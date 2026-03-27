param(
    [int]$RefreshSeconds = 5,
    [string]$Symbol = "EURUSD",
    [int]$TotalTimesteps = 3000000,
    [string]$CheckpointsRoot = "checkpoints",
    [string]$TrainLogPath = "train_run.log",
    [switch]$NoLoop
)

Set-Location "C:\dev\trading"

function Get-TrainProcess {
  Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -match 'train_agent\.py' } |
    Sort-Object CreationDate -Descending |
    Select-Object -First 1
}

function Get-LatestHeartbeat {
  param([string]$RootPath)
  Get-ChildItem $RootPath -Recurse -Filter 'training_heartbeat.json' -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
}

function Get-CurrentTrainingRunContext {
  param([string]$RootPath)
  $contextPath = Join-Path $RootPath 'current_training_run.json'
  if (-not (Test-Path $contextPath)) { return $null }
  try {
    return Get-Content $contextPath -Raw | ConvertFrom-Json
  } catch {
    return $null
  }
}

function Get-HeartbeatObject {
  param([string]$Path)
  if (-not $Path -or -not (Test-Path $Path)) { return $null }
  try {
    return Get-Content $Path -Raw | ConvertFrom-Json
  } catch {
    return $null
  }
}

function Format-Duration {
  param([double]$Seconds)
  if ($Seconds -lt 0 -or [double]::IsNaN($Seconds) -or [double]::IsInfinity($Seconds)) {
    return "ismeretlen"
  }
  $ts = [TimeSpan]::FromSeconds([math]::Round($Seconds))
  if ($ts.TotalHours -ge 1) {
    return ("{0:00}:{1:00}:{2:00}" -f [math]::Floor($ts.TotalHours), $ts.Minutes, $ts.Seconds)
  }
  return ("{0:00}:{1:00}" -f $ts.Minutes, $ts.Seconds)
}

function Get-GpuLine {
  try {
    $line = & nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1
    if (-not $line) { return "GPU adat nem elérhető" }
    $parts = $line -split ','
    if ($parts.Count -lt 4) { return $line.Trim() }
    $name = $parts[0].Trim()
    $util = $parts[1].Trim()
    $used = $parts[2].Trim()
    $total = $parts[3].Trim()
    return ("{0} | GPU: {1}% | VRAM: {2}/{3} MB" -f $name, $util, $used, $total)
  } catch {
    return "GPU adat nem elérhető"
  }
}

function Get-LatestLogLines {
  param([string]$Path)
  if (-not (Test-Path $Path)) { return @("A $Path még nem jött létre.") }
  return Get-Content $Path -Tail 12
}

function Get-HeartbeatThroughput {
  param(
    [Parameter(Mandatory = $true)] $CurrentHeartbeat,
    [object] $PreviousHeartbeat,
    [datetime] $NowUtc,
    [datetime] $ProcessStartTime,
    [int] $TotalTimesteps
  )

  $result = [ordered]@{
    mode = "unknown"
    speed_text = "ismeretlen"
    eta_text = "ismeretlen"
    progress_pct = 0.0
    steps = 0
    heartbeat_age_text = "nincs heartbeat"
    heartbeat_age_seconds = $null
    used_rolling = $false
  }

  if (-not $CurrentHeartbeat) {
    return [pscustomobject]$result
  }

  $steps = 0
  try {
    $steps = [int]$CurrentHeartbeat.num_timesteps
  } catch {
    $steps = 0
  }
  $result.steps = $steps

  if ($TotalTimesteps -gt 0) {
    $result.progress_pct = [math]::Min(100.0, [math]::Round((100.0 * $steps / $TotalTimesteps), 2))
  }

  $currentTimestamp = $null
  try {
    $currentTimestamp = [datetimeoffset]::Parse([string]$CurrentHeartbeat.timestamp_utc)
  } catch {
    $currentTimestamp = $null
  }
  if ($currentTimestamp) {
    $age = ($NowUtc - $currentTimestamp.UtcDateTime).TotalSeconds
    $result.heartbeat_age_seconds = $age
    $result.heartbeat_age_text = ("{0} mp" -f [math]::Round($age, 1))
  }

  $previousSteps = $null
  $previousTimestamp = $null
  if ($PreviousHeartbeat) {
    try {
      $previousSteps = [int]$PreviousHeartbeat.num_timesteps
    } catch {
      $previousSteps = $null
    }
    try {
      $previousTimestamp = [datetimeoffset]::Parse([string]$PreviousHeartbeat.timestamp_utc)
    } catch {
      $previousTimestamp = $null
    }
  }

  if ($PreviousHeartbeat -and $currentTimestamp -and $previousTimestamp -and $steps -gt $previousSteps) {
    $deltaSteps = [double]($steps - $previousSteps)
    $deltaSeconds = ($currentTimestamp.UtcDateTime - $previousTimestamp.UtcDateTime).TotalSeconds
    if ($deltaSeconds -gt 0 -and $deltaSteps -gt 0) {
      $sps = $deltaSteps / $deltaSeconds
      $result.mode = "rolling"
      $result.used_rolling = $true
      $result.speed_text = ("{0} step/sec" -f [math]::Round($sps, 1))
      if ($TotalTimesteps -gt $steps -and $sps -gt 0) {
        $result.eta_text = Format-Duration -Seconds (($TotalTimesteps - $steps) / $sps)
      }
      return [pscustomobject]$result
    }
  }

  if ($ProcessStartTime) {
    $elapsedSeconds = ($NowUtc - $ProcessStartTime).TotalSeconds
    if ($elapsedSeconds -gt 0 -and $steps -gt 0) {
      $sps = $steps / $elapsedSeconds
      $result.mode = "lifetime"
      $result.speed_text = ("{0} step/sec" -f [math]::Round($sps, 1))
      if ($TotalTimesteps -gt $steps -and $sps -gt 0) {
        $result.eta_text = Format-Duration -Seconds (($TotalTimesteps - $steps) / $sps)
      }
    }
  }

  return [pscustomobject]$result
}

function Invoke-TrainingMonitorIteration {
  param(
    [object] $PreviousHeartbeat,
    [datetime] $Now,
    [datetime] $NowUtc,
    [string] $CheckpointsRoot,
    [string] $TrainLogPath,
    [int] $TotalTimesteps,
    [string] $Symbol
  )

  $procInfo = Get-TrainProcess
  $runContext = Get-CurrentTrainingRunContext -RootPath $CheckpointsRoot
  $heartbeatRoot = $CheckpointsRoot
  if ($runContext -and $runContext.checkpoints_root) {
    $heartbeatRoot = [string]$runContext.checkpoints_root
  }
  $heartbeatFile = Get-LatestHeartbeat -RootPath $heartbeatRoot
  $heartbeat = $null
  if ($heartbeatFile) {
    $heartbeat = Get-HeartbeatObject -Path $heartbeatFile.FullName
  }

  $pidText = "nem fut"
  $cpuText = "-"
  $procStartTime = $null
  if ($procInfo) {
    $proc = Get-Process -Id $procInfo.ProcessId -ErrorAction SilentlyContinue
    if ($proc) {
      $pidText = [string]$proc.Id
      $cpuText = [string]([math]::Round($proc.CPU, 2))
      $procStartTime = $proc.StartTime.ToUniversalTime()
    } else {
      $pidText = [string]$procInfo.ProcessId
    }
  }

  $steps = 0
  $progressPct = 0.0
  $hbAgeText = "nincs heartbeat"
  $etaText = "ismeretlen"
  $speedText = "ismeretlen"
  $speedMode = "unknown"
  $hbTimestampText = "-"
  $explainedVariance = "-"
  $approxKl = "-"
  $valueLossStable = "-"
  $blockers = @()

  $throughput = Get-HeartbeatThroughput -CurrentHeartbeat $heartbeat -PreviousHeartbeat $PreviousHeartbeat -NowUtc $NowUtc -ProcessStartTime $procStartTime -TotalTimesteps $TotalTimesteps
  $steps = [int]$throughput.steps
  $progressPct = [double]$throughput.progress_pct
  $hbAgeText = [string]$throughput.heartbeat_age_text
  $etaText = [string]$throughput.eta_text
  $speedText = [string]$throughput.speed_text
  $speedMode = [string]$throughput.mode

  if ($heartbeat) {
    try {
      $hbTimestamp = [datetimeoffset]::Parse([string]$heartbeat.timestamp_utc)
      $hbTimestampText = $hbTimestamp.ToString("yyyy-MM-dd HH:mm:ss 'UTC'")
    } catch {
      $hbTimestampText = "ismeretlen"
    }

    $ppo = $heartbeat.ppo_diagnostics
    if ($ppo) {
      $explainedVariance = [string]$ppo.explained_variance
      $approxKl = [string]$ppo.approx_kl
      $valueLossStable = [string]$ppo.value_loss_stable
      if ($ppo.blockers) {
        $blockers = @($ppo.blockers)
      }
    }
  }

  Write-Host ("=" * 88) -ForegroundColor DarkCyan
  Write-Host "TRAINING MONITOR - $Symbol" -ForegroundColor Cyan
  Write-Host ("=" * 88) -ForegroundColor DarkCyan
  if ($runContext) {
    Write-Host ("Run ID:              {0}" -f $runContext.run_id)
    Write-Host ("Run state:           {0}" -f $runContext.state)
    Write-Host ("Checkpoint root:     {0}" -f $heartbeatRoot)
  }
  Write-Host ("Idő:                {0}" -f $Now.ToString("yyyy-MM-dd HH:mm:ss"))
  Write-Host ("Processz PID:       {0}" -f $pidText)
  Write-Host ("CPU idő:            {0}" -f $cpuText)
  Write-Host ("Előrehaladás:       {0} / {1} step  ({2}%)" -f $steps, $TotalTimesteps, $progressPct)
  Write-Host ("Becsült sebesség:   {0} [{1}]" -f $speedText, $speedMode)
  Write-Host ("Becsült hátralévő:  {0}" -f $etaText)
  Write-Host ("Utolsó heartbeat:   {0}" -f $hbTimestampText)
  Write-Host ("Heartbeat kora:     {0}" -f $hbAgeText)
  Write-Host ("GPU állapot:        {0}" -f (Get-GpuLine))
  if ($runContext -and -not $heartbeat) {
    Write-Host ("Megjegyzés:          current run már elindult, de még nincs első heartbeat.") -ForegroundColor Yellow
  }

  Write-Host ("-" * 88) -ForegroundColor DarkGray
  Write-Host "Mit nézz?" -ForegroundColor Yellow
  Write-Host ("- Előrehaladás: ha a step szám nő, a training halad.")
  Write-Host ("- Heartbeat kora: ha túl nagy és nem frissül, valami beragadt.")
  Write-Host ("- GPU állapot: ha van folyamat és VRAM használat, a GPU be van fogva.")
  Write-Host ("- Becsült sebesség / hátralévő idő: ez rolling heartbeatből jön, fallbackkal.")

  Write-Host ("-" * 88) -ForegroundColor DarkGray
  Write-Host "PPO diagnosztika" -ForegroundColor Yellow
  Write-Host ("- explained_variance: {0}" -f $explainedVariance)
  Write-Host ("- approx_kl:          {0}" -f $approxKl)
  Write-Host ("- value_loss_stable:  {0}" -f $valueLossStable)

  if ($blockers.Count -gt 0) {
    Write-Host "Jelenlegi blokkolók:" -ForegroundColor Red
    foreach ($blocker in $blockers) {
      Write-Host ("- {0}" -f $blocker) -ForegroundColor Red
    }
  } else {
    Write-Host "Jelenleg nincs blocker a heartbeatben." -ForegroundColor Green
  }

  Write-Host ("-" * 88) -ForegroundColor DarkGray
  Write-Host "Log vége" -ForegroundColor Yellow
  Get-LatestLogLines -Path $TrainLogPath | ForEach-Object { Write-Host $_ }

  return [pscustomobject]@{
    heartbeat = $heartbeat
    throughput = $throughput
    process_id = $pidText
  }
}

if (-not $NoLoop -and $env:TRAINING_TELEMETRY_TEST -ne "1") {
  $prevHeartbeat = $null
  $currHeartbeat = $null
  while ($true) {
    $now = Get-Date
    $nowUtc = (Get-Date).ToUniversalTime()
    $iteration = Invoke-TrainingMonitorIteration -PreviousHeartbeat $prevHeartbeat -Now $now -NowUtc $nowUtc -CheckpointsRoot $CheckpointsRoot -TrainLogPath $TrainLogPath -TotalTimesteps $TotalTimesteps -Symbol $Symbol
    
    $fileHb = $iteration.heartbeat
    if ($null -ne $fileHb) {
      if ($null -eq $currHeartbeat) {
        $currHeartbeat = $fileHb
      } elseif ($fileHb.num_timesteps -ne $currHeartbeat.num_timesteps) {
        $prevHeartbeat = $currHeartbeat
        $currHeartbeat = $fileHb
      }
    }
    
    Start-Sleep -Seconds $RefreshSeconds
  }
}
