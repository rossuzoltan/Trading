param(
  [string]$WatchPath,
  [string]$CheckpointsRoot = "checkpoints",
  [int]$RefreshSeconds = 10,
  [int]$StaleAfterSeconds = 30,
  [switch]$NoLoop
)

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

if (-not $WatchPath) {
    $WatchPath = Join-Path $ProjectRoot "training_watch.log"
}

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

function Get-HeartbeatStatus {
  param(
    [object]$Heartbeat,
    [object]$PreviousHeartbeat,
    [datetime]$NowUtc,
    [int]$StaleAfterSeconds
  )

  $status = "stale_heartbeat"
  $reason = "heartbeat missing or unreadable"
  $ageSeconds = $null
  $steps = $null
  $timestamp = $null

  if (-not $Heartbeat) {
    return [pscustomobject]@{
      status = $status
      reason = $reason
      age_seconds = $ageSeconds
      steps = $steps
      timestamp = $timestamp
    }
  }

  try {
    $steps = [int]$Heartbeat.num_timesteps
  } catch {
    $steps = $null
  }

  try {
    $timestamp = [datetimeoffset]::Parse([string]$Heartbeat.timestamp_utc)
    $ageSeconds = ($NowUtc - $timestamp.UtcDateTime).TotalSeconds
  } catch {
    $timestamp = $null
    $ageSeconds = $null
  }

  if (-not $timestamp -or $null -eq $steps) {
    return [pscustomobject]@{
      status = "stale_heartbeat"
      reason = "heartbeat missing required fields"
      age_seconds = $ageSeconds
      steps = $steps
      timestamp = $timestamp
    }
  }

  if ($ageSeconds -gt [double]$StaleAfterSeconds) {
    return [pscustomobject]@{
      status = "stale_heartbeat"
      reason = ("heartbeat age {0} s > {1} s" -f [math]::Round($ageSeconds, 1), $StaleAfterSeconds)
      age_seconds = $ageSeconds
      steps = $steps
      timestamp = $timestamp
    }
  }

  $previousSteps = $null
  if ($PreviousHeartbeat) {
    try {
      $previousSteps = [int]$PreviousHeartbeat.num_timesteps
    } catch {
      $previousSteps = $null
    }
  }

  if ($PreviousHeartbeat -and $null -ne $previousSteps -and $steps -le $previousSteps) {
    return [pscustomobject]@{
      status = "no_progress"
      reason = ("num_timesteps did not increase ({0} -> {1})" -f $previousSteps, $steps)
      age_seconds = $ageSeconds
      steps = $steps
      timestamp = $timestamp
    }
  }

  return [pscustomobject]@{
    status = "healthy"
    reason = "heartbeat fresh and progress observed"
    age_seconds = $ageSeconds
    steps = $steps
    timestamp = $timestamp
  }
}

function Invoke-TrainingWatchIteration {
  param(
    [object]$PreviousHeartbeat,
    [string]$WatchPath,
    [string]$CheckpointsRoot,
    [int]$StaleAfterSeconds
  )

  $ts = Get-Date -Format o
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

  if ($procInfo) {
    $proc = Get-Process -Id $procInfo.ProcessId -ErrorAction SilentlyContinue
    if ($proc -and $heartbeat) {
      $status = Get-HeartbeatStatus -Heartbeat $heartbeat -PreviousHeartbeat $PreviousHeartbeat -NowUtc (Get-Date).ToUniversalTime() -StaleAfterSeconds $StaleAfterSeconds
      $timestampText = if ($status.timestamp) { [datetimeoffset]$status.timestamp } else { $null }
      $line = "[$ts] status=$($status.status) pid=$($proc.Id) cpu=$([math]::Round($proc.CPU,2)) heartbeat=$($heartbeatFile.FullName) steps=$($status.steps) hb_ts=$($timestampText) reason=$($status.reason)"
    } elseif ($proc -and $runContext) {
      $line = "[$ts] status=waiting_for_first_heartbeat pid=$($proc.Id) cpu=$([math]::Round($proc.CPU,2)) checkpoints_root=$heartbeatRoot reason=current run started but no heartbeat has been written yet"
    } elseif ($proc) {
      $line = "[$ts] status=stale_heartbeat pid=$($proc.Id) cpu=$([math]::Round($proc.CPU,2)) heartbeat=missing reason=heartbeat missing or unreadable"
    } else {
      $line = "[$ts] status=stopped pid=$($procInfo.ProcessId)"
      Add-Content -Path $WatchPath -Value $line
      return [pscustomobject]@{ line = $line; heartbeat = $heartbeat }
    }
  } else {
    $line = "[$ts] status=stopped train_agent.py not running"
    Add-Content -Path $WatchPath -Value $line
    return [pscustomobject]@{ line = $line; heartbeat = $heartbeat }
  }

  Add-Content -Path $WatchPath -Value $line
  return [pscustomobject]@{ line = $line; heartbeat = $heartbeat }
}

if (-not $NoLoop -and $env:TRAINING_TELEMETRY_TEST -ne "1") {
  $previousHeartbeat = $null
  while ($true) {
    $iteration = Invoke-TrainingWatchIteration -PreviousHeartbeat $previousHeartbeat -WatchPath $WatchPath -CheckpointsRoot $CheckpointsRoot -StaleAfterSeconds $StaleAfterSeconds
    $previousHeartbeat = $iteration.heartbeat
    Start-Sleep -Seconds $RefreshSeconds
  }
}
