param(
  [string]$Symbol = "EURUSD",
  [int]$NumEnvs = 6,
  [int]$TotalTimesteps = 3000000,
  [int]$PpoNSteps = 1024,
  [int]$PpoBatchSize = 1024,
  [int]$PpoNEpochs = 10,
  [int]$EvalFreq = 10000,
  [int]$HeartbeatEverySteps = 2048,
  [int]$LogInterval = 5,
  [int]$CheckEverySeconds = 600,
  [int]$StaleAfterSeconds = 1800,
  [int]$BadRunStepGate = 1000000,
  [double]$BadRunSharpeMax = 0.0,
  [double]$BadRunExplainedVarianceMin = 0.10,
  [switch]$AllowBaselineBypass,
  [switch]$ResumeLatest,
  [switch]$RunOnce
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

$SupervisorLogPath = Join-Path $ProjectRoot "training_supervisor.log"
$SupervisorStatePath = Join-Path $ProjectRoot "training_supervisor_state.json"
$StdoutLogPath = Join-Path $ProjectRoot "train_run.log"
$StderrLogPath = Join-Path $ProjectRoot "train_err.log"
$StartScriptPath = Join-Path $ProjectRoot "start_training_bg.ps1"

function Write-SupervisorLog {
  param(
    [string]$Message,
    [string]$Level = "INFO"
  )

  $line = "{0} | {1} | {2}" -f (Get-Date).ToString("o"), $Level.ToUpperInvariant(), $Message
  Add-Content -Path $SupervisorLogPath -Value $line
  Write-Host $line
}

function Get-RepoTrainProcesses {
  $escapedRoot = [regex]::Escape($ProjectRoot)
  Get-CimInstance Win32_Process |
    Where-Object {
      $_.CommandLine -match 'train_agent\.py' -and
      $_.CommandLine -match $escapedRoot
    } |
    Sort-Object CreationDate
}

function Get-RepoTrainProcess {
  Get-RepoTrainProcesses | Select-Object -Last 1
}

function Stop-RepoTraining {
  $procs = @(Get-RepoTrainProcesses)
  foreach ($proc in $procs) {
    try {
      Stop-Process -Id $proc.ProcessId -Force -ErrorAction Stop
      Write-SupervisorLog -Message ("Stopped train_agent.py PID={0}" -f $proc.ProcessId) -Level "WARN"
    } catch {
      Write-SupervisorLog -Message ("Failed to stop PID={0}: {1}" -f $proc.ProcessId, $_.Exception.Message) -Level "WARN"
    }
  }
}

function Get-CurrentTrainingRunContext {
  $contextPath = Join-Path $ProjectRoot "checkpoints\current_training_run.json"
  if (-not (Test-Path $contextPath)) {
    return $null
  }
  try {
    return Get-Content -Path $contextPath -Raw | ConvertFrom-Json
  } catch {
    return $null
  }
}

function Get-LatestHeartbeatFile {
  param([string]$RootPath)
  if (-not $RootPath -or -not (Test-Path $RootPath)) {
    return $null
  }
  Get-ChildItem -Path $RootPath -Recurse -Filter 'training_heartbeat.json' -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
}

function Get-HeartbeatObject {
  param([string]$Path)
  if (-not $Path -or -not (Test-Path $Path)) {
    return $null
  }
  try {
    return Get-Content -Path $Path -Raw | ConvertFrom-Json
  } catch {
    return $null
  }
}

function Get-HeartbeatSnapshot {
  $runContext = Get-CurrentTrainingRunContext
  $heartbeatRoot = Join-Path $ProjectRoot "checkpoints"
  if ($runContext -and $runContext.checkpoints_root) {
    $heartbeatRoot = Join-Path $ProjectRoot ([string]$runContext.checkpoints_root)
  }
  $heartbeatFile = Get-LatestHeartbeatFile -RootPath $heartbeatRoot
  $heartbeat = $null
  if ($heartbeatFile) {
    $heartbeat = Get-HeartbeatObject -Path $heartbeatFile.FullName
  }
  [pscustomobject]@{
    run_context = $runContext
    heartbeat_file = $heartbeatFile
    heartbeat = $heartbeat
  }
}

function Get-RecentErrorText {
  if (-not (Test-Path $StderrLogPath)) {
    return ""
  }
  try {
    return (Get-Content -Path $StderrLogPath -Tail 120) -join "`n"
  } catch {
    return ""
  }
}

function Get-RecentStdoutText {
  if (-not (Test-Path $StdoutLogPath)) {
    return ""
  }
  try {
    return (Get-Content -Path $StdoutLogPath -Tail 120) -join "`n"
  } catch {
    return ""
  }
}

function Get-SupervisorState {
  if (-not (Test-Path $SupervisorStatePath)) {
    return [pscustomobject]@{
      active_num_envs = $NumEnvs
      active_profile = "default"
      restart_count = 0
      recent_failure_count = 0
      bad_run_restart_count = 0
      last_start_utc = $null
      last_restart_reason = $null
      last_seen_steps = $null
      last_seen_hb_utc = $null
    }
  }
  try {
    $state = Get-Content -Path $SupervisorStatePath -Raw | ConvertFrom-Json
    if ($null -eq $state.active_num_envs) {
      $state | Add-Member -NotePropertyName active_num_envs -NotePropertyValue $NumEnvs
    }
    if ($null -eq $state.active_profile) {
      $state | Add-Member -NotePropertyName active_profile -NotePropertyValue "default"
    }
    if ($null -eq $state.bad_run_restart_count) {
      $state | Add-Member -NotePropertyName bad_run_restart_count -NotePropertyValue 0
    }
    return $state
  } catch {
    return [pscustomobject]@{
      active_num_envs = $NumEnvs
      active_profile = "default"
      restart_count = 0
      recent_failure_count = 0
      bad_run_restart_count = 0
      last_start_utc = $null
      last_restart_reason = $null
      last_seen_steps = $null
      last_seen_hb_utc = $null
    }
  }
}

function Save-SupervisorState {
  param([object]$State)
  $State | ConvertTo-Json -Depth 6 | Set-Content -Path $SupervisorStatePath
}

function Get-RemediationPlan {
  param(
    [string]$Reason,
    [object]$State
  )

  $activeNumEnvs = [int]$State.active_num_envs
  $newFailureCount = [int]$State.recent_failure_count + 1
  $nextNumEnvs = $activeNumEnvs
  $action = "restart_same_config"

  if ($Reason -match 'BrokenPipe|pipe|subproc|spawn') {
    if ($newFailureCount -ge 2 -and $activeNumEnvs -gt 2) {
      $nextNumEnvs = $activeNumEnvs - 1
      $action = "decrease_env_workers"
      $newFailureCount = 0
    }
  } else {
    $newFailureCount = 0
  }

  [pscustomobject]@{
    action = $action
    next_num_envs = $nextNumEnvs
    next_profile = [string]$State.active_profile
    recent_failure_count = $newFailureCount
  }
}

function Get-TrainingProfile {
  param([string]$ProfileName)

  switch ($ProfileName) {
    "repair_kl_push" {
      return [pscustomobject]@{
        name = "repair_kl_push"
        ppo_learning_rate = 0.0015
        ppo_min_learning_rate = 0.00015
        ppo_ent_coef = 0.02
        ppo_batch_size = 512
        ppo_n_epochs = 12
        resume_latest = $false
      }
    }
    default {
      return [pscustomobject]@{
        name = "default"
        ppo_learning_rate = 0.0
        ppo_min_learning_rate = 0.0
        ppo_ent_coef = 0.0
        ppo_batch_size = $PpoBatchSize
        ppo_n_epochs = $PpoNEpochs
        resume_latest = [bool]$ResumeLatest
      }
    }
  }
}

function Start-RepoTraining {
  param(
    [int]$EffectiveNumEnvs,
    [string]$Reason,
    [string]$ProfileName
  )

  $profile = Get-TrainingProfile -ProfileName $ProfileName
  $args = @(
    '-ExecutionPolicy', 'Bypass',
    '-File', $StartScriptPath,
    '-Symbol', $Symbol,
    '-NumEnvs', [string]$EffectiveNumEnvs,
    '-TotalTimesteps', [string]$TotalTimesteps,
    '-PpoNSteps', [string]$PpoNSteps,
    '-PpoBatchSize', [string]$profile.ppo_batch_size,
    '-PpoNEpochs', [string]$profile.ppo_n_epochs,
    '-EvalFreq', [string]$EvalFreq,
    '-HeartbeatEverySteps', [string]$HeartbeatEverySteps,
    '-LogInterval', [string]$LogInterval
  )
  if ([double]$profile.ppo_learning_rate -gt 0) {
    $args += @('-PpoLearningRate', [string]$profile.ppo_learning_rate)
  }
  if ([double]$profile.ppo_min_learning_rate -gt 0) {
    $args += @('-PpoMinLearningRate', [string]$profile.ppo_min_learning_rate)
  }
  if ([double]$profile.ppo_ent_coef -gt 0) {
    $args += @('-PpoEntCoef', [string]$profile.ppo_ent_coef)
  }
  if ($AllowBaselineBypass) {
    $args += '-AllowBaselineBypass'
  }
  if ($profile.resume_latest) {
    $args += '-ResumeLatest'
  }

  $output = & powershell.exe @args 2>&1
  foreach ($line in @($output)) {
    if ($line) {
      Write-SupervisorLog -Message ("launcher> {0}" -f $line)
    }
  }
  Write-SupervisorLog -Message ("Training launch requested | symbol={0} | envs={1} | profile={2} | reason={3}" -f $Symbol, $EffectiveNumEnvs, $profile.name, $Reason)
}

function Test-BadRunAtMillion {
  param(
    [object]$State,
    [object]$Snapshot
  )

  if ([int]$State.bad_run_restart_count -ge 1) {
    return $null
  }
  if (-not $Snapshot.heartbeat) {
    return $null
  }

  $steps = $null
  try {
    $steps = [int]$Snapshot.heartbeat.num_timesteps
  } catch {
    $steps = $null
  }
  if ($null -eq $steps -or $steps -lt $BadRunStepGate) {
    return $null
  }

  $latestEval = $Snapshot.heartbeat.latest_eval
  $ppo = $Snapshot.heartbeat.ppo_diagnostics
  if (-not $latestEval -or -not $ppo) {
    return $null
  }

  $sharpe = $null
  $ev = $null
  try { $sharpe = [double]$latestEval.timed_sharpe } catch { $sharpe = $null }
  try { $ev = [double]$ppo.explained_variance } catch { $ev = $null }
  if ($null -eq $sharpe -or $null -eq $ev) {
    return $null
  }

  if ($sharpe -le $BadRunSharpeMax -and $ev -lt $BadRunExplainedVarianceMin) {
    return [pscustomobject]@{
      restart = $true
      reason = ("bad_run_at_{0}_steps sharpe={1:N3} ev={2:N3}" -f $BadRunStepGate, $sharpe, $ev)
      next_profile = "repair_kl_push"
    }
  }

  return $null
}

function Test-NeedsRestart {
  param(
    [object]$State,
    [object]$Snapshot
  )

  $procInfo = Get-RepoTrainProcess
  if (-not $procInfo) {
    $errorText = "{0}`n{1}" -f (Get-RecentErrorText), (Get-RecentStdoutText)
    $reason = "train_agent.py not running"
    if ($errorText -match 'BrokenPipeError') {
      $reason = "BrokenPipeError after process exit"
    }
    return [pscustomobject]@{
      restart = $true
      reason = $reason
    }
  }

  $heartbeat = $Snapshot.heartbeat
  if (-not $heartbeat) {
    return [pscustomobject]@{
      restart = $false
      reason = "waiting_for_heartbeat"
    }
  }

  $hbTimestamp = $null
  try {
    $hbTimestamp = [datetimeoffset]::Parse([string]$heartbeat.timestamp_utc)
  } catch {
    $hbTimestamp = $null
  }
  if (-not $hbTimestamp) {
    return [pscustomobject]@{
      restart = $true
      reason = "heartbeat unreadable"
    }
  }

  $hbAgeSeconds = ((Get-Date).ToUniversalTime() - $hbTimestamp.UtcDateTime).TotalSeconds
  if ($hbAgeSeconds -gt $StaleAfterSeconds) {
    return [pscustomobject]@{
      restart = $true
      reason = ("stale heartbeat ({0:N1}s)" -f $hbAgeSeconds)
    }
  }

  $currentSteps = $null
  try {
    $currentSteps = [int]$heartbeat.num_timesteps
  } catch {
    $currentSteps = $null
  }

  if ($null -ne $State.last_seen_steps -and $null -ne $currentSteps) {
    $lastSeen = [int]$State.last_seen_steps
    if ($currentSteps -le $lastSeen) {
      return [pscustomobject]@{
        restart = $true
        reason = ("no step progress ({0} -> {1})" -f $lastSeen, $currentSteps)
      }
    }
  }

  return [pscustomobject]@{
    restart = $false
    reason = "healthy"
  }
}

function Update-StateFromSnapshot {
  param(
    [object]$State,
    [object]$Snapshot
  )

  if ($Snapshot.heartbeat) {
    try {
      $State.last_seen_steps = [int]$Snapshot.heartbeat.num_timesteps
    } catch {
      $State.last_seen_steps = $null
    }
    $State.last_seen_hb_utc = [string]$Snapshot.heartbeat.timestamp_utc
  }
  return $State
}

Write-SupervisorLog -Message ("Supervisor active | symbol={0} | requested_envs={1} | check_every={2}s | stale_after={3}s" -f $Symbol, $NumEnvs, $CheckEverySeconds, $StaleAfterSeconds)

while ($true) {
  $state = Get-SupervisorState
  $snapshot = Get-HeartbeatSnapshot
  $badRunDecision = Test-BadRunAtMillion -State $state -Snapshot $snapshot
  if ($badRunDecision) {
    $decision = [pscustomobject]@{
      restart = $true
      reason = $badRunDecision.reason
    }
  } else {
    $decision = Test-NeedsRestart -State $state -Snapshot $snapshot
  }

  if ($decision.restart) {
    $plan = Get-RemediationPlan -Reason $decision.reason -State $state
    if ($badRunDecision) {
      $plan.next_profile = [string]$badRunDecision.next_profile
      $state.bad_run_restart_count = [int]$state.bad_run_restart_count + 1
      Write-SupervisorLog -Message ("Bad run gate triggered at >= {0} steps; switching profile {1} -> {2}" -f $BadRunStepGate, $state.active_profile, $plan.next_profile) -Level "WARN"
    }
    if ([int]$plan.next_num_envs -ne [int]$state.active_num_envs) {
      Write-SupervisorLog -Message ("Remediation: reducing env workers {0} -> {1} after reason={2}" -f $state.active_num_envs, $plan.next_num_envs, $decision.reason) -Level "WARN"
    } else {
      Write-SupervisorLog -Message ("Remediation: restart with same env worker count={0} after reason={1}" -f $state.active_num_envs, $decision.reason) -Level "WARN"
    }

    Stop-RepoTraining
    $state.active_num_envs = [int]$plan.next_num_envs
    $state.active_profile = [string]$plan.next_profile
    $state.recent_failure_count = [int]$plan.recent_failure_count
    $state.restart_count = [int]$state.restart_count + 1
    $state.last_restart_reason = $decision.reason
    $state.last_start_utc = (Get-Date).ToUniversalTime().ToString("o")
    $state.last_seen_steps = $null
    $state.last_seen_hb_utc = $null
    Save-SupervisorState -State $state
    Start-RepoTraining -EffectiveNumEnvs ([int]$state.active_num_envs) -Reason $decision.reason -ProfileName ([string]$state.active_profile)
  } else {
    $state = Update-StateFromSnapshot -State $state -Snapshot $snapshot
    Save-SupervisorState -State $state
    $stepsText = if ($null -ne $state.last_seen_steps) { [string]$state.last_seen_steps } else { "n/a" }
    Write-SupervisorLog -Message ("Health check OK | envs={0} | profile={1} | steps={2} | reason={3}" -f $state.active_num_envs, $state.active_profile, $stepsText, $decision.reason)
  }

  if ($RunOnce) {
    break
  }
  Start-Sleep -Seconds $CheckEverySeconds
}
