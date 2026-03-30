param(
    [string]$Symbol = "EURUSD",
    [int]$NumEnvs = 6,
    [int]$TotalTimesteps = 200000,
    [int]$PpoNSteps = 1024,
    [int]$PpoBatchSize = 1024,
    [int]$PpoNEpochs = 10,
    [int]$EvalFreq = 10000,
    [int]$HeartbeatEverySteps = 2048,
    [int]$LogInterval = 5,
    [double]$PpoLearningRate = 0.0,
    [double]$PpoMinLearningRate = 0.0,
    [double]$PpoEntCoef = 0.0,
    [switch]$AllowBaselineBypass,
    [switch]$ResumeLatest
)

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptRoot

$env:TRAIN_SYMBOL = $Symbol
$env:TRAIN_NUM_ENVS = [string]$NumEnvs
$env:TRAIN_TOTAL_TIMESTEPS = [string]$TotalTimesteps
$env:TRAIN_PPO_N_STEPS = [string]$PpoNSteps
$env:TRAIN_PPO_BATCH_SIZE = [string]$PpoBatchSize
$env:TRAIN_PPO_N_EPOCHS = [string]$PpoNEpochs
$env:TRAIN_EVAL_FREQ = [string]$EvalFreq
$env:TRAIN_HEARTBEAT_EVERY_STEPS = [string]$HeartbeatEverySteps
$env:TRAIN_LOG_INTERVAL = [string]$LogInterval
$env:TRAIN_PROGRESS_VERBOSE = "1"
$env:TRAIN_DEBUG_ALLOW_BASELINE_BYPASS = if ($AllowBaselineBypass) { "1" } else { "0" }
$env:TRAIN_RESUME_LATEST = if ($ResumeLatest) { "1" } else { "0" }
if ($PpoLearningRate -gt 0) {
    $env:TRAIN_PPO_LEARNING_RATE = [string]$PpoLearningRate
}
if ($PpoMinLearningRate -gt 0) {
    $env:TRAIN_PPO_MIN_LEARNING_RATE = [string]$PpoMinLearningRate
}
if ($PpoEntCoef -gt 0) {
    $env:TRAIN_PPO_ENT_COEF = [string]$PpoEntCoef
}

Remove-Item Env:TRAIN_FORCE_DUMMY_VEC -ErrorAction SilentlyContinue

# Only kill train_agent.py processes that belong to this repo (repo-filtered kill)
$escapedRoot = [regex]::Escape($ScriptRoot)
Get-CimInstance Win32_Process |
  Where-Object {
    $_.CommandLine -match 'train_agent\.py' -and
    $_.CommandLine -match $escapedRoot
  } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

$stdoutLog = Join-Path $ScriptRoot "train_run.log"
$stderrLog = Join-Path $ScriptRoot "train_err.log"
Remove-Item $stdoutLog, $stderrLog -Force -ErrorAction SilentlyContinue

$pythonExe = Join-Path $ScriptRoot ".venv\Scripts\python.exe"
$process = Start-Process `
  -FilePath $pythonExe `
  -ArgumentList "-u", ".\train_agent.py" `
  -WorkingDirectory $ScriptRoot `
  -RedirectStandardOutput $stdoutLog `
  -RedirectStandardError $stderrLog `
  -PassThru

Set-Content -Path (Join-Path $ScriptRoot "train_pid.txt") -Value $process.Id
Write-Output "Started train_agent.py (PID=$($process.Id))"
Write-Output "Stdout: $stdoutLog"
Write-Output "Stderr: $stderrLog"
