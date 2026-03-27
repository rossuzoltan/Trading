param(
    [string]$Symbol = "EURUSD",
    [int]$NumEnvs = 6,
    [int]$TotalTimesteps = 200000,
    [int]$PpoNSteps = 1024,
    [int]$EvalFreq = 10000,
    [int]$HeartbeatEverySteps = 2048,
    [int]$LogInterval = 5
)

Set-Location "C:\dev\trading"

$env:TRAIN_SYMBOL = $Symbol
$env:TRAIN_NUM_ENVS = [string]$NumEnvs
$env:TRAIN_TOTAL_TIMESTEPS = [string]$TotalTimesteps
$env:TRAIN_PPO_N_STEPS = [string]$PpoNSteps
$env:TRAIN_EVAL_FREQ = [string]$EvalFreq
$env:TRAIN_HEARTBEAT_EVERY_STEPS = [string]$HeartbeatEverySteps
$env:TRAIN_LOG_INTERVAL = [string]$LogInterval
$env:TRAIN_PROGRESS_VERBOSE = "1"

Remove-Item Env:TRAIN_FORCE_DUMMY_VEC -ErrorAction SilentlyContinue

Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match 'train_agent\.py' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

$stdoutLog = "C:\dev\trading\train_run.log"
$stderrLog = "C:\dev\trading\train_err.log"
Remove-Item $stdoutLog, $stderrLog -Force -ErrorAction SilentlyContinue

$process = Start-Process `
  -FilePath "C:\dev\trading\.venv\Scripts\python.exe" `
  -ArgumentList "-u", ".\train_agent.py" `
  -WorkingDirectory "C:\dev\trading" `
  -RedirectStandardOutput $stdoutLog `
  -RedirectStandardError $stderrLog `
  -PassThru

Set-Content -Path "C:\dev\trading\train_pid.txt" -Value $process.Id
Write-Output "Started train_agent.py (PID=$($process.Id))"
Write-Output "Stdout: $stdoutLog"
Write-Output "Stderr: $stderrLog"
