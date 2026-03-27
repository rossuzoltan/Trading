$env:TRAIN_SYMBOL = if ($env:TRAIN_SYMBOL) { $env:TRAIN_SYMBOL } else { "EURUSD" }
if (-not $env:TRAIN_NUM_ENVS) { $env:TRAIN_NUM_ENVS = "6" }

Set-Location "C:\dev\trading"

$log = "C:\dev\trading\train_run.log"
if (Test-Path $log) {
    Remove-Item $log -Force
}

& ".\.venv\Scripts\python.exe" ".\train_agent.py" *>&1 | Tee-Object -FilePath $log
