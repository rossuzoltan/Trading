param(
    [int]$RefreshSeconds = 5,
    [string]$Symbol = "EURUSD",
    [int]$TotalTimesteps = 3000000,
    [string]$CheckpointsRoot = "checkpoints",
    [string]$TrainLogPath = "train_run.log",
    [switch]$NoLoop
)

$ImplPath = Join-Path $PSScriptRoot "tools\monitor_training.ps1"
. $ImplPath `
    -RefreshSeconds $RefreshSeconds `
    -Symbol $Symbol `
    -TotalTimesteps $TotalTimesteps `
    -CheckpointsRoot $CheckpointsRoot `
    -TrainLogPath $TrainLogPath `
    -NoLoop:$NoLoop
