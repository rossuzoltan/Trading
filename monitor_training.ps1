param(
    [int]$RefreshSeconds = 5,
    [string]$Symbol = "EURUSD",
    [int]$TotalTimesteps = 3000000,
    [string]$CheckpointsRoot = "checkpoints",
    [string]$TrainLogPath = "train_run.log",
    [switch]$NoLoop
)

. (Join-Path $PSScriptRoot "tools\monitor_training.ps1") @PSBoundParameters
