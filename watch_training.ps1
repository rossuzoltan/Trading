param(
    [string]$WatchPath,
    [string]$CheckpointsRoot = "checkpoints",
    [int]$RefreshSeconds = 10,
    [int]$StaleAfterSeconds = 30,
    [switch]$NoLoop
)

. (Join-Path $PSScriptRoot "tools\watch_training.ps1") @PSBoundParameters
