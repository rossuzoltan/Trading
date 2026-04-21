param(
  [string]$WatchPath,
  [string]$CheckpointsRoot = "checkpoints",
  [int]$RefreshSeconds = 10,
  [int]$StaleAfterSeconds = 30,
  [switch]$NoLoop
)

$ImplPath = Join-Path $PSScriptRoot "tools\watch_training.ps1"
. $ImplPath `
  -WatchPath $WatchPath `
  -CheckpointsRoot $CheckpointsRoot `
  -RefreshSeconds $RefreshSeconds `
  -StaleAfterSeconds $StaleAfterSeconds `
  -NoLoop:$NoLoop
