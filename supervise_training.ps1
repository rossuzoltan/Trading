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
  [int]$CheckEverySeconds = 300,
  [int]$StaleAfterSeconds = 900,
  [int]$BadRunStepGate = 1000000,
  [double]$BadRunSharpeMax = 0.0,
  [double]$BadRunExplainedVarianceMin = 0.10,
  [switch]$AllowBaselineBypass,
  [switch]$ResumeLatest,
  [switch]$RunOnce
)

. (Join-Path $PSScriptRoot "tools\training_supervisor.ps1") @PSBoundParameters
