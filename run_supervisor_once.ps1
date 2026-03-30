Set-Location $PSScriptRoot

powershell.exe -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "supervise_training.ps1") `
  -Symbol EURUSD `
  -NumEnvs 6 `
  -TotalTimesteps 3000000 `
  -PpoNSteps 1024 `
  -PpoBatchSize 1024 `
  -PpoNEpochs 10 `
  -EvalFreq 10000 `
  -HeartbeatEverySteps 2048 `
  -LogInterval 5 `
  -CheckEverySeconds 300 `
  -StaleAfterSeconds 900 `
  -BadRunStepGate 1000000 `
  -BadRunSharpeMax 0.0 `
  -BadRunExplainedVarianceMin 0.10 `
  -AllowBaselineBypass `
  -RunOnce
