param(
    [int]$NumEnvs = 16,
    [int]$TotalTimesteps = 3000000
)

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptRoot

# Hard stop any previous repo training run before starting a new one.
.\tools\stop_repo_training.ps1 -RepoRoot $ScriptRoot | Out-Null

# Throughput-oriented PPO defaults (override here if needed).
$env:TRAIN_SYMBOL = "EURUSD"
$env:TRAIN_TOTAL_TIMESTEPS = [string]$TotalTimesteps
$env:TRAIN_NUM_ENVS = [string]$NumEnvs
$env:TRAIN_PPO_N_STEPS = "512"
$env:TRAIN_PPO_BATCH_SIZE = "2048"
$env:TRAIN_PPO_N_EPOCHS = "3"
$env:FEATURE_ENGINE_FAST = "1"
$env:TRAIN_ADAPTIVE_KL_LR = "1"
$env:TRAIN_ALPHA_GATE_ENABLED = "1"
$env:TRAIN_EVAL_STOCHASTIC_RUNS = "5"
$env:TRAIN_COLLAPSE_WARMUP_STEPS = "500000"
$env:TRAIN_COLLAPSE_CONSECUTIVE = "2"
$env:TRAIN_TX_PENALTY_START = "0.30"
$env:TRAIN_TX_PENALTY_END = "1.00"
$env:TRAIN_TX_PENALTY_RAMP_STEPS = "1000000"

# Keep eval/logging overhead modest for long runs.
$env:TRAIN_EVAL_FREQ = "200000"
$env:TRAIN_HEARTBEAT_EVERY_STEPS = "20000"
$env:TRAIN_LOG_INTERVAL = "20"
$env:TRAIN_REDUCE_LOGGING = "1"

# Prefer exploration early (avoid reward_strip profiles that disable this).
Remove-Item Env:TRAIN_EXPERIMENT_PROFILE -ErrorAction SilentlyContinue
Remove-Item Env:TRAIN_RECOVERY_PARTICIPATION_BONUS_ENABLED -ErrorAction SilentlyContinue

# CUDA AMP for speed if available.
$env:TRAIN_USE_AMP = "1"
$env:TRAIN_AMP_DTYPE = "bf16"

.\start_training_bg.ps1 -Symbol "EURUSD" -NumEnvs $NumEnvs -TotalTimesteps $TotalTimesteps -PpoNSteps 512 -PpoBatchSize 2048 -PpoNEpochs 3 -EvalFreq 200000 -HeartbeatEverySteps 20000 -LogInterval 20

"Started 3M EURUSD run. Tail logs: Get-Content .\\train_run.log -Tail 50"
