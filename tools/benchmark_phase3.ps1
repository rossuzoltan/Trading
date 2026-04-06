param (
    [string]$RunId = "phase3_benchmark_$(Get-Date -UFormat %s)",
    [int]$Timesteps = 20000,
    [string]$Symbol = "EURUSD"
)

$Env:TRAIN_SYMBOL = $Symbol
$Env:TRAIN_TOTAL_TIMESTEPS = $Timesteps
$Env:TRAIN_RESUME_LATEST = "0"
$Env:TRAIN_NUM_ENVS = "8"
$Env:TRAIN_PPO_BATCH_SIZE = "128"
$Env:TRAIN_N_FOLDS = "1"
$Env:TRAIN_EVAL_FREQ = "100000"
$Env:TRAIN_PURGE_GAP_BARS = "0"

function Start-Benchmark {
    param (
        [string]$Name,
        [hashtable]$Config
    )
    
    Write-Host "`n=======================================================" -ForegroundColor Cyan
    Write-Host "Running Benchmark: $Name" -ForegroundColor Cyan
    Write-Host "=======================================================" -ForegroundColor Cyan
    
    # Apply Config
    foreach ($key in $Config.Keys) {
        Set-Item -Path "Env:$key" -Value $Config[$key]
        Write-Host "  ENV: $key = $($Config[$key])" -ForegroundColor DarkGray
    }

    $startTime = Get-Date

    # We will log the stdout to check time and SPS.
    $logPath = "checkpoints\bench_${Name}.log"
    .venv\Scripts\python.exe train_agent.py > $logPath 2>&1
    
    $endTime = Get-Date
    $elapsed = $endTime - $startTime
    Write-Host "-> $Name finished in $($elapsed.TotalSeconds.ToString('0.0')) seconds" -ForegroundColor Green
    
    # Reset config back to default so tests don't leak into each other
    foreach ($key in $Config.Keys) {
        Remove-Item -Path "Env:$key" -ErrorAction SilentlyContinue
    }
}

# 1. Baseline
Start-Benchmark -Name "Baseline" -Config @{
    "TRAIN_TORCH_COMPILE" = "0"
    "TRAIN_REDUCE_LOGGING" = "0"
    "TRAIN_ASYNC_EVAL" = "0"
    "TRAIN_SHARED_DATASET" = "0"
    "TRAIN_USE_AMP" = "0"
}

# 2. Compile Only
Start-Benchmark -Name "TorchCompile" -Config @{
    "TRAIN_TORCH_COMPILE" = "1"
    "TRAIN_REDUCE_LOGGING" = "0"
    "TRAIN_ASYNC_EVAL" = "0"
    "TRAIN_SHARED_DATASET" = "0"
    "TRAIN_USE_AMP" = "0"
}

# 3. Reduce Logging and Eval
Start-Benchmark -Name "FastLogging" -Config @{
    "TRAIN_TORCH_COMPILE" = "0"
    "TRAIN_REDUCE_LOGGING" = "1"
    "TRAIN_ASYNC_EVAL" = "1"
    "TRAIN_SHARED_DATASET" = "0"
    "TRAIN_USE_AMP" = "0"
}

# 4. Memmap Shared Dataset
Start-Benchmark -Name "SharedMemmap" -Config @{
    "TRAIN_TORCH_COMPILE" = "0"
    "TRAIN_REDUCE_LOGGING" = "0"
    "TRAIN_ASYNC_EVAL" = "0"
    "TRAIN_SHARED_DATASET" = "1"
    "TRAIN_USE_AMP" = "0"
}

# 5. Combined Best (No AMP)
Start-Benchmark -Name "CombinedBest" -Config @{
    "TRAIN_TORCH_COMPILE" = "1"
    "TRAIN_REDUCE_LOGGING" = "1"
    "TRAIN_ASYNC_EVAL" = "1"
    "TRAIN_SHARED_DATASET" = "1"
    "TRAIN_USE_AMP" = "0"
}

Write-Host "`n[Benchmark Suite Finished]" -ForegroundColor Green
