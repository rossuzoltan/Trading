param(
    [string]$CheckpointsRoot = "checkpoints",
    [int]$RefreshSeconds = 2,
    [int]$StaleAfterSeconds = 15
)

function Get-LatestHeartbeat {
    param([string]$RootPath)
    $files = Get-ChildItem -Path $RootPath -Recurse -Filter 'training_heartbeat.json' -ErrorAction SilentlyContinue
    if ($null -eq $files) { return $null }
    return $files | Sort-Object LastWriteTime -Descending | Select-Object -First 1
}

function Get-CurrentTrainingRunContext {
    param([string]$RootPath)
    $contextPath = Join-Path $RootPath 'current_training_run.json'
    if (-not (Test-Path $contextPath)) { return $null }
    try {
        $content = Get-Content $contextPath -Raw -ErrorAction SilentlyContinue
        if ($null -eq $content -or [string]::IsNullOrWhiteSpace($content)) { return $null }
        return $content | ConvertFrom-Json
    } catch { return $null }
}

function Clear-Screen {
    [Console]::Clear()
}

$previousHeartbeat = $null
$startTime = Get-Date

while ($true) {
    $Now = Get-Date
    $runContext = Get-CurrentTrainingRunContext -RootPath $CheckpointsRoot
    $heartbeatRoot = $CheckpointsRoot
    if ($null -ne $runContext -and $null -ne $runContext.checkpoints_root) {
        $heartbeatRoot = [string]$runContext.checkpoints_root
    }
    
    $hbFile = Get-LatestHeartbeat -RootPath $heartbeatRoot
    $hb = $null
    if ($null -ne $hbFile) {
        try { 
            $rawHb = Get-Content $hbFile.FullName -Raw -ErrorAction SilentlyContinue
            if ($null -ne $rawHb) { $hb = $rawHb | ConvertFrom-Json }
        } catch {}
    }

    Clear-Screen
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "       TURBO RL TRAINING DASHBOARD (REAL-TIME)               " -ForegroundColor Yellow -BackgroundColor Black
    Write-Host "============================================================" -ForegroundColor Cyan
    
    if ($null -ne $runContext) {
        Write-Host "Symbol: $($runContext.symbol) | Run ID: $($runContext.run_id)"
        Write-Host "PID: $($runContext.pid) | State: $($runContext.state)"
    } else {
        Write-Host "Searching for active training run..." -ForegroundColor DarkGray
    }

    if ($null -ne $hb) {
        $hbTime = [datetimeoffset]::Parse($hb.timestamp_utc).UtcDateTime
        $age = ($Now.ToUniversalTime() - $hbTime).TotalSeconds
        $statusStr = "HEALTHY"
        $statusColor = "Green"
        
        if ($age -gt $StaleAfterSeconds) {
            $statusStr = "STALE"
            $statusColor = "Yellow"
        }
        
        Write-Host "`nStatus: $statusStr ($([math]::Round($age, 1))s ago)" -ForegroundColor $statusColor
        
        Write-Host "`n--- PROGRESS ---" -ForegroundColor Gray
        Write-Host "Steps: $($hb.num_timesteps)"
        $fps = $hb.fps
        $fpsColor = if ($fps -gt 200) { "Green" } else { "Yellow" }
        Write-Host "SPS:   $fps" -ForegroundColor $fpsColor
        
        Write-Host "`n--- REWARDS ---" -ForegroundColor Gray
        $reward = [math]::Round($hb.training_metrics.ep_rew_mean, 4)
        $rewColor = if ($reward -gt 0) { "Green" } else { "Red" }
        Write-Host "Mean Reward: $reward" -ForegroundColor $rewColor
        
        Write-Host "`n--- STABILITY ---" -ForegroundColor Gray
        Write-Host "Entropy:   $([math]::Round($hb.training_metrics.entropy_loss, 4))"
        Write-Host "KL Div:    $([math]::Round($hb.training_metrics.approx_kl, 6))"
        
        # Draw a mini progress bar for 1.2M steps
        $progress = ($hb.num_timesteps / 1200000) * 100
        $barWidth = 30
        $filled = [int]($progress * $barWidth / 100)
        $bar = "[" + ("#" * $filled) + ("-" * ($barWidth - $filled)) + "]"
        Write-Host "`nProgress: $bar $([math]::Round($progress, 1))%" -ForegroundColor White
    } else {
        Write-Host "`nWaiting for first heartbeat..." -ForegroundColor DarkGray
    }

    Write-Host "`n`n[Last Refresh: $(Get-Date -Format T)]" -ForegroundColor DarkGray
    Write-Host "Press Ctrl+C to stop dashboard." -ForegroundColor DarkBlue

    Start-Sleep -Seconds $RefreshSeconds
}
