param(
    [string]$ManifestPath = "",
    [string]$Symbol = "",
    [int]$TicksPerBar = 0,
    [string]$AuditDir = "",
    [int]$PollIntervalMs = 250,
    [switch]$LogFullFeatures,
    [string]$StdoutPath = "",
    [string]$StderrPath = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Resolve-DefaultManifest {
    param([string]$RequestedSymbol)

    $normalized = ""
    if (-not [string]::IsNullOrWhiteSpace($RequestedSymbol)) {
        $normalized = $RequestedSymbol.ToUpperInvariant()
    }

    switch ($normalized) {
        "GBPUSD" { return Join-Path $root "models/rc1/gbpusd_10k_v1_mr_rc1/manifest.json" }
        default { return Join-Path $root "models/rc1/eurusd_5k_v1_mr_rc1/manifest.json" }
    }
}

if ([string]::IsNullOrWhiteSpace($ManifestPath)) {
    $ManifestPath = Resolve-DefaultManifest -RequestedSymbol $Symbol
}
if (-not (Test-Path $ManifestPath)) {
    throw "Manifest not found: $ManifestPath"
}

$python = Join-Path $root ".venv/Scripts/python.exe"
if (-not (Test-Path $python)) {
    throw "Project virtualenv python not found: $python"
}

if ([string]::IsNullOrWhiteSpace($StdoutPath) -or [string]::IsNullOrWhiteSpace($StderrPath)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    if ([string]::IsNullOrWhiteSpace($StdoutPath)) {
        $StdoutPath = Join-Path $root "models/shadow_simulator_${stamp}_stdout.log"
    }
    if ([string]::IsNullOrWhiteSpace($StderrPath)) {
        $StderrPath = Join-Path $root "models/shadow_simulator_${stamp}_stderr.log"
    }
}

$args = @(
    "-m", "runtime.shadow_broker",
    "--manifest", $ManifestPath,
    "--poll-interval-ms", "$PollIntervalMs"
)
if (-not [string]::IsNullOrWhiteSpace($Symbol)) {
    $args += @("--symbol", $Symbol)
}
if ($TicksPerBar -gt 0) {
    $args += @("--ticks-per-bar", "$TicksPerBar")
}
if (-not [string]::IsNullOrWhiteSpace($AuditDir)) {
    $args += @("--audit-dir", $AuditDir)
}
if ($LogFullFeatures.IsPresent) {
    $args += @("--log-full-features")
}

Write-Output "Starting shadow simulator in background with manifest: $ManifestPath"
Write-Output "Stdout: $StdoutPath"
Write-Output "Stderr: $StderrPath"

$proc = Start-Process -FilePath $python -ArgumentList $args -NoNewWindow -PassThru -RedirectStandardOutput $StdoutPath -RedirectStandardError $StderrPath
Write-Output "PID: $($proc.Id)"
Write-Output "Stop with: Stop-Process -Id $($proc.Id)"
