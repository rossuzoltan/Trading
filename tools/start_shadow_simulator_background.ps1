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
$ManifestPath = (Resolve-Path $ManifestPath).Path
$manifestInfo = Get-Content $ManifestPath -Raw | ConvertFrom-Json
$symbolName = "$($manifestInfo.strategy_symbol)".ToUpperInvariant()
$manifestHash = "$($manifestInfo.manifest_hash)"

# Refuse to start duplicate shadow brokers for the same manifest, since multiple
# processes can corrupt evidence and make drift/debugging non-actionable.
$existing = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object {
        ($_.CommandLine -like "*runtime.shadow_broker*" -or $_.CommandLine -like "*shadow_broker.py*") -and
        $_.CommandLine -like "*$ManifestPath*"
    } |
    Select-Object ProcessId
if ($existing) {
    $pids = ($existing | ForEach-Object { $_.ProcessId }) -join ", "
    Write-Output "Shadow simulator already running for this manifest. PIDs: $pids"
    Write-Output "Stop with: Stop-Process -Id $pids"
    exit 0
}

$python = Join-Path $root ".venv/Scripts/python.exe"
if (-not (Test-Path $python)) {
    throw "Project virtualenv python not found: $python"
}

if ([string]::IsNullOrWhiteSpace($StdoutPath) -or [string]::IsNullOrWhiteSpace($StderrPath)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    if ([string]::IsNullOrWhiteSpace($StdoutPath)) {
        $StdoutPath = Join-Path $root "logs/shadow_simulator_${stamp}_stdout.log"
    }
    if ([string]::IsNullOrWhiteSpace($StderrPath)) {
        $StderrPath = Join-Path $root "logs/shadow_simulator_${stamp}_stderr.log"
    }
}

$args = @(
    "runtime/shadow_broker.py",
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

$logDir = Split-Path -Parent $StdoutPath
if (-not [string]::IsNullOrWhiteSpace($logDir)) {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
}
$proc = Start-Process -FilePath $python -ArgumentList $args -NoNewWindow -PassThru -RedirectStandardOutput $StdoutPath -RedirectStandardError $StderrPath
$effectiveShadowRoot = if ([string]::IsNullOrWhiteSpace($AuditDir)) { Join-Path $root "artifacts/shadow" } else { $AuditDir }
$evidenceDir = Join-Path $effectiveShadowRoot (Join-Path $symbolName $manifestHash)
$metadata = [ordered]@{
    started_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    pid = $proc.Id
    manifest_path = $ManifestPath
    manifest_hash = $manifestHash
    symbol = $symbolName
    audit_root = $effectiveShadowRoot
    evidence_dir = $evidenceDir
    stdout_path = $StdoutPath
    stderr_path = $StderrPath
    poll_interval_ms = $PollIntervalMs
    command = @($python) + $args
}
$metadataJson = $metadata | ConvertTo-Json -Depth 4
$metadataPath = Join-Path $root "logs/shadow_launch_${stamp}.json"
$latestMetadataPath = Join-Path $root ("logs/shadow_latest_{0}_{1}.json" -f $symbolName, $manifestHash)
$metadataJson | Set-Content -Path $metadataPath -Encoding UTF8
$metadataJson | Set-Content -Path $latestMetadataPath -Encoding UTF8
Write-Output "Metadata: $metadataPath"
Write-Output "PID: $($proc.Id)"
Write-Output "Stop with: Stop-Process -Id $($proc.Id)"
