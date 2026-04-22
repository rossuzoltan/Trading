param(
    [string]$ManifestPath = "models/rc1/eurusd_5k_v1_mr_rc1/manifest.json",
    [string]$AuditDir = "artifacts/shadow_sweep",
    [int]$PollIntervalMs = 250
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path $ManifestPath)) {
    throw "Manifest not found: $ManifestPath"
}

$python = Join-Path $root ".venv/Scripts/python.exe"
if (-not (Test-Path $python)) {
    throw "Project virtualenv python not found: $python"
}

$createScript = Join-Path $root "tools/create_shadow_sweep_manifests.py"
$payload = & $python $createScript --base-manifest $ManifestPath | ConvertFrom-Json
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"

$existing = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object {
        ($_.CommandLine -like "*shadow_sweep_broker.py*") -and
        $_.CommandLine -like "*$AuditDir*"
    } |
    Select-Object ProcessId
if ($existing) {
    $pids = ($existing | ForEach-Object { $_.ProcessId }) -join ", "
    Write-Output "Shadow sweep already running for audit root '$AuditDir'. PIDs: $pids"
    Write-Output "Stop with: Stop-Process -Id $pids"
    exit 0
}

Write-Output "Base manifest: $($payload.base_manifest_path)"
Write-Output "Base manifest hash: $($payload.base_manifest_hash)"
Write-Output "Audit root: $AuditDir"
Write-Output ""

$args = @(
    "runtime/shadow_sweep_broker.py",
    "--poll-interval-ms", "$PollIntervalMs",
    "--audit-dir", $AuditDir
)

foreach ($item in $payload.generated) {
    $evidenceDir = Join-Path $AuditDir (Join-Path $item.symbol $item.manifest_hash)
    New-Item -ItemType Directory -Force -Path $evidenceDir | Out-Null
    Write-Output ("[{0}] manifest={1}" -f $item.profile_id, $item.manifest_path)
    Write-Output ("[{0}] hash={1}" -f $item.profile_id, $item.manifest_hash)
    Write-Output ("[{0}] rule_params={1}" -f $item.profile_id, (($item.rule_params | ConvertTo-Json -Compress)))
    $args += @("--manifest", $item.manifest_path)
    Write-Output ""
}

$stdoutPath = Join-Path $root ("logs/shadow_sweep_{0}_stdout.log" -f $stamp)
$stderrPath = Join-Path $root ("logs/shadow_sweep_{0}_stderr.log" -f $stamp)

Write-Output "Starting shadow sweep in background"
Write-Output "Stdout: $stdoutPath"
Write-Output "Stderr: $stderrPath"

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $stdoutPath) | Out-Null
$proc = Start-Process -FilePath $python -ArgumentList $args -NoNewWindow -PassThru -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
Write-Output "PID: $($proc.Id)"
Write-Output "Stop with: Stop-Process -Id $($proc.Id)"
