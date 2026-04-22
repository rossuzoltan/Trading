param(
    [string]$ManifestDir = "models/rc1/eurusd_5k_v1_mr_rc1",
    [string]$BaselineShadowRoot = "artifacts/shadow",
    [string]$TunedShadowRoot = "artifacts/shadow_tuned",
    [string]$SweepShadowRoot = "artifacts/shadow_sweep",
    [int]$TailLines = 3
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$statusScript = Join-Path $root "tools/shadow_status.ps1"
$entries = @()

$baselineManifest = Join-Path $ManifestDir "manifest.json"
if (Test-Path $baselineManifest) {
    $entries += [PSCustomObject]@{
        Label = "baseline"
        ManifestPath = $baselineManifest
        ShadowRoot = $BaselineShadowRoot
    }
}

$tunedManifest = Join-Path $ManifestDir "manifest.tuned.json"
if (Test-Path $tunedManifest) {
    $entries += [PSCustomObject]@{
        Label = "tuned_balanced"
        ManifestPath = $tunedManifest
        ShadowRoot = $TunedShadowRoot
    }
}

Get-ChildItem $ManifestDir -Filter "manifest.shadow_*.json" -File -ErrorAction SilentlyContinue |
    Sort-Object Name |
    ForEach-Object {
        $entries += [PSCustomObject]@{
            Label = $_.BaseName.Replace("manifest.", "")
            ManifestPath = $_.FullName
            ShadowRoot = $SweepShadowRoot
        }
    }

foreach ($entry in $entries) {
    Write-Output ("=== {0} ===" -f $entry.Label)
    & $statusScript -ManifestPath $entry.ManifestPath -ShadowRoot $entry.ShadowRoot -TailLines $TailLines
    Write-Output ""
}
