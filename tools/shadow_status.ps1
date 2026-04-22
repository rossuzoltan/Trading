param(
    [string]$ManifestPath = "models/rc1/eurusd_5k_v1_mr_rc1/manifest.json",
    [string]$ShadowRoot = "artifacts/shadow",
    [int]$TailLines = 5
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path $ManifestPath)) {
    throw "Manifest not found: $ManifestPath"
}

$manifest = Get-Content $ManifestPath -Raw | ConvertFrom-Json
$symbol = ($manifest.strategy_symbol | ForEach-Object { "$_".ToUpperInvariant() })
$manifestHash = "$($manifest.manifest_hash)"

$evidenceDir = Join-Path $ShadowRoot (Join-Path $symbol $manifestHash)
$eventsPath = Join-Path $evidenceDir "events.jsonl"
$summaryJson = Join-Path $evidenceDir "shadow_summary.json"
$summaryMd = Join-Path $evidenceDir "shadow_summary.md"

$pidRows = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object {
        (
            $_.CommandLine -like "*runtime*shadow_broker*" -or
            $_.CommandLine -like "*shadow_broker.py*" -or
            $_.CommandLine -like "*shadow_sweep_broker.py*"
        ) -and
        $_.CommandLine -like "*$ManifestPath*"
    } |
    Select-Object ProcessId, ExecutablePath, CommandLine

Write-Output "Shadow status: $symbol"
Write-Output "Manifest: $ManifestPath"
Write-Output "Manifest hash: $manifestHash"
Write-Output "Evidence dir: $evidenceDir"
Write-Output ""

if ($pidRows) {
    Write-Output "Running processes:"
    $pidRows | ForEach-Object { Write-Output ("  pid=" + $_.ProcessId + " exe=" + $_.ExecutablePath) }
} else {
    Write-Output "Running processes: none"
}

Write-Output ""
if (Test-Path $eventsPath) {
    $lineCount = (Get-Content $eventsPath | Measure-Object -Line).Lines
    $lastLine = Get-Content $eventsPath -Tail 1
    Write-Output "events.jsonl: present, lines=$lineCount, last_write=$((Get-Item $eventsPath).LastWriteTime)"
    if (-not [string]::IsNullOrWhiteSpace($lastLine)) {
        try {
            $last = $lastLine | ConvertFrom-Json
            Write-Output ("last_bar_ts=" + $last.bar_ts + " reason=" + $last.no_trade_reason)
            if ($null -ne $last.context_in_blackout) {
                Write-Output ("context: in_blackout=" + $last.context_in_blackout + " kind=" + $last.context_blackout_kind + " policy=" + $last.context_block_policy)
            }
        } catch {
            Write-Output "last_bar_ts=(unparsed)"
        }
    }
} else {
    Write-Output "events.jsonl: missing"
}

Write-Output ""
if (Test-Path $summaryJson) {
    $summary = Get-Content $summaryJson -Raw | ConvertFrom-Json
    $counts = $summary.counts
    $shortfall = $summary.evidence_shortfall
    Write-Output "shadow_summary.json: present, last_write=$((Get-Item $summaryJson).LastWriteTime)"
    Write-Output ("events=" + $summary.event_count + " trading_days=" + $summary.trading_days + " actionable=" + $summary.actionable_event_count)
    Write-Output ("context: macro_day_bars=" + $counts.context_macro_day_count + " blackout_bars=" + $counts.context_blackout_count + " blocked_entry=" + $counts.context_block_entry_count + " close_only_rev=" + $counts.context_close_only_reversal_count)
    Write-Output ("shortfall: days_remaining=" + $shortfall.trading_days_remaining + " actionable_remaining=" + $shortfall.actionable_events_remaining)
} else {
    Write-Output "shadow_summary.json: missing"
}

Write-Output ""
if (Test-Path $summaryMd) {
    Write-Output "shadow_summary.md (tail):"
    Get-Content $summaryMd -Tail $TailLines
} else {
    Write-Output "shadow_summary.md: missing"
}
