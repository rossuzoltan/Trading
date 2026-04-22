param(
    [string]$ManifestPath = "models/rc1/eurusd_5k_v1_mr_rc1/manifest.json",
    [string]$ShadowRoot = "artifacts/shadow",
    [string]$LogsDir = "logs",
    [int]$TailLines = 40,
    [int]$FreshLogSeconds = 180
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path $ManifestPath)) {
    throw "Manifest not found: $ManifestPath"
}

$manifestPathResolved = (Resolve-Path $ManifestPath).Path
$manifest = Get-Content $manifestPathResolved -Raw | ConvertFrom-Json
$symbol = "$($manifest.strategy_symbol)".ToUpperInvariant()
$manifestHash = "$($manifest.manifest_hash)"

$latestMetadataPath = Join-Path $root ("logs/shadow_latest_{0}_{1}.json" -f $symbol, $manifestHash)
$metadata = $null
if (Test-Path $latestMetadataPath) {
    $metadata = Get-Content $latestMetadataPath -Raw | ConvertFrom-Json
}

$effectiveShadowRoot = if ($metadata -and $metadata.audit_root) { "$($metadata.audit_root)" } else { $ShadowRoot }
$effectiveLogsDir = if ($LogsDir -match '^[A-Za-z]:\\') { $LogsDir } else { Join-Path $root $LogsDir }
$evidenceDir = if ($metadata -and $metadata.evidence_dir) {
    "$($metadata.evidence_dir)"
} else {
    Join-Path $effectiveShadowRoot (Join-Path $symbol $manifestHash)
}
$eventsPath = Join-Path $evidenceDir "events.jsonl"
$summaryPath = Join-Path $evidenceDir "shadow_summary.json"

$pidRows = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object {
        ($_.CommandLine -like "*runtime.shadow_broker*" -or $_.CommandLine -like "*shadow_broker.py*") -and
        $_.CommandLine -like "*$manifestPathResolved*"
    } |
    Select-Object ProcessId, ParentProcessId, ExecutablePath, CommandLine

$stderrPath = $null
if ($metadata -and $metadata.stderr_path -and (Test-Path "$($metadata.stderr_path)")) {
    $stderrPath = "$($metadata.stderr_path)"
} else {
    $stderrCandidate = Get-ChildItem -Path $effectiveLogsDir -Filter "shadow*_stderr.log" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($stderrCandidate) {
        $stderrPath = $stderrCandidate.FullName
    }
}

$logLines = @()
$logAgeSeconds = $null
$heartbeatLine = $null
$offsetLine = $null
$reconnectCount = 0
$errorCount = 0
$lastTickAgeSeconds = $null
$ticksInBar = $null
$ticksPerBar = $null
$detectedOffset = $null
if ($stderrPath -and (Test-Path $stderrPath)) {
    $stderrItem = Get-Item $stderrPath
    $logAgeSeconds = [int]((New-TimeSpan -Start $stderrItem.LastWriteTimeUtc -End (Get-Date).ToUniversalTime()).TotalSeconds)
    $logLines = @(Get-Content $stderrPath -Tail $TailLines)
    $heartbeatLine = $logLines | Where-Object { $_ -match 'shadow heartbeat ticks_fetched=' } | Select-Object -Last 1
    $offsetLine = $logLines | Where-Object { $_ -match 'MT5 server UTC offset hours=' } | Select-Object -Last 1
    $reconnectCount = @($logLines | Where-Object { $_ -match 'reconnect|initialize\(\) failed|login\(\) failed|copy_ticks_from\(\) returned None' }).Count
    $errorCount = @($logLines | Where-Object { $_ -match 'ERROR|Traceback|RuntimeError|FATAL' }).Count
    if ($heartbeatLine -and $heartbeatLine -match 'ticks_in_bar=(\d+)/(\d+).*last_tick_utc=([0-9T:\.\+\-Z]+)') {
        $ticksInBar = [int]$Matches[1]
        $ticksPerBar = [int]$Matches[2]
        try {
            $lastTickUtc = [datetimeoffset]::Parse($Matches[3]).UtcDateTime
            $lastTickAgeSeconds = [int]((New-TimeSpan -Start $lastTickUtc -End (Get-Date).ToUniversalTime()).TotalSeconds)
        } catch {
            $lastTickAgeSeconds = $null
        }
    }
    if ($offsetLine -and $offsetLine -match 'MT5 server UTC offset hours=([\-0-9]+)') {
        $detectedOffset = [int]$Matches[1]
    }
}

$eventLines = 0
$eventsAgeSeconds = $null
$summaryAgeSeconds = $null
$summary = $null
if (Test-Path $eventsPath) {
    $eventLines = (Get-Content $eventsPath | Measure-Object -Line).Lines
    $eventsAgeSeconds = [int]((New-TimeSpan -Start (Get-Item $eventsPath).LastWriteTimeUtc -End (Get-Date).ToUniversalTime()).TotalSeconds)
}
if (Test-Path $summaryPath) {
    $summary = Get-Content $summaryPath -Raw | ConvertFrom-Json
    $summaryAgeSeconds = [int]((New-TimeSpan -Start (Get-Item $summaryPath).LastWriteTimeUtc -End (Get-Date).ToUniversalTime()).TotalSeconds)
}

$brokenReasons = @()
$suspiciousReasons = @()

if ($errorCount -gt 0) {
    $brokenReasons += "error_lines_in_stderr"
}
if (-not $pidRows -and -not (Test-Path $eventsPath) -and -not $stderrPath) {
    $brokenReasons += "no_process_no_log_no_evidence"
}
if ($pidRows -and $logAgeSeconds -ne $null -and $logAgeSeconds -gt ($FreshLogSeconds * 5)) {
    $brokenReasons += "stale_stderr_log"
}
if ($detectedOffset -eq $null) {
    $suspiciousReasons += "missing_logged_utc_offset"
} elseif ([math]::Abs($detectedOffset) -gt 12) {
    $brokenReasons += "invalid_utc_offset"
}
if ($pidRows -and $logAgeSeconds -ne $null -and $logAgeSeconds -gt $FreshLogSeconds) {
    $suspiciousReasons += "stderr_not_fresh"
}
if ($pidRows -and -not $heartbeatLine) {
    $suspiciousReasons += "missing_heartbeat_line"
}
if ($lastTickAgeSeconds -ne $null -and $lastTickAgeSeconds -gt $FreshLogSeconds) {
    $suspiciousReasons += "last_tick_stale"
}
if ($reconnectCount -gt 0) {
    $suspiciousReasons += "reconnect_or_mt5_errors_seen"
}
if ((Test-Path $eventsPath) -and -not (Test-Path $summaryPath)) {
    $brokenReasons += "events_present_summary_missing"
}
if ($summary -and $summaryAgeSeconds -ne $null -and $summaryAgeSeconds -gt ($FreshLogSeconds * 5)) {
    $suspiciousReasons += "summary_stale"
}

$status = "HEALTHY"
if ($brokenReasons.Count -gt 0) {
    $status = "BROKEN"
} elseif ($suspiciousReasons.Count -gt 0) {
    $status = "SUSPICIOUS"
}

Write-Output "Operator shadow check: $status"
Write-Output "Manifest: $manifestPathResolved"
Write-Output "Manifest hash: $manifestHash"
Write-Output "Evidence dir: $evidenceDir"
Write-Output ""

if ($pidRows) {
    Write-Output "Process: RUNNING"
    $pidRows | ForEach-Object { Write-Output ("  pid=" + $_.ProcessId + " parent=" + $_.ParentProcessId + " exe=" + $_.ExecutablePath) }
} else {
    Write-Output "Process: NOT RUNNING"
}

Write-Output ""
Write-Output ("stderr: " + ($(if ($stderrPath) { $stderrPath } else { "missing" })))
if ($logAgeSeconds -ne $null) {
    Write-Output ("stderr_age_seconds=" + $logAgeSeconds)
}
if ($detectedOffset -ne $null) {
    Write-Output ("detected_utc_offset_hours=" + $detectedOffset)
}
if ($ticksInBar -ne $null -and $ticksPerBar -ne $null) {
    Write-Output ("ticks_in_bar=" + $ticksInBar + "/" + $ticksPerBar)
}
if ($lastTickAgeSeconds -ne $null) {
    Write-Output ("last_tick_age_seconds=" + $lastTickAgeSeconds)
}
Write-Output ("reconnect_or_mt5_error_lines=" + $reconnectCount)
Write-Output ("stderr_error_lines=" + $errorCount)

Write-Output ""
if (Test-Path $eventsPath) {
    Write-Output ("events.jsonl: present lines=" + $eventLines + " age_seconds=" + $eventsAgeSeconds)
} else {
    Write-Output "events.jsonl: missing"
}
if ($summary) {
    Write-Output ("shadow_summary.json: present age_seconds=" + $summaryAgeSeconds)
    Write-Output ("summary_events=" + $summary.event_count + " trading_days=" + $summary.trading_days + " actionable=" + $summary.actionable_event_count)
    Write-Output ("summary_blackout_bars=" + $summary.counts.context_blackout_count + " close_only_reversal=" + $summary.counts.context_close_only_reversal_count)
    Write-Output ("evidence_sufficient=" + $summary.evidence_sufficient + " days_remaining=" + $summary.evidence_shortfall.trading_days_remaining + " actionable_remaining=" + $summary.evidence_shortfall.actionable_events_remaining)
} else {
    Write-Output "shadow_summary.json: missing"
}

Write-Output ""
Write-Output "Healthy means:"
Write-Output "- process is running"
Write-Output "- stderr log is fresh"
Write-Output "- heartbeat is present with a recent last_tick_utc"
Write-Output "- UTC offset is logged and plausible"
Write-Output "- no reconnect/error loop is visible"
Write-Output "- if events exist, summary exists and is fresh"

Write-Output ""
Write-Output "Suspicious means:"
Write-Output "- process exists but log or tick freshness is slipping"
Write-Output "- reconnect/login/init noise appears in stderr"
Write-Output "- summary or evidence is stale relative to the running process"
Write-Output "- UTC offset line is missing"

Write-Output ""
Write-Output "Broken means:"
Write-Output "- traceback/error lines are present"
Write-Output "- summary is missing while events exist"
Write-Output "- no process/log/evidence chain is present"
Write-Output "- UTC offset is invalid"

if ($suspiciousReasons.Count -gt 0) {
    Write-Output ""
    Write-Output ("Suspicious reasons: " + ($suspiciousReasons -join ", "))
}
if ($brokenReasons.Count -gt 0) {
    Write-Output ""
    Write-Output ("Broken reasons: " + ($brokenReasons -join ", "))
}
