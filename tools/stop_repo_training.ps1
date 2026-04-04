param(
    [string]$RepoRoot = "C:\\dev\\trading"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "SilentlyContinue"

function Stop-ProcessTree {
    param([int]$RootPid)

    $procs = Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,CommandLine
    $children = @{}
    foreach ($p in $procs) {
        $ppid = [int]$p.ParentProcessId
        if (-not $children.ContainsKey($ppid)) {
            $children[$ppid] = New-Object System.Collections.Generic.List[int]
        }
        $children[$ppid].Add([int]$p.ProcessId)
    }

    $toStop = New-Object System.Collections.Generic.List[int]
    $stack = New-Object System.Collections.Generic.Stack[int]
    $stack.Push([int]$RootPid)
    while ($stack.Count -gt 0) {
        $pid = $stack.Pop()
        if ($toStop.Contains($pid)) { continue }
        $toStop.Add($pid)
        if ($children.ContainsKey($pid)) {
            foreach ($c in $children[$pid]) { $stack.Push([int]$c) }
        }
    }

    foreach ($pid in ($toStop | Sort-Object -Descending)) {
        Stop-Process -Id $pid -Force | Out-Null
    }
}

$repo = [System.IO.Path]::GetFullPath($RepoRoot)
$ctxPath = Join-Path $repo "checkpoints\\current_training_run.json"
$trainPidPath = Join-Path $repo "train_pid.txt"
$stoppedAtUtc = (Get-Date).ToUniversalTime().ToString("o")

# 1) Stop PID recorded by current run context (authoritative)
if (Test-Path $ctxPath) {
    try {
        $ctx = Get-Content -LiteralPath $ctxPath -Raw | ConvertFrom-Json
        if ($ctx -and $ctx.pid) {
            Stop-ProcessTree -RootPid ([int]$ctx.pid)
        }
    } catch { }
}

# 2) Stop PID recorded by train_pid.txt (best-effort)
if (Test-Path $trainPidPath) {
    try {
        $pid2 = (Get-Content -LiteralPath $trainPidPath -Raw).Trim()
        if ($pid2) {
            Stop-ProcessTree -RootPid ([int]$pid2)
        }
    } catch { }
}

# 3) Stop any remaining python that is still running train_agent.py and is likely descended from this repo run
$all = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Select-Object ProcessId,ParentProcessId,CommandLine
$byPid = @{}
foreach ($p in $all) { $byPid[[int]$p.ProcessId] = $p }

function IsLikelyRepoTrainAgent {
    param([int]$Pid)
    $maxHops = 20
    $cur = $Pid
    for ($i=0; $i -lt $maxHops; $i++) {
        if (-not $byPid.ContainsKey($cur)) { return $false }
        $cmd = [string]($byPid[$cur].CommandLine)
        if ($cmd -match [regex]::Escape($repo)) { return $true }
        $cur = [int]$byPid[$cur].ParentProcessId
        if ($cur -le 0) { break }
    }
    return $false
}

$targets = @()
foreach ($p in $all) {
    $cmd = [string]$p.CommandLine
    if ($cmd -notmatch "train_agent\\.py") { continue }
    if ($cmd -match [regex]::Escape($repo) -or (IsLikelyRepoTrainAgent -Pid ([int]$p.ProcessId))) {
        $targets += [int]$p.ProcessId
    }
}

foreach ($pid3 in ($targets | Sort-Object -Unique)) {
    Stop-ProcessTree -RootPid $pid3
}

if (Test-Path $ctxPath) {
    try {
        $ctxText = Get-Content -LiteralPath $ctxPath -Raw
        $ctx = $ctxText | ConvertFrom-Json
        if ($ctx) {
            $ctx | Add-Member -NotePropertyName state -NotePropertyValue "stopped" -Force
            $ctx | Add-Member -NotePropertyName updated_at_utc -NotePropertyValue $stoppedAtUtc -Force
            $ctx | Add-Member -NotePropertyName stopped_at_utc -NotePropertyValue $stoppedAtUtc -Force
            $ctx | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $ctxPath -Encoding utf8
        }
    } catch { }
}

"Stopped repo training processes (if any)."
