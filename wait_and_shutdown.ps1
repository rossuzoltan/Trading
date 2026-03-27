# wait_and_shutdown.ps1
# Monitor the Dukascopy downloader and shutdown once complete.

$scriptName = "download_dukascopy.py"
$checkIntervalSeconds = 600  # 10 minutes

Write-Output "Starting monitor for $scriptName..."

while ($true) {
    # Check if the process is still active by looking at command lines
    $process = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*$scriptName*" }
    
    if (-not $process) {
        Write-Output "Process $scriptName NOT found. Initiating shutdown..."
        # Final safety check to avoid accidental shutdown if the script hasn't started yet
        # (Though we know it's already running in this session)
        Stop-Computer -Force
        break
    }
    
    Write-Output "Process still running. Sleeping for $checkIntervalSeconds seconds..."
    Start-Sleep -Seconds $checkIntervalSeconds
}
