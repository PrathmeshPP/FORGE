# ==============================================================================
# F.O.R.G.E. Website — Stop Backend Service
# ==============================================================================

$port = 8000
$connection = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue | Where-Object { $_.State -eq 'Listen' }

if ($connection) {
    $pidToKill = $connection.OwningProcess
    $proc = Get-Process -Id $pidToKill -ErrorAction SilentlyContinue
    $cmdLine = (Get-CimInstance Win32_Process -Filter "ProcessId = $pidToKill" -ErrorAction SilentlyContinue).CommandLine

    # Safety check: Verify this is indeed the F.O.R.G.E. Python/uvicorn backend process
    $isForgeProcess = ($proc.ProcessName -match "python" -and ($cmdLine -match "web_backend" -or $cmdLine -match "uvicorn" -or $cmdLine -match "FORGE_website"))

    if ($isForgeProcess) {
        Write-Host "[*] Stopping verified F.O.R.G.E. backend process on port $port (PID: $pidToKill, Name: $($proc.ProcessName))..."
        Stop-Process -Id $pidToKill -Force -ErrorAction SilentlyContinue
        Write-Host "[+] F.O.R.G.E. backend stopped successfully."
    } else {
        Write-Warning "[!] Process on port $port (PID: $pidToKill, Name: $($proc.ProcessName)) does not match F.O.R.G.E. backend signature. Aborting termination to protect unrelated services."
        Write-Host "    CommandLine: $cmdLine"
    }
} else {
    Write-Host "[*] No backend service was listening on port $port."
}
