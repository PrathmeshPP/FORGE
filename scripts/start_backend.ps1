# ==============================================================================
# F.O.R.G.E. Website — Start Backend Service (Loopback 127.0.0.1:8000)
# ==============================================================================

$workspaceRoot = Split-Path -Parent $PSScriptRoot
Set-Location $workspaceRoot

$venvPython = Join-Path $workspaceRoot ".venv\Scripts\python.exe"

if (!(Test-Path $venvPython)) {
    Write-Error "[ERROR] Virtual environment not found at $venvPython. Run 'py -3 -m venv .venv' first."
    exit 1
}

# Check if port 8000 is already occupied
$existing = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Where-Object { $_.State -eq 'Listen' }
if ($existing) {
    Write-Host "[*] Service is already listening on port 8000 (PID: $($existing.OwningProcess))."
    exit 0
}

Write-Host "==============================================================================="
Write-Host "  F.O.R.G.E. Website — Starting Local Workstation Backend"
Write-Host "  Binding: 127.0.0.1:8000 (Loopback only)"
Write-Host "  Data Root: C:\forge_data"
Write-Host "==============================================================================="

& $venvPython -m uvicorn web_backend.main:app --host 127.0.0.1 --port 8000

