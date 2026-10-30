# ==============================================================================
# F.O.R.G.E. Website — Test Runner Script
# Runs Python backend tests and Frontend TypeScript / Oxlint verification.
# ==============================================================================

$workspaceRoot = Split-Path -Parent $PSScriptRoot
Set-Location $workspaceRoot

$venvPython = Join-Path $workspaceRoot ".venv\Scripts\python.exe"

Write-Host "==============================================================================="
Write-Host "  1. Running Backend Pytest Suite..."
Write-Host "==============================================================================="
& $venvPython -m pytest tests -v
if ($LASTEXITCODE -ne 0) {
    Write-Error "[FAIL] Backend tests failed."
    exit $LASTEXITCODE
}

Write-Host "`n==============================================================================="
Write-Host "  2. Running Frontend Typecheck & Build..."
Write-Host "==============================================================================="
Set-Location (Join-Path $workspaceRoot "viewer")
npm run build
if ($LASTEXITCODE -ne 0) {
    Write-Error "[FAIL] Frontend build failed."
    exit $LASTEXITCODE
}

Write-Host "`n==============================================================================="
Write-Host "  3. Running Oxlint..."
Write-Host "==============================================================================="
npm run lint
if ($LASTEXITCODE -ne 0) {
    Write-Error "[FAIL] Linter found issues."
    exit $LASTEXITCODE
}

Write-Host "`n[SUCCESS] All backend tests, builds, and lint audits passed cleanly!"

