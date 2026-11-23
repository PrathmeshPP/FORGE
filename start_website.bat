@echo off
setlocal enabledelayedexpansion
title F.O.R.G.E. 3D Model Navigator

cd /d "%~dp0"

echo ===============================================================================
echo   F.O.R.G.E. - 3D Digital Twin Model Navigator
echo   Workstation Server: http://127.0.0.1:8000
echo ===============================================================================
echo.

set "VENV_PYTHON=.venv\Scripts\python.exe"
if not exist "%VENV_PYTHON%" (
    echo [ERROR] Python virtual environment was not found at %VENV_PYTHON%.
    pause
    exit /b 1
)

:: Build frontend bundle if needed
if not exist "viewer\dist\index.html" (
    echo [*] Building frontend application bundle...
    cd viewer
    call npm run build
    cd ..
)

:: Check if server is already running on port 8000
set "PORT=8000"
set "SERVER_RUNNING=0"
for /f "tokens=5" %%a in ('netstat -aon ^| findstr /r ":%PORT%\>.*LISTENING"') do (
    set "SERVER_RUNNING=1"
)

if "!SERVER_RUNNING!"=="1" (
    echo [*] F.O.R.G.E. backend server is already active on port %PORT%.
) else (
    echo [*] Starting F.O.R.G.E. backend server on 127.0.0.1:%PORT%...
    start "FORGE Server" /min "%VENV_PYTHON%" -m uvicorn web_backend.main:app --host 127.0.0.1 --port %PORT%
    timeout /t 2 /nobreak >nul
)

set "APP_URL=http://localhost:%PORT%/"
echo.
echo ===============================================================================
echo   F.O.R.G.E. Navigator is LIVE at: %APP_URL%
echo ===============================================================================
echo.
echo   Press any key to open in default browser...
echo ===============================================================================
pause >nul
start "" "%APP_URL%"
