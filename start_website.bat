@echo off
setlocal enabledelayedexpansion
title F.O.R.G.E. Indoor 3D Reconstruction Platform

cd /d "%~dp0"

echo ===============================================================================
echo   F.O.R.G.E. — Indoor 3D Digital Twin Reconstruction Platform
echo   Local Workstation Launcher (127.0.0.1:8000)
echo ===============================================================================
echo.

set "VENV_PYTHON=.venv\Scripts\python.exe"
if not exist "%VENV_PYTHON%" (
    echo [ERROR] Python virtual environment was not found at %VENV_PYTHON%.
    echo Please run 'py -3 -m venv .venv' and install requirements.txt.
    echo.
    pause
    exit /b 1
)

:: Ensure frontend production build exists
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
echo   F.O.R.G.E. is live at: %APP_URL%
echo ===============================================================================
echo.
echo Opening default web browser...
start "" "%APP_URL%"
echo.
pause

