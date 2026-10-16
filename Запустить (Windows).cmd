@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set "HF_HOME=%CD%\.tools\hf"
if not exist ".venv\Scripts\python.exe" (
  echo Environment not found - running the installer first...
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1"
  if errorlevel 1 goto :fail
)
".venv\Scripts\python.exe" launch_reel_factory.py %*
if errorlevel 1 goto :fail
echo Reel Factory is running in the background: http://127.0.0.1:8765  (you can close this window)
timeout /t 5 >nul
exit /b 0
:fail
echo.
echo Failed to start. If this is the first time, run the installer first (Установить (Windows).cmd).
echo Details: server.err.log in this folder.
pause
exit /b 1
