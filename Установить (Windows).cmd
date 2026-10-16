@echo off
chcp 65001 >nul
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1"
set RC=%ERRORLEVEL%
echo.
if not %RC%==0 goto :fail
echo Installation finished. Now double-click the launcher file (starts with Запустить).
goto :end
:fail
echo Installation FAILED, code %RC%. Scroll up to see the reason.
:end
pause
exit /b %RC%
