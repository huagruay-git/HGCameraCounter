@echo off
REM One-command source install for HG Camera Counter.
REM Double-click this file, or run:  setup.bat   (pass flags too, e.g. setup.bat -Watchdog)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
echo.
pause
