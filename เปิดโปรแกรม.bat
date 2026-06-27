@echo off
REM Double-click to open HG Camera Counter (first time: sets the PIN; no command line needed).
cd /d "%~dp0"
start "" ".venv\Scripts\pythonw.exe" controller\main.py --autostart
