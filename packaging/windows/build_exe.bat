@echo off
setlocal

REM Run from project root
cd /d "%~dp0\..\.."

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv\Scripts\python.exe not found. Create Windows venv first.
  exit /b 1
)

set PY=.venv\Scripts\python.exe

echo [1/2] Installing build tools...
%PY% -m pip install --upgrade pip setuptools wheel pyinstaller
if errorlevel 1 exit /b 1

echo [2/2] Building HGCameraCounter (single exe: GUI + --runtime + --recorder)...
%PY% -m PyInstaller --noconfirm --clean packaging\pyinstaller\hgcc.spec
if errorlevel 1 exit /b 1

if not exist "dist\HGCameraCounter\HGCameraCounter.exe" (
  echo [ERROR] dist\HGCameraCounter\HGCameraCounter.exe not found.
  exit /b 1
)

echo [DONE] Build complete.
echo   dist\HGCameraCounter\HGCameraCounter.exe
echo.
echo To create Setup.exe with Inno Setup:
echo   iscc packaging\windows\HGCameraCounter.iss

endlocal
