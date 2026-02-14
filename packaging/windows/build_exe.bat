
@echo off
setlocal

REM Run from project root
cd /d "%~dp0\..\.."

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv\Scripts\python.exe not found. Create Windows venv first.
  exit /b 1
)

set PY=.venv\Scripts\python.exe

echo [1/4] Installing build tools...
%PY% -m pip install --upgrade pip setuptools wheel pyinstaller
if errorlevel 1 exit /b 1

echo [2/4] Building runtime_service...
%PY% -m PyInstaller --noconfirm --clean packaging\pyinstaller\runtime_service.spec
if errorlevel 1 exit /b 1

echo [3/4] Building HGCameraCounter...
%PY% -m PyInstaller --noconfirm --clean packaging\pyinstaller\controller.spec
if errorlevel 1 exit /b 1

echo [4/4] Placing runtime_service next to controller exe...
if not exist "dist\HGCameraCounter" (
  echo [ERROR] dist\HGCameraCounter not found.
  exit /b 1
)
if not exist "dist\runtime_service\runtime_service.exe" (
  echo [ERROR] dist\runtime_service\runtime_service.exe not found.
  exit /b 1
)
copy /Y "dist\runtime_service\runtime_service.exe" "dist\HGCameraCounter\runtime_service.exe" >nul
if errorlevel 1 exit /b 1

echo [DONE] Build complete.
echo Output folders:
echo   dist\runtime_service\
echo   dist\HGCameraCounter\

echo.
echo To create Setup.exe with Inno Setup:
echo   iscc packaging\windows\HGCameraCounter.iss

endlocal
