# Windows EXE / Installer Guide

## Prerequisites (Windows machine)
- Python 3.11 (same major/minor as project)
- Inno Setup 6 (for `Setup.exe`)
- Repo source code

## 1) Create virtual environment
```bat
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip setuptools wheel
.venv\Scripts\pip install -r requirements.txt
```

## 2) Build EXE files (PyInstaller)
```bat
packaging\windows\build_exe.bat
```

Expected output:
- `dist\HGCameraCounter\HGCameraCounter.exe`
- `dist\HGCameraCounter\runtime_service.exe` (copied automatically)
- `dist\runtime_service\runtime_service.exe`

## 3) Build Setup.exe (Inno Setup)
```bat
iscc packaging\windows\HGCameraCounter.iss
```

Expected output:
- `dist\installer\HGCameraCounter_Setup.exe`

## Runtime layout requirement
`runtime_service.exe` must stay next to `HGCameraCounter.exe` in install folder.
Controller will auto-launch runtime executable from the same directory.

## Notes
- Default install path is user scope:
  - `%LOCALAPPDATA%\HGCameraCounter`
- This avoids permission issues for logs/reports/snapshots/config writes.
- If your Supabase table is `vision_events`, set in `data/config/config.yaml`:
  - `supabase.events_table: vision_events`
