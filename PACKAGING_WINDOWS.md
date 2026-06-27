# Windows EXE / Installer Guide

The app ships as a **single executable** (`HGCameraCounter.exe`) that runs three modes
from one bundle (so torch/ultralytics ship once):

| Command | Mode |
|---------|------|
| `HGCameraCounter.exe` | Controller GUI (default) — shows the PIN login gate |
| `HGCameraCounter.exe --runtime` | Counting runtime (processor) — spawned by the GUI |
| `HGCameraCounter.exe --recorder` | Clip recorder — spawned by the GUI |
| `HGCameraCounter.exe --model-ota` | Pull model/config OTA from Supabase (run on a schedule) |

## Prerequisites (Windows build machine)
- Python 3.11 (the project venv: `.venv\Scripts\python.exe`)
- Inno Setup 6 (for `Setup.exe`) — optional
- The trained models in `models\` and `tools\ffmpeg\` present

## 1) Build the EXE (PyInstaller)
```bat
packaging\windows\build_exe.bat
```
Output: `dist\HGCameraCounter\` (onedir) — `HGCameraCounter.exe` (~48 MB code) +
`_internal\` (torch/ultralytics/PySide6, ~900 MB total).

The build is driven by `packaging\pyinstaller\hgcc.spec` (entry `packaging\launcher.py`).

## 2) Build Setup.exe (Inno Setup) — optional
```bat
iscc packaging\windows\HGCameraCounter.iss
```
Output: `dist\installer\HGCameraCounter_Setup.exe`. Installs to `%LOCALAPPDATA%\HGCameraCounter`
(user scope — no admin needed; logs/reports/config are writable there).

## Install layout (next to the exe)
```
HGCameraCounter\
  HGCameraCounter.exe
  _internal\                      ← bundled deps (do not edit)
  models\*.pt                     ← YOLO + chair-service models
  tools\ffmpeg\...\bin\ffmpeg.exe ← camera capture
  data\config\config.yaml         ← PER-DEVICE (see provisioning) — NOT in the installer
  data\zones\                     ← per-camera zones
  logs\ reports\ snapshots\       ← created at runtime
```
The frozen app resolves these relative to the exe folder (see `_app_base()` in
`shared/config.py` and `_project_root()` in `controller/main.py`).

## Per-device provisioning (important — security)
`config.yaml` holds the Supabase anon key + `device_token`, which are **DPAPI-encrypted
and machine-bound** (see `shared/secure.py`). A config from one machine **cannot be
decrypted on another**, so the installer ships only `config.template.yaml`. On each
device:
1. Put the device's `config.yaml` in `…\HGCameraCounter\data\config\` (with its
   own `device_token` from the CCTV enrollment / `register_cctv_device`).
2. Run once to encrypt the secrets at rest:
   `HGCameraCounter.exe` is GUI-only, so encrypt with the venv on the build machine
   before copying, or ship a small `--encrypt-config` helper. (Currently:
   `python scripts/encrypt_config_secrets.py` on a machine with the venv.)
3. First launch asks the operator to set the **login PIN** (binds the app to that PC).

## Updating a deployed .exe (OTA)
- **App/code update** — host a new onedir package (`build_update_package.py --onedir dist\HGCameraCounter`)
  + metadata, then the GUI's **Install Update** swaps the whole install via an external
  helper (`shared/self_update.py`) and relaunches. Source (.py) installs use the
  loose-file `install_code_update` path instead.
- **Model/config update** — `HGCameraCounter.exe --model-ota` (schedule it) pulls the
  branch's active model + config from Supabase (device-token RPC), verifies sha256, swaps
  the model file, and hot-applies config. Works on the frozen .exe because models/config
  live next to the exe, not inside it.

## Notes
- The console window is on while validating; set `console=False` in `hgcc.spec` for a
  windowless release build.
