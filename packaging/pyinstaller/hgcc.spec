# -*- mode: python ; coding: utf-8 -*-
# Single-exe onedir build of the launcher -> HGCameraCounter.exe
# Build: .venv\Scripts\python -m PyInstaller --noconfirm --clean packaging/pyinstaller/hgcc.spec
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

ROOT = Path(SPECPATH).resolve().parents[1]
sys.setrecursionlimit(5000)

datas, binaries, hiddenimports = [], [], []

# ultralytics ships *.yaml configs it loads at runtime -> collect everything.
_d, _b, _h = collect_all("ultralytics")
datas += _d; binaries += _b; hiddenimports += _h

# The launcher reaches the real entry points via runpy (dynamic), so PyInstaller
# can't follow them statically -> force-include the app packages.
for pkg in ("controller", "runtime", "shared", "services", "detectors", "tracking", "app", "ui"):
    try:
        hiddenimports += collect_submodules(pkg)
    except Exception:
        pass

# Dynamically-imported deps PyInstaller tends to miss.
hiddenimports += [
    "supabase", "postgrest", "gotrue", "storage3", "realtime", "httpx", "httpcore",
    "websockets", "yaml", "scipy", "scipy.special", "sklearn.utils._typedefs",
    "PySide6.QtSvg", "PySide6.QtNetwork", "PySide6.QtPrintSupport",
]
try:
    datas += collect_data_files("timm")
except Exception:
    pass

a = Analysis(
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tensorflow", "PyQt5", "PyQt6", "PySide2", "tkinter",
              "pytest", "mypy", "mypyc", "matplotlib.tests", "notebook"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="HGCameraCounter",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,          # keep console while validating; flip to False for release
    disable_windowed_traceback=False,
    argv_emulation=False,
)
coll = COLLECT(
    exe, a.binaries, a.datas,
    strip=False, upx=False, upx_exclude=[],
    name="HGCameraCounter",
)
