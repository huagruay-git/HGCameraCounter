"""
Self-update for the frozen (.exe) build: swap the whole onedir install in place.

A running .exe can't overwrite its own files, so a code/app update means:
  1) download + sha256-verify the new onedir package  (shared.updater)
  2) extract it to a staging folder                    (stage_onedir)
  3) hand off to an EXTERNAL PowerShell helper that waits for this process to exit,
     mirrors the new app over the install dir (preserving the device's
     data/models/config/logs), then relaunches the exe   (spawn_swap_and_relaunch)

Only used when getattr(sys, "frozen", False). Source installs use
shared.updater.install_code_update (loose .py copy) instead.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

# Under the install dir, these are the device's OWN state — never overwritten/purged
# when the app folder is swapped.
PRESERVE_DIRS = ["data", "models", "logs", "reports", "snapshots", "tools", "updates", "experiments"]
PRESERVE_FILES = ["config.yaml"]


def stage_onedir(archive_path, staging_root=None) -> Path:
    """Extract a onedir update zip; return the folder that contains the new exe."""
    archive_path = Path(archive_path)
    staging_root = Path(staging_root or tempfile.mkdtemp(prefix="hgcc_update_"))
    staging_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(staging_root)
    candidates = [staging_root, *[p for p in staging_root.iterdir() if p.is_dir()]]
    for cand in candidates:
        if any(cand.glob("*.exe")):
            return cand
    return staging_root


def spawn_swap_and_relaunch(staging_dir, install_dir, exe_name, parent_pid=None) -> Path:
    """Write + launch a detached PowerShell helper that performs the swap.

    The caller MUST exit right after this returns so the helper can replace files.
    """
    staging_dir = Path(staging_dir).resolve()
    install_dir = Path(install_dir).resolve()
    parent_pid = int(parent_pid or os.getpid())

    xd = " ".join(f'"{install_dir / d}"' for d in PRESERVE_DIRS)
    xf = " ".join(f'"{install_dir / f}"' for f in PRESERVE_FILES)
    target_exe = install_dir / exe_name

    ps1 = (
        "$ErrorActionPreference='SilentlyContinue'\n"
        f"for ($i=0; $i -lt 240; $i++) {{ if (-not (Get-Process -Id {parent_pid} "
        "-ErrorAction SilentlyContinue)) { break }; Start-Sleep -Milliseconds 500 }\n"
        "Start-Sleep -Seconds 1\n"
        f'robocopy "{staging_dir}" "{install_dir}" /MIR /R:2 /W:1 /NFL /NDL /NJH /NJS '
        f"/XD {xd} /XF {xf} | Out-Null\n"
        "Start-Sleep -Seconds 1\n"
        f'Start-Process -FilePath "{target_exe}" -WorkingDirectory "{install_dir}"\n'
    )
    helper = Path(tempfile.gettempdir()) / f"hgcc_selfupdate_{int(time.time())}.ps1"
    helper.write_text(ps1, encoding="utf-8")

    DETACHED_PROCESS = 0x00000008
    CREATE_NEW_PROCESS_GROUP = 0x00000200
    subprocess.Popen(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-WindowStyle", "Hidden", "-File", str(helper)],
        creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
        close_fds=True,
    )
    return helper


def apply_frozen_update(archive_path) -> Path:
    """Stage a verified onedir package and launch the swap helper. Caller then exits."""
    if not getattr(sys, "frozen", False):
        raise RuntimeError("apply_frozen_update is only valid in a frozen (.exe) build")
    install_dir = Path(sys.executable).resolve().parent
    exe_name = Path(sys.executable).name
    staging = stage_onedir(archive_path)
    return spawn_swap_and_relaunch(staging, install_dir, exe_name)
