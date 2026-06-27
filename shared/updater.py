"""
Lightweight updater utilities: check metadata, download asset, verify sha256, stage update.

This is intentionally conservative: it downloads and verifies update packages and stages
them under `updates/` but does not auto-replace application files. Installation must be
explicit (either run an installer or a separate deploy step).

Config (optional):
- updates.metadata_url (string): URL to JSON metadata describing latest release
- updates.download_dir (string): where to store staged updates (default: updates/)
- updates.auto_install (bool): not used by default (manual is safer)

Expected metadata format (example):
{
  "version": "1.2.3",
  "notes": "Fixes and improvements",
  "assets": [
    {"name": "HGCameraCounter-1.2.3.zip", "url": "https://.../HGCameraCounter-1.2.3.zip", "sha256": "..."}
  ]
}
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import urllib.request
import logging
import time
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

from shared.logger import setup_logger


# Code paths an OTA update is allowed to replace. Everything else on the device
# (.venv, data/, models/, logs/, reports/, snapshots/, config) is preserved.
CODE_UPDATE_PATHS = [
    "controller", "runtime", "shared", "services", "detectors",
    "tracking", "app", "ui", "scripts", "requirements.txt", "VERSION",
]


def parse_version(text) -> tuple:
    """Parse 'v1.2.3' / '1.2.3' -> (1, 2, 3). Non-numeric chunks become 0."""
    parts = []
    for chunk in str(text or "0").strip().lstrip("vV").split("."):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts) if parts else (0,)


def is_newer(latest, current) -> bool:
    """True if `latest` version string is strictly newer than `current`."""
    return parse_version(latest) > parse_version(current)


def read_version_file(root) -> Optional[str]:
    """Installed app version from the top-level VERSION file (shipped in CODE_UPDATE_PATHS).

    This is the authoritative current version: a code update overwrites VERSION, so the
    device stops re-offering the same release (config.yaml `version` is never rewritten
    by install_code_update, so it must NOT be the source of truth). Returns None if absent.
    """
    try:
        p = Path(root) / "VERSION"
        if p.exists():
            return p.read_text(encoding="utf-8").strip() or None
    except Exception:
        pass
    return None


class Updater:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        updates_cfg = self.config.get('updates', {}) if isinstance(self.config, dict) else {}
        self.download_dir = Path(updates_cfg.get('download_dir', 'updates'))
        self.download_dir.mkdir(parents=True, exist_ok=True)
        # Setup logger for updater using configured logs path when available
        logs_dir = None
        if isinstance(self.config, dict):
            logs_dir = self.config.get('paths', {}).get('logs') if self.config.get('paths') else None
        self.logger = setup_logger('updater', logs_dir or 'logs')
        self.logger.info(f'Updater initialized, download_dir={self.download_dir}')

    def check_for_update(self, metadata_url: str) -> Dict:
        """Fetch update metadata JSON from `metadata_url` and return parsed dict.

        Raises urllib.error.URLError on network problems or ValueError on bad JSON.
        """
        with urllib.request.urlopen(metadata_url, timeout=15) as r:
            data = r.read()
        return json.loads(data.decode('utf-8'))

    def select_primary_asset(self, metadata: Dict) -> Optional[Dict]:
        assets = metadata.get('assets') or []
        if not assets:
            return None
        # Prefer zip or exe, else first
        for ext in ('.zip', '.exe', '.tar.gz'):
            for a in assets:
                if a.get('name', '').endswith(ext):
                    return a
        return assets[0]

    def extract_archive(self, file_path: Path, dest_dir: Path) -> None:
        """Extract zip/tar archives into dest_dir."""
        self.logger.info(f'Extracting {file_path} -> {dest_dir}')
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.unpack_archive(str(file_path), str(dest_dir))
        except Exception as e:
            self.logger.error(f'Failed to extract archive: {e}')
            raise

    def install_update_atomic(self, file_path: Path, deploy_target: Optional[Path] = None, restart_command: Optional[list] = None) -> Path:
        """Perform atomic deploy of an update archive (zip/tar).

        Steps:
        - extract archive to a staging folder
        - move staging folder to a releases folder with timestamp
        - rename existing deploy_target to backup
        - move new release into place (rename)
        - optionally run restart_command

        Returns path to release directory on success.
        """
        self.logger.info(f'Atomic install requested: {file_path} to {deploy_target}')
        if not file_path.exists():
            raise FileNotFoundError(file_path)

        if deploy_target is None:
            deploy_target = Path('.').resolve()
        else:
            deploy_target = Path(deploy_target).resolve()

        now = int(time.time())
        staging = self.download_dir / f'staging_{now}'
        release = self.download_dir / 'releases' / f'release_{now}'
        release.parent.mkdir(parents=True, exist_ok=True)

        # extract
        self.extract_archive(file_path, staging)

        # Move staging to release (atomic move when possible)
        shutil.move(str(staging), str(release))
        self.logger.info(f'Release prepared at {release}')

        # Backup existing target
        backup = None
        if deploy_target.exists():
            backup = deploy_target.with_name(deploy_target.name + f'.bak.{now}')
            self.logger.info(f'Backing up {deploy_target} -> {backup}')
            shutil.move(str(deploy_target), str(backup))

        # Move new release into place
        try:
            self.logger.info(f'Deploying {release} -> {deploy_target}')
            shutil.move(str(release), str(deploy_target))
        except Exception as e:
            self.logger.error(f'Deploy move failed: {e}')
            # attempt rollback
            if backup and backup.exists():
                shutil.move(str(backup), str(deploy_target))
                self.logger.info('Rollback: restored backup')
            raise

        self.logger.info('Deploy completed')

        # Optionally restart services/processes
        if restart_command:
            try:
                self.logger.info(f'Running restart command: {restart_command}')
                subprocess.Popen(restart_command)
            except Exception as e:
                self.logger.error(f'Failed to run restart command: {e}')

        return deploy_target

    def download_asset(self, url: str, name: Optional[str] = None) -> Path:
        """Download asset to a temporary file, return Path to it.

        Caller should call `verify_sha256` before trusting the file.
        """
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp_path, 'wb') as fd:
            shutil.copyfileobj(r, fd)
        # rename to include original filename for clarity
        if name:
            dest = tmp_path.with_name(name)
            tmp_path.rename(dest)
            return dest
        return tmp_path

    def verify_sha256(self, file_path: Path, expected_hex: str) -> bool:
        h = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        got = h.hexdigest()
        return got.lower() == expected_hex.lower()

    def stage_update(self, file_path: Path) -> Path:
        """Move verified file into the downloads staging dir and return staged path."""
        target = self.download_dir / file_path.name
        shutil.move(str(file_path), str(target))
        return target

    def inspect_metadata_for_update(self, metadata: Dict) -> Tuple[Optional[str], Optional[str]]:
        """Return (version, notes) if present"""
        return (metadata.get('version'), metadata.get('notes'))

    def _find_code_root(self, staging: Path) -> Path:
        """Locate the code root inside an extracted archive.

        Handles archives that wrap everything in a single top-level folder.
        """
        for marker in ("controller", "requirements.txt", "VERSION"):
            if (staging / marker).exists():
                return staging
        entries = list(staging.iterdir())
        subdirs = [p for p in entries if p.is_dir()]
        if len(subdirs) == 1 and not any(p.is_file() for p in entries):
            return subdirs[0]
        return staging

    def install_code_update(self, archive_path: Path, project_root: Path,
                            version: Optional[str] = None) -> Dict:
        """Safely apply a code-only update.

        Copies ONLY whitelisted code paths (CODE_UPDATE_PATHS) from the archive
        over the install, backing up the replaced files first. The running venv,
        data/, models/, logs/, reports/ and local config are never touched, so the
        deploy cannot wipe data or move locked files (the failure mode of the old
        install_update_atomic on Windows). Returns {copied, backup, version}.
        """
        archive_path = Path(archive_path)
        project_root = Path(project_root).resolve()
        if not archive_path.exists():
            raise FileNotFoundError(archive_path)

        now = int(time.time())
        staging = self.download_dir / f"staging_{now}"
        backup = self.download_dir / "backup" / f"{version or now}"
        self.extract_archive(archive_path, staging)
        src_root = self._find_code_root(staging)

        copied, backed_up = [], []
        try:
            for rel in CODE_UPDATE_PATHS:
                src = src_root / rel
                if not src.exists():
                    continue
                dst = project_root / rel
                if dst.exists():
                    bdst = backup / rel
                    bdst.parent.mkdir(parents=True, exist_ok=True)
                    if dst.is_dir():
                        shutil.copytree(dst, bdst, dirs_exist_ok=True)
                    else:
                        shutil.copy2(dst, bdst)
                    backed_up.append(rel)
                if src.is_dir():
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                copied.append(rel)
            self.logger.info(
                f"Code update applied: copied={copied} backup={backup if backed_up else 'none'}"
            )
            return {"copied": copied, "backup": str(backup) if backed_up else None,
                    "version": version}
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def rollback_code_update(self, project_root: Path, backup_dir: Path) -> None:
        """Restore code paths from a backup created by install_code_update."""
        project_root = Path(project_root).resolve()
        backup_dir = Path(backup_dir)
        if not backup_dir.exists():
            raise FileNotFoundError(backup_dir)
        for item in backup_dir.iterdir():
            dst = project_root / item.name
            if item.is_dir():
                shutil.copytree(item, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dst)
        self.logger.info(f"Rolled back code from {backup_dir}")

    # Note: deliberate omission of auto-install to avoid overwriting runtime files without
    # explicit operator action. Implementers can add `install_update(path)` for platform-specific
    # installer execution.


if __name__ == '__main__':
    print('Updater module - import and use Updater(config)')
