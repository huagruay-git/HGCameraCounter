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

    # Note: deliberate omission of auto-install to avoid overwriting runtime files without
    # explicit operator action. Implementers can add `install_update(path)` for platform-specific
    # installer execution.


if __name__ == '__main__':
    print('Updater module - import and use Updater(config)')
