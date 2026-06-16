"""
FFmpeg discovery/installer for Windows deployments.

Behavior:
- Check configured path in config (`runtime.ffmpeg_path` or `paths.ffmpeg`)
- Check PATH (`ffmpeg`, `ffmpeg.exe`)
- Check common local paths under project
- On Windows, optionally download and extract FFmpeg, then write runtime.ffmpeg_path
"""

from __future__ import annotations

import logging
import os
import subprocess
import shutil
import zipfile
from urllib.request import Request, urlopen
from pathlib import Path
from typing import Any, Optional, Tuple


FFMPEG_ZIP_CANDIDATE_URLS = [
    "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
    "https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-essentials.zip",
]


def _project_root(project_root: Optional[Path] = None) -> Path:
    if project_root is not None:
        return Path(project_root)
    return Path(__file__).resolve().parent.parent


def _log(logger: Optional[logging.Logger], level: str, msg: str) -> None:
    if logger is None:
        return
    fn = getattr(logger, level, None)
    if callable(fn):
        fn(msg)


def _config_get(config_obj: Any, key: str, default=None):
    try:
        return config_obj.get(key, default)
    except Exception:
        return default


def _save_ffmpeg_path_to_config(config_obj: Any, ffmpeg_path: str, logger: Optional[logging.Logger] = None) -> None:
    if config_obj is None:
        return
    try:
        runtime_cfg = dict(_config_get(config_obj, "runtime", {}) or {})
        runtime_cfg["ffmpeg_path"] = ffmpeg_path
        if hasattr(config_obj, "set"):
            config_obj.set("runtime", runtime_cfg)
            if hasattr(config_obj, "save"):
                config_obj.save()
        elif isinstance(config_obj, dict):
            config_obj["runtime"] = runtime_cfg
        _log(logger, "info", f"Configured runtime.ffmpeg_path = {ffmpeg_path}")
    except Exception as e:
        _log(logger, "warning", f"Failed to persist ffmpeg_path to config: {e}")


def find_ffmpeg_binary(config_obj: Any = None, project_root: Optional[Path] = None) -> Optional[str]:
    root = _project_root(project_root)

    if config_obj is not None:
        runtime_cfg = _config_get(config_obj, "runtime", {}) or {}
        paths_cfg = _config_get(config_obj, "paths", {}) or {}
        configured = runtime_cfg.get("ffmpeg_path") or paths_cfg.get("ffmpeg")
        if configured:
            p = Path(str(configured)).expanduser()
            if p.exists():
                return str(p)

    for name in ("ffmpeg", "ffmpeg.exe"):
        found = shutil.which(name)
        if found:
            return found

    candidates = [
        root / "ffmpeg.exe",
        root / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe",
        root / "bin" / "ffmpeg.exe",
    ]

    tools_root = root / "tools" / "ffmpeg"
    if tools_root.exists():
        # Search extracted layouts like ffmpeg-*-essentials_build/bin/ffmpeg.exe
        candidates.extend(sorted(tools_root.glob("**/bin/ffmpeg.exe")))

    for c in candidates:
        if c.exists():
            return str(c)
    return None


def _download_and_extract_ffmpeg_windows(root: Path, logger: Optional[logging.Logger] = None) -> Optional[str]:
    tools_dir = root / "tools" / "ffmpeg"
    tools_dir.mkdir(parents=True, exist_ok=True)
    zip_path = tools_dir / "ffmpeg_download.zip"
    extract_dir = tools_dir / "downloads"
    extract_dir.mkdir(parents=True, exist_ok=True)

    last_err = None
    for url in FFMPEG_ZIP_CANDIDATE_URLS:
        try:
            _log(logger, "info", f"Downloading FFmpeg from {url}")
            _download_file(url, zip_path, logger=logger, timeout_sec=25)
            _log(logger, "info", f"Downloaded FFmpeg archive to {zip_path}")
            break
        except Exception as e:
            last_err = e
            _log(logger, "warning", f"FFmpeg download failed from {url}: {e}")
    else:
        _log(logger, "error", f"All FFmpeg download URLs failed: {last_err}")
        return None

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        _log(logger, "info", f"Extracted FFmpeg archive to {extract_dir}")
    except Exception as e:
        _log(logger, "error", f"Failed to extract FFmpeg archive: {e}")
        return None

    found = find_ffmpeg_binary(project_root=root)
    if found:
        return found
    return None


def _download_file(url: str, dest_path: Path, logger: Optional[logging.Logger] = None, timeout_sec: int = 25) -> None:
    """Download with progress + timeout. Fallback to curl.exe on Windows."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest_path.with_suffix(dest_path.suffix + ".part")
    if tmp.exists():
        try:
            tmp.unlink()
        except Exception:
            pass

    # Primary: urllib with explicit timeout and user-agent
    try:
        req = Request(url, headers={"User-Agent": "HGCameraCounter/1.0"})
        with urlopen(req, timeout=timeout_sec) as resp, open(tmp, "wb") as out:
            total = resp.headers.get("Content-Length")
            total_i = int(total) if total and str(total).isdigit() else 0
            downloaded = 0
            next_log = 5 * 1024 * 1024
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_log:
                    if total_i > 0:
                        pct = (downloaded / total_i) * 100.0
                        _log(logger, "info", f"FFmpeg download progress: {pct:.1f}% ({downloaded // (1024*1024)} MB)")
                    else:
                        _log(logger, "info", f"FFmpeg download progress: {downloaded // (1024*1024)} MB")
                    next_log += 5 * 1024 * 1024
        if tmp.stat().st_size < 1024 * 1024:
            raise RuntimeError("Downloaded FFmpeg archive is unexpectedly small")
        tmp.replace(dest_path)
        return
    except Exception as e:
        _log(logger, "warning", f"urllib download failed: {e}")
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass

    # Fallback: curl.exe (usually present on Windows 10+)
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not curl:
        raise RuntimeError("urllib download failed and curl is not available")
    cmd = [curl, "-L", "--fail", "--connect-timeout", str(timeout_sec), "-o", str(tmp), url]
    _log(logger, "info", f"Retrying FFmpeg download with curl: {' '.join(cmd[:4])} ...")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"curl download failed ({proc.returncode}): {proc.stderr.strip()[:300]}")
    if not tmp.exists() or tmp.stat().st_size < 1024 * 1024:
        raise RuntimeError("curl download produced an invalid/small file")
    tmp.replace(dest_path)


def ensure_ffmpeg_available(
    config_obj: Any = None,
    project_root: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
    auto_download: bool = True,
) -> Tuple[Optional[str], str]:
    """
    Returns (path_or_none, message).
    """
    root = _project_root(project_root)
    found = find_ffmpeg_binary(config_obj=config_obj, project_root=root)
    if found:
        _save_ffmpeg_path_to_config(config_obj, found, logger=logger)
        return found, "found"

    if os.name != "nt" or not auto_download:
        return None, "not_found"

    downloaded = _download_and_extract_ffmpeg_windows(root, logger=logger)
    if downloaded:
        _save_ffmpeg_path_to_config(config_obj, downloaded, logger=logger)
        return downloaded, "downloaded"

    return None, "download_failed"
