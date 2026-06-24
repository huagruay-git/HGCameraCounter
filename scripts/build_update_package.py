"""
Build an OTA update package: a code-only zip + metadata JSON for shared.updater.

Produces (in --out-dir, default dist/):
  HGCameraCounter-<version>.zip   # only CODE_UPDATE_PATHS, minus venv/data/models/logs/state
  update_metadata.json            # {version, notes, assets:[{name,url,sha256}]}

The metadata `url` is <base-url>/<zip-name>; host both files together (e.g. a Supabase
Storage public bucket) and point config `updates.metadata_url` at the JSON. The device's
check_updates() compares the package version against config `version` and installs only
code — the venv, data/, models/ and config.yaml on the device are never touched.

Usage:
  python scripts/build_update_package.py --version 0.2.0 --notes "new feature" \
      --base-url https://<proj>.supabase.co/storage/v1/object/public/app-updates
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, os.path.abspath("."))
from shared.updater import CODE_UPDATE_PATHS

ROOT = Path(__file__).resolve().parent.parent

# Never ship transient/state/binary files even though they sit under code paths.
EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", "dist", "build", ".git", "node_modules"}
EXCLUDE_EXT = {".pyc", ".pyo", ".log", ".cache", ".pt", ".onnx", ".tmp", ".bak"}
EXCLUDE_NAMES = {
    "runtime_settings.override.json", "processor_sync_state.json",
    "chair_service_autotrain_state.json", "dashboard_state.json",
    "watchdog_state.json",
}


def _excluded(p: Path) -> bool:
    if any(part in EXCLUDE_DIRS for part in p.parts):
        return True
    if p.suffix.lower() in EXCLUDE_EXT:
        return True
    if p.name in EXCLUDE_NAMES or p.name.startswith("dashboard_state"):
        return True
    return False


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _add(zf: zipfile.ZipFile, src: Path, arc: Path) -> int:
    n = 0
    if src.is_dir():
        for p in sorted(src.rglob("*")):
            if p.is_file() and not _excluded(p.relative_to(ROOT)):
                zf.write(p, arc / p.relative_to(src))
                n += 1
    elif src.is_file() and not _excluded(src.relative_to(ROOT)):
        zf.write(src, arc)
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="Build an OTA code update package")
    ap.add_argument("--version", required=True)
    ap.add_argument("--notes", default="")
    ap.add_argument("--base-url", default="", help="where the zip will be hosted")
    ap.add_argument("--out-dir", default="dist")
    ap.add_argument("--onedir", default="",
                    help="package a PyInstaller onedir folder (e.g. dist/HGCameraCounter) for .exe self-update")
    args = ap.parse_args()

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    zip_name = f"HGCameraCounter-{args.version}.zip"
    zip_path = out / zip_name

    included, total_files = [], 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        if args.onedir:
            # Package a built PyInstaller onedir (exe + _internal) for frozen self-update.
            # Skip per-device state dirs — the self-update swap preserves those on the device.
            onedir = Path(args.onedir).resolve()
            _skip = {"data", "models", "logs", "reports", "snapshots", "tools", "updates", "experiments"}
            for p in sorted(onedir.rglob("*")):
                if not p.is_file():
                    continue
                rel = p.relative_to(onedir)
                if (rel.parts and rel.parts[0] in _skip) or p.suffix.lower() == ".zip":
                    continue
                zf.write(p, rel)
                total_files += 1
            included.append(f"onedir:{onedir.name} (app only, {total_files} files)")
        else:
            for rel in CODE_UPDATE_PATHS:
                src = ROOT / rel
                if src.exists():
                    c = _add(zf, src, Path(rel))
                    if c:
                        included.append(f"{rel}({c})")
                        total_files += c

    sha = _sha256(zip_path)
    base = args.base_url.rstrip("/")
    url = f"{base}/{zip_name}" if base else zip_name
    meta = {
        "version": args.version,
        "notes": args.notes,
        "kind": "onedir" if args.onedir else "source",
        "assets": [{"name": zip_name, "url": url, "sha256": sha}],
    }
    meta_path = out / "update_metadata.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Built {zip_path}  ({zip_path.stat().st_size // 1024} KB, {total_files} files)")
    print(f"  included : {', '.join(included)}")
    print(f"  sha256   : {sha}")
    print(f"  metadata : {meta_path}")
    print(f"  asset url: {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
