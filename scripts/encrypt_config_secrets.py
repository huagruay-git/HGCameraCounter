"""
One-time migration: DPAPI-encrypt the machine-bound secrets in config.yaml.

Encrypts `supabase.key` (anon key) and `supabase.cloud_sync.device_token` in place
so they are no longer stored as plaintext. DPAPI uses the machine scope, so after
this the config only works on THIS machine — a copied config cannot decrypt the
secrets (anti-theft / device binding).

Safe to run repeatedly: already-encrypted values are skipped. A timestamped backup
of the original (still-plaintext) file is written first.

Usage:
  python scripts/encrypt_config_secrets.py                  # data/config/config.yaml
  python scripts/encrypt_config_secrets.py --config X.yaml  # a specific file
  python scripts/encrypt_config_secrets.py --dry-run        # report only, no write
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath("."))
import yaml
from shared.secure import dpapi_available, encrypt_secret, is_encrypted

SECRET_PATHS = (
    ("supabase", "key"),
    ("supabase", "cloud_sync", "device_token"),
)


def _resolve(d, path):
    """Return (parent_dict, leaf_key) for a nested path, or (None, None)."""
    for p in path[:-1]:
        d = d.get(p) if isinstance(d, dict) else None
        if not isinstance(d, dict):
            return None, None
    return d, path[-1]


def main() -> int:
    ap = argparse.ArgumentParser(description="DPAPI-encrypt secrets in config.yaml")
    ap.add_argument("--config", default="data/config/config.yaml")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not dpapi_available():
        print("ERROR: DPAPI is only available on Windows. Aborting.")
        return 2

    path = Path(args.config)
    if not path.exists():
        print(f"ERROR: config not found: {path}")
        return 2

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    todo = []
    for sp in SECRET_PATHS:
        parent, leaf = _resolve(data, sp)
        if parent is None:
            continue
        val = parent.get(leaf)
        if isinstance(val, str) and val and not is_encrypted(val):
            todo.append((sp, parent, leaf, val))

    if not todo:
        print("Nothing to do: all target secrets are already encrypted (or absent).")
        return 0

    print("Will encrypt:")
    for sp, _, _, val in todo:
        print(f"  - {'.'.join(sp)}  (len={len(val)}, preview={val[:6]}...)")

    if args.dry_run:
        print("\n(dry-run) no changes written.")
        return 0

    backups = path.parent / "backups"   # gitignored — keeps the plaintext copy out of git
    backups.mkdir(parents=True, exist_ok=True)
    bak = backups / f"{path.name}.plain.bak.{int(time.time())}"
    shutil.copy2(path, bak)
    print(f"\nBackup of plaintext original: {bak}")

    for sp, parent, leaf, val in todo:
        parent[leaf] = encrypt_secret(val)

    path.write_text(
        yaml.safe_dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False),
        encoding="utf-8")
    print(f"Encrypted {len(todo)} secret(s) in {path}")
    print("NOTE: the .plain.bak file still contains plaintext — move it offline or delete it once verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
