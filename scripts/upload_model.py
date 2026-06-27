"""Publish an AI model version so branch devices can download it from the GUI.

Company-side tool. Uploads a .pt to a PUBLIC Supabase Storage bucket and updates the
models manifest JSON that the device's "อัปเดตโมเดล" tab reads. Requires the Supabase
SERVICE key (never the anon key, and never ship the service key to devices).

Usage:
  set SUPABASE_SERVICE_KEY=<service-role key>
  python scripts/upload_model.py --model models/best.pt --version 2026-06-27 \
      --notes "better staff detection" --recommended

  # url defaults to data/config/config.yaml's supabase.url; override with --url.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _default_url() -> str:
    try:
        import yaml
        data = yaml.safe_load((ROOT / "data/config/config.yaml").read_text(encoding="utf-8")) or {}
        return str((data.get("supabase", {}) or {}).get("url", "") or "")
    except Exception:
        return ""


def _fetch_manifest(url: str) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
        if isinstance(data, list):
            data = {"models": data}
        return data if isinstance(data, dict) else {"models": []}
    except Exception:
        return {"models": []}


def main() -> int:
    ap = argparse.ArgumentParser(description="Publish a model version to the public manifest")
    ap.add_argument("--model", required=True, help="path to the .pt to upload")
    ap.add_argument("--version", required=True, help="version label, e.g. 2026-06-27 or 1.3.0")
    ap.add_argument("--notes", default="", help="release notes shown in the GUI")
    ap.add_argument("--recommended", action="store_true", help="mark this the recommended version")
    ap.add_argument("--name", default="best.pt", help="local filename devices save it as")
    ap.add_argument("--bucket", default="app-updates", help="public storage bucket")
    ap.add_argument("--prefix", default="models", help="path prefix inside the bucket")
    ap.add_argument("--url", default=os.getenv("SUPABASE_URL", "") or _default_url())
    ap.add_argument("--service-key", default=os.getenv("SUPABASE_SERVICE_KEY", ""))
    args = ap.parse_args()

    model = Path(args.model)
    if not model.exists():
        print(f"ERROR: model not found: {model}"); return 2
    if not args.url:
        print("ERROR: no Supabase url (--url or SUPABASE_URL or config.yaml)"); return 2
    if not args.service_key:
        print("ERROR: no service key (--service-key or SUPABASE_SERVICE_KEY)"); return 2

    url = args.url.rstrip("/")
    sha = _sha256(model)
    size = model.stat().st_size
    remote_model_path = f"{args.prefix}/{args.version}/{args.name}"
    manifest_path = f"{args.prefix}/models_manifest.json"
    public_base = f"{url}/storage/v1/object/public/{args.bucket}"
    model_url = f"{public_base}/{remote_model_path}"
    manifest_url = f"{public_base}/{manifest_path}"

    try:
        from supabase import create_client
    except Exception as e:
        print(f"ERROR: supabase package not available: {e}"); return 2
    client = create_client(url, args.service_key)

    print(f"Uploading {model} ({size} bytes) -> {args.bucket}/{remote_model_path}")
    data = model.read_bytes()
    try:
        client.storage.from_(args.bucket).upload(
            remote_model_path, data,
            {"content-type": "application/octet-stream", "upsert": "true"})
    except Exception as e:
        print(f"ERROR: model upload failed: {e}"); return 1

    # Merge into the manifest: drop any existing entry for this version, optionally
    # clear other 'recommended' flags, append this one, newest first.
    manifest = _fetch_manifest(manifest_url)
    models = [m for m in (manifest.get("models") or []) if isinstance(m, dict) and m.get("version") != args.version]
    if args.recommended:
        for m in models:
            m["recommended"] = False
    models.append({
        "version": args.version,
        "name": args.name,
        "url": model_url,
        "sha256": sha,
        "size": size,
        "notes": args.notes,
        "recommended": bool(args.recommended),
        "created": date.today().isoformat(),
    })
    models.sort(key=lambda m: str(m.get("created", "")), reverse=True)
    manifest_bytes = json.dumps({"models": models}, ensure_ascii=False, indent=2).encode("utf-8")

    print(f"Updating manifest ({len(models)} versions) -> {args.bucket}/{manifest_path}")
    try:
        client.storage.from_(args.bucket).upload(
            manifest_path, manifest_bytes,
            {"content-type": "application/json", "upsert": "true"})
    except Exception as e:
        print(f"ERROR: manifest upload failed: {e}"); return 1

    print("\nDone.")
    print(f"  model    : {model_url}")
    print(f"  manifest : {manifest_url}")
    print(f"  sha256   : {sha}")
    print("Devices will see it in the อัปเดตโมเดล tab after Refresh.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
