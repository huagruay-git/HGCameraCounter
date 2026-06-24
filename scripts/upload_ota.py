"""
Upload an OTA package (zip + update_metadata.json) to the Supabase `app-updates` bucket.

Hosts the release so devices' Check Updates / auto-update can fetch it. Uses the Supabase
Storage REST API with the SERVICE ROLE key (admin) — the only key allowed to WRITE this
bucket (anon stays read-only on purpose; granting anon write = supply-chain risk).

The service key stays on YOUR machine — pass it via env, never commit it:
  set SUPABASE_URL=https://doafupjlqkydaoxmsqtc.supabase.co
  set SUPABASE_SERVICE_KEY=<your service_role key>
  python scripts/upload_ota.py

By default uploads dist/HGCameraCounter-*.zip + dist/update_metadata.json (upsert).
"""
from __future__ import annotations

import argparse
import mimetypes
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _upload(url: str, key: str, bucket: str, name: str, data: bytes, content_type: str) -> None:
    endpoint = f"{url.rstrip('/')}/storage/v1/object/{bucket}/{name}"
    req = urllib.request.Request(endpoint, data=data, method="POST")
    req.add_header("Authorization", f"Bearer {key}")
    req.add_header("apikey", key)
    req.add_header("x-upsert", "true")
    req.add_header("Content-Type", content_type)
    with urllib.request.urlopen(req, timeout=120) as r:
        if r.status not in (200, 201):
            raise RuntimeError(f"HTTP {r.status}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Upload OTA package to Supabase Storage")
    ap.add_argument("--url", default=os.getenv("SUPABASE_URL", ""))
    ap.add_argument("--service-key", default=os.getenv("SUPABASE_SERVICE_KEY", ""))
    ap.add_argument("--bucket", default="app-updates")
    ap.add_argument("--dir", default="dist", help="folder holding the zip + metadata")
    ap.add_argument("--files", nargs="*", help="explicit files (default: newest zip + update_metadata.json)")
    args = ap.parse_args()

    if not args.url or not args.service_key:
        print("ERROR: set SUPABASE_URL and SUPABASE_SERVICE_KEY (or pass --url/--service-key).")
        print("       The service_role key is under Project Settings > API in the Supabase dashboard.")
        return 2

    out = (ROOT / args.dir)
    if args.files:
        files = [Path(f) for f in args.files]
    else:
        zips = sorted(out.glob("HGCameraCounter-*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
        meta = out / "update_metadata.json"
        if not zips or not meta.exists():
            print(f"ERROR: could not find a zip and update_metadata.json in {out}. Build first.")
            return 2
        files = [zips[0], meta]

    base_public = f"{args.url.rstrip('/')}/storage/v1/object/public/{args.bucket}"
    for f in files:
        if not f.exists():
            print(f"  SKIP (missing): {f}")
            continue
        ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        if f.suffix == ".zip":
            ctype = "application/zip"
        elif f.suffix == ".json":
            ctype = "application/json"
        try:
            _upload(args.url, args.service_key, args.bucket, f.name, f.read_bytes(), ctype)
            print(f"  uploaded: {f.name}  ->  {base_public}/{f.name}")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "ignore")[:300]
            print(f"  FAILED {f.name}: HTTP {e.code} {body}")
            return 1
        except Exception as e:
            print(f"  FAILED {f.name}: {e}")
            return 1

    print("\nDone. Point devices' updates.metadata_url at:")
    print(f"  {base_public}/update_metadata.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
