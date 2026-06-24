"""
Find IP cameras of all kinds on the local network — headless CLI.

Wraps shared/camera_discovery.py (multi-subnet + multi-port + multi-brand RTSP +
ONVIF WS-Discovery). Prints discovered cameras and ready-to-use RTSP URLs, and can
write them straight into data/config/config.yaml.

Examples
  # Scan every local subnet + ONVIF (default), Dahua/Imou + other brands:
  python scripts/discover_cameras.py --user admin --pass L2EA2EF7

  # Scan a specific subnet, NVR channels 1-5 (pulling through an NVR):
  python scripts/discover_cameras.py --subnets 10.66.0.0/24 --channels 1-5 --user admin --pass L2EA2EF7

  # Discover and add the results to config.yaml:
  python scripts/discover_cameras.py --user admin --pass L2EA2EF7 --write
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared import camera_discovery as cd  # noqa: E402


def _parse_channels(s: str) -> tuple[int, int]:
    s = (s or "1").strip()
    if "-" in s:
        a, b = s.split("-", 1)
        return int(a), int(b)
    return int(s), int(s)


def _parse_ports(s: str) -> list[int]:
    out: list[int] = []
    for tok in (s or "").split(","):
        tok = tok.strip()
        if tok.isdigit():
            p = int(tok)
            if 1 <= p <= 65535 and p not in out:
                out.append(p)
    return out


def _cam_name(ip: str) -> str:
    return "Camera_" + ip.replace(".", "_")


def main() -> int:
    ap = argparse.ArgumentParser(description="Discover IP cameras of all kinds")
    ap.add_argument("--subnets", default="auto",
                    help='"auto" (all local /24) or comma list e.g. 10.66.0.0/24,192.168.1.0/24')
    ap.add_argument("--ports", default="554,8554", help="comma list (default 554,8554)")
    ap.add_argument("--user", default="", help="camera/NVR username")
    ap.add_argument("--pass", dest="password", default="", help="camera/NVR password")
    ap.add_argument("--channels", default="1", help='channel range, e.g. "1" or "1-5"')
    ap.add_argument("--template", default="", help="extra custom RTSP path (tried first)")
    ap.add_argument("--timeout", type=int, default=1200, help="per-probe timeout ms")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--no-onvif", action="store_true", help="skip ONVIF WS-Discovery")
    ap.add_argument("--write", action="store_true", help="add results to config.yaml")
    ap.add_argument("--config", default="data/config/config.yaml")
    args = ap.parse_args()

    if args.subnets.strip().lower() == "auto":
        subnets = cd.list_local_subnets()
    else:
        subnets = [s.strip() for s in args.subnets.split(",") if s.strip()]

    ports = _parse_ports(args.ports) or list(cd.DEFAULT_PORTS)
    ch_lo, ch_hi = _parse_channels(args.channels)

    print(f"Subnets : {', '.join(subnets)}")
    print(f"Ports   : {ports}   Channels: {ch_lo}-{ch_hi}   ONVIF: {not args.no_onvif}")
    print("Scanning... (Ctrl+C to stop)\n")

    def on_found(r: dict) -> None:
        url = r["rtsp_url"] or "(set path manually)"
        print(f"  [+] {r['ip']:<15} :{r['port']:<5} {r['brand']:<11} {r['note']}")

    def on_progress(done: int, total: int, msg: str) -> None:
        if msg.endswith("scanned"):
            print(f"\r  ...{done}/{total}", end="", flush=True)

    results = cd.scan(
        subnets=subnets, username=args.user, password=args.password, ports=ports,
        channel_start=ch_lo, channel_end=ch_hi, extra_template=args.template,
        timeout_ms=args.timeout, max_workers=args.workers,
        use_onvif=not args.no_onvif, on_found=on_found, on_progress=on_progress,
    )

    print("\n\n=== RESULT ===")
    if not results:
        print("No cameras found. Tips: make sure this PC is ON the camera subnet "
              "(ping a camera first), try --subnets 10.66.0.0/24, or widen --channels.")
        return 0

    print(f"Found {len(results)} endpoint(s):\n")
    yaml_lines = ["cameras:"]
    for r in results:
        name = _cam_name(r["ip"])
        url = r["rtsp_url"]
        flag = "" if url else "  <-- no RTSP path matched; edit manually"
        print(f"  {r['ip']:<15} {r['brand']:<11} {r['note']}")
        if url:
            print(f"      {url}")
        yaml_lines += [
            f"  {name}:",
            f"    rtsp_url: \"{url}\"",
            f"    enabled: {'true' if url else 'false'}",
            f"    zones_file: \"data/zones/zones_{name}.json\"{flag}",
        ]

    print("\n--- paste-ready config block ---")
    print("\n".join(yaml_lines))

    if args.write:
        rc = _write_config(args.config, results)
        return rc
    else:
        print("\n(Use --write to add these to config.yaml automatically.)")
    return 0


def _write_config(config_path: str, results: list[dict]) -> int:
    try:
        from shared.config import Config
    except Exception as e:
        print(f"\nERROR: cannot import Config to write ({e}). Paste the block above manually.")
        return 1

    cfg = Config(config_path)
    cameras = cfg.get("cameras", {})
    if not isinstance(cameras, dict):
        cameras = {}
    existing_urls = {str(c.get("rtsp_url", "")).strip()
                     for c in cameras.values() if isinstance(c, dict)}

    added = skipped = 0
    for r in results:
        url = r["rtsp_url"].strip()
        if not url:
            skipped += 1
            continue
        if url in existing_urls:
            skipped += 1
            continue
        name = _cam_name(r["ip"])
        # avoid clobbering a differently-configured camera of the same name
        base, i = name, 2
        while name in cameras:
            name = f"{base}_{i}"; i += 1
        cameras[name] = {
            "rtsp_url": url,
            "enabled": True,
            "zones_file": f"data/zones/zones_{name}.json",
            "note": f"auto-discovered {r['brand']}",
        }
        existing_urls.add(url)
        added += 1

    cfg.set("cameras", cameras)
    cfg.save()
    print(f"\nconfig.yaml updated: +{added} camera(s), {skipped} skipped "
          f"({cfg.config_path})")
    print("NOTE: saving rewrites the YAML (comments are not preserved); a backup of "
          "your config is recommended before --write on a production device.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
