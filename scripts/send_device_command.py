"""
Send a remote command to a branch device, or list devices + online status.

Commands: reboot | restart_app | shutdown | power_cycle | wake | update_now | ping
The device polls its command queue on the same heartbeat channel and executes it.

AUTH: uses the Supabase SERVICE ROLE key (admin). Keep it SECRET — never put it on a
device. Provide via env SUPABASE_URL + SUPABASE_SERVICE_KEY, or --url/--service-key.

Examples:
  python scripts/send_device_command.py --list
  python scripts/send_device_command.py --device 1 --command reboot
  python scripts/send_device_command.py --device 1 --command restart_app
  # hard power-cycle via a smart plug the device can reach (PC must still be on):
  python scripts/send_device_command.py --device 1 --command power_cycle \
      --args '{"url":"http://10.66.0.9/relay/0?turn=off"}'
"""
from __future__ import annotations

import argparse
import json
import os
import sys

try:
    from supabase import create_client
except ImportError:
    create_client = None

VALID = {"reboot", "restart_app", "restart_pc", "shutdown", "power_cycle", "wake", "update_now", "ping"}


def main() -> int:
    ap = argparse.ArgumentParser(description="Send a remote command to a branch device")
    ap.add_argument("--url", default=os.getenv("SUPABASE_URL", ""))
    ap.add_argument("--service-key", default=os.getenv("SUPABASE_SERVICE_KEY", ""))
    ap.add_argument("--device", type=int, help="cctv_devices.id")
    ap.add_argument("--command", help="reboot|restart_app|shutdown|power_cycle|wake|update_now|ping")
    ap.add_argument("--args", default="{}", help="JSON args, e.g. '{\"delay_sec\":30}'")
    ap.add_argument("--expires", type=int, default=60, help="command expiry (minutes)")
    ap.add_argument("--list", action="store_true", help="list devices + last-seen status")
    args = ap.parse_args()

    if create_client is None:
        print("ERROR: supabase library not installed (pip install supabase)")
        return 2
    if not args.url or not args.service_key:
        print("ERROR: set SUPABASE_URL and SUPABASE_SERVICE_KEY (or pass --url/--service-key)")
        return 2

    client = create_client(args.url, args.service_key)

    if args.list or not (args.device and args.command):
        rows = client.table("cctv_devices").select(
            "id,branch_id,device_name,last_seen_at,is_active").order("last_seen_at", desc=True).execute()
        print(f"{'id':>3}  {'branch':>6}  {'name':<18}  {'active':<6}  last_seen_at (UTC)")
        for r in (rows.data or []):
            print(f"{r.get('id'):>3}  {str(r.get('branch_id')):>6}  "
                  f"{str(r.get('device_name'))[:18]:<18}  {str(r.get('is_active')):<6}  {r.get('last_seen_at')}")
        if not args.device or not args.command:
            print("\n(Use --device <id> --command <cmd> to send.)")
            return 0

    cmd = args.command.strip().lower()
    if cmd not in VALID:
        print(f"WARNING: '{cmd}' is not a known command {sorted(VALID)} — sending anyway.")
    try:
        cmd_args = json.loads(args.args or "{}")
    except json.JSONDecodeError as e:
        print(f"ERROR: --args is not valid JSON: {e}")
        return 2

    resp = client.rpc("admin_enqueue_cctv_command", {
        "p_device_id": args.device,
        "p_command": cmd,
        "p_args": cmd_args,
        "p_expires_in_minutes": args.expires,
    }).execute()
    print(f"Queued '{cmd}' for device {args.device}. command_id = {resp.data}")
    print("The device will pick it up within ~one heartbeat interval and execute it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
