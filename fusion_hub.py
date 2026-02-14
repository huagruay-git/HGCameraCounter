# fusion_hub.py
# Local-only Fusion Hub:
# - Receives UDP packets from multiple agents
# - Cross-camera matching => global_id
# - Staff tagging:
#     - If in STAFF_AREA long enough => staff (ignored in counts)
# - "Still in shop" memory:
#     - If global_id last seen in SHOP recently => treat as still in shop (no re-count)
# - Outputs:
#     - prints occupancy every N seconds
#     - writes a local JSONL log (optional)
#
# NOTE: For testing with mp4, "in_shop" will be true if you provided SHOP polygon (or agent fallback = True).

import os, time, json, socket
from typing import Dict, Tuple, Optional

import math

FUSION_HOST = os.getenv("FUSION_HOST", "0.0.0.0")
FUSION_PORT = int(os.getenv("FUSION_PORT", "9999"))

PRINT_EVERY_SEC = float(os.getenv("PRINT_EVERY_SEC", "2.0"))
LOG_JSONL = os.getenv("LOG_JSONL", "").strip()  # e.g. ./fusion_log.jsonl (optional)

# Matching thresholds (in shop-map normalized coordinates)
MATCH_MAX_DIST = float(os.getenv("MATCH_MAX_DIST", "0.10"))      # distance on map
MATCH_MAX_SEC  = float(os.getenv("MATCH_MAX_SEC", "2.0"))        # time window

# Staff logic
STAFF_DWELL_SEC  = float(os.getenv("STAFF_DWELL_SEC", "10.0"))
STAFF_FORGET_SEC = float(os.getenv("STAFF_FORGET_SEC", "120.0"))

# Shop memory
SHOP_EXIT_GRACE_SEC = float(os.getenv("SHOP_EXIT_GRACE_SEC", "12.0"))

# Which zones to count as "customer zones"
COUNT_ZONES = os.getenv("COUNT_ZONES", "WAIT,WASH,CHAIR_1,CHAIR_2,CHAIR_3").split(",")

def d2(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    dx=a[0]-b[0]; dy=a[1]-b[1]
    return dx*dx+dy*dy

class FusionState:
    def __init__(self):
        self.next_gid = 1
        self.cam_vid_to_gid: Dict[Tuple[str,int], int] = {}

        self.gid_last_xy: Dict[int, Tuple[float,float]] = {}
        self.gid_last_ts: Dict[int, float] = {}

        # shop memory
        self.gid_shop_last: Dict[int, float] = {}

        # staff memory
        self.gid_staff_enter_ts: Dict[int, float] = {}
        self.gid_is_staff: Dict[int, bool] = {}
        self.gid_last_seen: Dict[int, float] = {}

        # latest zone membership
        self.gid_zones: Dict[int, set] = {}

    def alloc_gid(self) -> int:
        gid = self.next_gid
        self.next_gid += 1
        return gid

    def assign_gid(self, camera_id: str, vid: int, mx: float, my: float, ts: float) -> int:
        key = (camera_id, vid)
        if key in self.cam_vid_to_gid:
            gid = self.cam_vid_to_gid[key]
            self.gid_last_xy[gid] = (mx,my)
            self.gid_last_ts[gid] = ts
            self.gid_last_seen[gid] = ts
            return gid

        # try match to existing gid by proximity + time
        best_gid = None
        best_d2 = 1e9
        for gid, last_xy in self.gid_last_xy.items():
            dt = ts - self.gid_last_ts.get(gid, ts)
            if dt < 0 or dt > MATCH_MAX_SEC:
                continue
            dd = d2((mx,my), last_xy)
            if dd < best_d2:
                best_d2 = dd
                best_gid = gid

        if best_gid is not None and best_d2 <= (MATCH_MAX_DIST * MATCH_MAX_DIST):
            gid = best_gid
        else:
            gid = self.alloc_gid()
            self.gid_is_staff.setdefault(gid, False)

        self.cam_vid_to_gid[key] = gid
        self.gid_last_xy[gid] = (mx,my)
        self.gid_last_ts[gid] = ts
        self.gid_last_seen[gid] = ts
        return gid

    def mark_shop_seen(self, gid: int, ts: float):
        self.gid_shop_last[gid] = ts

    def is_inside_shop(self, gid: int, ts: float) -> bool:
        last = self.gid_shop_last.get(gid)
        if last is None:
            return False
        return (ts - last) <= SHOP_EXIT_GRACE_SEC

    def update_staff(self, gid: int, in_staff_area: bool, ts: float):
        # forget staff if unseen too long
        last_seen = self.gid_last_seen.get(gid, ts)
        if self.gid_is_staff.get(gid, False) and (ts - last_seen) > STAFF_FORGET_SEC:
            self.gid_is_staff[gid] = False
            self.gid_staff_enter_ts.pop(gid, None)

        if in_staff_area:
            if gid not in self.gid_staff_enter_ts:
                self.gid_staff_enter_ts[gid] = ts
            else:
                if (ts - self.gid_staff_enter_ts[gid]) >= STAFF_DWELL_SEC:
                    self.gid_is_staff[gid] = True
        else:
            # if not already staff, clear dwell timer
            if not self.gid_is_staff.get(gid, False):
                self.gid_staff_enter_ts.pop(gid, None)

    def is_staff(self, gid: int) -> bool:
        return bool(self.gid_is_staff.get(gid, False))

    def set_zones(self, gid: int, zones: set):
        self.gid_zones[gid] = zones


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((FUSION_HOST, FUSION_PORT))
    sock.setblocking(False)

    st = FusionState()
    last_print = 0.0
    logf = open(LOG_JSONL, "a", encoding="utf-8") if LOG_JSONL else None

    print(f"[hub] Listening UDP on {FUSION_HOST}:{FUSION_PORT}")
    print(f"[hub] MATCH_MAX_DIST={MATCH_MAX_DIST} MATCH_MAX_SEC={MATCH_MAX_SEC}")
    print(f"[hub] STAFF_DWELL_SEC={STAFF_DWELL_SEC} SHOP_EXIT_GRACE_SEC={SHOP_EXIT_GRACE_SEC}")
    print(f"[hub] COUNT_ZONES={COUNT_ZONES}")
    if LOG_JSONL:
        print(f"[hub] Logging -> {LOG_JSONL}")

    while True:
        now = time.time()
        # receive as many packets as possible
        while True:
            try:
                data, _addr = sock.recvfrom(2_000_000)
            except BlockingIOError:
                break

            msg = json.loads(data.decode("utf-8"))
            cam = msg["camera_id"]
            ts = float(msg["ts"])
            dets = msg.get("dets", [])

            for d in dets:
                vid = int(d["vid"])
                mx = float(d.get("mx", d.get("cx", 0.0)))
                my = float(d.get("my", d.get("cy", 0.0)))
                gid = st.assign_gid(cam, vid, mx, my, ts)

                # SHOP memory
                if bool(d.get("in_shop", True)):
                    st.mark_shop_seen(gid, ts)

                # STAFF tag
                st.update_staff(gid, bool(d.get("in_staff_area", False)), ts)

                # zones
                zones = set(d.get("in_zones", []))
                st.set_zones(gid, zones)

                if logf:
                    logf.write(json.dumps({
                        "ts": ts,
                        "camera_id": cam,
                        "vid": vid,
                        "gid": gid,
                        "mx": mx,
                        "my": my,
                        "in_shop": bool(d.get("in_shop", True)),
                        "in_staff_area": bool(d.get("in_staff_area", False)),
                        "zones": list(zones),
                        "is_staff": st.is_staff(gid),
                    }) + "\n")

        # compute occupancy
        if (now - last_print) >= PRINT_EVERY_SEC:
            last_print = now
            occ = {z: 0 for z in COUNT_ZONES}
            active_gids = []

            for gid, zones in st.gid_zones.items():
                if not st.is_inside_shop(gid, now):
                    continue
                if st.is_staff(gid):
                    continue
                # count if inside zone (any)
                for z in COUNT_ZONES:
                    if z in zones:
                        occ[z] += 1
                active_gids.append(gid)

            print(f"[hub] active_customers={len(set(active_gids))} occ={occ}")

        time.sleep(0.01)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[hub] bye")
