import os
import time
import json
import threading
import csv
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import numpy as np
import cv2
from ultralytics import YOLO
import torch

# =========================
# CONFIG
# =========================
CAMERAS_CONFIG = {
    "Camera_01": os.getenv("CAM1_URL", "rtsp://admin:112113114@192.168.1.24:554/ch01/0"),
    "Camera_02": os.getenv("CAM2_URL", "rtsp://admin:112113114@192.168.1.83:554/ch01/0"),
    "Camera_03": os.getenv("CAM3_URL", ""),  # ว่างได้ ถ้าไม่ใช้
}

USE_VIDEO = os.getenv("USE_VIDEO", "0") == "1"
CAM1_VIDEO = os.getenv("CAM1_VIDEO", "cam1.mp4")
CAM2_VIDEO = os.getenv("CAM2_VIDEO", "cam2.mp4")
CAM3_VIDEO = os.getenv("CAM3_VIDEO", "cam3.mp4")

# YOLO
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL", "yolov8m.pt")
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.35"))
YOLO_IOU = float(os.getenv("YOLO_IOU", "0.5"))
IMGSZ = int(os.getenv("IMGSZ", "640"))
TARGET_FPS = int(os.getenv("TARGET_FPS", "10"))
SHOW_WINDOW = os.getenv("SHOW_WINDOW", "true").lower() == "true"

# zone point
ZONE_POINT_MODE = os.getenv("ZONE_POINT_MODE", "foot")  # foot|center

# local anti-dup
VID_MERGE_MAX_SEC = float(os.getenv("VID_MERGE_MAX_SEC", "3.0"))
VID_MERGE_MAX_DIST = float(os.getenv("VID_MERGE_MAX_DIST", "0.10"))

# staff tagging by zone
STAFF_DWELL_SEC = float(os.getenv("STAFF_DWELL_SEC", "12"))

# staff tagging by color heuristic
USE_STAFF_COLOR = os.getenv("USE_STAFF_COLOR", "1") == "1"
STAFF_COLOR_DWELL_SEC = float(os.getenv("STAFF_COLOR_DWELL_SEC", "8"))
# ระบุ HSV ช่วงสีเสื้อยูนิฟอร์ม (ตัวอย่าง: ดำ/เทาเข้ม) -> ปรับเองตามร้าน
# format: "Hmin,Smin,Vmin;Hmax,Smax,Vmax|Hmin,Smin,Vmin;Hmax,Smax,Vmax"
STAFF_HSV_RANGES = os.getenv(
    "STAFF_HSV_RANGES",
    "0,0,0;180,255,70"  # ดำ/มืด
)

# shop session
SHOP_EXIT_GRACE_SEC = float(os.getenv("SHOP_EXIT_GRACE_SEC", "10"))

# Re-ID (appearance vector)
USE_REID = os.getenv("USE_REID", "1") == "1"
REID_HIST_BINS = int(os.getenv("REID_HIST_BINS", "24"))
REID_MAX_SEC = float(os.getenv("REID_MAX_SEC", "8.0"))     # memory window
REID_MAX_DIST = float(os.getenv("REID_MAX_DIST", "0.18"))  # normalized xy distance gate
REID_SIM_TH = float(os.getenv("REID_SIM_TH", "0.78"))       # cosine similarity threshold (0-1)

# global matching fallback
GLOBAL_MATCH_MAX_SEC = float(os.getenv("GLOBAL_MATCH_MAX_SEC", "2.0"))
GLOBAL_MATCH_MAX_DIST = float(os.getenv("GLOBAL_MATCH_MAX_DIST", "0.12"))

# ✅ นับเมื่อ "นั่งค้าง" >= X วิ
SIT_MIN_SEC = float(os.getenv("SIT_MIN_SEC", "30"))  # แนะนำ 30-60 สำหรับร้านจริง

# chair occupancy robust
VACANT_GRACE_SEC = float(os.getenv("VACANT_GRACE_SEC", "10"))  # เพิ่มได้เพื่อกันผ้าคลุม/หลุด detect

# Double confirmation (Chair + Wash)
REQUIRE_WASH = os.getenv("REQUIRE_WASH", "0") == "1"
WASH_MIN_SEC = float(os.getenv("WASH_MIN_SEC", "15"))
WASH_CONFIRM_WINDOW_SEC = float(os.getenv("WASH_CONFIRM_WINDOW_SEC", "60"))  # ต้องเจอสระภายในกี่วินาที หลังนั่งเก้าอี้

# zones
STAFF_NAMES = {"STAFF", "STAFF_AREA", "STAFF_ZONE"}
SHOP_NAMES = {"SHOP"}

# optional zone mapping across cameras
# หมายเหตุ: ถ้ากล้อง 3 เห็นเก้าอี้คนละชุด ให้ map เป็น CHAIR_4/5/6 แทนเพื่อไม่ชนกัน
ZONE_MAP = {
    "Camera_01": {"CHAIR_1": "CHAIR_1", "CHAIR_2": "CHAIR_2", "CHAIR_3": "CHAIR_3",
                  "WASH_1": "WASH_1", "WASH_2": "WASH_2"},
    "Camera_02": {"CHAIR_1": "CHAIR_1", "CHAIR_2": "CHAIR_2", "CHAIR_3": "CHAIR_3",
                  "WASH_1": "WASH_1", "WASH_2": "WASH_2"},
    "Camera_03": {"CHAIR_1": "CHAIR_1", "CHAIR_2": "CHAIR_2", "CHAIR_3": "CHAIR_3",
                  "WASH_1": "WASH_1", "WASH_2": "WASH_2"},
}

# Snapshot
SNAPSHOT_DIR = os.getenv("SNAPSHOT_DIR", "snapshots")
SNAPSHOT_PAD = int(os.getenv("SNAPSHOT_PAD", "30"))
REPORT_DIR = os.getenv("REPORT_DIR", "reports")


# =========================
# UTILS
# =========================
def today_key():
    return time.strftime("%Y-%m-%d", time.localtime())

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def now_hms():
    return time.strftime("%H:%M:%S", time.localtime())

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return -1.0
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / (na * nb))


# =========================
# GEOMETRY
# =========================
def point_in_poly(x: float, y: float, poly: List[Dict]) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]["x"], poly[i]["y"]
        x2, y2 = poly[(i + 1) % n]["x"], poly[(i + 1) % n]["y"]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / ((y2 - y1) + 1e-9) + x1):
            inside = not inside
    return inside

def dist2(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2

def l2dist(a, b):
    return float(np.sqrt(dist2(a, b)))


# =========================
# ZONES
# =========================
@dataclass
class Zone:
    name: str
    polygon: List[Dict]
    min_dwell_sec: int = 3
    exit_grace_sec: int = 6

def load_zones(cam_name: str) -> List[Zone]:
    fp = f"zones_{cam_name}.json"
    if not os.path.exists(fp):
        print(f"⚠️ {cam_name}: zones file not found: {fp}")
        return []
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data["zones"] if isinstance(data, dict) and "zones" in data else data

    zones = []
    for it in items:
        name = it.get("name") or it.get("zone_type")
        poly = it.get("polygon_json") or it.get("polygon")
        if not name or not poly:
            continue
        zones.append(Zone(name=name, polygon=poly))
    return zones

def zone_by_name(zones: List[Zone], name: str) -> Optional[Zone]:
    for z in zones:
        if z.name == name:
            return z
    return None

def zones_containing_point(zones: List[Zone], cx: float, cy: float) -> List[str]:
    hit = []
    for z in zones:
        if z.polygon and point_in_poly(cx, cy, z.polygon):
            hit.append(z.name)
    return hit


# =========================
# CSV LOGGER
# =========================
class DailyCSVLogger:
    def __init__(self, report_dir: str):
        self.report_dir = report_dir
        ensure_dir(report_dir)
        self.day = None
        self.fp = None
        self.writer = None

    def _open_for_today(self):
        day = today_key()
        if self.day == day and self.writer is not None:
            return
        self.day = day
        path = os.path.join(self.report_dir, f"report_{day}.csv")
        new_file = not os.path.exists(path)
        self.fp = open(path, "a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.fp)
        if new_file:
            self.writer.writerow([
                "seq", "date", "time",
                "event", "camera", "zone",
                "pid", "gid", "local_vid",
                "dwell_sec", "confidence",
                "snapshot_path"
            ])
            self.fp.flush()

    def log_event(self, seq: int, event: str, cam: str, zone: str, pid: int, gid: int, local_vid: int,
                  dwell_sec: float, confidence: float, snapshot_path: str):
        self._open_for_today()
        self.writer.writerow([
            seq, self.day, now_hms(),
            event, cam, zone,
            pid, gid, local_vid,
            f"{dwell_sec:.1f}", f"{confidence:.2f}",
            snapshot_path
        ])
        self.fp.flush()

    def close(self):
        try:
            if self.fp:
                self.fp.close()
        except Exception:
            pass


# =========================
# SNAPSHOT
# =========================
def crop_and_save_snapshot(frame: np.ndarray, bbox, out_path: str, pad: int = 30) -> str:
    ensure_dir(os.path.dirname(out_path))
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad)
    y2 = min(h - 1, y2 + pad)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return ""
    cv2.imwrite(out_path, crop)
    return out_path


# =========================
# STAFF COLOR HEURISTIC
# =========================
def parse_hsv_ranges(spec: str):
    ranges = []
    for part in spec.split("|"):
        part = part.strip()
        if not part:
            continue
        lo_s, hi_s = part.split(";")
        lo = [int(x) for x in lo_s.split(",")]
        hi = [int(x) for x in hi_s.split(",")]
        if len(lo) == 3 and len(hi) == 3:
            ranges.append((np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8)))
    return ranges

STAFF_HSV_PARSED = parse_hsv_ranges(STAFF_HSV_RANGES)

def staff_color_score(frame: np.ndarray, bbox: List[float]) -> float:
    """
    คืนค่า 0..1 ว่าคล้ายยูนิฟอร์มพนักงานแค่ไหน
    ใช้ ROI ช่วงอก/ลำตัว (upper body) เพื่อดูสีเสื้อ
    """
    if frame is None or bbox is None:
        return 0.0
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = clamp(x1, 0, w-1); x2 = clamp(x2, 0, w-1)
    y1 = clamp(y1, 0, h-1); y2 = clamp(y2, 0, h-1)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    # upper body ROI
    yy1 = y1 + int(0.15 * (y2 - y1))
    yy2 = y1 + int(0.55 * (y2 - y1))
    xx1 = x1 + int(0.15 * (x2 - x1))
    xx2 = x1 + int(0.85 * (x2 - x1))
    yy1 = clamp(yy1, 0, h-1); yy2 = clamp(yy2, 0, h-1)
    xx1 = clamp(xx1, 0, w-1); xx2 = clamp(xx2, 0, w-1)

    roi = frame[yy1:yy2, xx1:xx2]
    if roi.size == 0:
        return 0.0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    total = hsv.shape[0] * hsv.shape[1]
    if total <= 0:
        return 0.0

    mask_total = 0
    for lo, hi in STAFF_HSV_PARSED:
        mask = cv2.inRange(hsv, lo, hi)
        mask_total += int(np.count_nonzero(mask))
    return float(mask_total / max(total, 1))


# =========================
# RE-ID FEATURE (HSV Histogram Vector)
# =========================
def extract_reid_vector(frame: np.ndarray, bbox: List[float], bins: int = 24) -> Optional[np.ndarray]:
    if frame is None or bbox is None:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = clamp(x1, 0, w-1); x2 = clamp(x2, 0, w-1)
    y1 = clamp(y1, 0, h-1); y2 = clamp(y2, 0, h-1)
    if x2 <= x1 or y2 <= y1:
        return None

    # ใช้ ROI ช่วงตัวมากกว่าหน้า เพื่อทนมุมกล้อง/ผ้าคลุมบางส่วน
    yy1 = y1 + int(0.10 * (y2 - y1))
    yy2 = y2 - int(0.05 * (y2 - y1))
    xx1 = x1 + int(0.10 * (x2 - x1))
    xx2 = x2 - int(0.10 * (x2 - x1))
    yy1 = clamp(yy1, 0, h-1); yy2 = clamp(yy2, 0, h-1)
    xx1 = clamp(xx1, 0, w-1); xx2 = clamp(xx2, 0, w-1)

    roi = frame[yy1:yy2, xx1:xx2]
    if roi.size == 0:
        return None

    roi = cv2.resize(roi, (128, 256), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # histogram H,S,V
    h_hist = cv2.calcHist([hsv], [0], None, [bins], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [bins], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [bins], [0, 256]).flatten()

    vec = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
    vec /= (np.linalg.norm(vec) + 1e-9)
    return vec


# =========================
# LOCAL VIRTUAL ID (track-id stabilizer)
# =========================
class VirtualIDManager:
    def __init__(self):
        self.next_vid = 1
        self.track_to_vid = {}
        self.vid_last = {}      # vid -> {ts,xy}
        self.vid_is_staff = {}
        self.vid_staff_enter_ts = {}
        self.vid_shop_last = {} # vid -> ts
        self.vid_color_staff_enter_ts = {}
        self.vid_color_staff_score = {}

    def assign(self, raw_track_id: int, xy: Tuple[float, float], ts: float) -> int:
        if raw_track_id in self.track_to_vid:
            vid = self.track_to_vid[raw_track_id]
            self.vid_last[vid] = {"ts": ts, "xy": xy}
            return vid

        best_vid = None
        best_d2 = 1e9
        for vid, info in self.vid_last.items():
            dt = ts - info["ts"]
            if 0 <= dt <= VID_MERGE_MAX_SEC:
                d2v = dist2(xy, info["xy"])
                if d2v < best_d2:
                    best_d2 = d2v
                    best_vid = vid

        if best_vid is not None and best_d2 <= (VID_MERGE_MAX_DIST**2):
            vid = best_vid
        else:
            vid = self.next_vid
            self.next_vid += 1

        self.track_to_vid[raw_track_id] = vid
        self.vid_last[vid] = {"ts": ts, "xy": xy}
        self.vid_is_staff.setdefault(vid, False)
        return vid

    def update_staff_tag_by_zone(self, vid: int, in_staff_area: bool, ts: float):
        if in_staff_area:
            if vid not in self.vid_staff_enter_ts:
                self.vid_staff_enter_ts[vid] = ts
            elif (ts - self.vid_staff_enter_ts[vid]) >= STAFF_DWELL_SEC:
                self.vid_is_staff[vid] = True
        else:
            if not self.vid_is_staff.get(vid, False):
                self.vid_staff_enter_ts.pop(vid, None)

    def update_staff_tag_by_color(self, vid: int, staff_score: float, ts: float):
        if not USE_STAFF_COLOR:
            return
        # ถ้า score สูงต่อเนื่อง -> mark staff
        # (กันลูกค้าที่เสื้อคล้ายยูนิฟอร์มด้วย dwell time)
        if staff_score >= 0.65:
            if vid not in self.vid_color_staff_enter_ts:
                self.vid_color_staff_enter_ts[vid] = ts
            elif (ts - self.vid_color_staff_enter_ts[vid]) >= STAFF_COLOR_DWELL_SEC:
                self.vid_is_staff[vid] = True
        else:
            if not self.vid_is_staff.get(vid, False):
                self.vid_color_staff_enter_ts.pop(vid, None)

    def is_staff(self, vid: int) -> bool:
        return bool(self.vid_is_staff.get(vid, False))

    def mark_in_shop(self, vid: int, ts: float):
        self.vid_shop_last[vid] = ts

    def is_in_shop(self, vid: int, ts: float) -> bool:
        last = self.vid_shop_last.get(vid)
        if last is None:
            return False
        return (ts - last) <= SHOP_EXIT_GRACE_SEC


# =========================
# GLOBAL PERSON ID MANAGER (Re-ID)
# =========================
class PersonIDManager:
    """
    pid: คนเดียวกันข้ามกล้อง/หลุด track
    - ใช้ gate ตามเวลา/ระยะ + appearance vector (HSV hist)
    - ถ้าหาไม่เจอ -> alloc pid ใหม่
    """
    def __init__(self):
        self.next_pid = 1
        self.last = {}  # pid -> {ts, xy, cam, chair, vec}
        self.local_to_pid = {}  # (cam, local_vid) -> pid

    def _alloc(self):
        pid = self.next_pid
        self.next_pid += 1
        return pid

    def assign(self, det: dict) -> int:
        key = (det["cam"], det["local_vid"])
        ts = det["ts"]
        xy = (det["cx"], det["cy"])
        chair = det.get("chair")  # normalized chair
        vec = det.get("reid_vec")

        # same local key -> same pid
        if key in self.local_to_pid:
            pid = self.local_to_pid[key]
            self.last[pid] = {"ts": ts, "xy": xy, "cam": det["cam"], "chair": chair, "vec": vec}
            return pid

        # try match existing pid
        best_pid = None
        best_score = -1e9
        for pid, info in self.last.items():
            dt = abs(ts - info["ts"])
            if dt > REID_MAX_SEC:
                continue

            # chair constraint: ถ้าทั้งคู่มี chair และไม่ตรง -> ข้าม
            if chair is not None and info.get("chair") is not None:
                if chair != info["chair"]:
                    continue

            # distance gate
            if l2dist(xy, info["xy"]) > REID_MAX_DIST:
                continue

            # similarity (ถ้าไม่มี vec -> fallback)
            if USE_REID and vec is not None and info.get("vec") is not None:
                sim = cosine_sim(vec, info["vec"])
            else:
                # fallback score = ระยะ+เวลา (ยิ่งใกล้ยิ่งดี)
                sim = 1.0 - (l2dist(xy, info["xy"]) / max(REID_MAX_DIST, 1e-6)) - (dt / max(REID_MAX_SEC, 1e-6))

            if sim > best_score:
                best_score = sim
                best_pid = pid

        if best_pid is not None:
            # threshold for appearance sim
            if USE_REID and vec is not None and self.last[best_pid].get("vec") is not None:
                if best_score < REID_SIM_TH:
                    best_pid = None

        pid = self._alloc() if best_pid is None else best_pid
        self.local_to_pid[key] = pid
        self.last[pid] = {"ts": ts, "xy": xy, "cam": det["cam"], "chair": chair, "vec": vec}
        return pid


# =========================
# CAMERA PIPELINE
# =========================
class CameraPipeline:
    def __init__(self, cam_name: str, source: str):
        self.cam_name = cam_name
        self.source = source
        self.running = True

        self.zones = load_zones(cam_name)
        self.z_shop = zone_by_name(self.zones, "SHOP")

        self.z_staff = None
        for zn in STAFF_NAMES:
            z = zone_by_name(self.zones, zn)
            if z:
                self.z_staff = z
                break

        self.vid_mgr = VirtualIDManager()

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = YOLO(YOLO_MODEL_PATH)
        print(f"📷 {cam_name}: device={self.device} zones={len(self.zones)} source={'VIDEO' if USE_VIDEO else 'RTSP'}")

        self._lock = threading.Lock()
        self.latest_frame = None
        self.latest_frame_ts = 0.0

        self.latest_dets = []
        self.latest_ts = 0.0

        self._grab_thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._proc_thread = threading.Thread(target=self._proc_loop, daemon=True)

    def start(self):
        self._grab_thread.start()
        self._proc_thread.start()

    def stop(self):
        self.running = False

    def _open_cap(self):
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _grab_loop(self):
        cap = self._open_cap()
        if not cap.isOpened():
            print(f"❌ {self.cam_name}: cannot open source")
            return

        while self.running:
            ok, frame = cap.read()
            if not ok or frame is None:
                if USE_VIDEO:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.2)
                    continue
                cap.release()
                time.sleep(1.0)
                cap = self._open_cap()
                continue

            ts = time.time()
            with self._lock:
                self.latest_frame = frame
                self.latest_frame_ts = ts

        cap.release()

    def _normalize_zone(self, zn: str) -> str:
        return ZONE_MAP.get(self.cam_name, {}).get(zn, zn)

    def _get_chair_or_wash(self, zones: List[str]) -> Tuple[Optional[str], Optional[str]]:
        chair = None
        wash = None
        for zn in zones:
            if zn.startswith("CHAIR_"):
                chair = self._normalize_zone(zn)
            if zn.startswith("WASH"):
                wash = self._normalize_zone(zn)
        return chair, wash

    def _proc_loop(self):
        interval = 1.0 / max(TARGET_FPS, 1)
        last = 0.0

        while self.running:
            now = time.time()
            if (now - last) < interval:
                time.sleep(0.001)
                continue
            last = now

            with self._lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()

            if frame is None:
                continue

            H, W = frame.shape[:2]

            results = self.model.track(
                frame,
                persist=True,
                verbose=False,
                conf=YOLO_CONF,
                iou=YOLO_IOU,
                classes=[0],  # person
                imgsz=IMGSZ,
                device=self.device,
                tracker="bytetrack.yaml",
            )

            dets = []
            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes
                for box, tid in zip(boxes.xyxy, boxes.id):
                    x1, y1, x2, y2 = box.cpu().numpy().tolist()
                    cx = ((x1 + x2) / 2) / max(W, 1)
                    cy = (y2 / max(H, 1)) if ZONE_POINT_MODE == "foot" else (((y1 + y2) / 2) / max(H, 1))

                    vid = self.vid_mgr.assign(int(tid), (cx, cy), now)

                    in_shop = True if (self.z_shop is None) else point_in_poly(cx, cy, self.z_shop.polygon)
                    if in_shop:
                        self.vid_mgr.mark_in_shop(vid, now)

                    in_staff_zone = False
                    if self.z_staff is not None:
                        in_staff_zone = point_in_poly(cx, cy, self.z_staff.polygon)
                    self.vid_mgr.update_staff_tag_by_zone(vid, in_staff_zone, now)

                    # staff color heuristic
                    sc = staff_color_score(frame, [x1, y1, x2, y2]) if USE_STAFF_COLOR else 0.0
                    self.vid_mgr.update_staff_tag_by_color(vid, sc, now)

                    hit_zones_raw = zones_containing_point(self.zones, cx, cy)
                    hit_zones = [self._normalize_zone(z) for z in hit_zones_raw]

                    chair, wash = self._get_chair_or_wash(hit_zones)

                    # ReID vector (appearance)
                    reid_vec = extract_reid_vector(frame, [x1, y1, x2, y2], bins=REID_HIST_BINS) if USE_REID else None

                    dets.append({
                        "cam": self.cam_name,
                        "local_vid": vid,
                        "cx": cx, "cy": cy,
                        "bbox": [x1, y1, x2, y2],
                        "staff_color_score": sc,
                        "is_staff": self.vid_mgr.is_staff(vid),
                        "in_shop": self.vid_mgr.is_in_shop(vid, now),
                        "zones": hit_zones,
                        "chair": chair,
                        "wash": wash,
                        "reid_vec": reid_vec,
                        "ts": now,
                    })

            self.latest_ts = now
            self.latest_dets = dets


# =========================
# OCCUPANCY + DOUBLE CONFIRM + VERIFICATION
# =========================
class CustomerLogic:
    """
    - Chair dwell filter (นั่งต่อเนื่อง)
    - Occupancy memory effect (กันหลุด detect จาก cape)
    - Optional: require wash confirmation (chair + wash)
    - Snapshot + CSV verification
    """
    def __init__(self, csv_logger: DailyCSVLogger):
        self.csv = csv_logger
        self.day = today_key()
        self.seq = 0

        self.total_customer_count = 0

        # chair state: chair -> session
        self.chair_state: Dict[str, dict] = {}
        # wash state: wash -> session
        self.wash_state: Dict[str, dict] = {}

        # per pid flags
        self.pid_flags: Dict[int, dict] = {}  # pid -> {chair_counted_ts, wash_ok_ts, counted, confidence}

        # stats
        self.chair_total: Dict[str, int] = {}
        self.wash_total: Dict[str, int] = {}

    def _ensure_day(self):
        d = today_key()
        if d != self.day:
            self.day = d
            self.seq = 0
            self.total_customer_count = 0
            self.chair_state = {}
            self.wash_state = {}
            self.pid_flags = {}
            self.chair_total = {}
            self.wash_total = {}
            print(f"🗓️ NEW DAY RESET: {self.day}")

    def _best_by_zone(self, dets: List[dict], key_field: str) -> Dict[str, dict]:
        """
        เลือก det ที่ "ใหญ่สุด" ต่อ zone (chair/wash) เพื่อกันซ้อน/ทับ
        """
        best = {}
        for d in dets:
            z = d.get(key_field)
            if not z:
                continue
            x1, y1, x2, y2 = d["bbox"]
            area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
            if (z not in best) or (area > best[z]["_bbox_area"]):
                dd = dict(d)
                dd["_bbox_area"] = area
                best[z] = dd
        return best

    def _get_pidflag(self, pid: int) -> dict:
        return self.pid_flags.setdefault(pid, {
            "chair_counted_ts": None,
            "wash_ok_ts": None,
            "counted": False,
            "confidence": 0.0,
        })

    def _maybe_mark_wash_ok(self, pid: int, ts: float):
        pf = self._get_pidflag(pid)
        pf["wash_ok_ts"] = ts

    def _can_count_customer(self, pid: int, now: float) -> Tuple[bool, float]:
        pf = self._get_pidflag(pid)
        if pf["counted"]:
            return False, pf["confidence"]

        if not REQUIRE_WASH:
            return True, 0.85  # chair-only confidence baseline

        # require wash within window
        chair_ts = pf["chair_counted_ts"]
        wash_ts = pf["wash_ok_ts"]
        if chair_ts is None or wash_ts is None:
            return False, 0.0

        if 0 <= (wash_ts - chair_ts) <= WASH_CONFIRM_WINDOW_SEC:
            return True, 0.98
        return False, 0.0

    def update(self,
               dets: List[dict],
               assign_pid_func,
               get_latest_frame_func):
        self._ensure_day()
        now = time.time()

        # ---------------------------
        # 1) Process WASH dwell -> mark wash_ok(pid)
        # ---------------------------
        best_wash = self._best_by_zone(dets, "wash")
        for wash_zone, d in best_wash.items():
            pid = assign_pid_func(d)
            st = self.wash_state.setdefault(wash_zone, {
                "occupied": False,
                "enter_ts": 0.0,
                "last_seen": 0.0,
                "ok_marked": False,
                "last_pid": None,
                "last_cam": None,
                "last_vid": None,
                "last_bbox": None,
            })

            st["last_seen"] = now
            st["last_pid"] = pid
            st["last_cam"] = d["cam"]
            st["last_vid"] = d["local_vid"]
            st["last_bbox"] = d["bbox"]

            if not st["occupied"]:
                st["occupied"] = True
                st["enter_ts"] = now
                st["ok_marked"] = False

            dwell = now - st["enter_ts"]
            if (not st["ok_marked"]) and (dwell >= WASH_MIN_SEC):
                st["ok_marked"] = True
                self._maybe_mark_wash_ok(pid, now)

                self.wash_total[wash_zone] = self.wash_total.get(wash_zone, 0) + 1
                print(f"🧼 WASH_OK | zone={wash_zone} pid={pid} dwell={dwell:.1f}s")

        # end wash session on vacant
        for wash_zone, st in list(self.wash_state.items()):
            if wash_zone in best_wash:
                continue
            if st["occupied"] and (now - st["last_seen"]) > VACANT_GRACE_SEC:
                st["occupied"] = False
                st["enter_ts"] = 0.0
                st["ok_marked"] = False
                st["last_pid"] = None
                st["last_cam"] = None
                st["last_vid"] = None
                st["last_bbox"] = None

        # ---------------------------
        # 2) Process CHAIR dwell -> "chair candidate"
        # ---------------------------
        best_chair = self._best_by_zone(dets, "chair")
        for chair_zone, d in best_chair.items():
            pid = assign_pid_func(d)

            st = self.chair_state.setdefault(chair_zone, {
                "occupied": False,
                "enter_ts": 0.0,
                "last_seen": 0.0,
                "chair_dwell_counted": False,
                "last_pid": None,
                "last_cam": None,
                "last_vid": None,
                "last_bbox": None,
            })

            st["last_seen"] = now
            st["last_pid"] = pid
            st["last_cam"] = d["cam"]
            st["last_vid"] = d["local_vid"]
            st["last_bbox"] = d["bbox"]

            if not st["occupied"]:
                st["occupied"] = True
                st["enter_ts"] = now
                st["chair_dwell_counted"] = False

            dwell = now - st["enter_ts"]

            if (not st["chair_dwell_counted"]) and (dwell >= SIT_MIN_SEC):
                st["chair_dwell_counted"] = True

                pf = self._get_pidflag(pid)
                pf["chair_counted_ts"] = now

                # decide if can count customer now (chair-only or chair+wash)
                ok, conf = self._can_count_customer(pid, now)
                if ok:
                    pf["counted"] = True
                    pf["confidence"] = conf
                    self.total_customer_count += 1
                    self.chair_total[chair_zone] = self.chair_total.get(chair_zone, 0) + 1
                    self.seq += 1

                    # snapshot
                    snapshot_path = ""
                    frame = get_latest_frame_func(st["last_cam"])
                    if frame is not None and st["last_bbox"] is not None:
                        ensure_dir(SNAPSHOT_DIR)
                        snapshot_path = os.path.join(
                            SNAPSHOT_DIR,
                            f"{self.day}_seq{self.seq:04d}_{st['last_cam']}_{chair_zone}_pid{pid}.jpg"
                        )
                        p = crop_and_save_snapshot(frame, st["last_bbox"], snapshot_path, pad=SNAPSHOT_PAD)
                        snapshot_path = p or ""

                    # csv
                    self.csv.log_event(
                        seq=self.seq,
                        event="CUSTOMER_COUNT",
                        cam=st["last_cam"] or "",
                        zone=chair_zone,
                        pid=int(pid),
                        gid=int(pid),   # ให้ gid=pid เพื่ออ่านง่าย (คนเดียวกัน)
                        local_vid=int(st["last_vid"] or 0),
                        dwell_sec=float(dwell),
                        confidence=float(conf),
                        snapshot_path=snapshot_path
                    )

                    print(f"✅ COUNTED | day={self.day} seq={self.seq} chair={chair_zone} total={self.total_customer_count} pid={pid} dwell={dwell:.1f}s conf={conf:.2f} snap={bool(snapshot_path)}")
                else:
                    # ถ้า require wash และยังไม่ครบ -> ยังไม่นับ แต่จำ timestamp ไว้
                    print(f"⏳ CHAIR_OK_WAIT_WASH | chair={chair_zone} pid={pid} dwell={dwell:.1f}s require_wash={REQUIRE_WASH}")

        # ---------------------------
        # 3) Occupancy Memory Effect (cape) : end session only after vacancy grace
        # ---------------------------
        for chair_zone, st in list(self.chair_state.items()):
            if chair_zone in best_chair:
                continue
            if st["occupied"] and (now - st["last_seen"]) > VACANT_GRACE_SEC:
                # จบ session
                st["occupied"] = False
                st["enter_ts"] = 0.0
                st["chair_dwell_counted"] = False
                st["last_pid"] = None
                st["last_cam"] = None
                st["last_vid"] = None
                st["last_bbox"] = None

    def live_waiting_count(self) -> int:
        return sum(1 for st in self.chair_state.values() if st.get("occupied", False))


# =========================
# UI DRAW
# =========================
def draw_zones(frame, zones: List[Zone]):
    h, w = frame.shape[:2]
    for z in zones:
        pts = np.array([[int(p["x"] * w), int(p["y"] * h)] for p in z.polygon], np.int32)
        if z.name in STAFF_NAMES:
            color = (255, 0, 255)
        elif z.name in SHOP_NAMES:
            color = (0, 255, 255)
        elif str(z.name).startswith("WASH"):
            color = (255, 255, 0)
        else:
            color = (0, 255, 0)
        cv2.polylines(frame, [pts], True, color, 2)
        cv2.putText(frame, z.name, (pts[0][0], max(20, pts[0][1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

def draw_people(frame, dets, pid_map):
    for d in dets:
        x1, y1, x2, y2 = map(int, d["bbox"])
        is_staff = d["is_staff"]
        color = (0, 0, 255) if is_staff else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        pid = pid_map.get((d["cam"], d["local_vid"]))
        chair = d.get("chair")
        wash = d.get("wash")
        sc = d.get("staff_color_score", 0.0)
        label = f"P:{pid} L:{d['local_vid']}" + (" STAFF" if is_staff else "")
        if chair:
            label += f" {chair}"
        if wash:
            label += f" {wash}"
        if USE_STAFF_COLOR:
            label += f" C:{sc:.2f}"

        cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


# =========================
# MAIN
# =========================
def main():
    print("💈 Global Multi-Camera Counter Starting...")

    ensure_dir(SNAPSHOT_DIR)
    ensure_dir(REPORT_DIR)

    csv_logger = DailyCSVLogger(REPORT_DIR)
    logic = CustomerLogic(csv_logger)

    # sources
    if USE_VIDEO:
        sources = {
            "Camera_01": CAM1_VIDEO,
            "Camera_02": CAM2_VIDEO,
            "Camera_03": CAM3_VIDEO,
        }
    else:
        sources = dict(CAMERAS_CONFIG)

    # remove empty sources (camera 03 may be blank)
    sources = {k: v for k, v in sources.items() if v and str(v).strip()}

    # start cams
    cams: Dict[str, CameraPipeline] = {}
    for cam_name, src in sources.items():
        cams[cam_name] = CameraPipeline(cam_name, src)
        cams[cam_name].start()

    pid_mgr = PersonIDManager()

    last_log_ts = 0.0

    def get_latest_frame(cam_name: str):
        cam = cams.get(cam_name)
        if cam is None:
            return None
        with cam._lock:
            return None if cam.latest_frame is None else cam.latest_frame.copy()

    try:
        while True:
            now = time.time()

            all_dets = []
            frames = {}
            for cam_name, cam in cams.items():
                with cam._lock:
                    frames[cam_name] = None if cam.latest_frame is None else cam.latest_frame.copy()
                all_dets.extend(cam.latest_dets or [])

            # filter: customers only (in_shop AND not staff)
            customers = []
            for d in all_dets:
                if not d.get("in_shop", False):
                    continue
                if d.get("is_staff", False):
                    continue
                customers.append(d)

            # pid assignment wrapper
            def pid_of(det: dict) -> int:
                return pid_mgr.assign(det)

            # ✅ business logic update (chair/wash/occupancy + snapshot + csv)
            logic.update(customers, pid_of, get_latest_frame)

            # console log
            if (now - last_log_ts) >= 2.0:
                last_log_ts = now

                live_pids = set()
                for d in customers:
                    key = (d["cam"], d["local_vid"])
                    if key in pid_mgr.local_to_pid:
                        live_pids.add(pid_mgr.local_to_pid[key])

                waiting_now = logic.live_waiting_count()
                chairs = sorted(logic.chair_total.keys())
                chair_lines = []
                for c in chairs:
                    cnt = logic.chair_total.get(c, 0)
                    chair_lines.append(f"{c}:count={cnt}")

                print(
                    f"📊 DAY={today_key()} | LIVE_CUSTOMERS={len(live_pids)} | WAITING_NOW={waiting_now} "
                    f"| TOTAL_CUSTOMER_COUNT={logic.total_customer_count} | customers_frames={len(customers)} "
                    f"| REQUIRE_WASH={REQUIRE_WASH}"
                )
                if chair_lines:
                    print("🪑 " + " | ".join(chair_lines))

            # UI
            if SHOW_WINDOW:
                for cam_name, frame in frames.items():
                    if frame is None:
                        continue
                    cam = cams[cam_name]
                    draw_zones(frame, cam.zones)
                    draw_people(frame, cam.latest_dets or [], pid_mgr.local_to_pid)

                    cv2.rectangle(frame, (0, 0), (frame.shape[1], 95), (0, 0, 0), -1)
                    cv2.putText(
                        frame,
                        f"{cam_name} | DAY: {today_key()} | COUNT: {logic.total_customer_count} | WAITING: {logic.live_waiting_count()} | WASH_REQ: {REQUIRE_WASH}",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2
                    )
                    cv2.putText(
                        frame,
                        f"YOLO:{os.path.basename(YOLO_MODEL_PATH)} CONF:{YOLO_CONF} IOU:{YOLO_IOU} REID:{USE_REID} STAFF_COLOR:{USE_STAFF_COLOR}",
                        (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2
                    )
                    cv2.imshow(cam_name, frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            time.sleep(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        for cam in cams.values():
            cam.stop()
        cv2.destroyAllWindows()
        csv_logger.close()
        print("🛑 Stopped.")


if __name__ == "__main__":
    main()
