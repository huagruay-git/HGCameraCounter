# edge_agent_full.py
# ============================================================
# ✅ Global Multi-Camera Counter (FULL FILE) + HUD (not blocking)
# Business meaning:
#   - WAIT  = โซนนั่งรอ (โชว์จำนวนคนรอแบบ realtime)
#   - CHAIR_1/2/3 = เก้าอี้ตัดผม (นับ haircut_count ให้แม่นที่สุด)
#   - WASH  = จุดสระผม (นับ wash_count แยก)
#
# Added:
#   - draw_hud() แบบโปร่ง เตี้ย ไม่บังภาพ
#   - BOOT scan staff: เริ่มระบบแล้วสแกนช่างในเฟรมเพื่อ tag เป็น staff ทันที (ต้อง ENABLE_REID=1)
#
# Run:
#   python edge_agent_full.py
#   ENABLE_REID=0 python edge_agent_full.py   # ถ้า ReID โหลด weight ไม่ได้
# ============================================================

import os
import time
import json
import csv
import shutil
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Set

import numpy as np
import cv2
import torch
from ultralytics import YOLO

# torchvision for ReID encoder
import torchvision.transforms as T
import torchvision.models as models

# =========================
# CONFIG
# =========================
CAMERAS_CONFIG = {
    "Camera_01": os.getenv("CAM1_URL", "rtsp://admin:112113114@192.168.1.24:554/ch01/0"),
    "Camera_02": os.getenv("CAM2_URL", "rtsp://admin:112113114@192.168.1.83:554/ch01/0"),
    "Camera_03": os.getenv("CAM3_URL", "rtsp://admin:112113114@192.168.1.91:554/ch01/0"),
}

USE_VIDEO = os.getenv("USE_VIDEO", "0") == "1"
CAM1_VIDEO = os.getenv("CAM1_VIDEO", "cam1.mp4")
CAM2_VIDEO = os.getenv("CAM2_VIDEO", "cam2.mp4")
CAM3_VIDEO = os.getenv("CAM3_VIDEO", "cam3.mp4")

YOLO_MODEL_PATH = os.getenv("YOLO_MODEL", "yolov8m.pt")
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.35"))
YOLO_IOU = float(os.getenv("YOLO_IOU", "0.5"))
IMGSZ = int(os.getenv("IMGSZ", "640"))
TARGET_FPS = int(os.getenv("TARGET_FPS", "10"))
SHOW_WINDOW = os.getenv("SHOW_WINDOW", "true").lower() == "true"

# Zone point mode
ZONE_POINT_MODE = os.getenv("ZONE_POINT_MODE", "foot")  # foot|center

# local anti-dup (raw track merge)
VID_MERGE_MAX_SEC = float(os.getenv("VID_MERGE_MAX_SEC", "3.0"))
VID_MERGE_MAX_DIST = float(os.getenv("VID_MERGE_MAX_DIST", "0.10"))

# staff tagging by staff-zone dwell (optional)
STAFF_DWELL_SEC = float(os.getenv("STAFF_DWELL_SEC", "12"))

# shop session
SHOP_EXIT_GRACE_SEC = float(os.getenv("SHOP_EXIT_GRACE_SEC", "10"))

# global matching (cross-cam)
GLOBAL_MATCH_MAX_SEC = float(os.getenv("GLOBAL_MATCH_MAX_SEC", "2.0"))
GLOBAL_MATCH_MAX_DIST = float(os.getenv("GLOBAL_MATCH_MAX_DIST", "0.12"))

# count when dwell >= N sec
SIT_MIN_SEC = float(os.getenv("SIT_MIN_SEC", "10"))

# end zone session if not seen > grace sec
VACANT_GRACE_SEC = float(os.getenv("VACANT_GRACE_SEC", "6"))

# (optional) extra cooldown per (zone,pid)
RECOUNT_COOLDOWN_SEC = float(os.getenv("RECOUNT_COOLDOWN_SEC", "300"))

# Zone naming
STAFF_NAMES = {"STAFF", "STAFF_AREA", "STAFF_ZONE"}
SHOP_NAMES = {"SHOP"}

WAIT_ZONE_NAMES = {"WAIT", "WAITING", "WAIT_AREA"}  # คุณใช้ WAIT เป็นหลัก
WASH_ZONE_PREFIX = "WASH"                            # WASH, WASH_1, WASH_ZONE ฯลฯ
CHAIR_ZONE_PREFIX = "CHAIR_"                         # CHAIR_1,2,3

# optional mapping across cameras
ZONE_MAP = {
    "Camera_01": {"CHAIR_1": "CHAIR_1", "CHAIR_2": "CHAIR_2", "CHAIR_3": "CHAIR_3", "WASH": "WASH", "WAIT": "WAIT"},
    "Camera_02": {"CHAIR_1": "CHAIR_1", "CHAIR_2": "CHAIR_2", "CHAIR_3": "CHAIR_3", "WASH": "WASH", "WAIT": "WAIT"},
    "Camera_03": {"CHAIR_1": "CHAIR_1", "CHAIR_2": "CHAIR_2", "CHAIR_3": "CHAIR_3", "WASH": "WASH", "WAIT": "WAIT"},
}

# snapshot
SNAPSHOT_DIR = os.getenv("SNAPSHOT_DIR", "snapshots")
SNAPSHOT_PAD = int(os.getenv("SNAPSHOT_PAD", "30"))
SNAPSHOT_MULTI_CAM = os.getenv("SNAPSHOT_MULTI_CAM", "true").lower() == "true"
SNAPSHOT_CAM_LIMIT = int(os.getenv("SNAPSHOT_CAM_LIMIT", "2"))  # save 2 angles
SNAPSHOT_DAY_KEEP = int(os.getenv("SNAPSHOT_DAY_KEEP", "2"))

# report
REPORT_DIR = os.getenv("REPORT_DIR", "reports")

# ReID / Staff Gallery
ENABLE_REID = os.getenv("ENABLE_REID", "1") == "1"
REID_SIM_THRESHOLD = float(os.getenv("REID_SIM_THRESHOLD", "0.80"))        # person same
REID_ACTIVE_EXPIRE_SEC = float(os.getenv("REID_ACTIVE_EXPIRE_SEC", "15"))  # not seen => expire

STAFF_GALLERY_DIR = os.getenv("STAFF_GALLERY_DIR", "staff_gallery")
STAFF_DB_PATH = os.getenv("STAFF_DB_PATH", "staff_gallery/staff_db.json")
STAFF_SIM_THRESHOLD = float(os.getenv("STAFF_SIM_THRESHOLD", "0.78"))      # 👈 แนะนำ 0.75-0.80 สำหรับกล้องวงจรปิด

# Encoder device: cpu safe (set REID_DEVICE=mps/cuda/cpu)
REID_DEVICE = os.getenv("REID_DEVICE", "cpu").lower()

# BOOT scan staff
BOOT_SCAN_SEC = float(os.getenv("BOOT_SCAN_SEC", "8"))
BOOT_SCAN_FPS = int(os.getenv("BOOT_SCAN_FPS", "4"))

# =========================
# UTILS
# =========================
def today_key():
    return time.strftime("%Y-%m-%d", time.localtime())

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def now_hms():
    return time.strftime("%H:%M:%S", time.localtime())

def ts_key():
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")[:-3]

def day_dir(base: str, day: str) -> str:
    return os.path.join(base, day)

def cleanup_old_days(base_dir: str, keep_days: int = 2):
    ensure_dir(base_dir)
    items = []
    for name in os.listdir(base_dir):
        p = os.path.join(base_dir, name)
        if os.path.isdir(p):
            try:
                datetime.strptime(name, "%Y-%m-%d")
                items.append(name)
            except:
                pass
    items.sort()
    remove = items[:-keep_days] if len(items) > keep_days else []
    for d in remove:
        shutil.rmtree(os.path.join(base_dir, d), ignore_errors=True)

def dist2(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2

# =========================
# GEOMETRY
# =========================
def point_in_poly(x: float, y: float, poly: List[dict]) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]["x"], poly[i]["y"]
        x2, y2 = poly[(i + 1) % n]["x"], poly[(i + 1) % n]["y"]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / ((y2 - y1) + 1e-9) + x1):
            inside = not inside
    return inside

@dataclass
class Zone:
    name: str
    polygon: List[dict]
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
# HUD (✅ PRO) - NOT BLOCKING
# =========================
def draw_hud(frame, cam_label: str, haircut_count: int, wash_count: int, wait_realtime: int,
             cut_in_progress: int, wash_in_progress: int):
    # กล่อง HUD โปร่ง ไม่บังมาก
    h = 58  # ความสูง HUD (ลดจาก 110)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], h), (0, 0, 0), -1)
    alpha = 0.45  # ความโปร่ง (0=ไม่เห็น, 1=ทึบ)
    frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    cv2.putText(
        frame,
        f"{cam_label} | DAY: {today_key()} | HAIRCUT: {haircut_count} | WASH: {wash_count} | WAIT: {wait_realtime}",
        (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2
    )
    cv2.putText(
        frame,
        f"CUT_IN_PROGRESS: {cut_in_progress} | WASH_IN_PROGRESS: {wash_in_progress}",
        (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
    )

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
                "seq", "date", "time", "event_id",
                "event_type", "zone",
                "pid", "gid", "dwell_sec",
                "snapshot_paths"
            ])
            self.fp.flush()

    def log_event(self, seq: int, event_id: str, event_type: str, zone: str,
                  pid: int, gid: int, dwell_sec: float, snapshot_paths: str):
        self._open_for_today()
        self.writer.writerow([
            seq, self.day, now_hms(), event_id,
            event_type, zone,
            pid, gid, f"{dwell_sec:.1f}",
            snapshot_paths
        ])
        self.fp.flush()

    def close(self):
        try:
            if self.fp:
                self.fp.close()
        except Exception:
            pass

# =========================
# ReID Encoder + Staff Gallery
# =========================
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

class ReIDEncoder:
    def __init__(self, device: str = "cpu"):
        self.device = device
        # NOTE: if SSL fails to download weights -> set ENABLE_REID=0 or fix cert/preload weights.
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        m.fc = torch.nn.Identity()
        m.eval()
        self.model = m.to(device)

        self.tf = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def encode_crop_bgr(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        x = self.tf(rgb).unsqueeze(0).to(self.device)
        feat = self.model(x).squeeze(0).detach().float().cpu().numpy()
        n = np.linalg.norm(feat) + 1e-9
        return feat / n

class StaffGallery:
    def __init__(self, gallery_dir: str, db_path: str, encoder: ReIDEncoder):
        self.gallery_dir = gallery_dir
        self.db_path = db_path
        self.encoder = encoder
        self.staff_embs: List[Tuple[str, np.ndarray]] = []
        ensure_dir(gallery_dir)
        self._load_or_build()

    def _load_or_build(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.staff_embs = [
                    (it["person_id"], np.array(it["emb"], dtype=np.float32))
                    for it in data.get("items", [])
                ]
                print(f"🧑‍💼 Staff DB loaded: {len(self.staff_embs)} embeddings")
                return
            except Exception as e:
                print(f"⚠️ Staff DB load failed, rebuild... ({e})")
        self.rebuild()

    def rebuild(self):
        items = []
        self.staff_embs = []
        for person_id in os.listdir(self.gallery_dir):
            pdir = os.path.join(self.gallery_dir, person_id)
            if not os.path.isdir(pdir):
                continue
            for fn in os.listdir(pdir):
                if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                img = cv2.imread(os.path.join(pdir, fn))
                if img is None:
                    continue
                emb = self.encoder.encode_crop_bgr(img)
                if emb is None:
                    continue
                self.staff_embs.append((person_id, emb))
                items.append({"person_id": person_id, "emb": emb.tolist()})

        ensure_dir(os.path.dirname(self.db_path))
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump({"items": items}, f, ensure_ascii=False)

        print(f"🧑‍💼 Staff DB rebuilt: {len(self.staff_embs)} embeddings")

    def match_staff(self, emb: Optional[np.ndarray], thr: float) -> Optional[str]:
        if emb is None or not self.staff_embs:
            return None
        best_id, best = None, -1.0
        for pid, e in self.staff_embs:
            s = cosine_sim(emb, e)
            if s > best:
                best = s
                best_id = pid
        return best_id if best_id is not None and best >= thr else None

# =========================
# ACTIVE PERSON MEMORY (no recount while active)
# =========================
class ActivePersonMemory:
    def __init__(self):
        self.next_pid = 1
        self.pid_last: Dict[int, Dict] = {}   # pid -> {"ts":..., "emb":...}
        self.gid_to_pid: Dict[int, int] = {}  # gid -> pid

    def expire(self, now_ts: float, expire_sec: float):
        dead = []
        for pid, info in self.pid_last.items():
            if (now_ts - info["ts"]) > expire_sec:
                dead.append(pid)
        for pid in dead:
            self.pid_last.pop(pid, None)
        self.gid_to_pid = {g: p for g, p in self.gid_to_pid.items() if p in self.pid_last}

    def resolve_pid(self, gid: int, emb: Optional[np.ndarray], now_ts: float, thr: float) -> int:
        if gid in self.gid_to_pid:
            pid = self.gid_to_pid[gid]
            info = self.pid_last.get(pid)
            if info is not None:
                info["ts"] = now_ts
                if emb is not None:
                    info["emb"] = emb
                return pid

        best_pid, best = None, -1.0
        if emb is not None:
            for pid, info in self.pid_last.items():
                e = info.get("emb")
                if e is None:
                    continue
                s = cosine_sim(emb, e)
                if s > best:
                    best = s
                    best_pid = pid

        if best_pid is not None and best >= thr:
            pid = best_pid
        else:
            pid = self.next_pid
            self.next_pid += 1

        self.pid_last.setdefault(pid, {"ts": now_ts, "emb": emb})
        self.pid_last[pid]["ts"] = now_ts
        if emb is not None:
            self.pid_last[pid]["emb"] = emb
        self.gid_to_pid[gid] = pid
        return pid

# =========================
# LOCAL VIRTUAL ID (camera)
# =========================
class VirtualIDManager:
    def __init__(self):
        self.next_vid = 1
        self.track_to_vid = {}
        self.vid_last = {}      # vid -> {ts,xy}
        self.vid_is_staff = {}
        self.vid_staff_enter_ts = {}
        self.vid_shop_last = {} # vid -> ts

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

    def update_staff_tag(self, vid: int, in_staff_area: bool, ts: float):
        if in_staff_area:
            if vid not in self.vid_staff_enter_ts:
                self.vid_staff_enter_ts[vid] = ts
            elif (ts - self.vid_staff_enter_ts[vid]) >= STAFF_DWELL_SEC:
                self.vid_is_staff[vid] = True
        else:
            if not self.vid_is_staff.get(vid, False):
                self.vid_staff_enter_ts.pop(vid, None)

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
# CAMERA PIPELINE
# =========================
class CameraPipeline:
    def __init__(self, cam_name: str, source: str, staff_gallery: Optional[StaffGallery], encoder: Optional[ReIDEncoder]):
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

        self.yolo_device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = YOLO(YOLO_MODEL_PATH)
        print(f"📷 {cam_name}: yolo_device={self.yolo_device} zones={len(self.zones)} source={'VIDEO' if USE_VIDEO else 'RTSP'}")

        self.encoder = encoder
        self.staff_gallery = staff_gallery

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
                classes=[0],
                imgsz=IMGSZ,
                device=self.yolo_device,
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
                    self.vid_mgr.update_staff_tag(vid, in_staff_zone, now)

                    hit_zones = zones_containing_point(self.zones, cx, cy)

                    emb = None
                    staff_id = None
                    if ENABLE_REID and self.encoder is not None:
                        x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
                        x1i = max(0, x1i); y1i = max(0, y1i)
                        x2i = min(W - 1, x2i); y2i = min(H - 1, y2i)
                        if (x2i - x1i) >= 50 and (y2i - y1i) >= 80:
                            crop = frame[y1i:y2i, x1i:x2i]
                            if crop is not None and crop.size > 0:
                                emb = self.encoder.encode_crop_bgr(crop)
                                if emb is not None and self.staff_gallery is not None:
                                    staff_id = self.staff_gallery.match_staff(emb, STAFF_SIM_THRESHOLD)
                                    if staff_id is not None:
                                        self.vid_mgr.vid_is_staff[vid] = True

                    dets.append({
                        "cam": self.cam_name,
                        "local_vid": vid,
                        "cx": cx, "cy": cy,
                        "bbox": [x1, y1, x2, y2],
                        "is_staff": self.vid_mgr.is_staff(vid),
                        "in_shop": self.vid_mgr.is_in_shop(vid, now),
                        "zones": hit_zones,
                        "ts": now,
                        "emb": emb,
                        "staff_id": staff_id,
                    })

            self.latest_ts = now
            self.latest_dets = dets

# =========================
# GLOBAL ID MANAGER
# =========================
class GlobalIDManager:
    def __init__(self):
        self.next_gid = 1
        self.local_to_gid = {}
        self.gid_last = {}  # gid -> {ts, xy, cam, zone}

    def _alloc_gid(self):
        gid = self.next_gid
        self.next_gid += 1
        return gid

    def _get_primary_zone(self, cam_name: str, zones: List[str]) -> Optional[str]:
        chairs = [z for z in zones if z.startswith(CHAIR_ZONE_PREFIX)]
        if chairs:
            return ZONE_MAP.get(cam_name, {}).get(chairs[0], chairs[0])
        washes = [z for z in zones if z.startswith(WASH_ZONE_PREFIX)]
        if washes:
            return ZONE_MAP.get(cam_name, {}).get(washes[0], washes[0])
        waits = [z for z in zones if z in WAIT_ZONE_NAMES]
        if waits:
            return ZONE_MAP.get(cam_name, {}).get(waits[0], waits[0])
        return None

    def assign_gid(self, det: dict) -> int:
        key = (det["cam"], det["local_vid"])
        ts = det["ts"]
        xy = (det["cx"], det["cy"])
        primary_zone = self._get_primary_zone(det["cam"], det["zones"])

        if key in self.local_to_gid:
            gid = self.local_to_gid[key]
            self.gid_last[gid] = {"ts": ts, "xy": xy, "cam": det["cam"], "zone": primary_zone}
            return gid

        best_gid = None
        best_score = 1e9
        for gid, info in self.gid_last.items():
            dt = abs(ts - info["ts"])
            if dt > GLOBAL_MATCH_MAX_SEC:
                continue

            if primary_zone is not None and info.get("zone") is not None:
                if primary_zone.startswith(CHAIR_ZONE_PREFIX) and info["zone"].startswith(CHAIR_ZONE_PREFIX):
                    if primary_zone != info["zone"]:
                        continue
                score = 0.0 + dt
            else:
                d2v = dist2(xy, info["xy"])
                if d2v > (GLOBAL_MATCH_MAX_DIST**2):
                    continue
                score = d2v + dt

            if score < best_score:
                best_score = score
                best_gid = gid

        gid = self._alloc_gid() if best_gid is None else best_gid
        self.local_to_gid[key] = gid
        self.gid_last[gid] = {"ts": ts, "xy": xy, "cam": det["cam"], "zone": primary_zone}
        return gid

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

class SnapshotManager:
    def __init__(self, base_dir: str, pad: int = 30, cam_limit: int = 2):
        self.base_dir = base_dir
        self.pad = pad
        self.cam_limit = max(1, cam_limit)

    def save_multi_cam(self, day: str, event_id: str, event_type: str, zone: str,
                       pid: int, gid: int, cams_order: List[str],
                       frames_by_cam: Dict[str, np.ndarray],
                       dets_by_cam: Dict[str, List[dict]]) -> List[str]:
        out_dir = day_dir(self.base_dir, day)
        ensure_dir(out_dir)

        saved_paths = []
        for cam_name in cams_order:
            if len(saved_paths) >= self.cam_limit:
                break
            frame = frames_by_cam.get(cam_name)
            if frame is None:
                continue

            dets = dets_by_cam.get(cam_name, [])
            best = None
            best_area = -1.0
            for d in dets:
                if d.get("pid") != pid and d.get("gid") != gid:
                    continue
                x1, y1, x2, y2 = d["bbox"]
                area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
                if area > best_area:
                    best_area = area
                    best = d
            if best is None:
                continue

            path = os.path.join(out_dir, f"{event_id}_{event_type}_{zone}_{cam_name}_pid{pid}_gid{gid}.jpg")
            p = crop_and_save_snapshot(frame, best["bbox"], path, pad=self.pad)
            if p:
                saved_paths.append(p)

        return saved_paths

# =========================
# ZONE HELPERS (Priority)
# =========================
def normalize_zone(cam: str, z: str) -> str:
    return ZONE_MAP.get(cam, {}).get(z, z)

def get_zone_hit_sets(det: dict) -> Set[str]:
    cam = det["cam"]
    return set(normalize_zone(cam, z) for z in det.get("zones", []))

def pick_primary_business_zone(det: dict) -> Optional[str]:
    zs = get_zone_hit_sets(det)

    chairs = [z for z in zs if z.startswith(CHAIR_ZONE_PREFIX)]
    if chairs:
        chairs.sort()
        return chairs[0]

    washes = [z for z in zs if z.startswith(WASH_ZONE_PREFIX)]
    if washes:
        washes.sort()
        return washes[0]

    waits = [z for z in zs if z in WAIT_ZONE_NAMES]
    if waits:
        return "WAIT"

    return None

# =========================
# SESSION COUNTERS
# =========================
class ZoneSessionCounter:
    def __init__(self, name: str, sit_min_sec: float, vacant_grace_sec: float):
        self.name = name
        self.sit_min_sec = sit_min_sec
        self.vacant_grace_sec = vacant_grace_sec

        self.day = today_key()
        self.seq = 0

        self.state: Dict[str, Dict] = {}
        self.total_count = 0
        self.zone_total: Dict[str, int] = {}
        self.zone_wait_seconds_sum: Dict[str, float] = {}
        self.zone_wait_last: Dict[str, float] = {}

        self.counted_active_pid: Set[int] = set()
        self.last_count_by_zone_pid: Dict[Tuple[str, int], float] = {}

    def _ensure_day(self):
        d = today_key()
        if d != self.day:
            self.day = d
            self.seq = 0
            self.state = {}
            self.total_count = 0
            self.zone_total = {}
            self.zone_wait_seconds_sum = {}
            self.zone_wait_last = {}
            self.counted_active_pid = set()
            self.last_count_by_zone_pid = {}
            print(f"🗓️ NEW DAY RESET ({self.name}): {self.day}")

    def sync_active_pids(self, active_pids: Set[int]):
        self.counted_active_pid = {pid for pid in self.counted_active_pid if pid in active_pids}

    def update_and_collect_events(self, customers: List[dict], zone_selector_fn) -> List[dict]:
        self._ensure_day()
        now = time.time()
        events = []

        best_by_zone: Dict[str, dict] = {}
        for d in customers:
            zone = zone_selector_fn(d)
            if zone is None:
                continue
            x1, y1, x2, y2 = d["bbox"]
            area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
            if (zone not in best_by_zone) or (area > best_by_zone[zone]["_bbox_area"]):
                dd = dict(d)
                dd["_bbox_area"] = area
                dd["_zone"] = zone
                best_by_zone[zone] = dd

        for zone, d in best_by_zone.items():
            st = self.state.setdefault(zone, {
                "occupied": False,
                "enter_ts": 0.0,
                "last_seen": 0.0,
                "counted": False,
                "pid": None,
                "gid": None,
            })

            pid = int(d.get("pid", 0))
            gid = int(d.get("gid", 0))

            st["last_seen"] = now

            if (not st["occupied"]) or (st["pid"] != pid):
                st["occupied"] = True
                st["enter_ts"] = now
                st["counted"] = False
                st["pid"] = pid
                st["gid"] = gid

            dwell = now - st["enter_ts"]

            if (not st["counted"]) and (dwell >= self.sit_min_sec):
                if pid in self.counted_active_pid:
                    st["counted"] = True
                    continue

                last_ts = self.last_count_by_zone_pid.get((zone, pid))
                if last_ts is not None and (now - last_ts) < RECOUNT_COOLDOWN_SEC:
                    st["counted"] = True
                    continue

                st["counted"] = True
                self.counted_active_pid.add(pid)
                self.last_count_by_zone_pid[(zone, pid)] = now

                self.total_count += 1
                self.zone_total[zone] = self.zone_total.get(zone, 0) + 1
                self.seq += 1

                event_id = ts_key()
                events.append({
                    "seq": self.seq,
                    "day": self.day,
                    "event_id": event_id,
                    "zone": zone,
                    "pid": pid,
                    "gid": gid,
                    "dwell": float(dwell),
                })

        for zone, st in list(self.state.items()):
            if zone in best_by_zone:
                continue
            if st["occupied"] and (now - st["last_seen"]) > self.vacant_grace_sec:
                wait_sec = max(0.0, now - st["enter_ts"])
                self.zone_wait_seconds_sum[zone] = self.zone_wait_seconds_sum.get(zone, 0.0) + wait_sec
                self.zone_wait_last[zone] = wait_sec

                print(
                    f"🧾 SESSION_END({self.name}) | day={self.day} zone={zone} "
                    f"wait_min={wait_sec/60:.1f} total_wait_min={self.zone_wait_seconds_sum[zone]/60:.1f}"
                )

                st["occupied"] = False
                st["enter_ts"] = 0.0
                st["counted"] = False
                st["pid"] = None
                st["gid"] = None

        return events

    def live_occupied_count(self) -> int:
        return sum(1 for st in self.state.values() if st.get("occupied"))

# =========================
# UI DRAW
# =========================
def draw_zones(frame, zones: List[Zone]):
    h, w = frame.shape[:2]
    for z in zones:
        pts = np.array([[int(p["x"] * w), int(p["y"] * h)] for p in z.polygon], np.int32)

        name = z.name
        if name in STAFF_NAMES:
            color = (255, 0, 255)
        elif name in SHOP_NAMES:
            color = (0, 255, 255)
        elif name in WAIT_ZONE_NAMES or name == "WAIT":
            color = (255, 255, 0)
        elif name.startswith(WASH_ZONE_PREFIX):
            color = (255, 128, 0)
        elif name.startswith(CHAIR_ZONE_PREFIX):
            color = (0, 255, 0)
        else:
            color = (180, 180, 180)

        cv2.polylines(frame, [pts], True, color, 2)
        cv2.putText(frame, name, (pts[0][0], max(20, pts[0][1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

def draw_people(frame, dets, gid_map):
    for d in dets:
        x1, y1, x2, y2 = map(int, d["bbox"])
        is_staff = d["is_staff"]
        color = (0, 0, 255) if is_staff else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        gid = gid_map.get((d["cam"], d["local_vid"]))
        label = f"G:{gid} L:{d['local_vid']}" + (" STAFF" if is_staff else "")
        cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

# =========================
# MAIN
# =========================
def main():
    print("💈 Global Multi-Camera Counter Starting...")

    ensure_dir(SNAPSHOT_DIR)
    ensure_dir(REPORT_DIR)
    ensure_dir(STAFF_GALLERY_DIR)
    cleanup_old_days(SNAPSHOT_DIR, SNAPSHOT_DAY_KEEP)

    encoder = None
    staff_gallery = None
    if ENABLE_REID:
        enc_dev = REID_DEVICE
        if enc_dev == "mps" and not torch.backends.mps.is_available():
            enc_dev = "cpu"
        if enc_dev == "cuda" and not torch.cuda.is_available():
            enc_dev = "cpu"

        print(f"🧠 ReID enabled | device={enc_dev}")
        encoder = ReIDEncoder(device=enc_dev)
        staff_gallery = StaffGallery(STAFF_GALLERY_DIR, STAFF_DB_PATH, encoder)
    else:
        print("🧠 ReID disabled (ENABLE_REID=0)")

    csv_logger = DailyCSVLogger(REPORT_DIR)

    haircut_counter = ZoneSessionCounter("HAIRCUT", SIT_MIN_SEC, VACANT_GRACE_SEC)
    wash_counter = ZoneSessionCounter("WASH", SIT_MIN_SEC, VACANT_GRACE_SEC)

    snap_mgr = SnapshotManager(SNAPSHOT_DIR, pad=SNAPSHOT_PAD, cam_limit=SNAPSHOT_CAM_LIMIT)

    if USE_VIDEO:
        sources = {"Camera_01": CAM1_VIDEO, "Camera_02": CAM2_VIDEO, "Camera_03": CAM3_VIDEO}
    else:
        sources = CAMERAS_CONFIG

    cam1 = CameraPipeline("Camera_01", sources["Camera_01"], staff_gallery, encoder)
    cam2 = CameraPipeline("Camera_02", sources["Camera_02"], staff_gallery, encoder)
    cam3 = CameraPipeline("Camera_03", sources["Camera_03"], staff_gallery, encoder)

    cam1.start()
    cam2.start()
    cam3.start()

    gid_mgr = GlobalIDManager()
    active_mem = ActivePersonMemory()

    def get_latest_frame(cam_name: str):
        if cam_name == "Camera_01":
            with cam1._lock:
                return None if cam1.latest_frame is None else cam1.latest_frame.copy()
        if cam_name == "Camera_02":
            with cam2._lock:
                return None if cam2.latest_frame is None else cam2.latest_frame.copy()
        if cam_name == "Camera_03":
            with cam3._lock:
                return None if cam3.latest_frame is None else cam3.latest_frame.copy()
        return None

    # ✅ BOOTSTRAP SCAN STAFF (tag staff early)
    def bootstrap_scan_staff():
        if not ENABLE_REID or encoder is None or staff_gallery is None:
            print("⚠️ BOOT SCAN skipped (ReID disabled or staff_gallery not ready)")
            return

        print(f"🔎 BOOT SCAN staff for {BOOT_SCAN_SEC}s ...")
        end_ts = time.time() + BOOT_SCAN_SEC
        interval = 1.0 / max(1, BOOT_SCAN_FPS)
        last = 0.0
        tagged = 0

        while time.time() < end_ts:
            now = time.time()
            if (now - last) < interval:
                time.sleep(0.01)
                continue
            last = now

            for cam in [cam1, cam2, cam3]:
                dets = cam.latest_dets or []
                with cam._lock:
                    frame = None if cam.latest_frame is None else cam.latest_frame.copy()
                if frame is None:
                    continue

                H, W = frame.shape[:2]
                for d in dets:
                    if not d.get("in_shop", False):
                        continue
                    if d.get("is_staff", False):
                        continue

                    x1, y1, x2, y2 = map(int, d["bbox"])
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(W - 1, x2); y2 = min(H - 1, y2)
                    if (x2 - x1) < 50 or (y2 - y1) < 80:
                        continue

                    crop = frame[y1:y2, x1:x2]
                    emb = encoder.encode_crop_bgr(crop)
                    sid = staff_gallery.match_staff(emb, STAFF_SIM_THRESHOLD)
                    if sid is not None:
                        cam.vid_mgr.vid_is_staff[d["local_vid"]] = True
                        tagged += 1

            time.sleep(0.01)

        print(f"✅ BOOT SCAN done | tagged_staff={tagged}")

    bootstrap_scan_staff()

    last_log_ts = 0.0
    cams_order = ["Camera_01", "Camera_02", "Camera_03"]

    # selectors
    def select_haircut_zone(det: dict) -> Optional[str]:
        z = pick_primary_business_zone(det)
        if z and z.startswith(CHAIR_ZONE_PREFIX):
            return z
        return None

    def select_wash_zone(det: dict) -> Optional[str]:
        z = pick_primary_business_zone(det)
        if z and z.startswith(WASH_ZONE_PREFIX):
            return "WASH"
        return None

    def is_wait(det: dict) -> bool:
        z = pick_primary_business_zone(det)
        return (z == "WAIT")

    try:
        while True:
            now = time.time()

            f1 = get_latest_frame("Camera_01")
            f2 = get_latest_frame("Camera_02")
            f3 = get_latest_frame("Camera_03")

            dets1 = cam1.latest_dets or []
            dets2 = cam2.latest_dets or []
            dets3 = cam3.latest_dets or []

            active_mem.expire(now, REID_ACTIVE_EXPIRE_SEC)
            active_pids = set(active_mem.pid_last.keys())
            haircut_counter.sync_active_pids(active_pids)
            wash_counter.sync_active_pids(active_pids)

            customers: List[dict] = []
            per_cam_customers = {"Camera_01": [], "Camera_02": [], "Camera_03": []}

            for d in (dets1 + dets2 + dets3):
                if not d.get("in_shop", False):
                    continue

                # staff gallery match => staff
                if d.get("staff_id") is not None:
                    d["is_staff"] = True
                if d.get("is_staff", False):
                    continue

                gid = gid_mgr.assign_gid(d)

                pid = gid
                if ENABLE_REID:
                    emb = d.get("emb")
                    pid = active_mem.resolve_pid(gid, emb, now, REID_SIM_THRESHOLD)

                dd = dict(d)
                dd["gid"] = int(gid)
                dd["pid"] = int(pid)
                dd["primary_zone"] = pick_primary_business_zone(dd)

                customers.append(dd)
                per_cam_customers[dd["cam"]].append(dd)

            wait_pids = {d["pid"] for d in customers if is_wait(d)}
            wait_realtime = len(wait_pids)

            haircut_events = haircut_counter.update_and_collect_events(customers, select_haircut_zone)
            wash_events = wash_counter.update_and_collect_events(customers, select_wash_zone)

            if haircut_events or wash_events:
                cleanup_old_days(SNAPSHOT_DIR, SNAPSHOT_DAY_KEEP)

                frames_by_cam = {"Camera_01": f1, "Camera_02": f2, "Camera_03": f3}
                dets_by_cam = per_cam_customers

                def process_event(ev: dict, event_type: str):
                    event_id = ev["event_id"]
                    zone = ev["zone"]
                    pid = ev["pid"]
                    gid = ev["gid"]
                    dwell = ev["dwell"]

                    paths = []
                    if SNAPSHOT_MULTI_CAM:
                        paths = snap_mgr.save_multi_cam(
                            day=ev["day"],
                            event_id=event_id,
                            event_type=event_type,
                            zone=zone,
                            pid=pid,
                            gid=gid,
                            cams_order=cams_order,
                            frames_by_cam=frames_by_cam,
                            dets_by_cam=dets_by_cam,
                        )

                    csv_logger.log_event(
                        seq=ev["seq"],
                        event_id=event_id,
                        event_type=event_type,
                        zone=zone,
                        pid=pid,
                        gid=gid,
                        dwell_sec=dwell,
                        snapshot_paths="|".join(paths),
                    )

                    print(
                        f"✅ {event_type} | day={ev['day']} seq={ev['seq']} event={event_id} "
                        f"zone={zone} pid={pid} gid={gid} dwell={dwell:.1f}s snaps={len(paths)} "
                        f"haircut_total={haircut_counter.total_count} wash_total={wash_counter.total_count}"
                    )

                for ev in haircut_events:
                    process_event(ev, "HAIRCUT")
                for ev in wash_events:
                    process_event(ev, "WASH")

            if (now - last_log_ts) >= 2.0:
                last_log_ts = now
                live_pids = len(set([d["pid"] for d in customers]))

                print(
                    f"📊 DAY={today_key()} | LIVE_CUSTOMERS(pid)={live_pids} | WAIT_REALTIME={wait_realtime} "
                    f"| CUT_IN_PROGRESS={haircut_counter.live_occupied_count()} | WASH_IN_PROGRESS={wash_counter.live_occupied_count()} "
                    f"| HAIRCUT_COUNT={haircut_counter.total_count} | WASH_COUNT={wash_counter.total_count}"
                )

            if SHOW_WINDOW:
                if f1 is not None:
                    draw_zones(f1, cam1.zones)
                    draw_people(f1, cam1.latest_dets or [], gid_mgr.local_to_gid)
                    draw_hud(
                        f1, "Camera_01",
                        haircut_counter.total_count, wash_counter.total_count, wait_realtime,
                        haircut_counter.live_occupied_count(), wash_counter.live_occupied_count()
                    )
                    cv2.imshow("Camera_01", f1)

                if f2 is not None:
                    draw_zones(f2, cam2.zones)
                    draw_people(f2, cam2.latest_dets or [], gid_mgr.local_to_gid)
                    draw_hud(
                        f2, "Camera_02",
                        haircut_counter.total_count, wash_counter.total_count, wait_realtime,
                        haircut_counter.live_occupied_count(), wash_counter.live_occupied_count()
                    )
                    cv2.imshow("Camera_02", f2)

                if f3 is not None:
                    draw_zones(f3, cam3.zones)
                    draw_people(f3, cam3.latest_dets or [], gid_mgr.local_to_gid)
                    draw_hud(
                        f3, "Camera_03",
                        haircut_counter.total_count, wash_counter.total_count, wait_realtime,
                        haircut_counter.live_occupied_count(), wash_counter.live_occupied_count()
                    )
                    cv2.imshow("Camera_03", f3)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            time.sleep(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        cam1.stop()
        cam2.stop()
        cam3.stop()
        cv2.destroyAllWindows()
        csv_logger.close()
        print("🛑 Stopped.")

if __name__ == "__main__":
    main()
