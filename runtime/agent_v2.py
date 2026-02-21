"""
Enhanced Runtime Service with Event Tracking & Supabase

Features:
  - Multi-camera YOLO detection
  - Event tracking (haircut, wash, wait)
  - CSV report generation
  - Supabase event submission with retry
  - Heartbeat status updates
  - Zone-based counting
  - RTSP Watchdog (auto-reconnection)
  - Resource Guards (FPS/memory limits)
  - Health Checks (periodic diagnostics)
"""

import os
import sys
import time
import json
import platform
import socket
import threading
import logging
import tempfile
import gc
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Set
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

import numpy as np
import cv2
import torch
from ultralytics import YOLO
try:
    import torchvision.transforms as T
    import torchvision.models as tv_models
except Exception:
    T = None
    tv_models = None

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import Config
from shared.logger import setup_logger
from shared.event_tracker import EventTracker, EventType, Event
from shared.report_generator import ReportGenerator
from shared.supabase_client import SupabaseClient, SupabaseSync
from shared.dashboard_updater import init_dashboard_service, get_broadcaster
from shared.rtsp_watchdog import RTSPWatchdog, CameraHealth
from shared.resource_guard import ResourceGuard
from shared.health_checker import HealthChecker

# =========================
# SETUP
# =========================
CONFIG = Config("data/config/config.yaml")
logger = setup_logger("runtime", CONFIG.get("paths", {}).get("logs", "logs"))
RUNTIME_BUILD = "2026-02-16-stallfix-b3"

logger.info(f"Project: {CONFIG.get('project_name')}")
logger.info(f"Branch: {CONFIG.get('branch_code')}")
logger.info(f"Runtime build: {RUNTIME_BUILD}")

# =========================
# CONFIG
# =========================
CAMERAS_CONFIG = CONFIG.get("cameras", {})
YOLO_CONFIG = CONFIG.get("yolo", {})
RUNTIME_CONFIG = CONFIG.get("runtime", {})
PATHS_CONFIG = CONFIG.get("paths", {})
DWELL_TIME_CONFIG = CONFIG.get("dwell_time", {})
SUPABASE_CONFIG = CONFIG.get("supabase", {})

YOLO_MODEL_PATH = os.path.join(PATHS_CONFIG.get("models", "models"), YOLO_CONFIG.get("model", "yolov8m.pt"))
HEARTBEAT_INTERVAL = RUNTIME_CONFIG.get("heartbeat_interval", 30)

SNAPSHOT_DIR = PATHS_CONFIG.get("snapshots", "snapshots")
REPORTS_DIR = PATHS_CONFIG.get("reports", "reports")
ZONES_DIR = PATHS_CONFIG.get("zones", "data/zones")
DASHBOARD_STATE_FILE = PATHS_CONFIG.get("dashboard_state", "runtime/dashboard_state.json")
RESET_COUNTS_FLAG_FILE = PATHS_CONFIG.get("reset_counts_flag", "runtime/reset_counts.flag")
SETTINGS_OVERRIDE_FILE = PATHS_CONFIG.get("runtime_settings_override", "runtime/runtime_settings.override.json")
SNAPSHOT_PAD = int(RUNTIME_CONFIG.get("snapshot_pad", 30))

# =========================
# DATA CLASSES
# =========================

@dataclass
class Detection:
    """YOLO detection"""
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int
    
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def foot_point(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, self.y2)


@dataclass
class Track:
    """Tracked person"""
    track_id: int
    camera: str
    detections: List[Detection] = field(default_factory=list)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    
    def update(self, detection: Detection):
        self.detections.append(detection)
        self.last_seen = time.time()


# =========================
# ZONE UTILS
# =========================

def load_zones(camera_name: str) -> Dict[str, List[Tuple[float, float]]]:
    """Load zone polygons from JSON"""
    zones_file = os.path.join(ZONES_DIR, f"zones_{camera_name}.json")
    
    if not os.path.exists(zones_file):
        logger.warning(f"Zones file not found: {zones_file}")
        return {}
    
    try:
        with open(zones_file, 'r', encoding='utf-8') as f:
            zones_data = json.load(f)
        
        zones = {}
        for zone_obj in zones_data:
            zone_name = zone_obj.get("name", "")
            polygon = zone_obj.get("polygon_json", [])
            if not polygon:
                polygon = zone_obj.get("points", [])
            
            if not zone_name or not polygon:
                continue

            pts = []
            for p in polygon:
                if isinstance(p, dict):
                    pts.append((float(p.get("x", 0.0)), float(p.get("y", 0.0))))
                else:
                    pts.append((float(p[0]), float(p[1])))

            # Normalize if points look like pixel coordinates.
            if not all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in pts):
                max_x = max(x for x, _ in pts) if pts else 1.0
                max_y = max(y for _, y in pts) if pts else 1.0
                max_x = max(max_x, 1.0)
                max_y = max(max_y, 1.0)
                pts = [(x / max_x, y / max_y) for x, y in pts]

            zones[zone_name] = pts
        
        logger.info(f"Loaded {len(zones)} zones for {camera_name}")
        return zones
    except Exception as e:
        logger.error(f"Error loading zones for {camera_name}: {e}")
        return {}


def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """Check if point is inside polygon (normalized coordinates)"""
    if len(polygon) < 3:
        return False
    
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


WAIT_ZONE_NAMES = {"WAIT", "WAITING", "WAIT_AREA"}
WASH_ZONE_PREFIX = "WASH"
CHAIR_ZONE_PREFIX = "CHAIR_"


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return -1.0
    if a.shape != b.shape:
        return -1.0
    return float(np.dot(a, b))


class ReIDEncoder:
    """ResNet50 embedding encoder (same direction as edge_agent flow)."""
    def __init__(self, device: str = "cpu"):
        if tv_models is None or T is None:
            raise RuntimeError("torchvision is not available")
        self.device = device
        model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)
        model.fc = torch.nn.Identity()
        model.eval()
        self.model = model.to(device)
        self.tf = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.staff_embs: List[Tuple[str, np.ndarray]] = []
        self._load()

    def _load(self):
        if not os.path.exists(self.db_path):
            logger.warning(f"Staff DB not found: {self.db_path}")
            return
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            embs: List[Tuple[str, np.ndarray]] = []
            # Format A: {"items":[{"person_id":"X","emb":[...]}, ...]}
            if isinstance(data, dict) and isinstance(data.get("items"), list):
                for it in data.get("items", []):
                    if "person_id" in it and "emb" in it:
                        embs.append((str(it["person_id"]), np.array(it["emb"], dtype=np.float32)))
                    elif "staff_id" in it and "avg_vector" in it:
                        embs.append((str(it["staff_id"]), np.array(it["avg_vector"], dtype=np.float32)))
            # Format B (runtime/build_staff_db.py): {"STAFF_A":{"embeddings":[...], ...}, ...}
            elif isinstance(data, dict):
                for sid, info in data.items():
                    if not isinstance(info, dict):
                        continue
                    for emb in info.get("embeddings", []):
                        embs.append((str(sid), np.array(emb, dtype=np.float32)))

            self.staff_embs = embs
            logger.info(f"Loaded staff embeddings: {len(self.staff_embs)}")
        except Exception as e:
            logger.error(f"Failed loading staff DB: {e}")
            self.staff_embs = []

    def match_staff(self, emb: Optional[np.ndarray], thr: float) -> Optional[str]:
        if emb is None or not self.staff_embs:
            return None
        best_id, best = None, -1.0
        for pid, e in self.staff_embs:
            if e is None or e.shape != emb.shape:
                continue
            s = cosine_sim(emb, e)
            if s > best:
                best = s
                best_id = pid
        return best_id if best_id is not None and best >= thr else None


class ActivePersonMemory:
    def __init__(self):
        self.next_pid = 1
        self.pid_last: Dict[int, Dict] = {}
        self.gid_to_pid: Dict[int, int] = {}

    def expire(self, now_ts: float, expire_sec: float):
        dead = []
        for pid, info in self.pid_last.items():
            if (now_ts - info["ts"]) > expire_sec:
                dead.append(pid)
        for pid in dead:
            self.pid_last.pop(pid, None)
        self.gid_to_pid = {g: p for g, p in self.gid_to_pid.items() if p in self.pid_last}

    def resolve_pid(
        self,
        gid: int,
        emb: Optional[np.ndarray],
        now_ts: float,
        thr: float,
        match_window_sec: float,
    ) -> int:
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
                if (now_ts - float(info.get("ts", 0.0))) > match_window_sec:
                    continue
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

    def reset(self):
        self.next_pid = 1
        self.pid_last = {}
        self.gid_to_pid = {}


class GlobalIDManager:
    def __init__(self):
        self.next_gid = 1
        self.local_to_gid: Dict[Tuple[str, int], int] = {}
        self.gid_last: Dict[int, Dict] = {}

    def _alloc_gid(self):
        gid = self.next_gid
        self.next_gid += 1
        return gid

    def assign_gid(self, det: Dict) -> int:
        key = (det["cam"], int(det["local_vid"]))
        ts = float(det["ts"])
        xy = (float(det["cx"]), float(det["cy"]))
        primary_zone = det.get("primary_zone")

        if key in self.local_to_gid:
            gid = self.local_to_gid[key]
            self.gid_last[gid] = {"ts": ts, "xy": xy, "cam": det["cam"], "zone": primary_zone}
            return gid

        # Keep GID assignment stable per camera-local track id only.
        # Cross-camera merge is handled later by ActivePersonMemory (embedding similarity).
        # This avoids false merges when multiple people are present simultaneously.
        gid = self._alloc_gid()
        self.local_to_gid[key] = gid
        self.gid_last[gid] = {"ts": ts, "xy": xy, "cam": det["cam"], "zone": primary_zone}
        return gid

    def reset(self):
        self.next_gid = 1
        self.local_to_gid = {}
        self.gid_last = {}


class ZoneSessionCounter:
    def __init__(self, name: str, sit_min_sec: float, vacant_grace_sec: float):
        self.name = name
        self.sit_min_sec = sit_min_sec
        self.vacant_grace_sec = vacant_grace_sec
        self.day = datetime.now().strftime("%Y-%m-%d")
        self.seq = 0
        self.state: Dict[str, Dict] = {}
        self.total_count = 0
        self.zone_total: Dict[str, int] = {}
        self.counted_active_pid: Set[int] = set()
        self.last_count_by_zone_pid: Dict[Tuple[str, int], float] = {}

    def _ensure_day(self):
        d = datetime.now().strftime("%Y-%m-%d")
        if d != self.day:
            self.day = d
            self.seq = 0
            self.state = {}
            self.total_count = 0
            self.zone_total = {}
            self.counted_active_pid = set()
            self.last_count_by_zone_pid = {}

    def sync_active_pids(self, active_pids: Set[int]):
        self.counted_active_pid = {pid for pid in self.counted_active_pid if pid in active_pids}

    def update_and_collect_events(self, customers: List[dict], zone_selector_fn) -> List[dict]:
        self._ensure_day()
        now = time.time()
        events = []
        
        # Deduplication window: If a haircut was counted in this zone < 15 mins ago, don't count another.
        # This prevents double counting when tracker ID changes but it's likely the same customer.
        DEDUPE_WINDOW_SEC = 900.0  # 15 minutes
        if not hasattr(self, "zone_last_event_ts"):
            self.zone_last_event_ts = {}

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

            if not st["occupied"]:
                st["occupied"] = True
                st["enter_ts"] = now
                st["counted"] = False
                st["pid"] = pid
                st["gid"] = gid
            elif st["pid"] != pid:
                # Keep dwell timer when tracker/reid id jitters while the zone is
                # still continuously occupied. This prevents false "never reaches
                # sit_min_sec" behavior.
                st["pid"] = pid
                st["gid"] = gid

            dwell = now - st["enter_ts"]
            if (not st["counted"]) and (dwell >= self.sit_min_sec):
                if pid in self.counted_active_pid:
                    st["counted"] = True
                    continue

                # Check if this zone already had an event recently (debounce)
                last_ts = self.zone_last_event_ts.get(zone, 0.0)
                if (now - last_ts) < DEDUPE_WINDOW_SEC:
                    # Valid event already counted recently.
                    # Just mark this PID as counted so we don't try again for this session.
                    st["counted"] = True
                    self.counted_active_pid.add(pid)
                    continue

                st["counted"] = True
                self.counted_active_pid.add(pid)
                self.last_count_by_zone_pid[(zone, pid)] = now
                self.zone_last_event_ts[zone] = now  # Update debounce timestamp
                self.total_count += 1
                self.zone_total[zone] = self.zone_total.get(zone, 0) + 1
                self.seq += 1
                events.append({
                    "seq": self.seq,
                    "day": self.day,
                    "event_id": datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")[:-3],
                    "zone": zone,
                    "pid": pid,
                    "gid": gid,
                    "dwell": float(dwell),
                    "camera": d.get("cam", ""),
                    "bbox": d.get("bbox"),
                })

        for zone, st in list(self.state.items()):
            if zone in best_by_zone:
                continue
            if st["occupied"] and (now - st["last_seen"]) > self.vacant_grace_sec:
                st["occupied"] = False
                st["enter_ts"] = 0.0
                st["counted"] = False
                st["pid"] = None
                st["gid"] = None
        return events

    def live_occupied_count(self) -> int:
        return sum(1 for st in self.state.values() if st.get("occupied"))

    def reset(self):
        self.seq = 0
        self.state = {}
        self.total_count = 0
        self.zone_total = {}
        self.counted_active_pid = set()
        self.last_count_by_zone_pid = {}


# =========================
# CAMERA STREAM
# =========================

class CameraStream:
    """Capture from RTSP camera"""
    
    def __init__(
        self,
        camera_name: str,
        rtsp_url: str,
        freeze_max_same_frames: int = 120,
        freeze_diff_threshold: float = 1.2,
        connect_policy: Optional[Dict[str, bool]] = None,
    ):
        self.camera_name = camera_name
        self.rtsp_url = rtsp_url
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_connected = False
        self.frame: Optional[np.ndarray] = None
        self.fps = 0
        self.lock = threading.Lock()
        self.cap_lock = threading.RLock()
        self.freeze_max_same_frames = int(max(10, freeze_max_same_frames))
        self.freeze_diff_threshold = float(max(0.0, freeze_diff_threshold))
        self._last_small_gray: Optional[np.ndarray] = None
        self._same_frame_count = 0
        self.reconnect_requested = False
        self._connect_round = 0
        self.connect_policy = dict(connect_policy or {})
        
        self.connect()
    
    def connect(self):
        """Connect to camera"""
        if not self.rtsp_url:
            logger.warning(f"No RTSP URL for {self.camera_name}")
            return
        
        try:
            with self.cap_lock:
                if self.cap is not None:
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = None
                self.is_connected = False

            # Fast network pre-check to avoid long VideoCapture hangs when camera is unreachable.
            parsed = urlparse(self.rtsp_url)
            host = parsed.hostname
            port = parsed.port or 554
            if host:
                try:
                    with socket.create_connection((host, port), timeout=2.0):
                        pass
                except Exception as e:
                    logger.error(
                        f"Cannot reach camera endpoint for {self.camera_name}: "
                        f"{host}:{port} ({e})"
                    )
                    return

            attempts = []
            # Backend priority is runtime-tunable because some environments are
            # more stable on default/udp than ffmpeg_tcp (and vice versa).
            prefer_tcp = bool(self.connect_policy.get("prefer_tcp", RUNTIME_CONFIG.get("rtsp_prefer_tcp", False)))
            prefer_ffmpeg_only = bool(self.connect_policy.get("force_ffmpeg", RUNTIME_CONFIG.get("rtsp_force_ffmpeg", False)))
            allow_udp_fallback = bool(self.connect_policy.get("allow_udp_fallback", RUNTIME_CONFIG.get("rtsp_allow_udp_fallback", False)))
            rotate_backends = bool(self.connect_policy.get("rotate_backends", RUNTIME_CONFIG.get("rtsp_rotate_backends", False)))

            def _append_rtsp_params(url: str, kv: Dict[str, str]) -> str:
                try:
                    p = urlparse(url)
                    q = dict(parse_qsl(p.query, keep_blank_values=True))
                    changed = False
                    for k, v in kv.items():
                        if k not in q:
                            q[k] = v
                            changed = True
                    if not changed:
                        return url
                    return urlunparse((p.scheme, p.netloc, p.path, p.params, urlencode(q), p.fragment))
                except Exception:
                    return url

            ffmpeg_common = {
                # 3s IO timeout (microseconds) to avoid indefinite read() blocks.
                "stimeout": "3000000",
                "rw_timeout": "3000000",
            }
            tcp_url = _append_rtsp_params(self.rtsp_url, {"rtsp_transport": "tcp", **ffmpeg_common})
            ffmpeg_url = _append_rtsp_params(self.rtsp_url, ffmpeg_common)

            if prefer_ffmpeg_only:
                # In low-spec environments, UDP fallback often "connects" but then
                # stalls. Keep TCP strict by default unless explicitly enabled.
                if prefer_tcp:
                    attempts.append(("ffmpeg_tcp", tcp_url, cv2.CAP_FFMPEG))
                    if allow_udp_fallback:
                        attempts.append(("ffmpeg", ffmpeg_url, cv2.CAP_FFMPEG))
                else:
                    attempts.append(("ffmpeg", ffmpeg_url, cv2.CAP_FFMPEG))
                    attempts.append(("ffmpeg_tcp", tcp_url, cv2.CAP_FFMPEG))
            elif prefer_tcp:
                attempts.append(("ffmpeg_tcp", tcp_url, cv2.CAP_FFMPEG))
                if allow_udp_fallback:
                    attempts.append(("ffmpeg", ffmpeg_url, cv2.CAP_FFMPEG))
                attempts.append(("default", self.rtsp_url, None))
            else:
                # Default first tends to be more resilient on some low-spec macOS systems.
                attempts.append(("default", self.rtsp_url, None))
                attempts.append(("ffmpeg", ffmpeg_url, cv2.CAP_FFMPEG))
                attempts.append(("ffmpeg_tcp", tcp_url, cv2.CAP_FFMPEG))

            logger.info(
                f"{self.camera_name}: connect attempts={[x[0] for x in attempts]} "
                f"(prefer_tcp={prefer_tcp}, force_ffmpeg={prefer_ffmpeg_only}, "
                f"udp_fallback={allow_udp_fallback}, rotate={rotate_backends})"
            )

            # Rotate first-choice backend only when explicitly enabled.
            if rotate_backends and len(attempts) > 1:
                offset = self._connect_round % len(attempts)
                attempts = attempts[offset:] + attempts[:offset]
            self._connect_round += 1

            last_error = ""
            for backend_name, url, backend in attempts:
                cap = cv2.VideoCapture(url) if backend is None else cv2.VideoCapture(url, backend)
                if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
                if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)
                if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                if cap.isOpened():
                    with self.cap_lock:
                        self.cap = cap
                        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
                        self.is_connected = True
                    logger.info(f"Connected to {self.camera_name} at {self.fps:.1f} FPS via {backend_name}")
                    return

                cap.release()
                last_error = f"backend={backend_name}, url={url}"

            logger.error(f"Cannot open camera: {self.camera_name} ({last_error})")
        except Exception as e:
            logger.error(f"Error connecting to {self.camera_name}: {e}")
    
    def read_frame(self) -> Optional[np.ndarray]:
        """Read next frame"""
        try:
            with self.cap_lock:
                if not self.is_connected or self.cap is None:
                    return None
                cap = self.cap
            ret, frame = cap.read()
            
            if not ret:
                logger.warning(f"Cannot read frame from {self.camera_name}")
                with self.cap_lock:
                    self.is_connected = False
                    self.reconnect_requested = True
                return None
            
            with self.lock:
                self.frame = frame

            # Detect frozen RTSP stream (identical/almost-identical frames for too long).
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                small = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA)
                if self._last_small_gray is not None:
                    diff = cv2.absdiff(small, self._last_small_gray)
                    mean_diff = float(np.mean(diff))
                    if mean_diff <= self.freeze_diff_threshold:
                        self._same_frame_count += 1
                    else:
                        self._same_frame_count = 0
                self._last_small_gray = small
                if self._same_frame_count >= self.freeze_max_same_frames:
                    logger.warning(
                        f"{self.camera_name}: stream appears frozen "
                        f"(same frame x{self._same_frame_count}), forcing reconnect"
                    )
                    with self.cap_lock:
                        self.is_connected = False
                        self.reconnect_requested = True
                    return None
            except Exception:
                # Freeze detection must never break normal frame processing.
                pass
            
            return frame
        except Exception as e:
            logger.error(f"Error reading frame from {self.camera_name}: {e}")
            with self.cap_lock:
                self.is_connected = False
                self.reconnect_requested = True
            return None
    
    def test_connection(self) -> bool:
        """Test and reconnect to camera"""
        try:
            # Try to disconnect first
            self.disconnect()
            
            # Wait a bit
            time.sleep(1)
            
            # Try to reconnect
            self.connect()
            
            # Try to read a frame to verify
            with self.cap_lock:
                cap = self.cap
            if cap and cap.isOpened():
                with self.cap_lock:
                    ret, _ = cap.read()
                if ret:
                    with self.cap_lock:
                        self.is_connected = True
                    logger.info(f"Successfully reconnected to {self.camera_name}")
                    return True
            
            with self.cap_lock:
                self.is_connected = False
                self.reconnect_requested = True
            return False
        except Exception as e:
            logger.error(f"Error testing connection for {self.camera_name}: {e}")
            with self.cap_lock:
                self.is_connected = False
                self.reconnect_requested = True
            return False
    
    def disconnect(self):
        """Disconnect from camera"""
        with self.cap_lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.is_connected = False
            self.reconnect_requested = False
            self._last_small_gray = None
            self._same_frame_count = 0
        logger.info(f"Disconnected from {self.camera_name}")


# =========================
# TRACKER
# =========================

class MultiCameraTracker:
    """Track people across cameras"""
    
    def __init__(self):
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        self.lock = threading.Lock()
    
    def update(self, camera: str, detections: List[Detection]):
        """Update tracker with detections"""
        with self.lock:
            for detection in detections:
                closest_track = None
                min_dist = float('inf')
                
                for track in self.tracks.values():
                    if track.camera != camera:
                        continue
                    
                    if len(track.detections) == 0:
                        continue
                    
                    last_det = track.detections[-1]
                    last_center = last_det.center()
                    curr_center = detection.center()
                    
                    dist = ((last_center[0] - curr_center[0])**2 + 
                           (last_center[1] - curr_center[1])**2)**0.5
                    
                    if dist < min_dist and dist < 0.1:
                        min_dist = dist
                        closest_track = track
                
                if closest_track:
                    closest_track.update(detection)
                else:
                    new_track = Track(
                        track_id=self.next_track_id,
                        camera=camera,
                        detections=[detection]
                    )
                    self.tracks[self.next_track_id] = new_track
                    self.next_track_id += 1
    
    def get_active_tracks(self, camera: str, max_age: float = 5.0) -> List[Track]:
        """Get active tracks"""
        current_time = time.time()
        active = []
        
        with self.lock:
            for track in self.tracks.values():
                if track.camera == camera:
                    if current_time - track.last_seen < max_age:
                        active.append(track)
        
        return active

    def get_active_count(self, max_age: float = 5.0) -> int:
        """Get total active tracks across all cameras."""
        current_time = time.time()
        with self.lock:
            return sum(1 for track in self.tracks.values() if (current_time - track.last_seen) < max_age)

    def reset(self):
        """Reset tracker state."""
        with self.lock:
            self.tracks = {}
            self.next_track_id = 1


# =========================
# RUNTIME SERVICE
# =========================

class RuntimeService:
    """Main runtime service with reliability features"""
    
    def __init__(self):
        self.yolo_model, self.yolo_device = self._load_yolo()
        self.yolo_lock = threading.Lock()
        self.cameras: Dict[str, CameraStream] = {}
        self.tracker = MultiCameraTracker()
        self.zones: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
        
        # Event tracking
        self.event_tracker = EventTracker(CONFIG.data)
        self.report_gen = ReportGenerator(REPORTS_DIR)
        self.all_events: List = []
        self.pending_events: List = []
        self.pending_lock = threading.Lock()
        self.recent_snapshots: List[Dict] = []
        
        # Supabase
        self.supabase_client: Optional[SupabaseClient] = None
        self.supabase_sync: Optional[SupabaseSync] = None
        
        # Dashboard broadcasting
        self.broadcaster = init_dashboard_service()
        self.last_status_broadcast = 0
        self.last_summary_broadcast = 0
        
        # Phase 3B: Reliability features
        self.watchdog = RTSPWatchdog(logger, max_retries=10)
        self.resource_guard = ResourceGuard(
            logger,
            max_fps=30.0,
            max_memory_percent=80.0
        )
        self.health_checker = HealthChecker(logger, check_interval=30.0)
        
        # Setup watchdog callbacks
        self.watchdog.on_camera_offline = self._on_camera_offline
        self.watchdog.on_camera_online = self._on_camera_online
        self.health_checker.on_check_failed = self._on_health_check_failed
        
        self.running = False
        self.threads: List[threading.Thread] = []
        self.last_report_time = datetime.now()
        self.last_health_broadcast = 0
        self.customers_lock = threading.Lock()
        self.latest_customers: Dict[str, List[Dict]] = {}
        self.latest_frames: Dict[str, np.ndarray] = {}
        self._last_frame_ok_ts: Dict[str, float] = {}
        self._fallback_track_lock = threading.Lock()
        self._fallback_tracks: Dict[str, Dict[int, Dict[str, float]]] = {}
        self._next_fallback_tid = 1
        self._last_noid_log_ts: Dict[str, float] = {}
        self._last_forced_recover_ts: Dict[str, float] = {}
        self._camera_generation: Dict[str, int] = {}
        self._camera_lock = threading.Lock()
        self._restart_backoff_sec: Dict[str, float] = {}
        self._restart_backoff_until: Dict[str, float] = {}
        self._stall_restart_count: Dict[str, int] = {}
        self._camera_connect_policy: Dict[str, Dict[str, bool]] = {}
        self._camera_good_frame_count: Dict[str, int] = {}

        runtime_cfg = CONFIG.get("runtime", {})
        staff_cfg = CONFIG.get("staff", {})
        self.resource_guard.max_memory_percent = float(runtime_cfg.get("resource_max_memory_percent", 95.0))
        self.throttle_log_interval_sec = float(runtime_cfg.get("throttle_log_interval_sec", 5.0))
        self._last_throttle_log_ts: Dict[str, float] = {}
        self.zone_point_mode = runtime_cfg.get("zone_point_mode", "foot")
        self.sit_min_sec = float(runtime_cfg.get("sit_min_sec", 10))
        self.vacant_grace_sec = float(runtime_cfg.get("vacant_grace_sec", 6))
        
        # Missing attributes restored
        self.yolo_conf = 0.5
        self.yolo_iou = 0.5
        self.yolo_imgsz = 640
        self.yolo_mode = "track"  # default
        self.camera_stall_timeout_sec = 60.0
        self.camera_freeze_max_same_frames = 120
        self.camera_freeze_diff_threshold = 1.2
        self.inference_mode = "always"
        self.motion_diff_threshold = 25.0
        self.motion_min_area_ratio = 0.005
        self.motion_recheck_sec = 2.0
        self.motion_hold_sec = 5.0
        self.motion_downscale_width = 128
        self.customer_active_ttl_sec = 10.0
        self.dashboard_presence_ttl_sec = 60.0
        self.no_detection_hold_sec = 2.0
        self.reid_similarity_threshold = 0.75
        self.staff_similarity_threshold = 0.82
        self.reid_active_expire_sec = 3600.0  # 1 hour memory by default
        self.reid_match_window_sec = 30.0
        self.cross_camera_dedupe_window_sec = 15.0
        self.cross_camera_dedupe_similarity = 0.75
        self.reid_merge_for_counts = True
        self.exclude_staff_from_counts = True
        self._last_det_stats_log_ts = {}
        self._last_process_ts = {}
        self._last_camera_heartbeat = {}
        self._last_pid_canon_log_ts = 0
        self._last_pid_canon_ok_log_ts = 0
        self._last_camera_nonzero = {}
        self.camera_heartbeat_sec = 30.0
        self.camera_no_detection_warn_sec = 300.0
        # Custom classes for fine-grained salon detection.
        self.person_class_id = int(YOLO_CONFIG.get("person_class_id", 0))
        self.head_class_id = int(YOLO_CONFIG.get("head_class_id", 1))
        self.staff_uniform_class_id = int(YOLO_CONFIG.get("staff_uniform_class_id", 2))
        class_ids_cfg = YOLO_CONFIG.get(
            "detect_class_ids",
            [self.person_class_id, self.head_class_id, self.staff_uniform_class_id],
        )
        self.yolo_detect_class_ids = [int(x) for x in class_ids_cfg if x is not None]
        self.staff_uniform_iou_threshold = float(
            runtime_cfg.get("staff_uniform_iou_threshold", 0.20)
        )
        self.enable_staff_uniform = bool(runtime_cfg.get("enable_staff_uniform", True))
        
        self._motion_prev_gray: Dict[str, np.ndarray] = {}
        self._motion_last_trigger_ts: Dict[str, float] = {}
        self._motion_last_infer_ts: Dict[str, float] = {}
        self._motion_last_ratio: Dict[str, float] = {}
        self._motion_last_mean_diff: Dict[str, float] = {}
        
        # Load specific dwell times from config (Fix for duplicate counting)
        # Default to robust values if config is missing: Chair=2min, Wash=1min
        dwell_cfg = CONFIG.get("dwell_time", {})
        self.sit_time_haircut = float(dwell_cfg.get("haircut", 120.0))
        self.sit_time_wash = float(dwell_cfg.get("wash", 60.0))
        
        # Robust vacancy grace period for services
        # If a person is occluded for < 2 mins (120s), do NOT reset the session.
        # This prevents "double counting" when detection flickers or barber blocks view.
        self.service_vacant_grace_sec = 120.0

        configured_stall_timeout = float(runtime_cfg.get("camera_stall_timeout_sec", 8.0))
        # ... (lines 922-1002 unchanged) ...
        self.global_id_manager = GlobalIDManager()
        self.active_person_memory = ActivePersonMemory()
        
        # Use specific dwell times and robust grace period
        self.haircut_counter = ZoneSessionCounter("HAIRCUT", self.sit_time_haircut, self.service_vacant_grace_sec)
        self.wash_counter = ZoneSessionCounter("WASH", self.sit_time_wash, self.service_vacant_grace_sec)
        
        # Log the configuration for verification
        logger.info(
            f"Counter Config: Haircut(dwell={self.sit_time_haircut}s, grace={self.service_vacant_grace_sec}s), "
            f"Wash(dwell={self.sit_time_wash}s, grace={self.service_vacant_grace_sec}s)"
        )
        
        self.wait_realtime = 0
        self.live_customers = 0
        self.realtime_counts = {
            "chairs_total": 0,
            "washes_total": 0,
            "waits_total": 0,
            "chairs_by_zone": {},
            "washes_by_zone": {},
        }
        self.recent_haircut_reid = []
        
        # Staff identification: prefer uniform class over ReID embeddings.
        self.enable_reid = bool(staff_cfg.get("enable_reid", False))
        if self.enable_staff_uniform and self.enable_reid:
            logger.warning("enable_reid is ignored because runtime.enable_staff_uniform=True")
            self.enable_reid = False
        self.reid_encoder = None
        if self.enable_reid:
            try:
                # MPS support for ReID (ResNet50) can be flaky on some MacOS versions, fallback to CPU if needed
                reid_device = self.yolo_device if self.yolo_device != "mps" else "cpu"
                self.reid_encoder = ReIDEncoder(device=reid_device)
            except Exception as e:
                logger.error(f"Failed to init ReID encoder: {e}")
                self.enable_reid = False
        
        self.staff_gallery = None
        if self.enable_reid:
             self.staff_gallery = StaffGallery("data/staff_db.json")

        self._apply_runtime_settings(CONFIG.get("runtime_tunable", {}))
        
        # Restore dashboard state
        if os.path.exists(DASHBOARD_STATE_FILE):
             try:
                 with open(DASHBOARD_STATE_FILE, "r") as f:
                     state = json.load(f)
                     self.haircut_counter.total_count = int(state.get("haircut_count", 0))
                     self.wash_counter.total_count = int(state.get("wash_count", 0))
                     if "haircut_zone_total" in state:
                         self.haircut_counter.zone_total = state["haircut_zone_total"]
                     if "wash_zone_total" in state:
                         self.wash_counter.zone_total = state["wash_zone_total"]
                     logger.info(f"Restored dashboard state: haircuts={self.haircut_counter.total_count}")
             except Exception as e:
                 logger.error(f"Failed to restore dashboard state: {e}")

        self._setup_cameras()
        
    def _apply_runtime_settings(self, settings: Dict):
        """Apply runtime-tunable parameters from controller."""
        try:
            if "yolo_conf" in settings:
                self.yolo_conf = float(settings["yolo_conf"])
            if "yolo_iou" in settings:
                self.yolo_iou = float(settings["yolo_iou"])
            if "yolo_imgsz" in settings:
                self.yolo_imgsz = int(settings["yolo_imgsz"])
            if "yolo_mode" in settings:
                mode = str(settings["yolo_mode"]).lower().strip()
                if mode in ("predict", "track"):
                    self.yolo_mode = mode
            if "sit_min_sec" in settings:
                # Update global default, but DO NOT overwrite specific service counters
                self.sit_min_sec = float(settings["sit_min_sec"])
                # self.haircut_counter.sit_min_sec = self.sit_min_sec
                # self.wash_counter.sit_min_sec = self.sit_min_sec
            if "vacant_grace_sec" in settings:
                # Update global default, but DO NOT overwrite specific service counters
                self.vacant_grace_sec = float(settings["vacant_grace_sec"])
                # self.haircut_counter.vacant_grace_sec = self.vacant_grace_sec
                # self.wash_counter.vacant_grace_sec = self.vacant_grace_sec
            if "camera_stall_timeout_sec" in settings:
                requested_stall = float(settings["camera_stall_timeout_sec"])
                min_stall = 60.0 if self.yolo_device == "cpu" else 20.0
                self.camera_stall_timeout_sec = max(min_stall, requested_stall)
                if self.camera_stall_timeout_sec != requested_stall:
                    logger.warning(
                        f"camera_stall_timeout_sec={requested_stall:.1f}s is too low for {self.yolo_device} mode; "
                        f"clamped to {self.camera_stall_timeout_sec:.1f}s"
                    )
            if "camera_freeze_max_same_frames" in settings:
                self.camera_freeze_max_same_frames = int(settings["camera_freeze_max_same_frames"])
            if "camera_freeze_diff_threshold" in settings:
                self.camera_freeze_diff_threshold = float(settings["camera_freeze_diff_threshold"])
            if "inference_mode" in settings:
                mode = str(settings["inference_mode"]).lower().strip()
                if mode in ("always", "motion_gated"):
                    self.inference_mode = mode
            if "motion_diff_threshold" in settings:
                self.motion_diff_threshold = max(0.5, float(settings["motion_diff_threshold"]))
            if "motion_min_area_ratio" in settings:
                self.motion_min_area_ratio = min(max(float(settings["motion_min_area_ratio"]), 0.0001), 1.0)
            if "motion_recheck_sec" in settings:
                self.motion_recheck_sec = max(0.5, float(settings["motion_recheck_sec"]))
            if "motion_hold_sec" in settings:
                self.motion_hold_sec = max(0.0, float(settings["motion_hold_sec"]))
            if "motion_downscale_width" in settings:
                self.motion_downscale_width = max(64, int(settings["motion_downscale_width"]))
            if "enable_staff_uniform" in settings:
                self.enable_staff_uniform = bool(settings["enable_staff_uniform"])
            if "staff_uniform_iou_threshold" in settings:
                self.staff_uniform_iou_threshold = max(0.0, min(1.0, float(settings["staff_uniform_iou_threshold"])))
            if "customer_active_ttl_sec" in settings:
                self.customer_active_ttl_sec = float(settings["customer_active_ttl_sec"])
            if "dashboard_presence_ttl_sec" in settings:
                self.dashboard_presence_ttl_sec = float(settings["dashboard_presence_ttl_sec"])
            if "no_detection_hold_sec" in settings:
                self.no_detection_hold_sec = float(settings["no_detection_hold_sec"])
            if "resource_max_memory_percent" in settings:
                self.resource_guard.max_memory_percent = float(settings["resource_max_memory_percent"])
            if "throttle_log_interval_sec" in settings:
                self.throttle_log_interval_sec = float(settings["throttle_log_interval_sec"])
            if "zone_point_mode" in settings:
                zpm = str(settings["zone_point_mode"]).lower().strip()
                if zpm in ("foot", "center"):
                    self.zone_point_mode = zpm
            if "reid_similarity_threshold" in settings:
                self.reid_similarity_threshold = float(settings["reid_similarity_threshold"])
            if "staff_similarity_threshold" in settings:
                self.staff_similarity_threshold = float(settings["staff_similarity_threshold"])
            if "reid_active_expire_sec" in settings:
                self.reid_active_expire_sec = float(settings["reid_active_expire_sec"])
            if "reid_match_window_sec" in settings:
                self.reid_match_window_sec = float(settings["reid_match_window_sec"])
            if "cross_camera_dedupe_window_sec" in settings:
                self.cross_camera_dedupe_window_sec = float(settings["cross_camera_dedupe_window_sec"])
            if "cross_camera_dedupe_similarity" in settings:
                self.cross_camera_dedupe_similarity = float(settings["cross_camera_dedupe_similarity"])
            if "reid_merge_for_counts" in settings:
                self.reid_merge_for_counts = bool(settings["reid_merge_for_counts"])
            if "exclude_staff_from_counts" in settings:
                self.exclude_staff_from_counts = bool(settings["exclude_staff_from_counts"])
            logger.info(f"Applied runtime settings: {settings}")
        except Exception as e:
            logger.error(f"Failed applying settings: {e}")

    def _capture_event_snapshot(self, event) -> str:
        """Capture cropped snapshot for an event using latest frame and latest track bbox."""
        try:
            cam_name = event.camera
            with self.customers_lock:
                frame = self.latest_frames.get(cam_name)
                frame = None if frame is None else frame.copy()
            if frame is None:
                return ""
            h, w = frame.shape[:2]
            bbox = None
            gid = 0
            if isinstance(event.metadata, dict):
                bbox = event.metadata.get("bbox")
                gid = int(event.metadata.get("gid", 0))
            if not bbox:
                return ""
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1 = max(0, x1 - SNAPSHOT_PAD)
            y1 = max(0, y1 - SNAPSHOT_PAD)
            x2 = min(w - 1, x2 + SNAPSHOT_PAD)
            y2 = min(h - 1, y2 + SNAPSHOT_PAD)
            if x2 <= x1 or y2 <= y1:
                return ""

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return ""
            out = self._snapshot_path(event.event_type.value.upper(), cam_name, event.person_id, gid=gid)
            ok = cv2.imwrite(str(out), crop)
            return str(out) if ok else ""
        except Exception as e:
            logger.debug(f"Failed to capture event snapshot: {e}")
            return ""

    def _dedupe_haircut_events_by_reid(self, haircut_events: List[dict], merged: List[dict]) -> List[dict]:
        """Drop duplicate haircut events likely caused by the same person seen by different cameras."""
        if not haircut_events:
            return haircut_events
        now = time.time()
        pid_to_emb: Dict[int, np.ndarray] = {}
        for d in merged:
            pid = int(d.get("pid", 0))
            emb = d.get("emb")
            if pid and emb is not None:
                pid_to_emb[pid] = emb

        # expire old dedupe memory
        self.recent_haircut_reid = [
            x for x in self.recent_haircut_reid
            if (now - float(x.get("ts", 0.0))) <= self.cross_camera_dedupe_window_sec
        ]

        kept: List[dict] = []
        for ev in haircut_events:
            pid = int(ev.get("pid", 0))
            emb = pid_to_emb.get(pid)
            if emb is None:
                kept.append(ev)
                continue

            is_dup = False
            for prev in self.recent_haircut_reid:
                p_emb = prev.get("emb")
                if p_emb is None:
                    continue
                if cosine_sim(emb, p_emb) >= self.cross_camera_dedupe_similarity:
                    is_dup = True
                    break
            if is_dup:
                zone = str(ev.get("zone", ""))
                self.haircut_counter.total_count = max(0, self.haircut_counter.total_count - 1)
                if zone in self.haircut_counter.zone_total:
                    self.haircut_counter.zone_total[zone] = max(0, self.haircut_counter.zone_total[zone] - 1)
                logger.info(
                    f"Dropped duplicate haircut event by ReID dedupe: pid={pid}, zone={zone}, "
                    f"similarity>={self.cross_camera_dedupe_similarity}"
                )
                continue

            kept.append(ev)
            self.recent_haircut_reid.append({"ts": now, "emb": emb})
        return kept

    def _canonicalize_pids_cross_camera(self, merged: List[dict]) -> List[dict]:
        """Merge PIDs across cameras using ReID embeddings before counting."""
        if not merged:
            return merged

        by_pid: Dict[int, dict] = {}
        for d in merged:
            pid = int(d.get("pid", 0))
            if pid <= 0:
                continue
            prev = by_pid.get(pid)
            if prev is None or float(d.get("ts", 0.0)) > float(prev.get("ts", 0.0)):
                by_pid[pid] = d

        pids = sorted(by_pid.keys())
        if len(pids) < 2:
            return merged

        parent = {pid: pid for pid in pids}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int):
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb
            return True

        merged_groups = 0
        best_cross_sim = -1.0
        for i in range(len(pids)):
            di = by_pid[pids[i]]
            ci = str(di.get("cam", ""))
            ti = float(di.get("ts", 0.0))
            ei = di.get("emb")
            zi = str(di.get("primary_zone", "") or "")
            if ei is None:
                ei = None
            for j in range(i + 1, len(pids)):
                dj = by_pid[pids[j]]
                cj = str(dj.get("cam", ""))
                if ci == cj:
                    continue
                tj = float(dj.get("ts", 0.0))
                if abs(ti - tj) > self.cross_camera_dedupe_window_sec:
                    continue
                zj = str(dj.get("primary_zone", "") or "")

                # Edge-agent style: if both views point to the same canonical business zone
                # (e.g. CHAIR_1 / WASH), treat as same person across cameras.
                if zi and zj and zi == zj and (
                    zi.startswith(CHAIR_ZONE_PREFIX) or zi.startswith(WASH_ZONE_PREFIX)
                ):
                    if union(pids[i], pids[j]):
                        merged_groups += 1
                    continue

                # ReID-disabled fallback: if both are in the same business zone family
                # (CHAIR_* or WASH*), near-simultaneous, and from different cameras,
                # treat as the same person to avoid duplicate counting across angles.
                same_family = (
                    (zi.startswith(CHAIR_ZONE_PREFIX) and zj.startswith(CHAIR_ZONE_PREFIX))
                    or (zi.startswith(WASH_ZONE_PREFIX) and zj.startswith(WASH_ZONE_PREFIX))
                )
                if same_family and (ei is None and dj.get("emb") is None):
                    if union(pids[i], pids[j]):
                        merged_groups += 1
                    continue

                ej = dj.get("emb")
                if ej is None:
                    continue
                if ei is None:
                    continue
                sim = cosine_sim(ei, ej)
                if sim > best_cross_sim:
                    best_cross_sim = sim
                if sim >= self.cross_camera_dedupe_similarity:
                    if union(pids[i], pids[j]):
                        merged_groups += 1

        if merged_groups == 0:
            now = time.time()
            if (now - self._last_pid_canon_log_ts) >= 5.0:
                self._last_pid_canon_log_ts = now
                logger.info(
                    f"Canonicalize PID skipped: pids={len(pids)} best_sim={best_cross_sim:.3f} "
                    f"zone_or_reid_threshold={self.cross_camera_dedupe_similarity}"
                )
            return merged

        pid_map = {pid: find(pid) for pid in pids}
        out: List[dict] = []
        for d in merged:
            nd = dict(d)
            pid = int(nd.get("pid", 0))
            nd["pid"] = int(pid_map.get(pid, pid))
            out.append(nd)

        now = time.time()
        if (now - self._last_pid_canon_ok_log_ts) >= 5.0:
            self._last_pid_canon_ok_log_ts = now
            logger.info(
                f"Canonicalized cross-camera PIDs: merged_groups={merged_groups}, "
                f"sim>={self.cross_camera_dedupe_similarity}, window={self.cross_camera_dedupe_window_sec}s"
            )
        return out

    def _global_counting_loop(self):
        """Edge-agent style global counting over merged customers across cameras."""
        logger.info("Global counting loop started")
        while self.running:
            try:
                now = time.time()
                with self.customers_lock:
                    merged = []
                    for items in self.latest_customers.values():
                        for d in items:
                            ts = float(d.get("ts", 0.0))
                            if ts > 0 and (now - ts) <= self.customer_active_ttl_sec:
                                merged.append(d)

                self.active_person_memory.expire(now, self.reid_active_expire_sec)
                merged = self._canonicalize_pids_cross_camera(merged)
                active_pids = {int(d.get("pid", 0)) for d in merged if int(d.get("pid", 0)) > 0}
                self.haircut_counter.sync_active_pids(active_pids)
                self.wash_counter.sync_active_pids(active_pids)

                def select_haircut_zone(d):
                    z = d.get("primary_zone")
                    return z if z and str(z).startswith(CHAIR_ZONE_PREFIX) else None

                def select_wash_zone(d):
                    zones_hit = d.get("zones") or []
                    z = d.get("primary_zone")
                    if z and str(z).startswith(WASH_ZONE_PREFIX):
                        return str(z)
                    wash_hits = sorted([zn for zn in zones_hit if str(zn).startswith(WASH_ZONE_PREFIX)])
                    if wash_hits:
                        return wash_hits[0]
                    return None

                haircut_events = self.haircut_counter.update_and_collect_events(merged, select_haircut_zone)
                wash_events = self.wash_counter.update_and_collect_events(merged, select_wash_zone)
                haircut_events = self._dedupe_haircut_events_by_reid(haircut_events, merged)

                chairs_by_zone: Dict[str, set] = {}
                washes_by_zone: Dict[str, set] = {}
                wait_pids = set()
                for d in merged:
                    zones_hit = d.get("zones") or []
                    pid = int(d.get("pid", 0))
                    pz = d.get("primary_zone")
                    if pz and str(pz).startswith(CHAIR_ZONE_PREFIX):
                        chairs_by_zone.setdefault(str(pz), set()).add(pid)
                    if pz and str(pz).startswith(WASH_ZONE_PREFIX):
                        washes_by_zone.setdefault(str(pz), set()).add(pid)
                    in_wait = any(str(zn) in WAIT_ZONE_NAMES for zn in zones_hit)
                    if not in_wait and d.get("primary_zone") == "WAIT":
                        in_wait = True
                    if in_wait:
                        wait_pids.add(pid)
                self.wait_realtime = len(wait_pids)
                self.live_customers = len(set(d["pid"] for d in merged))
                chairs_total = len(set().union(*chairs_by_zone.values())) if chairs_by_zone else 0
                washes_total = len(set().union(*washes_by_zone.values())) if washes_by_zone else 0
                self.realtime_counts = {
                    "chairs_total": chairs_total,
                    "washes_total": washes_total,
                    "waits_total": self.wait_realtime,
                    "chairs_by_zone": {k: len(v) for k, v in chairs_by_zone.items()},
                    "washes_by_zone": {k: len(v) for k, v in washes_by_zone.items()},
                }

                # Build Event objects for submit loop
                with self.pending_lock:
                    for ev in haircut_events:
                        e = Event(
                            timestamp=datetime.now(),
                            camera=ev.get("camera", ""),
                            event_type=EventType.CHAIR,
                            person_id=int(ev.get("pid", 0)),
                            zone_name=ev.get("zone", ""),
                            dwell_seconds=float(ev.get("dwell", 0.0)),
                            metadata={"gid": int(ev.get("gid", 0)), "bbox": ev.get("bbox")},
                        )
                        self.pending_events.append(e)
                    for ev in wash_events:
                        e = Event(
                            timestamp=datetime.now(),
                            camera=ev.get("camera", ""),
                            event_type=EventType.WASH,
                            person_id=int(ev.get("pid", 0)),
                            zone_name=ev.get("zone", ""),
                            dwell_seconds=float(ev.get("dwell", 0.0)),
                            metadata={"gid": int(ev.get("gid", 0)), "bbox": ev.get("bbox")},
                        )
                        self.pending_events.append(e)

                time.sleep(0.2)
            except Exception as e:
                logger.error(f"Error in global counting loop: {e}")
                time.sleep(0.5)

    def _check_control_commands(self):
        """Handle runtime control files (reset counters)."""
        try:
            settings_file = Path(SETTINGS_OVERRIDE_FILE)
            if settings_file.exists():
                try:
                    with open(settings_file, "r", encoding="utf-8") as f:
                        settings = json.load(f) or {}
                    self._apply_runtime_settings(settings)
                finally:
                    try:
                        settings_file.unlink()
                    except Exception:
                        pass

            flag = Path(RESET_COUNTS_FLAG_FILE)
            if not flag.exists():
                return
            self.event_tracker.reset()
            self.tracker.reset()
            self.global_id_manager.reset()
            self.active_person_memory.reset()
            self.haircut_counter.reset()
            self.wash_counter.reset()
            self.wait_realtime = 0
            self.live_customers = 0
            self.realtime_counts = {
                "chairs_total": 0,
                "washes_total": 0,
                "waits_total": 0,
                "chairs_by_zone": {},
                "washes_by_zone": {},
            }
            self.recent_haircut_reid = []
            with self._fallback_track_lock:
                self._fallback_tracks = {}
            with self.pending_lock:
                self.pending_events = []
                self.all_events = []
            try:
                flag.unlink()
            except Exception:
                pass
            logger.info("Runtime counters reset by control flag")
            self._write_dashboard_state()
        except Exception as e:
            logger.error(f"Control command error: {e}")
    
    def process_frame(self, camera_name: str, frame: np.ndarray):
        """Process frame with YOLO"""
        if frame is None or frame.size == 0:
            return
        
        try:
            h, w = frame.shape[:2]
            # Guard ultralytics model forward pass with a lock to avoid concurrent
            # model mutation/runtime crashes across camera threads on CPU/MPS.
            # Use timeout so one slow inference does not block all camera loops.
            acquired = self.yolo_lock.acquire(timeout=2.0)
            if not acquired:
                now_ts = time.time()
                last_log = float(self._last_det_stats_log_ts.get(camera_name, 0.0))
                if (now_ts - last_log) >= 5.0:
                    self._last_det_stats_log_ts[camera_name] = now_ts
                    logger.warning(f"{camera_name}: YOLO busy >2s, skipping frame to keep stream alive")
                return
            try:
                detect_classes = self.yolo_detect_class_ids if self.yolo_detect_class_ids else None
                if self.yolo_mode == "track":
                    try:
                        results = self.yolo_model.track(
                            frame,
                            persist=True,
                            conf=self.yolo_conf,
                            iou=self.yolo_iou,
                            imgsz=self.yolo_imgsz,
                            classes=detect_classes,
                            tracker="bytetrack.yaml",
                            verbose=False
                        )
                    except Exception as e:
                        logger.warning(f"YOLO track failed, fallback to predict: {e}")
                        results = self.yolo_model.predict(
                            frame,
                            conf=self.yolo_conf,
                            iou=self.yolo_iou,
                            imgsz=self.yolo_imgsz,
                            classes=detect_classes,
                            verbose=False
                        )
                else:
                    # Predict mode is more stable on CPU-heavy environments.
                    results = self.yolo_model.predict(
                        frame,
                        conf=self.yolo_conf,
                        iou=self.yolo_iou,
                        imgsz=self.yolo_imgsz,
                        classes=detect_classes,
                        verbose=False
                    )
            finally:
                self.yolo_lock.release()

            now_ts = time.time()
            zones = self.zones.get(camera_name, {})
            detections: List[Detection] = []
            customers: List[Dict] = []
            raw_people = 0
            raw_heads = 0
            raw_staff_uniform = 0
            staff_filtered = 0
            emb_ready = 0

            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                ids = boxes.id
                cls_vals = boxes.cls
                if ids is None:
                    last_log = float(self._last_noid_log_ts.get(camera_name, 0.0))
                    if (now_ts - last_log) >= 5.0:
                        logger.info(f"{camera_name}: tracker ids unavailable, using fallback local ids")
                        self._last_noid_log_ts[camera_name] = now_ts
                xyxy_list = list(boxes.xyxy)
                staff_uniform_boxes: List[List[float]] = []
                for i, box in enumerate(xyxy_list):
                    cls_id = (
                        int(float(cls_vals[i].item()))
                        if cls_vals is not None and i < len(cls_vals)
                        else self.person_class_id
                    )
                    if cls_id == self.staff_uniform_class_id:
                        x1s, y1s, x2s, y2s = box.cpu().numpy().tolist()
                        raw_staff_uniform += 1
                        staff_uniform_boxes.append([float(x1s), float(y1s), float(x2s), float(y2s)])

                for i, box in enumerate(xyxy_list):
                    cls_id = (
                        int(float(cls_vals[i].item()))
                        if cls_vals is not None and i < len(cls_vals)
                        else self.person_class_id
                    )
                    if cls_id not in (self.person_class_id, self.head_class_id):
                        continue
                    x1, y1, x2, y2 = box.cpu().numpy().tolist()
                    conf = float(boxes.conf[i]) if boxes.conf is not None else 0.0
                    is_head = cls_id == self.head_class_id
                    if is_head:
                        raw_heads += 1
                    else:
                        raw_people += 1
                    x1_norm, y1_norm = x1 / w, y1 / h
                    x2_norm, y2_norm = x2 / w, y2 / h
                    detections.append(Detection(x1_norm, y1_norm, x2_norm, y2_norm, conf, int(cls_id)))

                    cx = ((x1 + x2) / 2.0) / max(w, 1)
                    cy = y2 / max(h, 1)
                    # For head-only detections, center point is more stable than foot point.
                    zone_hits = self._detect_zone_hits(
                        zones,
                        x1,
                        y1,
                        x2,
                        y2,
                        w,
                        h,
                        forced_mode=("center" if is_head else None),
                    )
                    primary_zone = self._pick_primary_business_zone(
                        zone_hits,
                        center_xy=(cx, cy),
                        zones=zones,
                    )
                    # Head-only objects should contribute to dwell mainly in service zones.
                    if is_head and primary_zone is not None:
                        pz = str(primary_zone)
                        if (not pz.startswith(CHAIR_ZONE_PREFIX)) and \
                           (not pz.startswith(WASH_ZONE_PREFIX)) and \
                           (pz not in WAIT_ZONE_NAMES):
                            primary_zone = None

                    emb = None
                    staff_id = None
                    if self.enable_staff_uniform and staff_uniform_boxes:
                        cur_box = [float(x1), float(y1), float(x2), float(y2)]
                        for sb in staff_uniform_boxes:
                            if self._bbox_iou_xyxy(cur_box, sb) >= self.staff_uniform_iou_threshold:
                                staff_id = "staff_uniform"
                                break

                    if staff_id is None and self.enable_reid and self.reid_encoder is not None:
                        x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
                        x1i = max(0, x1i)
                        y1i = max(0, y1i)
                        x2i = min(w - 1, x2i)
                        y2i = min(h - 1, y2i)
                        if (x2i - x1i) >= 50 and (y2i - y1i) >= 80:
                            crop = frame[y1i:y2i, x1i:x2i]
                            if crop is not None and crop.size > 0:
                                emb = self.reid_encoder.encode_crop_bgr(crop)
                                if emb is not None:
                                    emb_ready += 1
                                if self.staff_gallery is not None:
                                    staff_id = self.staff_gallery.match_staff(emb, self.staff_similarity_threshold)
                    if staff_id is not None and self.exclude_staff_from_counts:
                        staff_filtered += 1
                        continue

                    if ids is not None and i < len(ids):
                        local_vid = int(float(ids[i].item()))
                    else:
                        local_vid = self._assign_fallback_local_vid(camera_name, float(cx), float(cy), now_ts)

                    det = {
                        "cam": camera_name,
                        "local_vid": int(local_vid),
                        "cx": float(cx),
                        "cy": float(cy),
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "zones": zone_hits,
                        "primary_zone": primary_zone,
                        "staff_id": staff_id,
                        "det_type": "head" if is_head else "person",
                        "ts": now_ts,
                        "emb": emb,
                    }
                    gid = self.global_id_manager.assign_gid(det)
                    pid = gid
                    if self.enable_reid and self.reid_merge_for_counts:
                        pid = self.active_person_memory.resolve_pid(
                            gid,
                            emb,
                            now_ts,
                            self.reid_similarity_threshold,
                            self.reid_match_window_sec,
                        )
                    det["gid"] = int(gid)
                    det["pid"] = int(pid)
                    customers.append(det)

            self.tracker.update(camera_name, detections)
            if (raw_people + raw_heads) > 0:
                self._last_camera_nonzero[camera_name] = now_ts
            with self.customers_lock:
                if customers:
                    self.latest_customers[camera_name] = customers
                else:
                    # Keep recent detections for a short TTL to avoid flickering to 0
                    # on brief detector misses (matches dashboard "real-time but stable").
                    prev = self.latest_customers.get(camera_name, [])
                    kept = []
                    last_nonzero = float(self._last_camera_nonzero.get(camera_name, 0.0))
                    hold_stale = (last_nonzero > 0.0) and ((now_ts - last_nonzero) <= self.no_detection_hold_sec)
                    for d in prev:
                        ts = float(d.get("ts", 0.0))
                        if ts > 0 and (now_ts - ts) <= self.dashboard_presence_ttl_sec:
                            if hold_stale:
                                nd = dict(d)
                                nd["ts"] = now_ts
                                kept.append(nd)
                            else:
                                kept.append(d)
                    self.latest_customers[camera_name] = kept
                self.latest_frames[camera_name] = frame.copy()

            last_log = float(self._last_det_stats_log_ts.get(camera_name, 0.0))
            if (now_ts - last_log) >= 5.0:
                self._last_det_stats_log_ts[camera_name] = now_ts
                logger.info(
                    f"{camera_name}: raw_person={raw_people} raw_head={raw_heads} "
                    f"raw_staff_uniform={raw_staff_uniform} accepted={len(customers)} "
                    f"staff_filtered={staff_filtered} emb_ready={emb_ready} "
                    f"ids={'ok' if (results and results[0].boxes is not None and results[0].boxes.id is not None) else 'fallback'}"
                )
        
        except Exception as e:
            logger.error(f"Error processing frame from {camera_name}: {e}")
    
    def run_camera_thread(self, camera_name: str, generation: Optional[int] = None):
        """Thread for single camera with watchdog"""
        if generation is None:
            generation = int(self._camera_generation.get(camera_name, 0))
        logger.info(f"Starting camera thread: {camera_name} gen={generation}")
        frame_count = 0
        failure_count = 0

        while self.running:
            try:
                current_gen = int(self._camera_generation.get(camera_name, 0))
                if generation != current_gen:
                    logger.info(f"Stopping stale camera thread: {camera_name} gen={generation} current={current_gen}")
                    return

                with self._camera_lock:
                    cam_stream = self.cameras.get(camera_name)
                if cam_stream is None:
                    time.sleep(0.5)
                    continue

                if cam_stream.reconnect_requested or (not cam_stream.is_connected):
                    try:
                        cam_stream.connect()
                    except Exception:
                        logger.exception(f"{camera_name}: reconnect request failed")
                    finally:
                        cam_stream.reconnect_requested = False

                # Check throttle status
                self._check_control_commands()
                if self.resource_guard.should_throttle():
                    reason = self.resource_guard.get_throttle_reason()
                    now_ts = time.time()
                    last_log = float(self._last_throttle_log_ts.get(camera_name, 0.0))
                    if (now_ts - last_log) >= self.throttle_log_interval_sec:
                        self._last_throttle_log_ts[camera_name] = now_ts
                        logger.warning(f"Throttling {camera_name}: {reason}")

                    # When throttling due to high resource usage we still want the
                    # dashboard to reflect people who are likely still present.
                    # Refresh recent latest_customers timestamps (within a grace
                    # window) so the controller doesn't drop counts to zero during
                    # brief throttling periods.
                    try:
                        with self.customers_lock:
                            grace = max(
                                1.0,
                                float(max(self.customer_active_ttl_sec, self.dashboard_presence_ttl_sec)),
                            )
                            for cam, prev in list(self.latest_customers.items()):
                                refreshed: List[Dict] = []
                                for d in prev:
                                    ts = float(d.get("ts", 0.0))
                                    if ts > 0 and (now_ts - ts) <= (grace * 2):
                                        nd = dict(d)
                                        nd["ts"] = now_ts
                                        refreshed.append(nd)
                                if refreshed:
                                    self.latest_customers[cam] = refreshed
                    except Exception:
                        pass

                    time.sleep(1.0)
                    continue

                frame = cam_stream.read_frame()

                if frame is None:
                    # Record failure for watchdog
                    self.watchdog.mark_frame_failed(camera_name, "Frame read failed")
                    failure_count += 1
                    # After several consecutive failures try reconnecting
                    if failure_count >= 3:
                        logger.info(f"{camera_name}: repeated frame failures, attempting reconnect")
                        try:
                            cam_stream.connect()
                        except Exception:
                            logger.exception(f"Reconnect attempt failed for {camera_name}")
                        failure_count = 0
                    time.sleep(0.5)
                    continue

                # Record success for watchdog
                self.watchdog.mark_frame_received(camera_name)

                # Record frame for resource guard
                self.resource_guard.record_frame(camera_name)

                # successful read, reset failure counter
                failure_count = 0
                now_ts = time.time()
                self._last_frame_ok_ts[camera_name] = now_ts
                good_frames = int(self._camera_good_frame_count.get(camera_name, 0)) + 1
                self._camera_good_frame_count[camera_name] = good_frames
                # Camera is healthy again, clear reconnect backoff.
                self._restart_backoff_sec[camera_name] = 0.0
                self._restart_backoff_until[camera_name] = 0.0
                if good_frames >= int(max(30, self.target_fps * 20)):
                    if self._stall_restart_count.get(camera_name, 0) != 0:
                        logger.info(f"{camera_name}: stable frame reads restored, resetting stall counter")
                    self._stall_restart_count[camera_name] = 0
                    base_policy = {
                        "prefer_tcp": bool(RUNTIME_CONFIG.get("rtsp_prefer_tcp", False)),
                        "force_ffmpeg": bool(RUNTIME_CONFIG.get("rtsp_force_ffmpeg", False)),
                        "allow_udp_fallback": bool(RUNTIME_CONFIG.get("rtsp_allow_udp_fallback", False)),
                        "rotate_backends": bool(RUNTIME_CONFIG.get("rtsp_rotate_backends", False)),
                    }
                    if self._camera_connect_policy.get(camera_name) != base_policy:
                        self._camera_connect_policy[camera_name] = dict(base_policy)
                        logger.info(f"{camera_name}: restoring base connect policy after stable read period")

                # Keep reading frames continuously to prevent RTSP/FFmpeg buffer
                # buildup; only run heavy inference at target_fps cadence.
                min_process_interval = 1.0 / max(float(self.target_fps), 1.0)
                last_proc = float(self._last_process_ts.get(camera_name, 0.0))
                
                # OPTIMIZATION: Also skip UI frame updates if they are happening too fast.
                # Cap UI updates at 10 FPS to save memory bandwidth (copies).
                min_ui_interval = 0.1 
                
                if (now_ts - last_proc) < min_process_interval:
                    # Skip inference
                    
                    # Check if we should update UI frame
                    if not hasattr(self, "_last_ui_update_ts"):
                         self._last_ui_update_ts = {}
                    
                    if (now_ts - float(self._last_ui_update_ts.get(camera_name, 0.0))) >= min_ui_interval:
                        with self.customers_lock:
                            self.latest_frames[camera_name] = frame.copy()
                        self._last_ui_update_ts[camera_name] = now_ts

                    # short yield, but do not sleep by target_fps here
                    time.sleep(0.002)
                    continue
                self._last_process_ts[camera_name] = now_ts

                should_infer, infer_reason = self._should_run_inference(camera_name, frame, now_ts)
                if should_infer:
                    self.process_frame(camera_name, frame)
                    self._motion_last_infer_ts[camera_name] = now_ts
                else:
                    # Skip heavy YOLO pass during motion-gated idle windows.
                    # Keep recent customers alive briefly to avoid dashboard flicker.
                    with self.customers_lock:
                        prev = self.latest_customers.get(camera_name, [])
                        refreshed: List[Dict] = []
                        keep_sec = max(self.motion_recheck_sec * 2.0, self.customer_active_ttl_sec)
                        for d in prev:
                            ts = float(d.get("ts", 0.0))
                            if ts > 0 and (now_ts - ts) <= keep_sec:
                                nd = dict(d)
                                nd["ts"] = now_ts
                                refreshed.append(nd)
                        self.latest_customers[camera_name] = refreshed
                        self.latest_frames[camera_name] = frame.copy()
                    last_log = float(self._last_det_stats_log_ts.get(camera_name, 0.0))
                    if (now_ts - last_log) >= 5.0:
                        self._last_det_stats_log_ts[camera_name] = now_ts
                        logger.info(f"{camera_name}: inference skipped ({infer_reason})")

                # Per-camera heartbeat log to show ongoing detection activity
                try:
                    now = time.time()
                    last_hb = float(self._last_camera_heartbeat.get(camera_name, 0.0))
                    if (now - last_hb) >= max(1.0, float(self.camera_heartbeat_sec)):
                        det_count = 0
                        with self.customers_lock:
                            det_count = len(self.latest_customers.get(camera_name, []) or [])
                        logger.info(
                            f"CameraHeartbeat: {camera_name} connected={getattr(cam_stream, 'is_connected', False)} "
                            f"fps={getattr(cam_stream, 'fps', 0):.1f} detections={det_count}"
                        )
                        # Track last time camera had non-zero detections
                        if det_count > 0:
                            self._last_camera_nonzero[camera_name] = now
                        else:
                            last_nonzero = float(self._last_camera_nonzero.get(camera_name, 0.0))
                            if last_nonzero > 0 and (now - last_nonzero) >= float(self.camera_no_detection_warn_sec):
                                logger.warning(f"Camera {camera_name} has had no detections for {now-last_nonzero:.1f}s")
                        self._last_camera_heartbeat[camera_name] = now
                except Exception:
                    pass
                frame_count += 1

                if frame_count % 100 == 0:
                    logger.debug(f"{camera_name}: {frame_count} frames")

                # Broadcast status every 2 seconds
                current_time = time.time()
                if current_time - self.last_status_broadcast > 2.0:
                    self._broadcast_status()
                    self.last_health_broadcast = current_time

                # Keep capture loop responsive; inference cadence is controlled above.
                time.sleep(0.002)
            except Exception as e:
                logger.exception(f"Unexpected error in camera thread {camera_name}: {e}")
                time.sleep(1.0)

    
    def submit_events_loop(self):
        """Background thread to submit events"""
        logger.info("Event submission thread started")
        
        while self.running:
            try:
                self._check_control_commands()
                with self.pending_lock:
                    events = self.pending_events
                    self.pending_events = []
                
                if events:
                    event_dicts = []
                    for e in events:
                        snap = self._capture_event_snapshot(e)
                        if snap:
                            e.metadata["snapshot_path"] = snap
                            item = {
                                "timestamp": datetime.now().isoformat(),
                                "path": snap,
                                "camera": e.camera,
                                "gid": int(e.metadata.get("gid", 0)) if isinstance(e.metadata, dict) else 0,
                                "pid": int(e.person_id),
                                "event_type": e.event_type.value,
                            }
                            self.recent_snapshots.append(item)
                            if len(self.recent_snapshots) > 200:
                                self.recent_snapshots = self.recent_snapshots[-200:]
                        event_dicts.append(e.to_dict())

                    if self.supabase_sync:
                        for event_dict in event_dicts:
                            self.supabase_sync.add_event(event_dict)
                        logger.info(f"Queued {len(event_dicts)} events for submission")
                    else:
                        logger.info(f"Processed {len(event_dicts)} events (Supabase disabled)")

                    # Broadcast individual events even if Supabase is disabled.
                    for event_dict in event_dicts:
                        self._broadcast_event(event_dict)
                    self.all_events.extend(events)
                
                # Generate daily report (once per hour)
                now = datetime.now()
                if (now - self.last_report_time).total_seconds() > 3600:
                    self.report_gen.generate_daily_report(self.all_events)
                    self.last_report_time = now
                
                # Log and broadcast summary
                status_now = self.get_status()
                summary = {
                    "active_people": int(status_now.get("active_tracks", self.live_customers)),
                    "haircuts": self.haircut_counter.total_count,
                    "washes": self.wash_counter.total_count,
                    "waits": self.wait_realtime,
                    "total_events": len(self.all_events),
                }
                logger.info(f"Summary: {summary}")
                
                current_time = time.time()
                if current_time - self.last_summary_broadcast > 5.0:
                    self._broadcast_summary(summary)
                    self.last_summary_broadcast = current_time
                self._write_dashboard_state()
                
                # Explicit GC to prevent memory creep
                gc.collect()
                time.sleep(10)
            
            except Exception as e:
                logger.error(f"Error in event submission loop: {e}")
                time.sleep(5)
    
    def _load_yolo(self) -> Tuple[YOLO, str]:
        """Load YOLO model and detect best device."""
        model_name = YOLO_CONFIG.get("model", "yolov8m.pt")
        model_path = Path(YOLO_MODEL_PATH)
        
        model_to_load = ""
        if model_path.exists():
            model_to_load = str(model_path)
        else:
            logger.warning(f"YOLO model not found at {model_path}. Attempting to download '{model_name}'...")
            model_to_load = model_name
            
        device = str(YOLO_CONFIG.get("device", "auto")).lower().strip()
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif (
                platform.system() != "Darwin"
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                device = "mps"
            else:
                device = "cpu"
                
        logger.info(f"Loading YOLO model: {model_to_load} on {device}")
        
        try:
            model = YOLO(model_to_load)
        except Exception as e:
            logger.error(f"Failed to load/download YOLO model '{model_to_load}': {e}")
            raise e
            
        try:
            model.to(device)
        except Exception as e:
            if device != "cpu":
                logger.warning(f"YOLO device '{device}' failed ({e}), falling back to cpu")
                model.to("cpu")
                device = "cpu"
            else:
                raise e
                
        return model, device

    def _snapshot_path(self, event_type: str, camera: str, pid: int, gid: int = 0) -> Path:
        """Generate snapshot path."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_cam = "".join(c for c in camera if c.isalnum() or c in ('_', '-'))
        filename = f"{ts}_{event_type}_{safe_cam}_P{pid}_G{gid}.jpg"
        return Path(SNAPSHOT_DIR) / filename

    def _broadcast_status(self):
        """Broadcast heartbeat status to dashboard."""
        try:
            now = time.time()
            if (now - self.last_status_broadcast) < 1.0:
                return

            status = {
                "timestamp": now,
                "uptime": 0,  # calculated by dashboard
                "cameras": {},
                "people_count": self.live_customers,
                "waiting_count": self.wait_realtime,
                "haircut_count": self.realtime_counts.get("chairs_total", 0),
                "wash_count": self.realtime_counts.get("washes_total", 0),
                "cpu_usage": 0, # placeholder
                "memory_usage": 0, # placeholder
            }
            
            for name, cam in self.cameras.items():
                status["cameras"][name] = {
                    "connected": cam.is_connected,
                    "fps": getattr(cam, "fps", 0),
                    "last_frame": self._last_frame_ok_ts.get(name, 0),
                }

            self.broadcaster.broadcast("status", status)
            self.last_status_broadcast = now
            
            # Send Summary for charts
            if (now - self.last_summary_broadcast) >= 5.0:
                summary = {
                    "active_people": self.live_customers,
                    "haircuts": self.haircut_counter.total_count,
                    "washes": self.wash_counter.total_count,
                    "waits": self.wait_realtime,
                    "total_events": len(self.all_events),
                }
                self.broadcaster.broadcast("summary", summary)
                self.last_summary_broadcast = now
                logger.info(f"Summary: {summary}")

        except Exception as e:
            logger.error(f"Broadcast error: {e}")

    def _broadcast_event(self, event_dict: Dict):
        """Broadcast new event to dashboard."""
        try:
            self.broadcaster.broadcast("new_event", event_dict)
        except Exception as e:
            logger.error(f"Event broadcast error: {e}")

    def _write_dashboard_state(self, last_event: Optional[Dict] = None):
        """Persist state for cross-process dashboard polling."""
        try:
            status = self.get_status()
            data = {
                "timestamp": time.time(),
                "status": status,
                "summary": status.get("summary", {}),
            }
            if last_event is not None:
                data["last_event"] = last_event

            state_path = Path(DASHBOARD_STATE_FILE)
            state_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                prefix="dashboard_state_",
                suffix=".tmp",
                dir=str(state_path.parent),
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)
                os.replace(tmp_path, str(state_path))
            finally:
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Failed to write dashboard state: {e}")

    def _setup_cameras(self):
        """Initialize cameras from config."""
        for name, cfg in CAMERAS_CONFIG.items():
            # Handle both string (legacy) and dict (new) config formats
            if isinstance(cfg, str):
                url = cfg
                enabled = True
            elif isinstance(cfg, dict):
                url = cfg.get("rtsp_url", "")
                enabled = bool(cfg.get("enabled", True))
            else:
                logger.warning(f"Invalid config for camera {name}: {cfg}")
                continue

            if not enabled:
                logger.info(f"Skipping disabled camera: {name}")
                continue

            logger.info(f"Adding camera: {name}")
            self._camera_generation[name] = 1
            self.cameras[name] = CameraStream(name, url)
            self.zones[name] = load_zones(name)

    def _on_camera_offline(self, camera_name: str):
        logger.warning(f"Watchdog: Camera {camera_name} is OFFLINE")
        self.broadcaster.broadcast("camera_status", {"camera": camera_name, "status": "offline"})

    def _on_camera_online(self, camera_name: str):
        logger.info(f"Watchdog: Camera {camera_name} is ONLINE")
        self.broadcaster.broadcast("camera_status", {"camera": camera_name, "status": "online"})

    def _on_health_check_failed(self, check_name: str, details: str):
        logger.error(f"Health Check Failed: {check_name} - {details}")
        
    def _detect_zone_hits(
        self,
        zones: Dict[str, List[Tuple[float, float]]],
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        w: int,
        h: int,
        forced_mode: Optional[str] = None,
    ) -> List[str]:
        hits = []
        if not zones:
            return hits
        
        # Check foot point
        foot_x = (x1 + x2) / 2.0 / w
        foot_y = y2 / h
        foot_pt = (foot_x, foot_y)

        # Check center point
        center_x = (x1 + x2) / 2.0 / w
        center_y = (y1 + y2) / 2.0 / h
        center_pt = (center_x, center_y)

        for name, poly in zones.items():
            # Use foot point for better floor-plan accuracy
            mode = forced_mode if forced_mode in ("foot", "center") else self.zone_point_mode
            target_pt = foot_pt if mode == "foot" else center_pt
            if point_in_polygon(target_pt, poly):
                hits.append(name)
        return hits

    @staticmethod
    def _bbox_iou_xyxy(a: List[float], b: List[float]) -> float:
        """IoU between two XYXY pixel bboxes."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter
        if denom <= 0.0:
            return 0.0
        return inter / denom

    def _pick_primary_business_zone(self, hits: List[str], center_xy: Tuple[float, float], zones: Dict) -> Optional[str]:
        # Priority: CHAIR > WASH > WAIT > others
        if not hits:
            return None
        
        chair = [z for z in hits if z.startswith(CHAIR_ZONE_PREFIX)]
        if chair:
            return chair[0]  # Just pick first if multiple
            
        wash = [z for z in hits if z.startswith(WASH_ZONE_PREFIX)]
        if wash:
            return wash[0]

        wait = [z for z in hits if z in WAIT_ZONE_NAMES]
        if wait:
            return wait[0]
            
        return hits[0]

    def _assign_fallback_local_vid(self, camera: str, cx: float, cy: float, now: float) -> int:
        """Assign temporary ID when YOLO tracking fails."""
        with self._fallback_track_lock:
            tracks = self._fallback_tracks.setdefault(camera, {})
            best_id = -1
            min_dist = 0.15 # Max distance match
            
            # Simple distance matching
            dead = []
            for tid, info in tracks.items():
                if (now - info["ts"]) > 2.0:
                    dead.append(tid)
                    continue
                dist = ((cx - info["cx"])**2 + (cy - info["cy"])**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    best_id = tid
            
            for tid in dead:
                tracks.pop(tid, None)

            if best_id != -1:
                tracks[best_id] = {"cx": cx, "cy": cy, "ts": now}
                return best_id
            
            new_id = self._next_fallback_tid
            self._next_fallback_tid += 1
            tracks[new_id] = {"cx": cx, "cy": cy, "ts": now}
            return new_id

    def _should_run_inference(self, camera_name: str, frame: np.ndarray, now_ts: float) -> Tuple[bool, str]:
        """Motion gating logic."""
        if self.inference_mode == "always":
            return True, "always"

        # Initialize motion data for camera
        if camera_name not in self._motion_prev_gray:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (self.motion_downscale_width, int(self.motion_downscale_width * frame.shape[0] / frame.shape[1])))
            self._motion_prev_gray[camera_name] = small
            self._motion_last_trigger_ts[camera_name] = now_ts
            return True, "init"

        # Check hold time
        last_trig = self._motion_last_trigger_ts.get(camera_name, 0.0)
        if (now_ts - last_trig) < self.motion_hold_sec:
            return True, "hold"

        # Check recheck interval
        last_infer = self._motion_last_infer_ts.get(camera_name, 0.0)
        if (now_ts - last_infer) < self.motion_recheck_sec:
            return False, "interval"

        # Calculate motion
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (self._motion_prev_gray[camera_name].shape[1], self._motion_prev_gray[camera_name].shape[0]))
        
        diff = cv2.absdiff(small, self._motion_prev_gray[camera_name])
        thresh = cv2.threshold(diff, self.motion_diff_threshold, 255, cv2.THRESH_BINARY)[1]
        
        motion_ratio = np.sum(thresh > 0) / thresh.size
        self._motion_last_ratio[camera_name] = motion_ratio
        
        self._motion_prev_gray[camera_name] = small
        
        if motion_ratio > self.motion_min_area_ratio:
            self._motion_last_trigger_ts[camera_name] = now_ts
            return True, f"motion {motion_ratio:.4f}"
            
        return False, f"idle {motion_ratio:.4f}"
        
    def start(self):
        """Start service with reliability features"""
        logger.info("Starting runtime service...")
        self.running = True
        self._write_dashboard_state()
        
        # Start reliability components
        self.resource_guard.start()
        self.health_checker.start()
        
        # Start camera threads
        for camera_name in self.cameras:
            gen = int(self._camera_generation.get(camera_name, 0))
            self._spawn_camera_worker(camera_name, gen)
        
        # Start event submission thread
        counting_thread = threading.Thread(target=self._global_counting_loop)
        counting_thread.daemon = True
        counting_thread.start()
        self.threads.append(counting_thread)

        # Start event submission thread
        event_thread = threading.Thread(target=self.submit_events_loop)
        event_thread.daemon = True
        event_thread.start()
        self.threads.append(event_thread)
        
        # Start watchdog thread
        watchdog_thread = threading.Thread(target=self._watchdog_loop)
        watchdog_thread.daemon = True
        watchdog_thread.start()
        self.threads.append(watchdog_thread)
        
        # Start Supabase sync
        if self.supabase_sync:
            self.supabase_sync.start()
        
        logger.info(f"Started {len(self.threads)} threads with reliability features")
    
    def stop(self):
        """Stop service"""
        logger.info("Stopping runtime service...")
        self.running = False
        self._write_dashboard_state()
        
        # Stop reliability components
        self.resource_guard.stop()
        self.health_checker.stop()
        
        # Stop Supabase sync
        if self.supabase_sync:
            self.supabase_sync.stop()
        
        # Wait for threads
        for thread in self.threads:
            thread.join(timeout=5)
        
        # Disconnect cameras
        for cam_stream in self.cameras.values():
            cam_stream.disconnect()

        self._write_dashboard_state()
        
        logger.info("Runtime service stopped")
    
    def get_status(self) -> Dict:
        """Get service status"""
        cameras_status = {}
        for cam_name, cam_cfg in CAMERAS_CONFIG.items():
            cam_stream = self.cameras.get(cam_name)
            cameras_status[cam_name] = {
                "enabled": bool(cam_cfg.get("enabled", True)),
                "connected": bool(cam_stream.is_connected) if cam_stream else False,
                "fps": float(cam_stream.fps) if cam_stream else 0.0,
            }

        now = time.time()
        with self.customers_lock:
            latest_customers = {}
            for cam in self.cameras.keys():
                dets = self.latest_customers.get(cam, [])
                filtered = []
                for d in dets:
                    ts = float(d.get("ts", 0.0))
                    if ts <= 0 or (now - ts) > self.dashboard_presence_ttl_sec:
                        continue
                    filtered.append({
                        "pid": int(d.get("pid", 0)),
                        "gid": int(d.get("gid", 0)),
                        "bbox": [float(v) for v in d.get("bbox", [])],
                        "primary_zone": d.get("primary_zone"),
                    })
                latest_customers[cam] = filtered

        camera_people = {cam: len(latest_customers.get(cam, [])) for cam in self.cameras.keys()}
        camera_gids = {
            cam: sorted(list({d["gid"] for d in latest_customers.get(cam, []) if d.get("gid")}))
            for cam in self.cameras.keys()
        }
        gid_to_cams: Dict[int, List[str]] = {}
        for cam, gids in camera_gids.items():
            for gid in gids:
                gid_to_cams.setdefault(gid, []).append(cam)
        same_person_multi_cam = {
            str(gid): cams for gid, cams in gid_to_cams.items() if len(cams) > 1
        }
        motion_state = {
            cam: {
                "last_ratio": float(self._motion_last_ratio.get(cam, 0.0)),
                "last_mean_diff": float(self._motion_last_mean_diff.get(cam, 0.0)),
                "last_infer_ago_sec": max(0.0, now - float(self._motion_last_infer_ts.get(cam, 0.0)))
                if self._motion_last_infer_ts.get(cam, 0.0) > 0
                else -1.0,
            }
            for cam in self.cameras.keys()
        }
        
        # Prefer the computed global live_customers but fall back to per-camera
        # counts when the global counter is zero. This avoids brief resets to
        # 0 in the controller when small timing gaps occur between threads.
        fallback_active = sum(camera_people.values())
        active_tracks_val = self.live_customers if (self.live_customers and self.live_customers > 0) else int(fallback_active)

        return {
            "running": self.running,
            "branch": CONFIG.get("branch_code"),
            "cameras": cameras_status,
            "active_tracks": active_tracks_val,
            "events_queued": self.supabase_sync.get_queue_size() if self.supabase_sync else 0,
            "camera_people": camera_people,
            "camera_gids": camera_gids,
            "same_person_multi_cam": same_person_multi_cam,
            "motion": motion_state,
            "latest_customers": latest_customers,
            "realtime_counts": self.realtime_counts,
            "recent_snapshots": self.recent_snapshots[-50:],
            "tuning": {
                "yolo_conf": self.yolo_conf,
                "yolo_iou": self.yolo_iou,
                "yolo_imgsz": self.yolo_imgsz,
                "yolo_mode": self.yolo_mode,
                "person_class_id": self.person_class_id,
                "head_class_id": self.head_class_id,
                "staff_uniform_class_id": self.staff_uniform_class_id,
                "yolo_detect_class_ids": self.yolo_detect_class_ids,
                "inference_mode": self.inference_mode,
                "enable_staff_uniform": self.enable_staff_uniform,
                "staff_uniform_iou_threshold": self.staff_uniform_iou_threshold,
                "motion_diff_threshold": self.motion_diff_threshold,
                "motion_min_area_ratio": self.motion_min_area_ratio,
                "motion_recheck_sec": self.motion_recheck_sec,
                "motion_hold_sec": self.motion_hold_sec,
                "motion_downscale_width": self.motion_downscale_width,
                "sit_min_sec": self.sit_min_sec,
                "vacant_grace_sec": self.vacant_grace_sec,
                "camera_stall_timeout_sec": self.camera_stall_timeout_sec,
                "camera_freeze_max_same_frames": self.camera_freeze_max_same_frames,
                "camera_freeze_diff_threshold": self.camera_freeze_diff_threshold,
                "customer_active_ttl_sec": self.customer_active_ttl_sec,
                "dashboard_presence_ttl_sec": self.dashboard_presence_ttl_sec,
                "no_detection_hold_sec": self.no_detection_hold_sec,
                "resource_max_memory_percent": self.resource_guard.max_memory_percent,
                "zone_point_mode": self.zone_point_mode,
                "reid_similarity_threshold": self.reid_similarity_threshold,
                "staff_similarity_threshold": self.staff_similarity_threshold,
                "reid_active_expire_sec": self.reid_active_expire_sec,
                "reid_match_window_sec": self.reid_match_window_sec,
                "cross_camera_dedupe_window_sec": self.cross_camera_dedupe_window_sec,
                "cross_camera_dedupe_similarity": self.cross_camera_dedupe_similarity,
                "reid_merge_for_counts": self.reid_merge_for_counts,
                "exclude_staff_from_counts": self.exclude_staff_from_counts,
            },
            "summary": {
                "active_people": active_tracks_val,
                "haircuts": self.haircut_counter.total_count,
                "washes": self.wash_counter.total_count,
                "waits": self.wait_realtime,
                "haircut_confirmed": self.haircut_counter.total_count,
                "total_events": len(self.all_events),
            },
        }
    
    def _broadcast_status(self):
        """Broadcast status to dashboard"""
        try:
            status = self.get_status()
            self._write_dashboard_state()
            if self.broadcaster and self.broadcaster.get_subscriber_count() > 0:
                self.broadcaster.broadcast_status(status)
        except Exception as e:
            logger.debug(f"Error broadcasting status: {e}")
    
    def _broadcast_event(self, event: Dict):
        """Broadcast individual event to dashboard"""
        try:
            self._write_dashboard_state(last_event=event)
            if self.broadcaster and self.broadcaster.get_subscriber_count() > 0:
                self.broadcaster.broadcast_event(event)
        except Exception as e:
            logger.debug(f"Error broadcasting event: {e}")
    
    def _broadcast_summary(self, summary: Dict):
        """Broadcast summary to dashboard"""
        try:
            self._write_dashboard_state()
            if self.broadcaster and self.broadcaster.get_subscriber_count() > 0:
                self.broadcaster.broadcast_summary(summary)
        except Exception as e:
            logger.debug(f"Error broadcasting summary: {e}")
    
    def _watchdog_loop(self):
        """Background thread for RTSP watchdog and camera reconnection"""
        logger.info("Watchdog loop started")
        
        while self.running:
            try:
                with self._camera_lock:
                    camera_items = list(self.cameras.items())
                for camera_name, cam_stream in camera_items:
                    # Detect stalled camera loop even when stream still reports connected.
                    last_ok = float(self._last_frame_ok_ts.get(camera_name, 0.0))
                    camera_connected = bool(cam_stream.is_connected)
                    if camera_connected and last_ok > 0.0:
                        idle_sec = time.time() - last_ok
                        if idle_sec >= self.camera_stall_timeout_sec:
                            logger.warning(
                                f"Watchdog: {camera_name} frame stalled for {idle_sec:.1f}s, marking degraded and scheduling reconnect"
                            )
                            cam_stream.is_connected = False
                            cam_stream.reconnect_requested = True
                            self.watchdog.mark_frame_failed(
                                camera_name, f"Frame stall > {self.camera_stall_timeout_sec:.1f}s"
                            )
                            # If camera thread is blocked inside cap.read(), it may
                            # never reach reconnect branch. Force-close capture from
                            # watchdog side to break the blocking read.
                            now_ts = time.time()
                            last_force = float(self._last_forced_recover_ts.get(camera_name, 0.0))
                            backoff_until = float(self._restart_backoff_until.get(camera_name, 0.0))
                            if now_ts < backoff_until:
                                continue
                            if (now_ts - last_force) >= 10.0:
                                self._last_forced_recover_ts[camera_name] = now_ts
                                try:
                                    logger.warning(f"Watchdog: forcing stream reset for {camera_name} to unblock reader")
                                    self._restart_camera_worker(camera_name, reason="stall_recovery")
                                except Exception as e:
                                    logger.error(f"Watchdog: force-reset failed for {camera_name}: {e}")
                    
                    # Reconnect is handled in camera thread to avoid cross-thread
                    # VideoCapture release/read races on OpenCV/FFmpeg.
                
                time.sleep(5)  # Check every 5 seconds
            
            except Exception as e:
                logger.error(f"Error in watchdog loop: {e}")
                time.sleep(5)
    
    def _on_camera_offline(self, camera_name: str):
        """Callback when camera goes offline"""
        logger.warning(f"Camera {camera_name} is offline - attempting reconnection")
        # Broadcast health status
        self._broadcast_health_status()
    
    def _on_camera_online(self, camera_name: str):
        """Callback when camera comes back online"""
        logger.info(f"Camera {camera_name} is back online")
        # Broadcast health status
        self._broadcast_health_status()
    
    def _on_health_check_failed(self, check_name: str, message: str):
        """Callback when health check fails"""
        logger.error(f"Health check '{check_name}' failed: {message}")
        # Broadcast health status
        self._broadcast_health_status()
    
    def _broadcast_health_status(self):
        """Broadcast health status to dashboard"""
        try:
            current_time = time.time()
            if current_time - self.last_health_broadcast > 5.0:
                health = {
                    "watchdog": self.watchdog.get_status_summary(),
                    "resources": {
                        "fps": self.resource_guard.get_metrics().fps,
                        "memory_percent": self.resource_guard.get_metrics().memory_percent,
                        "cpu_percent": self.resource_guard.get_metrics().cpu_percent,
                    },
                    "health_checks": self.health_checker.get_status(),
                }
                
                if self.broadcaster:
                    # Broadcast as status update with health info
                    status = self.get_status()
                    status["health"] = health
                    self.broadcaster.broadcast_status(status)
                    self._write_dashboard_state()
                
                self.last_health_broadcast = current_time
        except Exception as e:
            logger.debug(f"Error broadcasting health status: {e}")



# =========================
# MAIN
# =========================

if __name__ == "__main__":
    service = RuntimeService()
    
    try:
        service.start()
        
        while service.running:
            status = service.get_status()
            logger.info(f"Status: {status}")
            time.sleep(60)
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    finally:
        service.stop()
