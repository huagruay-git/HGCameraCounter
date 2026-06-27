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
import queue
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Set
from collections import deque
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
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG = Config(str(PROJECT_ROOT / "data" / "config" / "config.yaml"))
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

def resolve_model_path(model_name: str, fallback_name: str) -> str:
    """Resolve model path from config while preserving Ultralytics filename fallback."""
    raw_name = str(model_name or fallback_name or "").strip()
    if not raw_name:
        raw_name = fallback_name
    candidate = Path(raw_name)
    if candidate.is_absolute():
        return str(candidate)

    search_roots = [
        PROJECT_ROOT,
        PROJECT_ROOT / PATHS_CONFIG.get("models", "models"),
    ]
    for root in search_roots:
        resolved = (root / raw_name).resolve()
        if resolved.exists():
            return str(resolved)

    if os.path.sep in raw_name or "/" in raw_name:
        return str((PROJECT_ROOT / raw_name).resolve())

    # Return the plain filename for built-in model auto-download if needed.
    return raw_name


def _normalize_label_name(name: str) -> str:
    s = str(name or "").strip().lower()
    return "".join(ch if ch.isalnum() else "_" for ch in s)


def _model_names_dict(model: YOLO) -> Dict[int, str]:
    names = getattr(model, "names", None)
    if names is None and hasattr(model, "model"):
        names = getattr(model.model, "names", None)
    out: Dict[int, str] = {}
    if isinstance(names, dict):
        for k, v in names.items():
            try:
                out[int(k)] = str(v)
            except Exception:
                continue
    elif isinstance(names, (list, tuple)):
        for i, v in enumerate(names):
            out[int(i)] = str(v)
    return out


def _label_contains_tokens(name: str, tokens: List[str]) -> bool:
    normalized = _normalize_label_name(name)
    return all(tok in normalized for tok in tokens)


def _normalize_label_set(raw: object) -> Set[str]:
    out: Set[str] = set()
    if isinstance(raw, str):
        candidates = raw.replace(";", ",").split(",")
    elif isinstance(raw, (list, tuple, set)):
        candidates = list(raw)
    else:
        candidates = []
    for item in candidates:
        token = _normalize_label_name(str(item or ""))
        if token:
            out.add(token)
    return out


YOLO_DISCOVERY_MODEL_PATH = resolve_model_path(
    YOLO_CONFIG.get("discovery_model", "best.pt"),
    "best.pt",
)
YOLO_VERIFY_MODEL_PATH = resolve_model_path(
    YOLO_CONFIG.get("worker_model", YOLO_CONFIG.get("model", "best.pt")),
    "best.pt",
)
TIER2_CLIP_DURATION_SEC = float(RUNTIME_CONFIG.get("tier2_clip_duration_sec", 4.0))

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

    def match_staff_with_score(self, emb: Optional[np.ndarray]) -> Tuple[Optional[str], float]:
        if emb is None or not self.staff_embs:
            return None, -1.0
        best_id, best = None, -1.0
        for pid, e in self.staff_embs:
            if e is None or e.shape != emb.shape:
                continue
            s = cosine_sim(emb, e)
            if s > best:
                best = s
                best_id = pid
        return best_id, best

    def match_staff(self, emb: Optional[np.ndarray], thr: float) -> Optional[str]:
        best_id, best = self.match_staff_with_score(emb)
        return best_id if best_id is not None and best >= thr else None


class RoleGallery:
    """Embedding gallery for role classification: barber_uniform vs customer."""

    def __init__(self, db_path: str, max_embeddings_per_role: int = 512):
        self.db_path = db_path
        self.max_embeddings_per_role = max(1, int(max_embeddings_per_role))
        self.role_mats: Dict[str, np.ndarray] = {}
        self.role_avgs: Dict[str, np.ndarray] = {}
        self.role_counts: Dict[str, int] = {}
        self.total_embeddings = 0
        self._load()

    @staticmethod
    def _normalize_role(role_name: str) -> str:
        txt = str(role_name or "").strip().lower()
        # Keep wash-customer as a distinct role so future counting logic
        # can use it separately from generic customer.
        if ("wash" in txt or "shampoo" in txt) and ("customer" in txt or "client" in txt):
            return "wash_customer"
        if txt in {"wash_customer", "customer_wash", "shampoo_customer"}:
            return "wash_customer"
        if ("barber" in txt) or ("staff" in txt) or ("uniform" in txt):
            return "barber_uniform"
        if "customer" in txt:
            return "customer"
        return txt or "unknown"

    def _append_role_vectors(self, role: str, vectors: List[np.ndarray]):
        if not vectors:
            return
        clean = []
        for v in vectors:
            if v is None or not isinstance(v, np.ndarray) or v.ndim != 1:
                continue
            n = float(np.linalg.norm(v)) + 1e-9
            clean.append((v / n).astype(np.float32))
        if not clean:
            return
        mat = np.stack(clean, axis=0).astype(np.float32)
        self.role_mats[role] = mat
        self.role_counts[role] = int(mat.shape[0])
        self.total_embeddings += int(mat.shape[0])

    def _load(self):
        if not os.path.exists(self.db_path):
            logger.warning(f"Role DB not found: {self.db_path}")
            return
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed loading role DB: {e}")
            return

        role_entries: Dict[str, Dict] = {}
        if isinstance(data, dict) and isinstance(data.get("roles"), dict):
            for role_name, entry in data.get("roles", {}).items():
                if isinstance(entry, dict):
                    role_entries[self._normalize_role(role_name)] = entry
        elif isinstance(data, dict):
            # Minimal fallback format: {"barber_uniform":{"embeddings":[...]}, "customer":{"embeddings":[...]}}
            for role_name, entry in data.items():
                if isinstance(entry, dict):
                    role_entries[self._normalize_role(role_name)] = entry

        for role, entry in role_entries.items():
            vectors: List[np.ndarray] = []
            avg_vector = entry.get("avg_vector")
            if isinstance(avg_vector, list) and avg_vector:
                try:
                    vectors.append(np.array(avg_vector, dtype=np.float32))
                except Exception:
                    pass
            embeddings = entry.get("embeddings", [])
            if isinstance(embeddings, list):
                for emb in embeddings[: self.max_embeddings_per_role]:
                    if not isinstance(emb, list):
                        continue
                    try:
                        vectors.append(np.array(emb, dtype=np.float32))
                    except Exception:
                        continue
            self._append_role_vectors(role, vectors)

        logger.info(
            f"Loaded role DB: roles={len(self.role_mats)} total_embs={self.total_embeddings} "
            f"counts={self.role_counts}"
        )

    def has_data(self) -> bool:
        return bool(self.role_mats)

    def match_role_scores(self, emb: Optional[np.ndarray]) -> Dict[str, float]:
        if emb is None or emb.ndim != 1 or not self.role_mats:
            return {}
        n = float(np.linalg.norm(emb)) + 1e-9
        emb_n = (emb / n).astype(np.float32)
        scores: Dict[str, float] = {}
        for role, mat in self.role_mats.items():
            if mat.size == 0 or mat.shape[1] != emb_n.shape[0]:
                continue
            sims = np.dot(mat, emb_n)
            if sims.size == 0:
                continue
            scores[role] = float(np.max(sims))
        return scores

    def match_role_with_score(self, emb: Optional[np.ndarray]) -> Tuple[Optional[str], float, Dict[str, float]]:
        scores = self.match_role_scores(emb)
        if not scores:
            return None, -1.0, {}
        best_role, best_score = max(scores.items(), key=lambda kv: float(kv[1]))
        return str(best_role), float(best_score), scores


class ActivePersonMemory:
    def __init__(self, max_prototypes_per_pid: int = 5):
        self.next_pid = 1
        self.pid_last: Dict[int, Dict] = {}
        self.gid_to_pid: Dict[int, int] = {}
        self.max_prototypes_per_pid = max(1, int(max_prototypes_per_pid))

    def _append_embedding(self, st: Dict, emb: Optional[np.ndarray], appearance_tag: Optional[str] = None):
        if emb is None:
            return
        tag = str(appearance_tag or "default").strip().lower() or "default"
        proto_map = st.setdefault("emb_proto", {})
        samples = proto_map.setdefault(tag, [])
        samples.append(emb)
        if len(samples) > self.max_prototypes_per_pid:
            del samples[:-self.max_prototypes_per_pid]
        # Keep a flattened short-list for backward compatibility/readability in debug dumps.
        flat = []
        for vecs in proto_map.values():
            flat.extend(vecs[-self.max_prototypes_per_pid:])
        st["emb_samples"] = flat[-self.max_prototypes_per_pid:]

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
        staff_id: Optional[str] = None,
        clear_staff_id: bool = False,
        role_id: Optional[str] = None,
        role_score: float = -1.0,
        appearance_tag: Optional[str] = None,
    ) -> int:
        if gid in self.gid_to_pid:
            pid = self.gid_to_pid[gid]
            info = self.pid_last.get(pid)
            if info is not None:
                info["ts"] = now_ts
                if emb is not None:
                    info["emb"] = emb
                self._append_embedding(info, emb, appearance_tag=appearance_tag)
                if clear_staff_id:
                    info["staff_id"] = None
                elif staff_id is not None:
                    info["staff_id"] = staff_id
                if role_id is not None:
                    info["role_id"] = role_id
                    info["role_score"] = float(role_score)
                return pid

        best_pid, best = None, -1.0
        if emb is not None:
            for pid, info in self.pid_last.items():
                if (now_ts - float(info.get("ts", 0.0))) > match_window_sec:
                    continue
                seen = False
                proto_map = info.get("emb_proto", {})
                if isinstance(proto_map, dict):
                    for vecs in proto_map.values():
                        for e in vecs:
                            if e is None:
                                continue
                            seen = True
                            s = cosine_sim(emb, e)
                            if s > best:
                                best = s
                                best_pid = pid
                if seen:
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

        self.pid_last.setdefault(pid, {"ts": now_ts, "emb": emb, "first_seen": now_ts, "emb_proto": {}})
        st = self.pid_last[pid]
        st["ts"] = now_ts
        if emb is not None:
            st["emb"] = emb
        self._append_embedding(st, emb, appearance_tag=appearance_tag)
        if clear_staff_id:
            st["staff_id"] = None
        elif staff_id is not None:
            st["staff_id"] = staff_id
        if role_id is not None:
            st["role_id"] = role_id
            st["role_score"] = float(role_score)
        
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

    def update_and_collect_events(self, customers: List[dict], zone_selector_fn, event_validator_fn=None, wait_tracker: Optional[Dict[int, float]] = None) -> List[dict]:
        self._ensure_day()
        now = time.time()
        events = []
        
        # Deduplication window: If a haircut was counted in this zone < 15 mins ago, don't count another.
        # This prevents double counting when tracker ID changes but it's likely the same customer.
        DEDUPE_WINDOW_SEC = 900.0  # 15 minutes
        if not hasattr(self, "zone_last_event_ts"):
            self.zone_last_event_ts = {}

        best_by_zone: Dict[str, dict] = {}
        best_score_by_zone: Dict[str, tuple] = {}
        for d in customers:
            zone = zone_selector_fn(d)
            if zone is None:
                continue
            x1, y1, x2, y2 = d["bbox"]
            area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
            det_type = str(d.get("det_type", ""))
            conf = float(d.get("conf", 0.0))
            # Prefer person over head, then higher confidence.
            # For head-only detections, prefer tighter boxes to avoid giant chair/floor false positives.
            score = (
                1 if det_type == "person" else 0,
                conf,
                (-area if det_type == "head" else area),
            )
            if (zone not in best_by_zone) or (score > best_score_by_zone.get(zone, (-1, -1.0, -1e18))):
                dd = dict(d)
                dd["_bbox_area"] = area
                best_by_zone[zone] = dd
                best_score_by_zone[zone] = score

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
                if callable(event_validator_fn):
                    ok, reason = event_validator_fn(d, zone, float(dwell))
                    if not ok:
                        if reason:
                            d["_event_reject_reason"] = str(reason)
                        continue
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
                    "wait_duration": wait_tracker.get(pid, 0.0) if wait_tracker and pid in wait_tracker else 0.0,
                    "camera": d.get("cam", ""),
                    "bbox": d.get("bbox"),
                    "conf": float(d.get("conf", 0.0)),
                    "det_type": d.get("det_type"),
                    "det_ts": float(d.get("ts", 0.0)),
                    "det_stale": bool(d.get("stale", False)),
                    "frame_w": int(d.get("frame_w", 0) or 0),
                    "frame_h": int(d.get("frame_h", 0) or 0),
                    "reject_reason": d.get("_event_reject_reason", ""),
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


@dataclass
class ServiceSession:
    pid: int
    state: str = "new"
    created_ts: float = 0.0
    last_seen_ts: float = 0.0
    state_since_ts: float = 0.0
    zone_name: str = ""
    zone_type: str = "OTHER"
    zone_since_ts: float = 0.0
    haircut_counted: bool = False
    wash_counted: bool = False
    last_haircut_ts: float = 0.0
    last_wash_ts: float = 0.0
    last_return_ts: float = 0.0
    gid: int = 0
    camera: str = ""
    last_reason: str = ""


class ServiceSessionFSM:
    """Service state machine for per-customer flow with transition reason codes."""

    def __init__(
        self,
        haircut_dwell_sec: float,
        wash_dwell_sec: float,
        return_dwell_sec: float = 3.0,
        close_timeout_sec: float = 600.0,
        return_window_sec: float = 1200.0,
        max_recent_transitions: int = 400,
    ):
        self.haircut_dwell_sec = max(1.0, float(haircut_dwell_sec))
        self.wash_dwell_sec = max(1.0, float(wash_dwell_sec))
        self.return_dwell_sec = max(0.5, float(return_dwell_sec))
        self.close_timeout_sec = max(30.0, float(close_timeout_sec))
        self.return_window_sec = max(60.0, float(return_window_sec))
        self.max_recent_transitions = max(20, int(max_recent_transitions))
        self.sessions: Dict[int, ServiceSession] = {}
        self.recent_transitions = deque(maxlen=self.max_recent_transitions)
        self.transition_seq = 0
        self.day = datetime.now().strftime("%Y-%m-%d")
        self.count_seq = 0
        self.haircut_total = 0
        self.wash_total = 0
        self.haircut_zone_total: Dict[str, int] = {}
        self.wash_zone_total: Dict[str, int] = {}

    def _ensure_day(self):
        d = datetime.now().strftime("%Y-%m-%d")
        if d != self.day:
            self.day = d
            self.count_seq = 0
            self.haircut_total = 0
            self.wash_total = 0
            self.haircut_zone_total = {}
            self.wash_zone_total = {}
            self.sessions = {}
            self.recent_transitions.clear()

    @staticmethod
    def _best_detections_by_pid(customers: List[dict]) -> Dict[int, dict]:
        best: Dict[int, dict] = {}
        best_score: Dict[int, tuple] = {}
        for d in customers:
            pid = int(d.get("pid", 0))
            if pid <= 0:
                continue
            bbox = d.get("bbox") or []
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox]
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            det_type = str(d.get("det_type", ""))
            conf = float(d.get("conf", 0.0))
            score = (1 if det_type == "person" else 0, conf, area)
            if pid not in best or score > best_score.get(pid, (-1, -1.0, -1.0)):
                best[pid] = d
                best_score[pid] = score
        return best

    @staticmethod
    def _zone_from_detection(d: dict) -> Tuple[str, str]:
        pz = str(d.get("primary_zone", "") or "")
        zones_hit = [str(z) for z in (d.get("zones") or [])]
        if pz.startswith(CHAIR_ZONE_PREFIX):
            return pz, "CHAIR"
        if pz.startswith(WASH_ZONE_PREFIX):
            return pz, "WASH"
        if pz in WAIT_ZONE_NAMES:
            return pz, "WAIT"
        chair_hits = [z for z in zones_hit if z.startswith(CHAIR_ZONE_PREFIX)]
        if chair_hits:
            return chair_hits[0], "CHAIR"
        wash_hits = [z for z in zones_hit if z.startswith(WASH_ZONE_PREFIX)]
        if wash_hits:
            return wash_hits[0], "WASH"
        wait_hits = [z for z in zones_hit if z in WAIT_ZONE_NAMES]
        if wait_hits:
            return wait_hits[0], "WAIT"
        return pz, "OTHER"

    def _transition(self, sess: ServiceSession, to_state: str, reason_code: str, ts: float, det: Optional[dict] = None) -> Optional[dict]:
        from_state = str(sess.state)
        if from_state == str(to_state) and str(reason_code) == str(sess.last_reason):
            return None
        self.transition_seq += 1
        rec = {
            "transition_id": int(self.transition_seq),
            "timestamp": float(ts),
            "pid": int(sess.pid),
            "gid": int(sess.gid),
            "camera": str(sess.camera),
            "from_state": from_state,
            "to_state": str(to_state),
            "reason_code": str(reason_code),
            "zone": str(sess.zone_name),
            "zone_type": str(sess.zone_type),
            "state_age_sec": float(max(0.0, float(ts) - float(sess.state_since_ts))),
        }
        if isinstance(det, dict):
            rec["bbox"] = det.get("bbox")
            rec["det_conf"] = float(det.get("conf", 0.0))
            rec["det_type"] = det.get("det_type")
        sess.state = str(to_state)
        sess.state_since_ts = float(ts)
        sess.last_reason = str(reason_code)
        self.recent_transitions.append(rec)
        return rec

    def _count_event(
        self,
        kind: str,
        sess: ServiceSession,
        det: dict,
        reason_code: str,
        zone_dwell_sec: float,
        ts: float,
        wait_tracker: Optional[Dict[int, float]] = None,
        from_state: Optional[str] = None,
        to_state: Optional[str] = None,
        transition_id: int = 0,
    ) -> dict:
        self.count_seq += 1
        zone_name = str(sess.zone_name or det.get("primary_zone", ""))
        if kind == "haircut":
            self.haircut_total += 1
            self.haircut_zone_total[zone_name] = self.haircut_zone_total.get(zone_name, 0) + 1
        elif kind == "wash":
            self.wash_total += 1
            self.wash_zone_total[zone_name] = self.wash_zone_total.get(zone_name, 0) + 1
        return {
            "seq": int(self.count_seq),
            "day": str(self.day),
            "event_id": datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")[:-3],
            "zone": zone_name,
            "pid": int(sess.pid),
            "gid": int(sess.gid),
            "dwell": float(max(0.0, zone_dwell_sec)),
            "zone_dwell_sec": float(max(0.0, zone_dwell_sec)),
            "camera": str(det.get("cam", sess.camera)),
            "bbox": det.get("bbox"),
            "conf": float(det.get("conf", 0.0)),
            "det_type": det.get("det_type"),
            "det_ts": float(det.get("ts", 0.0)),
            "det_stale": bool(det.get("stale", False)),
            "frame_w": int(det.get("frame_w", 0) or 0),
            "frame_h": int(det.get("frame_h", 0) or 0),
            "wait_duration": float(wait_tracker.get(sess.pid, 0.0)) if (kind == "haircut" and wait_tracker) else 0.0,
            "staff_id": det.get("staff_id"),
            "role_id": det.get("role_id"),
            "role_vote": det.get("role_vote"),
            "role_vote_ratio": float(det.get("role_vote_ratio", 0.0) or 0.0),
            "role_vote_samples": int(det.get("role_vote_samples", 0) or 0),
            "role_barber_score": float(det.get("role_barber_score", -1.0) or -1.0),
            "role_customer_score": float(det.get("role_customer_score", -1.0) or -1.0),
            "reason_code": str(reason_code),
            "from_state": str(from_state or ""),
            "to_state": str(to_state or sess.state),
            "session_state": str(sess.state),
            "transition_id": int(transition_id),
        }

    def update(
        self,
        customers: List[dict],
        now_ts: float,
        wait_tracker: Optional[Dict[int, float]] = None,
        event_validator_fn=None,
    ) -> Dict[str, object]:
        self._ensure_day()
        haircut_events: List[dict] = []
        wash_events: List[dict] = []
        transitions: List[dict] = []
        best_by_pid = self._best_detections_by_pid(customers)
        active_pids = set(best_by_pid.keys())

        for pid, d in best_by_pid.items():
            sess = self.sessions.get(pid)
            if sess is None:
                sess = ServiceSession(
                    pid=int(pid),
                    state="new",
                    created_ts=float(now_ts),
                    last_seen_ts=float(now_ts),
                    state_since_ts=float(now_ts),
                    zone_since_ts=float(now_ts),
                )
                self.sessions[int(pid)] = sess
                tr = self._transition(sess, "waiting", "session_started", now_ts, d)
                if tr is not None:
                    transitions.append(tr)

            sess.gid = int(d.get("gid", sess.gid) or sess.gid)
            sess.camera = str(d.get("cam", sess.camera) or sess.camera)
            sess.last_seen_ts = float(now_ts)

            zone_name, zone_type = self._zone_from_detection(d)
            if zone_name != sess.zone_name or zone_type != sess.zone_type:
                sess.zone_name = zone_name
                sess.zone_type = zone_type
                sess.zone_since_ts = float(now_ts)

            zone_dwell = float(max(0.0, float(now_ts) - float(sess.zone_since_ts)))

            if sess.zone_type == "WAIT":
                if sess.state not in {"waiting", "closed"}:
                    tr = self._transition(sess, "waiting", "enter_wait_zone", now_ts, d)
                    if tr is not None:
                        transitions.append(tr)
                continue

            if sess.zone_type == "CHAIR":
                if sess.wash_counted:
                    if sess.state != "returned":
                        if (float(now_ts) - float(sess.last_wash_ts)) <= self.return_window_sec and zone_dwell >= self.return_dwell_sec:
                            tr = self._transition(sess, "returned", "returned_to_chair_after_wash", now_ts, d)
                            if tr is not None:
                                transitions.append(tr)
                                sess.last_return_ts = float(now_ts)
                    continue
                if (not sess.haircut_counted) and zone_dwell >= self.haircut_dwell_sec:
                    if callable(event_validator_fn):
                        ok, _reason = event_validator_fn("haircut", d, str(sess.zone_name), float(zone_dwell))
                        if not ok:
                            continue
                    tr = self._transition(sess, "haircut", "chair_dwell_met_first_service", now_ts, d)
                    tr_id = int(tr.get("transition_id", 0)) if isinstance(tr, dict) else 0
                    if tr is not None:
                        transitions.append(tr)
                    sess.haircut_counted = True
                    sess.last_haircut_ts = float(now_ts)
                    haircut_events.append(
                        self._count_event(
                            "haircut",
                            sess,
                            d,
                            reason_code="chair_dwell_met_first_service",
                            zone_dwell_sec=zone_dwell,
                            ts=now_ts,
                            wait_tracker=wait_tracker,
                            from_state=(tr.get("from_state") if isinstance(tr, dict) else sess.state),
                            to_state="haircut",
                            transition_id=tr_id,
                        )
                    )
                continue

            if sess.zone_type == "WASH":
                if (not sess.wash_counted) and zone_dwell >= self.wash_dwell_sec:
                    if callable(event_validator_fn):
                        ok, _reason = event_validator_fn("wash", d, str(sess.zone_name), float(zone_dwell))
                        if not ok:
                            continue
                    reason_code = (
                        "wash_zone_dwell_met_after_haircut"
                        if sess.haircut_counted
                        else "wash_zone_dwell_met_first_service"
                    )
                    tr = self._transition(sess, "wash", reason_code, now_ts, d)
                    tr_id = int(tr.get("transition_id", 0)) if isinstance(tr, dict) else 0
                    if tr is not None:
                        transitions.append(tr)
                    sess.wash_counted = True
                    sess.last_wash_ts = float(now_ts)
                    wash_events.append(
                        self._count_event(
                            "wash",
                            sess,
                            d,
                            reason_code=reason_code,
                            zone_dwell_sec=zone_dwell,
                            ts=now_ts,
                            wait_tracker=wait_tracker,
                            from_state=(tr.get("from_state") if isinstance(tr, dict) else sess.state),
                            to_state="wash",
                            transition_id=tr_id,
                        )
                    )
                continue

        for pid, sess in list(self.sessions.items()):
            if int(pid) in active_pids:
                continue
            idle = float(now_ts) - float(sess.last_seen_ts)
            if idle < self.close_timeout_sec:
                continue
            tr = self._transition(sess, "closed", "inactive_timeout", now_ts, None)
            if tr is not None:
                transitions.append(tr)
            self.sessions.pop(int(pid), None)

        state_counts: Dict[str, int] = {}
        for sess in self.sessions.values():
            st = str(sess.state)
            state_counts[st] = state_counts.get(st, 0) + 1

        return {
            "haircut_events": haircut_events,
            "wash_events": wash_events,
            "transitions": transitions,
            "state_counts": state_counts,
            "open_sessions": int(len(self.sessions)),
            "active_pids": active_pids,
        }

    def reset(self):
        self.day = datetime.now().strftime("%Y-%m-%d")
        self.count_seq = 0
        self.transition_seq = 0
        self.haircut_total = 0
        self.wash_total = 0
        self.haircut_zone_total = {}
        self.wash_zone_total = {}
        self.sessions = {}
        self.recent_transitions.clear()


# =========================
# CAMERA STREAM
# =========================

class VerificationWorker(threading.Thread):
    """Tier-2 background analysis worker"""
    def __init__(self, runtime_service: "RuntimeService", job_queue: "queue.Queue", model: YOLO):
        super().__init__(daemon=True)
        self.runtime_service = runtime_service
        self.job_queue = job_queue
        self.model = model
        self.sample_stride = 5
        self.stop_event = threading.Event()

    def stop(self):
        self.stop_event.set()
        try:
            self.job_queue.put_nowait(None)
        except queue.Full:
            pass

    def run(self):
        if self.model is None:
            logger.error("Tier-2 Worker: verification model is not available")
            return
        logger.info("Tier-2 Worker: Ready.")

        while not self.stop_event.is_set():
            job = None
            try:
                job = self.job_queue.get(timeout=1.0)
                if job is None:
                    self.job_queue.task_done()
                    continue

                clip_path = job["file"]
                cam_name = job["cam"]
                logger.info(f"Tier-2 Worker: Verifying clip {clip_path} (Cam: {cam_name})")
                self.runtime_service._record_tier2_handoff(
                    cam_name,
                    {
                        "action": "processing",
                        "pid": int(job.get("tid", 0) or 0),
                        "zone_name": str(job.get("zone", "") or ""),
                        "queue_depth": int(self.job_queue.qsize()),
                    },
                )

                result = self._analyze_clip(clip_path)
                if result is not None:
                    self.runtime_service._handle_verification_result(job, result)
                else:
                    self.runtime_service._handle_verification_miss(job)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Tier-2 Worker Error: {e}")
                time.sleep(1.0)
            finally:
                if job is not None:
                    clip_path = str(job.get("file", "") or "")
                    if clip_path and os.path.exists(clip_path):
                        try:
                            os.remove(clip_path)
                        except Exception as cleanup_err:
                            logger.warning(f"Tier-2 Worker: cleanup failed for {clip_path}: {cleanup_err}")
                    try:
                        self.job_queue.task_done()
                    except ValueError:
                        pass
        return

    def _analyze_clip(self, clip_path: str) -> Optional[Dict]:
        cap = cv2.VideoCapture(clip_path)
        best_result: Optional[Dict] = None
        frame_count = 0
        verify_classes = [
            self.runtime_service.head_class_id,
            self.runtime_service.staff_uniform_class_id,
        ]

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                if (frame_count % self.sample_stride) != 0:
                    continue

                results = self.model.predict(
                    frame,
                    conf=max(0.25, float(self.runtime_service.yolo_conf)),
                    iou=float(self.runtime_service.yolo_iou),
                    imgsz=int(self.runtime_service.yolo_imgsz),
                    classes=verify_classes,
                    verbose=False,
                )
                if not results or results[0].boxes is None:
                    continue

                boxes = results[0].boxes
                cls_vals = boxes.cls
                conf_vals = boxes.conf
                head_count = 0
                staff_count = 0
                conf_total = 0.0

                for i in range(len(boxes)):
                    cls_id = int(float(cls_vals[i].item())) if cls_vals is not None else -1
                    conf_val = float(conf_vals[i].item()) if conf_vals is not None else 0.0
                    conf_total += conf_val
                    if cls_id == self.runtime_service.head_class_id:
                        head_count += 1
                    elif cls_id == self.runtime_service.staff_uniform_class_id:
                        staff_count += 1

                if head_count <= 0 and staff_count <= 0:
                    continue

                candidate = {
                    "head_count": int(head_count),
                    "staff_count": int(staff_count),
                    "confidence_total": float(conf_total),
                    "best_frame": frame.copy(),
                    "sampled_frame_index": int(frame_count),
                }
                if self._is_better_result(candidate, best_result):
                    best_result = candidate
        finally:
            cap.release()

        return best_result

    @staticmethod
    def _is_better_result(candidate: Dict, current: Optional[Dict]) -> bool:
        if current is None:
            return True
        cand_pair = (int(candidate.get("head_count", 0)), int(candidate.get("staff_count", 0)))
        curr_pair = (int(current.get("head_count", 0)), int(current.get("staff_count", 0)))
        if cand_pair != curr_pair:
            return cand_pair > curr_pair
        return float(candidate.get("confidence_total", 0.0)) >= float(current.get("confidence_total", 0.0))


class CameraStream:
    """Capture from RTSP camera"""
    
    def __init__(
        self,
        camera_name: str,
        rtsp_url: str,
        freeze_max_same_frames: int = 120,
        freeze_diff_threshold: float = 1.2,
        connect_policy: Optional[Dict[str, bool]] = None,
        verification_queue_ref: Optional["queue.Queue"] = None,
        clip_duration_sec: float = TIER2_CLIP_DURATION_SEC,
        clip_dir: str = "temp_clips",
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
        self.rolling_buffer = deque(maxlen=150)
        self.recording_event: Optional[Dict] = None # { 'tid': int, 'frames_to_go': int, 'buffer': [] }
        self.verification_queue = verification_queue_ref
        self.clip_duration_sec = max(1.0, float(clip_duration_sec))
        self.clip_dir = clip_dir
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
            
            if frame is not None:
                self.rolling_buffer.append(frame.copy())
                # Handle active recording
                if self.recording_event:
                    self.recording_event['buffer'].append(frame.copy())
                    self.recording_event['frames_to_go'] -= 1
                    if self.recording_event['frames_to_go'] <= 0:
                        # Save clip and put in queue
                        self._save_trigger_clip_v2()
                        self.recording_event = None

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
            self.disconnect()
            time.sleep(1)
            self.connect()
            with self.cap_lock:
                cap = self.cap
            if cap and cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    self.is_connected = True
                    logger.info(f"Successfully reconnected to {self.camera_name}")
                    return True
            return False
        except Exception as e:
            logger.error(f"Error testing connection for {self.camera_name}: {e}")
            return False

    def _save_trigger_clip_v2(self):
        """Save captured buffer to disk for Tier-2"""
        if not self.recording_event:
            return
        frames = self.recording_event.get('buffer', [])
        tid = int(self.recording_event.get('tid', 0))
        if not frames:
            return

        os.makedirs(self.clip_dir, exist_ok=True)
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = os.path.join(
            self.clip_dir,
            f"trigger_{self.camera_name.replace(' ', '_')}_ID{tid}_{ts_str}.mp4",
        )
        
        try:
            h, w = frames[0].shape[:2]
            fps = max(1.0, float(self.fps) if self.fps else 15.0)
            out = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            for f in frames:
                out.write(f)
            out.release()
            
            if self.verification_queue is None:
                logger.warning(f"Tier-2 queue unavailable, dropping clip {fname}")
                return

            job = {
                'file': fname,
                'cam': self.camera_name,
                'tid': tid,
                'zone': str(self.recording_event.get('zone', '') or ''),
                'trigger_ts': float(self.recording_event.get('trigger_ts', time.time())),
                'trigger_frame_count': int(len(frames)),
            }
            try:
                self.verification_queue.put_nowait(job)
                logger.info(f"Clip queued for Tier-2: {fname}")
            except queue.Full:
                logger.warning(f"Tier-2 queue full, dropping clip {fname}")
                if hasattr(self, "runtime_service") and self.runtime_service is not None:
                    try:
                        self.runtime_service._record_tier2_handoff(
                            self.camera_name,
                            {
                                "action": "dropped",
                                "eligible": False,
                                "pid": tid,
                                "zone_name": str(self.recording_event.get("zone", "") or ""),
                                "reason": "queue_full",
                            },
                        )
                    except Exception:
                        pass
                if os.path.exists(fname):
                    os.remove(fname)
        except Exception as e:
            logger.error(f"โ Error saving trigger clip: {e}")
    
    def get_clip_frame_budget(self) -> int:
        fps = max(1.0, float(self.fps) if self.fps else 15.0)
        return max(1, int(round(fps * self.clip_duration_sec)))

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

    @staticmethod
    def _normalize_count_mode(raw: object) -> str:
        mode = str(raw or "").strip().lower()
        return "recorded" if mode in {"recorded", "history", "batch", "เธขเนเธญเธเธซเธฅเธฑเธ"} else "live"
    
    def __init__(self):
        self.service_mode = "live"
        self.discovery_model, self.verify_model, self.yolo_device = self._load_yolo()
        self.yolo_lock = threading.Lock()
        self.verification_queue: "queue.Queue" = queue.Queue()
        self.verification_state_lock = threading.Lock()
        self.verified_events_total = 0
        self.verified_by_camera: Dict[str, int] = {}
        self.last_verified_event: Optional[Dict] = None
        self.tier2_latest_by_camera: Dict[str, Dict] = {}
        self.tier2_handoff_history = deque(maxlen=60)

        # Start Tier-2 Worker
        self.worker = VerificationWorker(self, self.verification_queue, self.verify_model)
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
        self._last_role_snapshot_ts: Dict[str, float] = {}
        self._last_role_snapshot_thumb: Dict[str, np.ndarray] = {}
        self._role_snapshot_daily_counts: Dict[str, int] = {}
        self._role_snapshot_last_day = datetime.now().strftime("%Y-%m-%d")
        
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
        self._runtime_day = datetime.now().strftime("%Y-%m-%d")
        self._daily_rollover_lock = threading.Lock()
        self.last_health_broadcast = 0
        self.customers_lock = threading.Lock()
        self.latest_customers: Dict[str, List[Dict]] = {}
        self.latest_detections: Dict[str, List[Dict]] = {}
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
        self.service_mode = self._normalize_count_mode(runtime_cfg.get("count_mode", self.service_mode))
        staff_cfg = CONFIG.get("staff", {})
        def _cfg_bool(v, default=False):
            if v is None:
                return bool(default)
            if isinstance(v, str):
                return v.strip().lower() in {"1", "true", "yes", "on"}
            return bool(v)
        self.resource_guard.max_memory_percent = float(runtime_cfg.get("resource_max_memory_percent", 95.0))
        self.throttle_log_interval_sec = float(runtime_cfg.get("throttle_log_interval_sec", 5.0))
        self.target_fps = max(1, int(runtime_cfg.get("target_fps", 10)))
        self.dashboard_ui_frame_interval_sec = max(
            0.03, float(runtime_cfg.get("dashboard_ui_frame_interval_sec", 0.20))
        )
        self.dashboard_state_min_interval_sec = max(
            0.20, float(runtime_cfg.get("dashboard_state_min_interval_sec", 1.00))
        )
        self.gc_collect_interval_sec = max(
            10.0, float(runtime_cfg.get("gc_collect_interval_sec", 60.0))
        )
        self.summary_log_interval_sec = max(
            5.0, float(runtime_cfg.get("summary_log_interval_sec", 30.0))
        )
        self._last_throttle_log_ts: Dict[str, float] = {}
        self._last_ui_update_ts: Dict[str, float] = {}
        self._last_dashboard_state_write_ts = 0.0
        self._last_dashboard_state_payload_hash = ""
        self._last_gc_collect_ts = time.time()
        self._last_summary_log_ts = 0.0
        self.zone_point_mode = runtime_cfg.get("zone_point_mode", "foot")
        self.sit_min_sec = float(runtime_cfg.get("sit_min_sec", 10))
        self.vacant_grace_sec = float(runtime_cfg.get("vacant_grace_sec", 6))
        self.tier2_clip_duration_sec = max(1.0, float(runtime_cfg.get("tier2_clip_duration_sec", TIER2_CLIP_DURATION_SEC)))
        self.tier2_trigger_dwell_sec = 3.0
        self.tier2_trigger_cooldown_sec = 60.0
        self.enable_business_hours_guard = _cfg_bool(
            runtime_cfg.get("enable_business_hours_guard", False),
            False,
        )
        self.business_hours_start = str(runtime_cfg.get("business_hours_start", "09:00"))
        self.business_hours_end = str(runtime_cfg.get("business_hours_end", "22:00"))
        self.restore_dashboard_state_on_startup = _cfg_bool(
            runtime_cfg.get("restore_dashboard_state_on_startup", False),
            False,
        )
        self.restore_dashboard_state_max_age_sec = max(
            0.0,
            float(runtime_cfg.get("restore_dashboard_state_max_age_sec", 900.0)),
        )
        
        # Missing attributes restored
        # Tiered Sensitivity Settings
        self.yolo_conf = 0.25 # Sensitive discovery
        self.yolo_iou = 0.45
        self.yolo_imgsz = 640
        self.yolo_mode = "track"  # default
        self.camera_stall_timeout_sec = 60.0
        self.camera_freeze_max_same_frames = 120
        self.camera_freeze_diff_threshold = 1.2
        infer_mode_cfg = str(runtime_cfg.get("inference_mode", "always")).strip().lower()
        if infer_mode_cfg not in {"always", "motion_gated"}:
            infer_mode_cfg = "always"
        self.inference_mode = infer_mode_cfg
        self._configured_inference_mode = infer_mode_cfg
        self.auto_motion_gate_enabled = _cfg_bool(
            runtime_cfg.get("auto_motion_gate_enabled", True), True
        )
        self.auto_motion_gate_cpu_threshold = min(
            99.0, max(10.0, float(runtime_cfg.get("auto_motion_gate_cpu_threshold", 85.0)))
        )
        self.auto_motion_gate_mem_threshold = min(
            99.0, max(10.0, float(runtime_cfg.get("auto_motion_gate_mem_threshold", 88.0)))
        )
        self.auto_motion_gate_recover_sec = max(
            5.0, float(runtime_cfg.get("auto_motion_gate_recover_sec", 20.0))
        )
        self.auto_motion_gate_min_hold_sec = max(
            3.0, float(runtime_cfg.get("auto_motion_gate_min_hold_sec", 15.0))
        )
        self._auto_motion_gate_active = False
        self._auto_motion_last_overload_ts = 0.0
        self._auto_motion_last_switch_ts = 0.0
        self._last_auto_motion_log_ts = 0.0
        self.motion_diff_threshold = 10.0 # More sensitive
        self.motion_min_area_ratio = 0.002
        self.motion_recheck_sec = 2.0
        self.motion_hold_sec = 5.0
        self.motion_downscale_width = 128
        self.customer_active_ttl_sec = 10.0
        self.dashboard_presence_ttl_sec = 60.0
        self.no_detection_hold_sec = 2.0
        self.reid_similarity_threshold = 0.75
        self.staff_similarity_threshold = 0.82
        self.staff_event_similarity_threshold = max(0.60, self.staff_similarity_threshold - 0.10)
        self.enable_role_snapshots = _cfg_bool(runtime_cfg.get("enable_role_snapshots", True), True)
        self.role_snapshot_cooldown_sec = max(
            1.0, float(runtime_cfg.get("role_snapshot_cooldown_sec", 20.0))
        )
        self.role_snapshot_customer_interval_sec = max(
            10.0, float(runtime_cfg.get("role_snapshot_customer_interval_sec", 600.0))
        )
        self.role_snapshot_barber_interval_sec = max(
            10.0, float(runtime_cfg.get("role_snapshot_barber_interval_sec", 300.0))
        )
        self.role_snapshot_wash_interval_sec = max(
            10.0, float(runtime_cfg.get("role_snapshot_wash_interval_sec", 300.0))
        )
        self.role_snapshot_unknown_interval_sec = max(
            10.0, float(runtime_cfg.get("role_snapshot_unknown_interval_sec", 180.0))
        )
        self.role_snapshot_enable_unknown = _cfg_bool(
            runtime_cfg.get("role_snapshot_enable_unknown", True),
            True,
        )
        self.role_snapshot_thumb_min_diff = max(
            1.0, float(runtime_cfg.get("role_snapshot_thumb_min_diff", 8.0))
        )
        self.role_snapshot_min_conf = min(
            1.0, max(0.0, float(runtime_cfg.get("role_snapshot_min_conf", 0.45)))
        )
        self.role_snapshot_min_area_ratio = min(
            1.0, max(0.0, float(runtime_cfg.get("role_snapshot_min_area_ratio", 0.01)))
        )
        self.role_snapshot_require_customer_chair_zone = _cfg_bool(
            runtime_cfg.get("role_snapshot_require_customer_chair_zone", True), True
        )
        self.role_snapshot_customer_max_per_camera_day = max(
            1, int(runtime_cfg.get("role_snapshot_customer_max_per_camera_day", 120))
        )
        self.role_snapshot_barber_max_per_camera_day = max(
            1, int(runtime_cfg.get("role_snapshot_barber_max_per_camera_day", 80))
        )
        self.role_snapshot_wash_max_per_camera_day = max(
            1, int(runtime_cfg.get("role_snapshot_wash_max_per_camera_day", 80))
        )
        self.role_snapshot_unknown_max_per_camera_day = max(
            0, int(runtime_cfg.get("role_snapshot_unknown_max_per_camera_day", 120))
        )
        self.barber_snapshot_dir = Path(
            PATHS_CONFIG.get(
                "barber_uniform_dir",
                str(Path(PATHS_CONFIG.get("staff_gallery", "data/staff_gallery")) / "BARBER_UNIFORM"),
            )
        )
        self.customer_snapshot_dir = Path(
            PATHS_CONFIG.get("customer_by_admin", "data/customer_by_admin")
        )
        self.wash_customer_snapshot_dir = Path(
            PATHS_CONFIG.get("customer_wash_by_admin", "data/customer_wash_by_admin")
        )
        self.unknown_snapshot_dir = Path(
            PATHS_CONFIG.get("unknown_by_admin", "data/unknown_by_admin")
        )
        self.enable_event_feedback_autocopy = _cfg_bool(
            runtime_cfg.get("enable_event_feedback_autocopy", True),
            True,
        )
        self.performance_feedback_root_dir = Path(
            PATHS_CONFIG.get("performance_feedback", "data/performance_feedback")
        )
        self.feedback_haircut_dir = self.performance_feedback_root_dir / "haircut"
        self.feedback_customerwash_dir = self.performance_feedback_root_dir / "customerwash"
        self.feedback_no_haircut_dir = self.performance_feedback_root_dir / "no haircut"
        self.feedback_no_customerwash_dir = self.performance_feedback_root_dir / "no customerwash"
        self.enable_no_haircut_feedback_autocopy = _cfg_bool(
            runtime_cfg.get("enable_no_haircut_feedback_autocopy", True),
            True,
        )
        self.no_haircut_feedback_cooldown_sec = max(
            1.0,
            float(runtime_cfg.get("no_haircut_feedback_cooldown_sec", 20.0)),
        )
        self._last_no_haircut_feedback_ts: Dict[str, float] = {}
        try:
            self.barber_snapshot_dir.mkdir(parents=True, exist_ok=True)
            self.customer_snapshot_dir.mkdir(parents=True, exist_ok=True)
            self.wash_customer_snapshot_dir.mkdir(parents=True, exist_ok=True)
            self.feedback_haircut_dir.mkdir(parents=True, exist_ok=True)
            self.feedback_customerwash_dir.mkdir(parents=True, exist_ok=True)
            self.feedback_no_haircut_dir.mkdir(parents=True, exist_ok=True)
            self.feedback_no_customerwash_dir.mkdir(parents=True, exist_ok=True)
            if self.role_snapshot_enable_unknown and self.role_snapshot_unknown_max_per_camera_day > 0:
                self.unknown_snapshot_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to create role snapshot directories: {e}")
            self.enable_role_snapshots = False
        self.reid_active_expire_sec = 3600.0  # 1 hour memory by default
        self.reid_match_window_sec = 30.0
        self.cross_camera_dedupe_window_sec = 15.0
        self.cross_camera_dedupe_similarity = 0.75
        self.reid_merge_for_counts = True
        self.enable_event_reid_dedupe = True
        self.haircut_event_dedupe_window_sec = 900.0
        self.haircut_event_dedupe_similarity = 0.82
        self.haircut_count_zones = self._parse_haircut_count_zones(
            runtime_cfg.get("haircut_count_zones", {})
        )
        self.enable_chair_service_classifier = _cfg_bool(
            runtime_cfg.get("enable_chair_service_classifier", False),
            False,
        )
        self.chair_service_classifier_model = str(
            runtime_cfg.get("chair_service_classifier_model", "models/chair_service_cls.pt")
        )
        self.chair_service_classifier_positive_labels = _normalize_label_set(
            runtime_cfg.get("chair_service_classifier_positive_labels", ["haircut"])
        ) or {"haircut"}
        self.chair_service_classifier_min_conf = min(
            1.0,
            max(0.0, float(runtime_cfg.get("chair_service_classifier_min_conf", 0.55))),
        )
        self.chair_service_classifier_imgsz = max(
            96,
            int(runtime_cfg.get("chair_service_classifier_imgsz", 224)),
        )
        self.chair_service_classifier: Optional[YOLO] = None
        self.chair_service_classifier_device = "cpu"
        self.chair_service_classifier_lock = threading.Lock()
        self._last_chair_service_classifier_log_ts = 0.0
        # Auto-retrain haircut/not_haircut classifier from performance feedback folders.
        self.chair_service_autotrain_enabled = _cfg_bool(
            runtime_cfg.get("chair_service_autotrain_enabled", True),
            True,
        )
        self.chair_service_autotrain_interval_hours = max(
            0.25,
            float(runtime_cfg.get("chair_service_autotrain_interval_hours", 6.0)),
        )
        self.chair_service_autotrain_min_positive = max(
            1,
            int(runtime_cfg.get("chair_service_autotrain_min_positive", 20)),
        )
        self.chair_service_autotrain_min_negative = max(
            1,
            int(runtime_cfg.get("chair_service_autotrain_min_negative", 20)),
        )
        self.chair_service_autotrain_positive_dir = str(
            runtime_cfg.get(
                "chair_service_autotrain_positive_dir",
                "data/performance_feedback/haircut",
            )
        )
        neg_dirs_cfg = runtime_cfg.get(
            "chair_service_autotrain_negative_dirs",
            [
                "data/performance_feedback/no haircut",
            ],
        )
        if isinstance(neg_dirs_cfg, str):
            neg_dirs = [x.strip() for x in neg_dirs_cfg.split(",") if x.strip()]
        elif isinstance(neg_dirs_cfg, (list, tuple, set)):
            neg_dirs = [str(x).strip() for x in neg_dirs_cfg if str(x).strip()]
        else:
            neg_dirs = ["data/performance_feedback/no haircut"]
        self.chair_service_autotrain_negative_dirs = neg_dirs or ["data/performance_feedback/no haircut"]
        self.chair_service_autotrain_epochs = max(
            1, int(runtime_cfg.get("chair_service_autotrain_epochs", 30))
        )
        self.chair_service_autotrain_batch = max(
            1, int(runtime_cfg.get("chair_service_autotrain_batch", 32))
        )
        self.chair_service_autotrain_imgsz = max(
            96, int(runtime_cfg.get("chair_service_autotrain_imgsz", 224))
        )
        self.chair_service_autotrain_patience = max(
            1, int(runtime_cfg.get("chair_service_autotrain_patience", 12))
        )
        self.chair_service_autotrain_workers = max(
            0, int(runtime_cfg.get("chair_service_autotrain_workers", 2))
        )
        self.chair_service_autotrain_train_split = min(
            0.99,
            max(0.01, float(runtime_cfg.get("chair_service_autotrain_train_split", 0.85))),
        )
        self.chair_service_autotrain_device = str(
            runtime_cfg.get("chair_service_autotrain_device", "auto")
        ).strip() or "auto"
        self.chair_service_autotrain_timeout_min = max(
            10, int(runtime_cfg.get("chair_service_autotrain_timeout_min", 120))
        )
        self.chair_service_autotrain_state_file = PROJECT_ROOT / "runtime" / "chair_service_autotrain_state.json"
        self._chair_service_autotrain_thread: Optional[threading.Thread] = None
        self._chair_service_autotrain_lock = threading.Lock()
        self._chair_service_autotrain_running = False
        self._chair_service_autotrain_next_check_ts = 0.0
        self._chair_service_autotrain_last_signature = ""
        self._chair_service_autotrain_last_train_ts = 0.0
        self._chair_service_autotrain_last_status = "idle"
        self._chair_service_autotrain_last_error = ""
        self._chair_service_autotrain_last_data_counts = {"positive": 0, "negative": 0}
        self._load_chair_service_autotrain_state()
        # Guard against false "new customer" counts by checking empty-chair similarity
        # and recent identity memory before creating haircut events.
        self.enable_chair_empty_guard = _cfg_bool(
            runtime_cfg.get("enable_chair_empty_guard", True),
            True,
        )
        self.chair_empty_gallery_dir = str(
            runtime_cfg.get("chair_empty_gallery_dir", "data/chair_emty")
        )
        self.chair_empty_similarity_threshold = min(
            1.0,
            max(0.0, float(runtime_cfg.get("chair_empty_similarity_threshold", 0.90))),
        )
        self.event_identity_similarity_threshold = min(
            1.0,
            max(0.0, float(runtime_cfg.get("event_identity_similarity_threshold", 0.85))),
        )
        self.event_same_customer_block_sec = max(
            10.0,
            float(runtime_cfg.get("event_same_customer_block_sec", 1800.0)),
        )
        self.event_same_seat_cooldown_sec = max(
            0.0,
            float(runtime_cfg.get("event_same_seat_cooldown_sec", 1800.0)),
        )
        self.event_wash_return_block_sec = max(
            10.0,
            float(runtime_cfg.get("event_wash_return_block_sec", 1800.0)),
        )
        self.require_chair_vacant_between_haircuts = _cfg_bool(
            runtime_cfg.get("require_chair_vacant_between_haircuts", True),
            True,
        )
        self.chair_empty_hist_weight = min(
            1.0,
            max(0.0, float(runtime_cfg.get("chair_empty_hist_weight", 0.35))),
        )
        self.exclude_staff_from_counts = True
        self._last_det_stats_log_ts = {}
        self._last_process_ts = {}
        self._last_camera_heartbeat = {}
        self._last_pid_canon_log_ts = 0
        self._last_pid_canon_ok_log_ts = 0
        self._last_camera_nonzero = {}
        self.camera_heartbeat_sec = 30.0
        self.camera_no_detection_warn_sec = 300.0
        self._latency_lock = threading.Lock()
        self._infer_latency_ms = deque(maxlen=300)
        self._process_latency_ms = deque(maxlen=300)
        # Custom classes for fine-grained salon detection.
        self.person_class_id = int(YOLO_CONFIG.get("person_class_id", 0))
        self.head_class_id = int(YOLO_CONFIG.get("head_class_id", 1))
        self.staff_uniform_class_id = int(YOLO_CONFIG.get("staff_uniform_class_id", 2))
        class_ids_cfg = YOLO_CONFIG.get(
            "detect_class_ids",
            [self.person_class_id, self.head_class_id, self.staff_uniform_class_id],
        )
        self.yolo_detect_class_ids = [int(x) for x in class_ids_cfg if x is not None]
        self.discovery_person_class_id: int = self.person_class_id
        self.discovery_head_class_id: Optional[int] = self.head_class_id
        self.discovery_staff_uniform_class_id: Optional[int] = self.staff_uniform_class_id
        self.staff_uniform_iou_threshold = float(
            runtime_cfg.get("staff_uniform_iou_threshold", 0.20)
        )
        self.staff_event_similarity_threshold = float(
            runtime_cfg.get(
                "staff_event_similarity_threshold",
                max(0.60, self.staff_similarity_threshold - 0.10),
            )
        )
        self.enable_staff_uniform = _cfg_bool(runtime_cfg.get("enable_staff_uniform", True), True)
        self._configure_discovery_class_schema()
        self._load_chair_service_classifier()
        self.enable_head_size_guard = _cfg_bool(runtime_cfg.get("enable_head_size_guard", False), False)
        # Head-only false-positive guardrails (prevent giant chair/floor boxes from being treated as heads).
        self.head_max_width_ratio = float(runtime_cfg.get("head_max_width_ratio", 0.90))
        self.head_max_height_ratio = float(runtime_cfg.get("head_max_height_ratio", 0.90))
        self.head_max_area_ratio = float(runtime_cfg.get("head_max_area_ratio", 0.45))
        # Event-level validation (stricter than per-frame detection) to reduce empty-chair/empty-bed snapshots.
        self.event_require_fresh_detection = _cfg_bool(runtime_cfg.get("event_require_fresh_detection", True), True)
        self.event_reject_stale_hold = _cfg_bool(runtime_cfg.get("event_reject_stale_hold", True), True)
        self.event_max_detection_age_sec = float(runtime_cfg.get("event_max_detection_age_sec", 1.5))
        self.event_head_min_conf = float(runtime_cfg.get("event_head_min_conf", 0.55))
        self.event_person_min_conf = float(runtime_cfg.get("event_person_min_conf", 0.35))
        self.event_head_min_area_ratio = float(runtime_cfg.get("event_head_min_area_ratio", 0.001))
        self.event_head_max_area_ratio = float(runtime_cfg.get("event_head_max_area_ratio", 0.18))
        self.event_head_max_width_ratio = float(runtime_cfg.get("event_head_max_width_ratio", 0.60))
        self.event_head_max_height_ratio = float(runtime_cfg.get("event_head_max_height_ratio", 0.60))
        self.event_snapshot_require_live_match = _cfg_bool(runtime_cfg.get("event_snapshot_require_live_match", True), True)
        self.event_snapshot_fallback_to_event_bbox = _cfg_bool(
            runtime_cfg.get("event_snapshot_fallback_to_event_bbox", True),
            True,
        )
        self.event_snapshot_live_match_max_age_sec = float(runtime_cfg.get("event_snapshot_live_match_max_age_sec", 2.0))
        # Live detection filters (affect overlay/IDs/realtime counts before event stage)
        self.enable_head_live_filter = _cfg_bool(runtime_cfg.get("enable_head_live_filter", True), True)
        self.head_live_min_conf = float(runtime_cfg.get("head_live_min_conf", 0.55))
        self.head_live_min_area_ratio = float(runtime_cfg.get("head_live_min_area_ratio", 0.0005))
        self.head_live_max_area_ratio = float(runtime_cfg.get("head_live_max_area_ratio", 0.12))
        self.head_live_max_width_ratio = float(runtime_cfg.get("head_live_max_width_ratio", 0.45))
        self.head_live_max_height_ratio = float(runtime_cfg.get("head_live_max_height_ratio", 0.45))
        self.hold_stale_for_head = _cfg_bool(runtime_cfg.get("hold_stale_for_head", False), False)
        self._last_event_reject_log_ts: Dict[Tuple[str, str, str, str], float] = {}
        
        self._motion_prev_gray: Dict[str, np.ndarray] = {}
        self._motion_last_trigger_ts: Dict[str, float] = {}
        self._motion_last_infer_ts: Dict[str, float] = {}
        self._motion_last_ratio: Dict[str, float] = {}
        self._motion_last_mean_diff: Dict[str, float] = {}
        
        # Load specific dwell times from config.
        # Accept both legacy "haircut" and the actual config key "chair".
        dwell_cfg = CONFIG.get("dwell_time", {})
        self.sit_time_haircut = float(
            dwell_cfg.get("haircut", dwell_cfg.get("chair", self.sit_min_sec))
        )
        self.sit_time_wash = float(dwell_cfg.get("wash", self.sit_min_sec))

        # Use runtime-configured vacancy grace for service counters instead of
        # hard-coding 120s, otherwise back-to-back customers are merged.
        self.service_vacant_grace_sec = float(
            runtime_cfg.get("service_vacant_grace_sec", self.vacant_grace_sec)
        )

        configured_stall_timeout = float(runtime_cfg.get("camera_stall_timeout_sec", 8.0))
        # ... (lines 922-1002 unchanged) ...
        self.global_id_manager = GlobalIDManager()
        self.reid_max_prototypes_per_pid = max(
            1, int(runtime_cfg.get("reid_max_prototypes_per_pid", 5))
        )
        self.active_person_memory = ActivePersonMemory(
            max_prototypes_per_pid=self.reid_max_prototypes_per_pid
        )
        
        # Use specific dwell times and robust grace period
        self.haircut_counter = ZoneSessionCounter("HAIRCUT", self.sit_time_haircut, self.service_vacant_grace_sec)
        self.wash_counter = ZoneSessionCounter("WASH", self.sit_time_wash, self.service_vacant_grace_sec)
        self.fsm_enabled = _cfg_bool(runtime_cfg.get("fsm_enabled", True), True)
        self.fsm_return_dwell_sec = max(0.5, float(runtime_cfg.get("fsm_return_dwell_sec", 3.0)))
        self.fsm_close_timeout_sec = max(
            60.0,
            float(runtime_cfg.get("fsm_close_timeout_sec", runtime_cfg.get("service_session_timeout_sec", 600.0))),
        )
        self.fsm_return_window_sec = max(60.0, float(runtime_cfg.get("fsm_return_window_sec", 1200.0)))
        self.service_fsm = ServiceSessionFSM(
            haircut_dwell_sec=self.sit_time_haircut,
            wash_dwell_sec=self.sit_time_wash,
            return_dwell_sec=self.fsm_return_dwell_sec,
            close_timeout_sec=self.fsm_close_timeout_sec,
            return_window_sec=self.fsm_return_window_sec,
            max_recent_transitions=400,
        )
        self.fsm_state_counts: Dict[str, int] = {}
        self.fsm_recent_transitions: List[Dict] = []
        
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
        # Returning customer re-sitting in a chair shortly after a haircut → wash
        self.enable_chair_return_wash = _cfg_bool(runtime_cfg.get("enable_chair_return_wash", True), True)
        self.event_chair_return_wash_window_sec = max(
            60.0, float(runtime_cfg.get("event_chair_return_wash_window_sec", 900.0))
        )
        self.recent_service_identity_events: List[Dict[str, object]] = []
        self.chair_empty_gallery_features: List[Dict[str, object]] = []
        self._chair_zone_occupied_keys: Set[str] = set()
        self._chair_zone_last_vacant_ts: Dict[str, float] = {}
        self.pid_wait_start = {}  # Track entry time to WAIT zone per PID
        self.staff_filtered_total = 0 # Cumulative filtered count
        self.staff_realtime = 0      # Real-time staff identified
        self.camera_people_counts = {} # Realtime people per camera
        self.camera_staff_counts = {}  # Realtime staff per camera
        
        # ReID:
        # - enable_reid controls live embedding for PID/global dedupe.
        # - staff.enable_reid controls staff-gallery matching (separate concern).
        enabled_camera_count = sum(1 for cfg in CAMERAS_CONFIG.values() if bool((cfg or {}).get("enabled", True)))
        default_enable_reid = enabled_camera_count > 1
        self.enable_reid = _cfg_bool(runtime_cfg.get("enable_reid", default_enable_reid), default_enable_reid)
        self.enable_staff_gallery_reid = _cfg_bool(staff_cfg.get("enable_reid", False), False)
        self.enable_role_db_reid = _cfg_bool(runtime_cfg.get("enable_role_db_reid", True), True)
        self.role_db_barber_threshold = min(
            1.0,
            max(
                0.0,
                float(
                    runtime_cfg.get(
                        "role_db_barber_threshold",
                        max(0.75, self.staff_similarity_threshold - 0.02),
                    )
                ),
            ),
        )
        self.role_db_customer_threshold = min(
            1.0,
            max(0.0, float(runtime_cfg.get("role_db_customer_threshold", 0.76))),
        )
        self.role_db_customer_margin = min(
            0.5,
            max(0.0, float(runtime_cfg.get("role_db_customer_margin", 0.04))),
        )
        # Chair-view recovery: top-view chair scenes may contain both barber + seated customer
        # in overlapping boxes; avoid over-tagging both as barber.
        self.enable_chair_customer_recovery = _cfg_bool(
            runtime_cfg.get("enable_chair_customer_recovery", True),
            True,
        )
        self.chair_customer_recovery_margin = min(
            0.25,
            max(0.0, float(runtime_cfg.get("chair_customer_recovery_margin", 0.03))),
        )
        self.chair_customer_recovery_min_area_ratio = min(
            1.0,
            max(0.0, float(runtime_cfg.get("chair_customer_recovery_min_area_ratio", 0.035))),
        )
        self.chair_staff_strict_barber_margin = min(
            0.30,
            max(0.0, float(runtime_cfg.get("chair_staff_strict_barber_margin", 0.08))),
        )
        self.role_db_customer_override_staff = _cfg_bool(
            runtime_cfg.get("role_db_customer_override_staff", True),
            True,
        )
        self.role_db_max_embeddings_per_role = max(
            1,
            int(runtime_cfg.get("role_db_max_embeddings_per_role", 512)),
        )
        self.role_vote_window = max(3, int(runtime_cfg.get("role_vote_window", 15)))
        self.role_vote_min_samples = max(1, int(runtime_cfg.get("role_vote_min_samples", 5)))
        self.role_vote_min_ratio = min(
            1.0,
            max(0.34, float(runtime_cfg.get("role_vote_min_ratio", 0.60))),
        )
        self.role_unknown_min_conf = min(
            1.0,
            max(0.0, float(runtime_cfg.get("role_unknown_min_conf", 0.35))),
        )
        self.require_stable_customer_role_for_service = _cfg_bool(
            runtime_cfg.get("require_stable_customer_role_for_service", False),
            False,
        )
        self.service_session_timeout_sec = max(
            60.0, float(runtime_cfg.get("service_session_timeout_sec", 600.0))
        )
        self.pid_role_votes: Dict[int, deque] = {}
        self.pid_role_last_ts: Dict[int, float] = {}
        self.role_vote_total_frames = 0
        self.role_vote_unknown_frames = 0
        self.reid_encoder = None
        self.event_reid_encoder = None
        if self.enable_reid or self.enable_staff_gallery_reid or self.enable_role_db_reid:
            try:
                # MPS support for ReID (ResNet50) can be flaky on some MacOS versions, fallback to CPU if needed
                reid_device = self.yolo_device if self.yolo_device != "mps" else "cpu"
                self.reid_encoder = ReIDEncoder(device=reid_device)
            except Exception as e:
                logger.error(f"Failed to init ReID encoder: {e}")
                self.enable_reid = False
                self.enable_staff_gallery_reid = False
                self.enable_role_db_reid = False
        
        self.staff_gallery = None
        if self.enable_staff_gallery_reid and self.reid_encoder is not None:
            self.staff_gallery = StaffGallery(str(PATHS_CONFIG.get("staff_db", "data/staff_gallery/staff_db.json")))
        self.role_gallery = None
        if self.enable_role_db_reid and self.reid_encoder is not None:
            role_db_path = str(PATHS_CONFIG.get("role_db", "data/staff_gallery/barber_customer_db.json"))
            role_gallery = RoleGallery(
                role_db_path,
                max_embeddings_per_role=self.role_db_max_embeddings_per_role,
            )
            if role_gallery.has_data():
                self.role_gallery = role_gallery
            else:
                self.enable_role_db_reid = False
                logger.warning(f"Role DB is empty or invalid, role classifier disabled: {role_db_path}")
        logger.info(
            "ReID settings: "
            f"live_pid_reid={self.enable_reid}, "
            f"staff_gallery_reid={self.enable_staff_gallery_reid}, "
            f"role_db_reid={self.enable_role_db_reid}, "
            f"encoder={'ready' if self.reid_encoder is not None else 'off'}"
        )

        self.enable_event_reid_dedupe = _cfg_bool(runtime_cfg.get("enable_event_reid_dedupe", True), True)
        self.haircut_event_dedupe_window_sec = float(
            runtime_cfg.get("haircut_event_dedupe_window_sec", max(300.0, self.sit_time_haircut * 2.0))
        )
        self.haircut_event_dedupe_similarity = float(
            runtime_cfg.get("haircut_event_dedupe_similarity", 0.82)
        )
        self._load_chair_empty_gallery_features()

        self._apply_runtime_settings(CONFIG.get("runtime_tunable", {}))
        
        # Optional startup restore of dashboard state (disabled by default).
        if os.path.exists(DASHBOARD_STATE_FILE):
             try:
                 with open(DASHBOARD_STATE_FILE, "r", encoding="utf-8") as f:
                     state = json.load(f)
                 now_day = datetime.now().strftime("%Y-%m-%d")
                 now_ts = time.time()
                 state_day = ""
                 state_age_sec = -1.0
                 state_ts = 0.0
                 try:
                     state_ts = float(state.get("timestamp", 0.0) or 0.0)
                     if state_ts > 0:
                         state_day = datetime.fromtimestamp(state_ts).strftime("%Y-%m-%d")
                         state_age_sec = max(0.0, float(now_ts - state_ts))
                 except Exception:
                     state_day = ""
                     state_age_sec = -1.0
                 if not state_day:
                     try:
                         mtime_ts = float(Path(DASHBOARD_STATE_FILE).stat().st_mtime)
                         state_day = datetime.fromtimestamp(mtime_ts).strftime("%Y-%m-%d")
                         state_age_sec = max(0.0, float(now_ts - mtime_ts))
                     except Exception:
                         state_day = ""
                         state_age_sec = -1.0

                 summary_state = state.get("summary", {}) if isinstance(state.get("summary", {}), dict) else {}
                 status_state = state.get("status", {}) if isinstance(state.get("status", {}), dict) else {}

                 if not self.restore_dashboard_state_on_startup:
                     logger.info("Skip dashboard restore on startup (restore_dashboard_state_on_startup=false)")
                 elif state_day != now_day:
                     logger.info(
                         f"Skip stale dashboard restore: state_day={state_day or '-'} current_day={now_day}"
                     )
                 elif (
                     self.restore_dashboard_state_max_age_sec > 0.0
                     and state_age_sec >= 0.0
                     and state_age_sec > self.restore_dashboard_state_max_age_sec
                 ):
                     logger.info(
                         f"Skip stale dashboard restore: state_age_sec={state_age_sec:.1f} > "
                         f"restore_dashboard_state_max_age_sec={self.restore_dashboard_state_max_age_sec:.1f}"
                     )
                 else:
                     restored_haircuts = int(
                         summary_state.get(
                             "haircuts",
                             status_state.get("haircut_count", state.get("haircut_count", 0)),
                         )
                         or 0
                     )
                     restored_washes = int(
                         summary_state.get(
                             "washes",
                             status_state.get("wash_count", state.get("wash_count", 0)),
                         )
                         or 0
                     )
                     restored_verified = int(
                         summary_state.get("verified", state.get("verified_count", 0)) or 0
                     )
                     self.haircut_counter.total_count = restored_haircuts
                     self.wash_counter.total_count = restored_washes
                     self.verified_events_total = restored_verified

                     if "haircut_zone_total" in state:
                         self.haircut_counter.zone_total = state["haircut_zone_total"]
                     if "wash_zone_total" in state:
                         self.wash_counter.zone_total = state["wash_zone_total"]
                     self.service_fsm.haircut_total = int(self.haircut_counter.total_count)
                     self.service_fsm.wash_total = int(self.wash_counter.total_count)
                     self.service_fsm.haircut_zone_total = dict(self.haircut_counter.zone_total)
                     self.service_fsm.wash_zone_total = dict(self.wash_counter.zone_total)
                     logger.info(
                         f"Restored startup dashboard state ({state_day} age={state_age_sec:.1f}s): "
                         f"haircuts={self.haircut_counter.total_count}, washes={self.wash_counter.total_count}, "
                         f"verified={self.verified_events_total}"
                     )
             except Exception as e:
                 logger.error(f"Failed to restore dashboard state: {e}")

        self.worker.start()
        self._setup_cameras()
        
    def _apply_runtime_settings(self, settings: Dict):
        """Apply runtime-tunable parameters from controller."""
        try:
            role_db_reload = False
            def _as_bool(v, default=False):
                if v is None:
                    return bool(default)
                if isinstance(v, str):
                    return v.strip().lower() in {"1", "true", "yes", "on"}
                return bool(v)
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
            if "target_fps" in settings:
                self.target_fps = max(1, int(settings["target_fps"]))
            if "dashboard_ui_frame_interval_sec" in settings:
                self.dashboard_ui_frame_interval_sec = max(
                    0.03, float(settings["dashboard_ui_frame_interval_sec"])
                )
            if "dashboard_state_min_interval_sec" in settings:
                self.dashboard_state_min_interval_sec = max(
                    0.20, float(settings["dashboard_state_min_interval_sec"])
                )
            if "gc_collect_interval_sec" in settings:
                self.gc_collect_interval_sec = max(
                    10.0, float(settings["gc_collect_interval_sec"])
                )
            if "summary_log_interval_sec" in settings:
                self.summary_log_interval_sec = max(
                    5.0, float(settings["summary_log_interval_sec"])
                )
            if "sit_min_sec" in settings:
                self.sit_min_sec = float(settings["sit_min_sec"])
                self.sit_time_haircut = self.sit_min_sec
                self.sit_time_wash = self.sit_min_sec
                self.haircut_counter.sit_min_sec = self.sit_min_sec
                self.wash_counter.sit_min_sec = self.sit_min_sec
                self.service_fsm.haircut_dwell_sec = self.sit_time_haircut
                self.service_fsm.wash_dwell_sec = self.sit_time_wash
            if "vacant_grace_sec" in settings:
                self.vacant_grace_sec = float(settings["vacant_grace_sec"])
                self.service_vacant_grace_sec = self.vacant_grace_sec
                self.haircut_counter.vacant_grace_sec = self.vacant_grace_sec
                self.wash_counter.vacant_grace_sec = self.vacant_grace_sec
            if "service_vacant_grace_sec" in settings:
                self.service_vacant_grace_sec = max(0.1, float(settings["service_vacant_grace_sec"]))
                self.haircut_counter.vacant_grace_sec = self.service_vacant_grace_sec
                self.wash_counter.vacant_grace_sec = self.service_vacant_grace_sec
            if "fsm_enabled" in settings:
                self.fsm_enabled = _as_bool(settings["fsm_enabled"], True)
            if "fsm_return_dwell_sec" in settings:
                self.fsm_return_dwell_sec = max(0.5, float(settings["fsm_return_dwell_sec"]))
                self.service_fsm.return_dwell_sec = self.fsm_return_dwell_sec
            if "fsm_close_timeout_sec" in settings:
                self.fsm_close_timeout_sec = max(60.0, float(settings["fsm_close_timeout_sec"]))
                self.service_fsm.close_timeout_sec = self.fsm_close_timeout_sec
            if "fsm_return_window_sec" in settings:
                self.fsm_return_window_sec = max(60.0, float(settings["fsm_return_window_sec"]))
                self.service_fsm.return_window_sec = self.fsm_return_window_sec
            if "tier2_trigger_dwell_sec" in settings:
                self.tier2_trigger_dwell_sec = max(0.1, float(settings["tier2_trigger_dwell_sec"]))
            if "tier2_trigger_cooldown_sec" in settings:
                self.tier2_trigger_cooldown_sec = max(0.1, float(settings["tier2_trigger_cooldown_sec"]))
            if "tier2_clip_duration_sec" in settings:
                self.tier2_clip_duration_sec = max(1.0, float(settings["tier2_clip_duration_sec"]))
            if "enable_business_hours_guard" in settings:
                self.enable_business_hours_guard = _as_bool(settings["enable_business_hours_guard"], self.enable_business_hours_guard)
            if "business_hours_start" in settings:
                self.business_hours_start = str(settings["business_hours_start"])
            if "business_hours_end" in settings:
                self.business_hours_end = str(settings["business_hours_end"])
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
                    self._configured_inference_mode = mode
                    self.inference_mode = mode
                    if mode != "always":
                        self._auto_motion_gate_active = False
            if "auto_motion_gate_enabled" in settings:
                self.auto_motion_gate_enabled = _as_bool(
                    settings["auto_motion_gate_enabled"],
                    self.auto_motion_gate_enabled,
                )
            if "auto_motion_gate_cpu_threshold" in settings:
                self.auto_motion_gate_cpu_threshold = min(
                    99.0, max(10.0, float(settings["auto_motion_gate_cpu_threshold"]))
                )
            if "auto_motion_gate_mem_threshold" in settings:
                self.auto_motion_gate_mem_threshold = min(
                    99.0, max(10.0, float(settings["auto_motion_gate_mem_threshold"]))
                )
            if "auto_motion_gate_recover_sec" in settings:
                self.auto_motion_gate_recover_sec = max(
                    5.0, float(settings["auto_motion_gate_recover_sec"])
                )
            if "auto_motion_gate_min_hold_sec" in settings:
                self.auto_motion_gate_min_hold_sec = max(
                    3.0, float(settings["auto_motion_gate_min_hold_sec"])
                )
            if "count_mode" in settings:
                self.service_mode = self._normalize_count_mode(settings.get("count_mode"))
            if "service_mode" in settings:
                self.service_mode = self._normalize_count_mode(settings.get("service_mode"))
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
                self.enable_staff_uniform = _as_bool(settings["enable_staff_uniform"], True)
                self._configure_discovery_class_schema()
            if "staff_uniform_iou_threshold" in settings:
                self.staff_uniform_iou_threshold = max(0.0, min(1.0, float(settings["staff_uniform_iou_threshold"])))
            if "enable_role_snapshots" in settings:
                self.enable_role_snapshots = _as_bool(settings["enable_role_snapshots"], True)
            if "role_snapshot_cooldown_sec" in settings:
                self.role_snapshot_cooldown_sec = max(1.0, float(settings["role_snapshot_cooldown_sec"]))
            if "role_snapshot_customer_interval_sec" in settings:
                self.role_snapshot_customer_interval_sec = max(
                    10.0, float(settings["role_snapshot_customer_interval_sec"])
                )
            if "role_snapshot_barber_interval_sec" in settings:
                self.role_snapshot_barber_interval_sec = max(
                    10.0, float(settings["role_snapshot_barber_interval_sec"])
                )
            if "role_snapshot_wash_interval_sec" in settings:
                self.role_snapshot_wash_interval_sec = max(
                    10.0, float(settings["role_snapshot_wash_interval_sec"])
                )
            if "role_snapshot_unknown_interval_sec" in settings:
                self.role_snapshot_unknown_interval_sec = max(
                    10.0, float(settings["role_snapshot_unknown_interval_sec"])
                )
            if "role_snapshot_enable_unknown" in settings:
                self.role_snapshot_enable_unknown = _as_bool(
                    settings["role_snapshot_enable_unknown"],
                    self.role_snapshot_enable_unknown,
                )
                if self.role_snapshot_enable_unknown and self.role_snapshot_unknown_max_per_camera_day > 0:
                    try:
                        self.unknown_snapshot_dir.mkdir(parents=True, exist_ok=True)
                    except Exception:
                        pass
            if "role_snapshot_thumb_min_diff" in settings:
                self.role_snapshot_thumb_min_diff = max(1.0, float(settings["role_snapshot_thumb_min_diff"]))
            if "role_snapshot_min_conf" in settings:
                self.role_snapshot_min_conf = min(1.0, max(0.0, float(settings["role_snapshot_min_conf"])))
            if "role_snapshot_min_area_ratio" in settings:
                self.role_snapshot_min_area_ratio = min(1.0, max(0.0, float(settings["role_snapshot_min_area_ratio"])))
            if "role_snapshot_require_customer_chair_zone" in settings:
                self.role_snapshot_require_customer_chair_zone = _as_bool(
                    settings["role_snapshot_require_customer_chair_zone"], True
                )
            if "role_snapshot_customer_max_per_camera_day" in settings:
                self.role_snapshot_customer_max_per_camera_day = max(
                    1, int(settings["role_snapshot_customer_max_per_camera_day"])
                )
            if "role_snapshot_barber_max_per_camera_day" in settings:
                self.role_snapshot_barber_max_per_camera_day = max(
                    1, int(settings["role_snapshot_barber_max_per_camera_day"])
                )
            if "role_snapshot_wash_max_per_camera_day" in settings:
                self.role_snapshot_wash_max_per_camera_day = max(
                    1, int(settings["role_snapshot_wash_max_per_camera_day"])
                )
            if "role_snapshot_unknown_max_per_camera_day" in settings:
                self.role_snapshot_unknown_max_per_camera_day = max(
                    0, int(settings["role_snapshot_unknown_max_per_camera_day"])
                )
                if self.role_snapshot_enable_unknown and self.role_snapshot_unknown_max_per_camera_day > 0:
                    try:
                        self.unknown_snapshot_dir.mkdir(parents=True, exist_ok=True)
                    except Exception:
                        pass
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
            if "staff_event_similarity_threshold" in settings:
                self.staff_event_similarity_threshold = float(settings["staff_event_similarity_threshold"])
            if "reid_active_expire_sec" in settings:
                self.reid_active_expire_sec = float(settings["reid_active_expire_sec"])
            if "reid_match_window_sec" in settings:
                self.reid_match_window_sec = float(settings["reid_match_window_sec"])
            if "cross_camera_dedupe_window_sec" in settings:
                self.cross_camera_dedupe_window_sec = float(settings["cross_camera_dedupe_window_sec"])
            if "cross_camera_dedupe_similarity" in settings:
                self.cross_camera_dedupe_similarity = float(settings["cross_camera_dedupe_similarity"])
            if "reid_merge_for_counts" in settings:
                self.reid_merge_for_counts = _as_bool(settings["reid_merge_for_counts"], True)
            if "enable_reid" in settings:
                self.enable_reid = _as_bool(settings["enable_reid"], self.enable_reid)
            if "exclude_staff_from_counts" in settings:
                self.exclude_staff_from_counts = _as_bool(settings["exclude_staff_from_counts"], True)
            if "enable_role_db_reid" in settings:
                self.enable_role_db_reid = _as_bool(settings["enable_role_db_reid"], self.enable_role_db_reid)
                role_db_reload = True
            if "role_db_barber_threshold" in settings:
                self.role_db_barber_threshold = min(
                    1.0,
                    max(0.0, float(settings["role_db_barber_threshold"])),
                )
            if "role_db_customer_threshold" in settings:
                self.role_db_customer_threshold = min(
                    1.0,
                    max(0.0, float(settings["role_db_customer_threshold"])),
                )
            if "role_db_customer_margin" in settings:
                self.role_db_customer_margin = min(
                    0.5,
                    max(0.0, float(settings["role_db_customer_margin"])),
                )
            if "enable_chair_customer_recovery" in settings:
                self.enable_chair_customer_recovery = _as_bool(
                    settings["enable_chair_customer_recovery"],
                    self.enable_chair_customer_recovery,
                )
            if "chair_customer_recovery_margin" in settings:
                self.chair_customer_recovery_margin = min(
                    0.25,
                    max(0.0, float(settings["chair_customer_recovery_margin"])),
                )
            if "chair_customer_recovery_min_area_ratio" in settings:
                self.chair_customer_recovery_min_area_ratio = min(
                    1.0,
                    max(0.0, float(settings["chair_customer_recovery_min_area_ratio"])),
                )
            if "chair_staff_strict_barber_margin" in settings:
                self.chair_staff_strict_barber_margin = min(
                    0.30,
                    max(0.0, float(settings["chair_staff_strict_barber_margin"])),
                )
            if "role_db_customer_override_staff" in settings:
                self.role_db_customer_override_staff = _as_bool(
                    settings["role_db_customer_override_staff"],
                    self.role_db_customer_override_staff,
                )
            if "role_db_max_embeddings_per_role" in settings:
                self.role_db_max_embeddings_per_role = max(
                    1,
                    int(settings["role_db_max_embeddings_per_role"]),
                )
                role_db_reload = True
            if "role_vote_window" in settings:
                role_vote_window = max(3, int(settings["role_vote_window"]))
                if role_vote_window != self.role_vote_window:
                    self.role_vote_window = role_vote_window
                    for pid, dq in list(self.pid_role_votes.items()):
                        ndq = deque(dq, maxlen=self.role_vote_window)
                        self.pid_role_votes[pid] = ndq
            if "role_vote_min_samples" in settings:
                self.role_vote_min_samples = max(1, int(settings["role_vote_min_samples"]))
            if "role_vote_min_ratio" in settings:
                self.role_vote_min_ratio = min(1.0, max(0.34, float(settings["role_vote_min_ratio"])))
            if "role_unknown_min_conf" in settings:
                self.role_unknown_min_conf = min(1.0, max(0.0, float(settings["role_unknown_min_conf"])))
            if "require_stable_customer_role_for_service" in settings:
                self.require_stable_customer_role_for_service = _as_bool(
                    settings["require_stable_customer_role_for_service"],
                    False,
                )
            if "service_session_timeout_sec" in settings:
                self.service_session_timeout_sec = max(60.0, float(settings["service_session_timeout_sec"]))
                if "fsm_close_timeout_sec" not in settings:
                    self.fsm_close_timeout_sec = self.service_session_timeout_sec
                    self.service_fsm.close_timeout_sec = self.fsm_close_timeout_sec
            if "reid_max_prototypes_per_pid" in settings:
                self.reid_max_prototypes_per_pid = max(1, int(settings["reid_max_prototypes_per_pid"]))
                self.active_person_memory.max_prototypes_per_pid = self.reid_max_prototypes_per_pid
            if "enable_head_size_guard" in settings:
                self.enable_head_size_guard = _as_bool(settings["enable_head_size_guard"], False)
            if "head_max_width_ratio" in settings:
                self.head_max_width_ratio = min(max(float(settings["head_max_width_ratio"]), 0.0), 1.0)
            if "head_max_height_ratio" in settings:
                self.head_max_height_ratio = min(max(float(settings["head_max_height_ratio"]), 0.0), 1.0)
            if "head_max_area_ratio" in settings:
                self.head_max_area_ratio = min(max(float(settings["head_max_area_ratio"]), 0.0), 1.0)
            if "event_require_fresh_detection" in settings:
                self.event_require_fresh_detection = _as_bool(settings["event_require_fresh_detection"], True)
            if "event_reject_stale_hold" in settings:
                self.event_reject_stale_hold = _as_bool(settings["event_reject_stale_hold"], True)
            if "event_snapshot_require_live_match" in settings:
                self.event_snapshot_require_live_match = _as_bool(settings["event_snapshot_require_live_match"], True)
            if "event_snapshot_fallback_to_event_bbox" in settings:
                self.event_snapshot_fallback_to_event_bbox = _as_bool(
                    settings["event_snapshot_fallback_to_event_bbox"],
                    True,
                )
            if "event_max_detection_age_sec" in settings:
                self.event_max_detection_age_sec = max(0.1, float(settings["event_max_detection_age_sec"]))
            if "event_snapshot_live_match_max_age_sec" in settings:
                self.event_snapshot_live_match_max_age_sec = max(0.1, float(settings["event_snapshot_live_match_max_age_sec"]))
            if "enable_event_feedback_autocopy" in settings:
                self.enable_event_feedback_autocopy = _as_bool(
                    settings["enable_event_feedback_autocopy"],
                    self.enable_event_feedback_autocopy,
                )
            if "enable_no_haircut_feedback_autocopy" in settings:
                self.enable_no_haircut_feedback_autocopy = _as_bool(
                    settings["enable_no_haircut_feedback_autocopy"],
                    self.enable_no_haircut_feedback_autocopy,
                )
            if "no_haircut_feedback_cooldown_sec" in settings:
                self.no_haircut_feedback_cooldown_sec = max(
                    1.0,
                    float(settings["no_haircut_feedback_cooldown_sec"]),
                )
            if "haircut_count_zones" in settings:
                self.haircut_count_zones = self._parse_haircut_count_zones(
                    settings.get("haircut_count_zones", {})
                )
            chair_cls_reload = False
            chair_empty_reload = False
            chair_cls_autotrain_changed = False
            if "enable_chair_service_classifier" in settings:
                self.enable_chair_service_classifier = _as_bool(
                    settings["enable_chair_service_classifier"],
                    self.enable_chair_service_classifier,
                )
                chair_cls_reload = True
            if "chair_service_classifier_model" in settings:
                self.chair_service_classifier_model = str(settings["chair_service_classifier_model"])
                chair_cls_reload = True
            if "chair_service_classifier_positive_labels" in settings:
                labels = _normalize_label_set(settings["chair_service_classifier_positive_labels"])
                self.chair_service_classifier_positive_labels = labels or {"haircut"}
            if "chair_service_classifier_min_conf" in settings:
                self.chair_service_classifier_min_conf = min(
                    1.0,
                    max(0.0, float(settings["chair_service_classifier_min_conf"])),
                )
            if "chair_service_classifier_imgsz" in settings:
                self.chair_service_classifier_imgsz = max(
                    96,
                    int(settings["chair_service_classifier_imgsz"]),
                )
            if "chair_service_autotrain_enabled" in settings:
                self.chair_service_autotrain_enabled = _as_bool(
                    settings["chair_service_autotrain_enabled"],
                    self.chair_service_autotrain_enabled,
                )
                chair_cls_autotrain_changed = True
            if "chair_service_autotrain_interval_hours" in settings:
                self.chair_service_autotrain_interval_hours = max(
                    0.25,
                    float(settings["chair_service_autotrain_interval_hours"]),
                )
                chair_cls_autotrain_changed = True
            if "chair_service_autotrain_min_positive" in settings:
                self.chair_service_autotrain_min_positive = max(
                    1, int(settings["chair_service_autotrain_min_positive"])
                )
                chair_cls_autotrain_changed = True
            if "chair_service_autotrain_min_negative" in settings:
                self.chair_service_autotrain_min_negative = max(
                    1, int(settings["chair_service_autotrain_min_negative"])
                )
                chair_cls_autotrain_changed = True
            if "chair_service_autotrain_positive_dir" in settings:
                self.chair_service_autotrain_positive_dir = str(
                    settings["chair_service_autotrain_positive_dir"] or ""
                ).strip()
                chair_cls_autotrain_changed = True
            if "chair_service_autotrain_negative_dirs" in settings:
                raw_neg = settings.get("chair_service_autotrain_negative_dirs")
                if isinstance(raw_neg, str):
                    neg_dirs = [x.strip() for x in raw_neg.split(",") if x.strip()]
                elif isinstance(raw_neg, (list, tuple, set)):
                    neg_dirs = [str(x).strip() for x in raw_neg if str(x).strip()]
                else:
                    neg_dirs = []
                if neg_dirs:
                    self.chair_service_autotrain_negative_dirs = neg_dirs
                    chair_cls_autotrain_changed = True
            if "chair_service_autotrain_epochs" in settings:
                self.chair_service_autotrain_epochs = max(
                    1, int(settings["chair_service_autotrain_epochs"])
                )
                chair_cls_autotrain_changed = True
            if "chair_service_autotrain_batch" in settings:
                self.chair_service_autotrain_batch = max(
                    1, int(settings["chair_service_autotrain_batch"])
                )
                chair_cls_autotrain_changed = True
            if "chair_service_autotrain_imgsz" in settings:
                self.chair_service_autotrain_imgsz = max(
                    96, int(settings["chair_service_autotrain_imgsz"])
                )
                chair_cls_autotrain_changed = True
            if "chair_service_autotrain_patience" in settings:
                self.chair_service_autotrain_patience = max(
                    1, int(settings["chair_service_autotrain_patience"])
                )
                chair_cls_autotrain_changed = True
            if "chair_service_autotrain_workers" in settings:
                self.chair_service_autotrain_workers = max(
                    0, int(settings["chair_service_autotrain_workers"])
                )
                chair_cls_autotrain_changed = True
            if "chair_service_autotrain_train_split" in settings:
                self.chair_service_autotrain_train_split = min(
                    0.99,
                    max(0.01, float(settings["chair_service_autotrain_train_split"])),
                )
                chair_cls_autotrain_changed = True
            if "chair_service_autotrain_device" in settings:
                self.chair_service_autotrain_device = str(
                    settings["chair_service_autotrain_device"] or ""
                ).strip() or "auto"
                chair_cls_autotrain_changed = True
            if "chair_service_autotrain_timeout_min" in settings:
                self.chair_service_autotrain_timeout_min = max(
                    10, int(settings["chair_service_autotrain_timeout_min"])
                )
                chair_cls_autotrain_changed = True
            if "enable_chair_empty_guard" in settings:
                self.enable_chair_empty_guard = _as_bool(
                    settings["enable_chair_empty_guard"],
                    self.enable_chair_empty_guard,
                )
                chair_empty_reload = True
            if "chair_empty_gallery_dir" in settings:
                self.chair_empty_gallery_dir = str(settings["chair_empty_gallery_dir"] or "").strip()
                chair_empty_reload = True
            if "chair_empty_similarity_threshold" in settings:
                self.chair_empty_similarity_threshold = min(
                    1.0,
                    max(0.0, float(settings["chair_empty_similarity_threshold"])),
                )
            if "chair_empty_hist_weight" in settings:
                self.chair_empty_hist_weight = min(
                    1.0,
                    max(0.0, float(settings["chair_empty_hist_weight"])),
                )
                chair_empty_reload = True
            if "event_identity_similarity_threshold" in settings:
                self.event_identity_similarity_threshold = min(
                    1.0,
                    max(0.0, float(settings["event_identity_similarity_threshold"])),
                )
            if "event_same_customer_block_sec" in settings:
                self.event_same_customer_block_sec = max(
                    10.0,
                    float(settings["event_same_customer_block_sec"]),
                )
            if "event_same_seat_cooldown_sec" in settings:
                self.event_same_seat_cooldown_sec = max(
                    0.0,
                    float(settings["event_same_seat_cooldown_sec"]),
                )
            if "event_wash_return_block_sec" in settings:
                self.event_wash_return_block_sec = max(
                    10.0,
                    float(settings["event_wash_return_block_sec"]),
                )
            if "require_chair_vacant_between_haircuts" in settings:
                self.require_chair_vacant_between_haircuts = _as_bool(
                    settings["require_chair_vacant_between_haircuts"],
                    self.require_chair_vacant_between_haircuts,
                )
            if chair_cls_reload:
                self._load_chair_service_classifier()
            if chair_cls_autotrain_changed:
                self._chair_service_autotrain_next_check_ts = 0.0
                self._save_chair_service_autotrain_state()
            if self.running and self.chair_service_autotrain_enabled:
                self._start_chair_service_autotrain_thread()
            if chair_empty_reload:
                self._load_chair_empty_gallery_features()
            if "event_head_min_conf" in settings:
                self.event_head_min_conf = min(max(float(settings["event_head_min_conf"]), 0.0), 1.0)
            if "event_person_min_conf" in settings:
                self.event_person_min_conf = min(max(float(settings["event_person_min_conf"]), 0.0), 1.0)
            if "event_head_min_area_ratio" in settings:
                self.event_head_min_area_ratio = min(max(float(settings["event_head_min_area_ratio"]), 0.0), 1.0)
            if "event_head_max_area_ratio" in settings:
                self.event_head_max_area_ratio = min(max(float(settings["event_head_max_area_ratio"]), 0.0), 1.0)
            if "event_head_max_width_ratio" in settings:
                self.event_head_max_width_ratio = min(max(float(settings["event_head_max_width_ratio"]), 0.0), 1.0)
            if "event_head_max_height_ratio" in settings:
                self.event_head_max_height_ratio = min(max(float(settings["event_head_max_height_ratio"]), 0.0), 1.0)
            if "enable_head_live_filter" in settings:
                self.enable_head_live_filter = _as_bool(settings["enable_head_live_filter"], True)
            if "hold_stale_for_head" in settings:
                self.hold_stale_for_head = _as_bool(settings["hold_stale_for_head"], False)
            if "head_live_min_conf" in settings:
                self.head_live_min_conf = min(max(float(settings["head_live_min_conf"]), 0.0), 1.0)
            if "head_live_min_area_ratio" in settings:
                self.head_live_min_area_ratio = min(max(float(settings["head_live_min_area_ratio"]), 0.0), 1.0)
            if "head_live_max_area_ratio" in settings:
                self.head_live_max_area_ratio = min(max(float(settings["head_live_max_area_ratio"]), 0.0), 1.0)
            if "head_live_max_width_ratio" in settings:
                self.head_live_max_width_ratio = min(max(float(settings["head_live_max_width_ratio"]), 0.0), 1.0)
            if "head_live_max_height_ratio" in settings:
                self.head_live_max_height_ratio = min(max(float(settings["head_live_max_height_ratio"]), 0.0), 1.0)
            if role_db_reload:
                if self.enable_role_db_reid and self.reid_encoder is not None:
                    role_db_path = str(PATHS_CONFIG.get("role_db", "data/staff_gallery/barber_customer_db.json"))
                    role_gallery = RoleGallery(
                        role_db_path,
                        max_embeddings_per_role=self.role_db_max_embeddings_per_role,
                    )
                    if role_gallery.has_data():
                        self.role_gallery = role_gallery
                    else:
                        self.role_gallery = None
                        self.enable_role_db_reid = False
                        logger.warning(
                            f"Role DB reload failed (empty/invalid), role classifier disabled: {role_db_path}"
                        )
                else:
                    self.role_gallery = None
            logger.info(f"Applied runtime settings: {settings}")
        except Exception as e:
            logger.error(f"Failed applying settings: {e}")

    def _capture_event_snapshot(self, event) -> str:
        """Capture cropped snapshot for an event using latest frame and latest track bbox."""
        try:
            cam_name = event.camera
            event_bbox = None
            if isinstance(getattr(event, "metadata", None), dict):
                event_bbox = event.metadata.get("bbox")
            matched = None
            if self.event_snapshot_require_live_match:
                matched = self._find_live_match_for_event(event)
                if matched is None:
                    if not (self.event_snapshot_fallback_to_event_bbox and event_bbox):
                        logger.info(
                            f"Skip snapshot (no live person evidence): type={getattr(event.event_type, 'value', event.event_type)} "
                            f"cam={cam_name} pid={getattr(event, 'person_id', 0)} zone={getattr(event, 'zone_name', '')}"
                        )
                        return ""
                    logger.info(
                        f"Snapshot fallback to event bbox: type={getattr(event.event_type, 'value', event.event_type)} "
                        f"cam={cam_name} pid={getattr(event, 'person_id', 0)} zone={getattr(event, 'zone_name', '')}"
                    )
            with self.customers_lock:
                frame = self.latest_frames.get(cam_name)
                frame = None if frame is None else frame.copy()
            if frame is None:
                return ""
            h, w = frame.shape[:2]
            bbox = None
            gid = 0
            det_type = ""
            if isinstance(event.metadata, dict):
                bbox = event.metadata.get("bbox")
                gid = int(event.metadata.get("gid", 0))
                det_type = str(event.metadata.get("det_type", "") or "")
                if matched and isinstance(matched, dict) and matched.get("bbox"):
                    bbox = matched.get("bbox")
                    det_type = str(matched.get("det_type", det_type) or det_type)
            if not bbox:
                return ""
            x1f, y1f, x2f, y2f = [float(v) for v in bbox]
            orig_x1, orig_y1, orig_x2, orig_y2 = [int(v) for v in [x1f, y1f, x2f, y2f]]
            bw = max(1.0, x2f - x1f)
            bh = max(1.0, y2f - y1f)

            if det_type == "head":
                # Head detections can still be oversized; bias the snapshot to the upper section.
                focus_x1 = x1f - max(SNAPSHOT_PAD * 0.3, bw * 0.10)
                focus_x2 = x2f + max(SNAPSHOT_PAD * 0.3, bw * 0.10)
                focus_y1 = y1f - max(SNAPSHOT_PAD * 0.6, bh * 0.08)
                focus_y2 = y1f + max(72.0, min(bh * 0.38, bh - 8.0)) + SNAPSHOT_PAD * 0.35
            else:
                # For person boxes, still bias upward so snapshots show head/torso instead of feet.
                focus_x1 = x1f - SNAPSHOT_PAD
                focus_x2 = x2f + SNAPSHOT_PAD
                focus_y1 = y1f - SNAPSHOT_PAD
                focus_y2 = y1f + max(96.0, min(bh * 0.68, bh - 4.0)) + SNAPSHOT_PAD * 0.25

            x1 = max(0, int(round(focus_x1)))
            y1 = max(0, int(round(focus_y1)))
            x2 = min(w - 1, int(round(focus_x2)))
            y2 = min(h - 1, int(round(focus_y2)))
            if x2 <= x1 or y2 <= y1:
                return ""

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return ""
            # Draw original bbox inside padded crop to make audits easier.
            try:
                rx1 = max(0, orig_x1 - x1)
                ry1 = max(0, orig_y1 - y1)
                rx2 = min(crop.shape[1] - 1, orig_x2 - x1)
                ry2 = min(crop.shape[0] - 1, orig_y2 - y1)
                cv2.rectangle(crop, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
                if isinstance(event.metadata, dict):
                    det_type = str(event.metadata.get("det_type", ""))
                    det_conf = float(event.metadata.get("det_conf", event.metadata.get("conf", 0.0)) or 0.0)
                    label = f"{event.event_type.value}:{det_type} {det_conf:.2f}".strip()
                    cv2.putText(
                        crop,
                        label,
                        (6, max(18, ry1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
            except Exception:
                pass
            out = self._snapshot_path(event.event_type.value.upper(), cam_name, event.person_id, gid=gid)
            out.parent.mkdir(parents=True, exist_ok=True)
            ok = self._cv2_imwrite_unicode_safe(out, crop)
            if not ok:
                logger.warning(f"Snapshot write failed: {out}")
            return str(out) if ok else ""
        except Exception as e:
            logger.debug(f"Failed to capture event snapshot: {e}")
            return ""

    def _log_event_reject(self, event_name: str, d: dict, zone: str, reason: str):
        try:
            now = time.time()
            cam = str(d.get("cam", ""))
            key = (event_name, cam, str(zone), str(reason))
            last = float(self._last_event_reject_log_ts.get(key, 0.0))
            if (now - last) < 5.0:
                return
            self._last_event_reject_log_ts[key] = now
            logger.info(
                f"Reject {event_name} event candidate: cam={cam} zone={zone} "
                f"pid={int(d.get('pid', 0))} gid={int(d.get('gid', 0))} "
                f"type={d.get('det_type')} conf={float(d.get('conf', 0.0)):.2f} reason={reason}"
            )
            self._maybe_save_rejected_haircut_feedback(event_name, d, zone, reason)
        except Exception:
            pass

    def _maybe_save_rejected_haircut_feedback(self, event_name: str, d: dict, zone: str, reason: str):
        if not self.enable_no_haircut_feedback_autocopy:
            return
        if str(event_name or "").strip().lower() != "haircut":
            return
        reason_l = str(reason or "").strip().lower()
        if (
            "chair_cls_negative" not in reason_l
            and "chair_cls_conf_low" not in reason_l
            and "chair_cls_no_" not in reason_l
        ):
            return
        cam = str(d.get("cam", d.get("camera", "")) or "").strip()
        if not cam:
            return
        bbox = d.get("bbox") or []
        if len(bbox) != 4:
            return
        pid = int(d.get("pid", 0) or 0)
        gid = int(d.get("gid", 0) or 0)
        key = f"{cam}:{gid if gid > 0 else pid}:{str(zone or '')}:{reason_l[:64]}"
        now = time.time()
        prev = float(self._last_no_haircut_feedback_ts.get(key, 0.0))
        if (now - prev) < float(self.no_haircut_feedback_cooldown_sec):
            return
        with self.customers_lock:
            frame = self.latest_frames.get(cam)
            frame = None if frame is None else frame.copy()
        if frame is None:
            return
        h, w = frame.shape[:2]
        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        except Exception:
            return
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return
        crop = frame[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            return
        try:
            self.feedback_no_haircut_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            safe_cam = "".join(c for c in cam if c.isalnum() or c in ("_", "-"))
            reason_tag = "".join(c if c.isalnum() else "_" for c in reason_l[:48]).strip("_") or "reject"
            out_path = self.feedback_no_haircut_dir / (
                f"{ts}_NO_HAIRCUT_{safe_cam}_P{pid}_G{gid}_{reason_tag}.jpg"
            )
            ok = self._cv2_imwrite_unicode_safe(out_path, crop)
            if ok:
                self._last_no_haircut_feedback_ts[key] = now
        except Exception:
            return

    @staticmethod
    def _parse_haircut_count_zones(raw: object) -> Dict[str, Set[str]]:
        """Parse config mapping: camera -> list of allowed CHAIR zones for haircut counting."""
        out: Dict[str, Set[str]] = {}
        if not isinstance(raw, dict):
            return out
        for cam, zones_raw in raw.items():
            cam_key = str(cam or "").strip()
            if not cam_key:
                continue
            zones: Set[str] = set()
            if isinstance(zones_raw, str):
                tokens = [z.strip() for z in zones_raw.replace(";", ",").split(",")]
                for token in tokens:
                    if token:
                        zones.add(token.upper())
            elif isinstance(zones_raw, (list, tuple, set)):
                for z in zones_raw:
                    token = str(z or "").strip()
                    if token:
                        zones.add(token.upper())
            if zones:
                out[cam_key] = zones
        return out

    @staticmethod
    def _match_zone_pattern(zone_name: str, pattern: str) -> bool:
        z = str(zone_name or "").strip().upper()
        p = str(pattern or "").strip().upper()
        if not z or not p:
            return False
        if p in {"*", "ALL"}:
            return True
        if p.endswith("*"):
            return z.startswith(p[:-1])
        return z == p

    def _is_haircut_count_enabled_for(self, camera_name: str, zone_name: str) -> bool:
        """Gate haircut counting by camera+zone allowlist.

        Behavior:
        - Empty allowlist => allow any CHAIR_* zone (legacy-safe fallback).
        - Camera configured => enforce configured zone patterns for that camera.
        - Camera not configured and no wildcard => allow CHAIR_* zone (prevents silent hard-block).
        - Wildcard configured (__all__/*/all) => enforce wildcard patterns.
        """
        cam = str(camera_name or "").strip()
        zone = str(zone_name or "").strip()
        if not zone:
            return False
        is_chair_zone = str(zone).upper().startswith(CHAIR_ZONE_PREFIX)
        if not self.haircut_count_zones:
            return bool(is_chair_zone)
        if not cam:
            return bool(is_chair_zone)

        # Camera key match should be case-insensitive for robustness.
        lower_map = {str(k).strip().lower(): v for k, v in (self.haircut_count_zones or {}).items()}
        cam_allowed = lower_map.get(cam.lower())
        if cam_allowed:
            for pattern in cam_allowed:
                if self._match_zone_pattern(zone, pattern):
                    return True
            return False

        wildcard_allowed = None
        for k in ("__all__", "*", "all"):
            v = lower_map.get(k)
            if v:
                wildcard_allowed = v
                break
        if wildcard_allowed:
            for pattern in wildcard_allowed:
                if self._match_zone_pattern(zone, pattern):
                    return True
            return False

        # No camera-specific or wildcard entry: do not hard-block this camera.
        return bool(is_chair_zone)

    @staticmethod
    def _parse_hhmm(value: str) -> Optional[dt_time]:
        try:
            txt = str(value or "").strip()
            if not txt:
                return None
            return datetime.strptime(txt, "%H:%M").time()
        except Exception:
            return None

    def _is_within_business_hours(self, dt_now: Optional[datetime] = None) -> bool:
        if not self.enable_business_hours_guard:
            return True
        now_dt = dt_now or datetime.now()
        start_t = self._parse_hhmm(self.business_hours_start)
        end_t = self._parse_hhmm(self.business_hours_end)
        if start_t is None or end_t is None:
            return True
        now_t = now_dt.time()
        if start_t <= end_t:
            return start_t <= now_t <= end_t
        # Overnight window, e.g. 20:00 -> 04:00
        return now_t >= start_t or now_t <= end_t

    def _classify_role_from_embedding(self, emb: Optional[np.ndarray]) -> Tuple[Optional[str], float, Dict[str, float]]:
        """Classify embedding into barber_uniform/customer using role_db thresholds."""
        try:
            if (not self.enable_role_db_reid) or (self.role_gallery is None) or emb is None:
                return None, -1.0, {}
            _best_role, _best_score, scores = self.role_gallery.match_role_with_score(emb)
            if not scores:
                return None, -1.0, {}

            barber_score = float(scores.get("barber_uniform", -1.0))
            customer_score = float(scores.get("customer", -1.0))
            barber_ok = barber_score >= float(self.role_db_barber_threshold)
            customer_ok = customer_score >= float(self.role_db_customer_threshold)
            margin = float(self.role_db_customer_margin)

            if barber_ok and customer_ok:
                if customer_score >= (barber_score + margin):
                    return "customer", customer_score, scores
                return "barber_uniform", barber_score, scores
            if barber_ok:
                return "barber_uniform", barber_score, scores
            if customer_ok:
                return "customer", customer_score, scores
            return None, max(barber_score, customer_score), scores
        except Exception:
            return None, -1.0, {}

    @staticmethod
    def _is_chair_zone_name(zone_name: Optional[str]) -> bool:
        try:
            return str(zone_name or "").strip().upper().startswith(CHAIR_ZONE_PREFIX)
        except Exception:
            return False

    def _chair_customer_recovery_applies(self, d: dict) -> bool:
        """Recover customer role in CHAIR when barber evidence is weak/ambiguous."""
        try:
            if not self.enable_chair_customer_recovery:
                return False
            if not self._is_chair_zone_name(d.get("primary_zone")):
                return False

            bbox = d.get("bbox") or []
            if len(bbox) != 4:
                return False
            x1, y1, x2, y2 = [float(v) for v in bbox]
            fw = float(max(int(d.get("frame_w", 0) or 0), 1))
            fh = float(max(int(d.get("frame_h", 0) or 0), 1))
            area_ratio = (max(0.0, x2 - x1) * max(0.0, y2 - y1)) / max(fw * fh, 1.0)
            if area_ratio < float(self.chair_customer_recovery_min_area_ratio):
                return False

            # Do not override strong stable barber vote.
            vote = str(d.get("role_vote", "") or "").strip().lower()
            vote_ratio = float(d.get("role_vote_ratio", 0.0) or 0.0)
            vote_samples = int(d.get("role_vote_samples", 0) or 0)
            if (
                vote == "barber"
                and vote_samples >= int(self.role_vote_min_samples)
                and vote_ratio >= max(0.80, float(self.role_vote_min_ratio) + 0.10)
            ):
                return False

            # Keep explicit staff-gallery IDs as staff.
            sid = str(d.get("staff_id", "") or "").strip().lower()
            if sid and (not sid.startswith("role_")) and sid not in {"staff_uniform"}:
                return False

            barber_score = float(d.get("role_barber_score", -1.0) or -1.0)
            customer_score = float(d.get("role_customer_score", -1.0) or -1.0)
            if customer_score < float(self.role_db_customer_threshold):
                return False
            if customer_score < (barber_score - float(self.chair_customer_recovery_margin)):
                return False
            return True
        except Exception:
            return False

    def _frame_role_vote(
        self,
        staff_id: Optional[str],
        role_id: Optional[str],
        role_barber_score: float,
        role_customer_score: float,
        conf: float,
        bbox: List[float],
        frame_w: int,
        frame_h: int,
        primary_zone: Optional[str] = None,
    ) -> str:
        """Classify one frame into barber/customer/unknown before temporal smoothing."""
        try:
            b = float(role_barber_score)
            c = float(role_customer_score)
            in_chair = self._is_chair_zone_name(primary_zone)
            area_ratio = 0.0
            if len(bbox) == 4:
                x1, y1, x2, y2 = [float(v) for v in bbox]
                bw = max(0.0, x2 - x1)
                bh = max(0.0, y2 - y1)
                area_ratio = (bw * bh) / float(max(1.0, float(frame_w) * float(frame_h)))
                if area_ratio < max(0.001, self.role_snapshot_min_area_ratio * 0.5):
                    return "unknown"

            chair_customer_recover = (
                bool(in_chair)
                and area_ratio >= float(self.chair_customer_recovery_min_area_ratio)
                and c >= float(self.role_db_customer_threshold)
                and c >= (b - float(self.chair_customer_recovery_margin))
            )

            if staff_id is not None:
                sid = str(staff_id or "").strip().lower()
                if chair_customer_recover:
                    return "customer"
                # role-derived staff labels in CHAIR need stronger barber margin.
                if (
                    in_chair
                    and sid.startswith("role_")
                    and b < (float(self.role_db_barber_threshold) + float(self.chair_staff_strict_barber_margin))
                ):
                    return "unknown"
                return "barber"

            rid = str(role_id or "").strip().lower()
            if rid in {"barber_uniform", "barber", "staff_uniform", "staff"}:
                if chair_customer_recover:
                    return "customer"
                return "barber"
            if rid == "customer":
                # Keep customer only when confidence is sane; else defer to unknown.
                if float(conf) >= float(self.role_unknown_min_conf):
                    return "customer"
                return "unknown"

            if float(conf) < float(self.role_unknown_min_conf):
                return "unknown"
            margin = float(self.role_db_customer_margin)
            if b >= float(self.role_db_barber_threshold):
                if c < float(self.role_db_customer_threshold):
                    if in_chair and b < (float(self.role_db_barber_threshold) + float(self.chair_staff_strict_barber_margin)):
                        return "unknown"
                    return "barber"
                if c < (b + margin):
                    if in_chair and b < (float(self.role_db_barber_threshold) + float(self.chair_staff_strict_barber_margin)):
                        return "unknown"
                    return "barber"
            if c >= float(self.role_db_customer_threshold):
                if c >= (b + margin):
                    return "customer"
            return "unknown"
        except Exception:
            return "unknown"

    def _update_pid_role_vote(self, pid: int, frame_vote: str, now_ts: float) -> Tuple[str, float, int]:
        try:
            pid = int(pid)
            if pid <= 0:
                return "unknown", 0.0, 0
            dq = self.pid_role_votes.get(pid)
            if dq is None or dq.maxlen != self.role_vote_window:
                dq = deque(list(dq) if dq is not None else [], maxlen=self.role_vote_window)
                self.pid_role_votes[pid] = dq
            vote = str(frame_vote or "unknown").strip().lower()
            if vote not in {"barber", "customer", "unknown"}:
                vote = "unknown"
            dq.append(vote)
            self.pid_role_last_ts[pid] = float(now_ts)

            counts = {"barber": 0, "customer": 0, "unknown": 0}
            for v in dq:
                counts[v] = counts.get(v, 0) + 1
            samples = int(len(dq))
            best_role = max(counts.items(), key=lambda kv: kv[1])[0] if samples > 0 else "unknown"
            best_ratio = float(counts.get(best_role, 0)) / float(max(samples, 1))

            stable_role = "unknown"
            if samples >= int(self.role_vote_min_samples) and best_ratio >= float(self.role_vote_min_ratio):
                stable_role = str(best_role)
            return stable_role, best_ratio, samples
        except Exception:
            return "unknown", 0.0, 0

    def _close_stale_role_sessions(self, now_ts: float, active_pids: Set[int]):
        """Close inactive PID role sessions after timeout to avoid stale state carry-over."""
        try:
            active_set = {int(p) for p in active_pids if int(p) > 0}
            timeout = float(self.service_session_timeout_sec)
            for pid, last_ts in list(self.pid_role_last_ts.items()):
                p = int(pid)
                if p in active_set:
                    continue
                if (float(now_ts) - float(last_ts)) < timeout:
                    continue
                self.pid_role_last_ts.pop(p, None)
                self.pid_role_votes.pop(p, None)
        except Exception:
            pass

    def _is_staff_for_service_counts(self, d: dict) -> bool:
        """Return True when a detection is staff-like and should be excluded from service counts."""
        try:
            if self._chair_customer_recovery_applies(d):
                return False
            vote = str(d.get("role_vote", "") or "").strip().lower()
            if vote == "barber":
                return True
            if d.get("staff_id") is not None:
                return True

            role_id = str(d.get("role_id", "") or "").strip().lower()
            if role_id in {"barber_uniform", "barber", "staff_uniform", "staff"}:
                return True

            barber_score = float(d.get("role_barber_score", -1.0) or -1.0)
            customer_score = float(d.get("role_customer_score", -1.0) or -1.0)
            if barber_score >= float(self.role_db_barber_threshold):
                if self._is_chair_zone_name(d.get("primary_zone")):
                    if barber_score < (float(self.role_db_barber_threshold) + float(self.chair_staff_strict_barber_margin)):
                        return False
                if customer_score < float(self.role_db_customer_threshold):
                    return True
                if customer_score < (barber_score + float(self.role_db_customer_margin)):
                    return True
            return False
        except Exception:
            return d.get("staff_id") is not None

    def _is_customer_for_service_counts(self, d: dict) -> bool:
        try:
            if self._is_staff_for_service_counts(d):
                return False
            if not self.require_stable_customer_role_for_service:
                return True
            role_vote = str(d.get("role_vote", "unknown") or "unknown").strip().lower()
            if role_vote != "customer":
                return False
            samples = int(d.get("role_vote_samples", 0) or 0)
            ratio = float(d.get("role_vote_ratio", 0.0) or 0.0)
            if samples < int(self.role_vote_min_samples):
                return False
            if ratio < float(self.role_vote_min_ratio):
                return False
            return True
        except Exception:
            return False

    def _validate_service_event_candidate(self, d: dict, zone: str, event_name: str, dwell_sec: float) -> Tuple[bool, str]:
        try:
            if str(event_name or "").strip().lower() == "haircut":
                cam = str(d.get("cam", d.get("camera", "")) or "")
                if not self._is_haircut_count_enabled_for(cam, str(zone or "")):
                    return False, "haircut_zone_not_enabled"
                cls_ok, cls_reason = self._validate_haircut_with_chair_service_classifier(d)
                if not cls_ok:
                    return False, cls_reason
                if self.enable_chair_empty_guard:
                    empty_sim, _empty_path = self._empty_chair_similarity_from_detection(d)
                    if empty_sim >= float(self.chair_empty_similarity_threshold):
                        return False, f"chair_empty_like_{empty_sim:.2f}"
                id_ok, id_reason = self._validate_haircut_identity_and_vacancy_guard(d, str(zone or ""))
                if not id_ok:
                    return False, id_reason
            if not self._is_within_business_hours():
                return False, "outside_business_hours"
            now = time.time()
            ts = float(d.get("ts", 0.0))
            if self.event_reject_stale_hold and bool(d.get("stale", False)):
                return False, "stale_hold"
            if self.event_require_fresh_detection and ts > 0 and (now - ts) > self.event_max_detection_age_sec:
                return False, "stale_age"
            conf = float(d.get("conf", 0.0))
            det_type = str(d.get("det_type", ""))

            # Event-stage staff guard: keep barber snapshots out even if live-frame
            # staff tagging missed in a specific frame.
            enforce_staff_guard = self.exclude_staff_from_counts or str(event_name).lower() in {"haircut", "wash"}
            if enforce_staff_guard:
                if self.require_stable_customer_role_for_service and str(event_name).lower() in {"haircut", "wash"}:
                    role_vote = str(d.get("role_vote", "unknown") or "unknown").strip().lower()
                    role_vote_ratio = float(d.get("role_vote_ratio", 0.0) or 0.0)
                    role_vote_samples = int(d.get("role_vote_samples", 0) or 0)
                    if role_vote != "customer":
                        return False, f"role_unstable_{role_vote}"
                    if role_vote_samples < int(self.role_vote_min_samples):
                        return False, f"role_vote_samples_low_{role_vote_samples}"
                    if role_vote_ratio < float(self.role_vote_min_ratio):
                        return False, f"role_vote_ratio_low_{role_vote_ratio:.2f}"
                if self._is_staff_for_service_counts(d):
                    return False, "staff_filtered"
                emb = d.get("emb")
                if emb is None:
                    emb = d.get("_event_emb")
                if emb is None:
                    emb = self._compute_event_embedding_from_detection(d)
                    if emb is not None:
                        d["_event_emb"] = emb

                role_id, role_score, _role_scores = self._classify_role_from_embedding(emb)
                if role_id == "barber_uniform":
                    d["staff_id"] = d.get("staff_id") or "role_barber_db_event"
                    d["role_id"] = role_id
                    d["role_score"] = float(role_score)
                    return False, f"role_db_barber_{role_score:.2f}"

                if self.staff_gallery is not None:
                    if emb is not None:
                        sid, sim = self.staff_gallery.match_staff_with_score(emb)
                        if sid is not None and sim >= float(self.staff_event_similarity_threshold):
                            d["staff_id"] = sid
                            return False, f"staff_reid_{sim:.2f}"

            bbox = d.get("bbox") or []
            if len(bbox) != 4:
                return False, "no_bbox"
            x1, y1, x2, y2 = [float(v) for v in bbox]
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            if bw <= 1.0 or bh <= 1.0:
                return False, "tiny_bbox_px"
            fw = float(max(int(d.get("frame_w", 0) or 0), 1))
            fh = float(max(int(d.get("frame_h", 0) or 0), 1))
            width_ratio = bw / fw
            height_ratio = bh / fh
            area_ratio = (bw * bh) / max(fw * fh, 1.0)

            if det_type == "person":
                if conf < self.event_person_min_conf:
                    return False, "person_conf_low"
            else:
                if conf < self.event_head_min_conf:
                    return False, "head_conf_low"
                if area_ratio < self.event_head_min_area_ratio:
                    return False, "head_too_small"
                if area_ratio > self.event_head_max_area_ratio:
                    return False, "head_too_large_area"
                if width_ratio > self.event_head_max_width_ratio:
                    return False, "head_too_wide"
                if height_ratio > self.event_head_max_height_ratio:
                    return False, "head_too_tall"
            return True, "ok"
        except Exception:
            return False, "validator_error"

    def _head_live_reject_reason(self, x1: float, y1: float, x2: float, y2: float, conf: float, w: int, h: int) -> str:
        """Return reject reason for implausible head detection in live overlay/counting, or '' if ok."""
        try:
            if not self.enable_head_live_filter:
                return ""
            if conf < self.head_live_min_conf:
                return "head_live_conf_low"
            bw = max(0.0, float(x2 - x1))
            bh = max(0.0, float(y2 - y1))
            fw = float(max(int(w), 1))
            fh = float(max(int(h), 1))
            area_ratio = (bw * bh) / max(fw * fh, 1.0)
            if area_ratio < self.head_live_min_area_ratio:
                return "head_live_too_small"
            if area_ratio > self.head_live_max_area_ratio:
                return "head_live_too_large_area"
            if (bw / fw) > self.head_live_max_width_ratio:
                return "head_live_too_wide"
            if (bh / fh) > self.head_live_max_height_ratio:
                return "head_live_too_tall"
            return ""
        except Exception:
            return "head_live_filter_error"

    def _find_live_match_for_event(self, event) -> Optional[dict]:
        try:
            cam = str(getattr(event, "camera", "") or "")
            if not cam:
                return None
            now = time.time()
            target_pid = int(getattr(event, "person_id", 0) or 0)
            target_gid = 0
            target_bbox = None
            if isinstance(getattr(event, "metadata", None), dict):
                target_gid = int(event.metadata.get("gid", 0) or 0)
                target_bbox = event.metadata.get("bbox")
            with self.customers_lock:
                items = [dict(x) for x in (self.latest_customers.get(cam) or [])]
            best = None
            best_score = -1.0
            for d in items:
                ts = float(d.get("ts", 0.0))
                if ts <= 0 or (now - ts) > self.event_snapshot_live_match_max_age_sec:
                    continue
                if self.event_reject_stale_hold and bool(d.get("stale", False)):
                    continue
                pid = int(d.get("pid", 0) or 0)
                gid = int(d.get("gid", 0) or 0)
                if target_gid > 0 and gid != target_gid:
                    continue
                if target_gid <= 0 and target_pid > 0 and pid != target_pid:
                    continue
                score = float(d.get("conf", 0.0))
                if target_bbox and isinstance(target_bbox, (list, tuple)) and len(target_bbox) == 4:
                    try:
                        score += 0.5 * self._bbox_iou_xyxy(
                            [float(v) for v in d.get("bbox", [])],
                            [float(v) for v in target_bbox],
                        )
                    except Exception:
                        pass
                if score > best_score:
                    best_score = score
                    best = d
            return best
        except Exception:
            return None

    def _get_event_reid_encoder(self) -> Optional[ReIDEncoder]:
        if self.reid_encoder is not None:
            return self.reid_encoder
        if not self.enable_event_reid_dedupe:
            return None
        if self.event_reid_encoder is not None:
            return self.event_reid_encoder
        try:
            reid_device = self.yolo_device if self.yolo_device != "mps" else "cpu"
            self.event_reid_encoder = ReIDEncoder(device=reid_device)
            logger.info(f"Initialized event dedupe ReID encoder on {reid_device}")
        except Exception as e:
            logger.warning(f"Event dedupe ReID unavailable: {e}")
            self.enable_event_reid_dedupe = False
            self.event_reid_encoder = None
        return self.event_reid_encoder

    def _compute_event_embedding_from_detection(self, d: dict) -> Optional[np.ndarray]:
        enc = self._get_event_reid_encoder()
        if enc is None:
            return None
        try:
            cam = str(d.get("cam", d.get("camera", "")) or "")
            bbox = d.get("bbox") or []
            if not cam or len(bbox) != 4:
                return None
            with self.customers_lock:
                frame = self.latest_frames.get(cam)
                if frame is None:
                    return None
                frame = frame.copy()
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            if x2 <= x1 or y2 <= y1:
                return None
            crop = frame[y1:y2, x1:x2]
            if crop is None or crop.size == 0:
                return None
            return enc.encode_crop_bgr(crop)
        except Exception:
            return None

    @staticmethod
    def _build_seat_key(camera: str, zone: str) -> str:
        cam = str(camera or "").strip()
        zn = str(zone or "").strip()
        if not zn:
            return ""
        return f"{cam}::{zn}" if cam else zn

    def _extract_detection_crop_bgr(self, d: dict, context_scale: float = 1.12) -> Optional[np.ndarray]:
        try:
            cam = str(d.get("cam", d.get("camera", "")) or "")
            bbox = d.get("bbox") or []
            if not cam or len(bbox) != 4:
                return None
            with self.customers_lock:
                frame = self.latest_frames.get(cam)
                if frame is None:
                    return None
                frame = frame.copy()
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = [float(v) for v in bbox]
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            bw = max(1.0, (x2 - x1) * float(context_scale))
            bh = max(1.0, (y2 - y1) * float(context_scale))
            nx1 = int(max(0, min(w - 1, round(cx - 0.5 * bw))))
            ny1 = int(max(0, min(h - 1, round(cy - 0.5 * bh))))
            nx2 = int(max(0, min(w, round(cx + 0.5 * bw))))
            ny2 = int(max(0, min(h, round(cy + 0.5 * bh))))
            if nx2 <= nx1 or ny2 <= ny1:
                return None
            crop = frame[ny1:ny2, nx1:nx2]
            if crop is None or crop.size == 0:
                return None
            return crop
        except Exception:
            return None

    @staticmethod
    def _compute_hsv_hist_feature(crop_bgr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        try:
            if crop_bgr is None or crop_bgr.size == 0:
                return None
            hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
            n = float(np.linalg.norm(hist) + 1e-9)
            return hist / n
        except Exception:
            return None

    def _extract_visual_features_from_crop(self, crop_bgr: Optional[np.ndarray]) -> Dict[str, Optional[np.ndarray]]:
        emb = None
        try:
            enc = self.reid_encoder
            if enc is None:
                enc = self.event_reid_encoder if self.event_reid_encoder is not None else self._get_event_reid_encoder()
            if enc is not None and crop_bgr is not None and crop_bgr.size > 0:
                emb = enc.encode_crop_bgr(crop_bgr)
        except Exception:
            emb = None
        hist = self._compute_hsv_hist_feature(crop_bgr)
        return {"emb": emb, "hist": hist}

    def _load_chair_empty_gallery_features(self):
        self.chair_empty_gallery_features = []
        if not bool(self.enable_chair_empty_guard):
            return
        raw_dir = str(self.chair_empty_gallery_dir or "").strip()
        if not raw_dir:
            return
        gallery_dir = Path(raw_dir)
        if not gallery_dir.is_absolute():
            gallery_dir = (PROJECT_ROOT / gallery_dir).resolve()
        if not gallery_dir.exists():
            logger.warning(f"Chair-empty gallery directory not found: {gallery_dir}")
            return
        image_paths: List[Path] = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            image_paths.extend(gallery_dir.rglob(ext))
        if not image_paths:
            logger.warning(f"Chair-empty gallery has no images: {gallery_dir}")
            return

        loaded: List[Dict[str, object]] = []
        for p in sorted(set(image_paths)):
            try:
                img = cv2.imread(str(p))
                if img is None or img.size == 0:
                    continue
                feats = self._extract_visual_features_from_crop(img)
                if feats.get("emb") is None and feats.get("hist") is None:
                    continue
                loaded.append(
                    {
                        "path": str(p),
                        "emb": feats.get("emb"),
                        "hist": feats.get("hist"),
                    }
                )
            except Exception:
                continue
        self.chair_empty_gallery_features = loaded
        logger.info(
            f"Loaded chair-empty gallery features: {len(loaded)} images from {gallery_dir}"
        )

    def _empty_chair_similarity_from_detection(self, d: dict) -> Tuple[float, str]:
        gallery = self.chair_empty_gallery_features or []
        if not gallery:
            return -1.0, ""
        crop = self._extract_detection_crop_bgr(d, context_scale=1.10)
        feats = self._extract_visual_features_from_crop(crop)
        emb = feats.get("emb")
        hist = feats.get("hist")
        if emb is None and hist is None:
            return -1.0, ""

        hist_w = float(self.chair_empty_hist_weight)
        emb_w = max(0.0, 1.0 - hist_w)
        best_sim = -1.0
        best_path = ""
        for g in gallery:
            g_emb = g.get("emb")
            g_hist = g.get("hist")
            sim_emb = cosine_sim(emb, g_emb) if emb is not None and g_emb is not None else -1.0
            sim_hist = cosine_sim(hist, g_hist) if hist is not None and g_hist is not None else -1.0
            if sim_emb < 0.0 and sim_hist < 0.0:
                continue
            if sim_emb >= 0.0 and sim_hist >= 0.0:
                sim = (emb_w * sim_emb) + (hist_w * sim_hist)
            else:
                sim = max(sim_emb, sim_hist)
            if sim > best_sim:
                best_sim = sim
                best_path = str(g.get("path", ""))
        try:
            d["chair_empty_similarity"] = float(best_sim)
            if best_path:
                d["chair_empty_match_path"] = best_path
        except Exception:
            pass
        return float(best_sim), best_path

    def _prune_recent_service_identity_events(self, now_ts: Optional[float] = None):
        now = float(now_ts if now_ts is not None else time.time())
        keep_sec = max(
            float(self.event_same_customer_block_sec),
            float(self.event_same_seat_cooldown_sec),
            float(self.event_wash_return_block_sec),
            float(self.haircut_event_dedupe_window_sec),
        ) + 30.0
        self.recent_service_identity_events = [
            x
            for x in (self.recent_service_identity_events or [])
            if (now - float(x.get("ts", 0.0))) <= keep_sec
        ]

    def _is_same_identity_against_memory(
        self,
        prev: Dict[str, object],
        pid: int,
        gid: int,
        emb: Optional[np.ndarray],
    ) -> Tuple[bool, float]:
        prev_pid = int(prev.get("pid", 0) or 0)
        prev_gid = int(prev.get("gid", 0) or 0)
        if gid > 0 and prev_gid > 0 and gid == prev_gid:
            return True, 1.0
        if pid > 0 and prev_pid > 0 and pid == prev_pid:
            return True, 1.0
        prev_emb = prev.get("emb")
        if emb is not None and prev_emb is not None:
            sim = cosine_sim(emb, prev_emb)
            if sim >= float(self.event_identity_similarity_threshold):
                return True, float(sim)
            return False, float(sim)
        return False, -1.0

    def _validate_haircut_identity_and_vacancy_guard(self, d: dict, zone: str) -> Tuple[bool, str]:
        now = time.time()
        cam = str(d.get("cam", d.get("camera", "")) or "")
        seat_key = self._build_seat_key(cam, str(zone or ""))
        pid = int(d.get("pid", 0) or 0)
        gid = int(d.get("gid", 0) or 0)
        emb = d.get("_event_emb")
        if emb is None:
            emb = self._compute_event_embedding_from_detection(d)
            if emb is not None:
                d["_event_emb"] = emb

        self._prune_recent_service_identity_events(now)
        memory = list(self.recent_service_identity_events or [])
        if not memory:
            return True, "identity_guard_no_history"

        latest_haircut_same_seat_ts = 0.0
        for prev in memory:
            if str(prev.get("event", "")).strip().lower() != "haircut":
                continue
            if seat_key and seat_key == str(prev.get("seat_key", "")):
                pts = float(prev.get("ts", 0.0))
                if pts > latest_haircut_same_seat_ts:
                    latest_haircut_same_seat_ts = pts
        if latest_haircut_same_seat_ts > 0.0:
            age_same_seat = now - latest_haircut_same_seat_ts
            if age_same_seat <= float(self.event_same_seat_cooldown_sec):
                return False, f"same_seat_cooldown_{age_same_seat:.1f}s"
        if (
            self.require_chair_vacant_between_haircuts
            and latest_haircut_same_seat_ts > 0.0
            and (now - latest_haircut_same_seat_ts) <= float(self.event_same_customer_block_sec)
        ):
            last_vacant_ts = float(self._chair_zone_last_vacant_ts.get(seat_key, 0.0))
            if last_vacant_ts <= latest_haircut_same_seat_ts:
                return False, "chair_not_vacant_since_last_haircut"

        for prev in reversed(memory):
            pts = float(prev.get("ts", 0.0))
            if pts <= 0:
                continue
            age = now - pts
            if age < 0:
                continue
            same_identity, sim = self._is_same_identity_against_memory(prev, pid, gid, emb)
            if not same_identity:
                continue
            prev_event = str(prev.get("event", "")).strip().lower()
            prev_seat_key = str(prev.get("seat_key", ""))
            if prev_event == "wash" and age <= float(self.event_wash_return_block_sec):
                if seat_key and prev_seat_key and seat_key == prev_seat_key:
                    return False, f"wash_return_same_chair_sim_{sim:.2f}"
                return False, f"recent_wash_same_customer_sim_{sim:.2f}"
            if prev_event == "haircut" and age <= float(self.event_same_customer_block_sec):
                if seat_key and prev_seat_key and seat_key == prev_seat_key:
                    return False, f"same_customer_same_chair_sim_{sim:.2f}"
                return False, f"same_customer_recent_haircut_sim_{sim:.2f}"
        return True, "identity_guard_ok"

    def _remember_service_events(self, events: List[dict], event_name: str):
        if not events:
            return
        now = time.time()
        self._prune_recent_service_identity_events(now)
        et = str(event_name or "").strip().lower()
        for ev in events:
            try:
                cam = str(ev.get("camera", ev.get("cam", "")) or "")
                zone = str(ev.get("zone", ev.get("primary_zone", "")) or "")
                pid = int(ev.get("pid", 0) or 0)
                gid = int(ev.get("gid", 0) or 0)
                emb = ev.get("_event_emb")
                if emb is None:
                    emb = self._compute_event_embedding_from_detection(ev)
                    if emb is not None:
                        ev["_event_emb"] = emb
                self.recent_service_identity_events.append(
                    {
                        "ts": float(now),
                        "event": et,
                        "camera": cam,
                        "zone": zone,
                        "seat_key": self._build_seat_key(cam, zone),
                        "pid": pid,
                        "gid": gid,
                        "emb": emb,
                    }
                )
            except Exception:
                continue
        self._prune_recent_service_identity_events(now)

    def _update_chair_zone_vacancy_memory(self, service_merged: List[dict], now_ts: float):
        occupied: Set[str] = set()
        for d in service_merged or []:
            zone = str(d.get("primary_zone", "") or "")
            if not zone.startswith(CHAIR_ZONE_PREFIX):
                continue
            cam = str(d.get("cam", d.get("camera", "")) or "")
            seat_key = self._build_seat_key(cam, zone)
            if seat_key:
                occupied.add(seat_key)
        prev_occupied = set(self._chair_zone_occupied_keys or set())
        for seat_key in (prev_occupied - occupied):
            self._chair_zone_last_vacant_ts[seat_key] = float(now_ts)
        self._chair_zone_occupied_keys = occupied
        prune_before = float(now_ts) - (
            max(
                float(self.event_same_customer_block_sec),
                float(self.event_wash_return_block_sec),
            )
            + 60.0
        )
        self._chair_zone_last_vacant_ts = {
            k: float(v)
            for k, v in (self._chair_zone_last_vacant_ts or {}).items()
            if float(v) >= prune_before
        }

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

        # Keep event dedupe memory much longer than live cross-camera canonicalization.
        self.recent_haircut_reid = [
            x for x in self.recent_haircut_reid
            if (now - float(x.get("ts", 0.0))) <= self.haircut_event_dedupe_window_sec
        ]

        kept: List[dict] = []
        for ev in haircut_events:
            pid = int(ev.get("pid", 0))
            emb = pid_to_emb.get(pid)
            if emb is None:
                emb = self._compute_event_embedding_from_detection(ev)
                if emb is not None and pid:
                    pid_to_emb[pid] = emb
            if emb is None:
                kept.append(ev)
                continue

            is_dup = False
            best_sim = -1.0
            for prev in self.recent_haircut_reid:
                p_emb = prev.get("emb")
                if p_emb is None:
                    continue
                sim = cosine_sim(emb, p_emb)
                if sim > best_sim:
                    best_sim = sim
                if sim >= self.haircut_event_dedupe_similarity:
                    is_dup = True
                    break
            if is_dup:
                zone = str(ev.get("zone", ""))
                self.haircut_counter.total_count = max(0, self.haircut_counter.total_count - 1)
                if zone in self.haircut_counter.zone_total:
                    self.haircut_counter.zone_total[zone] = max(0, self.haircut_counter.zone_total[zone] - 1)
                if bool(getattr(self, "fsm_enabled", False)) and hasattr(self, "service_fsm"):
                    self.service_fsm.haircut_total = max(0, int(self.service_fsm.haircut_total) - 1)
                    if zone in self.service_fsm.haircut_zone_total:
                        self.service_fsm.haircut_zone_total[zone] = max(
                            0,
                            int(self.service_fsm.haircut_zone_total.get(zone, 0)) - 1,
                        )
                logger.info(
                    f"Dropped duplicate haircut event by event ReID dedupe: pid={pid}, zone={zone}, "
                    f"best_sim={best_sim:.3f}, threshold={self.haircut_event_dedupe_similarity}, "
                    f"window={self.haircut_event_dedupe_window_sec}s"
                )
                continue

            kept.append(ev)
            self.recent_haircut_reid.append({
                "ts": now,
                "emb": emb,
                "camera": str(ev.get("camera", "")),
                "pid": pid,
                "zone": str(ev.get("zone", "")),
            })
        return kept

    def _detect_chair_return_washes(self, service_merged: List[dict], now: float) -> List[dict]:
        """A customer who re-sits in a CHAIR seat shortly after a haircut there (same
        identity) is treated as a WASH return — not a new haircut. Heuristic, bounded
        by event_chair_return_wash_window_sec; deduped via recent_service_identity_events
        (once a wash is remembered, the existing identity guard blocks re-counting it).
        Limited to the configured counting cameras (haircut_count_zones)."""
        if not getattr(self, "enable_chair_return_wash", True):
            return []
        window = float(getattr(self, "event_chair_return_wash_window_sec", 900.0))
        self._prune_recent_service_identity_events(now)
        memory = list(self.recent_service_identity_events or [])
        if not memory:
            return []
        out: List[dict] = []
        seen_seats: Set[str] = set()
        for d in service_merged or []:
            zone = str(d.get("primary_zone", "") or "")
            if not zone.startswith(CHAIR_ZONE_PREFIX):
                continue
            cam = str(d.get("cam", d.get("camera", "")) or "")
            if not self._is_haircut_count_enabled_for(cam, zone):
                continue
            seat_key = self._build_seat_key(cam, zone)
            if not seat_key or seat_key in seen_seats:
                continue
            pid = int(d.get("pid", 0) or 0)
            gid = int(d.get("gid", 0) or 0)
            emb = d.get("_event_emb")
            if emb is None:
                emb = self._compute_event_embedding_from_detection(d)
                if emb is not None:
                    d["_event_emb"] = emb
            matched_haircut = False
            already_washed = False
            for prev in memory:
                if str(prev.get("seat_key", "")) != seat_key:
                    continue
                age = now - float(prev.get("ts", 0.0))
                if age < 0 or age > window:
                    continue
                same, _sim = self._is_same_identity_against_memory(prev, pid, gid, emb)
                if not same:
                    continue
                ev_type = str(prev.get("event", "")).strip().lower()
                if ev_type == "wash":
                    already_washed = True
                    break
                if ev_type == "haircut":
                    matched_haircut = True
            if matched_haircut and not already_washed:
                out.append({
                    "camera": cam,
                    "pid": pid,
                    "gid": gid,
                    "zone": zone,
                    "dwell": float(d.get("zone_dwell_sec", d.get("dwell", 0.0)) or 0.0),
                    "bbox": d.get("bbox"),
                    "conf": float(d.get("conf", 0.0) or 0.0),
                    "det_type": d.get("det_type"),
                    "frame_w": int(d.get("frame_w", 0) or 0),
                    "frame_h": int(d.get("frame_h", 0) or 0),
                    "reason_code": "chair_return_wash",
                    "_event_emb": emb,
                })
                seen_seats.add(seat_key)
        return out

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

    def _today_event_count(self) -> int:
        """Count only today's events for dashboard summary."""
        today = datetime.now().date()
        with self.pending_lock:
            snapshot = list(self.all_events)
        count = 0
        for e in snapshot:
            ts = getattr(e, "timestamp", None)
            if isinstance(ts, datetime) and ts.date() == today:
                count += 1
        return count

    def _reset_daily_counters_if_needed(self):
        """Reset runtime counters when calendar day changes."""
        today = datetime.now().strftime("%Y-%m-%d")
        if today == self._runtime_day:
            return
        with self._daily_rollover_lock:
            if today == self._runtime_day:
                return
            prev_day = self._runtime_day
            try:
                prev_dt = datetime.strptime(prev_day, "%Y-%m-%d")
                self.report_gen.generate_daily_report(self.all_events, date=prev_dt)
            except Exception as e:
                logger.warning(f"Daily rollover report flush failed for {prev_day}: {e}")

            self.haircut_counter.reset()
            self.wash_counter.reset()
            self.service_fsm.reset()
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
            self.recent_service_identity_events = []
            self._chair_zone_occupied_keys = set()
            self._chair_zone_last_vacant_ts = {}
            self.pid_wait_start = {}
            self.pid_role_votes = {}
            self.pid_role_last_ts = {}
            self.role_vote_total_frames = 0
            self.role_vote_unknown_frames = 0
            self.fsm_state_counts = {}
            self.fsm_recent_transitions = []
            with self.verification_state_lock:
                self.verified_events_total = 0
                self.verified_by_camera = {}
                self.last_verified_event = None
            self._runtime_day = today
            logger.info(f"Daily rollover: reset counters for new day {today} (prev={prev_day})")
            self._write_dashboard_state(force=True)

    def _global_counting_loop(self):
        """Edge-agent style global counting over merged customers across cameras."""
        logger.info("Global counting loop started")
        while self.running:
            try:
                self._reset_daily_counters_if_needed()
                now = time.time()
                self._auto_manage_inference_mode(now)
                with self.customers_lock:
                    merged = []
                    for items in self.latest_customers.values():
                        for d in items:
                            ts = float(d.get("ts", 0.0))
                            if ts > 0 and (now - ts) <= self.customer_active_ttl_sec:
                                merged.append(d)

                self.active_person_memory.expire(now, self.reid_active_expire_sec)
                
                # Identify staff who are currently active (before canonicalization)
                # Actually, merged detections already have staff_id if persistent.
                
                merged = self._canonicalize_pids_cross_camera(merged)
                service_merged = [d for d in merged if not self._is_staff_for_service_counts(d)]
                role_active_pids = {int(d.get("pid", 0)) for d in service_merged if int(d.get("pid", 0)) > 0}
                self._close_stale_role_sessions(now, role_active_pids)

                # Use stable service identity (prefer gid) to avoid duplicate recount
                # when tracker PID changes for the same person.
                remapped_service: List[dict] = []
                for d in service_merged:
                    nd = dict(d)
                    raw_pid = int(nd.get("pid", 0) or 0)
                    gid = int(nd.get("gid", 0) or 0)
                    sid = gid if gid > 0 else raw_pid
                    if raw_pid > 0 and sid != raw_pid:
                        nd["track_pid"] = raw_pid
                    nd["pid"] = int(sid)
                    remapped_service.append(nd)
                service_merged = remapped_service
                self._update_chair_zone_vacancy_memory(service_merged, now)
                service_active_pids = {int(d.get("pid", 0)) for d in service_merged if int(d.get("pid", 0)) > 0}
                
                # Update per-camera counts and staff counts for dashboard
                cam_counts = {}
                cam_staff = {}
                active_staff_pids = set()
                
                for d in merged:
                    cname = d.get("cam")
                    pid = int(d.get("pid", 0))
                    is_staff = self._is_staff_for_service_counts(d)
                    
                    if cname:
                        if is_staff:
                            cam_staff[cname] = cam_staff.get(cname, 0) + 1
                            active_staff_pids.add(pid)
                        else:
                            cam_counts[cname] = cam_counts.get(cname, 0) + 1
                            
                self.camera_people_counts = cam_counts
                self.camera_staff_counts = cam_staff
                self.staff_realtime = len(active_staff_pids)

                self.haircut_counter.sync_active_pids(service_active_pids)
                self.wash_counter.sync_active_pids(service_active_pids)

                # Update wait-time tracking
                current_wait_pids = set()
                for d in service_merged:
                    pid = int(d.get("pid", 0))
                    zones_hit = d.get("zones") or []
                    in_wait = any(str(zn) in WAIT_ZONE_NAMES for zn in zones_hit)
                    if in_wait:
                        current_wait_pids.add(pid)
                        if pid not in self.pid_wait_start:
                            self.pid_wait_start[pid] = now
                
                # Cleanup wait_tracker for gone PIDs
                self.pid_wait_start = {p: t for p, t in self.pid_wait_start.items() if p in service_active_pids}
                if self.fsm_enabled:
                    fsm_input = [d for d in service_merged if self._is_customer_for_service_counts(d)]
                    def fsm_validate(kind: str, det: dict, zone: str, dwell_sec: float):
                        ok, reason = self._validate_service_event_candidate(det, zone, kind, dwell_sec)
                        if not ok:
                            self._log_event_reject(kind, det, zone, reason)
                        return ok, reason
                    fsm_out = self.service_fsm.update(
                        fsm_input,
                        now,
                        wait_tracker=self.pid_wait_start,
                        event_validator_fn=fsm_validate,
                    )
                    haircut_events = list(fsm_out.get("haircut_events", []))
                    wash_events = list(fsm_out.get("wash_events", []))
                    self.fsm_state_counts = dict(fsm_out.get("state_counts", {}))
                    self.fsm_recent_transitions = list(self.service_fsm.recent_transitions)[-60:]
                    self.haircut_counter.total_count = int(self.service_fsm.haircut_total)
                    self.wash_counter.total_count = int(self.service_fsm.wash_total)
                    self.haircut_counter.zone_total = dict(self.service_fsm.haircut_zone_total)
                    self.wash_counter.zone_total = dict(self.service_fsm.wash_zone_total)
                else:
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

                    def validate_haircut_event(d, zone, dwell_sec):
                        ok, reason = self._validate_service_event_candidate(d, zone, "haircut", dwell_sec)
                        if not ok:
                            self._log_event_reject("haircut", d, zone, reason)
                        return ok, reason

                    def validate_wash_event(d, zone, dwell_sec):
                        ok, reason = self._validate_service_event_candidate(d, zone, "wash", dwell_sec)
                        if not ok:
                            self._log_event_reject("wash", d, zone, reason)
                        return ok, reason

                    haircut_events = self.haircut_counter.update_and_collect_events(
                        service_merged, select_haircut_zone, validate_haircut_event, wait_tracker=self.pid_wait_start
                    )
                    wash_events = self.wash_counter.update_and_collect_events(
                        service_merged, select_wash_zone, validate_wash_event
                    )
                    self.fsm_state_counts = {}
                    self.fsm_recent_transitions = []
                haircut_events = self._dedupe_haircut_events_by_reid(haircut_events, service_merged)
                # Returning customer re-sitting in a chair shortly after a haircut → wash (not a new haircut)
                try:
                    chair_return_washes = self._detect_chair_return_washes(service_merged, time.time())
                except Exception as _e:
                    chair_return_washes = []
                    logger.debug(f"chair-return-wash detect failed: {_e}")
                if chair_return_washes:
                    for _wev in chair_return_washes:
                        _z = str(_wev.get("zone", ""))
                        self.wash_counter.total_count += 1
                        self.wash_counter.zone_total[_z] = self.wash_counter.zone_total.get(_z, 0) + 1
                        if bool(getattr(self, "fsm_enabled", False)) and hasattr(self, "service_fsm"):
                            self.service_fsm.wash_total = int(self.service_fsm.wash_total) + 1
                            self.service_fsm.wash_zone_total[_z] = int(self.service_fsm.wash_zone_total.get(_z, 0)) + 1
                    logger.info(f"Chair-return wash events: {len(chair_return_washes)}")
                    wash_events = list(wash_events) + chair_return_washes
                self._remember_service_events(haircut_events, "haircut")
                self._remember_service_events(wash_events, "wash")

                chairs_by_zone: Dict[str, set] = {}
                washes_by_zone: Dict[str, set] = {}
                wait_pids = set()
                for d in service_merged:
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
                self.live_customers = len(set(d["pid"] for d in service_merged))
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
                staged_events = []
                for ev in haircut_events:
                    e = Event(
                        timestamp=datetime.now(),
                        camera=ev.get("camera", ""),
                        event_type=EventType.CHAIR,
                        person_id=int(ev.get("pid", 0)),
                        zone_name=ev.get("zone", ""),
                        dwell_seconds=float(ev.get("dwell", 0.0)),
                        metadata={
                            "wait_duration": float(ev.get("wait_duration", 0.0)),
                            "gid": int(ev.get("gid", 0)),
                            "bbox": ev.get("bbox"),
                            "det_conf": float(ev.get("conf", 0.0)),
                            "det_type": ev.get("det_type"),
                            "det_ts": float(ev.get("det_ts", 0.0)),
                            "det_stale": bool(ev.get("det_stale", False)),
                            "frame_w": int(ev.get("frame_w", 0) or 0),
                            "frame_h": int(ev.get("frame_h", 0) or 0),
                            "reason_code": str(ev.get("reason_code", "")),
                            "from_state": str(ev.get("from_state", "")),
                            "to_state": str(ev.get("to_state", "")),
                            "session_state": str(ev.get("session_state", "")),
                            "fsm_transition_id": int(ev.get("transition_id", 0) or 0),
                            "fsm_zone_dwell_sec": float(ev.get("zone_dwell_sec", 0.0) or 0.0),
                        },
                    )
                    try:
                        snap = self._capture_event_snapshot(e)
                        if snap:
                            e.metadata["snapshot_path"] = snap
                    except Exception:
                        pass
                    staged_events.append(e)
                for ev in wash_events:
                    e = Event(
                        timestamp=datetime.now(),
                        camera=ev.get("camera", ""),
                        event_type=EventType.WASH,
                        person_id=int(ev.get("pid", 0)),
                        zone_name=ev.get("zone", ""),
                        dwell_seconds=float(ev.get("dwell", 0.0)),
                        metadata={
                            "gid": int(ev.get("gid", 0)),
                            "bbox": ev.get("bbox"),
                            "det_conf": float(ev.get("conf", 0.0)),
                            "det_type": ev.get("det_type"),
                            "det_ts": float(ev.get("det_ts", 0.0)),
                            "det_stale": bool(ev.get("det_stale", False)),
                            "frame_w": int(ev.get("frame_w", 0) or 0),
                            "frame_h": int(ev.get("frame_h", 0) or 0),
                            "reason_code": str(ev.get("reason_code", "")),
                            "from_state": str(ev.get("from_state", "")),
                            "to_state": str(ev.get("to_state", "")),
                            "session_state": str(ev.get("session_state", "")),
                            "fsm_transition_id": int(ev.get("transition_id", 0) or 0),
                            "fsm_zone_dwell_sec": float(ev.get("zone_dwell_sec", 0.0) or 0.0),
                        },
                    )
                    try:
                        snap = self._capture_event_snapshot(e)
                        if snap:
                            e.metadata["snapshot_path"] = snap
                    except Exception:
                        pass
                    staged_events.append(e)
                with self.pending_lock:
                    self.pending_events.extend(staged_events)

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
            self.service_fsm.reset()
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
            self.recent_service_identity_events = []
            self._chair_zone_occupied_keys = set()
            self._chair_zone_last_vacant_ts = {}
            self.pid_role_votes = {}
            self.pid_role_last_ts = {}
            self.role_vote_total_frames = 0
            self.role_vote_unknown_frames = 0
            self.fsm_state_counts = {}
            self.fsm_recent_transitions = []
            with self.verification_state_lock:
                self.verified_events_total = 0
                self.verified_by_camera = {}
                self.last_verified_event = None
            self._runtime_day = datetime.now().strftime("%Y-%m-%d")
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
            self._write_dashboard_state(force=True)
        except Exception as e:
            logger.error(f"Control command error: {e}")
    
    def _maybe_store_latest_frame(
        self,
        camera_name: str,
        frame: Optional[np.ndarray],
        now_ts: Optional[float] = None,
        force: bool = False,
    ):
        """Store dashboard frame at a capped cadence to reduce copy overhead."""
        if frame is None:
            return
        now_ts = float(now_ts if now_ts is not None else time.time())
        if not force:
            last_ui = float(self._last_ui_update_ts.get(camera_name, 0.0))
            if (now_ts - last_ui) < self.dashboard_ui_frame_interval_sec:
                return
        self.latest_frames[camera_name] = frame.copy()
        self._last_ui_update_ts[camera_name] = now_ts

    @staticmethod
    def _latency_stats(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"avg": 0.0, "p95": 0.0, "max": 0.0, "samples": 0}
        arr = np.array(values, dtype=np.float32)
        return {
            "avg": float(np.mean(arr)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(np.max(arr)),
            "samples": int(arr.size),
        }

    def _auto_manage_inference_mode(self, now_ts: Optional[float] = None):
        """Auto switch always<->motion_gated based on runtime CPU/MEM pressure."""
        now_ts = float(now_ts if now_ts is not None else time.time())
        configured = str(self._configured_inference_mode or self.inference_mode or "always").strip().lower()
        if configured not in {"always", "motion_gated"}:
            configured = "always"
            self._configured_inference_mode = configured

        if (not self.auto_motion_gate_enabled) or configured != "always":
            if self._auto_motion_gate_active:
                self._auto_motion_gate_active = False
            if self.inference_mode != configured:
                self.inference_mode = configured
            return

        metrics = self.resource_guard.get_latest_metrics(max_age_sec=max(1.0, self.resource_guard.check_interval * 2.0))
        cpu_pct = float(getattr(metrics, "cpu_percent", 0.0))
        mem_pct = float(getattr(metrics, "memory_percent", 0.0))
        overload = (
            cpu_pct >= float(self.auto_motion_gate_cpu_threshold)
            or mem_pct >= float(self.auto_motion_gate_mem_threshold)
            or self.resource_guard.should_throttle()
        )

        if overload:
            self._auto_motion_last_overload_ts = now_ts
            if self.inference_mode != "motion_gated":
                self.inference_mode = "motion_gated"
                self._auto_motion_gate_active = True
                self._auto_motion_last_switch_ts = now_ts
                logger.warning(
                    "Auto-switch inference mode -> motion_gated "
                    f"(cpu={cpu_pct:.1f}% mem={mem_pct:.1f}% thresholds="
                    f"{self.auto_motion_gate_cpu_threshold:.1f}/{self.auto_motion_gate_mem_threshold:.1f})"
                )
            return

        if self._auto_motion_gate_active and self.inference_mode == "motion_gated":
            hold_elapsed = now_ts - float(self._auto_motion_last_switch_ts)
            recover_elapsed = now_ts - float(self._auto_motion_last_overload_ts)
            if hold_elapsed >= float(self.auto_motion_gate_min_hold_sec) and recover_elapsed >= float(self.auto_motion_gate_recover_sec):
                self.inference_mode = "always"
                self._auto_motion_gate_active = False
                self._auto_motion_last_switch_ts = now_ts
                logger.info(
                    "Auto-switch inference mode -> always "
                    f"(cpu={cpu_pct:.1f}% mem={mem_pct:.1f}% recovered)"
                )

    def process_frame(self, camera_name: str, frame: np.ndarray):
        """Process frame with YOLO"""
        if frame is None or frame.size == 0:
            return
        process_start_pc = time.perf_counter()
        infer_ms = 0.0
        
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
                infer_start_pc = time.perf_counter()
                detect_classes = self.yolo_detect_class_ids if self.yolo_detect_class_ids else None
                if self.yolo_mode == "track":
                    try:
                        results = self.discovery_model.track(
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
                        results = self.discovery_model.predict(
                            frame,
                            conf=self.yolo_conf,
                            iou=self.yolo_iou,
                            imgsz=self.yolo_imgsz,
                            classes=detect_classes,
                            verbose=False
                        )
                else:
                    # Predict mode is more stable on CPU-heavy environments.
                    results = self.discovery_model.predict(
                        frame,
                        conf=self.yolo_conf,
                        iou=self.yolo_iou,
                        imgsz=self.yolo_imgsz,
                        classes=detect_classes,
                        verbose=False
                    )
                infer_ms = (time.perf_counter() - infer_start_pc) * 1000.0
            finally:
                self.yolo_lock.release()

            now_ts = time.time()
            zones = self.zones.get(camera_name, {})
            detections: List[Detection] = []
            customers: List[Dict] = []
            all_detections: List[Dict] = []
            raw_people = 0
            raw_heads = 0
            raw_staff_uniform = 0
            staff_filtered = 0
            emb_ready = 0
            head_rejected_size = 0
            head_rejected_live = 0

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
                        else self.discovery_person_class_id
                    )
                    if (
                        self.discovery_staff_uniform_class_id is not None
                        and cls_id == int(self.discovery_staff_uniform_class_id)
                    ):
                        x1s, y1s, x2s, y2s = box.cpu().numpy().tolist()
                        raw_staff_uniform += 1
                        staff_uniform_boxes.append([float(x1s), float(y1s), float(x2s), float(y2s)])

                for i, box in enumerate(xyxy_list):
                    cls_id = (
                        int(float(cls_vals[i].item()))
                        if cls_vals is not None and i < len(cls_vals)
                        else self.discovery_person_class_id
                    )
                    allowed_cls = {int(self.discovery_person_class_id)}
                    if self.discovery_head_class_id is not None:
                        allowed_cls.add(int(self.discovery_head_class_id))
                    if cls_id not in allowed_cls:
                        continue
                    x1, y1, x2, y2 = box.cpu().numpy().tolist()
                    conf = float(boxes.conf[i]) if boxes.conf is not None else 0.0
                    is_head = (
                        self.discovery_head_class_id is not None
                        and cls_id == int(self.discovery_head_class_id)
                    )
                    if is_head:
                        raw_heads += 1
                    else:
                        raw_people += 1
                    if is_head and self.enable_head_size_guard:
                        bw_px = max(0.0, float(x2 - x1))
                        bh_px = max(0.0, float(y2 - y1))
                        width_ratio = bw_px / float(max(w, 1))
                        height_ratio = bh_px / float(max(h, 1))
                        area_ratio = (bw_px * bh_px) / float(max(w * h, 1))
                        if (
                            width_ratio > self.head_max_width_ratio
                            or height_ratio > self.head_max_height_ratio
                            or area_ratio > self.head_max_area_ratio
                        ):
                            head_rejected_size += 1
                            continue
                    if is_head:
                        head_live_reason = self._head_live_reject_reason(x1, y1, x2, y2, conf, w, h)
                        if head_live_reason:
                            head_rejected_live += 1
                            continue
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
                    role_id = None
                    role_score = -1.0
                    role_scores: Dict[str, float] = {}
                    customer_override_staff = False
                    if self.enable_staff_uniform and staff_uniform_boxes:
                        cur_box = [float(x1), float(y1), float(x2), float(y2)]
                        for sb in staff_uniform_boxes:
                            if self._bbox_iou_xyxy(cur_box, sb) >= self.staff_uniform_iou_threshold:
                                staff_id = "staff_uniform"
                                break

                    need_embedding = self.enable_reid or (self.staff_gallery is not None) or (self.role_gallery is not None)
                    if need_embedding and self.reid_encoder is not None:
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
                                if staff_id is None and self.staff_gallery is not None:
                                    staff_id = self.staff_gallery.match_staff(emb, self.staff_similarity_threshold)
                                role_id, role_score, role_scores = self._classify_role_from_embedding(emb)
                                if role_id == "barber_uniform" and staff_id is None:
                                    staff_id = "role_barber_db"
                                elif (
                                    role_id == "customer"
                                    and staff_id is not None
                                    and self.role_db_customer_override_staff
                                ):
                                    staff_id = None
                                    customer_override_staff = True

                    # CHAIR-specific recovery before PID persistence:
                    # if this looks like a seated customer with ambiguous barber score,
                    # clear role-derived staff tags so staff state does not stick.
                    pre_det = {
                        "primary_zone": primary_zone,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "frame_w": int(w),
                        "frame_h": int(h),
                        "staff_id": staff_id,
                        "role_barber_score": float(role_scores.get("barber_uniform", -1.0)),
                        "role_customer_score": float(role_scores.get("customer", -1.0)),
                    }
                    if self._chair_customer_recovery_applies(pre_det):
                        sid_txt = str(staff_id or "").strip().lower()
                        if (not sid_txt) or sid_txt.startswith("role_") or sid_txt == "staff_uniform":
                            staff_id = None
                            customer_override_staff = True
                            role_id = "customer"
                    # Move staff check after pid resolution for persistence
                    # (but keep detection logic same)

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
                        "conf": float(conf),
                        "zones": zone_hits,
                        "primary_zone": primary_zone,
                        "staff_id": staff_id,
                        "role_id": role_id,
                        "role_score": float(role_score),
                        "role_barber_score": float(role_scores.get("barber_uniform", -1.0)),
                        "role_customer_score": float(role_scores.get("customer", -1.0)),
                        "det_type": "head" if is_head else "person",
                        "stale": False,
                        "frame_w": int(w),
                        "frame_h": int(h),
                        "ts": now_ts,
                        "emb": emb,
                    }
                    gid = self.global_id_manager.assign_gid(det)
                    pid = gid
                    appearance_tag = "default"
                    pz_tag = str(primary_zone or "")
                    if staff_id is not None:
                        appearance_tag = "barber"
                    elif pz_tag.startswith(WASH_ZONE_PREFIX):
                        appearance_tag = "wash"
                    elif pz_tag.startswith(CHAIR_ZONE_PREFIX):
                        appearance_tag = "chair"
                    if self.enable_reid and self.reid_merge_for_counts:
                        pid = self.active_person_memory.resolve_pid(
                            gid,
                            emb,
                            now_ts,
                            self.reid_similarity_threshold,
                            self.reid_match_window_sec,
                            staff_id=staff_id,
                            clear_staff_id=customer_override_staff,
                            role_id=role_id,
                            role_score=role_score,
                            appearance_tag=appearance_tag,
                        )
                    
                    # Persistence check from memory
                    mem_info = self.active_person_memory.pid_last.get(pid, {})
                    p_staff_id = mem_info.get("staff_id", staff_id)
                    det["gid"] = int(gid)
                    det["pid"] = int(pid)
                    det["staff_id"] = p_staff_id
                    det["role_id"] = mem_info.get("role_id", role_id)
                    det["role_score"] = float(mem_info.get("role_score", role_score))

                    frame_role_vote = self._frame_role_vote(
                        staff_id=det.get("staff_id"),
                        role_id=det.get("role_id"),
                        role_barber_score=float(det.get("role_barber_score", -1.0) or -1.0),
                        role_customer_score=float(det.get("role_customer_score", -1.0) or -1.0),
                        conf=float(det.get("conf", 0.0) or 0.0),
                        bbox=det.get("bbox") or [],
                        frame_w=int(det.get("frame_w", 0) or 0),
                        frame_h=int(det.get("frame_h", 0) or 0),
                        primary_zone=str(det.get("primary_zone", "") or ""),
                    )
                    stable_role, stable_ratio, stable_samples = self._update_pid_role_vote(
                        int(pid),
                        frame_role_vote,
                        now_ts,
                    )
                    det["role_vote"] = stable_role
                    det["role_vote_ratio"] = float(stable_ratio)
                    det["role_vote_samples"] = int(stable_samples)
                    self.role_vote_total_frames += 1
                    if stable_role == "unknown":
                        self.role_vote_unknown_frames += 1

                    if stable_role == "barber":
                        det["role_id"] = "barber_uniform"
                        if det.get("staff_id") is None:
                            det["staff_id"] = "role_vote_barber"
                    elif stable_role == "customer":
                        det["role_id"] = "customer"
                    else:
                        det["role_id"] = "unknown"

                    if self._chair_customer_recovery_applies(det):
                        sid_txt = str(det.get("staff_id", "") or "").strip().lower()
                        if (not sid_txt) or sid_txt.startswith("role_") or sid_txt == "staff_uniform":
                            det["staff_id"] = None
                            det["role_id"] = "customer"
                            det["role_vote"] = "customer"
                            det["role_vote_ratio"] = max(
                                float(det.get("role_vote_ratio", 0.0) or 0.0),
                                float(self.role_vote_min_ratio),
                            )
                            det["role_vote_samples"] = max(
                                int(det.get("role_vote_samples", 0) or 0),
                                int(self.role_vote_min_samples),
                            )

                    try:
                        mem_info["role_id"] = det.get("role_id")
                        mem_info["role_vote"] = det.get("role_vote", stable_role)
                        mem_info["role_vote_ratio"] = float(det.get("role_vote_ratio", stable_ratio) or stable_ratio)
                        mem_info["role_vote_samples"] = int(det.get("role_vote_samples", stable_samples) or stable_samples)
                        if det.get("staff_id") is not None:
                            mem_info["staff_id"] = det.get("staff_id")
                        elif self._chair_customer_recovery_applies(det):
                            old_sid = str(mem_info.get("staff_id", "") or "").strip().lower()
                            if (not old_sid) or old_sid.startswith("role_") or old_sid == "staff_uniform":
                                mem_info["staff_id"] = None
                    except Exception:
                        pass
                    self._capture_role_snapshot(camera_name, frame, det, now_ts=now_ts)
                    all_detections.append(det)

                    if self._is_staff_for_service_counts(det) and self.exclude_staff_from_counts:
                        staff_filtered += 1
                        self.staff_filtered_total += 1 # Incremental filtered count
                        continue

                    customers.append(det)

            self.tracker.update(camera_name, detections)
            if (raw_people + raw_heads) > 0:
                self._last_camera_nonzero[camera_name] = now_ts
            with self.customers_lock:
                if all_detections:
                    self.latest_detections[camera_name] = all_detections
                else:
                    prev_all = self.latest_detections.get(camera_name, [])
                    kept_all = []
                    for d in prev_all:
                        ts = float(d.get("ts", 0.0))
                        if ts > 0 and (now_ts - ts) <= self.dashboard_presence_ttl_sec:
                            d["stale"] = bool(d.get("stale", False))
                            kept_all.append(d)
                    self.latest_detections[camera_name] = kept_all
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
                            if hold_stale and (self.hold_stale_for_head or str(d.get("det_type", "")) != "head"):
                                nd = dict(d)
                                nd["ts"] = now_ts
                                nd["stale"] = True
                                kept.append(nd)
                            elif not hold_stale:
                                d["stale"] = bool(d.get("stale", False))
                                kept.append(d)
                    self.latest_customers[camera_name] = kept
                self._maybe_store_latest_frame(camera_name, frame, now_ts=now_ts)

            last_log = float(self._last_det_stats_log_ts.get(camera_name, 0.0))
            if (now_ts - last_log) >= 5.0:
                self._last_det_stats_log_ts[camera_name] = now_ts
                logger.info(
                    f"{camera_name}: raw_person={raw_people} raw_head={raw_heads} "
                    f"raw_staff_uniform={raw_staff_uniform} accepted={len(customers)} "
                    f"staff_filtered={staff_filtered} head_rejected_size={head_rejected_size} "
                    f"head_rejected_live={head_rejected_live} emb_ready={emb_ready} "
                    f"ids={'ok' if (results and results[0].boxes is not None and results[0].boxes.id is not None) else 'fallback'}"
                )
            process_ms = (time.perf_counter() - process_start_pc) * 1000.0
            with self._latency_lock:
                if infer_ms > 0.0:
                    self._infer_latency_ms.append(float(infer_ms))
                self._process_latency_ms.append(float(process_ms))
        
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
                
                # OPTIMIZATION: Skip UI frame updates if they are happening too fast.
                # Configurable cap helps reduce memory bandwidth and CPU usage.
                min_ui_interval = float(self.dashboard_ui_frame_interval_sec)
                
                if (now_ts - last_proc) < min_process_interval:
                    # Skip inference
                    
                    # Check if we should update UI frame
                    if (now_ts - float(self._last_ui_update_ts.get(camera_name, 0.0))) >= min_ui_interval:
                        with self.customers_lock:
                            self._maybe_store_latest_frame(camera_name, frame, now_ts=now_ts)

                    # short yield, but do not sleep by target_fps here
                    time.sleep(0.002)
                    continue
                self._last_process_ts[camera_name] = now_ts

                should_infer, infer_reason = self._should_run_inference(camera_name, frame, now_ts)
                if should_infer:
                    self.process_frame(camera_name, frame)
                    self._maybe_start_tier2_capture(camera_name, time.time())
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
                        self._maybe_store_latest_frame(camera_name, frame, now_ts=now_ts)
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

    def _spawn_camera_worker(self, camera_name: str, generation: Optional[int] = None):
        """Spawn one camera worker thread bound to a generation id."""
        if generation is None:
            generation = int(self._camera_generation.get(camera_name, 0))
        worker = threading.Thread(
            target=self.run_camera_thread,
            args=(camera_name, int(generation)),
            daemon=True,
            name=f"cam_{camera_name}_g{int(generation)}",
        )
        worker.start()
        self.threads.append(worker)

    def _restart_camera_worker(self, camera_name: str, reason: str = "manual") -> bool:
        """Recreate camera stream and start a fresh worker generation."""
        now_ts = time.time()
        prev_backoff = float(self._restart_backoff_sec.get(camera_name, 0.0))
        # Exponential backoff (1s -> 2s -> 4s ... capped) to avoid restart loops.
        next_backoff = max(1.0, prev_backoff * 2.0) if prev_backoff > 0 else 1.0
        next_backoff = min(next_backoff, 60.0)

        with self._camera_lock:
            old_stream = self.cameras.get(camera_name)
            url = ""
            cfg = CAMERAS_CONFIG.get(camera_name, {})
            if isinstance(cfg, dict):
                url = str(cfg.get("rtsp_url", "") or "")
            elif isinstance(cfg, str):
                url = str(cfg)
            if (not url) and old_stream is not None:
                url = str(getattr(old_stream, "rtsp_url", "") or "")
            if not url:
                logger.error(f"{camera_name}: cannot restart worker, missing RTSP URL (reason={reason})")
                return False

            self._camera_generation[camera_name] = int(self._camera_generation.get(camera_name, 0)) + 1
            gen = int(self._camera_generation[camera_name])
            self._restart_backoff_sec[camera_name] = next_backoff
            self._restart_backoff_until[camera_name] = now_ts + next_backoff
            self._camera_good_frame_count[camera_name] = 0

            connect_policy = dict(
                self._camera_connect_policy.get(
                    camera_name,
                    {
                        "prefer_tcp": bool(RUNTIME_CONFIG.get("rtsp_prefer_tcp", False)),
                        "force_ffmpeg": bool(RUNTIME_CONFIG.get("rtsp_force_ffmpeg", False)),
                        "allow_udp_fallback": bool(RUNTIME_CONFIG.get("rtsp_allow_udp_fallback", False)),
                        "rotate_backends": bool(RUNTIME_CONFIG.get("rtsp_rotate_backends", False)),
                    },
                )
            )

            try:
                if old_stream is not None:
                    old_stream.disconnect()
            except Exception:
                logger.exception(f"{camera_name}: old stream disconnect failed during restart")

            self.cameras[camera_name] = CameraStream(
                camera_name,
                url,
                freeze_max_same_frames=self.camera_freeze_max_same_frames,
                freeze_diff_threshold=self.camera_freeze_diff_threshold,
                connect_policy=connect_policy,
                verification_queue_ref=self.verification_queue,
                clip_duration_sec=self.tier2_clip_duration_sec,
            )
            self._last_frame_ok_ts[camera_name] = now_ts

        logger.info(
            f"{camera_name}: camera worker restarted (gen={gen}, reason={reason}, backoff={next_backoff:.1f}s)"
        )
        self._spawn_camera_worker(camera_name, gen)
        return True

    
    def submit_events_loop(self):
        """Background thread to submit events"""
        logger.info("Event submission thread started")
        
        while self.running:
            try:
                self._reset_daily_counters_if_needed()
                self._check_control_commands()
                with self.pending_lock:
                    events = self.pending_events
                    self.pending_events = []
                
                if events:
                    event_dicts = []
                    for e in events:
                        snap = ""
                        if isinstance(e.metadata, dict):
                            snap = str(e.metadata.get("snapshot_path", "") or "")
                        if (not snap) or (not Path(snap).exists()):
                            snap = self._capture_event_snapshot(e)
                        if snap:
                            e.metadata["snapshot_path"] = snap
                            gid_val = int(e.metadata.get("gid", 0)) if isinstance(e.metadata, dict) else 0
                            self._remember_snapshot(
                                snap,
                                e.camera,
                                int(e.person_id),
                                e.event_type.value,
                                gid=gid_val,
                            )
                            feedback_path = self._copy_snapshot_to_event_feedback(
                                snap,
                                e.event_type.value,
                                e.camera,
                                int(e.person_id),
                                gid=gid_val,
                            )
                            if feedback_path:
                                e.metadata["feedback_snapshot_path"] = str(feedback_path)
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
                    "verified": self.verified_events_total,
                    "total_events": self._today_event_count(),
                }
                current_time = time.time()
                if (current_time - float(self._last_summary_log_ts)) >= self.summary_log_interval_sec:
                    logger.info(f"Summary: {summary}")
                    self._last_summary_log_ts = current_time
                
                if current_time - self.last_summary_broadcast > 5.0:
                    self._broadcast_summary(summary)
                    self.last_summary_broadcast = current_time
                self._write_dashboard_state(status_override=status_now)
                
                # Explicit GC on a slower cadence to avoid periodic CPU spikes.
                if (current_time - float(self._last_gc_collect_ts)) >= self.gc_collect_interval_sec:
                    gc.collect()
                    self._last_gc_collect_ts = current_time
                time.sleep(10)
            
            except Exception as e:
                logger.error(f"Error in event submission loop: {e}")
                time.sleep(5)
    
    def _load_yolo(self) -> Tuple[YOLO, YOLO, str]:
        """Load YOLO models (Discovery + Verification) and detect best device."""
        
        device = str(YOLO_CONFIG.get("device", "auto")).lower().strip()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
                
        logger.info(
            f"Loading Tiered Models on {device} "
            f"(discovery={YOLO_DISCOVERY_MODEL_PATH}, worker={YOLO_VERIFY_MODEL_PATH})..."
        )
        
        try:
            verify_path = YOLO_VERIFY_MODEL_PATH
            # Load Discovery Model (Nano)
            disc_path = YOLO_DISCOVERY_MODEL_PATH
            if not os.path.exists(disc_path):
                disc_path = verify_path if os.path.exists(verify_path) else "yolov8n.pt"
            discovery = YOLO(disc_path)
            discovery.to(device)
            
            # Load Verification Model (Best/Medium)
            verify = YOLO(verify_path)
            verify.to(device)
            
            logger.info("Discovery and Verification models loaded.")
            return discovery, verify, device
            
        except Exception as e:
            logger.error(f"Failed to load YOLO models: {e}")
            raise e

    def _configure_discovery_class_schema(self):
        """Resolve discovery class IDs against loaded model labels.

        Supports two modes:
        - salon custom model (person/head/staff_uniform)
        - generic COCO model (person only)
        """
        try:
            def _as_bool(v, default=False):
                if v is None:
                    return bool(default)
                if isinstance(v, str):
                    return v.strip().lower() in {"1", "true", "yes", "on"}
                return bool(v)

            names = _model_names_dict(self.discovery_model)
            verify_names = _model_names_dict(self.verify_model)

            pid_cfg = int(self.person_class_id)
            hid_cfg = int(self.head_class_id)
            sid_cfg = int(self.staff_uniform_class_id)

            disc_person_name = names.get(pid_cfg, "")
            disc_head_name = names.get(hid_cfg, "")
            disc_staff_name = names.get(sid_cfg, "")
            disc_has_person = _label_contains_tokens(disc_person_name, ["person"])
            disc_has_head = _label_contains_tokens(disc_head_name, ["head"])
            disc_has_staff = _label_contains_tokens(disc_staff_name, ["staff", "uniform"])

            auto_switch = _as_bool(
                YOLO_CONFIG.get("auto_switch_discovery_on_schema_mismatch", False),
                False,
            )
            mismatch = not (disc_has_person and disc_has_head and disc_has_staff)
            if mismatch:
                logger.warning(
                    "Discovery model class schema mismatch for configured ids "
                    f"(person={pid_cfg}:{disc_person_name or 'N/A'}, "
                    f"head={hid_cfg}:{disc_head_name or 'N/A'}, "
                    f"staff={sid_cfg}:{disc_staff_name or 'N/A'})."
                )
                if auto_switch:
                    verify_head_name = verify_names.get(hid_cfg, "")
                    verify_staff_name = verify_names.get(sid_cfg, "")
                    verify_has_head = _label_contains_tokens(verify_head_name, ["head"])
                    verify_has_staff = _label_contains_tokens(verify_staff_name, ["staff", "uniform"])
                    if verify_has_head and verify_has_staff:
                        logger.warning(
                            "Auto-switching discovery model to worker model for salon classes "
                            f"(auto_switch_discovery_on_schema_mismatch=true): {YOLO_VERIFY_MODEL_PATH}"
                        )
                        switched = YOLO(YOLO_VERIFY_MODEL_PATH)
                        switched.to(self.yolo_device)
                        self.discovery_model = switched
                        names = _model_names_dict(self.discovery_model)
                    else:
                        logger.warning(
                            "Cannot auto-switch: worker model does not expose expected head/staff labels."
                        )
                else:
                    logger.info(
                        "Keeping discovery model as-is (person-priority mode). "
                        "Head/Uniform class detection will be disabled if labels are absent."
                    )

            def _find_first(tokens: List[str]) -> Optional[int]:
                for cid, nm in sorted(names.items(), key=lambda kv: int(kv[0])):
                    if _label_contains_tokens(nm, tokens):
                        return int(cid)
                return None

            person_id = pid_cfg if _label_contains_tokens(names.get(pid_cfg, ""), ["person"]) else _find_first(["person"])
            head_id = hid_cfg if _label_contains_tokens(names.get(hid_cfg, ""), ["head"]) else _find_first(["head"])
            staff_id = sid_cfg if _label_contains_tokens(names.get(sid_cfg, ""), ["staff", "uniform"]) else _find_first(["staff", "uniform"])

            if person_id is None:
                person_id = pid_cfg

            self.discovery_person_class_id = int(person_id)
            self.discovery_head_class_id = int(head_id) if head_id is not None else None
            self.discovery_staff_uniform_class_id = int(staff_id) if staff_id is not None else None

            detect_ids: List[int] = [int(self.discovery_person_class_id)]
            if self.discovery_head_class_id is not None:
                detect_ids.append(int(self.discovery_head_class_id))
            if self.enable_staff_uniform and self.discovery_staff_uniform_class_id is not None:
                detect_ids.append(int(self.discovery_staff_uniform_class_id))
            self.yolo_detect_class_ids = sorted(set(detect_ids))

            logger.info(
                "Discovery class mapping: "
                f"person={self.discovery_person_class_id}, "
                f"head={self.discovery_head_class_id}, "
                f"staff_uniform={self.discovery_staff_uniform_class_id}, "
                f"detect_ids={self.yolo_detect_class_ids}"
            )
        except Exception as e:
            logger.warning(f"Failed to configure discovery class schema: {e}")

    def _load_chair_service_classifier(self):
        """Load optional chair-zone classifier used to confirm haircut events."""
        self.chair_service_classifier = None
        self.chair_service_classifier_device = "cpu"
        if not bool(self.enable_chair_service_classifier):
            logger.info("Chair service classifier disabled")
            return
        try:
            model_path = resolve_model_path(
                self.chair_service_classifier_model,
                "models/chair_service_cls.pt",
            )
            if not os.path.exists(model_path):
                logger.warning(
                    f"Chair service classifier model not found: {model_path} "
                    "(haircut counts will use non-classifier logic)"
                )
                return
            model = YOLO(model_path)
            cls_device = self.yolo_device if self.yolo_device != "mps" else "cpu"
            model.to(cls_device)
            self.chair_service_classifier = model
            self.chair_service_classifier_device = cls_device
            cls_names = _model_names_dict(model)
            logger.info(
                f"Chair service classifier loaded: path={model_path}, device={cls_device}, "
                f"labels={cls_names}, positive={sorted(self.chair_service_classifier_positive_labels)}, "
                f"min_conf={self.chair_service_classifier_min_conf:.2f}, imgsz={self.chair_service_classifier_imgsz}"
            )
        except Exception as e:
            self.chair_service_classifier = None
            logger.warning(f"Failed to load chair service classifier: {e}")

    def _resolve_project_path(self, path_like: str) -> Path:
        p = Path(str(path_like or "")).expanduser()
        if p.is_absolute():
            return p
        return (PROJECT_ROOT / p).resolve()

    def _feedback_image_stats(self, root_dir: Path) -> Tuple[int, float]:
        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".jp2"}
        if (not root_dir.exists()) or (not root_dir.is_dir()):
            return 0, 0.0
        count = 0
        newest_ts = 0.0
        try:
            for fp in root_dir.rglob("*"):
                if (not fp.is_file()) or (fp.suffix.lower() not in img_exts):
                    continue
                count += 1
                try:
                    mt = float(fp.stat().st_mtime)
                except Exception:
                    mt = 0.0
                if mt > newest_ts:
                    newest_ts = mt
        except Exception:
            return 0, 0.0
        return int(count), float(newest_ts)

    def _current_chair_service_feedback_signature(self) -> Tuple[str, int, int]:
        pos_dir = self._resolve_project_path(self.chair_service_autotrain_positive_dir)
        neg_dirs = [self._resolve_project_path(x) for x in (self.chair_service_autotrain_negative_dirs or [])]
        pos_count, pos_mtime = self._feedback_image_stats(pos_dir)
        neg_total = 0
        neg_mtime = 0.0
        parts = [f"pos:{pos_dir}:{pos_count}:{int(pos_mtime)}"]
        for nd in neg_dirs:
            n_count, n_mtime = self._feedback_image_stats(nd)
            neg_total += int(n_count)
            neg_mtime = max(neg_mtime, float(n_mtime))
            parts.append(f"neg:{nd}:{n_count}:{int(n_mtime)}")
        signature = "|".join(parts)
        self._chair_service_autotrain_last_data_counts = {
            "positive": int(pos_count),
            "negative": int(neg_total),
        }
        return signature, int(pos_count), int(neg_total)

    def _load_chair_service_autotrain_state(self):
        self._chair_service_autotrain_last_signature = ""
        self._chair_service_autotrain_last_train_ts = 0.0
        self._chair_service_autotrain_last_status = "idle"
        self._chair_service_autotrain_last_error = ""
        if not self.chair_service_autotrain_state_file.exists():
            return
        try:
            raw = json.loads(self.chair_service_autotrain_state_file.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return
            self._chair_service_autotrain_last_signature = str(raw.get("last_signature", "") or "")
            self._chair_service_autotrain_last_train_ts = float(raw.get("last_train_ts", 0.0) or 0.0)
            self._chair_service_autotrain_last_status = str(raw.get("last_status", "idle") or "idle")
            self._chair_service_autotrain_last_error = str(raw.get("last_error", "") or "")
        except Exception as e:
            logger.debug(f"Failed loading chair classifier autotrain state: {e}")

    def _save_chair_service_autotrain_state(self):
        try:
            self.chair_service_autotrain_state_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "updated_at": datetime.now().isoformat(),
                "last_signature": self._chair_service_autotrain_last_signature,
                "last_train_ts": float(self._chair_service_autotrain_last_train_ts),
                "last_status": self._chair_service_autotrain_last_status,
                "last_error": self._chair_service_autotrain_last_error,
                "data_counts": dict(self._chair_service_autotrain_last_data_counts or {}),
                "model_path": str(self.chair_service_classifier_model),
            }
            self.chair_service_autotrain_state_file.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.debug(f"Failed saving chair classifier autotrain state: {e}")

    def _start_chair_service_autotrain_thread(self):
        if not self.chair_service_autotrain_enabled:
            return
        if self._chair_service_autotrain_thread and self._chair_service_autotrain_thread.is_alive():
            return
        th = threading.Thread(
            target=self._chair_service_autotrain_loop,
            daemon=True,
            name="chair_cls_autotrain",
        )
        th.start()
        self._chair_service_autotrain_thread = th
        self.threads.append(th)
        logger.info(
            "Chair classifier autotrain thread started: interval=%.2fh, min_pos=%d, min_neg=%d",
            self.chair_service_autotrain_interval_hours,
            self.chair_service_autotrain_min_positive,
            self.chair_service_autotrain_min_negative,
        )

    def _chair_service_autotrain_loop(self):
        self._chair_service_autotrain_next_check_ts = time.time() + 10.0
        while self.running:
            try:
                if not self.chair_service_autotrain_enabled:
                    time.sleep(5.0)
                    continue
                now_ts = time.time()
                if now_ts < float(self._chair_service_autotrain_next_check_ts):
                    time.sleep(2.0)
                    continue
                interval_sec = max(900.0, float(self.chair_service_autotrain_interval_hours) * 3600.0)
                self._chair_service_autotrain_next_check_ts = now_ts + min(600.0, max(60.0, interval_sec / 6.0))
                self._run_chair_service_autotrain_once(now_ts=now_ts, interval_sec=interval_sec)
            except Exception as e:
                logger.warning(f"Chair classifier autotrain loop error: {e}")
                time.sleep(10.0)

    def _run_chair_service_autotrain_once(self, now_ts: float, interval_sec: float):
        signature, pos_count, neg_count = self._current_chair_service_feedback_signature()
        if pos_count < int(self.chair_service_autotrain_min_positive) or neg_count < int(self.chair_service_autotrain_min_negative):
            self._chair_service_autotrain_last_status = (
                f"waiting_data pos={pos_count}/{self.chair_service_autotrain_min_positive} "
                f"neg={neg_count}/{self.chair_service_autotrain_min_negative}"
            )
            self._chair_service_autotrain_last_error = ""
            self._save_chair_service_autotrain_state()
            return

        model_path = Path(resolve_model_path(self.chair_service_classifier_model, "models/chair_service_cls.pt"))
        model_exists = model_path.exists()
        data_changed = str(signature) != str(self._chair_service_autotrain_last_signature)
        enough_time_elapsed = (now_ts - float(self._chair_service_autotrain_last_train_ts)) >= float(interval_sec)

        if model_exists and (not data_changed):
            self._chair_service_autotrain_last_status = "idle_no_new_feedback"
            self._chair_service_autotrain_last_error = ""
            self._save_chair_service_autotrain_state()
            return

        if model_exists and (not enough_time_elapsed):
            remain = max(0.0, float(interval_sec) - (now_ts - float(self._chair_service_autotrain_last_train_ts)))
            self._chair_service_autotrain_last_status = f"cooldown_{int(remain)}s"
            self._chair_service_autotrain_last_error = ""
            self._save_chair_service_autotrain_state()
            return

        with self._chair_service_autotrain_lock:
            if self._chair_service_autotrain_running:
                return
            self._chair_service_autotrain_running = True
        try:
            script_path = (PROJECT_ROOT / "scripts" / "train_chair_service_classifier.py").resolve()
            if not script_path.exists():
                raise FileNotFoundError(f"Autotrain script not found: {script_path}")

            pos_dir = self._resolve_project_path(self.chair_service_autotrain_positive_dir)
            neg_dirs = [self._resolve_project_path(x) for x in (self.chair_service_autotrain_negative_dirs or [])]
            cmd = [
                sys.executable,
                str(script_path),
                "--positive-dir",
                str(pos_dir),
                "--negative-dirs",
                *[str(x) for x in neg_dirs],
                "--output-model",
                str(self.chair_service_classifier_model),
                "--epochs",
                str(int(self.chair_service_autotrain_epochs)),
                "--imgsz",
                str(int(self.chair_service_autotrain_imgsz)),
                "--batch",
                str(int(self.chair_service_autotrain_batch)),
                "--patience",
                str(int(self.chair_service_autotrain_patience)),
                "--workers",
                str(int(self.chair_service_autotrain_workers)),
                "--train-split",
                str(float(self.chair_service_autotrain_train_split)),
                "--device",
                str(self.chair_service_autotrain_device),
                "--project",
                "runs/classify",
                "--name",
                "chair_service_cls_autotrain",
                "--exist-ok",
            ]
            self._chair_service_autotrain_last_status = "training"
            self._chair_service_autotrain_last_error = ""
            self._save_chair_service_autotrain_state()
            logger.info(
                "Starting chair classifier auto-train: pos=%d neg=%d interval=%.2fh",
                pos_count,
                neg_count,
                self.chair_service_autotrain_interval_hours,
            )
            proc = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=max(600, int(self.chair_service_autotrain_timeout_min) * 60),
            )
            out_tail = "\n".join((proc.stdout or "").splitlines()[-12:])
            err_tail = "\n".join((proc.stderr or "").splitlines()[-12:])
            if proc.returncode != 0:
                raise RuntimeError(
                    f"autotrain rc={proc.returncode}\nSTDOUT:\n{out_tail}\nSTDERR:\n{err_tail}"
                )
            self._chair_service_autotrain_last_signature = str(signature)
            self._chair_service_autotrain_last_train_ts = float(time.time())
            self._chair_service_autotrain_last_status = "trained_ok"
            self._chair_service_autotrain_last_error = ""
            self._save_chair_service_autotrain_state()
            logger.info(f"Chair classifier auto-train done.\n{out_tail}")
            # Hot reload classifier weights in runtime immediately.
            self._load_chair_service_classifier()
        except Exception as e:
            self._chair_service_autotrain_last_status = "train_failed"
            self._chair_service_autotrain_last_error = str(e)
            self._save_chair_service_autotrain_state()
            logger.warning(f"Chair classifier auto-train failed: {e}")
        finally:
            self._chair_service_autotrain_running = False

    def _validate_haircut_with_chair_service_classifier(self, d: dict) -> Tuple[bool, str]:
        """Return False when chair classifier predicts this is not haircut."""
        if (not self.enable_chair_service_classifier) or (self.chair_service_classifier is None):
            return True, "chair_service_classifier_disabled"
        try:
            cam = str(d.get("cam", d.get("camera", "")) or "")
            bbox = d.get("bbox") or []
            if not cam or len(bbox) != 4:
                return False, "chair_cls_no_bbox"
            with self.customers_lock:
                frame = self.latest_frames.get(cam)
                frame = None if frame is None else frame.copy()
            if frame is None:
                return False, "chair_cls_no_frame"

            h, w = frame.shape[:2]
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            if x2 <= x1 or y2 <= y1:
                return False, "chair_cls_bad_bbox"
            crop = frame[y1:y2, x1:x2]
            if crop is None or crop.size == 0:
                return False, "chair_cls_empty_crop"

            with self.chair_service_classifier_lock:
                out = self.chair_service_classifier.predict(
                    source=crop,
                    imgsz=int(self.chair_service_classifier_imgsz),
                    device=self.chair_service_classifier_device,
                    verbose=False,
                )
            if not out:
                return False, "chair_cls_no_result"
            probs = getattr(out[0], "probs", None)
            if probs is None:
                return False, "chair_cls_no_probs"
            top1 = int(getattr(probs, "top1", -1))
            top1conf = float(getattr(probs, "top1conf", 0.0) or 0.0)
            names = _model_names_dict(self.chair_service_classifier)
            label = str(names.get(top1, top1)).strip()
            label_norm = _normalize_label_name(label)
            d["chair_service_cls_label"] = label_norm
            d["chair_service_cls_conf"] = top1conf

            if label_norm not in self.chair_service_classifier_positive_labels:
                return False, f"chair_cls_negative_{label_norm}"
            if top1conf < float(self.chair_service_classifier_min_conf):
                return False, f"chair_cls_conf_low_{top1conf:.2f}"
            return True, "chair_cls_ok"
        except Exception as e:
            now = time.time()
            if (now - float(self._last_chair_service_classifier_log_ts)) >= 5.0:
                self._last_chair_service_classifier_log_ts = now
                logger.warning(f"Chair service classifier inference failed: {e}")
            return False, "chair_cls_error"

    def _snapshot_path(self, event_type: str, camera: str, pid: int, gid: int = 0) -> Path:
        """Generate snapshot path."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_cam = "".join(c for c in camera if c.isalnum() or c in ('_', '-'))
        filename = f"{ts}_{event_type}_{safe_cam}_P{pid}_G{gid}.jpg"
        return Path(SNAPSHOT_DIR) / filename

    def _feedback_folder_for_event(self, event_type: str) -> Optional[Path]:
        et = str(event_type or "").strip().lower()
        if et in {EventType.CHAIR.value, "chair", "haircut"}:
            return self.feedback_haircut_dir
        if et in {EventType.WASH.value, "wash", "customerwash", "customer_wash"}:
            return self.feedback_customerwash_dir
        return None

    def _copy_snapshot_to_event_feedback(
        self,
        snapshot_path: str,
        event_type: str,
        camera: str,
        pid: int,
        gid: int = 0,
    ) -> str:
        """Copy counted-event snapshot into QA feedback folders (haircut/customerwash)."""
        if not self.enable_event_feedback_autocopy:
            return ""
        try:
            src = Path(str(snapshot_path or ""))
            if (not str(src)) or (not src.exists()) or (not src.is_file()):
                return ""

            dst_dir = self._feedback_folder_for_event(event_type)
            if dst_dir is None:
                return ""
            dst_dir.mkdir(parents=True, exist_ok=True)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            safe_cam = "".join(c for c in str(camera or "") if c.isalnum() or c in ("_", "-"))
            event_tag = str(event_type or "").strip().upper() or "EVENT"
            ext = src.suffix if src.suffix else ".jpg"
            filename = f"{ts}_{event_tag}_{safe_cam}_P{int(pid)}_G{int(gid)}{ext}"
            dst = dst_dir / filename
            shutil.copy2(str(src), str(dst))
            return str(dst)
        except Exception as e:
            logger.debug(f"Event feedback snapshot copy failed: {e}")
            return ""

    def _role_snapshot_path(self, role: str, camera: str, pid: int, gid: int = 0) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_cam = "".join(c for c in camera if c.isalnum() or c in ("_", "-"))
        role_tag_map = {
            "barber": "BARBER",
            "customer": "CUSTOMER",
            "wash_customer": "WASH_CUSTOMER",
            "unknown": "UNKNOWN",
        }
        role_tag = role_tag_map.get(role, "UNKNOWN")
        filename = f"{ts}_{role_tag}_{safe_cam}_P{pid}_G{gid}.jpg"
        if role == "barber":
            return self.barber_snapshot_dir / filename
        if role == "customer":
            return self.customer_snapshot_dir / filename
        if role == "wash_customer":
            return self.wash_customer_snapshot_dir / filename
        return self.unknown_snapshot_dir / filename

    def _decide_role_snapshot_role(self, det: Dict) -> str:
        """Choose role bucket for manual-training snapshots."""
        try:
            role_id = str(det.get("role_id", "") or "").strip().lower()
            role_vote = str(det.get("role_vote", "") or "").strip().lower()
            pzone = str(det.get("primary_zone", "") or "")
            zones_hit = [str(z) for z in (det.get("zones") or [])]
            in_wash_zone = bool(pzone.startswith(WASH_ZONE_PREFIX) or any(z.startswith(WASH_ZONE_PREFIX) for z in zones_hit))
            in_chair_zone = bool(pzone.startswith(CHAIR_ZONE_PREFIX) or any(z.startswith(CHAIR_ZONE_PREFIX) for z in zones_hit))

            if role_vote in {"barber", "customer", "unknown"}:
                if role_vote == "barber":
                    return "barber"
                if role_vote == "customer":
                    return "wash_customer" if in_wash_zone else "customer"
                if in_wash_zone:
                    return "wash_customer"
                # For unknown vote, continue with role_id/score/zone heuristics below.

            if det.get("staff_id") is not None:
                return "barber"
            if role_id in {"barber_uniform", "barber", "staff_uniform", "staff"}:
                return "barber"

            barber_score = float(det.get("role_barber_score", -1.0) or -1.0)
            customer_score = float(det.get("role_customer_score", -1.0) or -1.0)
            margin = float(self.role_db_customer_margin)
            if barber_score >= float(self.role_db_barber_threshold):
                if customer_score < float(self.role_db_customer_threshold):
                    return "barber"
                if customer_score < (barber_score + margin):
                    return "barber"

            if role_id in {"wash_customer", "customer_wash", "shampoo_customer"}:
                return "wash_customer"
            if in_wash_zone:
                return "wash_customer"

            role_score = float(det.get("role_score", -1.0) or -1.0)
            if role_id == "customer" and role_score >= float(self.role_db_customer_threshold):
                return "customer"
            if customer_score >= float(self.role_db_customer_threshold):
                if customer_score >= (barber_score + margin):
                    return "customer"

            # Fallback behavior when temporal vote is unavailable.
            if role_id == "customer" or in_chair_zone:
                return "customer"

            return "unknown"
        except Exception:
            return "unknown"

    def _role_snapshot_thumb(self, crop: np.ndarray) -> Optional[np.ndarray]:
        try:
            if crop is None or crop.size == 0:
                return None
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            thumb = cv2.resize(gray, (24, 24), interpolation=cv2.INTER_AREA)
            return thumb.astype(np.float32)
        except Exception:
            return None

    def _reset_role_snapshot_daily_counters_if_needed(self):
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self._role_snapshot_last_day:
            self._role_snapshot_daily_counts = {}
            self._role_snapshot_last_day = today

    def _capture_role_snapshot(self, camera_name: str, frame: np.ndarray, det: Dict, now_ts: Optional[float] = None):
        """Save manual-training snapshots for barber/customer/wash_customer/unknown classification."""
        if not self.enable_role_snapshots:
            return
        try:
            if frame is None or frame.size == 0:
                return
            if str(det.get("det_type", "person")) != "person":
                return
            bbox = det.get("bbox") or []
            if len(bbox) != 4:
                return
            conf = float(det.get("conf", 0.0) or 0.0)
            if conf < self.role_snapshot_min_conf:
                return
            gid = int(det.get("gid", 0) or 0)
            pid = int(det.get("pid", 0) or 0)
            if gid <= 0 and pid <= 0:
                return

            role = self._decide_role_snapshot_role(det)
            if role == "unknown":
                if (not self.role_snapshot_enable_unknown) or self.role_snapshot_unknown_max_per_camera_day <= 0:
                    return
            pzone = str(det.get("primary_zone", "") or "")
            if role == "customer" and self.role_snapshot_require_customer_chair_zone:
                if not pzone.startswith(CHAIR_ZONE_PREFIX):
                    return
            if role == "wash_customer" and not pzone.startswith(WASH_ZONE_PREFIX):
                return

            h, w = frame.shape[:2]
            x1f, y1f, x2f, y2f = [float(v) for v in bbox]
            bw = max(1.0, x2f - x1f)
            bh = max(1.0, y2f - y1f)
            area_ratio = (bw * bh) / float(max(1, w * h))
            if area_ratio < self.role_snapshot_min_area_ratio:
                return

            cx = float(det.get("cx", ((x1f + x2f) * 0.5) / float(max(w, 1))))
            cy = float(det.get("cy", (y2f / float(max(h, 1)))))
            grid_x = int(max(0, min(11, round(cx * 12))))
            grid_y = int(max(0, min(11, round(cy * 12))))
            identity = f"{camera_name}:{pzone or '-'}:g{grid_x}_{grid_y}"
            cooldown_key = f"{role}:{identity}"
            ts_now = float(now_ts if now_ts is not None else time.time())
            role_min_interval_map = {
                "barber": self.role_snapshot_barber_interval_sec,
                "customer": self.role_snapshot_customer_interval_sec,
                "wash_customer": self.role_snapshot_wash_interval_sec,
                "unknown": self.role_snapshot_unknown_interval_sec,
            }
            role_min_interval = float(role_min_interval_map.get(role, self.role_snapshot_unknown_interval_sec))
            min_interval = max(self.role_snapshot_cooldown_sec, role_min_interval)
            last_ts = float(self._last_role_snapshot_ts.get(cooldown_key, 0.0))
            if (ts_now - last_ts) < min_interval:
                return

            self._reset_role_snapshot_daily_counters_if_needed()
            day_key = f"{datetime.now().strftime('%Y-%m-%d')}:{role}:{camera_name}"
            day_count = int(self._role_snapshot_daily_counts.get(day_key, 0))
            day_limit_map = {
                "barber": self.role_snapshot_barber_max_per_camera_day,
                "customer": self.role_snapshot_customer_max_per_camera_day,
                "wash_customer": self.role_snapshot_wash_max_per_camera_day,
                "unknown": self.role_snapshot_unknown_max_per_camera_day,
            }
            day_limit = int(day_limit_map.get(role, self.role_snapshot_unknown_max_per_camera_day))
            if day_count >= day_limit:
                return

            pad = float(SNAPSHOT_PAD)
            x1 = max(0, int(round(x1f - pad)))
            y1 = max(0, int(round(y1f - pad)))
            x2 = min(w - 1, int(round(x2f + pad)))
            y2 = min(h - 1, int(round(y2f + pad)))
            if x2 <= x1 or y2 <= y1:
                return

            crop = frame[y1:y2, x1:x2]
            if crop is None or crop.size == 0:
                return
            thumb = self._role_snapshot_thumb(crop)
            prev_thumb = self._last_role_snapshot_thumb.get(cooldown_key)
            if thumb is not None and prev_thumb is not None:
                try:
                    diff = float(np.mean(np.abs(thumb - prev_thumb)))
                    if diff < self.role_snapshot_thumb_min_diff:
                        return
                except Exception:
                    pass
            try:
                rx1 = max(0, int(round(x1f)) - x1)
                ry1 = max(0, int(round(y1f)) - y1)
                rx2 = min(crop.shape[1] - 1, int(round(x2f)) - x1)
                ry2 = min(crop.shape[0] - 1, int(round(y2f)) - y1)
                role_color = {
                    "barber": (0, 165, 255),
                    "customer": (0, 255, 255),
                    "wash_customer": (255, 200, 0),
                    "unknown": (160, 160, 160),
                }
                color = role_color.get(role, (160, 160, 160))
                cv2.rectangle(crop, (rx1, ry1), (rx2, ry2), color, 2)
            except Exception:
                pass

            pid_for_name = pid if pid > 0 else gid
            out = self._role_snapshot_path(role, camera_name, pid_for_name, gid=gid)
            out.parent.mkdir(parents=True, exist_ok=True)
            if self._cv2_imwrite_unicode_safe(out, crop):
                role_event_map = {
                    "barber": "barber_uniform",
                    "customer": "customer_by_admin",
                    "wash_customer": "customer_wash_by_admin",
                    "unknown": "unknown_by_admin",
                }
                event_type = role_event_map.get(role, "unknown_by_admin")
                self._remember_snapshot(str(out), camera_name, pid_for_name, event_type, gid=gid)
                self._last_role_snapshot_ts[cooldown_key] = ts_now
                if thumb is not None:
                    self._last_role_snapshot_thumb[cooldown_key] = thumb
                self._role_snapshot_daily_counts[day_key] = day_count + 1
        except Exception as e:
            logger.debug(f"Role snapshot capture failed: {e}")

    @staticmethod
    def _cv2_imwrite_unicode_safe(path: Path, img) -> bool:
        """Write image on Windows paths that may contain non-ASCII characters."""
        try:
            ext = path.suffix if path.suffix else ".jpg"
            ok, buf = cv2.imencode(ext, img)
            if not ok:
                return False
            buf.tofile(str(path))
            return True
        except Exception:
            return False

    def _remember_snapshot(self, snapshot_path: str, camera: str, pid: int, event_type: str, gid: int = 0):
        if not snapshot_path:
            return
        item = {
            "timestamp": datetime.now().isoformat(),
            "path": snapshot_path,
            "camera": camera,
            "gid": int(gid),
            "pid": int(pid),
            "event_type": event_type,
        }
        with self.verification_state_lock:
            existing = {str(s.get("path", "")) for s in self.recent_snapshots[-200:]}
            if snapshot_path not in existing:
                self.recent_snapshots.append(item)
                if len(self.recent_snapshots) > 200:
                    self.recent_snapshots = self.recent_snapshots[-200:]

    def _record_tier2_handoff(self, camera: str, payload: Dict):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "camera": camera,
        }
        entry.update(payload or {})
        with self.verification_state_lock:
            self.tier2_latest_by_camera[camera] = dict(entry)
            self.tier2_handoff_history.append(dict(entry))

    def _handle_verification_result(self, job: Dict, result: Dict):
        cam_name = str(job.get("cam", "") or "")
        pid = int(job.get("tid", 0) or 0)
        zone_name = str(job.get("zone", "") or "")
        head_count = int(result.get("head_count", 0) or 0)
        staff_count = int(result.get("staff_count", 0) or 0)
        best_frame = result.get("best_frame")
        if best_frame is None:
            return

        gid = 0
        with self.customers_lock:
            for det in self.latest_customers.get(cam_name, []) or []:
                if int(det.get("pid", 0) or 0) == pid:
                    gid = int(det.get("gid", 0) or 0)
                    break

        snapshot_path = self._snapshot_path("VERIFIED", cam_name, pid, gid=gid)
        snapshot_saved = self._cv2_imwrite_unicode_safe(snapshot_path, best_frame)
        if snapshot_saved:
            self._remember_snapshot(str(snapshot_path), cam_name, pid, EventType.VERIFIED.value, gid=gid)

        event = Event(
            timestamp=datetime.now(),
            camera=cam_name,
            event_type=EventType.VERIFIED,
            person_id=pid,
            zone_name=zone_name,
            dwell_seconds=0.0,
            metadata={
                "gid": gid,
                "verified_heads": head_count,
                "verified_staff": staff_count,
                "sampled_frame_index": int(result.get("sampled_frame_index", 0) or 0),
                "trigger_frame_count": int(job.get("trigger_frame_count", 0) or 0),
                "trigger_ts": float(job.get("trigger_ts", 0.0) or 0.0),
                "model_path": YOLO_VERIFY_MODEL_PATH,
                "snapshot_path": str(snapshot_path) if snapshot_saved else "",
            },
        )
        if hasattr(self.event_tracker, "record_event"):
            try:
                self.event_tracker.record_event(event)
            except Exception:
                pass
        with self.pending_lock:
            self.pending_events.append(event)
        with self.verification_state_lock:
            self.verified_events_total += 1
            self.verified_by_camera[cam_name] = int(self.verified_by_camera.get(cam_name, 0)) + 1
            self.last_verified_event = {
                "timestamp": event.timestamp.isoformat(),
                "camera": cam_name,
                "pid": pid,
                "gid": gid,
                "zone_name": zone_name,
                "verified_heads": head_count,
                "verified_staff": staff_count,
                "snapshot_path": str(snapshot_path) if snapshot_saved else "",
            }
        logger.info(
            f"Tier-2 Verified: {cam_name} | heads={head_count} staff={staff_count} "
            f"| snapshot={snapshot_path if snapshot_saved else 'save_failed'}"
        )
        self._record_tier2_handoff(
            cam_name,
            {
                "action": "verified",
                "eligible": True,
                "pid": pid,
                "gid": gid,
                "zone_name": zone_name,
                "verified_heads": head_count,
                "verified_staff": staff_count,
                "queue_depth": int(self.verification_queue.qsize()),
                "snapshot_path": str(snapshot_path) if snapshot_saved else "",
            },
        )

    def _handle_verification_miss(self, job: Dict):
        cam_name = str(job.get("cam", "") or "")
        pid = int(job.get("tid", 0) or 0)
        zone_name = str(job.get("zone", "") or "")
        self._record_tier2_handoff(
            cam_name,
            {
                "action": "no_match",
                "eligible": True,
                "pid": pid,
                "zone_name": zone_name,
                "queue_depth": int(self.verification_queue.qsize()),
                "reason": "worker_found_no_valid_heads_or_staff",
            },
        )
        logger.info(f"Tier-2 Done: {cam_name} pid={pid} zone={zone_name} | no valid verification match")

    def _maybe_start_tier2_capture(self, camera_name: str, now_ts: Optional[float] = None):
        now_ts = float(now_ts if now_ts is not None else time.time())
        if not self._is_within_business_hours():
            self._record_tier2_handoff(
                camera_name,
                {
                    "action": "blocked",
                    "eligible": False,
                    "reason": "outside_business_hours",
                    "queue_depth": int(self.verification_queue.qsize()),
                },
            )
            return
        cam_thread = self.cameras.get(camera_name)
        if cam_thread is None:
            return
        if cam_thread.recording_event:
            self._record_tier2_handoff(
                camera_name,
                {
                    "action": "blocked",
                    "eligible": False,
                    "reason": "clip_already_recording",
                    "queue_depth": int(self.verification_queue.qsize()),
                },
            )
            return

        with self.customers_lock:
            camera_customers = list(self.latest_customers.get(camera_name, []) or [])

        best_blocked: Optional[Dict] = None
        for det in camera_customers:
            pid = int(det.get("pid", 0) or 0)
            zone_name = det.get("primary_zone")
            if pid <= 0 or not zone_name:
                continue

            mem_info = self.active_person_memory.pid_last.get(pid, {})
            dwell = now_ts - float(mem_info.get("first_seen", now_ts))
            last_clip = float(mem_info.get("last_clip_ts", 0.0))
            last_clip_ago = (now_ts - last_clip) if last_clip > 0 else -1.0
            is_stale = bool(det.get("stale", False))
            is_staff = mem_info.get("staff_id") is not None
            duplicate_cooldown = last_clip > 0 and last_clip_ago < self.tier2_trigger_cooldown_sec
            eligible = (not is_stale) and (not is_staff) and (dwell >= self.tier2_trigger_dwell_sec) and (not duplicate_cooldown)
            reason = "ready"
            if is_stale:
                reason = "stale_detection"
            elif is_staff:
                reason = "staff_filtered"
            elif dwell < self.tier2_trigger_dwell_sec:
                reason = "below_dwell_threshold"
            elif duplicate_cooldown:
                reason = "duplicate_pid_cooldown"

            candidate = {
                "action": "queued" if eligible else "blocked",
                "eligible": bool(eligible),
                "reason": reason,
                "pid": pid,
                "gid": int(det.get("gid", 0) or 0),
                "zone_name": str(zone_name),
                "dwell_sec": round(float(dwell), 1),
                "dwell_min": round(float(dwell) / 60.0, 2),
                "last_clip_ago_sec": round(float(last_clip_ago), 1) if last_clip_ago >= 0 else -1.0,
                "duplicate_pid_cooldown": bool(duplicate_cooldown),
                "is_stale": bool(is_stale),
                "is_staff": bool(is_staff),
                "queue_depth": int(self.verification_queue.qsize()),
            }
            if (best_blocked is None) or (float(candidate.get("dwell_sec", 0.0)) > float(best_blocked.get("dwell_sec", 0.0))):
                best_blocked = dict(candidate)

            if not eligible:
                continue

            cam_thread.recording_event = {
                "tid": pid,
                "zone": str(zone_name),
                "trigger_ts": now_ts,
                "frames_to_go": cam_thread.get_clip_frame_budget(),
                "buffer": list(cam_thread.rolling_buffer),
            }
            mem_info["last_clip_ts"] = now_ts
            logger.info(
                f"TRIGGER: {camera_name} Person {pid} in {zone_name} for {dwell:.1f}s "
                f"(clip={cam_thread.clip_duration_sec:.1f}s)"
            )
            self._record_tier2_handoff(camera_name, candidate)
            break
        else:
            if best_blocked is not None:
                self._record_tier2_handoff(camera_name, best_blocked)

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
                "staff_filtered": self.staff_filtered_total,
                "staff_realtime": self.staff_realtime,
                "camera_people": self.camera_people_counts,
                "camera_staff": self.camera_staff_counts,
                "realtime_counts": self.realtime_counts,
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
                    "verified": self.verified_events_total,
                    "total_events": self._today_event_count(),
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

    def _write_dashboard_state(
        self,
        last_event: Optional[Dict] = None,
        status_override: Optional[Dict] = None,
        force: bool = False,
    ):
        """Persist state for cross-process dashboard polling with write throttling."""
        try:
            now_ts = time.time()
            if (
                (not force)
                and (last_event is None)
                and ((now_ts - float(self._last_dashboard_state_write_ts)) < self.dashboard_state_min_interval_sec)
            ):
                return

            status = status_override if isinstance(status_override, dict) else self.get_status()
            data = {
                "timestamp": now_ts,
                "status": status,
                "summary": status.get("summary", {}),
            }
            if last_event is not None:
                data["last_event"] = last_event

            serialized = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if (not force) and (last_event is None) and (serialized == self._last_dashboard_state_payload_hash):
                return

            state_path = Path(DASHBOARD_STATE_FILE)
            state_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                prefix="dashboard_state_",
                suffix=".tmp",
                dir=str(state_path.parent),
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(serialized)
                last_replace_err = None
                for _ in range(8):
                    try:
                        os.replace(tmp_path, str(state_path))
                        last_replace_err = None
                        break
                    except PermissionError as e:
                        # Windows may briefly lock the file while controller reads it.
                        last_replace_err = e
                        time.sleep(0.03)
                if last_replace_err is not None:
                    raise last_replace_err
            finally:
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass

            self._last_dashboard_state_write_ts = now_ts
            self._last_dashboard_state_payload_hash = serialized
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
            self.cameras[name] = CameraStream(
                name,
                url,
                verification_queue_ref=self.verification_queue,
                clip_duration_sec=self.tier2_clip_duration_sec,
            )
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
        self._write_dashboard_state(force=True)
        
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

        # Background feedback auto-train for chair-service classifier.
        self._start_chair_service_autotrain_thread()
        
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
        self._write_dashboard_state(force=True)
        
        # Stop reliability components
        self.resource_guard.stop()
        self.health_checker.stop()
        
        # Stop Supabase sync
        if self.supabase_sync:
            self.supabase_sync.stop()

        if self.worker:
            self.worker.stop()
        
        # Wait for threads
        for thread in self.threads:
            thread.join(timeout=5)
        if self.worker:
            self.worker.join(timeout=5)
        
        # Disconnect cameras
        for cam_stream in self.cameras.values():
            cam_stream.disconnect()

        self._write_dashboard_state(force=True)
        
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
            latest_detections = {}
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
                all_dets = self.latest_detections.get(cam, [])
                filtered_all = []
                for d in all_dets:
                    ts = float(d.get("ts", 0.0))
                    if ts <= 0 or (now - ts) > self.dashboard_presence_ttl_sec:
                        continue
                    filtered_all.append({
                        "pid": int(d.get("pid", 0)),
                        "gid": int(d.get("gid", 0)),
                        "bbox": [float(v) for v in d.get("bbox", [])],
                        "primary_zone": d.get("primary_zone"),
                        "staff_id": d.get("staff_id"),
                    })
                latest_detections[cam] = filtered_all

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
        with self.verification_state_lock:
            verified_total = int(self.verified_events_total)
            verified_by_camera = dict(self.verified_by_camera)
            last_verified_event = dict(self.last_verified_event) if isinstance(self.last_verified_event, dict) else self.last_verified_event
            tier2_latest_by_camera = {
                cam: dict(item) for cam, item in self.tier2_latest_by_camera.items()
            }
            tier2_recent = [dict(item) for item in list(self.tier2_handoff_history)[-12:]]
        
        # Prefer the computed global live_customers but fall back to per-camera
        # counts when the global counter is zero. This avoids brief resets to
        # 0 in the controller when small timing gaps occur between threads.
        fallback_active = sum(camera_people.values())
        active_tracks_val = self.live_customers if (self.live_customers and self.live_customers > 0) else int(fallback_active)
        res_metrics = self.resource_guard.get_latest_metrics(max_age_sec=max(1.0, self.resource_guard.check_interval * 2.0))
        with self._latency_lock:
            infer_stats = self._latency_stats(list(self._infer_latency_ms))
            process_stats = self._latency_stats(list(self._process_latency_ms))

        return {
            "running": self.running,
            "count_mode": self.service_mode,
            "branch": CONFIG.get("branch_code"),
            "cameras": cameras_status,
            "active_tracks": active_tracks_val,
            "events_queued": self.supabase_sync.get_queue_size() if self.supabase_sync else 0,
            "verification_queue_depth": self.verification_queue.qsize(),
            "camera_people": camera_people,
            "camera_staff": dict(self.camera_staff_counts),
            "camera_gids": camera_gids,
            "same_person_multi_cam": same_person_multi_cam,
            "motion": motion_state,
            "latest_customers": latest_customers,
            "latest_detections": latest_detections,
            "realtime_counts": self.realtime_counts,
            "resources": {
                "fps": float(getattr(res_metrics, "fps", 0.0)),
                "cpu_percent": float(getattr(res_metrics, "cpu_percent", 0.0)),
                "memory_percent": float(getattr(res_metrics, "memory_percent", 0.0)),
                "memory_mb": float(getattr(res_metrics, "memory_mb", 0.0)),
            },
            "perf": {
                "inference_latency_ms_avg": float(infer_stats.get("avg", 0.0)),
                "inference_latency_ms_p95": float(infer_stats.get("p95", 0.0)),
                "inference_latency_ms_max": float(infer_stats.get("max", 0.0)),
                "inference_latency_samples": int(infer_stats.get("samples", 0)),
                "frame_process_latency_ms_avg": float(process_stats.get("avg", 0.0)),
                "frame_process_latency_ms_p95": float(process_stats.get("p95", 0.0)),
                "frame_process_latency_ms_max": float(process_stats.get("max", 0.0)),
                "frame_process_latency_samples": int(process_stats.get("samples", 0)),
            },
            "verified_counts": {
                "total": verified_total,
                "by_camera": verified_by_camera,
                "last_event": last_verified_event,
            },
            "tier2_debug": {
                "latest_by_camera": tier2_latest_by_camera,
                "recent": tier2_recent,
            },
            "fsm_debug": {
                "enabled": bool(self.fsm_enabled),
                "state_counts": dict(self.fsm_state_counts),
                "open_sessions": int(len(getattr(self.service_fsm, "sessions", {}))),
                "recent_transitions": list(self.fsm_recent_transitions[-30:]),
            },
            "recent_snapshots": self.recent_snapshots[-50:],
            "tuning": {
                "yolo_conf": self.yolo_conf,
                "yolo_iou": self.yolo_iou,
                "yolo_imgsz": self.yolo_imgsz,
                "count_mode": self.service_mode,
                "yolo_mode": self.yolo_mode,
                "person_class_id": self.person_class_id,
                "head_class_id": self.head_class_id,
                "staff_uniform_class_id": self.staff_uniform_class_id,
                "discovery_person_class_id": self.discovery_person_class_id,
                "discovery_head_class_id": self.discovery_head_class_id,
                "discovery_staff_uniform_class_id": self.discovery_staff_uniform_class_id,
                "yolo_detect_class_ids": self.yolo_detect_class_ids,
                "inference_mode": self.inference_mode,
                "configured_inference_mode": self._configured_inference_mode,
                "auto_motion_gate_enabled": self.auto_motion_gate_enabled,
                "auto_motion_gate_active": self._auto_motion_gate_active,
                "auto_motion_gate_cpu_threshold": self.auto_motion_gate_cpu_threshold,
                "auto_motion_gate_mem_threshold": self.auto_motion_gate_mem_threshold,
                "auto_motion_gate_recover_sec": self.auto_motion_gate_recover_sec,
                "auto_motion_gate_min_hold_sec": self.auto_motion_gate_min_hold_sec,
                "enable_staff_uniform": self.enable_staff_uniform,
                "staff_uniform_iou_threshold": self.staff_uniform_iou_threshold,
                "enable_role_snapshots": self.enable_role_snapshots,
                "role_snapshot_cooldown_sec": self.role_snapshot_cooldown_sec,
                "role_snapshot_customer_interval_sec": self.role_snapshot_customer_interval_sec,
                "role_snapshot_barber_interval_sec": self.role_snapshot_barber_interval_sec,
                "role_snapshot_wash_interval_sec": self.role_snapshot_wash_interval_sec,
                "role_snapshot_unknown_interval_sec": self.role_snapshot_unknown_interval_sec,
                "role_snapshot_enable_unknown": self.role_snapshot_enable_unknown,
                "role_snapshot_thumb_min_diff": self.role_snapshot_thumb_min_diff,
                "role_snapshot_min_conf": self.role_snapshot_min_conf,
                "role_snapshot_min_area_ratio": self.role_snapshot_min_area_ratio,
                "role_snapshot_require_customer_chair_zone": self.role_snapshot_require_customer_chair_zone,
                "role_snapshot_customer_max_per_camera_day": self.role_snapshot_customer_max_per_camera_day,
                "role_snapshot_barber_max_per_camera_day": self.role_snapshot_barber_max_per_camera_day,
                "role_snapshot_wash_max_per_camera_day": self.role_snapshot_wash_max_per_camera_day,
                "role_snapshot_unknown_max_per_camera_day": self.role_snapshot_unknown_max_per_camera_day,
                "motion_diff_threshold": self.motion_diff_threshold,
                "motion_min_area_ratio": self.motion_min_area_ratio,
                "motion_recheck_sec": self.motion_recheck_sec,
                "motion_hold_sec": self.motion_hold_sec,
                "motion_downscale_width": self.motion_downscale_width,
                "sit_min_sec": self.sit_min_sec,
                "vacant_grace_sec": self.vacant_grace_sec,
                "haircut_counter_sit_min_sec": self.haircut_counter.sit_min_sec,
                "wash_counter_sit_min_sec": self.wash_counter.sit_min_sec,
                "service_vacant_grace_sec": self.service_vacant_grace_sec,
                "haircut_count_zones": {
                    str(cam): sorted(list(zones))
                    for cam, zones in (self.haircut_count_zones or {}).items()
                },
                "enable_chair_service_classifier": self.enable_chair_service_classifier,
                "chair_service_classifier_model": self.chair_service_classifier_model,
                "chair_service_classifier_positive_labels": sorted(list(self.chair_service_classifier_positive_labels)),
                "chair_service_classifier_min_conf": self.chair_service_classifier_min_conf,
                "chair_service_classifier_imgsz": self.chair_service_classifier_imgsz,
                "chair_service_classifier_loaded": bool(self.chair_service_classifier is not None),
                "chair_service_autotrain_enabled": self.chair_service_autotrain_enabled,
                "chair_service_autotrain_interval_hours": self.chair_service_autotrain_interval_hours,
                "chair_service_autotrain_min_positive": self.chair_service_autotrain_min_positive,
                "chair_service_autotrain_min_negative": self.chair_service_autotrain_min_negative,
                "chair_service_autotrain_positive_dir": self.chair_service_autotrain_positive_dir,
                "chair_service_autotrain_negative_dirs": list(self.chair_service_autotrain_negative_dirs),
                "chair_service_autotrain_last_status": self._chair_service_autotrain_last_status,
                "chair_service_autotrain_last_error": self._chair_service_autotrain_last_error,
                "chair_service_autotrain_last_train_ts": self._chair_service_autotrain_last_train_ts,
                "chair_service_autotrain_running": self._chair_service_autotrain_running,
                "chair_service_autotrain_data_counts": dict(self._chair_service_autotrain_last_data_counts or {}),
                "enable_chair_empty_guard": self.enable_chair_empty_guard,
                "chair_empty_gallery_dir": self.chair_empty_gallery_dir,
                "chair_empty_similarity_threshold": self.chair_empty_similarity_threshold,
                "chair_empty_hist_weight": self.chair_empty_hist_weight,
                "chair_empty_gallery_size": len(self.chair_empty_gallery_features or []),
                "event_identity_similarity_threshold": self.event_identity_similarity_threshold,
                "event_same_customer_block_sec": self.event_same_customer_block_sec,
                "event_same_seat_cooldown_sec": self.event_same_seat_cooldown_sec,
                "event_wash_return_block_sec": self.event_wash_return_block_sec,
                "require_chair_vacant_between_haircuts": self.require_chair_vacant_between_haircuts,
                "camera_stall_timeout_sec": self.camera_stall_timeout_sec,
                "camera_freeze_max_same_frames": self.camera_freeze_max_same_frames,
                "camera_freeze_diff_threshold": self.camera_freeze_diff_threshold,
                "customer_active_ttl_sec": self.customer_active_ttl_sec,
                "dashboard_presence_ttl_sec": self.dashboard_presence_ttl_sec,
                "no_detection_hold_sec": self.no_detection_hold_sec,
                "tier2_trigger_dwell_sec": self.tier2_trigger_dwell_sec,
                "tier2_trigger_cooldown_sec": self.tier2_trigger_cooldown_sec,
                "tier2_clip_duration_sec": self.tier2_clip_duration_sec,
                "enable_business_hours_guard": self.enable_business_hours_guard,
                "business_hours_start": self.business_hours_start,
                "business_hours_end": self.business_hours_end,
                "enable_event_reid_dedupe": self.enable_event_reid_dedupe,
                "haircut_event_dedupe_window_sec": self.haircut_event_dedupe_window_sec,
                "haircut_event_dedupe_similarity": self.haircut_event_dedupe_similarity,
                "event_snapshot_require_live_match": self.event_snapshot_require_live_match,
                "event_snapshot_fallback_to_event_bbox": self.event_snapshot_fallback_to_event_bbox,
                "event_snapshot_live_match_max_age_sec": self.event_snapshot_live_match_max_age_sec,
                "enable_event_feedback_autocopy": self.enable_event_feedback_autocopy,
                "enable_no_haircut_feedback_autocopy": self.enable_no_haircut_feedback_autocopy,
                "no_haircut_feedback_cooldown_sec": self.no_haircut_feedback_cooldown_sec,
                "event_require_fresh_detection": self.event_require_fresh_detection,
                "event_reject_stale_hold": self.event_reject_stale_hold,
                "event_max_detection_age_sec": self.event_max_detection_age_sec,
                "event_person_min_conf": self.event_person_min_conf,
                "event_head_min_conf": self.event_head_min_conf,
                "event_head_min_area_ratio": self.event_head_min_area_ratio,
                "event_head_max_area_ratio": self.event_head_max_area_ratio,
                "event_head_max_width_ratio": self.event_head_max_width_ratio,
                "event_head_max_height_ratio": self.event_head_max_height_ratio,
                "enable_head_live_filter": self.enable_head_live_filter,
                "hold_stale_for_head": self.hold_stale_for_head,
                "head_live_min_conf": self.head_live_min_conf,
                "head_live_min_area_ratio": self.head_live_min_area_ratio,
                "head_live_max_area_ratio": self.head_live_max_area_ratio,
                "head_live_max_width_ratio": self.head_live_max_width_ratio,
                "head_live_max_height_ratio": self.head_live_max_height_ratio,
                "enable_head_size_guard": self.enable_head_size_guard,
                "head_max_width_ratio": self.head_max_width_ratio,
                "head_max_height_ratio": self.head_max_height_ratio,
                "head_max_area_ratio": self.head_max_area_ratio,
                "resource_max_memory_percent": self.resource_guard.max_memory_percent,
                "throttle_log_interval_sec": self.throttle_log_interval_sec,
                "dashboard_ui_frame_interval_sec": self.dashboard_ui_frame_interval_sec,
                "dashboard_state_min_interval_sec": self.dashboard_state_min_interval_sec,
                "gc_collect_interval_sec": self.gc_collect_interval_sec,
                "summary_log_interval_sec": self.summary_log_interval_sec,
                "zone_point_mode": self.zone_point_mode,
                "enable_reid": self.enable_reid,
                "reid_similarity_threshold": self.reid_similarity_threshold,
                "staff_similarity_threshold": self.staff_similarity_threshold,
                "staff_event_similarity_threshold": self.staff_event_similarity_threshold,
                "enable_role_db_reid": self.enable_role_db_reid,
                "role_db_barber_threshold": self.role_db_barber_threshold,
                "role_db_customer_threshold": self.role_db_customer_threshold,
                "role_db_customer_margin": self.role_db_customer_margin,
                "enable_chair_customer_recovery": self.enable_chair_customer_recovery,
                "chair_customer_recovery_margin": self.chair_customer_recovery_margin,
                "chair_customer_recovery_min_area_ratio": self.chair_customer_recovery_min_area_ratio,
                "chair_staff_strict_barber_margin": self.chair_staff_strict_barber_margin,
                "role_db_customer_override_staff": self.role_db_customer_override_staff,
                "role_db_max_embeddings_per_role": self.role_db_max_embeddings_per_role,
                "role_db_total_embeddings": int(self.role_gallery.total_embeddings) if self.role_gallery else 0,
                "role_db_role_counts": dict(self.role_gallery.role_counts) if self.role_gallery else {},
                "role_vote_window": self.role_vote_window,
                "role_vote_min_samples": self.role_vote_min_samples,
                "role_vote_min_ratio": self.role_vote_min_ratio,
                "role_unknown_min_conf": self.role_unknown_min_conf,
                "require_stable_customer_role_for_service": self.require_stable_customer_role_for_service,
                "service_session_timeout_sec": self.service_session_timeout_sec,
                "reid_max_prototypes_per_pid": self.reid_max_prototypes_per_pid,
                "fsm_enabled": self.fsm_enabled,
                "fsm_return_dwell_sec": self.fsm_return_dwell_sec,
                "fsm_close_timeout_sec": self.fsm_close_timeout_sec,
                "fsm_return_window_sec": self.fsm_return_window_sec,
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
                "verified": verified_total,
                "haircut_confirmed": self.haircut_counter.total_count,
                "fsm_open_sessions": int(len(getattr(self.service_fsm, "sessions", {}))),
                "role_unknown_rate": (
                    float(self.role_vote_unknown_frames) / float(max(1, self.role_vote_total_frames))
                ),
                "total_events": self._today_event_count(),
            },
        }
    
    def _broadcast_status(self):
        """Broadcast status to dashboard"""
        try:
            status = self.get_status()
            self._write_dashboard_state(status_override=status)
            if self.broadcaster and self.broadcaster.get_subscriber_count() > 0:
                self.broadcaster.broadcast_status(status)
        except Exception as e:
            logger.debug(f"Error broadcasting status: {e}")
    
    def _broadcast_event(self, event: Dict):
        """Broadcast individual event to dashboard"""
        try:
            self._write_dashboard_state(last_event=event, force=True)
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
                    "resources": {},
                    "health_checks": self.health_checker.get_status(),
                }
                m = self.resource_guard.get_latest_metrics(max_age_sec=max(1.0, self.resource_guard.check_interval * 2.0))
                health["resources"] = {
                    "fps": float(getattr(m, "fps", 0.0)),
                    "memory_percent": float(getattr(m, "memory_percent", 0.0)),
                    "cpu_percent": float(getattr(m, "cpu_percent", 0.0)),
                    "memory_mb": float(getattr(m, "memory_mb", 0.0)),
                }
                
                if self.broadcaster:
                    # Broadcast as status update with health info
                    status = self.get_status()
                    status["health"] = health
                    self.broadcaster.broadcast_status(status)
                    self._write_dashboard_state(status_override=status)
                
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
