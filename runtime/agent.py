"""
Runtime Service: Multi-Camera People Counter with YOLO + Zone Tracking

Business Logic:
  - WAIT = Waiting area (real-time count)
  - CHAIR_1/2/3 = Haircut chairs (precise counting)
  - WASH = Washing area (separate count)

Refactored to use centralized config.yaml
"""

import os
import sys
import time
import json
import csv
import threading
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Set

import numpy as np
import cv2
import torch
from ultralytics import YOLO

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent))

from shared.config import Config
from shared.logger import setup_logger

# =========================
# SETUP
# =========================
CONFIG = Config("data/config/config.yaml")
logger = setup_logger("edge_agent", CONFIG.get("paths", {}).get("logs", "logs"))

logger.info(f"Project: {CONFIG.get('project_name')}")
logger.info(f"Branch: {CONFIG.get('branch_code')}")

# =========================
# CAMERA CONFIG
# =========================
CAMERAS_CONFIG = CONFIG.get("cameras", {})
YOLO_CONFIG = CONFIG.get("yolo", {})
RUNTIME_CONFIG = CONFIG.get("runtime", {})
PATHS_CONFIG = CONFIG.get("paths", {})
DWELL_TIME_CONFIG = CONFIG.get("dwell_time", {})

YOLO_MODEL_PATH = os.path.join(PATHS_CONFIG.get("models", "models"), YOLO_CONFIG.get("model", "yolov8m.pt"))
YOLO_CONF = YOLO_CONFIG.get("conf", 0.35)
YOLO_IOU = YOLO_CONFIG.get("iou", 0.5)
IMGSZ = YOLO_CONFIG.get("imgsz", 640)
TARGET_FPS = RUNTIME_CONFIG.get("target_fps", 10)
SHOW_WINDOW = os.getenv("SHOW_WINDOW", "false").lower() == "true"

SNAPSHOT_DIR = PATHS_CONFIG.get("snapshots", "snapshots")
REPORTS_DIR = PATHS_CONFIG.get("reports", "reports")
ZONES_DIR = PATHS_CONFIG.get("zones", "data/zones")

# Legacy env vars for backward compatibility
USE_VIDEO = os.getenv("USE_VIDEO", "0") == "1"
ZONE_POINT_MODE = os.getenv("ZONE_POINT_MODE", "foot")
VID_MERGE_MAX_SEC = float(os.getenv("VID_MERGE_MAX_SEC", "3.0"))
VID_MERGE_MAX_DIST = float(os.getenv("VID_MERGE_MAX_DIST", "0.10"))
STAFF_DWELL_SEC = float(os.getenv("STAFF_DWELL_SEC", "12"))
SHOP_EXIT_GRACE_SEC = float(os.getenv("SHOP_EXIT_GRACE_SEC", "10"))
GLOBAL_MATCH_MAX_SEC = float(os.getenv("GLOBAL_MATCH_MAX_SEC", "2.0"))
GLOBAL_MATCH_MAX_DIST = float(os.getenv("GLOBAL_MATCH_MAX_DIST", "0.12"))
SIT_MIN_SEC = float(os.getenv("SIT_MIN_SEC", "10"))
VACANT_GRACE_SEC = float(os.getenv("VACANT_GRACE_SEC", "6"))
RECOUNT_COOLDOWN_SEC = float(os.getenv("RECOUNT_COOLDOWN_SEC", "300"))

# Zone naming
STAFF_NAMES = {"STAFF", "STAFF_AREA", "STAFF_ZONE"}
SHOP_NAMES = {"SHOP"}
WAIT_ZONE_NAMES = {"WAIT", "WAITING", "WAIT_AREA"}
WASH_ZONE_PREFIX = "WASH"
CHAIR_ZONE_PREFIX = "CHAIR_"

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
    zone_entries: Dict[str, float] = field(default_factory=dict)
    counted_in: Set[str] = field(default_factory=set)
    is_staff: bool = False
    
    def update(self, detection: Detection):
        """Update track with new detection"""
        self.detections.append(detection)
        self.last_seen = time.time()
    
    def dwell_time_in_zone(self, zone_name: str) -> float:
        """Calculate dwell time in zone (seconds)"""
        if zone_name not in self.zone_entries:
            return 0
        return time.time() - self.zone_entries[zone_name]


@dataclass
class EventLog:
    """Event for reporting"""
    timestamp: datetime
    camera: str
    event_type: str  # haircut, wash, wait, entrance, exit
    person_id: int
    zone: str
    metadata: Dict = field(default_factory=dict)
    
    def to_csv_row(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "camera": self.camera,
            "event_type": self.event_type,
            "person_id": self.person_id,
            "zone": self.zone,
            "metadata": json.dumps(self.metadata),
        }


# =========================
# ZONE LOADING
# =========================

def load_zones(camera_name: str) -> Dict[str, List[Tuple[float, float]]]:
    """Load zone polygon from JSON for camera"""
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
            
            if not zone_name or not polygon:
                continue
            
            # Convert normalized points to pixel coordinates
            pts_px = [(p["x"], p["y"]) for p in polygon]
            zones[zone_name] = pts_px
        
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


# =========================
# YOLO SETUP
# =========================

def load_yolo_model():
    """Load YOLO model"""
    if not os.path.exists(YOLO_MODEL_PATH):
        logger.error(f"YOLO model not found: {YOLO_MODEL_PATH}")
        raise FileNotFoundError(f"YOLO model: {YOLO_MODEL_PATH}")
    
    device = YOLO_CONFIG.get("device", "auto")
    logger.info(f"Loading YOLO model: {YOLO_MODEL_PATH} on device: {device}")
    
    model = YOLO(YOLO_MODEL_PATH)
    model.to(device)
    return model


# =========================
# CAMERA STREAM
# =========================

class CameraStream:
    """Capture video stream from camera"""
    
    def __init__(self, camera_name: str, rtsp_url: str):
        self.camera_name = camera_name
        self.rtsp_url = rtsp_url
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_connected = False
        self.frame: Optional[np.ndarray] = None
        self.fps = 0
        self.lock = threading.Lock()
        
        self.connect()
    
    def connect(self):
        """Connect to camera stream"""
        if not self.rtsp_url:
            logger.warning(f"No RTSP URL for {self.camera_name}")
            return
        
        try:
            self.cap = cv2.VideoCapture(self.rtsp_url)
            
            if not self.cap.isOpened():
                logger.error(f"Cannot open camera: {self.camera_name}")
                return
            
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            self.is_connected = True
            logger.info(f"Connected to {self.camera_name} at {self.fps:.1f} FPS")
        except Exception as e:
            logger.error(f"Error connecting to {self.camera_name}: {e}")
    
    def read_frame(self) -> Optional[np.ndarray]:
        """Read next frame"""
        if not self.is_connected or self.cap is None:
            return None
        
        try:
            ret, frame = self.cap.read()
            
            if not ret:
                logger.warning(f"Cannot read frame from {self.camera_name}")
                self.is_connected = False
                self.reconnect()
                return None
            
            with self.lock:
                self.frame = frame
            
            return frame
        except Exception as e:
            logger.error(f"Error reading frame from {self.camera_name}: {e}")
            self.is_connected = False
            return None
    
    def reconnect(self):
        """Attempt reconnection"""
        logger.info(f"Reconnecting to {self.camera_name}...")
        self.disconnect()
        time.sleep(2)
        self.connect()
    
    def disconnect(self):
        """Disconnect from camera"""
        if self.cap is not None:
            self.cap.release()
        self.is_connected = False
        logger.info(f"Disconnected from {self.camera_name}")
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get latest frame without blocking"""
        with self.lock:
            return self.frame


# =========================
# TRACKER
# =========================

class MultiCameraTracker:
    """Track people across cameras"""
    
    def __init__(self):
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        self.lock = threading.Lock()
        self.events: List[EventLog] = []
    
    def update(self, camera: str, detections: List[Detection]):
        """Update tracker with new detections"""
        with self.lock:
            # Simple association: closest detection to existing track
            for detection in detections:
                # Find closest track
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
                    
                    if dist < min_dist and dist < 0.1:  # Threshold
                        min_dist = dist
                        closest_track = track
                
                if closest_track:
                    closest_track.update(detection)
                else:
                    # Create new track
                    new_track = Track(
                        track_id=self.next_track_id,
                        camera=camera,
                        detections=[detection]
                    )
                    self.tracks[self.next_track_id] = new_track
                    self.next_track_id += 1
    
    def get_active_tracks(self, camera: str, max_age: float = 5.0) -> List[Track]:
        """Get active tracks for camera"""
        current_time = time.time()
        active = []
        
        with self.lock:
            for track in self.tracks.values():
                if track.camera == camera:
                    if current_time - track.last_seen < max_age:
                        active.append(track)
        
        return active


# =========================
# MAIN RUNTIME
# =========================

class RuntimeService:
    """Main runtime service"""
    
    def __init__(self):
        self.yolo_model = load_yolo_model()
        self.cameras: Dict[str, CameraStream] = {}
        self.tracker = MultiCameraTracker()
        self.zones: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
        
        self.running = False
        self.threads: List[threading.Thread] = []
        
        self._setup_cameras()
        self._load_zones()
        self._setup_dirs()
    
    def _setup_cameras(self):
        """Initialize camera streams"""
        for cam_name, cam_config in CAMERAS_CONFIG.items():
            if cam_config.get("enabled", True):
                rtsp_url = cam_config.get("rtsp_url", "")
                cam_stream = CameraStream(cam_name, rtsp_url)
                self.cameras[cam_name] = cam_stream
    
    def _load_zones(self):
        """Load zones for all cameras"""
        for cam_name in self.cameras:
            zones = load_zones(cam_name)
            self.zones[cam_name] = zones
    
    def _setup_dirs(self):
        """Create necessary directories"""
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        os.makedirs(REPORTS_DIR, exist_ok=True)
        os.makedirs(CONFIG.get("paths", {}).get("logs", "logs"), exist_ok=True)
    
    def process_frame(self, camera_name: str, frame: np.ndarray):
        """Process single frame with YOLO"""
        if frame is None or frame.size == 0:
            return
        
        try:
            h, w = frame.shape[:2]
            results = self.yolo_model.predict(
                frame,
                conf=YOLO_CONF,
                iou=YOLO_IOU,
                imgsz=IMGSZ,
                verbose=False
            )
            
            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        
                        # Normalize coordinates
                        x1_norm, y1_norm = x1 / w, y1 / h
                        x2_norm, y2_norm = x2 / w, y2 / h
                        
                        detection = Detection(x1_norm, y1_norm, x2_norm, y2_norm, conf, cls)
                        detections.append(detection)
            
            self.tracker.update(camera_name, detections)
        except Exception as e:
            logger.error(f"Error processing frame from {camera_name}: {e}")
    
    def run_camera_thread(self, camera_name: str):
        """Thread for single camera"""
        logger.info(f"Starting camera thread: {camera_name}")
        cam_stream = self.cameras[camera_name]
        frame_count = 0
        
        while self.running:
            frame = cam_stream.read_frame()
            
            if frame is None:
                time.sleep(0.1)
                continue
            
            self.process_frame(camera_name, frame)
            frame_count += 1
            
            if frame_count % 30 == 0:
                logger.debug(f"{camera_name}: {frame_count} frames processed")
            
            # FPS control
            time.sleep(1.0 / max(TARGET_FPS, 1))
    
    def start(self):
        """Start runtime service"""
        logger.info("Starting runtime service...")
        self.running = True
        
        for camera_name in self.cameras:
            thread = threading.Thread(target=self.run_camera_thread, args=(camera_name,))
            thread.daemon = True
            thread.start()
            self.threads.append(thread)
        
        logger.info(f"Started {len(self.threads)} camera threads")
    
    def stop(self):
        """Stop runtime service"""
        logger.info("Stopping runtime service...")
        self.running = False
        
        for thread in self.threads:
            thread.join(timeout=5)
        
        for cam_stream in self.cameras.values():
            cam_stream.disconnect()
        
        logger.info("Runtime service stopped")
    
    def get_status(self) -> Dict:
        """Get service status"""
        cameras_status = {}
        for cam_name, cam_stream in self.cameras.items():
            cameras_status[cam_name] = {
                "connected": cam_stream.is_connected,
                "fps": cam_stream.fps
            }
        
        return {
            "running": self.running,
            "branch": CONFIG.get("branch_code"),
            "cameras": cameras_status,
            "active_tracks": len(self.tracker.tracks)
        }


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    service = RuntimeService()
    
    try:
        service.start()
        
        # Status loop
        while service.running:
            status = service.get_status()
            logger.info(f"Status: {status}")
            time.sleep(30)
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    finally:
        service.stop()
