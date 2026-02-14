"""
Live Viewer Widget - Real-time camera feed with zone overlays and counting

Features:
- Select camera and connect to RTSP stream
- Display live video frame
- Overlay zone polygons (drawn from zones JSON)
- Show real-time counting metrics (people in zones)
- Display FPS and connection status
"""

import json
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QMessageBox
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QPixmap, QImage, QColor, QPainter, QPen
from PySide6.QtWidgets import QScrollArea


class FrameCapture(QThread):
    """Background thread to continuously capture frames from an RTSP stream."""
    status_changed = Signal(str)  # Emits status message

    def __init__(self, rtsp_url: str):
        super().__init__()
        self.rtsp_url = rtsp_url
        self.running = False
        self.cap = None
        self.latest_frame = None
        self._lock = threading.Lock()

    def run(self):
        """Continuously grab frames from the RTSP stream."""
        self.running = True
        try:
            self.status_changed.emit(f"Connecting to {self.rtsp_url[:40]}...")
            self.cap = cv2.VideoCapture(self.rtsp_url)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not self.cap.isOpened():
                self.status_changed.emit("Failed to open RTSP stream")
                return

            self.status_changed.emit("Connected ✓")

            while self.running:
                ret, frame = self.cap.read()
                if ret:
                    with self._lock:
                        self.latest_frame = frame
                else:
                    self.status_changed.emit("Frame read failed. Reconnecting...")
                    self.cap.release()
                    time.sleep(1)
                    self.cap = cv2.VideoCapture(self.rtsp_url)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if not self.cap.isOpened():
                        self.status_changed.emit("Reconnect failed.")
                        break

        except Exception as e:
            self.status_changed.emit(f"Error: {str(e)[:50]}")

        finally:
            if self.cap:
                self.cap.release()
            self.running = False

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest captured frame."""
        with self._lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def stop(self):
        """Stop frame capture"""
        self.running = False


class LiveViewerWidget(QWidget):
    """Live viewer widget with zone overlay and counting"""
    
    def __init__(self, parent=None, config=None):
        super().__init__(parent)
        self.config = config or {}
        self.zones_data: Dict[str, List] = {}
        self.current_camera = None
        self.frame_capture: Optional[FrameCapture] = None
        self.fps_counter = 0
        self.fps_timestamp = time.time()
        self.dashboard_state_file = Path(self.config.get("paths", {}).get("dashboard_state", "runtime/dashboard_state.json"))
        self.runtime_overlay: Dict[str, List[Dict]] = {}
        # detection cache keeps last-known detections to persist boxes while person remains in frame
        self.detection_cache: Dict[str, Dict] = {}
        # TTL for keeping detections visible (seconds)
        self.detection_cache_ttl = float(self.config.get('runtime', {}).get('detection_cache_ttl_sec', 5.0))
        self.last_snapshot_info = ""
        
        self.init_ui()
        
        # Timer to update display
        self.display_timer = QTimer()
        self.display_timer.timeout.connect(self.update_display)
        self.display_timer.start(33)  # Aim for ~30 FPS display rate
    
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()
        
        # Controls
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Camera:"))
        
        self.camera_combo = QComboBox()
        self.reload_cameras()
        controls_layout.addWidget(self.camera_combo)
        
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.connect_camera)
        controls_layout.addWidget(self.connect_btn)

        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.clicked.connect(self.disconnect_camera)
        self.disconnect_btn.setEnabled(False)
        controls_layout.addWidget(self.disconnect_btn)

        self.reload_zones_btn = QPushButton("Reload Zones")
        self.reload_zones_btn.clicked.connect(self.reload_current_zones)
        controls_layout.addWidget(self.reload_zones_btn)
        
        layout.addLayout(controls_layout)
        
        # Video display
        self.video_label = QLabel()
        # Reduce required minimum so the whole GUI can be smaller; label will
        # scale the video display down while preserving aspect ratio.
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setStyleSheet("background-color: black; border: 1px solid gray;")
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)
        
        # Status bar
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: orange;")
        status_layout.addWidget(self.status_label)
        
        self.fps_label = QLabel("FPS: 0")
        status_layout.addWidget(self.fps_label)
        
        self.people_count_label = QLabel("People: 0")
        status_layout.addWidget(self.people_count_label)
        
        status_layout.addStretch()
        layout.addLayout(status_layout)
        
        self.setLayout(layout)
    
    def reload_cameras(self):
        """Reload camera list"""
        self.camera_combo.clear()
        cameras = self.config.get("cameras", {})
        for name in cameras.keys():
            self.camera_combo.addItem(name)
    
    def load_zones_for_camera(self, camera_name: str):
        """Load zone data from zones JSON file"""
        zones_dir = Path(self.config.get("paths", {}).get("zones", "data/zones"))
        zones_file = zones_dir / f"zones_{camera_name}.json"
        
        self.zones_data = {}
        
        if zones_file.exists():
            try:
                with open(zones_file, 'r', encoding='utf-8') as f:
                    zones_list = json.load(f)
                    for zone in zones_list:
                        zone_name = zone.get("name", "")
                        points = zone.get("points", [])
                        if not points and "polygon_json" in zone:
                            points = zone.get("polygon_json", [])
                        self.zones_data[zone_name] = {
                            "points": self._parse_points(points),
                            "type": zone.get("type", "OTHER"),
                            "count": 0  # Real-time count would go here
                        }
            except Exception as e:
                self.status_label.setText(f"Error loading zones: {e}")
                self.status_label.setStyleSheet("color: red;")

    def _parse_points(self, points: List) -> List[List[float]]:
        """Parse points from [[x,y], ...] or [{x,y}, ...] to numeric list."""
        out: List[List[float]] = []
        for p in points or []:
            if isinstance(p, dict):
                out.append([float(p.get("x", 0.0)), float(p.get("y", 0.0))])
            else:
                out.append([float(p[0]), float(p[1])])
        return out

    def _to_display_points(self, points: List[List[float]], frame_w: int, frame_h: int) -> np.ndarray:
        """
        Convert zone points to display pixels.
        Supports normalized points (0..1) and absolute pixel points.
        """
        if not points:
            return np.array([], dtype=np.int32)

        is_normalized = all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in points)
        if is_normalized:
            pts = np.array([[int(x * frame_w), int(y * frame_h)] for x, y in points], dtype=np.int32)
            return pts

        max_x = max(x for x, _ in points) if points else 1.0
        max_y = max(y for _, y in points) if points else 1.0
        sx = frame_w / max(max_x, 1.0)
        sy = frame_h / max(max_y, 1.0)
        pts = np.array([[int(x * sx), int(y * sy)] for x, y in points], dtype=np.int32)
        return pts

    def _color_for_gid(self, gid: int):
        """Return a consistent BGR color tuple for a given global id (gid).

        If gid is falsy (0), return a default green.
        """
        if not gid:
            return (0, 220, 0)
        # deterministic color from gid
        h = (int(gid) * 2654435761) & 0xFFFFFFFF
        r = (h >> 16) & 0xFF
        g = (h >> 8) & 0xFF
        b = h & 0xFF
        # make colors brighter
        return (int(b), int(g), int(r))
    
    def connect_camera(self):
        """Connect to selected camera"""
        camera = self.camera_combo.currentText()
        if not camera:
            QMessageBox.warning(self, "Error", "No camera selected")
            return
        
        cam_config = self.config.get("cameras", {}).get(camera)
        if not cam_config:
            QMessageBox.warning(self, "Error", "Camera not found")
            return
        
        url = cam_config.get("rtsp_url")
        if not url:
            QMessageBox.warning(self, "Error", "RTSP URL not configured")
            return
        
        self.current_camera = camera
        self.load_zones_for_camera(camera)
        
        # Start frame capture
        self.frame_capture = FrameCapture(url)
        self.frame_capture.status_changed.connect(self.on_status_changed)
        self.frame_capture.start()
        
        self.connect_btn.setEnabled(False)
        self.disconnect_btn.setEnabled(True)
        self.camera_combo.setEnabled(False)
        
        self.status_label.setText(f"Connecting to {camera}...")
        self.status_label.setStyleSheet("color: blue;")
    
    def disconnect_camera(self):
        """Disconnect from camera"""
        if self.frame_capture:
            self.frame_capture.stop()
            self.frame_capture.wait()
            self.frame_capture = None
        
        self.current_camera = None
        
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.camera_combo.setEnabled(True)
        
        self.status_label.setText("Disconnected")
        self.status_label.setStyleSheet("color: orange;")
        self.video_label.clear()

    def reload_current_zones(self):
        """Reload zones for the selected/current camera without reconnecting stream."""
        camera = self.current_camera or self.camera_combo.currentText()
        if not camera:
            QMessageBox.information(self, "Info", "No camera selected")
            return
        self.load_zones_for_camera(camera)
        self.status_label.setText(f"Zones reloaded: {camera} ({len(self.zones_data)})")
        self.status_label.setStyleSheet("color: green;")
    
    def on_status_changed(self, status: str):
        """Called when stream status changes"""
        self.status_label.setText(status)
        if "Connected" in status:
            self.status_label.setStyleSheet("color: green;")
        elif "Error" in status or "Failed" in status:
            self.status_label.setStyleSheet("color: red;")
        else:
            self.status_label.setStyleSheet("color: blue;")

    def _load_runtime_overlay(self):
        """Load latest detection IDs from runtime dashboard_state file."""
        if not self.dashboard_state_file.exists():
            self.last_snapshot_info = ""
            return
        try:
            with open(self.dashboard_state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            status = state.get("status", {})
            latest = status.get("latest_customers", {}) or {}
            snaps = status.get("recent_snapshots", []) or []
            if snaps:
                s = snaps[-1]
                self.last_snapshot_info = f"Last snap: {s.get('camera','?')} GID {s.get('gid',0)}"
            else:
                self.last_snapshot_info = ""

            # Update detection cache with latest detections per camera
            now = time.time()
            for cam_name, people in latest.items():
                for det in people or []:
                    bbox = det.get('bbox', [])
                    if len(bbox) != 4:
                        continue
                    pid = det.get('pid', 0)
                    gid = det.get('gid', 0)
                    # prefer global id key if available to link across cameras
                    key = f"G{gid}" if gid else f"{cam_name}_P{pid}"
                    # if this gid already exists from another camera, update last_seen and bbox
                    existing = self.detection_cache.get(key)
                    self.detection_cache[key] = {
                        'bbox': [int(v) for v in bbox],
                        'pid': pid,
                        'gid': gid,
                        'camera': cam_name,
                        'last_seen': now,
                        'primary_zone': det.get('primary_zone', '')
                    }
                    # If gid present, also refresh/merge other keys with same gid to keep global tracking
                    if gid:
                        other_key = f"{cam_name}_P{pid}"
                        if other_key in self.detection_cache and other_key != key:
                            self.detection_cache[other_key]['last_seen'] = now

            # Prune old cache entries
            to_del = []
            for k, v in self.detection_cache.items():
                if now - v.get('last_seen', 0) > self.detection_cache_ttl * 3:
                    to_del.append(k)
            for k in to_del:
                del self.detection_cache[k]

            # Keep runtime_overlay for compatibility, but drawing uses detection_cache
            self.runtime_overlay = latest
        except Exception:
            self.last_snapshot_info = ""
    
    def update_display(self):
        """Update video display with zones overlay"""
        if self.frame_capture is None:
            return

        frame = self.frame_capture.get_frame()
        if frame is None:
            return
        self._load_runtime_overlay()

        # Work on original size for accurate overlay, resize after drawing.
        draw_frame = frame.copy()
        h, w = draw_frame.shape[:2]
        
        # Draw zones
        total_people = 0
        for zone_name, zone_data in self.zones_data.items():
            points = zone_data.get("points", [])
            zone_type = zone_data.get("type", "OTHER")
            
            if not points:
                continue
            
            # Convert points to numpy array
            pts = self._to_display_points(points, w, h)
            if len(pts) < 3:
                continue
            
            # Color based on type
            color_map = {
                "CHAIR": (0, 255, 0),      # Green
                "WAIT": (255, 165, 0),     # Orange
                "WASH": (0, 165, 255),     # Cyan
                "STAFF_AREA": (128, 0, 128),  # Purple
                "OTHER": (200, 200, 200)   # Gray
            }
            color = color_map.get(zone_type, (200, 200, 200))
            
            # Draw polygon
            cv2.polylines(draw_frame, [pts], True, color, 2)
            
            # Draw zone name
            if len(pts) > 0:
                centroid = pts.mean(axis=0).astype(int)
                cv2.putText(draw_frame, f"{zone_name}", tuple(centroid), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw cached detections (persist while person remains in frame)
        now = time.time()
        draw_items = []
        for key, info in list(self.detection_cache.items()):
            # Only draw detections for this camera and recently seen
            if info.get('camera') != (self.current_camera or ""):
                continue
            if now - info.get('last_seen', 0) > self.detection_cache_ttl:
                continue
            bbox = info.get('bbox', [])
            if len(bbox) != 4:
                continue
            draw_items.append(info)

        self.people_count_label.setText(f"People: {len(draw_items)}")
        for det in draw_items:
            x1, y1, x2, y2 = [int(v) for v in det.get('bbox', [])]
            pid = det.get('pid', 0)
            gid = det.get('gid', 0)
            pzone = det.get('primary_zone', '')
            color = self._color_for_gid(gid)
            cv2.rectangle(draw_frame, (x1, y1), (x2, y2), color, 2)
            label = f"PID:{pid} GID:{gid} {pzone}" if gid else f"PID:{pid} {pzone}"
            # put a filled background for readability
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(draw_frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(draw_frame, label, (x1 + 3, max(12, y1 - 4)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        
        # Add FPS counter
        self.fps_counter += 1
        now = time.time()
        if now - self.fps_timestamp >= 1.0:
            fps = self.fps_counter
            self.fps_label.setText(f"FPS: {fps}")
            self.fps_counter = 0
            self.fps_timestamp = now
        
        cv2.putText(draw_frame, f"FPS: {self.fps_counter}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(draw_frame, f"Camera: {self.current_camera or 'None'}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if self.last_snapshot_info:
            cv2.putText(draw_frame, self.last_snapshot_info, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 0), 2)

        # Resize for display widget
        frame = cv2.resize(draw_frame, (960, 540))
        
        # Convert BGR to RGB and display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap)
    
    def closeEvent(self, event):
        """Clean up on close"""
        self.disconnect_camera()
        self.display_timer.stop()
        event.accept()
