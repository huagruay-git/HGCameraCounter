"""
Live Viewer Widget - Real-time camera feed with zone overlays and counting

Supports:
- Single camera live view
- Multi-camera grid view (select multiple cameras)
- Zone overlays from zones JSON
- Runtime detection overlay from dashboard_state.json
"""

import json
import shutil
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal, QUrl
from PySide6.QtGui import QDesktopServices, QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class FrameCapture(QThread):
    """Background thread to continuously capture frames from an RTSP stream."""

    status_changed = Signal(str)

    def __init__(self, rtsp_url: str):
        super().__init__()
        self.rtsp_url = rtsp_url
        self.running = False
        self.cap = None
        self.latest_frame = None
        self._lock = threading.Lock()
        self.backend_name = "unknown"

    @staticmethod
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

    def _connect_attempts(self) -> List[Tuple[str, str, Optional[int]]]:
        ffmpeg_common = {
            "stimeout": "3000000",
            "rw_timeout": "3000000",
        }
        tcp_url = self._append_rtsp_params(self.rtsp_url, {"rtsp_transport": "tcp", **ffmpeg_common})
        ffmpeg_url = self._append_rtsp_params(self.rtsp_url, ffmpeg_common)
        return [
            ("ffmpeg_tcp", tcp_url, cv2.CAP_FFMPEG),
            ("default", self.rtsp_url, None),
            ("ffmpeg", ffmpeg_url, cv2.CAP_FFMPEG),
        ]

    def _open_capture(self) -> bool:
        for backend_name, url, backend in self._connect_attempts():
            cap = cv2.VideoCapture(url) if backend is None else cv2.VideoCapture(url, backend)
            if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
            if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)
            if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened():
                self.cap = cap
                self.backend_name = backend_name
                self.status_changed.emit(f"Connected ({backend_name})")
                return True
            cap.release()
        return False

    def run(self):
        self.running = True
        try:
            self.status_changed.emit(f"Connecting to {self.rtsp_url[:40]}...")
            if not self._open_capture():
                self.status_changed.emit("Failed to open RTSP stream")
                return

            while self.running:
                ret, frame = self.cap.read()
                if ret:
                    with self._lock:
                        self.latest_frame = frame
                else:
                    self.status_changed.emit("Frame read failed. Reconnecting...")
                    if self.cap:
                        self.cap.release()
                    time.sleep(1)
                    if not self._open_capture():
                        self.status_changed.emit("Reconnect failed")
                        break
        except Exception as e:
            self.status_changed.emit(f"Error: {str(e)[:50]}")
        finally:
            if self.cap:
                self.cap.release()
            self.running = False

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def stop(self):
        self.running = False


class LiveViewerWidget(QWidget):
    """Live viewer widget with zone overlay and counting."""

    def __init__(self, parent=None, config=None):
        super().__init__(parent)
        self.config = config or {}

        self.current_camera: Optional[str] = None
        self.frame_capture: Optional[FrameCapture] = None

        self.multi_frame_captures: Dict[str, FrameCapture] = {}
        self.multi_camera_order: List[str] = []
        self.multi_tile_labels: Dict[str, QLabel] = {}

        self.zones_data: Dict[str, Dict] = {}
        self.zones_by_camera: Dict[str, Dict[str, Dict]] = {}

        self.fps_counter = 0
        self.fps_timestamp = time.time()

        self.dashboard_state_file = Path(self.config.get("paths", {}).get("dashboard_state", "runtime/dashboard_state.json"))
        self.runtime_overlay: Dict[str, List[Dict]] = {}
        self.detection_cache: Dict[str, Dict] = {}
        self.detection_cache_ttl = float(self.config.get("runtime", {}).get("detection_cache_ttl_sec", 5.0))
        self.last_snapshot_info = ""
        self.current_count_mode = "live"
        self.recent_snapshots: List[Dict] = []
        self._snapshot_signature = ""
        paths_cfg = self.config.get("paths", {}) or {}
        self.barber_label_dir = Path(paths_cfg.get("staff_gallery", "data/staff_gallery")) / "BARBER_UNIFORM"
        self.customer_label_dir = Path(paths_cfg.get("customer_by_admin", "data/customer_by_admin"))
        self.wash_customer_label_dir = Path(
            paths_cfg.get("customer_wash_by_admin", "data/customer_wash_by_admin")
        )
        try:
            self.barber_label_dir.mkdir(parents=True, exist_ok=True)
            self.customer_label_dir.mkdir(parents=True, exist_ok=True)
            self.wash_customer_label_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        self._init_ui()

        self.display_timer = QTimer()
        self.display_timer.timeout.connect(self.update_display)
        self.display_timer.start(33)

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(10)

        # --- Top toolbar: mode + camera + connection actions ---
        top = QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(QLabel("Mode:"))
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems(["Single", "Multi"])
        self.view_mode_combo.currentTextChanged.connect(self._on_view_mode_changed)
        top.addWidget(self.view_mode_combo)
        top.addSpacing(10)
        top.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(180)
        top.addWidget(self.camera_combo)
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.connect_camera)
        top.addWidget(self.connect_btn)
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.clicked.connect(self.disconnect_camera)
        self.disconnect_btn.setEnabled(False)
        top.addWidget(self.disconnect_btn)
        self.reload_zones_btn = QPushButton("Reload Zones")
        self.reload_zones_btn.clicked.connect(self.reload_current_zones)
        top.addWidget(self.reload_zones_btn)
        top.addStretch()
        root.addLayout(top)

        # --- Main area: large video (left) + control panel (right) ---
        main = QHBoxLayout()
        main.setSpacing(10)

        video_col = QVBoxLayout()
        video_col.setSpacing(8)
        self.single_video_label = QLabel()
        self.single_video_label.setMinimumSize(320, 200)
        self.single_video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.single_video_label.setStyleSheet(
            "background-color:#0E0E0E; border:1px solid #2A2A2A; border-radius:8px; color:#777777;")
        self.single_video_label.setAlignment(Qt.AlignCenter)
        self.single_video_label.setScaledContents(True)
        self.single_video_label.setText("กดเชื่อมต่อกล้องเพื่อดูภาพสด")
        video_col.addWidget(self.single_video_label, 1)

        self.multi_grid_container = QWidget()
        self.multi_grid = QGridLayout(self.multi_grid_container)
        self.multi_grid.setContentsMargins(0, 0, 0, 0)
        self.multi_grid.setSpacing(8)
        self.multi_grid_container.hide()
        video_col.addWidget(self.multi_grid_container, 1)
        main.addLayout(video_col, 3)

        side = QVBoxLayout()
        side.setSpacing(10)

        self.multi_group = QGroupBox("กล้อง (Multi)")
        mg = QVBoxLayout(self.multi_group)
        self.multi_cameras_label = QLabel("เลือกกล้องที่จะดูพร้อมกัน")
        self.multi_cameras_label.setStyleSheet("color:#9A9A92;")
        mg.addWidget(self.multi_cameras_label)
        self.multi_camera_list = QListWidget()
        self.multi_camera_list.setSelectionMode(QListWidget.NoSelection)
        self.multi_camera_list.setMinimumHeight(120)
        mg.addWidget(self.multi_camera_list, 1)
        mrow = QHBoxLayout()
        self.multi_select_all_btn = QPushButton("Select All")
        self.multi_select_all_btn.clicked.connect(lambda: self._set_all_multi_checks(True))
        mrow.addWidget(self.multi_select_all_btn)
        self.multi_clear_btn = QPushButton("Clear")
        self.multi_clear_btn.clicked.connect(lambda: self._set_all_multi_checks(False))
        mrow.addWidget(self.multi_clear_btn)
        mg.addLayout(mrow)
        self.connect_multi_btn = QPushButton("Connect Selected")
        self.connect_multi_btn.clicked.connect(self.connect_selected_cameras)
        mg.addWidget(self.connect_multi_btn)
        side.addWidget(self.multi_group)

        snap_group = QGroupBox("Snapshots ล่าสุด")
        sg = QVBoxLayout(snap_group)
        self.snapshot_admin_list = QListWidget()
        self.snapshot_admin_list.setMinimumHeight(120)
        self.snapshot_admin_list.itemDoubleClicked.connect(self._open_selected_snapshot)
        sg.addWidget(self.snapshot_admin_list, 1)
        self.snapshot_open_btn = QPushButton("Open")
        self.snapshot_open_btn.clicked.connect(self._open_selected_snapshot)
        sg.addWidget(self.snapshot_open_btn)
        self.snapshot_mark_barber_btn = QPushButton("Mark as Barber")
        self.snapshot_mark_barber_btn.clicked.connect(lambda: self._copy_selected_snapshot_to("barber"))
        sg.addWidget(self.snapshot_mark_barber_btn)
        self.snapshot_mark_customer_btn = QPushButton("Mark as Customer")
        self.snapshot_mark_customer_btn.clicked.connect(lambda: self._copy_selected_snapshot_to("customer"))
        sg.addWidget(self.snapshot_mark_customer_btn)
        self.snapshot_mark_wash_customer_btn = QPushButton("Mark as Wash Customer")
        self.snapshot_mark_wash_customer_btn.clicked.connect(
            lambda: self._copy_selected_snapshot_to("wash_customer")
        )
        sg.addWidget(self.snapshot_mark_wash_customer_btn)
        side.addWidget(snap_group, 1)

        side_container = QWidget()
        side_container.setLayout(side)
        side_container.setFixedWidth(290)
        main.addWidget(side_container, 0)
        root.addLayout(main, 1)

        # --- Status bar ---
        status = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color:#E0922A;")
        status.addWidget(self.status_label)
        status.addSpacing(16)
        self.fps_label = QLabel("FPS: 0")
        status.addWidget(self.fps_label)
        status.addSpacing(16)
        self.people_count_label = QLabel("People: 0")
        status.addWidget(self.people_count_label)
        status.addStretch()
        root.addLayout(status)

        self.reload_cameras()
        self._on_view_mode_changed(self.view_mode_combo.currentText())

    def _on_view_mode_changed(self, mode_text: str):
        is_multi = str(mode_text).strip().lower() == "multi"
        self.camera_combo.setEnabled(not is_multi and self.frame_capture is None)
        self.connect_btn.setVisible(not is_multi)
        self.multi_group.setVisible(is_multi)
        self.single_video_label.setVisible(not is_multi)
        self.multi_grid_container.setVisible(is_multi)
        if is_multi and self.frame_capture is not None:
            self.disconnect_camera()
        if (not is_multi) and self.multi_frame_captures:
            self.disconnect_camera()

    def reload_cameras(self):
        """Reload camera list for both single and multi selectors."""
        cameras = list((self.config.get("cameras", {}) or {}).keys())

        selected_single = self.camera_combo.currentText() if hasattr(self, "camera_combo") else ""
        checked_multi = set(self._checked_multi_cameras()) if hasattr(self, "multi_camera_list") else set()

        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        for name in cameras:
            self.camera_combo.addItem(name)
        if selected_single and selected_single in cameras:
            self.camera_combo.setCurrentText(selected_single)
        self.camera_combo.blockSignals(False)

        self.multi_camera_list.clear()
        for name in cameras:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if name in checked_multi else Qt.Unchecked)
            self.multi_camera_list.addItem(item)

        if self.current_camera and self.current_camera not in cameras:
            self.disconnect_camera()
        for cam in list(self.multi_frame_captures.keys()):
            if cam not in cameras:
                self.disconnect_camera()
                break

    def _checked_multi_cameras(self) -> List[str]:
        names: List[str] = []
        for i in range(self.multi_camera_list.count()):
            item = self.multi_camera_list.item(i)
            if item and item.checkState() == Qt.Checked:
                names.append(item.text())
        return names

    def _set_all_multi_checks(self, checked: bool):
        state = Qt.Checked if checked else Qt.Unchecked
        for i in range(self.multi_camera_list.count()):
            item = self.multi_camera_list.item(i)
            if item:
                item.setCheckState(state)

    def load_zones_for_camera(self, camera_name: str):
        """Load zone data from zones JSON file and cache by camera."""
        zones_dir = Path(self.config.get("paths", {}).get("zones", "data/zones"))
        zones_file = zones_dir / f"zones_{camera_name}.json"

        parsed: Dict[str, Dict] = {}
        if zones_file.exists():
            try:
                with open(zones_file, "r", encoding="utf-8") as f:
                    zones_list = json.load(f)
                for zone in zones_list or []:
                    zone_name = zone.get("name", "")
                    points = zone.get("points", [])
                    if not points and "polygon_json" in zone:
                        points = zone.get("polygon_json", [])
                    parsed[zone_name] = {
                        "points": self._parse_points(points),
                        "type": zone.get("type", "OTHER"),
                        "count": 0,
                    }
            except Exception as e:
                self.status_label.setText(f"Error loading zones: {e}")
                self.status_label.setStyleSheet("color: red;")

        self.zones_by_camera[camera_name] = parsed
        if camera_name == self.current_camera:
            self.zones_data = parsed
        return parsed

    def _zones_for_camera(self, camera_name: str) -> Dict[str, Dict]:
        if camera_name not in self.zones_by_camera:
            return self.load_zones_for_camera(camera_name)
        return self.zones_by_camera.get(camera_name, {})

    def _parse_points(self, points: List) -> List[List[float]]:
        out: List[List[float]] = []
        for p in points or []:
            if isinstance(p, dict):
                out.append([float(p.get("x", 0.0)), float(p.get("y", 0.0))])
            else:
                out.append([float(p[0]), float(p[1])])
        return out

    def _to_display_points(self, points: List[List[float]], frame_w: int, frame_h: int) -> np.ndarray:
        if not points:
            return np.array([], dtype=np.int32)
        is_normalized = all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in points)
        if is_normalized:
            return np.array([[int(x * frame_w), int(y * frame_h)] for x, y in points], dtype=np.int32)
        max_x = max(x for x, _ in points) if points else 1.0
        max_y = max(y for _, y in points) if points else 1.0
        sx = frame_w / max(max_x, 1.0)
        sy = frame_h / max(max_y, 1.0)
        return np.array([[int(x * sx), int(y * sy)] for x, y in points], dtype=np.int32)

    def _color_for_gid(self, gid: int):
        if not gid:
            return (0, 220, 0)
        h = (int(gid) * 2654435761) & 0xFFFFFFFF
        r = (h >> 16) & 0xFF
        g = (h >> 8) & 0xFF
        b = h & 0xFF
        return (int(b), int(g), int(r))

    def _camera_url(self, camera_name: str) -> Optional[str]:
        cam_config = (self.config.get("cameras", {}) or {}).get(camera_name)
        if not cam_config:
            return None
        return cam_config.get("rtsp_url") or None

    def _start_capture(self, camera_name: str, on_status):
        url = self._camera_url(camera_name)
        if not url:
            raise ValueError(f"RTSP URL not configured for {camera_name}")
        cap = FrameCapture(url)
        cap.status_changed.connect(on_status)
        cap.start()
        return cap

    def connect_camera(self):
        camera = self.camera_combo.currentText()
        if not camera:
            QMessageBox.warning(self, "Error", "No camera selected")
            return
        try:
            self.disconnect_camera()
            self.current_camera = camera
            self.load_zones_for_camera(camera)
            self.frame_capture = self._start_capture(camera, self.on_status_changed)
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            if self.view_mode_combo.currentText() == "Single":
                self.camera_combo.setEnabled(False)
            self.status_label.setText(f"Connecting to {camera}...")
            self.status_label.setStyleSheet("color: blue;")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def connect_selected_cameras(self):
        selected = self._checked_multi_cameras()
        if not selected:
            QMessageBox.information(self, "Live View", "Please check at least one camera")
            return
        self.view_mode_combo.setCurrentText("Multi")
        try:
            self.disconnect_camera()
            self.multi_camera_order = list(selected)
            self._rebuild_multi_grid(self.multi_camera_order)
            for cam in self.multi_camera_order:
                self.load_zones_for_camera(cam)
                self.multi_frame_captures[cam] = self._start_capture(
                    cam, lambda status, c=cam: self.on_multi_status_changed(c, status)
                )
            self.disconnect_btn.setEnabled(True)
            self.status_label.setText(f"Connecting multi view: {', '.join(self.multi_camera_order)}")
            self.status_label.setStyleSheet("color: blue;")
        except Exception as e:
            self.disconnect_camera()
            QMessageBox.critical(self, "Live View", f"Failed to start multi-camera view: {e}")

    def _rebuild_multi_grid(self, camera_names: List[str]):
        while self.multi_grid.count():
            item = self.multi_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.multi_tile_labels = {}

        cols = 2 if len(camera_names) > 1 else 1
        for idx, cam in enumerate(camera_names):
            tile = QLabel(cam)
            tile.setAlignment(Qt.AlignCenter)
            tile.setMinimumSize(240, 135)
            tile.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            tile.setStyleSheet("background-color: black; border: 1px solid gray; color: white;")
            tile.setScaledContents(True)
            tile.setText(f"{cam}\n(waiting frame...)")
            self.multi_tile_labels[cam] = tile
            self.multi_grid.addWidget(tile, idx // cols, idx % cols)

    def _stop_capture(self, cap: Optional[FrameCapture]):
        if cap is None:
            return
        try:
            cap.stop()
            cap.wait(1500)
        except Exception:
            pass

    def disconnect_camera(self):
        self._stop_capture(self.frame_capture)
        self.frame_capture = None

        for cam, cap in list(self.multi_frame_captures.items()):
            self._stop_capture(cap)
        self.multi_frame_captures.clear()
        self.multi_camera_order = []

        self.current_camera = None
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.camera_combo.setEnabled(self.view_mode_combo.currentText() != "Multi")

        self.single_video_label.clear()
        self.single_video_label.setText("Single camera preview")
        for cam, lbl in list(self.multi_tile_labels.items()):
            lbl.clear()
            lbl.setText(f"{cam}\n(disconnected)")
        self.status_label.setText("Disconnected")
        self.status_label.setStyleSheet("color: orange;")
        self.people_count_label.setText("People: 0")

    def reload_current_zones(self):
        is_multi = self.view_mode_combo.currentText() == "Multi"
        if is_multi:
            cams = self.multi_camera_order or self._checked_multi_cameras()
            if not cams:
                QMessageBox.information(self, "Info", "No cameras selected")
                return
            for cam in cams:
                self.load_zones_for_camera(cam)
            self.status_label.setText(f"Zones reloaded ({len(cams)} cameras)")
            self.status_label.setStyleSheet("color: green;")
            return

        camera = self.current_camera or self.camera_combo.currentText()
        if not camera:
            QMessageBox.information(self, "Info", "No camera selected")
            return
        zones = self.load_zones_for_camera(camera)
        self.status_label.setText(f"Zones reloaded: {camera} ({len(zones)})")
        self.status_label.setStyleSheet("color: green;")

    def on_status_changed(self, status: str):
        self.status_label.setText(status)
        if "Connected" in status:
            self.status_label.setStyleSheet("color: green;")
            self.single_video_label.clear()
        elif "Error" in status or "Failed" in status:
            self.status_label.setStyleSheet("color: red;")
            self.single_video_label.setText(f"Live view error\n{status}")
        else:
            self.status_label.setStyleSheet("color: blue;")

    def on_multi_status_changed(self, camera_name: str, status: str):
        short = f"[{camera_name}] {status}"
        self.status_label.setText(short)
        tile = self.multi_tile_labels.get(camera_name)
        if "Connected" in status:
            self.status_label.setStyleSheet("color: green;")
            if tile is not None:
                tile.clear()
        elif "Error" in status or "Failed" in status:
            self.status_label.setStyleSheet("color: red;")
            if tile is not None:
                tile.setText(f"{camera_name}\n{status}")
        else:
            self.status_label.setStyleSheet("color: blue;")
            if tile is not None and "Connecting" in status:
                tile.setText(f"{camera_name}\nconnecting...")

    def _load_runtime_overlay(self):
        if not self.dashboard_state_file.exists():
            self.last_snapshot_info = ""
            return
        try:
            with open(self.dashboard_state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            status = state.get("status", {})
            raw_mode = status.get("count_mode", (self.config.get("runtime", {}) or {}).get("count_mode", "live"))
            mode = str(raw_mode or "").strip().lower()
            self.current_count_mode = "live" if mode in {"live", "realtime", "real_time"} else "recorded"
            latest = status.get("latest_detections", {}) or status.get("latest_customers", {}) or {}
            snaps = status.get("recent_snapshots", []) or []
            self._sync_recent_snapshots(snaps)
            if self.recent_snapshots:
                s = self.recent_snapshots[-1]
                self.last_snapshot_info = f"Last snap: {s.get('camera', '?')} GID {s.get('gid', 0)}"
            else:
                self.last_snapshot_info = ""

            # In recorded mode, runtime detections are from processed clips and may not
            # align with live RTSP frames, so hide overlay boxes in Live View.
            if self.current_count_mode != "live":
                self.runtime_overlay = {}
                self.detection_cache.clear()
                return

            now = time.time()
            for cam_name, people in latest.items():
                for det in people or []:
                    bbox = det.get("bbox", [])
                    if len(bbox) != 4:
                        continue
                    pid = det.get("pid", 0)
                    gid = det.get("gid", 0)
                    key = f"G{gid}" if gid else f"{cam_name}_P{pid}"
                    self.detection_cache[key] = {
                        "bbox": [int(v) for v in bbox],
                        "pid": pid,
                        "gid": gid,
                        "camera": cam_name,
                        "last_seen": now,
                        "primary_zone": det.get("primary_zone", ""),
                        "staff_id": det.get("staff_id"),
                        "is_staff": det.get("staff_id") is not None,
                    }
                    if gid:
                        other_key = f"{cam_name}_P{pid}"
                        if other_key in self.detection_cache and other_key != key:
                            self.detection_cache[other_key]["last_seen"] = now

            for k in list(self.detection_cache.keys()):
                if now - float(self.detection_cache.get(k, {}).get("last_seen", 0)) > self.detection_cache_ttl * 3:
                    self.detection_cache.pop(k, None)

            self.runtime_overlay = latest
        except Exception:
            self.last_snapshot_info = ""

    def _sync_recent_snapshots(self, snapshots: List[Dict]):
        recent = list((snapshots or [])[-80:])
        signature = "|".join(
            [f"{str(item.get('timestamp', ''))}:{str(item.get('path', ''))}" for item in recent]
        )
        if signature == self._snapshot_signature:
            return
        self._snapshot_signature = signature
        self.recent_snapshots = recent

        selected_path = ""
        current_item = self.snapshot_admin_list.currentItem()
        if current_item is not None:
            selected_path = str(current_item.data(Qt.UserRole) or "")

        self.snapshot_admin_list.blockSignals(True)
        self.snapshot_admin_list.clear()
        restored_row = None
        for item in reversed(recent[-50:]):
            path = str(item.get("path", "") or "")
            camera = str(item.get("camera", "") or "")
            gid = int(item.get("gid", 0) or 0)
            event_type = str(item.get("event_type", "") or "")
            ts = str(item.get("timestamp", "") or "")
            name = Path(path).name if path else "-"
            row = QListWidgetItem(f"{ts} | {event_type} | {camera} | GID {gid} | {name}")
            row.setData(Qt.UserRole, path)
            self.snapshot_admin_list.addItem(row)
            if selected_path and path == selected_path and restored_row is None:
                restored_row = row
        if restored_row is not None:
            self.snapshot_admin_list.setCurrentItem(restored_row)
        self.snapshot_admin_list.blockSignals(False)

    def _selected_snapshot_path(self) -> Optional[Path]:
        item = self.snapshot_admin_list.currentItem()
        if item is None:
            return None
        raw = str(item.data(Qt.UserRole) or "").strip()
        if not raw:
            return None
        return Path(raw)

    def _open_selected_snapshot(self, _item=None):
        path = self._selected_snapshot_path()
        if path is None:
            QMessageBox.information(self, "Live View", "Please select a snapshot first")
            return
        if not path.exists():
            QMessageBox.warning(self, "Live View", f"Snapshot not found:\n{path}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))

    def _copy_selected_snapshot_to(self, role: str):
        src = self._selected_snapshot_path()
        if src is None:
            QMessageBox.information(self, "Live View", "Please select a snapshot first")
            return
        if not src.exists():
            QMessageBox.warning(self, "Live View", f"Snapshot not found:\n{src}")
            return

        role_key = str(role).strip().lower()
        role_dirs = {
            "barber": self.barber_label_dir,
            "customer": self.customer_label_dir,
            "wash_customer": self.wash_customer_label_dir,
        }
        dst_dir = role_dirs.get(role_key, self.customer_label_dir)
        try:
            dst_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "Live View", f"Cannot create target folder:\n{dst_dir}\n\n{e}")
            return

        try:
            dst = dst_dir / src.name
            if dst.exists():
                stem = src.stem
                suffix = src.suffix or ".jpg"
                idx = 1
                while True:
                    cand = dst_dir / f"{stem}_{idx:03d}{suffix}"
                    if not cand.exists():
                        dst = cand
                        break
                    idx += 1
            shutil.copy2(str(src), str(dst))
            if role_key == "barber":
                self.status_label.setText(f"Copied to BARBER_UNIFORM: {dst.name}")
            elif role_key == "wash_customer":
                self.status_label.setText(f"Copied to CUSTOMER_WASH_BY_ADMIN: {dst.name}")
            else:
                self.status_label.setText(f"Copied to CUSTOMER_BY_ADMIN: {dst.name}")
            self.status_label.setStyleSheet("color: green;")
        except Exception as e:
            QMessageBox.critical(self, "Live View", f"Copy failed:\n{e}")

    def _draw_items_for_camera(self, camera_name: str) -> List[Dict]:
        now = time.time()
        out: List[Dict] = []
        for info in self.detection_cache.values():
            if info.get("camera") != camera_name:
                continue
            if now - float(info.get("last_seen", 0)) > self.detection_cache_ttl:
                continue
            bbox = info.get("bbox", [])
            if len(bbox) != 4:
                continue
            out.append(info)
        return out

    def _draw_frame_with_overlay(self, camera_name: str, frame: np.ndarray, include_global_info: bool = False) -> Tuple[np.ndarray, int]:
        draw_frame = frame.copy()
        h, w = draw_frame.shape[:2]

        for zone_name, zone_data in (self._zones_for_camera(camera_name) or {}).items():
            points = zone_data.get("points", [])
            zone_type = zone_data.get("type", "OTHER")
            if not points:
                continue
            pts = self._to_display_points(points, w, h)
            if len(pts) < 3:
                continue
            color_map = {
                "CHAIR": (0, 255, 0),
                "WAIT": (255, 165, 0),
                "WASH": (0, 165, 255),
                "STAFF_AREA": (128, 0, 128),
                "OTHER": (200, 200, 200),
            }
            color = color_map.get(zone_type, (200, 200, 200))
            cv2.polylines(draw_frame, [pts], True, color, 2)
            centroid = pts.mean(axis=0).astype(int)
            cv2.putText(draw_frame, str(zone_name), tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        draw_items = self._draw_items_for_camera(camera_name)
        customer_count = 0
        staff_count = 0
        for det in draw_items:
            x1, y1, x2, y2 = [int(v) for v in det.get("bbox", [])]
            pid = det.get("pid", 0)
            gid = det.get("gid", 0)
            pzone = det.get("primary_zone", "")
            is_staff = bool(det.get("is_staff", False))
            if is_staff:
                staff_count += 1
                color = (0, 165, 255)
            else:
                customer_count += 1
                color = self._color_for_gid(gid)
            cv2.rectangle(draw_frame, (x1, y1), (x2, y2), color, 2)
            if is_staff:
                label = f"BARBER PID:{pid} GID:{gid} {pzone}" if gid else f"BARBER PID:{pid} {pzone}"
            else:
                label = f"PID:{pid} GID:{gid} {pzone}" if gid else f"PID:{pid} {pzone}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(draw_frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(draw_frame, label, (x1 + 3, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        cv2.putText(draw_frame, f"Camera: {camera_name}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(draw_frame, f"People: {customer_count}", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(draw_frame, f"Barbers: {staff_count}", (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        if include_global_info and self.last_snapshot_info:
            cv2.putText(draw_frame, self.last_snapshot_info, (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 0), 2)

        return draw_frame, customer_count

    def _to_pixmap(self, frame_bgr: np.ndarray, size: Tuple[int, int]) -> QPixmap:
        resized = cv2.resize(frame_bgr, size)
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        qt_image = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(qt_image)

    def update_display(self):
        self._load_runtime_overlay()

        self.fps_counter += 1
        now = time.time()
        if now - self.fps_timestamp >= 1.0:
            self.fps_label.setText(f"FPS: {self.fps_counter}")
            self.fps_counter = 0
            self.fps_timestamp = now

        is_multi = self.view_mode_combo.currentText() == "Multi"
        if is_multi:
            total_people = 0
            active_streams = 0
            for cam in self.multi_camera_order:
                cap = self.multi_frame_captures.get(cam)
                lbl = self.multi_tile_labels.get(cam)
                if cap is None or lbl is None:
                    continue
                frame = cap.get_frame()
                if frame is None:
                    continue
                active_streams += 1
                draw_frame, people = self._draw_frame_with_overlay(cam, frame, include_global_info=False)
                total_people += people
                lbl.setPixmap(self._to_pixmap(draw_frame, (640, 360)))
            if self.multi_frame_captures:
                self.people_count_label.setText(f"People: {total_people}")
                if active_streams == 0:
                    self.status_label.setText("Waiting for multi-camera frames...")
                    self.status_label.setStyleSheet("color: blue;")
            return

        if self.frame_capture is None:
            return
        frame = self.frame_capture.get_frame()
        if frame is None:
            return

        camera = self.current_camera or self.camera_combo.currentText() or "Unknown"
        draw_frame, people = self._draw_frame_with_overlay(camera, frame, include_global_info=True)
        cv2.putText(draw_frame, f"UI FPS: {self.fps_counter}", (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        self.single_video_label.setPixmap(self._to_pixmap(draw_frame, (960, 540)))
        self.people_count_label.setText(f"People: {people}")

    def closeEvent(self, event):
        self.disconnect_camera()
        self.display_timer.stop()
        event.accept()
