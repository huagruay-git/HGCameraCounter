"""
Camera Management Dialog and Utilities

เธ•เธฑเธงเธเธฑเธ”เธเธฒเธฃเธเธฅเนเธญเธเธชเธณเธซเธฃเธฑเธ:
- เน€เธเธดเนเธก/เนเธเนเนเธ/เธฅเธเธเธฅเนเธญเธ
- เธ—เธ”เธชเธญเธ RTSP
- เธ”เธนเธ•เธฑเธงเธญเธขเนเธฒเธ + metrics
- Import/Export JSON
"""

import json
import ipaddress
import re
import socket
import time
import threading
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Callable, List
from urllib.parse import quote, urlparse

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QCheckBox, QMessageBox, QTextEdit, QComboBox, QFileDialog, QScrollArea,
    QFrame, QProgressBar, QTreeWidget, QTreeWidgetItem, QSpinBox,
    QListWidget, QListWidgetItem,
    QAbstractItemView
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QPixmap, QImage, QFont


class RTSPTester(QThread):
    """Background thread เธชเธณเธซเธฃเธฑเธเธ—เธ”เธชเธญเธ RTSP connection"""
    
    test_complete = Signal(bool, str, float, float)  # success, message, latency, fps
    
    def __init__(self, rtsp_url: str):
        super().__init__()
        self.rtsp_url = rtsp_url
        
    def run(self):
        try:
            # เน€เธเธทเนเธญเธกเธ•เนเธญเนเธฅเธฐเธเธฑเธเธ—เธถเธเน€เธงเธฅเธฒ
            start_time = time.time()
            cap = cv2.VideoCapture(self.rtsp_url)
            # Some OpenCV builds don't expose CAP_PROP_CONNECT_TIMEOUT; try alternatives and ignore if unsupported
            try:
                if hasattr(cv2, 'CAP_PROP_CONNECT_TIMEOUT'):
                    cap.set(cv2.CAP_PROP_CONNECT_TIMEOUT, 5000)
                elif hasattr(cv2, 'CAP_PROP_OPEN_TIMEOUT_MSEC'):
                    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            except Exception:
                # Not all builds support setting timeout via VideoCapture; ignore and proceed
                pass
            
            # เธเธขเธฒเธขเธฒเธกเธญเนเธฒเธ frame
            ret, frame = cap.read()
            latency = (time.time() - start_time) * 1000  # ms
            
            if not ret:
                cap.release()
                self.test_complete.emit(False, "Cannot read frames", latency, 0)
                return
            
            # เธเธณเธเธงเธ“ FPS (เธ•เธฑเธงเธญเธขเนเธฒเธ)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            
            cap.release()
            self.test_complete.emit(True, "Connection OK", latency, fps)
            
        except Exception as e:
            self.test_complete.emit(False, f"Error: {str(e)}", 0, 0)


class CameraFormDialog(QDialog):
    """Dialog เธชเธณเธซเธฃเธฑเธเน€เธเธดเนเธก/เนเธเนเนเธเธเธฅเนเธญเธ"""
    
    def __init__(
        self,
        parent=None,
        camera_name: str = "",
        camera_config: Dict = None,
        haircut_zones_text: str = "",
    ):
        super().__init__(parent)
        self.camera_name = camera_name
        self.camera_config = camera_config or {}
        self.haircut_zones_text = str(haircut_zones_text or "")
        self.test_result = None
        self.rtsp_tester = None
        self._zone_sync_guard = False
        
        self.init_ui()
        
    def init_ui(self):
        """เธชเธฃเนเธฒเธ UI"""
        layout = QVBoxLayout()
        
        # Camera Name
        layout.addWidget(QLabel("Camera Name:"))
        self.name_input = QLineEdit()
        self.name_input.setText(self.camera_name)
        if self.camera_name:  # เธ–เนเธฒเนเธเนเนเธ เนเธซเน disable name
            self.name_input.setReadOnly(True)
        layout.addWidget(self.name_input)
        
        # RTSP URL
        layout.addWidget(QLabel("RTSP URL:"))
        self.url_input = QLineEdit()
        self.url_input.setText(self.camera_config.get("rtsp_url", ""))
        self.url_input.setPlaceholderText("rtsp://user:pass@192.168.1.100:554/stream")
        layout.addWidget(self.url_input)
        
        # Enabled checkbox
        self.enabled_check = QCheckBox("Enabled")
        self.enabled_check.setChecked(self.camera_config.get("enabled", True))
        layout.addWidget(self.enabled_check)
        
        # Note
        layout.addWidget(QLabel("Note (Optional):"))
        self.note_input = QLineEdit()
        self.note_input.setText(self.camera_config.get("note", ""))
        self.note_input.setPlaceholderText("e.g., Main entrance")
        layout.addWidget(self.note_input)
        
        # Zones file
        layout.addWidget(QLabel("Zones File:"))
        zones_layout = QHBoxLayout()
        self.zones_input = QLineEdit()
        self.zones_input.setText(self.camera_config.get("zones_file", ""))
        self.zones_input.setReadOnly(True)
        zones_layout.addWidget(self.zones_input)
        browse_zones_btn = QPushButton("Browse")
        browse_zones_btn.clicked.connect(self.browse_zones)
        zones_layout.addWidget(browse_zones_btn)
        layout.addLayout(zones_layout)

        # Runtime haircut counting zones for this camera
        layout.addWidget(QLabel("Haircut Count Zones (comma-separated):"))
        self.haircut_zones_input = QLineEdit()
        self.haircut_zones_input.setText(self.haircut_zones_text)
        self.haircut_zones_input.setPlaceholderText("CHAIR_01, CHAIR_02 (empty = disable for this camera)")
        layout.addWidget(self.haircut_zones_input)
        layout.addWidget(QLabel("Tick zones for haircut count:"))
        self.haircut_zone_list = QListWidget()
        self.haircut_zone_list.setMaximumHeight(110)
        layout.addWidget(self.haircut_zone_list)
        reload_zone_btn = QPushButton("Reload Zone List")
        reload_zone_btn.clicked.connect(self._load_haircut_zone_options)
        layout.addWidget(reload_zone_btn)
        
        # Test RTSP button
        layout.addWidget(QLabel("Test Connection:"))
        test_layout = QHBoxLayout()
        self.test_btn = QPushButton("Test RTSP")
        self.test_btn.clicked.connect(self.test_rtsp)
        test_layout.addWidget(self.test_btn)
        self.test_status = QLabel("Not tested")
        test_layout.addWidget(self.test_status)
        layout.addLayout(test_layout)
        
        # Test result area
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(100)
        layout.addWidget(self.result_text)
        
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        
        # Dialog buttons
        layout.addStretch()
        buttons_layout = QHBoxLayout()
        
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        buttons_layout.addWidget(ok_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        buttons_layout.addWidget(cancel_btn)
        
        layout.addLayout(buttons_layout)
        
        self.setLayout(layout)
        self.setWindowTitle(f"{'Edit' if self.camera_name else 'Add'} Camera")
        # Allow slightly narrower dialogs on small screens
        self.setMinimumWidth(420)
        self.haircut_zone_list.itemChanged.connect(self._sync_haircut_zones_text_from_list)
        self.haircut_zones_input.textChanged.connect(self._sync_haircut_zone_list_from_text)
        self.name_input.textChanged.connect(self._on_camera_name_changed)
        self._load_haircut_zone_options()
        
    def browse_zones(self):
        """เน€เธฅเธทเธญเธ zones file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Zones File", "", "JSON Files (*.json)"
        )
        if file_path:
            self.zones_input.setText(file_path)
            self._load_haircut_zone_options()
    
    def _on_camera_name_changed(self, *_args):
        if not self.zones_input.text().strip():
            self._load_haircut_zone_options()

    @staticmethod
    def _parse_zone_tokens(raw: str) -> List[str]:
        out: List[str] = []
        seen = set()
        for token in str(raw or "").replace(";", ",").split(","):
            name = token.strip().upper()
            if not name or name in seen:
                continue
            seen.add(name)
            out.append(name)
        return out

    def _default_zones_file_path(self) -> Optional[Path]:
        cam_name = self.name_input.text().strip()
        if not cam_name:
            return None
        root = Path(__file__).resolve().parent.parent
        path = root / "data" / "zones" / f"zones_{cam_name}.json"
        return path if path.exists() else None

    def _resolve_zones_file_path(self) -> Optional[Path]:
        raw = self.zones_input.text().strip()
        if raw:
            p = Path(raw)
            if not p.is_absolute():
                p = Path(__file__).resolve().parent.parent / p
            if p.exists():
                return p
        return self._default_zones_file_path()

    def _load_haircut_zone_options(self):
        selected = set(self._parse_zone_tokens(self.haircut_zones_input.text()))
        self._zone_sync_guard = True
        try:
            self.haircut_zone_list.clear()
            path = self._resolve_zones_file_path()
            if path is None:
                item = QListWidgetItem("No zones file found for this camera")
                item.setFlags(Qt.ItemIsEnabled)
                self.haircut_zone_list.addItem(item)
                return

            with open(path, "r", encoding="utf-8") as f:
                zones_data = json.load(f) or []

            names: List[str] = []
            if isinstance(zones_data, list):
                for zone in zones_data:
                    if not isinstance(zone, dict):
                        continue
                    name = str(zone.get("name", "") or "").strip()
                    if name:
                        names.append(name)
            elif isinstance(zones_data, dict):
                names = [str(k).strip() for k in zones_data.keys() if str(k).strip()]

            if not names:
                item = QListWidgetItem("Zones file has no named zones")
                item.setFlags(Qt.ItemIsEnabled)
                self.haircut_zone_list.addItem(item)
                return

            names = sorted(names, key=lambda n: (0 if str(n).upper().startswith("CHAIR_") else 1, str(n).upper()))
            for name in names:
                zone_key = str(name).upper()
                item = QListWidgetItem(name)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked if zone_key in selected else Qt.Unchecked)
                self.haircut_zone_list.addItem(item)
        except Exception as e:
            self.haircut_zone_list.clear()
            item = QListWidgetItem(f"Failed to load zones: {e}")
            item.setFlags(Qt.ItemIsEnabled)
            self.haircut_zone_list.addItem(item)
        finally:
            self._zone_sync_guard = False

    def _sync_haircut_zones_text_from_list(self, _item=None):
        if self._zone_sync_guard:
            return
        selected: List[str] = []
        for i in range(self.haircut_zone_list.count()):
            item = self.haircut_zone_list.item(i)
            if not item:
                continue
            if not (item.flags() & Qt.ItemIsUserCheckable):
                continue
            if item.checkState() == Qt.Checked:
                selected.append(item.text().strip().upper())
        self._zone_sync_guard = True
        try:
            self.haircut_zones_input.setText(", ".join(selected))
        finally:
            self._zone_sync_guard = False

    def _sync_haircut_zone_list_from_text(self, text: str):
        if self._zone_sync_guard:
            return
        selected = set(self._parse_zone_tokens(text))
        self._zone_sync_guard = True
        try:
            for i in range(self.haircut_zone_list.count()):
                item = self.haircut_zone_list.item(i)
                if not item or not (item.flags() & Qt.ItemIsUserCheckable):
                    continue
                zone_key = item.text().strip().upper()
                item.setCheckState(Qt.Checked if zone_key in selected else Qt.Unchecked)
        finally:
            self._zone_sync_guard = False

    def test_rtsp(self):
        """เธ—เธ”เธชเธญเธ RTSP connection"""
        url = self.url_input.text().strip()
        if not url:
            QMessageBox.warning(self, "Error", "Please enter RTSP URL")
            return
        
        self.test_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # Indeterminate
        self.result_text.clear()
        self.result_text.setText("Testing connection...")
        
        # เน€เธฃเธดเนเธก background thread
        self.rtsp_tester = RTSPTester(url)
        self.rtsp_tester.test_complete.connect(self.on_test_complete)
        self.rtsp_tester.start()
    
    def on_test_complete(self, success: bool, message: str, latency: float, fps: float):
        """Callback เน€เธกเธทเนเธญ test เน€เธชเธฃเนเธ"""
        self.test_btn.setEnabled(True)
        self.progress.setVisible(False)
        
        if success:
            self.test_status.setText(f"OK (Latency: {latency:.0f}ms, FPS: {fps:.1f})")
            result_msg = f"""
Connection Successful

Latency: {latency:.2f} ms
FPS: {fps:.2f}
Status: Ready
"""
        else:
            self.test_status.setText(f"Failed: {message}")
            result_msg = f"""
Connection Failed

Message: {message}
Latency: {latency:.2f} ms
Status: Check URL and network
"""
        
        self.result_text.setText(result_msg)
        self.test_result = success
    
    def get_camera_data(self) -> Tuple[str, Dict]:
        """เนเธ”เนเธเนเธญเธกเธนเธฅเธเธฅเนเธญเธเธเธฒเธ form"""
        return (
            self.name_input.text(),
            {
                "rtsp_url": self.url_input.text(),
                "enabled": self.enabled_check.isChecked(),
                "note": self.note_input.text(),
                "zones_file": self.zones_input.text()
            }
        )

    def get_haircut_zones_text(self) -> str:
        return self.haircut_zones_input.text().strip()


def _guess_default_subnet() -> str:
    """Guess local /24 subnet. Fallback to common private subnet."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            local_ip = sock.getsockname()[0]
        octets = local_ip.split(".")
        if len(octets) == 4:
            return f"{octets[0]}.{octets[1]}.{octets[2]}.0/24"
    except Exception:
        pass
    return "192.168.1.0/24"


class LANCameraScannerWorker(QThread):
    """Scan IP range and discover RTSP camera endpoints."""

    progress = Signal(int, int, str)  # done, total, message
    camera_found = Signal(dict)  # payload with ip/url/status
    scan_finished = Signal(dict)  # summary payload
    scan_error = Signal(str)

    def __init__(
        self,
        subnet: str,
        username: str,
        password: str,
        path_template: str,
        channel_start: int,
        channel_end: int,
        ports: List[int],
        timeout_ms: int = 1200,
        max_workers: int = 64,
    ):
        super().__init__()
        self.subnet = subnet.strip()
        self.username = username
        self.password = password
        self.path_template = path_template.strip()
        self.channel_start = min(channel_start, channel_end)
        self.channel_end = max(channel_start, channel_end)
        self.ports = ports
        self.timeout_ms = max(200, timeout_ms)
        self.max_workers = max(1, min(max_workers, 256))
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        try:
            network = ipaddress.ip_network(self.subnet, strict=False)
        except ValueError as e:
            self.scan_error.emit(f"Invalid subnet '{self.subnet}': {e}")
            return

        hosts = [str(ip) for ip in network.hosts()]
        total_hosts = len(hosts)
        if total_hosts == 0:
            self.scan_error.emit("No hosts in subnet")
            return
        if total_hosts > 4096:
            self.scan_error.emit("Subnet too large. Please scan no more than 4096 hosts at a time.")
            return

        done = 0
        found = 0
        stopped = False

        pool = ThreadPoolExecutor(max_workers=self.max_workers)
        try:
            futures = {pool.submit(self._scan_host, ip): ip for ip in hosts}
            for future in as_completed(futures):
                if self._stop_requested:
                    stopped = True
                    for f in futures:
                        f.cancel()
                    break

                ip = futures[future]
                done += 1
                try:
                    result = future.result()
                except Exception as e:
                    self.progress.emit(done, total_hosts, f"{ip} error: {e}")
                    continue

                if result:
                    found += 1
                    self.camera_found.emit(result)
                    self.progress.emit(done, total_hosts, f"{ip} found")
                elif done == total_hosts or done % 16 == 0:
                    self.progress.emit(done, total_hosts, f"{ip} scanned")
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

        self.scan_finished.emit(
            {
                "total": total_hosts,
                "scanned": done,
                "found": found,
                "stopped": stopped,
            }
        )

    def _scan_host(self, ip: str) -> Optional[dict]:
        for port in self.ports:
            if self._stop_requested:
                return None
            if not self._is_tcp_port_open(ip, port):
                continue

            first_rtsp_response: Optional[Tuple[int, str]] = None
            for rtsp_url in self._candidate_urls(ip, port):
                if self._stop_requested:
                    return None
                code, status_line = self._rtsp_options(rtsp_url)
                if code is None:
                    continue
                if first_rtsp_response is None:
                    first_rtsp_response = (code, status_line)
                if code in (200, 401):
                    frame_ok = code == 200
                    frame_note = "RTSP endpoint reachable" if code == 200 else "Auth required"
                    return {
                        "ip": ip,
                        "port": port,
                        "rtsp_url": rtsp_url,
                        "status_code": code,
                        "status_text": status_line,
                        "frame_ok": frame_ok,
                        "note": frame_note,
                    }

            if first_rtsp_response is not None:
                code, status_line = first_rtsp_response
                return {
                    "ip": ip,
                    "port": port,
                    "rtsp_url": "",
                    "status_code": code,
                    "status_text": status_line,
                    "frame_ok": False,
                    "note": "RTSP service detected but no known stream path matched",
                }
        return None

    def _candidate_urls(self, ip: str, port: int) -> List[str]:
        auth = ""
        if self.username or self.password:
            auth = f"{quote(self.username, safe='')}:{quote(self.password, safe='')}@"
        base = f"rtsp://{auth}{ip}:{port}"

        def normalize_path(path: str) -> str:
            p = path.strip()
            if not p:
                return ""
            if not p.startswith("/"):
                p = f"/{p}"
            return p

        candidate_paths: List[str] = []
        template = self.path_template
        if template:
            if "{channel}" in template:
                for ch in range(self.channel_start, self.channel_end + 1):
                    candidate_paths.append(normalize_path(template.replace("{channel}", str(ch))))
            else:
                candidate_paths.append(normalize_path(template))
        else:
            candidate_paths.extend(
                [
                    "/cam/realmonitor?channel=1&subtype=0",
                    "/Streaming/Channels/101",
                    "/h264/ch1/main/av_stream",
                    "/live/ch00_0",
                ]
            )

        dedup: List[str] = []
        seen = set()
        for path in candidate_paths:
            if not path or path in seen:
                continue
            dedup.append(path)
            seen.add(path)

        return [f"{base}{path}" for path in dedup]

    def _is_tcp_port_open(self, ip: str, port: int) -> bool:
        timeout_sec = self.timeout_ms / 1000.0
        try:
            with socket.create_connection((ip, port), timeout=timeout_sec):
                return True
        except Exception:
            return False

    def _rtsp_options(self, rtsp_url: str) -> Tuple[Optional[int], str]:
        timeout_sec = self.timeout_ms / 1000.0
        try:
            parsed = urlparse(rtsp_url)
            host = parsed.hostname
            port = parsed.port or 554
            if not host:
                return None, "Invalid URL host"

            request = (
                f"OPTIONS {rtsp_url} RTSP/1.0\r\n"
                "CSeq: 1\r\n"
                "User-Agent: HGCameraCounter/1.0\r\n"
                "\r\n"
            ).encode("ascii", errors="ignore")

            with socket.create_connection((host, port), timeout=timeout_sec) as sock:
                sock.settimeout(timeout_sec)
                sock.sendall(request)
                data = sock.recv(4096)

            if not data:
                return None, "No RTSP response"

            header_text = data.decode("utf-8", errors="ignore")
            first_line = header_text.splitlines()[0].strip() if header_text else ""
            match = re.search(r"RTSP/\d\.\d\s+(\d{3})", first_line)
            if not match:
                return None, first_line or "Invalid RTSP header"
            return int(match.group(1)), first_line
        except Exception as e:
            return None, str(e)

    def _verify_frame(self, rtsp_url: str) -> Tuple[bool, str]:
        try:
            start = time.time()
            cap = cv2.VideoCapture(rtsp_url)
            try:
                if hasattr(cv2, "CAP_PROP_CONNECT_TIMEOUT"):
                    cap.set(cv2.CAP_PROP_CONNECT_TIMEOUT, self.timeout_ms)
                elif hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.timeout_ms)
            except Exception:
                pass
            ok, _ = cap.read()
            cap.release()
            elapsed = (time.time() - start) * 1000.0
            if ok:
                return True, f"Frame OK ({elapsed:.0f}ms)"
            return False, "Connected but no frame"
        except Exception as e:
            return False, f"Frame test failed: {e}"


class LANCameraScannerDialog(QDialog):
    """Dialog for scanning LAN IP range and adding discovered cameras."""

    def __init__(self, parent, manager: "CameraManagerWidget"):
        super().__init__(parent)
        self.manager = manager
        self.worker: Optional[LANCameraScannerWorker] = None
        self.results: List[dict] = []
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle("Scan LAN Cameras")
        self.setMinimumWidth(940)
        self.setMinimumHeight(640)

        layout = QVBoxLayout(self)

        subnet_row = QHBoxLayout()
        subnet_row.addWidget(QLabel("Subnet (CIDR):"))
        self.subnet_input = QLineEdit(_guess_default_subnet())
        subnet_row.addWidget(self.subnet_input, 2)
        subnet_row.addWidget(QLabel("Ports:"))
        self.ports_input = QLineEdit("554,8554")
        self.ports_input.setPlaceholderText("e.g. 554,8554")
        subnet_row.addWidget(self.ports_input, 1)
        layout.addLayout(subnet_row)

        auth_row = QHBoxLayout()
        auth_row.addWidget(QLabel("Username:"))
        self.username_input = QLineEdit("admin")
        auth_row.addWidget(self.username_input, 1)
        auth_row.addWidget(QLabel("Password:"))
        self.password_input = QLineEdit("")
        self.password_input.setEchoMode(QLineEdit.Password)
        auth_row.addWidget(self.password_input, 1)
        layout.addLayout(auth_row)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Path Template:"))
        self.path_template_input = QLineEdit("/cam/realmonitor?channel={channel}&subtype=0")
        self.path_template_input.setPlaceholderText("/cam/realmonitor?channel={channel}&subtype=0")
        path_row.addWidget(self.path_template_input, 3)
        path_row.addWidget(QLabel("Ch:"))
        self.channel_start_spin = QSpinBox()
        self.channel_start_spin.setRange(1, 64)
        self.channel_start_spin.setValue(1)
        path_row.addWidget(self.channel_start_spin)
        path_row.addWidget(QLabel("to"))
        self.channel_end_spin = QSpinBox()
        self.channel_end_spin.setRange(1, 64)
        self.channel_end_spin.setValue(4)
        path_row.addWidget(self.channel_end_spin)
        layout.addLayout(path_row)

        scan_row = QHBoxLayout()
        scan_row.addWidget(QLabel("Timeout (ms):"))
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(200, 10000)
        self.timeout_spin.setValue(1200)
        scan_row.addWidget(self.timeout_spin)
        scan_row.addWidget(QLabel("Workers:"))
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 256)
        self.workers_spin.setValue(64)
        scan_row.addWidget(self.workers_spin)
        scan_row.addStretch()
        layout.addLayout(scan_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        self.result_tree = QTreeWidget()
        self.result_tree.setHeaderLabels(["IP", "Port", "RTSP URL", "Status", "Frame", "Note"])
        self.result_tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.result_tree.setColumnWidth(0, 130)
        self.result_tree.setColumnWidth(1, 65)
        self.result_tree.setColumnWidth(2, 340)
        self.result_tree.setColumnWidth(3, 160)
        self.result_tree.setColumnWidth(4, 70)
        self.result_tree.setColumnWidth(5, 260)
        layout.addWidget(self.result_tree, 1)

        actions = QHBoxLayout()
        self.start_btn = QPushButton("Start Scan")
        self.start_btn.clicked.connect(self._start_scan)
        actions.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_scan)
        actions.addWidget(self.stop_btn)

        self.add_selected_btn = QPushButton("Add Selected")
        self.add_selected_btn.clicked.connect(self._add_selected)
        actions.addWidget(self.add_selected_btn)

        self.add_all_btn = QPushButton("Add All Stream URLs")
        self.add_all_btn.clicked.connect(self._add_all)
        actions.addWidget(self.add_all_btn)

        actions.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        actions.addWidget(close_btn)
        layout.addLayout(actions)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1500)
        super().closeEvent(event)

    def _parse_ports(self) -> Optional[List[int]]:
        raw = self.ports_input.text().strip()
        if not raw:
            return [554]
        ports: List[int] = []
        try:
            for part in raw.split(","):
                p = int(part.strip())
                if p < 1 or p > 65535:
                    raise ValueError(f"Port out of range: {p}")
                if p not in ports:
                    ports.append(p)
        except Exception:
            return None
        return ports

    def _start_scan(self):
        ports = self._parse_ports()
        if not ports:
            QMessageBox.warning(self, "Scan Error", "Invalid port list. Example: 554,8554")
            return

        subnet = self.subnet_input.text().strip()
        if not subnet:
            QMessageBox.warning(self, "Scan Error", "Subnet is required")
            return

        self.results = []
        self.result_tree.clear()
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting scan...")

        self.worker = LANCameraScannerWorker(
            subnet=subnet,
            username=self.username_input.text().strip(),
            password=self.password_input.text(),
            path_template=self.path_template_input.text().strip(),
            channel_start=self.channel_start_spin.value(),
            channel_end=self.channel_end_spin.value(),
            ports=ports,
            timeout_ms=self.timeout_spin.value(),
            max_workers=self.workers_spin.value(),
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.camera_found.connect(self._on_found)
        self.worker.scan_finished.connect(self._on_finished)
        self.worker.scan_error.connect(self._on_error)
        self.worker.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def _stop_scan(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.status_label.setText("Stopping scan...")

    def _on_progress(self, done: int, total: int, message: str):
        pct = int((done / total) * 100) if total else 0
        self.progress_bar.setValue(max(0, min(100, pct)))
        self.status_label.setText(f"Scanning {done}/{total}, found {len(self.results)} | {message}")

    def _on_found(self, payload: dict):
        self.results.append(payload)
        item = QTreeWidgetItem(
            [
                str(payload.get("ip", "")),
                str(payload.get("port", "")),
                str(payload.get("rtsp_url", "")),
                str(payload.get("status_text", "")),
                "Yes" if payload.get("frame_ok") else "No",
                str(payload.get("note", "")),
            ]
        )
        item.setData(0, Qt.UserRole, payload)
        self.result_tree.addTopLevelItem(item)

    def _on_finished(self, summary: dict):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if summary.get("stopped"):
            self.status_label.setText(
                f"Stopped: scanned {summary.get('scanned', 0)}/{summary.get('total', 0)} hosts, found {summary.get('found', 0)}"
            )
        else:
            self.progress_bar.setValue(100)
            self.status_label.setText(
                f"Done: scanned {summary.get('scanned', 0)} hosts, found {summary.get('found', 0)}"
            )

    def _on_error(self, message: str):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText(message)
        QMessageBox.warning(self, "Scan Error", message)

    def _selected_payloads(self) -> List[dict]:
        selected = []
        for item in self.result_tree.selectedItems():
            payload = item.data(0, Qt.UserRole)
            if isinstance(payload, dict):
                selected.append(payload)
        return selected

    def _add_selected(self):
        payloads = self._selected_payloads()
        if not payloads:
            QMessageBox.information(self, "Add Cameras", "Please select at least one discovered camera")
            return
        self._add_payloads_to_config(payloads)

    def _add_all(self):
        if not self.results:
            QMessageBox.information(self, "Add Cameras", "No discovered cameras")
            return
        self._add_payloads_to_config(self.results)

    def _add_payloads_to_config(self, payloads: List[dict]):
        try:
            cameras = self.manager.config.get("cameras", {})
            if not isinstance(cameras, dict):
                cameras = {}

            added = 0
            skipped = 0
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            existing_urls = {
                str(cfg.get("rtsp_url", "")).strip()
                for cfg in cameras.values()
                if isinstance(cfg, dict)
            }

            for payload in payloads:
                rtsp_url = str(payload.get("rtsp_url") or "").strip()
                if not rtsp_url:
                    skipped += 1
                    continue
                if rtsp_url in existing_urls:
                    skipped += 1
                    continue

                ip = str(payload.get("ip", "unknown")).replace(".", "_")
                base_name = f"Camera_{ip}"
                cam_name = base_name
                index = 2
                while cam_name in cameras:
                    cam_name = f"{base_name}_{index}"
                    index += 1

                cameras[cam_name] = {
                    "rtsp_url": rtsp_url,
                    "enabled": True,
                    "note": f"Discovered by LAN scan at {timestamp}",
                    "zones_file": "",
                }
                existing_urls.add(rtsp_url)
                added += 1

            if added > 0:
                self.manager.config["cameras"] = cameras
                self.manager.controller.save_config()
                self.manager.refresh_camera_list()
                self.manager._refresh_live_view_cameras()

            QMessageBox.information(
                self,
                "Add Cameras",
                f"Added {added} camera(s). Skipped {skipped} item(s) (missing or duplicate URL).",
            )
        except Exception as e:
            QMessageBox.critical(self, "Add Cameras Error", f"Failed to add cameras: {e}")


class CameraManagerWidget:
    """Widget utilities เธชเธณเธซเธฃเธฑเธเธเธฑเธ”เธเธฒเธฃเธเธฅเนเธญเธเนเธ main controller"""
    
    def __init__(self, main_controller):
        self.controller = main_controller
        self.config = main_controller.config  # This is a dict in main_controller

    @staticmethod
    def _normalize_zone_tokens(raw: object) -> List[str]:
        tokens: List[str] = []
        seen = set()
        if isinstance(raw, str):
            candidates = raw.replace(";", ",").split(",")
        elif isinstance(raw, (list, tuple, set)):
            candidates = list(raw)
        else:
            candidates = []
        for item in candidates:
            token = str(item or "").strip().upper()
            if not token or token in seen:
                continue
            seen.add(token)
            tokens.append(token)
        return tokens

    def _get_runtime_config(self) -> Dict:
        runtime_cfg = self.config.get("runtime", {})
        if not isinstance(runtime_cfg, dict):
            return {}
        return runtime_cfg

    def _get_haircut_zone_map(self) -> Dict[str, List[str]]:
        runtime_cfg = self._get_runtime_config()
        raw = runtime_cfg.get("haircut_count_zones", {})
        if not isinstance(raw, dict):
            return {}
        out: Dict[str, List[str]] = {}
        for cam, zones_raw in raw.items():
            cam_name = str(cam or "").strip()
            if not cam_name:
                continue
            zones = self._normalize_zone_tokens(zones_raw)
            if zones:
                out[cam_name] = zones
        return out

    def haircut_zones_text_for_camera(self, cam_name: str) -> str:
        zone_map = self._get_haircut_zone_map()
        zones = zone_map.get(str(cam_name or "").strip(), [])
        return ", ".join(zones)

    def _set_haircut_zones_for_camera(self, cam_name: str, zones_text: str):
        cam_key = str(cam_name or "").strip()
        if not cam_key:
            return
        runtime_cfg = dict(self._get_runtime_config())
        zone_map = dict(self._get_haircut_zone_map())
        zones = self._normalize_zone_tokens(zones_text)
        if zones:
            zone_map[cam_key] = zones
        else:
            zone_map.pop(cam_key, None)
        runtime_cfg["haircut_count_zones"] = zone_map
        if hasattr(self.config, "set"):
            self.config.set("runtime", runtime_cfg)
        else:
            self.config["runtime"] = runtime_cfg

    def _hot_apply_haircut_zone_map(self):
        override_path = getattr(self.controller, "runtime_settings_override_file", None)
        if override_path is None:
            return
        try:
            payload = {}
            override_file = Path(override_path)
            if override_file.exists():
                with open(override_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f) or {}
                    if isinstance(loaded, dict):
                        payload = loaded
            payload["haircut_count_zones"] = self._get_haircut_zone_map()
            override_file.parent.mkdir(parents=True, exist_ok=True)
            with open(override_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _refresh_live_view_cameras(self):
        live_viewer = getattr(self.controller, "live_viewer", None)
        if live_viewer and hasattr(live_viewer, "reload_cameras"):
            try:
                live_viewer.reload_cameras()
            except Exception:
                pass
        zone_editor = getattr(self.controller, "zone_editor", None)
        if zone_editor and hasattr(zone_editor, "reload_cameras"):
            try:
                zone_editor.reload_cameras()
            except Exception:
                pass

    def open_lan_scanner_dialog(self):
        dialog = LANCameraScannerDialog(self.controller, self)
        dialog.exec()
    
    def add_camera_dialog(self):
        """เน€เธเธดเธ” dialog เน€เธเธดเนเธกเธเธฅเนเธญเธ"""
        dialog = CameraFormDialog(self.controller)
        if dialog.exec() == QDialog.Accepted:
            cam_name, cam_config = dialog.get_camera_data()
            haircut_zones_text = dialog.get_haircut_zones_text()
            
            if not cam_name:
                QMessageBox.warning(self.controller, "Error", "Camera name is required")
                return
            
            if not cam_config["rtsp_url"]:
                QMessageBox.warning(self.controller, "Error", "RTSP URL is required")
                return
            
            # เน€เธเธดเนเธกเนเธเธเธฅเนเธญเธ config
            if "cameras" not in self.config:
                self.config["cameras"] = {}
            
            if cam_name in self.config["cameras"]:
                QMessageBox.warning(self.controller, "Error", f"Camera '{cam_name}' already exists")
                return
            
            self.config["cameras"][cam_name] = cam_config
            self._set_haircut_zones_for_camera(cam_name, haircut_zones_text)
            self.controller.save_config()
            self._hot_apply_haircut_zone_map()
            
            # เธญเธฑเธเน€เธ”เธ• UI
            self.refresh_camera_list()
            self._refresh_live_view_cameras()
            QMessageBox.information(self.controller, "Success", f"Camera '{cam_name}' added")
    
    def edit_camera_dialog(self):
        """เน€เธเธดเธ” dialog เนเธเนเนเธเธเธฅเนเธญเธ"""
        if not self.controller.camera_list.selectedItems():
            QMessageBox.warning(self.controller, "Error", "Please select a camera")
            return
        
        selected_item = self.controller.camera_list.selectedItems()[0]
        cam_name = selected_item.text(0)
        cam_config = self.config.get("cameras", {}).get(cam_name)
        
        if not cam_config:
            QMessageBox.warning(self.controller, "Error", "Camera not found")
            return
        
        dialog = CameraFormDialog(
            self.controller,
            cam_name,
            cam_config,
            haircut_zones_text=self.haircut_zones_text_for_camera(cam_name),
        )
        if dialog.exec() == QDialog.Accepted:
            _, updated_config = dialog.get_camera_data()
            haircut_zones_text = dialog.get_haircut_zones_text()
            cameras = self.config.get("cameras", {})
            cameras[cam_name] = {**cameras.get(cam_name, {}), **updated_config}
            self.config["cameras"] = cameras
            self._set_haircut_zones_for_camera(cam_name, haircut_zones_text)
            self.controller.save_config()
            self._hot_apply_haircut_zone_map()
            
            # เธญเธฑเธเน€เธ”เธ• UI
            self.refresh_camera_list()
            self._refresh_live_view_cameras()
            QMessageBox.information(self.controller, "Success", f"Camera '{cam_name}' updated")
    
    def delete_camera(self):
        """เธฅเธเธเธฅเนเธญเธ"""
        if not self.controller.camera_list.selectedItems():
            QMessageBox.warning(self.controller, "Error", "Please select a camera")
            return
        
        selected_item = self.controller.camera_list.selectedItems()[0]
        cam_name = selected_item.text(0)
        
        reply = QMessageBox.question(
            self.controller,
            "Confirm Delete",
            f"Delete camera '{cam_name}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            del self.config["cameras"][cam_name]
            self._set_haircut_zones_for_camera(cam_name, "")
            self.controller.save_config()
            self._hot_apply_haircut_zone_map()
            self.refresh_camera_list()
            self._refresh_live_view_cameras()
            QMessageBox.information(self.controller, "Success", f"Camera '{cam_name}' deleted")
    
    def test_camera(self):
        """เธ—เธ”เธชเธญเธเธเธฅเนเธญเธเธ—เธตเนเน€เธฅเธทเธญเธ"""
        if not self.controller.camera_list.selectedItems():
            QMessageBox.warning(self.controller, "Error", "Please select a camera")
            return
        
        selected_item = self.controller.camera_list.selectedItems()[0]
        cam_name = selected_item.text(0)
        cam_config = self.config.get("cameras", {}).get(cam_name)
        
        if not cam_config:
            QMessageBox.warning(self.controller, "Error", "Camera not found")
            return
        
        url = cam_config.get("rtsp_url")
        if not url:
            QMessageBox.warning(self.controller, "Error", "RTSP URL not configured")
            return
        
        # เธ—เธ”เธชเธญเธเนเธเธ inline
        dialog = CameraFormDialog(self.controller, cam_name, cam_config)
        dialog.url_input.setText(url)
        dialog.test_rtsp()
        dialog.exec()
    
    def test_all_cameras(self):
        """เธ—เธ”เธชเธญเธเธเธฅเนเธญเธเธ—เธฑเนเธเธซเธกเธ”"""
        cameras = self.config.get("cameras", {})
        if not cameras:
            QMessageBox.warning(self.controller, "Error", "No cameras configured")
            return
        
        results = {}
        for cam_name, cam_config in cameras.items():
            url = cam_config.get("rtsp_url")
            if not url:
                results[cam_name] = "โ No URL"
                continue
            
            try:
                start_time = time.time()
                cap = cv2.VideoCapture(url)
                try:
                    if hasattr(cv2, 'CAP_PROP_CONNECT_TIMEOUT'):
                        cap.set(cv2.CAP_PROP_CONNECT_TIMEOUT, 5000)
                    elif hasattr(cv2, 'CAP_PROP_OPEN_TIMEOUT_MSEC'):
                        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                except Exception:
                    pass
                ret, _ = cap.read()
                latency = (time.time() - start_time) * 1000
                cap.release()
                
                if ret:
                    results[cam_name] = f"OK ({latency:.0f}ms)"
                else:
                    results[cam_name] = f"โ Cannot read frames ({latency:.0f}ms)"
            except Exception as e:
                results[cam_name] = f"โ {str(e)[:50]}"
        
        # เนเธชเธ”เธเธเธฅ
        report = "Camera Test Results:\n" + "=" * 50 + "\n\n"
        for cam_name, result in results.items():
            report += f"{cam_name}: {result}\n"
        
        QMessageBox.information(self.controller, "Test Results", report)
    
    def import_cameras_json(self):
        """Import cameras เธเธฒเธเนเธเธฅเน JSON"""
        file_path, _ = QFileDialog.getOpenFileName(
            self.controller, "Import Cameras", "", "JSON Files (*.json)"
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'r') as f:
                imported_cameras = json.load(f)
            
            if not isinstance(imported_cameras, dict):
                QMessageBox.warning(self.controller, "Error", "Invalid JSON format")
                return
            
            # Merge cameras
            if "cameras" not in self.config:
                self.config["cameras"] = {}
            
            added = 0
            for cam_name, cam_config in imported_cameras.items():
                if cam_name not in self.config["cameras"]:
                    self.config["cameras"][cam_name] = cam_config
                    added += 1
            
            self.controller.save_config()
            self.refresh_camera_list()
            self._refresh_live_view_cameras()
            QMessageBox.information(self.controller, "Success", f"Imported {added} cameras")
        
        except Exception as e:
            QMessageBox.critical(self.controller, "Error", f"Import failed: {str(e)}")
    
    def export_cameras_json(self):
        """Export cameras เน€เธเนเธเนเธเธฅเน JSON"""
        cameras = self.config.get("cameras", {})
        if not cameras:
            QMessageBox.warning(self.controller, "Error", "No cameras to export")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self.controller, "Export Cameras", "cameras.json", "JSON Files (*.json)"
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'w') as f:
                json.dump(cameras, f, indent=2)
            
            QMessageBox.information(self.controller, "Success", f"Exported to {file_path}")
        
        except Exception as e:
            QMessageBox.critical(self.controller, "Error", f"Export failed: {str(e)}")
    
    def refresh_camera_list(self):
        """เธฃเธตเน€เธเธฃเธ camera list เนเธ UI"""
        self.controller.camera_list.clear()
        
        cameras = self.config.get("cameras", {})
        for cam_name, cam_config in cameras.items():
            item = self.controller.camera_list.insertTopLevelItem(
                self.controller.camera_list.topLevelItemCount(),
                self._create_camera_item(
                    cam_name,
                    cam_config,
                    self.haircut_zones_text_for_camera(cam_name),
                )
            )
    
    @staticmethod
    def _create_camera_item(cam_name: str, cam_config: Dict, haircut_zones_text: str = "") -> QTreeWidgetItem:
        """เธชเธฃเนเธฒเธ tree item เธชเธณเธซเธฃเธฑเธเธเธฅเนเธญเธ"""
        return QTreeWidgetItem([
            cam_name,
            cam_config.get("rtsp_url", "")[:40],
            "Yes" if cam_config.get("enabled", True) else "No",
            cam_config.get("zones_file", ""),
            haircut_zones_text,
        ])

