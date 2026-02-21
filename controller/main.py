"""
Main Controller Application

Features:
- Dashboard: Status overview
- Setup Wizard: Configuration steps
- Camera Management
- Zone Editing
- Staff DB Builder
- Diagnostics
- Logs Viewer
- Service Control (Start/Stop/Restart)
"""

import os
import sys
import json
import threading
import subprocess
import time
import random
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import yaml
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTabWidget, QStatusBar, QMessageBox,
    QTreeWidget, QTreeWidgetItem, QTextEdit, QSplitter, QSpinBox, QFileDialog,
    QListWidget, QListWidgetItem, QDoubleSpinBox, QComboBox, QLineEdit,
    QFormLayout, QGroupBox, QProgressBar, QTableWidget, QTableWidgetItem,
    QHeaderView
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject, QThread
from PySide6.QtGui import QIcon, QFont, QColor, QDesktopServices, QImage, QPixmap
from PySide6.QtCore import QUrl

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import Config
from shared.logger import setup_logger
from shared.dashboard_updater import init_dashboard_service, get_broadcaster
from shared.updater import Updater
from controller.setup_wizard import SetupWizard
from controller.dashboard_client import GUIDashboardClient
from controller.camera_manager import CameraManagerWidget
from controller.zone_editor import ZoneEditorWidget
from controller.staff_builder import StaffBuilderWidget
from controller.live_viewer import LiveViewerWidget

# =========================
# SETUP
# =========================
CONFIG = Config("data/config/config.yaml")
logger = setup_logger("controller", CONFIG.get("paths", {}).get("logs", "logs"))


class CommandWorker(QObject):
    """Run subprocess command without blocking the UI thread."""
    output = Signal(str)
    finished = Signal(int, str)

    def __init__(self, cmd: list[str], cwd: Optional[str] = None):
        super().__init__()
        self.cmd = cmd
        self.cwd = cwd

    def run(self):
        logs: list[str] = []
        try:
            proc = subprocess.Popen(
                self.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=self.cwd,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                logs.append(line)
                self.output.emit(line.rstrip("\n"))
            code = proc.wait()
            self.finished.emit(code, "".join(logs))
        except Exception as e:
            msg = f"Command failed: {e}"
            self.output.emit(msg)
            self.finished.emit(1, "\n".join(logs + [msg]))


# =========================
# MAIN CONTROLLER
# =========================

class MainController(QMainWindow):
    """Main Controller Application"""
    
    def __init__(self):
        super().__init__()
        self.config = CONFIG
        # Cache of last-known active people count (avoid UI resetting when state missing keys)
        self.cached_active_tracks: Optional[int] = None
        # Cache per-camera people counts keyed by camera name
        self.cached_camera_people: Dict[str, int] = {}
        self.recorder_process: Optional[Any] = None
        self.processor_process: Optional[Any] = None
        self.runtime_output_handle = None
        self.is_running = False
        root = self._project_root()
        self.dashboard_state_file = root / self.config.get("paths", {}).get("dashboard_state", "runtime/dashboard_state.json")
        self.reset_counts_flag_file = root / self.config.get("paths", {}).get("reset_counts_flag", "runtime/reset_counts.flag")
        self.runtime_settings_override_file = root / self.config.get("paths", {}).get("runtime_settings_override", "runtime/runtime_settings.override.json")
        self.runtime_stdout_file = root / self.config.get("paths", {}).get("logs", "logs") / f"runtime_{datetime.now().strftime('%Y-%m-%d')}.log"
        self.last_runtime_state_ts = 0.0
        self._live_log_pos = 0
        self._live_log_file: Optional[Path] = None
        self._live_log_cache: list[str] = []
        self.yolo_class_names = ["person", "head_customer", "staff_uniform"]
        self.lab_items: Dict[str, Dict[str, Any]] = {}
        self.lab_current_image_path: Optional[str] = None
        self.lab_current_detections: list[Dict[str, Any]] = []
        
        # Initialize dashboard service
        self.broadcaster = init_dashboard_service()
        self.dashboard_client: Optional[GUIDashboardClient] = None
        
        # Initialize camera manager
        self.camera_manager = CameraManagerWidget(self)
        
        self.setWindowTitle("HG Camera Counter - Controller")
        self.setGeometry(100, 100, 1200, 800)
        
        # Setup UI
        self.setup_ui()
        
        # Timers
        self.status_check_timer = QTimer()
        self.status_check_timer.timeout.connect(self.check_runtime_status)
        self.status_check_timer.start(2000)  # Check every 2 seconds
        
        self.status_check_timer.start(2000)  # Check every 2 seconds
        
        # self.check_runtime_status() # Skip initial check to avoid race conditions or errors if processes aren't set up yet? 
        # Actually better to just initialize variables to None in __init__ which I did.
        # But wait, the traceback said:
        # File "/Users/supachaimumdang/projectcount/HGCameraCounter/controller/main.py", line 102, in __init__
        # self.check_runtime_status()
        # So I should keep it but ensure variables exist.
        self.check_runtime_status()
    
    def closeEvent(self, event):
        """Ensure services are stopped on exit."""
        logger.info("Main window is closing. Stopping services...")
        self.stop_service()
        self.status_check_timer.stop()
        event.accept()

    def setup_ui(self):
        """Setup user interface"""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        
        # Header
        header_layout = QHBoxLayout()
        
        title = QLabel(f"HG Camera Counter - {self.config.get('branch_code', 'DEMO')}")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Start/Stop button
        self.start_btn = QPushButton("▶ Start Service")
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 5px;")
        self.start_btn.clicked.connect(self.start_service)
        header_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("⏹ Stop Service")
        self.stop_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 5px;")
        self.stop_btn.clicked.connect(self.stop_service)
        self.stop_btn.setEnabled(False)
        header_layout.addWidget(self.stop_btn)

        # Check updates button
        self.check_updates_btn = QPushButton("⬆ Check Updates")
        self.check_updates_btn.setToolTip("Check for remote updates")
        self.check_updates_btn.clicked.connect(self.check_updates)
        header_layout.addWidget(self.check_updates_btn)
        # Install updates button
        self.install_update_btn = QPushButton("⬇ Install Update")
        self.install_update_btn.setToolTip("Run a staged installer from updates/")
        self.install_update_btn.clicked.connect(self.install_update)
        header_layout.addWidget(self.install_update_btn)
        
        layout.addLayout(header_layout)
        
        # Tabs
        self.tabs = QTabWidget()
        
        self.tab_dashboard()
        self.tab_live_view()
        self.tab_setup()
        self.tab_cameras()
        # Staff DB Builder
        try:
            self.tab_staff_db()
        except Exception:
            logger.exception("Failed to initialize Staff DB tab")
        # Zones tab (Zone Editor)
        try:
            self.tab_zones()
        except Exception:
            # If zone editor fails to initialize, continue gracefully
            logger.exception("Failed to initialize Zones tab")
        self.tab_diagnostics()
        self.tab_logs()
        self.tab_model_train()
        self.tab_model_test()
        self.tab_dataset_lab()
        
        layout.addWidget(self.tabs)
        
        central_widget.setLayout(layout)
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def tab_dashboard(self):
        """Dashboard tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Status panel
        status_group_layout = QHBoxLayout()
        
        # Runtime status
        self.status_runtime = QLabel("⚫ Stopped")
        self.status_runtime.setFont(QFont("Arial", 12, QFont.Bold))
        status_group_layout.addWidget(QLabel("Runtime:"))
        status_group_layout.addWidget(self.status_runtime)
        
        status_group_layout.addSpacing(30)
        
        # Last heartbeat
        self.status_heartbeat = QLabel("Never")
        status_group_layout.addWidget(QLabel("Last Heartbeat:"))
        status_group_layout.addWidget(self.status_heartbeat)
        
        status_group_layout.addSpacing(30)
        
        # Active tracks
        self.status_active = QLabel("0")
        status_group_layout.addWidget(QLabel("Active People:"))
        status_group_layout.addWidget(self.status_active)
        
        status_group_layout.addStretch()
        layout.addLayout(status_group_layout)
        
        # Camera status tree
        layout.addWidget(QLabel("Camera Status (Real-time):"))
        self.camera_tree = QTreeWidget()
        self.camera_tree.setHeaderLabels(["Camera", "Status", "FPS", "Frames"])
        self.camera_tree.setMaximumHeight(120)
        layout.addWidget(self.camera_tree)
        
        # Event counts
        layout.addWidget(QLabel("Event Counts (Today):"))
        self.event_counts = QTextEdit()
        self.event_counts.setReadOnly(True)
        self.event_counts.setMaximumHeight(150)
        layout.addWidget(self.event_counts)

        # Real-time match info
        match_layout = QHBoxLayout()
        self.camera_people_label = QLabel("People by Camera: -")
        self.cross_camera_label = QLabel("Same Person Across Cameras: -")
        match_layout.addWidget(self.camera_people_label)
        match_layout.addWidget(self.cross_camera_label)
        layout.addLayout(match_layout)
        self.realtime_zone_label = QLabel("Realtime Zones: Chair=0 | Wait=0 | Wash=0")
        layout.addWidget(self.realtime_zone_label)

        # Snapshot list (click to open)
        layout.addWidget(QLabel("Recent Snapshots:"))
        self.snapshot_list = QListWidget()
        self.snapshot_list.setMaximumHeight(140)
        self.snapshot_list.itemDoubleClicked.connect(self.open_snapshot_item)
        layout.addWidget(self.snapshot_list)

        # Split lower dashboard into left/right panes.
        lower_splitter = QSplitter(Qt.Horizontal)

        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Runtime Resources:"))
        self.resources_text = QTextEdit()
        self.resources_text.setReadOnly(True)
        self.resources_text.setMaximumHeight(150)
        self.resources_text.setPlaceholderText("Runtime resources will appear here...")
        left_layout.addWidget(self.resources_text)

        left_layout.addWidget(QLabel("Effective Config / Tuning:"))
        self.config_text = QTextEdit()
        self.config_text.setReadOnly(True)
        self.config_text.setMaximumHeight(150)
        self.config_text.setPlaceholderText("Effective runtime tuning/config will appear here...")
        left_layout.addWidget(self.config_text)
        left_panel.setLayout(left_layout)

        right_panel = QWidget()
        right_layout = QVBoxLayout()
        live_log_header = QHBoxLayout()
        live_log_header.addWidget(QLabel("Live Runtime Log (Tail):"))
        live_log_header.addStretch()
        clear_live_btn = QPushButton("Clear Live Log")
        clear_live_btn.clicked.connect(self.clear_live_log_view)
        live_log_header.addWidget(clear_live_btn)
        right_layout.addLayout(live_log_header)

        self.live_log_text = QTextEdit()
        self.live_log_text.setReadOnly(True)
        self.live_log_text.setMaximumHeight(310)
        self.live_log_text.setPlaceholderText("Waiting for runtime log output...")
        right_layout.addWidget(self.live_log_text)
        right_panel.setLayout(right_layout)

        lower_splitter.addWidget(left_panel)
        lower_splitter.addWidget(right_panel)
        lower_splitter.setStretchFactor(0, 1)
        lower_splitter.setStretchFactor(1, 2)
        layout.addWidget(lower_splitter)

        # Runtime tuning controls
        tuning_box = QHBoxLayout()
        tuning_box.addWidget(QLabel("YOLO Conf"))
        self.tune_conf = QDoubleSpinBox()
        self.tune_conf.setRange(0.05, 0.95)
        self.tune_conf.setSingleStep(0.01)
        self.tune_conf.setValue(float(self.config.get("yolo", {}).get("conf", 0.35)))
        tuning_box.addWidget(self.tune_conf)

        tuning_box.addWidget(QLabel("YOLO IoU"))
        self.tune_iou = QDoubleSpinBox()
        self.tune_iou.setRange(0.05, 0.95)
        self.tune_iou.setSingleStep(0.01)
        self.tune_iou.setValue(float(self.config.get("yolo", {}).get("iou", 0.5)))
        tuning_box.addWidget(self.tune_iou)

        tuning_box.addWidget(QLabel("Seat Dwell (sec)"))
        self.tune_sit_min = QSpinBox()
        self.tune_sit_min.setRange(1, 1800)
        self.tune_sit_min.setValue(int(self.config.get("runtime", {}).get("sit_min_sec", 10)))
        tuning_box.addWidget(self.tune_sit_min)

        tuning_box.addWidget(QLabel("Vacant Grace (sec)"))
        self.tune_vacant = QSpinBox()
        self.tune_vacant.setRange(1, 120)
        self.tune_vacant.setValue(int(self.config.get("runtime", {}).get("vacant_grace_sec", 6)))
        tuning_box.addWidget(self.tune_vacant)

        tuning_box.addWidget(QLabel("Zone Point"))
        self.tune_zone_point = QComboBox()
        self.tune_zone_point.addItems(["foot", "center"])
        self.tune_zone_point.setCurrentText(str(self.config.get("runtime", {}).get("zone_point_mode", "foot")))
        tuning_box.addWidget(self.tune_zone_point)

        apply_tuning_btn = QPushButton("Apply Tuning")
        apply_tuning_btn.clicked.connect(self.apply_runtime_tuning)
        tuning_box.addWidget(apply_tuning_btn)
        layout.addLayout(tuning_box)
        
        # Auto-refresh indicator
        self.auto_refresh_label = QLabel("🔄 Auto-updating...")
        self.auto_refresh_label.setStyleSheet("color: green; font-weight: bold;")
        layout.addWidget(self.auto_refresh_label)
        
        # Manual refresh button
        refresh_btn = QPushButton("🔄 Manual Refresh")
        refresh_btn.clicked.connect(self.refresh_dashboard)
        layout.addWidget(refresh_btn)

        clear_btn = QPushButton("🧹 Clear Event Counts")
        clear_btn.clicked.connect(self.clear_event_counts)
        layout.addWidget(clear_btn)
        
        layout.addStretch()
        
        widget.setLayout(layout)
        self.tabs.addTab(widget, "Dashboard")
        
        # Store for update reference
        self.dashboard_widget = widget

    def _find_latest_runtime_log(self) -> Optional[Path]:
        logs_dir = self._project_root() / self.config.get("paths", {}).get("logs", "logs")
        if not logs_dir.exists():
            return None
        candidates = sorted(logs_dir.glob("runtime_*.log"))
        if not candidates:
            fallback = logs_dir / "runtime.log"
            return fallback if fallback.exists() else None
        return candidates[-1]

    def _read_log_tail_lines(self, path: Path, max_lines: int = 250, max_bytes: int = 262144) -> list[str]:
        """Read only the latest part of a potentially large log file."""
        try:
            size = path.stat().st_size
            with open(path, "rb") as f:
                if size > max_bytes:
                    f.seek(size - max_bytes)
                data = f.read()
            text = data.decode("utf-8", errors="replace")
            return text.splitlines()[-max_lines:]
        except Exception:
            return []

    def _update_live_log_tail(self):
        """Update dashboard live log area from runtime log file."""
        try:
            current = self._find_latest_runtime_log()
            if current is None:
                return
            if self._live_log_file != current:
                self._live_log_file = current
                self._live_log_cache = self._read_log_tail_lines(current, max_lines=250)
                self._live_log_pos = current.stat().st_size if current.exists() else 0
                self.live_log_text.setPlainText("\n".join(self._live_log_cache))
                sb = self.live_log_text.verticalScrollBar()
                sb.setValue(sb.maximum())

            with open(current, "r", encoding="utf-8", errors="replace") as f:
                f.seek(self._live_log_pos)
                chunk = f.read()
                self._live_log_pos = f.tell()

            if chunk:
                lines = chunk.splitlines()
                self._live_log_cache.extend(lines)
                self._live_log_cache = self._live_log_cache[-250:]
                self.live_log_text.setPlainText("\n".join(self._live_log_cache))
                sb = self.live_log_text.verticalScrollBar()
                sb.setValue(sb.maximum())
        except Exception as e:
            logger.debug(f"Live log tail update failed: {e}")

    def clear_live_log_view(self):
        """Clear dashboard live log view without deleting log files."""
        try:
            self._live_log_cache = []
            if self._live_log_file and self._live_log_file.exists():
                self._live_log_pos = self._live_log_file.stat().st_size
            self.live_log_text.clear()
            self.statusBar().showMessage("Live log view cleared", 1500)
        except Exception as e:
            logger.debug(f"Clear live log view failed: {e}")

    def _render_resources(self, status: Dict[str, Any]):
        """Render runtime resource usage and health status."""
        lines = []
        health = status.get("health", {}) if isinstance(status.get("health", {}), dict) else {}
        resources = health.get("resources", {}) if isinstance(health.get("resources", {}), dict) else {}
        checks = health.get("health_checks", {}) if isinstance(health.get("health_checks", {}), dict) else {}
        watchdog = health.get("watchdog", {}) if isinstance(health.get("watchdog", {}), dict) else {}

        if resources:
            lines.append("Runtime Resources:")
            for k in sorted(resources.keys()):
                v = resources.get(k)
                lines.append(f"  {k}: {v}")
        else:
            # Fallback to local process/system metrics if runtime did not send resource payload.
            try:
                import psutil
                vm = psutil.virtual_memory()
                lines.append("Local Resources (controller host):")
                lines.append(f"  cpu_percent: {psutil.cpu_percent(interval=0.0):.1f}")
                lines.append(f"  memory_percent: {vm.percent:.1f}")
                lines.append(f"  memory_used_gb: {vm.used / (1024**3):.2f}")
            except Exception:
                lines.append("Resources: N/A")

        if checks:
            lines.append("")
            lines.append("Health Checks:")
            for name, item in checks.items():
                lines.append(f"  {name}: {item}")

        if watchdog:
            lines.append("")
            lines.append("Watchdog:")
            for k, v in watchdog.items():
                if k == "cameras":
                    continue
                lines.append(f"  {k}: {v}")

        self.resources_text.setPlainText("\n".join(lines) if lines else "No resource data")

    def _render_effective_config(self, status: Dict[str, Any]):
        """Render runtime tuning/config currently effective."""
        tuning = status.get("tuning", {}) if isinstance(status.get("tuning", {}), dict) else {}
        cameras = status.get("cameras", {}) if isinstance(status.get("cameras", {}), dict) else {}
        lines = [
            f"branch: {status.get('branch', '-')}",
            f"running: {status.get('running', False)}",
            f"camera_count: {len(cameras)}",
        ]
        if tuning:
            lines.append("")
            lines.append("tuning:")
            for k in sorted(tuning.keys()):
                lines.append(f"  {k}: {tuning.get(k)}")
        else:
            lines.append("")
            lines.append("tuning: (no runtime tuning payload)")
        self.config_text.setPlainText("\n".join(lines))
    
    def tab_live_view(self):
        """Live Viewer tab to see camera feed with zone overlays."""
        widget = LiveViewerWidget(self, config=self.config.data)
        self.tabs.addTab(widget, "🎥 Live View")

    def tab_setup(self):
        """Setup tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        info = QLabel(
            "Setup Wizard\n\n"
            "Follow the guided steps to configure:\n"
            "1. Supabase connection\n"
            "2. Camera RTSP URLs\n"
            "3. Zone definitions\n"
            "4. Staff database\n"
            "5. System diagnostics"
        )
        layout.addWidget(info)
        
        wizard_btn = QPushButton("✧ Launch Setup Wizard")
        wizard_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 10px;")
        wizard_btn.clicked.connect(self.launch_wizard)
        layout.addWidget(wizard_btn)
        
        layout.addStretch()
        
        widget.setLayout(layout)
        self.tabs.addTab(widget, "Setup Wizard")
    
    def tab_cameras(self):
        """Cameras tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Camera Management")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Camera list
        layout.addWidget(QLabel("Configured Cameras:"))
        self.camera_list = QTreeWidget()
        self.camera_list.setHeaderLabels(["Camera", "RTSP URL", "Enabled", "Zones"])
        # setColumnWidths not available on QTreeWidget; set each column width
        col_widths = [150, 300, 80, 200]
        for i, w in enumerate(col_widths):
            self.camera_list.setColumnWidth(i, w)
        
        cameras = self.config.get("cameras", {})
        for cam_name, cam_config in cameras.items():
            item = QTreeWidgetItem([
                cam_name,
                cam_config.get("rtsp_url", "")[:40],
                "✓" if cam_config.get("enabled", True) else "✗",
                cam_config.get("zones_file", "")
            ])
            self.camera_list.addTopLevelItem(item)
        
        layout.addWidget(self.camera_list)
        
        # Actions
        actions_layout = QHBoxLayout()
        
        add_btn = QPushButton("+ Add Camera")
        add_btn.clicked.connect(self.camera_manager.add_camera_dialog)
        actions_layout.addWidget(add_btn)
        
        edit_btn = QPushButton("✎ Edit Camera")
        edit_btn.clicked.connect(self.camera_manager.edit_camera_dialog)
        actions_layout.addWidget(edit_btn)
        
        delete_btn = QPushButton("🗑 Delete Camera")
        delete_btn.clicked.connect(self.camera_manager.delete_camera)
        actions_layout.addWidget(delete_btn)
        
        test_btn = QPushButton("⚡ Test Selected")
        test_btn.clicked.connect(self.camera_manager.test_camera)
        actions_layout.addWidget(test_btn)
        
        test_all_btn = QPushButton("⚡ Test All")
        test_all_btn.clicked.connect(self.camera_manager.test_all_cameras)
        actions_layout.addWidget(test_all_btn)
        
        layout.addLayout(actions_layout)
        
        # Import/Export
        import_export_layout = QHBoxLayout()
        
        import_btn = QPushButton("📥 Import JSON")
        import_btn.clicked.connect(self.camera_manager.import_cameras_json)
        import_export_layout.addWidget(import_btn)
        
        export_btn = QPushButton("📤 Export JSON")
        export_btn.clicked.connect(self.camera_manager.export_cameras_json)
        import_export_layout.addWidget(export_btn)
        
        import_export_layout.addStretch()
        layout.addLayout(import_export_layout)
        
        layout.addStretch()
        
        widget.setLayout(layout)
        self.tabs.addTab(widget, "Cameras")

    def tab_staff_db(self):
        """Staff DB Builder tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        title = QLabel("Staff DB Builder")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        self.staff_builder = StaffBuilderWidget(self, config=self.config)
        layout.addWidget(self.staff_builder)

        widget.setLayout(layout)
        self.tabs.addTab(widget, "Staff DB")

    def tab_zones(self):
        """Zones tab - Zone Editor"""
        try:
            widget = QWidget()
            layout = QVBoxLayout()
            self.zone_editor = ZoneEditorWidget(self)
            layout.addWidget(self.zone_editor)
            widget.setLayout(layout)
            self.tabs.addTab(widget, "Zones")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load Zone Editor: {e}")
            logger.exception("Zone Editor initialization error")
    
    def tab_diagnostics(self):
        """Diagnostics tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        # Diagnostics results
        self.diagnostics_output = QTextEdit()
        self.diagnostics_output.setReadOnly(True)
        layout.addWidget(self.diagnostics_output)

        # Run button
        run_btn = QPushButton("🔍 Run Diagnostics")
        run_btn.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold; padding: 10px;")
        run_btn.clicked.connect(self.run_diagnostics)
        layout.addWidget(run_btn)
        
        widget.setLayout(layout)
        self.tabs.addTab(widget, "Diagnostics")
    
    def tab_logs(self):
        """Logs tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        # Logs display
        self.logs_display = QTextEdit()
        self.logs_display.setReadOnly(True)
        layout.addWidget(self.logs_display)

        # Filters + actions
        fl = QHBoxLayout()
        fl.addWidget(QLabel("Level:"))
        from PySide6.QtWidgets import QComboBox
        self.log_level = QComboBox()
        self.log_level.addItems(["ALL", "DEBUG", "INFO", "WARN", "ERROR"])
        fl.addWidget(self.log_level)

        fl.addWidget(QLabel("Camera:"))
        self.log_camera = QComboBox()
        self.log_camera.addItem("ALL")
        # populate cameras
        cams = list(self.config.get('cameras', {}).keys())
        for c in cams:
            self.log_camera.addItem(c)
        fl.addWidget(self.log_camera)

        refresh_btn = QPushButton("Refresh Logs")
        refresh_btn.clicked.connect(self.refresh_logs)
        fl.addWidget(refresh_btn)

        export_btn = QPushButton("Export Logs")
        export_btn.clicked.connect(self.export_logs)
        fl.addWidget(export_btn)

        open_btn = QPushButton("Open Logs Folder")
        open_btn.clicked.connect(self.open_logs_folder)
        fl.addWidget(open_btn)

        layout.addLayout(fl)
        
        widget.setLayout(layout)
        self.tabs.addTab(widget, "Logs")

    def tab_model_train(self):
        """Model Train tab for YOLO custom classes."""
        widget = QWidget()
        layout = QVBoxLayout()

        title = QLabel("YOLO Training")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title.setFont(title_font)
        layout.addWidget(title)

        form = QFormLayout()
        self.train_data_yaml = QLineEdit("data/yolo_head_dataset/dataset.yaml")
        self.train_model_base = QLineEdit("yolov8n.pt")
        browse_train_data_btn = QPushButton("Browse")
        browse_train_data_btn.clicked.connect(
            lambda: self._browse_line_edit_file(self.train_data_yaml, "YAML files (*.yaml *.yml);;All files (*)")
        )
        browse_train_model_btn = QPushButton("Browse")
        browse_train_model_btn.clicked.connect(
            lambda: self._browse_line_edit_file(self.train_model_base, "PyTorch model (*.pt);;All files (*)")
        )
        self.train_epochs = QSpinBox()
        self.train_epochs.setRange(1, 2000)
        self.train_epochs.setValue(100)
        self.train_imgsz = QSpinBox()
        self.train_imgsz.setRange(64, 4096)
        self.train_imgsz.setValue(640)
        self.train_batch = QSpinBox()
        self.train_batch.setRange(1, 256)
        self.train_batch.setValue(16)
        self.train_device = QComboBox()
        self.train_device.addItems(["auto", "cpu", "mps", "cuda"])
        self.train_device.setCurrentText("cpu")

        train_data_row = QWidget()
        train_data_row_l = QHBoxLayout()
        train_data_row_l.setContentsMargins(0, 0, 0, 0)
        train_data_row_l.addWidget(self.train_data_yaml)
        train_data_row_l.addWidget(browse_train_data_btn)
        train_data_row.setLayout(train_data_row_l)
        form.addRow("Dataset YAML", train_data_row)

        train_model_row = QWidget()
        train_model_row_l = QHBoxLayout()
        train_model_row_l.setContentsMargins(0, 0, 0, 0)
        train_model_row_l.addWidget(self.train_model_base)
        train_model_row_l.addWidget(browse_train_model_btn)
        train_model_row.setLayout(train_model_row_l)
        form.addRow("Base Model", train_model_row)
        form.addRow("Epochs", self.train_epochs)
        form.addRow("Image Size", self.train_imgsz)
        form.addRow("Batch", self.train_batch)
        form.addRow("Device", self.train_device)
        layout.addLayout(form)

        actions = QHBoxLayout()
        btn_validate = QPushButton("Validate Labels")
        btn_validate.clicked.connect(self.run_label_validation)
        actions.addWidget(btn_validate)

        btn_train = QPushButton("Start Training")
        btn_train.setStyleSheet("background-color: #1B5E20; color: white; font-weight: bold; padding: 8px;")
        btn_train.clicked.connect(self.run_model_training)
        actions.addWidget(btn_train)
        actions.addStretch()
        layout.addLayout(actions)

        self.train_output = QTextEdit()
        self.train_output.setReadOnly(True)
        self.train_output.setPlaceholderText("Training output will appear here...")
        layout.addWidget(self.train_output)

        widget.setLayout(layout)
        self.tabs.addTab(widget, "Model Train")

    def tab_model_test(self):
        """Model Test/Evaluate tab with percent metrics."""
        widget = QWidget()
        layout = QVBoxLayout()

        title = QLabel("YOLO Evaluation")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title.setFont(title_font)
        layout.addWidget(title)

        form = QFormLayout()
        self.eval_weights = QLineEdit("runs/train/yolo_head_staff/weights/best.pt")
        self.eval_data_yaml = QLineEdit("data/yolo_head_dataset/dataset.yaml")
        browse_eval_weights_btn = QPushButton("Browse")
        browse_eval_weights_btn.clicked.connect(
            lambda: self._browse_line_edit_file(self.eval_weights, "PyTorch model (*.pt);;All files (*)")
        )
        browse_eval_data_btn = QPushButton("Browse")
        browse_eval_data_btn.clicked.connect(
            lambda: self._browse_line_edit_file(self.eval_data_yaml, "YAML files (*.yaml *.yml);;All files (*)")
        )
        self.eval_imgsz = QSpinBox()
        self.eval_imgsz.setRange(64, 4096)
        self.eval_imgsz.setValue(640)
        self.eval_batch = QSpinBox()
        self.eval_batch.setRange(1, 256)
        self.eval_batch.setValue(16)
        self.eval_device = QComboBox()
        self.eval_device.addItems(["auto", "cpu", "mps", "cuda"])
        self.eval_device.setCurrentText("cpu")

        eval_weights_row = QWidget()
        eval_weights_row_l = QHBoxLayout()
        eval_weights_row_l.setContentsMargins(0, 0, 0, 0)
        eval_weights_row_l.addWidget(self.eval_weights)
        eval_weights_row_l.addWidget(browse_eval_weights_btn)
        eval_weights_row.setLayout(eval_weights_row_l)
        form.addRow("Weights", eval_weights_row)

        eval_data_row = QWidget()
        eval_data_row_l = QHBoxLayout()
        eval_data_row_l.setContentsMargins(0, 0, 0, 0)
        eval_data_row_l.addWidget(self.eval_data_yaml)
        eval_data_row_l.addWidget(browse_eval_data_btn)
        eval_data_row.setLayout(eval_data_row_l)
        form.addRow("Dataset YAML", eval_data_row)
        form.addRow("Image Size", self.eval_imgsz)
        form.addRow("Batch", self.eval_batch)
        form.addRow("Device", self.eval_device)
        layout.addLayout(form)

        eval_btn_row = QHBoxLayout()
        run_eval_btn = QPushButton("Run Evaluation")
        run_eval_btn.setStyleSheet("background-color: #0D47A1; color: white; font-weight: bold; padding: 8px;")
        run_eval_btn.clicked.connect(self.run_model_evaluation)
        eval_btn_row.addWidget(run_eval_btn)
        eval_btn_row.addStretch()
        layout.addLayout(eval_btn_row)

        metrics_box = QGroupBox("Metrics (%)")
        metrics_layout = QFormLayout()
        self.metric_precision = QLabel("0.00%")
        self.metric_recall = QLabel("0.00%")
        self.metric_map50 = QLabel("0.00%")
        self.metric_map = QLabel("0.00%")
        self.metric_bar_precision = QProgressBar()
        self.metric_bar_recall = QProgressBar()
        self.metric_bar_map50 = QProgressBar()
        self.metric_bar_map = QProgressBar()
        for bar in (
            self.metric_bar_precision,
            self.metric_bar_recall,
            self.metric_bar_map50,
            self.metric_bar_map,
        ):
            bar.setRange(0, 100)
            bar.setValue(0)

        metrics_layout.addRow("Precision", self._metric_row_widget(self.metric_precision, self.metric_bar_precision))
        metrics_layout.addRow("Recall", self._metric_row_widget(self.metric_recall, self.metric_bar_recall))
        metrics_layout.addRow("mAP@50", self._metric_row_widget(self.metric_map50, self.metric_bar_map50))
        metrics_layout.addRow("mAP@50:95", self._metric_row_widget(self.metric_map, self.metric_bar_map))
        metrics_box.setLayout(metrics_layout)
        layout.addWidget(metrics_box)

        self.eval_output = QTextEdit()
        self.eval_output.setReadOnly(True)
        self.eval_output.setPlaceholderText("Evaluation output will appear here...")
        layout.addWidget(self.eval_output)

        widget.setLayout(layout)
        self.tabs.addTab(widget, "Model Test")

    def _metric_row_widget(self, label: QLabel, bar: QProgressBar) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout()
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(label)
        lay.addWidget(bar)
        w.setLayout(lay)
        return w

    def _browse_line_edit_file(self, line_edit: QLineEdit, filters: str):
        path, _ = QFileDialog.getOpenFileName(self, "Select file", line_edit.text().strip() or "", filters)
        if path:
            line_edit.setText(path)

    def tab_dataset_lab(self):
        """Dataset workflow tab: import -> label/correct -> evaluate."""
        widget = QWidget()
        root_layout = QVBoxLayout()

        title = QLabel("Dataset Lab (Import / Auto-detect / Human Correct / Save)")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title.setFont(title_font)
        root_layout.addWidget(title)

        split = QSplitter(Qt.Horizontal)

        # Left panel: import and manual category selection
        left = QWidget()
        left_l = QVBoxLayout()
        left_form = QFormLayout()
        self.lab_dataset_yaml = QLineEdit("data/yolo_head_dataset/dataset.yaml")
        lab_data_browse = QPushButton("Browse")
        lab_data_browse.clicked.connect(
            lambda: self._browse_line_edit_file(self.lab_dataset_yaml, "YAML files (*.yaml *.yml);;All files (*)")
        )
        data_row = QWidget()
        data_row_l = QHBoxLayout()
        data_row_l.setContentsMargins(0, 0, 0, 0)
        data_row_l.addWidget(self.lab_dataset_yaml)
        data_row_l.addWidget(lab_data_browse)
        data_row.setLayout(data_row_l)
        left_form.addRow("Dataset YAML", data_row)
        left_l.addLayout(left_form)

        import_row = QHBoxLayout()
        add_images_btn = QPushButton("Add Images")
        add_images_btn.clicked.connect(self.lab_add_images)
        import_row.addWidget(add_images_btn)
        remove_images_btn = QPushButton("Remove Selected")
        remove_images_btn.clicked.connect(self.lab_remove_selected_images)
        import_row.addWidget(remove_images_btn)
        clear_images_btn = QPushButton("Clear")
        clear_images_btn.clicked.connect(self.lab_clear_images)
        import_row.addWidget(clear_images_btn)
        left_l.addLayout(import_row)

        self.lab_image_list = QListWidget()
        self.lab_image_list.currentItemChanged.connect(self.lab_on_image_changed)
        left_l.addWidget(self.lab_image_list)

        controls = QHBoxLayout()
        self.lab_split_combo = QComboBox()
        self.lab_split_combo.addItems(["train", "val"])
        controls.addWidget(QLabel("Split"))
        controls.addWidget(self.lab_split_combo)
        self.lab_manual_class_combo = QComboBox()
        self.lab_manual_class_combo.addItems(self.yolo_class_names)
        controls.addWidget(QLabel("Manual Class"))
        controls.addWidget(self.lab_manual_class_combo)
        apply_meta_btn = QPushButton("Apply to Selected")
        apply_meta_btn.clicked.connect(self.lab_apply_meta_to_selected)
        controls.addWidget(apply_meta_btn)
        left_l.addLayout(controls)

        split_row = QHBoxLayout()
        split_row.addWidget(QLabel("Auto Split Val %"))
        self.lab_val_percent = QSpinBox()
        self.lab_val_percent.setRange(5, 50)
        self.lab_val_percent.setValue(20)
        split_row.addWidget(self.lab_val_percent)
        auto_split_btn = QPushButton("Auto Split train/val")
        auto_split_btn.clicked.connect(self.lab_auto_split_items)
        split_row.addWidget(auto_split_btn)
        split_row.addStretch()
        left_l.addLayout(split_row)

        self.lab_preview = QLabel("Image preview")
        self.lab_preview.setAlignment(Qt.AlignCenter)
        self.lab_preview.setMinimumHeight(280)
        self.lab_preview.setStyleSheet("border: 1px solid #666;")
        left_l.addWidget(self.lab_preview)
        left.setLayout(left_l)

        # Right panel: evaluate + auto-detect + correction
        right = QWidget()
        right_l = QVBoxLayout()
        right_top = QFormLayout()
        self.lab_weights = QLineEdit("runs/train/yolo_head_staff/weights/best.pt")
        lab_weights_browse = QPushButton("Browse")
        lab_weights_browse.clicked.connect(
            lambda: self._browse_line_edit_file(self.lab_weights, "PyTorch model (*.pt);;All files (*)")
        )
        weights_row = QWidget()
        weights_row_l = QHBoxLayout()
        weights_row_l.setContentsMargins(0, 0, 0, 0)
        weights_row_l.addWidget(self.lab_weights)
        weights_row_l.addWidget(lab_weights_browse)
        weights_row.setLayout(weights_row_l)
        right_top.addRow("Weights", weights_row)
        self.lab_eval_device = QComboBox()
        self.lab_eval_device.addItems(["auto", "cpu", "mps", "cuda"])
        self.lab_eval_device.setCurrentText("cpu")
        right_top.addRow("Eval Device", self.lab_eval_device)
        right_l.addLayout(right_top)

        eval_row = QHBoxLayout()
        eval_btn = QPushButton("Evaluate Dataset")
        eval_btn.clicked.connect(self.lab_run_evaluation)
        eval_row.addWidget(eval_btn)
        detect_btn = QPushButton("Auto Detect Selected Image")
        detect_btn.clicked.connect(self.lab_run_detect_selected)
        eval_row.addWidget(detect_btn)
        save_btn = QPushButton("Save Corrected -> Dataset")
        save_btn.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold;")
        save_btn.clicked.connect(self.lab_save_corrected_to_dataset)
        eval_row.addWidget(save_btn)
        save_all_btn = QPushButton("Save All Imported -> Dataset")
        save_all_btn.setStyleSheet("background-color: #00695C; color: white; font-weight: bold;")
        save_all_btn.clicked.connect(self.lab_save_all_imported_to_dataset)
        eval_row.addWidget(save_all_btn)
        right_l.addLayout(eval_row)

        metric_row = QHBoxLayout()
        self.lab_metric_precision = QLabel("P: 0.00%")
        self.lab_metric_recall = QLabel("R: 0.00%")
        self.lab_metric_map50 = QLabel("mAP50: 0.00%")
        self.lab_metric_map = QLabel("mAP50-95: 0.00%")
        metric_row.addWidget(self.lab_metric_precision)
        metric_row.addWidget(self.lab_metric_recall)
        metric_row.addWidget(self.lab_metric_map50)
        metric_row.addWidget(self.lab_metric_map)
        metric_row.addStretch()
        right_l.addLayout(metric_row)

        summary_row = QHBoxLayout()
        summary_row.addWidget(QLabel("Dataset Summary"))
        summary_refresh_btn = QPushButton("Refresh Summary")
        summary_refresh_btn.clicked.connect(self.lab_refresh_dataset_summary)
        summary_row.addWidget(summary_refresh_btn)
        summary_row.addStretch()
        right_l.addLayout(summary_row)

        self.lab_summary_text = QTextEdit()
        self.lab_summary_text.setReadOnly(True)
        self.lab_summary_text.setMaximumHeight(150)
        right_l.addWidget(self.lab_summary_text)

        right_l.addWidget(QLabel("Detections (edit class before save)"))
        self.lab_det_table = QTableWidget(0, 8)
        self.lab_det_table.setHorizontalHeaderLabels(["idx", "class", "conf", "x1", "y1", "x2", "y2", "correct_class"])
        self.lab_det_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        right_l.addWidget(self.lab_det_table)

        self.lab_log = QTextEdit()
        self.lab_log.setReadOnly(True)
        self.lab_log.setPlaceholderText("Dataset lab output")
        right_l.addWidget(self.lab_log)
        right.setLayout(right_l)

        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)

        root_layout.addWidget(split)
        widget.setLayout(root_layout)
        self.tabs.addTab(widget, "Dataset Lab")
        self.lab_refresh_dataset_summary()
    
    # =========================
    # HANDLERS
    # =========================
    
    def launch_wizard(self):
        """Launch setup wizard"""
        # SetupWizard is a QMainWindow (not QDialog) — show it modelessly
        self.wizard_window = SetupWizard()
        self.wizard_window.show()

    def _project_root(self) -> Path:
        """Resolve project/app root for source and frozen builds."""
        if getattr(sys, "frozen", False):
            return Path(sys.executable).resolve().parent
        return Path(__file__).resolve().parent.parent

    def _recorder_command(self) -> list[str]:
        """Return subprocess command to launch recorder."""
        root = self._project_root()
        if getattr(sys, "frozen", False):
             # For frozen builds, similar logic to runtime_command (omitted for brevity)
             # assuming same executable or explicit wrapper
             pass
        return [sys.executable, "-u", str(root / "runtime" / "recorder.py")]

    def _processor_command(self) -> list[str]:
        """Return subprocess command to launch processor."""
        root = self._project_root()
        return [sys.executable, "-u", str(root / "runtime" / "processor.py")]
    
    def start_service(self):
        """Start runtime service"""
        logger.info("Starting runtime service...")

        try:
            root = self._project_root()
            
            # Use unbuffered runtime output and persist stdout/stderr to file.
            self.runtime_stdout_file = root / self.config.get("paths", {}).get("logs", "logs") / f"runtime_{datetime.now().strftime('%Y-%m-%d')}.log"
            self.runtime_stdout_file.parent.mkdir(parents=True, exist_ok=True)
            self.runtime_output_handle = open(self.runtime_stdout_file, "a", encoding="utf-8")
            self.runtime_output_handle.write(f"\n[{datetime.now().isoformat()}] [controller] START_SERVICE (Store-and-Forward)\n")
            self.runtime_output_handle.flush()
            
            # Start Recorder
            self.recorder_process = subprocess.Popen(
                self._recorder_command(),
                stdout=self.runtime_output_handle,
                stderr=subprocess.STDOUT,
                cwd=str(root)
            )
            
            # Start Processor
            self.processor_process = subprocess.Popen(
                self._processor_command(),
                stdout=self.runtime_output_handle,
                stderr=subprocess.STDOUT,
                cwd=str(root)
            )
            
            self.is_running = True
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            
            self.status_runtime.setText("🟢 Running")
            self.status_runtime.setStyleSheet("color: green;")
            self.statusBar().showMessage("Service started")
            
            # Initialize in-process dashboard client (optional, cross-process uses dashboard_state_file)
            if self.broadcaster:
                self.dashboard_client = GUIDashboardClient("controller", self.broadcaster)
                self.dashboard_client.status_updated.connect(self.on_status_updated)
                self.dashboard_client.summary_updated.connect(self.on_summary_updated)
                self.dashboard_client.event_received.connect(self.on_event_received)
                self.dashboard_client.connection_status.connect(self.on_connection_changed)
                self.dashboard_client.start()
                
                logger.info("Dashboard client connected")
            
            # Give service time to start
            self.status_check_timer.stop()
            self.status_check_timer.start(2000)
        
        except Exception as e:
            logger.error(f"Error starting service: {e}")
            QMessageBox.critical(self, "Error", f"Cannot start service: {e}")
            self.is_running = False
    
    def stop_service(self):
        """Stop runtime service"""
        logger.info("Stopping runtime service...")
        
        # Stop dashboard client first
        if self.dashboard_client:
            self.dashboard_client.stop()
            self.dashboard_client = None
        
        if self.recorder_process:
            self.recorder_process.terminate()
            try:
                # Give recorder enough time to shut down ffmpeg children gracefully.
                # Recorder waits up to 5s for ffmpeg, so we should wait at least 7-10s.
                self.recorder_process.wait(timeout=10)
            except:
                logger.warning("Recorder process timed out, killing...")
                self.recorder_process.kill()
            self.recorder_process = None
            
        if self.processor_process:
            self.processor_process.terminate()
            try:
                self.processor_process.wait(timeout=5)
            except:
                self.processor_process.kill()
            self.processor_process = None
        if self.runtime_output_handle:
            try:
                self.runtime_output_handle.write(f"[{datetime.now().isoformat()}] [controller] STOP_SERVICE\n")
                self.runtime_output_handle.flush()
                self.runtime_output_handle.close()
            except Exception:
                pass
            self.runtime_output_handle = None
        
        self.is_running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        self.status_runtime.setText("⚫ Stopped")
        self.status_runtime.setStyleSheet("color: red;")
        self.statusBar().showMessage("Service stopped")
        
        # Clear dashboard
        self.camera_tree.clear()
        self.event_counts.setText("")
        self.status_active.setText("0")
        self.status_heartbeat.setText("Never")
        self.camera_people_label.setText("People by Camera: -")
        self.realtime_zone_label.setText("Realtime Zones: Chair=0 | Wait=0 | Wash=0")
        self.cross_camera_label.setText("Same Person Across Cameras: -")
        self.snapshot_list.clear()
        self.last_runtime_state_ts = 0.0
        try:
            if self.dashboard_state_file.exists():
                self.dashboard_state_file.unlink()
        except Exception:
            pass
    
    def monitor_service(self):
        """Monitor service status"""
        # TODO: Implement proper monitoring
        pass
    
    def on_status_updated(self, status: Dict[str, Any]):
        """Handle status update from dashboard"""
        try:
            logger.debug(f"Status update received: {status}")
            
            # Update runtime status
            running = status.get("running", False)
            self.status_runtime.setText("🟢 Running" if running else "⚫ Stopped")
            self.status_runtime.setStyleSheet("color: green;" if running else "color: red;")
            
            # Update timestamp
            self.status_heartbeat.setText(datetime.now().strftime("%H:%M:%S"))
            
            # Update active tracks only if provided by runtime; otherwise keep cached value
            if "active_tracks" in status:
                active_tracks = status.get("active_tracks", 0)
                self.cached_active_tracks = int(active_tracks)
                self.status_active.setText(str(self.cached_active_tracks))
            elif self.cached_active_tracks is not None:
                # keep previous displayed value
                self.status_active.setText(str(self.cached_active_tracks))
            
            # Update camera list
            cameras = status.get("cameras", {})
            all_cam_names = list(self.config.get("cameras", {}).keys())
            for cam in all_cam_names:
                cameras.setdefault(cam, {"connected": False, "fps": 0.0, "enabled": True})
            self.camera_tree.clear()
            
            cam_people_map = status.get("camera_people")
            for cam_name, cam_info in cameras.items():
                connected = cam_info.get("connected", False)
                fps = cam_info.get("fps", 0)
                enabled = cam_info.get("enabled", True)
                if cam_people_map is not None and cam_name in cam_people_map:
                    cam_people = int(cam_people_map.get(cam_name, 0))
                    self.cached_camera_people[cam_name] = cam_people
                else:
                    # fall back to cached per-camera value or 0 if unknown
                    cam_people = int(self.cached_camera_people.get(cam_name, 0))
                
                status_icon = "✓" if connected else "✗"
                status_color = "green" if connected else ("gray" if not enabled else "red")
                
                item = QTreeWidgetItem([
                    cam_name,
                    f"{status_icon}{'' if enabled else ' (disabled)'}",
                    f"{fps:.1f}",
                    str(cam_people)
                ])
                item.setForeground(1, QColor(status_color))
                self.camera_tree.addTopLevelItem(item)

            # People by camera and cross-camera same-person view
            cam_people_map = status.get("camera_people", {})
            if cam_people_map:
                text = ", ".join([f"{cam}:{cnt}" for cam, cnt in cam_people_map.items()])
            else:
                text = "-"
            self.camera_people_label.setText(f"People by Camera: {text}")

            same_map = status.get("same_person_multi_cam", {})
            if same_map:
                s = ", ".join([f"GID {gid}:{'/'.join(cams)}" for gid, cams in same_map.items()])
            else:
                s = "-"
            self.cross_camera_label.setText(f"Same Person Across Cameras: {s}")

            rt = status.get("realtime_counts", {}) or {}
            chair_total = int(rt.get("chairs_total", 0))
            wait_total = int(rt.get("waits_total", 0))
            wash_total = int(rt.get("washes_total", 0))
            chair_by_zone = rt.get("chairs_by_zone", {}) or {}
            wash_by_zone = rt.get("washes_by_zone", {}) or {}
            chair_text = ", ".join([f"{z}:{c}" for z, c in sorted(chair_by_zone.items())]) if chair_by_zone else "-"
            wash_text = ", ".join([f"{z}:{c}" for z, c in sorted(wash_by_zone.items())]) if wash_by_zone else "-"
            self.realtime_zone_label.setText(
                f"Realtime Zones: Chair={chair_total} ({chair_text}) | Wait={wait_total} | Wash={wash_total} ({wash_text})"
            )

            # Snapshot list
            self.update_snapshot_list(status.get("recent_snapshots", []))
            self._render_resources(status)
            self._render_effective_config(status)
            self._update_live_log_tail()
            
            # Update auto-refresh indicator
            if self.auto_refresh_label:
                self.auto_refresh_label.setText("🔄 Auto-updating...")
        
        except Exception as e:
            logger.error(f"Error handling status update: {e}")
    
    def on_summary_updated(self, summary: Dict[str, Any]):
        """Handle summary update from dashboard"""
        try:
            logger.debug(f"Summary update received: {summary}")
            
            # Format event counts
            haircuts = summary.get("haircuts", summary.get("haircut", 0))
            washes = summary.get("washes", summary.get("wash", 0))
            waits = summary.get("waits", summary.get("wait", 0))
            haircut_confirmed = summary.get("haircut_confirmed", haircuts)
            
            text = f"""Haircuts: {haircuts}
Washes: {washes}
Waiting: {waits}
Haircut Confirmed: {haircut_confirmed}

Updated: {datetime.now().strftime("%H:%M:%S")}"""
            
            self.event_counts.setText(text)
        
        except Exception as e:
            logger.error(f"Error handling summary update: {e}")
    
    def on_event_received(self, event: Dict[str, Any]):
        """Handle individual event from dashboard"""
        try:
            event_type = event.get("event_type", "unknown")
            camera = event.get("camera", "unknown")
            dwell = event.get("dwell_seconds", 0)
            
            logger.info(f"Event received: {event_type} on {camera} (dwell: {dwell}s)")
            
            # Flash status to indicate event
            if self.status_runtime.isVisible():
                # Already updated in status, just log
                pass
        
        except Exception as e:
            logger.error(f"Error handling event: {e}")
    
    def on_connection_changed(self, connected: bool):
        """Handle dashboard connection status change"""
        if connected:
            logger.info("Dashboard client connected")
            self.auto_refresh_label.setText("🟢 Live")
            self.auto_refresh_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            logger.warning("Dashboard client disconnected")
            self.auto_refresh_label.setText("⚠️ No connection")
            self.auto_refresh_label.setStyleSheet("color: orange; font-weight: bold;")

    
    def check_runtime_status(self):
        """Check if runtime is already running"""
        runtime_alive = False
        # 1) Process liveness check
        # 1) Process liveness check
        if self.recorder_process:
            rc = self.recorder_process.poll()
            if rc is not None:
                logger.error(f"Recorder process exited with code {rc}")
                self.recorder_process = None
        
        if self.processor_process:
            rc = self.processor_process.poll()
            if rc is not None:
                logger.error(f"Processor process exited with code {rc}")
                self.processor_process = None
        
        # If we expect to be running but both processes are dead, update UI
        if self.is_running and (self.recorder_process is None and self.processor_process is None):
            self.is_running = False
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.status_runtime.setText("⚫ Stopped")
            self.status_runtime.setStyleSheet("color: red;")
            self.auto_refresh_label.setText("⚠️ Services exited")
            self.auto_refresh_label.setStyleSheet("color: orange; font-weight: bold;")
            self.statusBar().showMessage("Services stopped unexpectedly")
        
        runtime_alive = (self.recorder_process is not None or self.processor_process is not None)

        # 2) Cross-process dashboard state polling
        if self.dashboard_state_file.exists():
            try:
                with open(self.dashboard_state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                ts = float(state.get("timestamp", 0.0))
                if ts and ts >= self.last_runtime_state_ts:
                    self.last_runtime_state_ts = ts
                    status = state.get("status", {})
                    summary = state.get("summary", {})
                    # Backward compatibility: some runtime builds wrote a flat
                    # dashboard_state payload without status/summary wrappers.
                    if not status and isinstance(state, dict):
                        status = {
                            "running": bool(runtime_alive),
                            "branch": self.config.get("branch_code", "-"),
                            "cameras": {},
                            "active_tracks": int(state.get("active_people", 0)),
                            "camera_people": {},
                            "same_person_multi_cam": {},
                            "realtime_counts": {},
                            "recent_snapshots": [],
                            "tuning": {},
                        }
                    if not summary and isinstance(state, dict):
                        summary = {
                            "active_people": int(state.get("active_people", status.get("active_tracks", 0) if isinstance(status, dict) else 0)),
                            "haircuts": int(state.get("haircut_count", 0)),
                            "washes": int(state.get("wash_count", 0)),
                            "waits": int(state.get("wait_count", 0)),
                            "total_events": int(state.get("haircut_events", 0)),
                        }
                    # Treat stale state as not live.
                    is_fresh = (time.time() - ts) <= 15
                    if not is_fresh:
                        status["running"] = False
                    if status:
                        if not runtime_alive:
                            status["running"] = False
                        self.on_status_updated(status)
                    if summary:
                        self.on_summary_updated(summary)
                    if is_fresh and status.get("running", False):
                        self.auto_refresh_label.setText("🟢 Live")
                        self.auto_refresh_label.setStyleSheet("color: green; font-weight: bold;")
                    else:
                        self.auto_refresh_label.setText("⚠️ No recent runtime state")
                        self.auto_refresh_label.setStyleSheet("color: orange; font-weight: bold;")
            except Exception as e:
                logger.warning(f"Failed to read dashboard state: {e}")
        # Keep live log tail updating even when status file does not change.
        self._update_live_log_tail()
    
    def refresh_dashboard(self):
        """Refresh dashboard status manually"""
        if self.dashboard_client:
            status = self.dashboard_client.get_last_status()
            if status:
                self.on_status_updated(status)
            
            summary = self.dashboard_client.get_last_summary()
            if summary:
                self.on_summary_updated(summary)

        # Also refresh from cross-process state file.
        self.check_runtime_status()
        self._update_live_log_tail()
        
        self.statusBar().showMessage("Dashboard refreshed", 2000)

    def clear_event_counts(self):
        """Reset event counters in runtime + reset dashboard values."""
        try:
            self.reset_counts_flag_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.reset_counts_flag_file, "w", encoding="utf-8") as f:
                f.write(datetime.now().isoformat())
            self.event_counts.setText("Haircuts: 0\nWashes: 0\nWaiting: 0\nHaircut Confirmed: 0\n\nUpdated: -")
            self.status_active.setText("0")
            self.camera_people_label.setText("People by Camera: -")
            self.cross_camera_label.setText("Same Person Across Cameras: -")
            self.realtime_zone_label.setText("Realtime Zones: Chair=0 | Wait=0 | Wash=0")
            self.snapshot_list.clear()
            self.statusBar().showMessage("Reset requested")
            logger.info("Requested runtime counter reset")
        except Exception as e:
            logger.error(f"Failed to reset counters: {e}")
            QMessageBox.critical(self, "Error", f"Failed to reset counters: {e}")

    def apply_runtime_tuning(self):
        """Apply sensitivity/runtime parameters (and persist to config)."""
        settings = {
            "yolo_conf": float(self.tune_conf.value()),
            "yolo_iou": float(self.tune_iou.value()),
            "sit_min_sec": int(self.tune_sit_min.value()),
            "vacant_grace_sec": int(self.tune_vacant.value()),
            "zone_point_mode": self.tune_zone_point.currentText(),
        }
        try:
            # Persist to config file
            yolo = self.config.get("yolo", {})
            yolo["conf"] = settings["yolo_conf"]
            yolo["iou"] = settings["yolo_iou"]
            self.config.set("yolo", yolo)

            runtime_cfg = self.config.get("runtime", {})
            runtime_cfg["sit_min_sec"] = settings["sit_min_sec"]
            runtime_cfg["vacant_grace_sec"] = settings["vacant_grace_sec"]
            runtime_cfg["zone_point_mode"] = settings["zone_point_mode"]
            self.config.set("runtime", runtime_cfg)
            self.config.save()

            # Hot apply to running runtime
            self.runtime_settings_override_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.runtime_settings_override_file, "w", encoding="utf-8") as f:
                json.dump(settings, f, ensure_ascii=False, indent=2)
            self.statusBar().showMessage("Runtime tuning applied")
            logger.info(f"Applied runtime tuning: {settings}")
        except Exception as e:
            logger.error(f"Failed to apply tuning: {e}")
            QMessageBox.critical(self, "Error", f"Failed to apply tuning: {e}")

    def update_snapshot_list(self, snapshots):
        """Update dashboard snapshot file list."""
        self.snapshot_list.clear()
        if not snapshots:
            return
        for item in reversed(snapshots[-50:]):
            path = item.get("path", "")
            camera = item.get("camera", "")
            gid = item.get("gid", 0)
            event_type = item.get("event_type", "")
            ts = item.get("timestamp", "")
            label = f"{ts} | {event_type} | {camera} | GID {gid} | {Path(path).name}"
            row = QListWidgetItem(label)
            row.setData(Qt.UserRole, path)
            self.snapshot_list.addItem(row)

    def open_snapshot_item(self, item: QListWidgetItem):
        path = item.data(Qt.UserRole)
        if not path:
            return
        p = Path(path)
        if not p.exists():
            QMessageBox.warning(self, "Missing File", f"Snapshot not found:\n{path}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(p.resolve())))

    def check_updates(self):
        """Check remote metadata, download & verify primary asset, and stage it under `updates/`.

        This is conservative: it will not auto-install. It requires `updates.metadata_url` in config
        or will prompt for a URL.
        """
        try:
            upd_cfg = CONFIG.get('updates', {})
            metadata_url = upd_cfg.get('metadata_url')
            if not metadata_url:
                from PySide6.QtWidgets import QInputDialog
                metadata_url, ok = QInputDialog.getText(self, 'Update URL', 'Enter metadata JSON URL:')
                if not ok or not metadata_url:
                    return

            updater = Updater(CONFIG.data if hasattr(CONFIG, 'data') else CONFIG)
            self.statusBar().showMessage('Checking for updates...')
            metadata = updater.check_for_update(metadata_url)
            version, notes = updater.inspect_metadata_for_update(metadata)
            asset = updater.select_primary_asset(metadata)
            if not asset:
                QMessageBox.information(self, 'No Update', 'No downloadable assets found in metadata.')
                return

            name = asset.get('name')
            url = asset.get('url')
            sha = asset.get('sha256') or asset.get('sha') or asset.get('checksum')

            msg = f"Found version: {version or 'unknown'}\n{notes or ''}\n\nAsset: {name}\nURL: {url}\n\nDownload and verify?"
            reply = QMessageBox.question(self, 'Update Found', msg, QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                return

            tmp_path = updater.download_asset(url, name=name)
            if sha:
                ok = updater.verify_sha256(tmp_path, sha)
                if not ok:
                    QMessageBox.critical(self, 'Verify Failed', 'SHA256 mismatch; download may be corrupted.')
                    tmp_path.unlink(missing_ok=True)
                    return

            staged = updater.stage_update(tmp_path)
            QMessageBox.information(self, 'Downloaded', f'Update downloaded and staged: {staged}')
            self.statusBar().showMessage('Update downloaded and staged', 5000)

        except Exception as e:
            QMessageBox.critical(self, 'Update Error', f'Failed to check/download update: {e}')

    def install_update(self):
        """Allow operator to select a staged update and run the installer (.exe).

        This performs a user confirmation then attempts to launch the selected file.
        Currently implemented for Windows executables; on other platforms it will
        open the file with the default handler where possible.
        """
        try:
            upd_cfg = CONFIG.get('updates', {})
            updates_dir = Path(upd_cfg.get('download_dir', 'updates'))
            if not updates_dir.exists():
                QMessageBox.information(self, 'No Updates', f'No staged updates found in: {updates_dir}')
                return

            # Ask user to pick a file from updates dir
            out = QFileDialog.getOpenFileName(self, 'Select staged update to install', str(updates_dir), 'All files (*)')
            if not out or not out[0]:
                return
            chosen = Path(out[0])
            if not chosen.exists():
                QMessageBox.critical(self, 'Not found', f'Selected file not found: {chosen}')
                return

            # If archive, offer atomic deploy
            if chosen.suffix.lower() in ['.zip', '.tar', '.gz', '.bz2']:
                reply = QMessageBox.question(self, 'Install Update', f'Detected archive {chosen.name}. Perform atomic deploy?', QMessageBox.Yes | QMessageBox.No)
                if reply == QMessageBox.Yes:
                    try:
                        updater = Updater(CONFIG.data if hasattr(CONFIG, 'data') else CONFIG)
                        root = self._project_root()
                        # stop runtime service before deploying
                        was_running = self.is_running
                        if was_running:
                            self.stop_service()
                        deployed = updater.install_update_atomic(chosen, deploy_target=root)
                        QMessageBox.information(self, 'Deployed', f'Update deployed to: {deployed}')
                        # restart runtime service if it was running
                        if was_running:
                            self.start_service()
                    except Exception as e:
                        QMessageBox.critical(self, 'Deploy Failed', f'Atomic deploy failed: {e}')
                    return

            reply = QMessageBox.question(self, 'Install Update', f'Run installer: {chosen.name}?', QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                return

            # Launch installer
            if sys.platform.startswith('win'):
                # Run elevated via PowerShell Start-Process -Verb RunAs
                try:
                    ps_cmd = [
                        'powershell', '-NoProfile', '-Command',
                        f"Start-Process -FilePath '{str(chosen.resolve())}' -Verb runAs"
                    ]
                    subprocess.Popen(ps_cmd)
                    self.statusBar().showMessage(f'Installer launched (elevated): {chosen.name}', 5000)
                except Exception as e:
                    QMessageBox.critical(self, 'Launch Failed', f'Failed to launch installer elevated: {e}')
            else:
                # On non-Windows, try to open with system opener
                try:
                    subprocess.Popen(['xdg-open', str(chosen.resolve())])
                    self.statusBar().showMessage(f'Opened installer: {chosen.name}', 5000)
                except Exception as e:
                    QMessageBox.critical(self, 'Not supported', f'Auto-install not supported on this platform: {e}')

        except Exception as e:
            QMessageBox.critical(self, 'Install Error', f'Failed to install update: {e}')

    
    def run_diagnostics(self):
        """Run system diagnostics: network, RTSP per camera, model file, disk write, supabase keys."""
        lines = []
        # Network: DNS
        import socket, time
        try:
            t0 = time.time()
            ip = socket.gethostbyname('8.8.8.8')
            dns_ok = True
            t_dns = round((time.time() - t0) * 1000)
            lines.append(f"Network: DNS resolution to 8.8.8.8 -> {ip} ({t_dns} ms)")
        except Exception as e:
            lines.append(f"Network: DNS resolution failed: {e}")

        # RTSP per camera (quick attempt)
        cameras = self.config.get('cameras', {})
        if cameras:
            lines.append("RTSP checks:")
            for name, cam in cameras.items():
                url = cam.get('rtsp_url', '')
                if not url:
                    lines.append(f"  - {name}: no URL configured")
                    continue
                # More robust RTSP check
                try:
                    import cv2 as _cv2
                    import time
                    cap = _cv2.VideoCapture(url)
                    if not cap.isOpened():
                        lines.append(f"  - {name}: FAIL (could not open stream at {url})")
                        cap.release()
                        continue

                    frames_read = 0
                    start_time = time.time()
                    # Test for 3 seconds to check stability
                    while time.time() - start_time < 3:
                        ret, frame = cap.read()
                        if ret:
                            frames_read += 1
                        else:
                            # Allow for brief interruptions
                            time.sleep(0.1)
                    cap.release()
                    if frames_read > 5: # e.g., > ~2 FPS is acceptable for a test
                        lines.append(f"  - {name}: OK ({frames_read} frames read in 3s)")
                    elif frames_read > 0:
                        lines.append(f"  - {name}: WARN (Unstable stream? Only {frames_read} frames in 3s)")
                    else:
                        lines.append(f"  - {name}: FAIL (no frames read in 3s)")
                except Exception as e:
                    lines.append(f"  - {name}: ERROR ({e})")
        else:
            lines.append("RTSP checks: no cameras configured")

        # Model check
        models_dir = self.config.get("paths", {}).get("models", "models")
        model_name = self.config.get("yolo", {}).get("model", "yolov8m.pt")
        model_path = Path(models_dir) / model_name
        if model_path.exists():
            lines.append(f"Model: found {model_path}")
        else:
            lines.append(f"Model: missing {model_path} (check config paths:models and yolo:model)")

        # Disk/write checks
        try:
            tmp_dir = Path(self.config.get('paths', {}).get('snapshots', 'snapshots'))
            tmp_dir.mkdir(parents=True, exist_ok=True)
            test_file = tmp_dir / f"diag_test_{int(time.time())}.tmp"
            with open(test_file, 'w') as f:
                f.write('ok')
            test_file.unlink()
            lines.append(f"Storage: snapshots dir writable: {tmp_dir}")
        except Exception as e:
            lines.append(f"Storage: write test failed: {e}")

        # Device/GPU check (torch)
        try:
            import torch
            dev = 'cpu'
            if torch.cuda.is_available():
                dev = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                dev = 'mps'
            lines.append(f"Device: torch available, preferred device: {dev}")
        except Exception:
            lines.append("Device: torch not available (running on CPU)")

        # Supabase config check
        sup = self.config.get('supabase', {})
        if sup.get('url') and sup.get('key'):
            lines.append("Supabase: config present (url/key) — will require runtime test to verify connectivity")
        else:
            lines.append("Supabase: missing url/key in config")

        self.diagnostics_output.setPlainText('\n'.join(lines))

    def _script_command(self, script_name: str) -> list[str]:
        root = self._project_root()
        return [sys.executable, "-u", str(root / "scripts" / script_name)]

    def _resolve_dataset_root_from_yaml(self, yaml_path: str) -> Optional[Path]:
        try:
            yp = Path(yaml_path).expanduser()
            if not yp.is_absolute():
                yp = (self._project_root() / yp).resolve()
            if not yp.exists():
                return None
            cfg = yaml.safe_load(yp.read_text(encoding="utf-8")) or {}
            path_cfg = str(cfg.get("path", "."))
            root = Path(path_cfg)
            if not root.is_absolute():
                cand1 = (yp.parent / root).resolve()
                cand2 = (self._project_root() / root).resolve()
                if cand1.exists():
                    return cand1
                if cand2.exists():
                    return cand2
                return cand1
            return root
        except Exception:
            return None

    def _count_images_dir(self, dir_path: Path) -> int:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".jp2"}
        if not dir_path.exists():
            return 0
        return sum(1 for p in dir_path.rglob("*") if p.is_file() and p.suffix.lower() in exts)

    def _dataset_stats(self, yaml_path: str) -> Dict[str, Any]:
        root = self._resolve_dataset_root_from_yaml(yaml_path)
        stats: Dict[str, Any] = {
            "ok": False,
            "root": str(root) if root else "",
            "train_images": 0,
            "val_images": 0,
            "class_image_counts": {name: 0 for name in self.yolo_class_names},
            "class_box_counts": {name: 0 for name in self.yolo_class_names},
        }
        if root is None:
            return stats
        train_img_dir = root / "images" / "train"
        val_img_dir = root / "images" / "val"
        stats["train_images"] = self._count_images_dir(train_img_dir)
        stats["val_images"] = self._count_images_dir(val_img_dir)

        seen_img_by_class = {name: set() for name in self.yolo_class_names}
        for split in ("train", "val"):
            lbl_dir = root / "labels" / split
            if not lbl_dir.exists():
                continue
            for lf in lbl_dir.glob("*.txt"):
                try:
                    lines = [x.strip() for x in lf.read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]
                except Exception:
                    continue
                for line in lines:
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    try:
                        cid = int(parts[0])
                    except ValueError:
                        continue
                    if 0 <= cid < len(self.yolo_class_names):
                        cname = self.yolo_class_names[cid]
                        stats["class_box_counts"][cname] += 1
                        seen_img_by_class[cname].add(f"{split}:{lf.stem}")
        for cname in self.yolo_class_names:
            stats["class_image_counts"][cname] = len(seen_img_by_class[cname])

        stats["ok"] = True
        return stats

    def _resolve_weights_path_ui(self, raw_path: str) -> Optional[Path]:
        p = Path(raw_path).expanduser()
        if p.is_absolute() and p.exists():
            return p
        if p.exists():
            return p.resolve()
        root = self._project_root()
        candidate = root / raw_path
        if candidate.exists():
            return candidate.resolve()
        search_roots = [
            root / "runs" / "train",
            root / "runs" / "detect",
            Path.home() / "runs" / "train",
            Path.home() / "runs" / "detect",
        ]
        found: list[Path] = []
        for sr in search_roots:
            if sr.exists():
                found.extend(sr.glob("**/weights/best.pt"))
        candidates = sorted(found, key=lambda x: x.stat().st_mtime, reverse=True)
        if candidates:
            return candidates[0].resolve()
        return None

    def _start_command_worker(
        self,
        worker_name: str,
        thread_name: str,
        cmd: list[str],
        output_widget: QTextEdit,
        finish_handler,
    ) -> None:
        existing_thread: Optional[QThread] = getattr(self, thread_name, None)
        if existing_thread is not None and existing_thread.isRunning():
            QMessageBox.warning(self, "Busy", "A command is already running for this tab.")
            return

        root = self._project_root()
        worker = CommandWorker(cmd=cmd, cwd=str(root))
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.output.connect(lambda text: output_widget.append(text))

        def _finish(code: int, full_output: str):
            try:
                finish_handler(code, full_output)
            finally:
                thread.quit()
                thread.wait(2000)
                worker.deleteLater()
                thread.deleteLater()
                setattr(self, worker_name, None)
                setattr(self, thread_name, None)

        worker.finished.connect(_finish)
        setattr(self, worker_name, worker)
        setattr(self, thread_name, thread)
        thread.start()

    def run_label_validation(self):
        data_yaml = self.train_data_yaml.text().strip()
        if not data_yaml:
            QMessageBox.warning(self, "Input Error", "Please provide dataset yaml path")
            return
        dataset_root = str(Path(data_yaml).resolve().parent)
        cmd = self._script_command("yolo_validate_labels.py") + ["--dataset", dataset_root]
        self.train_output.append(f"$ {' '.join(cmd)}")

        def on_finish(code: int, _out: str):
            if code == 0:
                self.statusBar().showMessage("Label validation: OK", 3000)
            else:
                self.statusBar().showMessage("Label validation: failed", 5000)

        self._start_command_worker(
            worker_name="_validate_worker",
            thread_name="_validate_thread",
            cmd=cmd,
            output_widget=self.train_output,
            finish_handler=on_finish,
        )

    def run_model_training(self):
        stats = self._dataset_stats(self.train_data_yaml.text().strip())
        if not stats.get("ok", False):
            QMessageBox.warning(self, "Dataset Error", "Dataset YAML not found or unreadable.")
            return
        train_images = int(stats.get("train_images", 0))
        val_images = int(stats.get("val_images", 0))
        total_images = train_images + val_images
        if train_images <= 0:
            QMessageBox.warning(self, "Dataset Error", "No images in train split. Add images first.")
            return
        if total_images < 20:
            reply = QMessageBox.question(
                self,
                "Small Dataset Warning",
                (
                    f"Dataset is small (train={train_images}, val={val_images}, total={total_images}).\n"
                    "Training may overfit and metrics may be unstable.\n"
                    "Continue anyway?"
                ),
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        cmd = self._script_command("train_yolo_head_staff.py") + [
            "--data", self.train_data_yaml.text().strip(),
            "--model", self.train_model_base.text().strip(),
            "--epochs", str(int(self.train_epochs.value())),
            "--imgsz", str(int(self.train_imgsz.value())),
            "--batch", str(int(self.train_batch.value())),
            "--device", self.train_device.currentText(),
            "--exist-ok",
        ]
        self.train_output.append(f"$ {' '.join(cmd)}")

        def on_finish(code: int, output: str):
            if code == 0:
                self.statusBar().showMessage("Training finished", 4000)
                for line in output.splitlines():
                    if line.startswith("best_model="):
                        best = line.split("=", 1)[1].strip()
                        self.eval_weights.setText(best)
                        self.eval_data_yaml.setText(self.train_data_yaml.text().strip())
                        break
            else:
                self.statusBar().showMessage("Training failed", 5000)

        self._start_command_worker(
            worker_name="_train_worker",
            thread_name="_train_thread",
            cmd=cmd,
            output_widget=self.train_output,
            finish_handler=on_finish,
        )

    def _set_metric_percent(self, label: QLabel, bar: QProgressBar, value: float):
        pct = max(0.0, min(100.0, value))
        label.setText(f"{pct:.2f}%")
        bar.setValue(int(round(pct)))

    def run_model_evaluation(self):
        resolved = self._resolve_weights_path_ui(self.eval_weights.text().strip())
        if resolved is not None:
            self.eval_weights.setText(str(resolved))
        cmd = self._script_command("eval_yolo_head_staff.py") + [
            "--weights", self.eval_weights.text().strip(),
            "--data", self.eval_data_yaml.text().strip(),
            "--imgsz", str(int(self.eval_imgsz.value())),
            "--batch", str(int(self.eval_batch.value())),
            "--device", self.eval_device.currentText(),
        ]
        self.eval_output.append(f"$ {' '.join(cmd)}")

        def on_finish(code: int, output: str):
            if code != 0:
                self.statusBar().showMessage("Evaluation failed", 5000)
                self.eval_output.append("Hint: select .pt in Weights (or run training first).")
                return
            metrics_line = None
            using_weights = None
            for line in output.splitlines():
                if line.startswith("USING_WEIGHTS:"):
                    using_weights = line.split("USING_WEIGHTS:", 1)[1].strip()
                if line.startswith("METRICS_JSON:"):
                    metrics_line = line.split("METRICS_JSON:", 1)[1].strip()
            if using_weights:
                self.eval_weights.setText(using_weights)
            if not metrics_line:
                self.statusBar().showMessage("Evaluation done (metrics not parsed)", 5000)
                return
            try:
                data = json.loads(metrics_line)
                self._set_metric_percent(self.metric_precision, self.metric_bar_precision, float(data.get("precision_pct", 0.0)))
                self._set_metric_percent(self.metric_recall, self.metric_bar_recall, float(data.get("recall_pct", 0.0)))
                self._set_metric_percent(self.metric_map50, self.metric_bar_map50, float(data.get("map50_pct", 0.0)))
                self._set_metric_percent(self.metric_map, self.metric_bar_map, float(data.get("map50_95_pct", 0.0)))
                self.statusBar().showMessage("Evaluation finished", 4000)
            except Exception as e:
                self.eval_output.append(f"Failed to parse metrics JSON: {e}")
                self.statusBar().showMessage("Evaluation finished (parse error)", 5000)

        self._start_command_worker(
            worker_name="_eval_worker",
            thread_name="_eval_thread",
            cmd=cmd,
            output_widget=self.eval_output,
            finish_handler=on_finish,
        )

    def lab_add_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select images",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp);;All files (*)",
        )
        if not files:
            return
        for path in files:
            if path not in self.lab_items:
                self.lab_items[path] = {"split": "train", "manual_class": self.yolo_class_names[0]}
        self.lab_refresh_image_list()
        self.lab_log.append(f"Added {len(files)} image(s)")

    def lab_remove_selected_images(self):
        selected = self.lab_image_list.selectedItems()
        if not selected:
            return
        for item in selected:
            p = item.data(Qt.UserRole)
            if p in self.lab_items:
                del self.lab_items[p]
        self.lab_refresh_image_list()

    def lab_clear_images(self):
        self.lab_items = {}
        self.lab_current_image_path = None
        self.lab_current_detections = []
        self.lab_image_list.clear()
        self.lab_det_table.setRowCount(0)
        self.lab_preview.setText("Image preview")

    def lab_apply_meta_to_selected(self):
        selected = self.lab_image_list.selectedItems()
        if not selected:
            return
        split = self.lab_split_combo.currentText()
        cls_name = self.lab_manual_class_combo.currentText()
        for item in selected:
            p = item.data(Qt.UserRole)
            if p in self.lab_items:
                self.lab_items[p]["split"] = split
                self.lab_items[p]["manual_class"] = cls_name
        self.lab_refresh_image_list()

    def lab_auto_split_items(self):
        if not self.lab_items:
            QMessageBox.information(self, "No images", "Add images first.")
            return
        val_pct = int(self.lab_val_percent.value())
        paths = list(self.lab_items.keys())
        rnd = random.Random(42)
        rnd.shuffle(paths)
        val_n = max(1, int(round(len(paths) * (val_pct / 100.0)))) if len(paths) > 1 else 0
        val_set = set(paths[:val_n])
        for p in paths:
            self.lab_items[p]["split"] = "val" if p in val_set else "train"
        self.lab_refresh_image_list()
        self.lab_log.append(f"Auto split done: train={len(paths)-len(val_set)} val={len(val_set)} (val%={val_pct})")

    def lab_refresh_dataset_summary(self):
        stats = self._dataset_stats(self.lab_dataset_yaml.text().strip())
        if not stats.get("ok", False):
            self.lab_summary_text.setPlainText("Dataset YAML not found or unreadable.")
            return
        lines = [
            f"root: {stats.get('root', '-')}",
            f"images/train: {stats.get('train_images', 0)}",
            f"images/val: {stats.get('val_images', 0)}",
            "",
            "Per class:",
        ]
        class_image_counts = stats.get("class_image_counts", {})
        class_box_counts = stats.get("class_box_counts", {})
        for cname in self.yolo_class_names:
            lines.append(
                f"- {cname}: images={int(class_image_counts.get(cname, 0))}, "
                f"boxes={int(class_box_counts.get(cname, 0))}"
            )
        self.lab_summary_text.setPlainText("\n".join(lines))

    def lab_refresh_image_list(self):
        self.lab_image_list.clear()
        for path in sorted(self.lab_items.keys()):
            meta = self.lab_items[path]
            txt = f"[{meta.get('split','train')}] [{meta.get('manual_class','person')}] {Path(path).name}"
            row = QListWidgetItem(txt)
            row.setData(Qt.UserRole, path)
            self.lab_image_list.addItem(row)

    def lab_on_image_changed(self, current: Optional[QListWidgetItem], _previous: Optional[QListWidgetItem]):
        if current is None:
            self.lab_current_image_path = None
            self.lab_preview.setText("Image preview")
            return
        self.lab_current_image_path = str(current.data(Qt.UserRole))
        item_meta = self.lab_items.get(self.lab_current_image_path, {})
        dets = item_meta.get("detections", [])
        self.lab_current_detections = [dict(d) for d in dets] if dets else []
        self.lab_fill_detection_table()
        self.lab_render_preview(self.lab_current_image_path, self.lab_current_detections)

    def lab_render_preview(self, image_path: str, detections: list[Dict[str, Any]]):
        try:
            import cv2

            img = cv2.imread(image_path)
            if img is None:
                self.lab_preview.setText(f"Cannot open image: {image_path}")
                return
            for det in detections:
                x1, y1, x2, y2 = [int(v) for v in det.get("bbox", [0, 0, 0, 0])]
                cls_name = str(det.get("correct_class", det.get("class_name", "unknown")))
                conf = float(det.get("conf", 0.0))
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    f"{cls_name} {conf:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(
                self.lab_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.lab_preview.setPixmap(pix)
        except Exception as e:
            self.lab_preview.setText(f"Preview error: {e}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.lab_current_image_path:
            self.lab_render_preview(self.lab_current_image_path, self.lab_current_detections)

    def lab_run_evaluation(self):
        stats = self._dataset_stats(self.lab_dataset_yaml.text().strip())
        saved_total = int(stats.get("train_images", 0)) + int(stats.get("val_images", 0))
        if len(self.lab_items) > saved_total:
            self.lab_log.append(
                f"NOTE: Imported in Lab={len(self.lab_items)} but saved in dataset={saved_total}. "
                "Use 'Save Corrected -> Dataset' per image before evaluation."
            )
        resolved = self._resolve_weights_path_ui(self.lab_weights.text().strip())
        if resolved is not None:
            self.lab_weights.setText(str(resolved))
        cmd = self._script_command("eval_yolo_head_staff.py") + [
            "--weights", self.lab_weights.text().strip(),
            "--data", self.lab_dataset_yaml.text().strip(),
            "--device", self.lab_eval_device.currentText(),
        ]
        self.lab_log.append(f"$ {' '.join(cmd)}")

        def on_finish(code: int, output: str):
            if code != 0:
                self.lab_log.append("Evaluation failed")
                self.lab_log.append("Hint: select .pt in Weights (or run training first).")
                return
            metrics_line = None
            using_weights = None
            for line in output.splitlines():
                if line.startswith("USING_WEIGHTS:"):
                    using_weights = line.split("USING_WEIGHTS:", 1)[1].strip()
                if line.startswith("METRICS_JSON:"):
                    metrics_line = line.split("METRICS_JSON:", 1)[1].strip()
            if using_weights:
                self.lab_weights.setText(using_weights)
            if not metrics_line:
                return
            try:
                m = json.loads(metrics_line)
                self.lab_metric_precision.setText(f"P: {float(m.get('precision_pct', 0.0)):.2f}%")
                self.lab_metric_recall.setText(f"R: {float(m.get('recall_pct', 0.0)):.2f}%")
                self.lab_metric_map50.setText(f"mAP50: {float(m.get('map50_pct', 0.0)):.2f}%")
                self.lab_metric_map.setText(f"mAP50-95: {float(m.get('map50_95_pct', 0.0)):.2f}%")
            except Exception as e:
                self.lab_log.append(f"Metric parse error: {e}")

        self._start_command_worker(
            worker_name="_lab_eval_worker",
            thread_name="_lab_eval_thread",
            cmd=cmd,
            output_widget=self.lab_log,
            finish_handler=on_finish,
        )

    def lab_run_detect_selected(self):
        if not self.lab_current_image_path:
            QMessageBox.warning(self, "No image", "Select an image first")
            return
        try:
            import cv2
            from ultralytics import YOLO

            resolved = self._resolve_weights_path_ui(self.lab_weights.text().strip())
            if resolved is None:
                raise RuntimeError("Weights not found. Please Browse .pt model first.")
            self.lab_weights.setText(str(resolved))
            img = cv2.imread(self.lab_current_image_path)
            if img is None:
                raise RuntimeError("Cannot open selected image")
            model = YOLO(str(resolved))
            results = model.predict(img, conf=0.25, iou=0.45, verbose=False)
            dets: list[Dict[str, Any]] = []
            if results and results[0].boxes is not None:
                b = results[0].boxes
                cls_vals = b.cls
                conf_vals = b.conf
                for i, box in enumerate(list(b.xyxy)):
                    x1, y1, x2, y2 = box.cpu().numpy().tolist()
                    cid = int(float(cls_vals[i].item())) if cls_vals is not None else 0
                    conf = float(conf_vals[i].item()) if conf_vals is not None else 0.0
                    cls_name = self.yolo_class_names[cid] if 0 <= cid < len(self.yolo_class_names) else str(cid)
                    dets.append(
                        {
                            "idx": i,
                            "class_id": cid,
                            "class_name": cls_name,
                            "correct_class": cls_name,
                            "conf": conf,
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        }
                    )
            self.lab_current_detections = dets
            if self.lab_current_image_path in self.lab_items:
                self.lab_items[self.lab_current_image_path]["detections"] = [dict(d) for d in dets]
            self.lab_fill_detection_table()
            self.lab_render_preview(self.lab_current_image_path, self.lab_current_detections)
            self.lab_log.append(f"Auto-detected {len(dets)} object(s)")
        except Exception as e:
            QMessageBox.critical(self, "Detect Error", str(e))

    def lab_fill_detection_table(self):
        self.lab_det_table.setRowCount(0)
        for row, det in enumerate(self.lab_current_detections):
            self.lab_det_table.insertRow(row)
            x1, y1, x2, y2 = det.get("bbox", [0, 0, 0, 0])
            vals = [
                str(det.get("idx", row)),
                str(det.get("class_name", "")),
                f"{float(det.get('conf', 0.0)):.3f}",
                f"{x1:.1f}",
                f"{y1:.1f}",
                f"{x2:.1f}",
                f"{y2:.1f}",
            ]
            for col, text in enumerate(vals):
                self.lab_det_table.setItem(row, col, QTableWidgetItem(text))
            combo = QComboBox()
            combo.addItems(self.yolo_class_names)
            combo.setCurrentText(str(det.get("correct_class", self.yolo_class_names[0])))
            combo.currentTextChanged.connect(lambda text, r=row: self.lab_on_correct_class_changed(r, text))
            self.lab_det_table.setCellWidget(row, 7, combo)

    def lab_on_correct_class_changed(self, row: int, text: str):
        if 0 <= row < len(self.lab_current_detections):
            self.lab_current_detections[row]["correct_class"] = text
            if self.lab_current_image_path in self.lab_items:
                self.lab_items[self.lab_current_image_path]["detections"] = [dict(d) for d in self.lab_current_detections]
            if self.lab_current_image_path:
                self.lab_render_preview(self.lab_current_image_path, self.lab_current_detections)

    def _lab_save_one_image_to_dataset(self, image_path: str) -> tuple[bool, str]:
        dataset_root = Path(self.lab_dataset_yaml.text().strip()).resolve().parent
        meta = self.lab_items.get(image_path, {"split": "train", "manual_class": "person"})
        split = str(meta.get("split", "train"))
        manual_class = str(meta.get("manual_class", "person"))
        dets = meta.get("detections", []) or []

        img_src = Path(image_path)
        img_dst = dataset_root / "images" / split / img_src.name
        lbl_dst = dataset_root / "labels" / split / (img_src.stem + ".txt")
        img_dst.parent.mkdir(parents=True, exist_ok=True)
        lbl_dst.parent.mkdir(parents=True, exist_ok=True)

        try:
            import shutil
            import cv2

            shutil.copy2(str(img_src), str(img_dst))
            img = cv2.imread(str(img_src))
            if img is None:
                return False, f"Cannot read image: {img_src}"
            h, w = img.shape[:2]

            rows: list[str] = []
            if dets:
                for det in dets:
                    cls_name = str(det.get("correct_class", det.get("class_name", manual_class)))
                    if cls_name not in self.yolo_class_names:
                        continue
                    cid = self.yolo_class_names.index(cls_name)
                    x1, y1, x2, y2 = [float(v) for v in det.get("bbox", [0, 0, 0, 0])]
                    bw = max(0.0, x2 - x1)
                    bh = max(0.0, y2 - y1)
                    if bw <= 1.0 or bh <= 1.0:
                        continue
                    cx = x1 + (bw / 2.0)
                    cy = y1 + (bh / 2.0)
                    rows.append(f"{cid} {cx / w:.6f} {cy / h:.6f} {bw / w:.6f} {bh / h:.6f}")
            else:
                cid = self.yolo_class_names.index(manual_class) if manual_class in self.yolo_class_names else 0
                rows.append(f"{cid} 0.500000 0.500000 1.000000 1.000000")

            with open(lbl_dst, "w", encoding="utf-8") as f:
                f.write("\n".join(rows) + ("\n" if rows else ""))

            return True, f"Saved: {img_dst.name} -> {split} (rows={len(rows)})"
        except Exception as e:
            return False, f"Failed {img_src.name}: {e}"

    def lab_save_corrected_to_dataset(self):
        if not self.lab_current_image_path:
            QMessageBox.warning(self, "No image", "Select an image first")
            return
        if self.lab_current_image_path in self.lab_items:
            self.lab_items[self.lab_current_image_path]["detections"] = [dict(d) for d in self.lab_current_detections]
        ok, msg = self._lab_save_one_image_to_dataset(self.lab_current_image_path)
        if ok:
            self.lab_log.append(msg)
            self.lab_refresh_dataset_summary()
            self.statusBar().showMessage("Saved corrected labels to dataset", 4000)
        else:
            QMessageBox.critical(self, "Save Error", msg)

    def lab_save_all_imported_to_dataset(self):
        if not self.lab_items:
            QMessageBox.information(self, "No images", "Add images first.")
            return
        selected_class = self.lab_manual_class_combo.currentText()
        for meta in self.lab_items.values():
            meta["manual_class"] = selected_class
        if self.lab_current_image_path in self.lab_items and self.lab_current_detections is not None:
            self.lab_items[self.lab_current_image_path]["detections"] = [dict(d) for d in self.lab_current_detections]

        total = len(self.lab_items)
        ok_count = 0
        fail_msgs: list[str] = []
        for image_path in sorted(self.lab_items.keys()):
            ok, msg = self._lab_save_one_image_to_dataset(image_path)
            if ok:
                ok_count += 1
            else:
                fail_msgs.append(msg)

        self.lab_refresh_dataset_summary()
        self.lab_refresh_image_list()
        self.lab_log.append(f"Save all done: success={ok_count}/{total} failed={len(fail_msgs)}")
        for m in fail_msgs[:20]:
            self.lab_log.append(f"ERROR: {m}")
        if fail_msgs:
            QMessageBox.warning(
                self,
                "Save All Completed with Errors",
                f"Saved {ok_count}/{total}. Failed {len(fail_msgs)} item(s). Check Dataset Lab log.",
            )
        else:
            self.statusBar().showMessage(f"Saved all imported images: {ok_count}/{total}", 5000)
    
    def save_config(self):
        """Save config ไฟล์ (สำหรับ camera manager เป็นต้น)"""
        try:
            CONFIG.set_all(self.config)
            logger.info("Config saved successfully")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save config: {e}")
    
    def refresh_logs(self):
        """Refresh logs display"""
        logs_dir = Path(CONFIG.get("paths", {}).get("logs", "logs"))
        log_files = [
            logs_dir / "controller.log",
            logs_dir / "runtime.log",
            logs_dir / "runtime_stdout.log",
        ]
        log_files.extend(sorted(logs_dir.glob("runtime_*.log")))
        try:
            all_lines = []
            for log_file in log_files:
                if not log_file.exists():
                    continue
                with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                    lines = f.readlines()[-200:]
                all_lines.extend([f"[{log_file.name}] {ln}" for ln in lines])

            if not all_lines:
                self.logs_display.setText("No logs found")
                return

            level_filter = self.log_level.currentText()
            camera_filter = self.log_camera.currentText()
            filtered = []
            for ln in all_lines:
                if level_filter != "ALL" and f"[{level_filter}]" not in ln and f"[{level_filter.replace('WARN', 'WARNING')}]" not in ln:
                    continue
                if camera_filter != "ALL" and camera_filter not in ln:
                    continue
                filtered.append(ln)

            self.logs_display.setText(''.join(filtered[-300:]))
        except Exception as e:
            self.logs_display.setText(f"Error reading logs: {e}")

    def export_logs(self):
        logs_dir = CONFIG.get("paths", {}).get("logs", "logs")
        out = QFileDialog.getSaveFileName(self, "Export Logs", "controller_export.log", "Log files (*.log);;All files (*)")
        if out and out[0]:
            try:
                merged = []
                from glob import glob
                names = ["controller.log", "runtime.log", "runtime_stdout.log"]
                names.extend([os.path.basename(p) for p in glob(os.path.join(logs_dir, "runtime_*.log"))])
                for name in sorted(set(names)):
                    src = os.path.join(logs_dir, name)
                    if not os.path.exists(src):
                        continue
                    with open(src, 'r', encoding='utf-8', errors='replace') as fsrc:
                        merged.append(f"\n===== {name} =====\n")
                        merged.append(fsrc.read())
                with open(out[0], 'w', encoding='utf-8') as fdst:
                    fdst.write(''.join(merged) if merged else "No log files found.")
                QMessageBox.information(self, 'Export', f'Logs exported to {out[0]}')
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Export failed: {e}')

    def open_logs_folder(self):
        logs_dir = CONFIG.get("paths", {}).get("logs", "logs")
        try:
            # Ensure path exists
            from pathlib import Path
            p = Path(logs_dir)
            if not p.exists():
                p.mkdir(parents=True, exist_ok=True)

            # Platform-specific open
            if sys.platform.startswith('win'):
                # Windows: use os.startfile
                import os
                os.startfile(str(p.resolve()))
            else:
                import subprocess
                if sys.platform == 'darwin':
                    subprocess.Popen(['open', str(p.resolve())])
                else:
                    # Linux/Unix
                    subprocess.Popen(['xdg-open', str(p.resolve())])
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Cannot open logs folder: {e}')


# =========================
# MAIN
# =========================

def main():
    app = QApplication(sys.argv)
    controller = MainController()
    controller.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
