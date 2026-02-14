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
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTabWidget, QStatusBar, QMessageBox,
    QTreeWidget, QTreeWidgetItem, QTextEdit, QSplitter, QSpinBox, QFileDialog,
    QListWidget, QListWidgetItem, QDoubleSpinBox, QComboBox
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject, QThread
from PySide6.QtGui import QIcon, QFont, QColor, QDesktopServices
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
        self.runtime_process: Optional[Any] = None
        self.runtime_output_handle = None
        self.is_running = False
        root = self._project_root()
        self.dashboard_state_file = root / self.config.get("paths", {}).get("dashboard_state", "runtime/dashboard_state.json")
        self.reset_counts_flag_file = root / self.config.get("paths", {}).get("reset_counts_flag", "runtime/reset_counts.flag")
        self.runtime_settings_override_file = root / self.config.get("paths", {}).get("runtime_settings_override", "runtime/runtime_settings.override.json")
        self.runtime_stdout_file = root / self.config.get("paths", {}).get("logs", "logs") / f"runtime_{datetime.now().strftime('%Y-%m-%d')}.log"
        self.last_runtime_state_ts = 0.0
        
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

    def _runtime_command(self) -> list[str]:
        """Return subprocess command to launch runtime service."""
        root = self._project_root()

        if getattr(sys, "frozen", False):
            candidates = [
                root / "runtime_service.exe",
                root / "runtime_service",
                root / "agent_v2.exe",
                root / "agent_v2",
            ]
            for c in candidates:
                if c.exists():
                    return [str(c)]
            raise FileNotFoundError(
                f"Runtime executable not found in {root}. "
                "Expected runtime_service(.exe) next to controller binary."
            )

        return [sys.executable, "-u", str(root / "runtime" / "agent_v2.py")]
    
    def start_service(self):
        """Start runtime service"""
        logger.info("Starting runtime service...")

        try:
            root = self._project_root()
            runtime_cmd = self._runtime_command()
            # Use unbuffered runtime output and persist stdout/stderr to file.
            self.runtime_stdout_file = root / self.config.get("paths", {}).get("logs", "logs") / f"runtime_{datetime.now().strftime('%Y-%m-%d')}.log"
            self.runtime_stdout_file.parent.mkdir(parents=True, exist_ok=True)
            self.runtime_output_handle = open(self.runtime_stdout_file, "a", encoding="utf-8")
            self.runtime_output_handle.write(f"\n[{datetime.now().isoformat()}] [controller] START_SERVICE\n")
            self.runtime_output_handle.flush()
            self.runtime_process = subprocess.Popen(
                runtime_cmd,
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
        
        if self.runtime_process:
            self.runtime_process.terminate()
            try:
                self.runtime_process.wait(timeout=5)
            except:
                self.runtime_process.kill()
            self.runtime_process = None
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
        if self.runtime_process:
            rc = self.runtime_process.poll()
            if rc is not None and self.is_running:
                self.is_running = False
                self.start_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                self.status_runtime.setText("⚫ Stopped")
                self.status_runtime.setStyleSheet("color: red;")
                self.auto_refresh_label.setText(f"⚠️ Runtime exited (code {rc})")
                self.auto_refresh_label.setStyleSheet("color: orange; font-weight: bold;")
                self.statusBar().showMessage(f"Runtime stopped (exit code {rc})")
                logger.error(f"Runtime process exited with code {rc}")
            else:
                runtime_alive = (rc is None)

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
                    # Treat stale state as not live.
                    is_fresh = (time.time() - ts) <= 15
                    if not is_fresh:
                        status["running"] = False
                    if status:
                        if not runtime_alive and self.runtime_process is not None:
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
                logger.debug(f"Failed to read dashboard state: {e}")
    
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
                try:
                    import cv2 as _cv2
                    cap = _cv2.VideoCapture(url)
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        lines.append(f"  - {name}: OK (frame grabbed)")
                    else:
                        lines.append(f"  - {name}: FAIL (no frame)")
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

