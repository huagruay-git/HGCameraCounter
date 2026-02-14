"""
Camera Management Dialog and Utilities

ตัวจัดการกล้องสำหรับ:
- เพิ่ม/แก้ไข/ลบกล้อง
- ทดสอบ RTSP
- ดูตัวอย่าง + metrics
- Import/Export JSON
"""

import json
import time
import threading
import cv2
from pathlib import Path
from typing import Dict, Optional, Tuple, Callable

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QCheckBox, QMessageBox, QTextEdit, QComboBox, QFileDialog, QScrollArea,
    QFrame, QProgressBar
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QPixmap, QImage, QFont
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem


class RTSPTester(QThread):
    """Background thread สำหรับทดสอบ RTSP connection"""
    
    test_complete = Signal(bool, str, float, float)  # success, message, latency, fps
    
    def __init__(self, rtsp_url: str):
        super().__init__()
        self.rtsp_url = rtsp_url
        
    def run(self):
        try:
            # เชื่อมต่อและบันทึกเวลา
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
            
            # พยายามอ่าน frame
            ret, frame = cap.read()
            latency = (time.time() - start_time) * 1000  # ms
            
            if not ret:
                cap.release()
                self.test_complete.emit(False, "Cannot read frames", latency, 0)
                return
            
            # คำนวณ FPS (ตัวอย่าง)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            
            cap.release()
            self.test_complete.emit(True, "Connection OK", latency, fps)
            
        except Exception as e:
            self.test_complete.emit(False, f"Error: {str(e)}", 0, 0)


class CameraFormDialog(QDialog):
    """Dialog สำหรับเพิ่ม/แก้ไขกล้อง"""
    
    def __init__(self, parent=None, camera_name: str = "", camera_config: Dict = None):
        super().__init__(parent)
        self.camera_name = camera_name
        self.camera_config = camera_config or {}
        self.test_result = None
        self.rtsp_tester = None
        
        self.init_ui()
        
    def init_ui(self):
        """สร้าง UI"""
        layout = QVBoxLayout()
        
        # Camera Name
        layout.addWidget(QLabel("Camera Name:"))
        self.name_input = QLineEdit()
        self.name_input.setText(self.camera_name)
        if self.camera_name:  # ถ้าแก้ไข ให้ disable name
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
        
    def browse_zones(self):
        """เลือก zones file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Zones File", "", "JSON Files (*.json)"
        )
        if file_path:
            self.zones_input.setText(file_path)
    
    def test_rtsp(self):
        """ทดสอบ RTSP connection"""
        url = self.url_input.text().strip()
        if not url:
            QMessageBox.warning(self, "Error", "Please enter RTSP URL")
            return
        
        self.test_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # Indeterminate
        self.result_text.clear()
        self.result_text.setText("Testing connection...")
        
        # เริ่ม background thread
        self.rtsp_tester = RTSPTester(url)
        self.rtsp_tester.test_complete.connect(self.on_test_complete)
        self.rtsp_tester.start()
    
    def on_test_complete(self, success: bool, message: str, latency: float, fps: float):
        """Callback เมื่อ test เสร็จ"""
        self.test_btn.setEnabled(True)
        self.progress.setVisible(False)
        
        if success:
            self.test_status.setText(f"✓ OK (Latency: {latency:.0f}ms, FPS: {fps:.1f})")
            result_msg = f"""
✓ Connection Successful

Latency: {latency:.2f} ms
FPS: {fps:.2f}
Status: Ready
"""
        else:
            self.test_status.setText(f"✗ Failed: {message}")
            result_msg = f"""
✗ Connection Failed

Message: {message}
Latency: {latency:.2f} ms
Status: Check URL and network
"""
        
        self.result_text.setText(result_msg)
        self.test_result = success
    
    def get_camera_data(self) -> Tuple[str, Dict]:
        """ได้ข้อมูลกล้องจาก form"""
        return (
            self.name_input.text(),
            {
                "rtsp_url": self.url_input.text(),
                "enabled": self.enabled_check.isChecked(),
                "note": self.note_input.text(),
                "zones_file": self.zones_input.text()
            }
        )


class CameraManagerWidget:
    """Widget utilities สำหรับจัดการกล้องใน main controller"""
    
    def __init__(self, main_controller):
        self.controller = main_controller
        self.config = main_controller.config  # This is a dict in main_controller
    
    def add_camera_dialog(self):
        """เปิด dialog เพิ่มกล้อง"""
        dialog = CameraFormDialog(self.controller)
        if dialog.exec() == QDialog.Accepted:
            cam_name, cam_config = dialog.get_camera_data()
            
            if not cam_name:
                QMessageBox.warning(self.controller, "Error", "Camera name is required")
                return
            
            if not cam_config["rtsp_url"]:
                QMessageBox.warning(self.controller, "Error", "RTSP URL is required")
                return
            
            # เพิ่มในกล้อง config
            if "cameras" not in self.config:
                self.config["cameras"] = {}
            
            if cam_name in self.config["cameras"]:
                QMessageBox.warning(self.controller, "Error", f"Camera '{cam_name}' already exists")
                return
            
            self.config["cameras"][cam_name] = cam_config
            self.controller.save_config()
            
            # อัปเดต UI
            self.refresh_camera_list()
            QMessageBox.information(self.controller, "Success", f"Camera '{cam_name}' added")
    
    def edit_camera_dialog(self):
        """เปิด dialog แก้ไขกล้อง"""
        if not self.controller.camera_list.selectedItems():
            QMessageBox.warning(self.controller, "Error", "Please select a camera")
            return
        
        selected_item = self.controller.camera_list.selectedItems()[0]
        cam_name = selected_item.text(0)
        cam_config = self.config.get("cameras", {}).get(cam_name)
        
        if not cam_config:
            QMessageBox.warning(self.controller, "Error", "Camera not found")
            return
        
        dialog = CameraFormDialog(self.controller, cam_name, cam_config)
        if dialog.exec() == QDialog.Accepted:
            _, updated_config = dialog.get_camera_data()
            self.config["cameras"][cam_name].update(updated_config)
            self.controller.save_config()
            
            # อัปเดต UI
            self.refresh_camera_list()
            QMessageBox.information(self.controller, "Success", f"Camera '{cam_name}' updated")
    
    def delete_camera(self):
        """ลบกล้อง"""
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
            self.controller.save_config()
            self.refresh_camera_list()
            QMessageBox.information(self.controller, "Success", f"Camera '{cam_name}' deleted")
    
    def test_camera(self):
        """ทดสอบกล้องที่เลือก"""
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
        
        # ทดสอบแบบ inline
        dialog = CameraFormDialog(self.controller, cam_name, cam_config)
        dialog.url_input.setText(url)
        dialog.test_rtsp()
        dialog.exec()
    
    def test_all_cameras(self):
        """ทดสอบกล้องทั้งหมด"""
        cameras = self.config.get("cameras", {})
        if not cameras:
            QMessageBox.warning(self.controller, "Error", "No cameras configured")
            return
        
        results = {}
        for cam_name, cam_config in cameras.items():
            url = cam_config.get("rtsp_url")
            if not url:
                results[cam_name] = "❌ No URL"
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
                    results[cam_name] = f"✓ OK ({latency:.0f}ms)"
                else:
                    results[cam_name] = f"❌ Cannot read frames ({latency:.0f}ms)"
            except Exception as e:
                results[cam_name] = f"❌ {str(e)[:50]}"
        
        # แสดงผล
        report = "Camera Test Results:\n" + "=" * 50 + "\n\n"
        for cam_name, result in results.items():
            report += f"{cam_name}: {result}\n"
        
        QMessageBox.information(self.controller, "Test Results", report)
    
    def import_cameras_json(self):
        """Import cameras จากไฟล์ JSON"""
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
            QMessageBox.information(self.controller, "Success", f"Imported {added} cameras")
        
        except Exception as e:
            QMessageBox.critical(self.controller, "Error", f"Import failed: {str(e)}")
    
    def export_cameras_json(self):
        """Export cameras เป็นไฟล์ JSON"""
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
        """รีเฟรช camera list ใน UI"""
        self.controller.camera_list.clear()
        
        cameras = self.config.get("cameras", {})
        for cam_name, cam_config in cameras.items():
            item = self.controller.camera_list.insertTopLevelItem(
                self.controller.camera_list.topLevelItemCount(),
                __class__._create_camera_item(cam_name, cam_config)
            )
    
    @staticmethod
    def _create_camera_item(cam_name: str, cam_config: Dict) -> QTreeWidgetItem:
        """สร้าง tree item สำหรับกล้อง"""
        return QTreeWidgetItem([
            cam_name,
            cam_config.get("rtsp_url", "")[:40],
            "✓" if cam_config.get("enabled", True) else "✗",
            cam_config.get("zones_file", "")
        ])
