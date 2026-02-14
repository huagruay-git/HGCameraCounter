#!/usr/bin/env python
"""
Test script for B4 Camera Management
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from PySide6.QtWidgets import QApplication
from controller.main import MainController


def test_camera_management():
    """ทดสอบ Camera Management features"""
    
    app = QApplication(sys.argv)
    controller = MainController()
    
    print("✅ MainController initialized")
    print(f"📡 Camera Manager: {controller.camera_manager}")
    print(f"📱 Camera List Widget: {controller.camera_list}")
    
    # ตรวจสอบว่ามี cameras แต่แรก
    cameras = controller.config.get("cameras", {})
    print(f"\n📹 Current cameras: {list(cameras.keys())}")
    
    controller.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    test_camera_management()
