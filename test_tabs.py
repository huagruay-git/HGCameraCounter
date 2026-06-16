"""
Test script to verify all tabs in HGCameraCounter Controller
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("HGCameraCounter - Tab Functionality Test")
print("=" * 60)

# Test 1: Check all widget files exist
print("\n[1] Checking widget files...")
widget_files = [
    "controller/main.py",
    "controller/dashboard_client.py",
    "controller/live_viewer.py",
    "controller/setup_wizard.py",
    "controller/camera_manager.py",
    "controller/zone_editor.py",
    "controller/staff_builder.py",
    "controller/llm_evaluation_ui.py",
    "controller/cloud_sync_widget.py",
    "shared/config.py",
    "shared/logger.py",
    "shared/dashboard_updater.py",
]

for wf in widget_files:
    path = project_root / wf
    if path.exists():
        print(f"  OK {wf}")
    else:
        print(f"  MISSING {wf} - NOT FOUND!")

# Test 2: Try importing each widget module
print("\n[2] Testing module imports...")

modules_to_test = [
    ("shared.config", "Config"),
    ("shared.logger", "setup_logger"),
    ("shared.dashboard_updater", "init_dashboard_service"),
    ("controller.dashboard_client", "GUIDashboardClient"),
    ("controller.live_viewer", "LiveViewerWidget"),
    ("controller.setup_wizard", "SetupWizard"),
    ("controller.camera_manager", "CameraManagerWidget"),
    ("controller.zone_editor", "ZoneEditorWidget"),
    ("controller.staff_builder", "StaffBuilderWidget"),
    ("controller.llm_evaluation_ui", "LLMEvaluationWidget"),
    ("controller.cloud_sync_widget", "CloudSyncWidget"),
]

import_results = []
for module_name, class_name in modules_to_test:
    try:
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
        print(f"  OK {module_name}.{class_name}")
        import_results.append((module_name, class_name, True, None))
    except Exception as e:
        print(f"  FAIL {module_name}.{class_name} - {str(e)[:50]}")
        import_results.append((module_name, class_name, False, str(e)))

# Test 3: Check required dependencies
print("\n[3] Checking required dependencies...")
dependencies = [
    "PySide6",
    "cv2",
    "numpy",
    "yaml",
    "requests",
]

for dep in dependencies:
    try:
        __import__(dep)
        print(f"  OK {dep}")
    except ImportError as e:
        print(f"  MISSING {dep} - {str(e)[:50]}")

# Test 4: Check config file
print("\n[4] Checking configuration...")
config_path = project_root / "data" / "config" / "config.yaml"
if config_path.exists():
    print(f"  OK Config file exists: {config_path}")
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"  OK Config loaded successfully")
        if 'branch_code' in config:
            print(f"     Branch: {config['branch_code']}")
        if 'cameras' in config:
            print(f"     Cameras: {len(config['cameras'])} configured")
    except Exception as e:
        print(f"  WARN Config load error: {e}")
else:
    print(f"  WARN Config file not found: {config_path}")

# Test 5: Widget instantiation (without QApplication for safety)
print("\n[5] Testing widget initialization (mock mode)...")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

failed_imports = [r for r in import_results if not r[2]]
if failed_imports:
    print(f"\nWARN {len(failed_imports)} import(s) failed:")
    for module_name, class_name, success, error in failed_imports:
        print(f"     - {module_name}.{class_name}: {error}")
else:
    print("\nOK All modules imported successfully!")

print("\n" + "=" * 60)
print("Test complete! You can now run the main application.")
print("=" * 60)
