# HG Camera Counter - Phase 1 Implementation

**Status**: ✅ Foundation Complete

## What's Done

### ✅ Project Setup (A)
- [x] Folder structure: `controller/`, `runtime/`, `shared/`
- [x] Central config management: `data/config/config.yaml`
- [x] Standard logging with rotating handlers
- [x] Path convention: models/, zones/, staff_gallery/, logs/, reports/

### ✅ Shared Utilities
- [x] `shared/config.py` - Config loader (YAML/JSON)
- [x] `shared/logger.py` - Standard logging setup

### ✅ Runtime Service (C)
- [x] `runtime/agent.py` - Refactored multi-camera service
  - Multi-camera pipeline
  - YOLO detection + tracking
  - Zone-based counting logic
  - Event logging
  - Graceful shutdown

### ✅ Staff DB Builder (Refactored)
- [x] `runtime/build_staff_db.py` - Uses centralized config
  - YOLO person detection
  - Embedding extraction
  - Staff database generation
  - Build report

### ✅ Controller GUI (B - Partial)
- [x] `controller/setup_wizard.py` - Multi-step wizard
  - Step 1: Supabase configuration
  - Step 2: Camera management
  - Step 3: Zone editing
  - Step 4: Staff DB building
  - Step 5: Diagnostics

- [x] `controller/main.py` - Main application
  - Dashboard (status overview)
  - Setup wizard launcher
  - Camera management
  - Diagnostics panel
  - Logs viewer
  - Service control (Start/Stop)

### ✅ Dependencies
- [x] `requirements.txt` - All dependencies listed

## Quick Start

### 1. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

Edit `data/config/config.yaml`:
```yaml
supabase:
  url: "your-supabase-url"
  key: "your-anon-key"

cameras:
  Camera_01:
    rtsp_url: "rtsp://..."
    enabled: true
```

### 3. Run Controller GUI

```bash
python controller/main.py
```

Then use Setup Wizard to:
1. Configure Supabase
2. Add cameras
3. Edit zones (launch zone_picker.py)
4. Build staff database
5. Run diagnostics

### 4. Run Runtime Service (Optional)

```bash
python runtime/agent.py
```

## Configuration

### `data/config/config.yaml`

```yaml
project_name: "HG Camera Counter"
version: "0.1.0"
branch_code: "DEMO"

supabase:
  url: ""      # Set from env or manually
  key: ""      # Set from env or manually

cameras:
  Camera_01:
    rtsp_url: "rtsp://..."
    enabled: true
    zones_file: "data/zones/zones_Camera_01.json"

yolo:
  model: "yolov8m.pt"
  conf: 0.35
  iou: 0.5
  device: "auto"

runtime:
  target_fps: 10
  max_workers: 4
  heartbeat_interval: 30

paths:
  models: "models"
  zones: "data/zones"
  staff_gallery: "data/staff_gallery"
  reports: "reports"
  snapshots: "snapshots"
  logs: "logs"
```

## Logging

All modules use standard logging:

```python
from shared.logger import setup_logger

logger = setup_logger("module_name", "logs")
logger.info("Message")
logger.error("Error message")
```

Logs are stored in `logs/` with automatic rotation (10MB max per file, 5 backups).

## File Structure

```
project_count/
├── controller/
│   ├── __init__.py
│   ├── main.py              # Main GUI app
│   └── setup_wizard.py      # Setup wizard
│
├── runtime/
│   ├── __init__.py
│   ├── agent.py             # Runtime service
│   └── build_staff_db.py    # Staff DB builder
│
├── shared/
│   ├── __init__.py
│   ├── config.py            # Config management
│   └── logger.py            # Logging setup
│
├── data/
│   ├── config/
│   │   └── config.yaml      # Central config
│   ├── zones/
│   │   ├── zones_Camera_01.json
│   │   └── zones_Camera_02.json
│   └── staff_gallery/
│       ├── staff_db.json
│       └── [staff folders]
│
├── models/
│   ├── yolov8m.pt
│   └── yolov8n.pt
│
├── logs/                    # Application logs
├── reports/                 # CSV reports
├── snapshots/               # Frame captures
├── tests/                   # Tests (coming)
│
├── requirements.txt
├── MASTER_CHECKLIST.md
├── STRUCTURE.md
├── requirement_HGcam.md
└── [legacy files]
```

## Next Steps (Phase 2)

### Controller Features to Complete
- [ ] Real-time dashboard with Supabase integration
- [ ] Full camera management UI
- [ ] Zone editor in GUI
- [ ] Logs viewer with filtering
- [ ] System tray icon
- [ ] Auto-start service

### Runtime Features to Complete
- [ ] Full zone dwell tracking
- [ ] Event counting logic (haircut, wash, wait)
- [ ] Supabase event submission
- [ ] Heartbeat status updates
- [ ] RTSP reconnection watchdog
- [ ] Memory/resource guards

### Supabase Integration
- [ ] device_status table
- [ ] events table
- [ ] RLS policies
- [ ] Health check RPC

### Testing & Deployment
- [ ] Unit tests for core logic
- [ ] Integration tests
- [ ] PyInstaller build script
- [ ] Windows installer (NSIS)
- [ ] Service wrapper for Windows

## Key Design Decisions

1. **Config-Driven**: All settings in `config.yaml`, not hardcoded env vars
2. **Logging Standard**: Centralized setup in `shared/logger.py`
3. **Modular**: GUI, Runtime, and Shared are separate
4. **Backward Compatible**: Legacy env vars still supported
5. **Thread-Safe**: Multi-camera with proper locking

## Testing

To test the new structure:

```bash
# Test config loading
python -c "from shared.config import Config; c = Config('data/config/config.yaml'); print(c.get('project_name'))"

# Test logging
python -c "from shared.logger import setup_logger; l = setup_logger('test', 'logs'); l.info('Test message')"

# Launch GUI
python controller/main.py

# Run diagnostics
python -c "from runtime.build_staff_db import build_staff_db; build_staff_db()"
```

## Troubleshooting

### PySide6 Import Error
```bash
pip install PySide6 --upgrade
```

### YAML Import Error
```bash
pip install PyYAML
```

### YOLO Model Not Found
```bash
# Download model manually
cd models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
```

## Support

- Check `logs/` for detailed error messages
- Review `data/config/config.yaml` for settings
- See `STRUCTURE.md` for file organization
- Check `MASTER_CHECKLIST.md` for project progress
