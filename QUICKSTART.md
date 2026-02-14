# Quick Start Guide

## 5-Minute Setup

### 1️⃣ Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # or: .venv\Scripts\activate (Windows)
pip install -r requirements.txt
```

### 2️⃣ Configure Cameras
Edit `data/config/config.yaml`:
```yaml
cameras:
  Camera_01:
    rtsp_url: "rtsp://admin:112113114@192.168.1.24:554/ch01/0"
    enabled: true
  Camera_02:
    rtsp_url: "rtsp://admin:112113114@192.168.1.83:554/ch01/0"
    enabled: true
```

### 3️⃣ Configure Supabase (Optional)
```yaml
supabase:
  url: "https://xxxx.supabase.co"
  key: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
  branch_code: "BRANCH_NAME"
```

### 4️⃣ Launch GUI
```bash
python controller/main.py
```

### 5️⃣ Run Setup Wizard
1. Configure Supabase (Step 1)
2. Add cameras (Step 2)
3. Edit zones (Step 3) - uses zone_picker.py
4. Build staff DB (Step 4)
5. Run diagnostics (Step 5)

---

## Files Reference

### Core Files

| File | Purpose | Run Command |
|------|---------|------------|
| `controller/main.py` | GUI Application | `python controller/main.py` |
| `runtime/agent.py` | Background Service | `python runtime/agent.py` |
| `runtime/build_staff_db.py` | Build Staff DB | `python runtime/build_staff_db.py` |
| `zone_picker.py` | Zone Editor | `python zone_picker.py` |

### Configuration

| File | Purpose |
|------|---------|
| `data/config/config.yaml` | Main configuration |
| `data/config/config.template.yaml` | Configuration template |
| `.env` | Environment variables |

### Data

| Directory | Purpose |
|-----------|---------|
| `data/zones/` | Zone JSON files |
| `data/staff_gallery/` | Staff photos |
| `data/staff_gallery/staff_db.json` | Staff embeddings database |

### Output

| Directory | Purpose |
|-----------|---------|
| `logs/` | Application logs |
| `reports/` | CSV reports |
| `snapshots/` | Frame captures |

---

## Logging

All log messages go to:
- **Console**: Real-time output
- **Files**: `logs/` directory with automatic rotation

Check logs:
```bash
tail -f logs/controller.log
tail -f logs/edge_agent.log
```

---

## Troubleshooting

### PySide6 Not Found
```bash
pip install --upgrade PySide6
```

### YOLO Model Not Found
```bash
cd models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
```

### RTSP Connection Failed
- Check camera IP and credentials in config.yaml
- Test with VLC: `vlc rtsp://...`
- Check network connectivity

### Zone Editor Won't Open
```bash
python zone_picker.py
```

### GUI Won't Start
```bash
python -c "from controller.main import main; main()"
```

---

## Common Commands

### Check Configuration
```bash
python -c "from shared.config import Config; import json; c = Config('data/config/config.yaml'); print(json.dumps(c.data, indent=2))"
```

### Test YOLO Model
```bash
python -c "from ultralytics import YOLO; m = YOLO('models/yolov8m.pt'); print('✓ YOLO loaded')"
```

### Test Logger
```bash
python -c "from shared.logger import setup_logger; l = setup_logger('test'); l.info('Test message'); print('✓ Logger works')"
```

### Build Staff Database
```bash
python runtime/build_staff_db.py --gallery data/staff_gallery --save-crops
```

### List Cameras
```bash
python -c "from shared.config import Config; c = Config('data/config/config.yaml'); print('\n'.join(c.get('cameras', {}).keys()))"
```

---

## Project Structure

```
project_count/
├── controller/              # GUI App
│   ├── main.py             # Main application
│   └── setup_wizard.py     # Setup wizard
│
├── runtime/                # Background Service
│   ├── agent.py            # Main runtime
│   └── build_staff_db.py   # Staff DB builder
│
├── shared/                 # Shared Utilities
│   ├── config.py           # Config system
│   └── logger.py           # Logging
│
├── data/
│   ├── config/
│   │   ├── config.yaml     # Configuration
│   │   └── config.template.yaml
│   ├── zones/              # Zone definitions
│   ├── staff_gallery/      # Staff photos
│   └── staff_db.json       # Embeddings
│
├── models/                 # YOLO weights
├── logs/                   # Application logs
├── reports/                # CSV reports
├── snapshots/              # Frames
│
├── requirements.txt        # Dependencies
├── MASTER_CHECKLIST.md    # Progress tracking
├── PHASE1_README.md       # Phase 1 guide
├── PHASE1_SUMMARY.md      # Summary
├── STRUCTURE.md           # Structure docs
└── [legacy scripts]        # Original files
```

---

## Next Steps

1. **Customize Config**
   - Edit camera URLs
   - Set Supabase credentials
   - Adjust YOLO thresholds

2. **Setup Zone**
   - Launch zone_picker.py
   - Draw zones for each camera
   - Save zones_Camera_XX.json

3. **Add Staff**
   - Create folders in data/staff_gallery/
   - Add staff photos (>5 images per person)
   - Run Build Staff DB

4. **Test Diagnostics**
   - Run Setup Wizard Step 5
   - Verify all checks pass
   - Fix any issues

5. **Start Service**
   - Click "Start Service" in GUI
   - Monitor logs
   - Check dashboard for status

---

## Support

For issues or questions:
1. Check logs in `logs/` directory
2. Review `data/config/config.yaml` settings
3. See `PHASE1_README.md` for detailed documentation
4. Check `MASTER_CHECKLIST.md` for progress

---

**Version**: 0.1.0 (Phase 1)  
**Last Updated**: 2026-02-12
