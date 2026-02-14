# HG Camera Counter - Project Structure

```
project_count/
├── controller/              # GUI Application (ยังไม่เริ่ม)
│   └── __init__.py
│
├── runtime/                 # Background Service (edge_agent.py)
│   └── __init__.py
│
├── shared/                  # Shared utilities
│   ├── __init__.py
│   ├── logger.py           # Standard logging
│   └── config.py           # Configuration management
│
├── data/
│   ├── config/
│   │   └── config.yaml     # Central config file
│   ├── zones/              # Zone definitions per camera
│   │   ├── zones_Camera_01.json
│   │   └── zones_Camera_02.json
│   └── staff_gallery/      # Staff photos & embeddings
│       ├── staff_db.json
│       └── _crops/
│
├── models/                 # YOLO weights
│   ├── yolov8m.pt
│   └── yolov8n.pt
│
├── reports/                # CSV reports
│   └── report_2026-02-12.csv
│
├── snapshots/              # Frame captures
│   └── 2026-02-12/
│
├── logs/                   # Application logs
│
├── tests/                  # Unit/integration tests
│
├── MASTER_CHECKLIST.md     # Project progress tracking
├── requirement_HGcam.md    # Detailed requirements
├── .env                    # Environment variables
└── [legacy scripts]        # To be refactored
    ├── edge_agent.py
    ├── agent_cam.py
    ├── build_staff_db.py
    └── zone_picker.py
```

## Quick Start

### 1. Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure
- Edit `data/config/config.yaml`
- Set SUPABASE_URL, SUPABASE_ANON_KEY in `.env`

### 3. Prepare Data
- Add camera zones in `data/zones/`
- Add staff photos in `data/staff_gallery/`

### 4. Run
```bash
# Runtime service
python runtime/agent.py

# Controller GUI (coming soon)
python controller/main.py
```

## Configuration

See `data/config/config.yaml` for all settings.

**Key Paths:**
- Models: `models/`
- Zones: `data/zones/`
- Staff DB: `data/staff_gallery/staff_db.json`
- Reports: `reports/`
- Logs: `logs/`
- Snapshots: `snapshots/`
