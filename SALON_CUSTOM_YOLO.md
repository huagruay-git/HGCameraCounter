# HGCameraCounter Custom YOLO (Salon)

## Architecture
- `scripts/prepare_salon_dataset.py`:
  - Build YOLO dataset from:
    - `data/performance_feedback/haircut` => `customer_haircut`
    - `data/staff_gallery/BARBER_UNIFORM` => `staff_barber`
    - `data/performance_feedback/customerwash` => `customer_wash`
  - Class balancing by oversampling minority classes
  - Hard-negative injection from `data/failed`, `data/unknown_by_admin` (empty label images)
  - Auto person-box generation using YOLO person detector (`--autobox-model`)
- `scripts/train_salon_yolo.py`:
  - Transfer learning from YOLOv8/YOLO11 checkpoint
  - Augmentation via `data/yolo_salon_dataset/augment.yaml`
  - Optional hard-negative mining from validation mistakes
- Runtime stack:
  - `detectors/yolo_salon_detector.py`: YOLO + ByteTrack inference wrapper
    - `tracker_mode` supports `bytetrack` or `deepsort` from one config file
    - if `deep_sort_realtime` is missing, runtime auto-falls back to `bytetrack`
  - `tracking/temporal_smoother.py`: temporal class smoothing
  - `tracking/reid_memory.py`: ID persistence across track breaks/cameras
  - `services/zone_manager.py`: zone hit detection
  - `services/counter_service.py`: anti-double-count + staff exclusion logic
  - `services/camera_runtime.py`: multi-camera RTSP real-time service
  - `ui/salon_ai_bridge.py`: PySide6 bridge (`Signal`) for GUI integration
  - `app/salon_ai_system.py`: high-level system builder from YAML config

## Counting Policy
- `staff_barber`:
  - excluded from customer events
  - staff lock window prevents immediate class-flip counting
- Haircut:
  - count when `customer_haircut` enters `chair_*` zone
  - protected by cooldown + zone memory
- Wash return anti-double-count:
  - same person (`global_id`) with prior haircut is tracked
  - when `customer_wash` returns to same `chair_*` zone:
    - do not increment haircut
    - increment wash only once per cycle/cooldown

## Commands
```bash
python scripts/prepare_salon_dataset.py
python scripts/train_salon_yolo.py --mine-hard-negatives
python scripts/run_salon_runtime.py --config data/config/salon_ai.runtime.yaml
```

## Config
- Runtime config template:
  - `data/config/salon_ai.runtime.template.yaml`
- Active runtime config:
  - `data/config/salon_ai.runtime.yaml`
- Controller GUI tab:
  - `controller/main.py` adds **Salon AI Runtime** tab for:
    - selecting runtime config
    - switching tracker mode (`bytetrack`/`deepsort`) and saving into config
    - start/stop runtime bridge and inspect realtime counters
