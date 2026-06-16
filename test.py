import cv2
import threading
import time
import os
import yaml
import json
from collections import deque
import numpy as np
import queue
from ultralytics import YOLO

# Shared Resources for Performance
shared_model = None
model_lock = threading.Lock()
verification_queue = queue.Queue() # Queue for Tier-2 workers

# --- Configuration Paths ---
CONFIG_PATH = "data/config/config.yaml"
MODEL_PATH = os.path.abspath("yolov8n.pt") # Use absolute path
SNAPSHOT_DIR = "snapshots"

def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"⚠️ Error loading {CONFIG_PATH}: {e}")
    return {}

config_data = load_config()

# Extract Camera Info and Zones from config
CAMERAS = {}
CAMERA_ZONES = {} # Store zones by camera name

def load_zones_for_camera(filepath):
    """Loads zones from a JSON file and returns a list of (name, type, polygon_points)"""
    if not filepath or not os.path.exists(filepath):
        return []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            zones = []
            for z in data:
                # points are usually in [x, y] normalized format
                pts = np.array(z.get('points', []), dtype=np.float32)
                zones.append({
                    'name': z.get('name', 'UNKNOWN'),
                    'type': z.get('type', 'UNKNOWN'),
                    'polygon': pts
                })
            return zones
    except Exception as e:
        print(f"⚠️ Error loading zones from {filepath}: {e}")
    return []

if config_data and 'cameras' in config_data:
    for cam_id, cam_info in config_data['cameras'].items():
        if cam_info.get('enabled', True):
            name = cam_id.replace('_', ' ')
            url = cam_info.get('rtsp_url')
            if url:
                CAMERAS[name] = url
                zone_file = cam_info.get('zones_file')
                if zone_file:
                    CAMERA_ZONES[name] = load_zones_for_camera(zone_file)

# YOLO Defaults
YOLO_CONF = 0.15 # Lower confidence for discovery sensitivity
YOLO_IOU = config_data.get('yolo', {}).get('iou', 0.5)
YOLO_IMGSZ = config_data.get('yolo', {}).get('imgsz', 640)
YOLO_CLASSES = [0] # 0: person ONLY (Standard YOLO v8m classes)

class VerificationWorker(threading.Thread):
    def __init__(self, model_path="yolov8m.pt"):
        super().__init__(daemon=True)
        self.model_path = model_path
        self.model = None

    def run(self):
        print(f"🔬 Tier-2 Worker: Loading high-precision model {self.model_path}...")
        try:
            self.model = YOLO(self.model_path)
            print(f"🔬 Tier-2 Worker: Ready and waiting for clips.")
        except Exception as e:
            print(f"❌ Tier-2 Worker Error: Failed to load model: {e}")
            return

        while True:
            job = verification_queue.get()
            clip_path = job['file']
            cam_name = job['cam']
            tid = job['tid']
            
            print(f"🔬 Tier-2 Worker: Verifying clip {clip_path} (Cam: {cam_name}, ID: {tid})")
            
            cap = cv2.VideoCapture(clip_path)
            frame_count = 0
            max_heads = 0
            best_frame = None
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                frame_count += 1
                
                # Sample every 5th frame for speed in Tier-2
                if frame_count % 5 == 0:
                    results = self.model(frame, verbose=False, conf=0.3)
                    if results and len(results[0].boxes) > 0:
                        person_count = len(results[0].boxes)
                        if person_count >= max_heads:
                            max_heads = person_count
                            best_frame = frame.copy()
            
            cap.release()
            
            if max_heads > 0 and best_frame is not None:
                os.makedirs(SNAPSHOT_DIR, exist_ok=True)
                snap_time = time.strftime("%H%M%S")
                snap_name = f"{SNAPSHOT_DIR}/verified_{cam_name.replace(' ','_')}_ID{tid}_{snap_time}.jpg"
                cv2.imwrite(snap_name, best_frame)
                print(f"✅ Tier-2 Verified: {clip_path} | Snapshot Saved: {snap_name}")
            else:
                print(f"✅ Tier-2 Done: {clip_path} | No persons verified (Max: {max_heads})")
            
            # --- Auto Cleanup ---
            try:
                if os.path.exists(clip_path):
                    os.remove(clip_path)
                    print(f"🗑️ Tier-2 Cleanup: Deleted temporary clip {clip_path}")
            except Exception as e:
                print(f"⚠️ Tier-2 Cleanup Error: {e}")

            verification_queue.task_done()

class CameraStream:
    def __init__(self, name, url, model_path=None, zones=None):
        self.name = name
        self.url = url
        self.model_path = model_path
        self.zones = zones
        self.cap = None
        self.frame = None
        self.display_frame = None 
        self.ret = False
        self.running = True
        self.status = "INITIALIZING"
        self.last_success_ts = 0
        self.fps_tracker = deque(maxlen=30)
        # Local tracking state
        self.track_data = {} # pid -> { 'first_seen': ts, 'zone': str, 'zone_start': ts, 'counted': bool }
        self.frame_count = 0
        self.last_boxes = [] # For flickering-free display
        self.rolling_buffer = deque(maxlen=150) # ~5-10 seconds buffer at 15-30 fps
        self.recording_event = None # { 'start_ts': ts, 'tid': id, 'frames_to_go': int }
        
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()

    def _connect(self):
        self.status = f"CONNECTING"
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self.cap.isOpened():
            self.status = "CONNECTED"
            self.last_success_ts = time.time()
        else:
            self.status = "FAILED"
            if self.cap: self.cap.release()
            self.cap = None

    def update(self):
        self._connect()
        while self.running:
            if self.cap and self.cap.isOpened():
                success, frame = self.cap.read()
                if success:
                    now = time.time()
                    self.ret = True
                    self.frame = frame
                    self.last_success_ts = now
                    self.fps_tracker.append(now)
                    self.frame_count += 1
                    # Update Rolling Buffer
                    self.rolling_buffer.append(frame.copy())
                    # Process frame
                    tmp_frame = frame.copy()
                    h_img, w_img = tmp_frame.shape[:2]
                    
                    if shared_model is not None:
                        # Draw Zones every frame for visibility
                        for z in (self.zones or []):
                            poly_px = (z['polygon'] * [w_img, h_img]).astype(np.int32)
                            color = (255, 255, 0) # Cyan-ish
                            cv2.polylines(tmp_frame, [poly_px], isClosed=True, color=color, thickness=2)
                            cv2.putText(tmp_frame, z['name'], (poly_px[0][0], poly_px[0][1] - 5), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                        # Optimization: Skip inference on odd frames to balance load
                        results = []
                        if self.frame_count % 2 == 0:
                            acquired = model_lock.acquire(blocking=True)
                            if acquired:
                                try:
                                    results = shared_model.track(tmp_frame, persist=True, tracker="bytetrack.yaml", 
                                                             conf=YOLO_CONF, iou=YOLO_IOU, imgsz=YOLO_IMGSZ, 
                                                             classes=YOLO_CLASSES, agnostic_nms=True, verbose=False)
                                    if results and len(results[0].boxes) > 0:
                                        print(f"🔍 [{self.name}] Found {len(results[0].boxes)} detections.")
                                except Exception as e:
                                    print(f"❌ Inference Error: {e}")
                                finally:
                                    model_lock.release()
                        
                        active_ids = set()
                        current_boxes = [] 
                        if results:
                            for r in results:
                                if not r.boxes: continue
                                for box in r.boxes:
                                    cls = int(box.cls[0])
                                    conf = float(box.conf[0])
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                                    tid = int(box.id[0]) if box.id is not None else -1

                                    if (x2-x1) > w_img * 0.95 or (y2-y1) > h_img * 0.95: continue
                                    cx, cy = (x1 + x2) / 2 / w_img, (y1 + y2) / 2 / h_img
                                    is_head = (cls == 1)
                                    label = "head" if is_head else ("person" if cls == 0 else "staff")
                                    color = (0, 255, 0)

                                    if is_head:
                                        bw, bh = (x2 - x1) / w_img, (y2 - y1) / h_img
                                        if bw * bh > 0.15 or bw * bh < 0.0005: continue
                                        if bw > 0.4 or bh > 0.4: continue
                                    
                                    current_zone = "NONE"
                                    if cls == 2:
                                        label = "staff"; color = (255, 0, 255)
                                    else:
                                        for z in (self.zones or []):
                                            if is_point_in_zone((cx, cy), z['polygon']):
                                                current_zone = z['type']
                                                if "WAIT" in current_zone.upper():
                                                    label = "customer_wait"; color = (0, 165, 255)
                                                elif "WASH" in current_zone.upper():
                                                    label = "customer_wash"; color = (255, 200, 0)
                                                break

                                    current_boxes.append({
                                        'bbox': (x1, y1, x2, y2), 'label': label, 'color': color, 
                                        'conf': conf, 'cx': cx, 'cy': cy, 'tid': tid, 'zone': current_zone
                                    })

                                    if tid != -1:
                                        active_ids.add(tid)
                                        if tid not in self.track_data:
                                            self.track_data[tid] = {'first_seen': now, 'zone': current_zone, 'zone_start': now, 'is_staff': (cls==2)}
                                        d = self.track_data[tid]
                                        if d['zone'] != current_zone:
                                            d['zone'] = current_zone; d['zone_start'] = now

                            if current_boxes:
                                self.last_boxes = current_boxes

                        # Draw detections
                        for b in self.last_boxes:
                            x1, y1, x2, y2 = b['bbox']
                            color, label, conf, tid = b['color'], b['label'], b['conf'], b['tid']
                            dwell = now - self.track_data[tid]['zone_start'] if tid != -1 and tid in self.track_data else 0
                            cv2.rectangle(tmp_frame, (x1, y1), (x2, y2), color, 2)
                            id_str = f"#{tid} " if tid != -1 else "NEW "
                            cv2.putText(tmp_frame, f"{id_str}{label} {conf:.2f} {dwell:.0f}s", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        ai_status = "ON" if shared_model is not None else "OFF"
                        summary_txt = f"AI: {ai_status} | Active IDs: {len(active_ids)} | In-Zone: {sum(1 for d in self.track_data.values() if not d.get('is_staff') and d.get('zone') != 'NONE')}"
                        cv2.rectangle(tmp_frame, (5, 5), (500, 35), (0,0,0), -1)
                        cv2.putText(tmp_frame, summary_txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                        to_delete = [tid for tid, d in self.track_data.items() if (now - d.get('first_seen', now)) > 60 and tid not in active_ids]
                        for tid in to_delete: del self.track_data[tid]

                        self.display_frame = tmp_frame
                        self.status = "LIVE"

                        # --- Tier-1 Trigger Logic (Recording) ---
                        for tid, d in self.track_data.items():
                            if not d.get('is_staff') and d.get('zone') != 'NONE':
                                dwell = now - d.get('zone_start', now)
                                last_clip_ts = d.get('last_clip_ts', 0)
                                if dwell > 2.0 and self.recording_event is None and (now - last_clip_ts) > 60:
                                    print(f"🎬 TRIGGER: Person #{tid} in {d['zone']} for {dwell:.1f}s. Recording clip...")
                                    self.recording_event = {
                                        'start_ts': now, 'tid': tid, 'frames_to_go': 60,
                                        'buffer_snapshot': list(self.rolling_buffer)
                                    }
                                    d['last_clip_ts'] = now
                                    break

                        if self.recording_event:
                            self.recording_event['buffer_snapshot'].append(frame.copy())
                            self.recording_event['frames_to_go'] -= 1
                            if self.recording_event['frames_to_go'] <= 0:
                                self._save_trigger_clip()
                                self.recording_event = None
                    else: # This else belongs to 'if shared_model is not None:'
                        self.display_frame = frame.copy()
                        self.status = "LIVE (No AI)"
                else: # This else belongs to 'if success:'
                    if time.time() - self.last_success_ts > 5.0:
                        if self.cap: self.cap.release()
                        self.cap = None
                        time.sleep(2)
                        self._connect()
            else: # This else belongs to 'if self.cap and self.cap.isOpened():'
                time.sleep(1)
                self._connect()
            time.sleep(0.005)

    def _save_trigger_clip(self):
        if not self.recording_event: return
        os.makedirs("temp_clips", exist_ok=True)
        ts_str = time.strftime("%Y%m%d_%H%M%S")
        tid = self.recording_event['tid']
        fname = f"temp_clips/trigger_{self.name.replace(' ','_')}_ID{tid}_{ts_str}.mp4"
        
        frames = self.recording_event['buffer_snapshot']
        if not frames: return
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(fname, fourcc, 15.0, (w, h))
        for f in frames: out.write(f)
        out.release()
        print(f"✅ Clip Saved for Tier-2 Verification: {fname}")
        verification_queue.put({
            'file': fname, 'cam': self.name, 'tid': tid, 'zone': self.track_data.get(tid, {}).get('zone', 'N/A')
        })

    def get_display_frame(self):
        return self.ret, self.display_frame

    def get_fps(self):
        if len(self.fps_tracker) < 2: return 0.0
        return (len(self.fps_tracker) - 1) / (self.fps_tracker[-1] - self.fps_tracker[0] + 1e-6)

    def stop(self):
        self.running = False
        if self.cap: self.cap.release()

def is_point_in_zone(point, polygon):
    """point: (x, y) normalized, polygon: ndarray of (x, y) normalized"""
    return cv2.pointPolygonTest(polygon, point, False) >= 0

def main():
    print("==============================")
    print(" HG Camera Counter Test Tool")
    print("==============================")
    print(f"Config: {CONFIG_PATH}")
    print(f"Targeting: Head (1), Staff (2)")
    print("------------------------------")
    print("1. Test Model (Refined Labels)")
    print("2. Test Camera Connection")
    print("3. Live View & Snapshot")
    print("q. Exit")
    choice = input("Select mode: ").strip().lower()

    if choice == 'q': return

    if choice == '1' or choice == '3':
        print(f"⏳ Initializing Shared AI Model ({MODEL_PATH})...")
        try:
            global shared_model
            shared_model = YOLO(MODEL_PATH)
            print(f"✅ Shared AI Model ready.")
            # Start Tier-2 Worker
            worker = VerificationWorker("yolov8m.pt")
            worker.start()
        except Exception as e:
            print(f"❌ Critical Error loading model: {e}")
            if choice == '1': return # Exit if testing model

    if choice == '3' and not os.path.exists(SNAPSHOT_DIR):
        os.makedirs(SNAPSHOT_DIR)

    streams = []
    for name, url in CAMERAS.items():
        if not url: continue
        exists = os.path.exists(MODEL_PATH)
        print(f"🚀 Starting {name}... (AI Model found: {exists})")
        
        curr_model_path = MODEL_PATH if (choice in ['1', '3'] and exists) else None
        if curr_model_path is None and choice in ['1', '3']:
            print(f"⚠️ Warning: Model path set to None for {name} (File exists: {exists})")
            
        curr_zones = CAMERA_ZONES.get(name, [])
        streams.append(CameraStream(name, url, model_path=curr_model_path, zones=curr_zones))

    while True:
        all_frames = []
        for s in streams:
            success, frame = s.get_display_frame()
            if success and frame is not None:
                display_frame = frame
                h_img, w_img = display_frame.shape[:2]
                
                # Overlay Status
                fps = s.get_fps()
                status_text = f"[{s.name}] {s.status} | FPS: {fps:.1f}"
                cv2.putText(display_frame, status_text, (10, h_img - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                all_frames.append(display_frame)
            else:
                err_f = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(err_f, f"[{s.name}] FAILED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                all_frames.append(err_f)

        if all_frames:
            resized = [cv2.resize(f, (640, 480)) for f in all_frames]
            montage = np.hstack(resized) if len(resized) > 1 else resized[0]
            cv2.imshow("HG Counter Refined", montage)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('s') and (choice == '3' or choice == '1'):
            ts = time.strftime("%Y%m%d_%H%M%S")
            for s in streams:
                ret, snap = s.get_display_frame() 
                if ret and snap is not None:
                    fname = f"{SNAPSHOT_DIR}/{s.name.replace(' ', '_')}_{ts}.jpg"
                    cv2.imwrite(fname, snap)
                    print(f"📸 Saved: {fname}")

    for s in streams: s.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

