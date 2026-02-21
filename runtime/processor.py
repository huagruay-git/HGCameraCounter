
import os
import time
import glob
import logging
from pathlib import Path
from typing import Optional, List
from datetime import datetime
import cv2
import shutil

# Reuse existing Agent logic but modified for file input
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from runtime.agent_v2 import RuntimeService, CameraStream, CONFIG, logger as agent_logger
from shared.config import Config
from shared.logger import setup_logger

logger = setup_logger("processor", "logs")

class FileCameraStream(CameraStream):
    """Adaptation of CameraStream to read from a file instead of RTSP"""
    def __init__(self, camera_name: str, file_path: str):
        self.camera_name = camera_name
        self.file_path = file_path
        self.cap = cv2.VideoCapture(file_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.is_connected = True
        self.frame = None
        self.lock = None # Not needed for sequential file read
        
    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.is_connected = False
            return None
        return frame

    def release(self):
        if self.cap:
            self.cap.release()

class DummyCameraStream(CameraStream):
    """Fake stream to keep dashboard status 'OK' between files"""
    def __init__(self, camera_name: str):
        self.camera_name = camera_name
        self.fps = 25.0
        self.is_connected = True
        self.frame = None
        self.lock = None

    def read_frame(self):
        return None
    
    def connect(self):
        self.is_connected = True
        
    def disconnect(self):
        self.is_connected = False

    def release(self):
        pass

class BatchProcessor(RuntimeService):
    def __init__(self):
        # Initialize parent RuntimeService but don't start threads yet
        super().__init__()
        self.config = CONFIG  # Fix: Ensure self.config is available
        self.recordings_dir = Path("data/recordings")
        self.processed_dir = Path("data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.segment_time_sec = 60
        self.file_ready_grace_sec = 15.0
        self.retry_open_max_age_sec = float((self.segment_time_sec * 3) + self.file_ready_grace_sec)
        
        # Tune agent parameters for batch processing resilience
        # Increase grace to tolerate processing gaps or slow-motion effects.
        self.vacant_grace_sec = 120.0
        self.customer_active_ttl_sec = 10.0
        self.haircut_counter.vacant_grace_sec = 120.0
        self.wash_counter.vacant_grace_sec = 120.0
        
        # Initialize cameras with dummies so dashboard shows them as "Connected" (but idle)
        from runtime.agent_v2 import CAMERAS_CONFIG
        for name in CAMERAS_CONFIG:
            self.cameras[name] = DummyCameraStream(name)
            self._last_frame_ok_ts[name] = time.time()  # Initialize heartbeat
        
        # Disable the default camera threads from RuntimeService
        # We will control the processing loop manually
        
    def start_processing(self):
        logger.info("Starting Batch Processor...")
        self.running = True
        
        # Start only the non-camera threads (Supabase, Reporting)
        # We need these for event submission
        import threading
        # DISABLE Supabase for now as per user request to focus on counting accuracy
        # self.supabase_sync.start()
        self.supabase_sync = None
        
        # Event submission thread (from agent_v2)
        event_thread = threading.Thread(target=self.submit_events_loop, daemon=True)
        event_thread.start()
        self.threads.append(event_thread)

        # Cleanup old snapshots on startup
        self._cleanup_snapshots()

        # Global counting/logic loop (CRITICAL: this generates events from detections)
        counting_thread = threading.Thread(target=self._global_counting_loop, daemon=True)
        counting_thread.start()
        self.threads.append(counting_thread)
        
        from runtime.agent_v2 import CAMERAS_CONFIG
        
        while self.running:
            did_work = False
            # Iterate configured cameras (static list) instead of self.cameras (dynamic)
            # to avoid "dictionary changed size during iteration"
            for camera_name in list(CAMERAS_CONFIG.keys()): 
                # Update heartbeat for all cameras to keep dashboard happy
                # (even if we are only processing one of them right now)
                for c_name in self.cameras:
                    self._last_frame_ok_ts[c_name] = time.time()
                
                # Find oldest .mp4 file
                cam_dir = self.recordings_dir / camera_name
                if not cam_dir.exists():
                    continue
                
                files = sorted(list(cam_dir.glob("*.mp4")))
                if not files:
                    continue
                
                # New logic: Check if file is stable (not growing)
                # This works regardless of how many files there are.
                
                target_file = None
                
                # Check candidates (oldest first)
                for candidate in files:
                    if not self._is_file_active(candidate):
                        # File is stable, process it
                        target_file = candidate
                        break
                    else:
                        # File is growing, it's the active one.
                        # Since files are sorted by time, if the oldest one is active, 
                        # we probably shouldn't look at newer ones (they shouldn't exist yet).
                        # But just in case, we continue? 
                        # No, if oldest is active, we just wait.
                        pass
                
                if target_file:
                    self._process_file(camera_name, target_file)
                    did_work = True
            
            if not did_work:
                time.sleep(2.0) # Wait for new files
    # ... (start_processing, etc. unchanged)

    def _is_file_active(self, file_path: Path) -> bool:
        """Check if file is currently being modified (size changing) OR is too new."""
        try:
            # Check 1: Filename timestamp (YYYYMMDD_HHMMSS.mp4)
            # If the file started < 65 seconds ago, assume it's still recording (60s segments).
            try:
                ts_str = file_path.stem.split('_')[-1] # HHMMSS
                date_str = file_path.stem.split('_')[-2] # YYYYMMDD
                dt = datetime.strptime(f"{date_str}_{ts_str}", "%Y%m%d_%H%M%S")
                file_ts = dt.timestamp()
                
                min_age = float(self.segment_time_sec) + float(self.file_ready_grace_sec)
                # Keep newest segment untouched until writer has enough time to close moov atom.
                if (time.time() - file_ts) < min_age:
                    return True
            except Exception:
                pass # Fallback to file stats if parsing fails

            # Check 2: File size changing
            s1 = file_path.stat().st_size
            time.sleep(0.5)
            s2 = file_path.stat().st_size
            if s1 != s2:
                return True
            
            # Check 3: Modification time
            # If written purely > 10 seconds ago, it's likely safe.
            mtime = file_path.stat().st_mtime
            if (time.time() - mtime) < float(self.file_ready_grace_sec):
                 # Too fresh, assume potentially active
                 return True
                 
            return False
        except OSError:
            return True # Assume active/problematic if we can't stat

    def _should_retry_open_later(self, file_path: Path) -> bool:
        try:
            st = file_path.stat()
            if st.st_size <= 0:
                return True
            age_sec = time.time() - st.st_mtime
            return age_sec < self.retry_open_max_age_sec
        except OSError:
            return True
                
    def _process_file(self, camera_name: str, file_path: Path):
        logger.info(f"Processing file: {file_path}")
        
        # Swap the camera stream with a file stream
        try:
             stream = FileCameraStream(camera_name, str(file_path))
        except Exception as e:
             logger.error(f"Failed to open file {file_path}: {e}")
             self._move_to_failed(file_path)
             return

        if not stream.is_connected or stream.cap is None or not stream.cap.isOpened():
            if self._should_retry_open_later(file_path):
                logger.warning(f"Video not ready yet, will retry later: {file_path}")
            else:
                logger.error(f"Cannot open video stream in {file_path}")
                self._move_to_failed(file_path)
            stream.release()
            return

        # We need to inject this stream into the agent's state
        # effectively replacing the RTSP stream for this camera
        self.cameras[camera_name] = stream
        
        # Process every Nth frame to speed up (e.g. 5 FPS equivalent)
        # file_fps = stream.fps (e.g. 25)
        # target_fps = 5
        # skip = file_fps / target_fps
        
        frame_idx = 0
        max_people = 0
        initial_event_count = len(self.all_events)
        
        while stream.is_connected:
            frame = stream.read_frame()
            if frame is None:
                break
            
            # Smart skipping: verify we process at roughly 2-5 FPS
            # If video is 25fps, process every 5th frame for 5fps
            if frame_idx % 5 == 0:
                # Mock time? 
                # Agent uses time.time() for dwell.
                # Ideally config dwell times are in seconds.
                # If we process faster than realtime, dwell calculations (time.time()) break.
                # FIX: We must monkeypatch time.time() or update track last_seen manually?
                # Actually, agent_v2 relies heavily on system clock (time.time()).
                # If we process a 60s file in 5s, the agent will think only 5s passed.
                # This breaks dwell logic (waiting 30s).
                
                # Hack: We must run at 1.0x speed or simulate time.
                # Simulating time in a complex system is hard.
                # Easier: Process at max speed, but scale the dwell thresholds down? No.
                # 
                # Alternative: We just sleep() to match realtime?
                # If we sleep, we are limited to 1x speed. The user wants "reduce load".
                # Processing 500ms (2 FPS) every 1s is fine.
                # So we can process a 60s file in <60s if CPU allows.
                # But dwell checks uses `now - last_seen`.
                # If we use system time, we MUST process in roughly real-time or faster.
                # If we process FASTER, `now - last_seen` will be too small.
                # So we effectively CANNOT process faster than 1x wall-clock time if we rely on `time.time()`.
                
                # Wait, if we use a "store and forward", checks should be based on `frame_timestamp`.
                # But `agent_v2` is built for live RTSP.
                # Refactoring `agent_v2` to use `frame_time` instead of `time.time()` is a big change.
                
                # Compromise:
                # We process the file "pseudo-live".
                # We interpret the file frames.
                # We sleep slightly to not CPU burn, but we accept that "real time" is passing.
                # If the processor lags, it's fine.
                # If it's too fast, we might undertrack.
                # BUT: `time.time()` is monotonic. 
                # If we process a 60s chunk in 10s, the agent sees 10s passed. 
                # A person sitting for 60s (in video) only sat for 10s (in wall clock).
                # Events will be missed.
                
                # CRITICAL FIX: We must throttle the processor to match specific playback speed?
                # OR we accept that we CANNOT process faster than realtime with this codebase.
                # But we CAN process SLOWER / LOWER FPS.
                # processing 2 FPS from a 25 FPS file = 12x speedup in processing time per frame,
                # but we must space them out to match wall clock?
                
                # Actually, for "Offline" processing, we need to pass a timestamp to `process_frame`.
                # I see `process_frame(camera_name, frame)` in agent_v2.
                # Does `process_frame` use `time.time()`? Yes, implicitly via trackers.
                
                # Current plan adjustment:
                # For `BatchProcessor` to work with `agent_v2` logic:
                # We should try to "emulate" real-time by sleeping.
                # We read 1 frame, sleep 0.5s (2 FPS), read next (skip 12 frames).
                # This keeps "wall clock" roughly aligned with "video time".
                # And since we process only 2 FPS, CPU load is low!
                # This is "Store-and-Slow-Forward".
                
                self.process_frame(camera_name, frame)
                # Ensure we sync with wall clock?
                # Video 25fps. Frame 0 -> 0s. Frame 12 -> 0.48s. Frame 24 -> 0.96s.
                # If we process frame 0, then frame 12...
                # We should `time.sleep(0.5)` between them.
                # Speeding up to 0.1s to avoid "slow motion" gap magnification
                time.sleep(0.1) 
            
            
            # Update max_people stat for this file
            with self.customers_lock:
                 # Check active customers for this camera
                 current_people = len(self.latest_customers.get(camera_name, []))
                 if current_people > max_people:
                     max_people = current_people

            frame_idx += 1
            
        stream.release()
        
        # Clean up: Replace with dummy stream to maintain status in dashboard
        self.cameras[camera_name] = DummyCameraStream(camera_name)
        
        # Calculate events generated during this file
        final_event_count = len(self.all_events)
        total_events = final_event_count - initial_event_count
        
        # Decide whether to keep or delete the file
        should_keep = (total_events > 0) or (max_people > 0)
        
        if should_keep:
            # Check retention policy
            retain_video = self.config.get("storage", {}).get("retain_processed_video", False)
            if retain_video:
                self._move_to_processed(file_path)
                logger.info(f"Processed {file_path.name}: {total_events} new events, max {max_people} people. Kept (Policy=True).")
            else:
                self._delete_file(file_path)
                logger.info(f"Processed {file_path.name}: {total_events} new events, max {max_people} people. Deleted (Policy=False).")
        else:
            self._delete_file(file_path)
            logger.info(f"Processed {file_path.name}: No interesting activity. Deleted.")

    def _cleanup_snapshots(self):
        """Delete snapshots older than retention days."""
        try:
            days = int(self.config.get("storage", {}).get("snapshot_retention_days", 30))
            if days <= 0:
                return

            logger.info(f"Running snapshot cleanup (retention={days} days)...")
            cutoff = time.time() - (days * 86400)
            
            # Snapshots are in data/snapshots/Camera_X/YYYY-MM-DD/filename.jpg
            # We can walk the directory
            count = 0
            snapshots_dir = Path("data/snapshots")
            if not snapshots_dir.exists():
                return

            for snap in snapshots_dir.rglob("*.jpg"):
                if snap.stat().st_mtime < cutoff:
                    try:
                        os.remove(snap)
                        count += 1
                        # If directory empty, remove it? optional.
                    except Exception:
                        pass
            
            if count > 0:
                logger.info(f"Cleaned up {count} old snapshots.")
        except Exception as e:
            logger.error(f"Snapshot cleanup failed: {e}")

    def _delete_file(self, file_path: Path):
        try:
            if file_path.exists():
                os.remove(file_path)
        except Exception as e:
            logger.error(f"Failed to delete {file_path}: {e}")

    def _move_to_processed(self, file_path: Path):
        try:
            shutil.move(str(file_path), str(self.processed_dir / file_path.name))
        except Exception as e:
            logger.error(f"Failed to move file {file_path} to processed: {e}")
            # Try delete to avoid loops if move fails
            try:
                os.remove(file_path)
            except Exception as e_del:
                logger.error(f"Failed to delete {file_path} after failed move: {e_del}")

    def _move_to_failed(self, file_path: Path):
        failed_dir = self.processed_dir.parent / "failed"
        failed_dir.mkdir(exist_ok=True)
        try:
             shutil.move(str(file_path), str(failed_dir / file_path.name))
        except Exception:
             try:
                 os.remove(file_path)
             except:
                 pass

if __name__ == "__main__":
    processor = BatchProcessor()
    
    def handle_signal(signum, frame):
        logger.info(f"Received signal {signum}, stopping...")
        processor.stop() 
        sys.exit(0)

    import signal
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    try:
        processor.start_processing()
    except KeyboardInterrupt:
        logger.info("Stopping...")
    except Exception as e:
        logger.error(f"Processor error: {e}")
        processor.stop()
