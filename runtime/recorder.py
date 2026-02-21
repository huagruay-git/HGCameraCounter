
import os
import time
import subprocess
import threading
import logging
import signal
from typing import Dict, Optional
from datetime import datetime
from pathlib import Path

CONFIG_PATH = "data/config/config.yaml"
# We'll use the shared config loader
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.config import Config
from shared.logger import setup_logger

class RTSPRecorder:
    def __init__(self, camera_name: str, rtsp_url: str, output_dir: str, segment_time: int = 60):
        self.camera_name = camera_name
        self.rtsp_url = rtsp_url
        self.output_dir = Path(output_dir) / camera_name
        self.segment_time = segment_time
        self.process: Optional[subprocess.Popen] = None
        self.running = False
        self.logger = logging.getLogger(f"Recorder.{camera_name}")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def start(self):
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        self.logger.info(f"Recorder started for {self.camera_name}")

    def stop(self):
        self.running = False
        self._stop_ffmpeg()
        self.logger.info(f"Recorder stopped for {self.camera_name}")

    def _stop_ffmpeg(self):
        if self.process:
            self.logger.info("Stopping FFmpeg process...")
            # Send SIGTERM first
            self.process.terminate()
            try:
                # Wait briefly for graceful exit (close file propertly)
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.logger.warning("FFmpeg did not exit gracefully, killing...")
                self.process.kill()
                try:
                    self.process.wait(timeout=1)
                except:
                    pass
            self.process = None
        # Double check for orphans
        self._kill_orphans()

    def _kill_orphans(self):
        """Force kill any lingering ffmpeg processes for this camera."""
        try:
            # Find pids handling this RTSP URL
            # We match the URL in the command line using pgrep -f
            # Use shlex to safely quote the URL to avoid shell injection or grep errors
            import shlex
            safe_url = shlex.quote(self.rtsp_url)
            
            # Note: pgrep -f matches the full command line.
            cmd = f"pgrep -f {safe_url}"
            try:
                pids = subprocess.check_output(cmd, shell=True).decode().split()
            except subprocess.CalledProcessError:
                # No process found
                return

            my_pid = str(os.getpid())
            for pid in pids:
                if pid == my_pid: continue
                # Kill it
                self.logger.warning(f"Killing orphan ffmpeg pid {pid} for {self.camera_name}")
                try:
                    subprocess.call(["kill", "-9", pid])
                except Exception:
                    pass
        except Exception as e:
            self.logger.error(f"Error killing orphans: {e}")

    def _monitor_loop(self):
        while self.running:
            if not self.process or self.process.poll() is not None:
                # Process not running, start it
                self._start_ffmpeg()
            
            time.sleep(5)

    def _start_ffmpeg(self):
        # Ensure clean slate
        self._stop_ffmpeg()
        self._kill_orphans()
        
        # Pattern for segment filenames: YYYYMMDD_HHMMSS.mp4
        filename_pattern = str(self.output_dir / "%Y%m%d_%H%M%S.mp4")
        
        cmd = [
            "ffmpeg",
            "-y",
            "-rtsp_transport", "tcp",
            "-i", self.rtsp_url,
            "-c:v", "copy",  # Copy video stream
            "-an",           # Disable audio (pcm_mulaw causes mp4 container errors)
            "-f", "segment",
            "-segment_time", str(self.segment_time),
            "-segment_format", "mp4",
            "-reset_timestamps", "1",
            "-strftime", "1",
            filename_pattern
        ]
        
        self.logger.info(f"Starting FFmpeg: {' '.join(cmd)}")
        
        # Log ffmpeg output for debugging
        log_file = self.output_dir / "ffmpeg.log"
        self.log_handle = open(log_file, "a")
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=self.log_handle,
                stderr=subprocess.STDOUT, 
                stdin=subprocess.DEVNULL
            )
        except Exception as e:
            self.logger.error(f"Failed to start FFmpeg: {e}")
            time.sleep(10) # Backoff

class ServiceRecorder:
    def __init__(self):
        self.config = Config(CONFIG_PATH)
        self.logger = setup_logger("recorder", "logs")
        self.recorders: Dict[str, RTSPRecorder] = {}
        
    def start(self):
        self.logger.info("Starting ServiceRecorder...")
        cameras = self.config.get("cameras", {})
        
        for name, cfg in cameras.items():
            if not cfg.get("enabled", True):
                continue
                
            url = cfg.get("rtsp_url")
            if url:
                rec = RTSPRecorder(
                    name, 
                    url, 
                    output_dir="data/recordings",
                    segment_time=60
                )
                rec.start()
                self.recorders[name] = rec
                
    def stop(self):
        for rec in self.recorders.values():
            rec.stop()

if __name__ == "__main__":
    svc = ServiceRecorder()
    
    def handle_signal(signum, frame):
        print(f"Received signal {signum}, stopping...")
        svc.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    try:
        svc.start()
        # Use signal.pause() or a loop that can be interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        svc.stop()
    except Exception as e:
        print(f"Recorder error: {e}")
        svc.stop()
