"""
Resource Guards - Monitor and Limit Resource Usage

Prevents resource exhaustion by monitoring and enforcing limits on:
- FPS (frames per second)
- Memory usage
- Queue sizes
"""

import time
import threading
import psutil
import logging
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ResourceMetrics:
    """Current resource metrics"""
    fps: float
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    queue_sizes: Dict[str, int]
    timestamp: datetime


class ResourceGuard:
    """Monitor and enforce resource limits"""
    
    def __init__(
        self,
        logger: logging.Logger,
        max_fps: float = 30.0,
        max_memory_percent: float = 80.0,
        max_queue_size: int = 1000,
        check_interval: float = 5.0
    ):
        self.logger = logger
        self.max_fps = max_fps
        self.max_memory_percent = max_memory_percent
        self.max_queue_size = max_queue_size
        self.check_interval = check_interval
        
        # Current state
        self.frame_times: Dict[str, list] = {}
        self.queue_sizes: Dict[str, int] = {}
        self.lock = threading.Lock()
        
        # Process info
        self.process = psutil.Process()
        # Prime CPU sampler to make subsequent non-blocking cpu_percent() useful.
        try:
            self.process.cpu_percent(interval=None)
        except Exception:
            pass
        
        # Running state
        self.running = False
        self.check_thread: Optional[threading.Thread] = None

        # Cached metrics (updated by monitor loop and on-demand reads)
        self.metrics_lock = threading.Lock()
        self.last_metrics: Optional[ResourceMetrics] = None
        self.last_metrics_ts: float = 0.0
        
        # Alerts
        self.high_memory_alert = False
        self.high_cpu_alert = False
    
    def record_frame(self, camera_name: str):
        """Record frame processing time for FPS calculation"""
        with self.lock:
            if camera_name not in self.frame_times:
                self.frame_times[camera_name] = []
            
            self.frame_times[camera_name].append(time.time())
            
            # Keep only last 60 seconds of frames
            cutoff = time.time() - 60
            self.frame_times[camera_name] = [
                t for t in self.frame_times[camera_name] if t > cutoff
            ]
    
    def record_queue_size(self, queue_name: str, size: int):
        """Record queue size"""
        with self.lock:
            self.queue_sizes[queue_name] = size
    
    def get_fps(self, camera_name: str) -> float:
        """Get current FPS for camera"""
        with self.lock:
            if camera_name not in self.frame_times or not self.frame_times[camera_name]:
                return 0.0
            
            frames = self.frame_times[camera_name]
            if len(frames) < 2:
                return 0.0
            
            time_span = frames[-1] - frames[0]
            if time_span < 0.1:
                return 0.0
            
            fps = len(frames) / time_span
            return fps
    
    def get_metrics(self) -> ResourceMetrics:
        """Get current resource metrics (non-blocking CPU sample + safe FPS snapshot)."""
        # Get memory/cpu info from process
        mem_info = self.process.memory_info()
        mem_percent = self.process.memory_percent()
        cpu_percent = self.process.cpu_percent(interval=None)

        # Snapshot frame/queue data without nested lock recursion
        with self.lock:
            frame_snapshot = {cam: list(times) for cam, times in self.frame_times.items()}
            queues = dict(self.queue_sizes)

        fps_values = []
        for frames in frame_snapshot.values():
            if len(frames) < 2:
                continue
            time_span = frames[-1] - frames[0]
            if time_span < 0.1:
                continue
            fps_values.append(len(frames) / max(time_span, 1e-6))
        avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0.0

        metrics = ResourceMetrics(
            fps=avg_fps,
            cpu_percent=cpu_percent,
            memory_percent=mem_percent,
            memory_mb=mem_info.rss / (1024 * 1024),
            queue_sizes=queues,
            timestamp=datetime.now()
        )
        with self.metrics_lock:
            self.last_metrics = metrics
            self.last_metrics_ts = time.time()
        return metrics

    def get_latest_metrics(self, max_age_sec: float = 2.0) -> ResourceMetrics:
        """Return cached metrics when fresh, otherwise sample new metrics."""
        now_ts = time.time()
        with self.metrics_lock:
            if self.last_metrics is not None and (now_ts - self.last_metrics_ts) <= max(0.0, float(max_age_sec)):
                return self.last_metrics
        return self.get_metrics()
    
    def check_fps_limit(self, camera_name: str) -> bool:
        """Check if FPS is within limit"""
        fps = self.get_fps(camera_name)
        return fps <= self.max_fps
    
    def check_memory_limit(self) -> bool:
        """Check if memory is within limit"""
        mem_percent = self.process.memory_percent()
        return mem_percent <= self.max_memory_percent
    
    def check_queue_sizes(self) -> bool:
        """Check if all queues are within limit"""
        with self.lock:
            for size in self.queue_sizes.values():
                if size > self.max_queue_size:
                    return False
        return True
    
    def start(self):
        """Start resource monitoring thread"""
        if self.running:
            return
        
        self.running = True
        self.check_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.check_thread.start()
        self.logger.info("Resource Guard started")
    
    def stop(self):
        """Stop resource monitoring"""
        self.running = False
        if self.check_thread:
            self.check_thread.join(timeout=5)
        self.logger.info("Resource Guard stopped")
    
    def _monitor_loop(self):
        """Background monitoring thread"""
        while self.running:
            try:
                metrics = self.get_metrics()
                
                # Check memory
                if metrics.memory_percent > self.max_memory_percent:
                    if not self.high_memory_alert:
                        self.high_memory_alert = True
                        self.logger.warning(
                            f"HIGH MEMORY: {metrics.memory_mb:.1f}MB "
                            f"({metrics.memory_percent:.1f}%)"
                        )
                else:
                    if self.high_memory_alert:
                        self.high_memory_alert = False
                        self.logger.info(
                            f"Memory normal: {metrics.memory_mb:.1f}MB "
                            f"({metrics.memory_percent:.1f}%)"
                        )
                
                # Check CPU
                if metrics.cpu_percent > 80.0:
                    if not self.high_cpu_alert:
                        self.high_cpu_alert = True
                        self.logger.warning(f"HIGH CPU: {metrics.cpu_percent:.1f}%")
                else:
                    if self.high_cpu_alert:
                        self.high_cpu_alert = False
                        self.logger.info(f"CPU normal: {metrics.cpu_percent:.1f}%")
                
                # Check queue sizes
                oversized_queues = [
                    name for name, size in metrics.queue_sizes.items()
                    if size > self.max_queue_size
                ]
                if oversized_queues:
                    self.logger.warning(
                        f"OVERSIZED QUEUES: {', '.join(oversized_queues)}"
                    )
                
                time.sleep(self.check_interval)
            
            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {e}")
                time.sleep(self.check_interval)
    
    def should_throttle(self) -> bool:
        """Determine if processing should be throttled"""
        return (
            not self.check_memory_limit() or
            not self.check_queue_sizes()
        )
    
    def get_throttle_reason(self) -> Optional[str]:
        """Get reason for throttling if applicable"""
        if not self.check_memory_limit():
            return f"Memory high: {self.get_latest_metrics(max_age_sec=2.0).memory_percent:.1f}%"
        if not self.check_queue_sizes():
            return "Queue size exceeded"
        return None
    
    def reset_camera_fps(self, camera_name: str):
        """Reset FPS tracking for camera"""
        with self.lock:
            if camera_name in self.frame_times:
                self.frame_times[camera_name].clear()
