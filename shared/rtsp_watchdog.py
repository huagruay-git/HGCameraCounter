"""
RTSP Watchdog - Automatic Camera Reconnection

Monitors camera health and automatically reconnects on failure
with exponential backoff and health checks.
"""

import time
import threading
import logging
from typing import Dict, Callable, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class CameraHealth(Enum):
    """Camera health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    ATTEMPTING_RECONNECT = "attempting_reconnect"


@dataclass
class CameraHealthInfo:
    """Camera health information"""
    camera_name: str
    status: CameraHealth
    last_seen: datetime
    consecutive_failures: int = 0
    last_error: Optional[str] = None
    reconnect_attempts: int = 0
    last_reconnect_attempt: Optional[datetime] = None
    uptime_percent: float = 100.0
    
    def mark_success(self):
        """Mark camera as successfully connected"""
        self.status = CameraHealth.HEALTHY
        self.consecutive_failures = 0
        self.last_seen = datetime.now()
        self.last_error = None
    
    def mark_failure(self, error: str):
        """Mark camera as failed"""
        self.consecutive_failures += 1
        self.last_error = error
        
        if self.consecutive_failures > 5:
            self.status = CameraHealth.OFFLINE
        elif self.consecutive_failures > 2:
            self.status = CameraHealth.DEGRADED
        else:
            self.status = CameraHealth.ATTEMPTING_RECONNECT
    
    def get_backoff_seconds(self) -> float:
        """Calculate exponential backoff time"""
        # Start with 1 second, exponentially increase up to 60 seconds
        backoff = min(2 ** (self.consecutive_failures - 1), 60)
        return backoff


class RTSPWatchdog:
    """Monitor and manage RTSP camera connections"""
    
    def __init__(
        self,
        logger: logging.Logger,
        max_retries: int = 10,
        initial_backoff: float = 1.0,
        max_backoff: float = 60.0
    ):
        self.logger = logger
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        
        # Health tracking
        self.health_info: Dict[str, CameraHealthInfo] = {}
        self.lock = threading.Lock()
        
        # Callbacks
        self.on_camera_offline: Optional[Callable[[str], None]] = None
        self.on_camera_online: Optional[Callable[[str], None]] = None
        self.on_reconnect_attempt: Optional[Callable[[str, int], None]] = None
    
    def register_camera(self, camera_name: str):
        """Register camera for monitoring"""
        with self.lock:
            if camera_name not in self.health_info:
                self.health_info[camera_name] = CameraHealthInfo(
                    camera_name=camera_name,
                    status=CameraHealth.HEALTHY,
                    last_seen=datetime.now()
                )
                self.logger.info(f"Watchdog: Registered camera {camera_name}")
    
    def mark_frame_received(self, camera_name: str):
        """Mark successful frame reception"""
        with self.lock:
            if camera_name in self.health_info:
                info = self.health_info[camera_name]
                was_offline = info.status in (CameraHealth.OFFLINE, CameraHealth.DEGRADED)
                
                info.mark_success()
                
                if was_offline:
                    self.logger.info(f"Watchdog: {camera_name} back online")
                    if self.on_camera_online:
                        self.on_camera_online(camera_name)
    
    def mark_frame_failed(self, camera_name: str, error: str):
        """Mark failed frame reception"""
        with self.lock:
            if camera_name in self.health_info:
                info = self.health_info[camera_name]
                was_online = info.status == CameraHealth.HEALTHY
                
                info.mark_failure(error)
                
                if was_online and info.status == CameraHealth.OFFLINE:
                    self.logger.warning(f"Watchdog: {camera_name} offline - {error}")
                    if self.on_camera_offline:
                        self.on_camera_offline(camera_name)
    
    def should_attempt_reconnect(self, camera_name: str) -> bool:
        """Check if should attempt reconnect"""
        with self.lock:
            if camera_name not in self.health_info:
                return False
            
            info = self.health_info[camera_name]

            # Never reconnect healthy cameras.
            if info.status == CameraHealth.HEALTHY and info.consecutive_failures <= 0:
                return False
            
            # Don't retry if exceeded max attempts
            if info.consecutive_failures > self.max_retries:
                return False
            
            # Check if enough time has passed for backoff
            if info.last_reconnect_attempt:
                backoff = info.get_backoff_seconds()
                elapsed = (datetime.now() - info.last_reconnect_attempt).total_seconds()
                if elapsed < backoff:
                    return False
            
            return True
    
    def attempt_reconnect(
        self,
        camera_name: str,
        reconnect_fn: Callable[[], bool]
    ) -> bool:
        """Attempt to reconnect camera"""
        with self.lock:
            if camera_name not in self.health_info:
                return False
            
            info = self.health_info[camera_name]
            info.last_reconnect_attempt = datetime.now()
            info.reconnect_attempts += 1
            
            backoff = info.get_backoff_seconds()
            attempt_num = info.consecutive_failures
            
            self.logger.info(
                f"Watchdog: Attempting reconnect for {camera_name} "
                f"(attempt {attempt_num}/{self.max_retries}, backoff: {backoff}s)"
            )
        
        # Call reconnect outside lock
        try:
            success = reconnect_fn()
            
            with self.lock:
                if success:
                    self.health_info[camera_name].mark_success()
                    self.logger.info(f"Watchdog: {camera_name} reconnected successfully")
                    if self.on_camera_online:
                        self.on_camera_online(camera_name)
                    return True
                else:
                    self.health_info[camera_name].mark_failure("Reconnect failed")
                    return False
        
        except Exception as e:
            with self.lock:
                self.health_info[camera_name].mark_failure(str(e))
            self.logger.error(f"Watchdog: Error reconnecting {camera_name}: {e}")
            return False
    
    def get_health(self, camera_name: str) -> Optional[CameraHealthInfo]:
        """Get camera health info"""
        with self.lock:
            if camera_name in self.health_info:
                return CameraHealthInfo(
                    camera_name=self.health_info[camera_name].camera_name,
                    status=self.health_info[camera_name].status,
                    last_seen=self.health_info[camera_name].last_seen,
                    consecutive_failures=self.health_info[camera_name].consecutive_failures,
                    last_error=self.health_info[camera_name].last_error,
                    reconnect_attempts=self.health_info[camera_name].reconnect_attempts,
                    uptime_percent=self.health_info[camera_name].uptime_percent
                )
            return None
    
    def get_status_summary(self) -> Dict:
        """Get overall watchdog status"""
        with self.lock:
            healthy = sum(1 for info in self.health_info.values() 
                         if info.status == CameraHealth.HEALTHY)
            degraded = sum(1 for info in self.health_info.values() 
                          if info.status == CameraHealth.DEGRADED)
            offline = sum(1 for info in self.health_info.values() 
                         if info.status == CameraHealth.OFFLINE)
            
            return {
                "total": len(self.health_info),
                "healthy": healthy,
                "degraded": degraded,
                "offline": offline,
                "cameras": {
                    name: {
                        "status": info.status.value,
                        "failures": info.consecutive_failures,
                        "last_error": info.last_error,
                        "attempts": info.reconnect_attempts
                    }
                    for name, info in self.health_info.items()
                }
            }
