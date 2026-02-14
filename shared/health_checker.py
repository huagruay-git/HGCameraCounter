"""
Health Checks - Periodic System Diagnostics

Performs regular health checks on system components and reports issues.
"""

import time
import threading
import logging
import socket
import psutil
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class CheckStatus(Enum):
    """Health check status"""
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a single health check"""
    check_name: str
    status: CheckStatus
    message: str
    timestamp: datetime
    severity: str  # "info", "warning", "error"


class HealthChecker:
    """Perform periodic health checks"""
    
    def __init__(
        self,
        logger: logging.Logger,
        check_interval: float = 30.0
    ):
        self.logger = logger
        self.check_interval = check_interval
        
        # Results tracking
        self.last_results: Dict[str, HealthCheckResult] = {}
        self.results_history: List[HealthCheckResult] = []
        self.lock = threading.Lock()
        
        # Running state
        self.running = False
        self.check_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self.on_check_failed: Optional[Callable[[str, str], None]] = None
    
    def start(self):
        """Start health check thread"""
        if self.running:
            return
        
        self.running = True
        self.check_thread = threading.Thread(target=self._check_loop, daemon=True)
        self.check_thread.start()
        self.logger.info("Health Checker started")
    
    def stop(self):
        """Stop health checks"""
        self.running = False
        if self.check_thread:
            self.check_thread.join(timeout=5)
        self.logger.info("Health Checker stopped")
    
    def _check_loop(self):
        """Background health check loop"""
        while self.running:
            try:
                self._perform_checks()
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
                time.sleep(self.check_interval)
    
    def _perform_checks(self):
        """Run all health checks"""
        checks = [
            ("disk_space", self._check_disk_space),
            ("memory", self._check_memory),
            ("cpu", self._check_cpu),
            ("network", self._check_network),
            ("file_permissions", self._check_file_permissions),
        ]
        
        for check_name, check_fn in checks:
            try:
                result = check_fn()
                self._record_result(result)
                
                if result.status != CheckStatus.OK:
                    self.logger.warning(
                        f"Health check '{check_name}': {result.message}"
                    )
                    
                    if self.on_check_failed and result.status == CheckStatus.ERROR:
                        self.on_check_failed(check_name, result.message)
            
            except Exception as e:
                self.logger.error(f"Error running check '{check_name}': {e}")
    
    def _check_disk_space(self) -> HealthCheckResult:
        """Check available disk space"""
        try:
            disk = psutil.disk_usage("/")
            percent_free = 100 - disk.percent
            
            if disk.percent > 95:
                status = CheckStatus.ERROR
                severity = "error"
                message = f"Disk full: {disk.percent:.1f}% used"
            elif disk.percent > 85:
                status = CheckStatus.WARNING
                severity = "warning"
                message = f"Disk almost full: {disk.percent:.1f}% used"
            else:
                status = CheckStatus.OK
                severity = "info"
                message = f"Disk OK: {percent_free:.1f}% free"
            
            return HealthCheckResult(
                check_name="disk_space",
                status=status,
                message=message,
                timestamp=datetime.now(),
                severity=severity
            )
        
        except Exception as e:
            return HealthCheckResult(
                check_name="disk_space",
                status=CheckStatus.UNKNOWN,
                message=f"Could not check disk: {e}",
                timestamp=datetime.now(),
                severity="error"
            )
    
    def _check_memory(self) -> HealthCheckResult:
        """Check memory usage"""
        try:
            mem = psutil.virtual_memory()
            
            if mem.percent > 90:
                status = CheckStatus.ERROR
                severity = "error"
                message = f"Memory critical: {mem.percent:.1f}% used"
            elif mem.percent > 80:
                status = CheckStatus.WARNING
                severity = "warning"
                message = f"Memory high: {mem.percent:.1f}% used"
            else:
                status = CheckStatus.OK
                severity = "info"
                message = f"Memory OK: {mem.percent:.1f}% used"
            
            return HealthCheckResult(
                check_name="memory",
                status=status,
                message=message,
                timestamp=datetime.now(),
                severity=severity
            )
        
        except Exception as e:
            return HealthCheckResult(
                check_name="memory",
                status=CheckStatus.UNKNOWN,
                message=f"Could not check memory: {e}",
                timestamp=datetime.now(),
                severity="error"
            )
    
    def _check_cpu(self) -> HealthCheckResult:
        """Check CPU usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.5)
            
            if cpu_percent > 90:
                status = CheckStatus.WARNING
                severity = "warning"
                message = f"CPU high: {cpu_percent:.1f}%"
            else:
                status = CheckStatus.OK
                severity = "info"
                message = f"CPU OK: {cpu_percent:.1f}%"
            
            return HealthCheckResult(
                check_name="cpu",
                status=status,
                message=message,
                timestamp=datetime.now(),
                severity=severity
            )
        
        except Exception as e:
            return HealthCheckResult(
                check_name="cpu",
                status=CheckStatus.UNKNOWN,
                message=f"Could not check CPU: {e}",
                timestamp=datetime.now(),
                severity="error"
            )
    
    def _check_network(self) -> HealthCheckResult:
        """Check network connectivity"""
        try:
            # Try to resolve DNS
            socket.gethostbyname("8.8.8.8")
            
            return HealthCheckResult(
                check_name="network",
                status=CheckStatus.OK,
                message="Network OK",
                timestamp=datetime.now(),
                severity="info"
            )
        
        except Exception as e:
            return HealthCheckResult(
                check_name="network",
                status=CheckStatus.WARNING,
                message=f"Network issue: {e}",
                timestamp=datetime.now(),
                severity="warning"
            )
    
    def _check_file_permissions(self) -> HealthCheckResult:
        """Check file write permissions"""
        try:
            import os
            import tempfile
            
            # Test write to temp directory
            with tempfile.NamedTemporaryFile(delete=True) as tmp:
                pass
            
            return HealthCheckResult(
                check_name="file_permissions",
                status=CheckStatus.OK,
                message="File permissions OK",
                timestamp=datetime.now(),
                severity="info"
            )
        
        except Exception as e:
            return HealthCheckResult(
                check_name="file_permissions",
                status=CheckStatus.ERROR,
                message=f"File permission issue: {e}",
                timestamp=datetime.now(),
                severity="error"
            )
    
    def _record_result(self, result: HealthCheckResult):
        """Record health check result"""
        with self.lock:
            self.last_results[result.check_name] = result
            self.results_history.append(result)
            
            # Keep only last 1000 results
            if len(self.results_history) > 1000:
                self.results_history = self.results_history[-1000:]
    
    def get_status(self) -> Dict:
        """Get current health status"""
        with self.lock:
            return {
                "timestamp": datetime.now().isoformat(),
                "checks": {
                    name: {
                        "status": result.status.value,
                        "message": result.message,
                        "severity": result.severity
                    }
                    for name, result in self.last_results.items()
                },
                "overall": self._calculate_overall_status()
            }
    
    def _calculate_overall_status(self) -> str:
        """Calculate overall health status"""
        if not self.last_results:
            return "unknown"
        
        statuses = [r.status for r in self.last_results.values()]
        
        if CheckStatus.ERROR in statuses:
            return "error"
        elif CheckStatus.WARNING in statuses:
            return "warning"
        else:
            return "ok"
    
    def get_errors(self) -> List[HealthCheckResult]:
        """Get all error results"""
        with self.lock:
            return [
                r for r in self.last_results.values()
                if r.status == CheckStatus.ERROR
            ]
    
    def get_warnings(self) -> List[HealthCheckResult]:
        """Get all warning results"""
        with self.lock:
            return [
                r for r in self.last_results.values()
                if r.status == CheckStatus.WARNING
            ]
