"""
Dashboard Real-time Updates

Thread-safe communication between runtime service and GUI
"""

import json
import threading
import queue
import time
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict


@dataclass
class DashboardUpdate:
    """Dashboard update message"""
    timestamp: datetime
    update_type: str  # status, event, summary, log
    data: Dict[str, Any]
    
    def to_json(self) -> str:
        """Convert to JSON"""
        return json.dumps({
            "timestamp": self.timestamp.isoformat(),
            "update_type": self.update_type,
            "data": self.data,
        })


class DashboardBroadcaster:
    """Broadcast dashboard updates to subscribers"""
    
    def __init__(self, max_queue_size: int = 100):
        self.subscribers: Dict[str, queue.Queue] = {}
        self.max_queue_size = max_queue_size
        self.lock = threading.Lock()
    
    def subscribe(self, client_id: str) -> queue.Queue:
        """Subscribe to updates"""
        with self.lock:
            q = queue.Queue(maxsize=self.max_queue_size)
            self.subscribers[client_id] = q
            return q
    
    def unsubscribe(self, client_id: str):
        """Unsubscribe from updates"""
        with self.lock:
            if client_id in self.subscribers:
                del self.subscribers[client_id]
    
    def broadcast(self, update: DashboardUpdate):
        """Broadcast update to all subscribers"""
        with self.lock:
            for client_id, q in list(self.subscribers.items()):
                try:
                    q.put_nowait(update)
                except queue.Full:
                    # Remove client if queue is full (not responding)
                    del self.subscribers[client_id]
    
    def broadcast_status(self, status: Dict[str, Any]):
        """Broadcast status update"""
        update = DashboardUpdate(
            timestamp=datetime.now(),
            update_type="status",
            data=status
        )
        self.broadcast(update)
    
    def broadcast_event(self, event: Dict[str, Any]):
        """Broadcast event"""
        update = DashboardUpdate(
            timestamp=datetime.now(),
            update_type="event",
            data=event
        )
        self.broadcast(update)
    
    def broadcast_summary(self, summary: Dict[str, Any]):
        """Broadcast summary"""
        update = DashboardUpdate(
            timestamp=datetime.now(),
            update_type="summary",
            data=summary
        )
        self.broadcast(update)
    
    def get_subscriber_count(self) -> int:
        """Get number of subscribers"""
        with self.lock:
            return len(self.subscribers)


class DashboardUpdater:
    """Update dashboard in real-time"""
    
    def __init__(self, broadcaster: DashboardBroadcaster):
        self.broadcaster = broadcaster
        self.running = False
        self.update_thread: Optional[threading.Thread] = None
        self.last_status = {}
        self.lock = threading.Lock()
    
    def start(self, update_callback: Optional[Callable] = None):
        """Start updater thread"""
        if self.running:
            return
        
        self.running = True
        self.update_callback = update_callback
        
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
    
    def stop(self):
        """Stop updater thread"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
    
    def _update_loop(self):
        """Background update loop"""
        while self.running:
            try:
                if self.update_callback:
                    status = self.update_callback()
                    
                    with self.lock:
                        if status != self.last_status:
                            self.broadcaster.broadcast_status(status)
                            self.last_status = status
                
                time.sleep(1)
            
            except Exception as e:
                print(f"Error in update loop: {e}")
    
    def update_status(self, status: Dict[str, Any]):
        """Manual status update"""
        with self.lock:
            self.last_status = status
        
        self.broadcaster.broadcast_status(status)
    
    def update_event(self, event: Dict[str, Any]):
        """Manual event update"""
        self.broadcaster.broadcast_event(event)
    
    def update_summary(self, summary: Dict[str, Any]):
        """Manual summary update"""
        self.broadcaster.broadcast_summary(summary)


class DashboardClient:
    """Dashboard client for GUI"""
    
    def __init__(self, client_id: str, broadcaster: DashboardBroadcaster):
        self.client_id = client_id
        self.broadcaster = broadcaster
        self.queue = broadcaster.subscribe(client_id)
    
    def get_updates(self, timeout: float = 0.1) -> list[DashboardUpdate]:
        """Get pending updates"""
        updates = []
        
        try:
            while True:
                update = self.queue.get(timeout=timeout)
                updates.append(update)
        except queue.Empty:
            pass
        
        return updates
    
    def close(self):
        """Disconnect from broadcaster"""
        self.broadcaster.unsubscribe(self.client_id)


# Global singleton
_broadcaster: Optional[DashboardBroadcaster] = None
_updater: Optional[DashboardUpdater] = None


def init_dashboard_service() -> DashboardBroadcaster:
    """Initialize global dashboard service"""
    global _broadcaster, _updater
    
    if _broadcaster is None:
        _broadcaster = DashboardBroadcaster()
        _updater = DashboardUpdater(_broadcaster)
    
    return _broadcaster


def get_broadcaster() -> Optional[DashboardBroadcaster]:
    """Get global broadcaster"""
    return _broadcaster


def get_updater() -> Optional[DashboardUpdater]:
    """Get global updater"""
    return _updater
