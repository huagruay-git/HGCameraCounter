"""
GUI-Friendly Dashboard Client

Wraps DashboardBroadcaster for use in PySide6 GUI with signals
"""

import threading
import time
from typing import Optional, Callable, Dict, Any
from datetime import datetime
from PySide6.QtCore import QObject, Signal

from shared.dashboard_updater import (
    DashboardClient as BaseDashboardClient,
    DashboardBroadcaster,
    DashboardUpdate
)


class GUIDashboardClient(QObject):
    """
    GUI-compatible dashboard client with Qt signals
    
    Signals:
        status_updated: Emitted when status changes (Dict[str, Any])
        event_received: Emitted when event occurs (Dict[str, Any])
        summary_updated: Emitted when summary changes (Dict[str, Any])
        connection_status: Emitted when connection status changes (bool)
    """
    
    # Qt Signals
    status_updated = Signal(dict)
    event_received = Signal(dict)
    summary_updated = Signal(dict)
    connection_status = Signal(bool)
    
    def __init__(self, client_id: str, broadcaster: DashboardBroadcaster):
        super().__init__()
        self.client_id = client_id
        self.broadcaster = broadcaster
        self.base_client = BaseDashboardClient(client_id, broadcaster)
        
        # State
        self.connected = True
        self.last_status = {}
        self.last_summary = {}
        self.running = False
        
        # Update thread
        self.update_thread: Optional[threading.Thread] = None
        
    def start(self):
        """Start receiving updates"""
        if self.running:
            return
        
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
    
    def stop(self):
        """Stop receiving updates and cleanup"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
        
        self.base_client.close()
    
    def _update_loop(self):
        """Background thread that processes updates"""
        while self.running:
            try:
                updates = self.base_client.get_updates(timeout=0.5)
                
                for update in updates:
                    try:
                        if update.update_type == "status":
                            self.last_status = update.data
                            self.status_updated.emit(update.data)
                        
                        elif update.update_type == "event":
                            self.event_received.emit(update.data)
                        
                        elif update.update_type == "summary":
                            self.last_summary = update.data
                            self.summary_updated.emit(update.data)
                    
                    except Exception as e:
                        print(f"Error processing update: {e}")
                
                # Check connection status
                is_connected = self.broadcaster.get_subscriber_count() > 0
                if is_connected != self.connected:
                    self.connected = is_connected
                    self.connection_status.emit(is_connected)
            
            except Exception as e:
                print(f"Error in dashboard client update loop: {e}")
                time.sleep(1)
    
    def get_last_status(self) -> Dict[str, Any]:
        """Get last received status"""
        return self.last_status.copy() if self.last_status else {}
    
    def get_last_summary(self) -> Dict[str, Any]:
        """Get last received summary"""
        return self.last_summary.copy() if self.last_summary else {}
    
    def is_alive(self) -> bool:
        """Check if client is receiving updates"""
        return self.connected and len(self.last_status) > 0
