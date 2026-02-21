"""
Supabase Integration Client

Handle device status, events, and heartbeat
"""

import time
import re
import logging
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any

try:
    from supabase import create_client
except ImportError:
    create_client = None


class SupabaseClient:
    """Supabase client for event submission and heartbeat"""
    
    def __init__(
        self,
        url: str,
        key: str,
        branch_code: str,
        events_table: str = "events",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize Supabase client
        
        Args:
            url: Supabase project URL
            key: Supabase anon key
            branch_code: Branch/shop identifier
            logger: Logger instance
        """
        self.url = url
        self.key = key
        self.branch_code = branch_code
        self.events_table = events_table or "events"
        self.logger = logger or logging.getLogger(__name__)
        
        self.client = None
        self.is_connected = False
        self.known_missing_columns = set()
        
        self._connect()
    
    def _connect(self) -> bool:
        """Connect to Supabase"""
        if not self.url or not self.key:
            self.logger.warning("Supabase credentials not configured")
            return False
        
        if create_client is None:
            self.logger.error("supabase library not installed")
            return False
        
        try:
            self.client = create_client(self.url, self.key)
            self.is_connected = True
            self.logger.info(f"Connected to Supabase: {self.url}")
            return True
        except Exception as e:
            self.logger.error(f"Error connecting to Supabase: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Test Supabase connection"""
        if not self.client:
            return False
        
        try:
            # Try a simple query
            response = self.client.table("device_status").select("*").limit(1).execute()
            self.logger.info("✓ Supabase connection test passed")
            return True
        except Exception as e:
            self.logger.error(f"Supabase connection test failed: {e}")
            return False
    
    def update_device_status(
        self,
        device_id: str,
        status: Dict[str, Any]
    ) -> bool:
        """
        Update or insert device status
        
        Args:
            device_id: Device identifier
            status: Status dictionary
        
        Returns:
            True if successful
        """
        if not self.client:
            return False
        
        try:
            data = {
                "device_id": device_id,
                "branch_code": self.branch_code,
                "status": status.get("status", "online"),
                "last_heartbeat": datetime.now().isoformat(),
                "cameras_ok": status.get("cameras_ok", 0),
                "active_people": status.get("active_people", 0),
                "haircuts_today": status.get("haircuts_today", 0),
                "metadata": status.get("metadata", {}),
            }
            
            # Upsert
            response = self.client.table("device_status").upsert(
                data,
                on_conflict="device_id"
            ).execute()
            
            return True
        except Exception as e:
            self.logger.error(f"Error updating device status: {e}")
            return False

    def submit_events(
        self,
        events: List[Dict[str, Any]],
        retry_attempts: int = 3,
        retry_delay: float = 2.0
    ) -> bool:
        """
        Submit events to Supabase with retry
        
        Args:
            events: List of event dictionaries
            retry_attempts: Number of retry attempts
            retry_delay: Delay between retries (seconds)
        
        Returns:
            True if successful
        """
        if not self.client or not events:
            return False
        
        # Add branch code to all events (some schemas may not have this column).
        payload = [dict(event) for event in events]
        for event in payload:
            event["branch_code"] = self.branch_code
        
        # Filter out known missing columns immediately
        if self.known_missing_columns:
            for event in payload:
                for col in self.known_missing_columns:
                    event.pop(col, None)

        # Try to submit with retries
        for attempt in range(retry_attempts):
            try:
                response = self.client.table(self.events_table).insert(payload).execute()
                self.logger.info(f"Submitted {len(events)} events to Supabase")
                return True
            except Exception as e:
                msg = str(e)
                # Auto-adapt to schema differences by removing missing columns
                # reported by PostgREST (e.g. "Could not find the 'camera' column ...").
                m = re.search(r"Could not find the '([^']+)' column", msg)
                if m:
                    missing_col = m.group(1)
                    self.known_missing_columns.add(missing_col)
                    
                    # Remove from current payload for retry
                    for event in payload:
                        event.pop(missing_col, None)
                        
                    self.logger.warning(
                        f"Supabase table '{self.events_table}' missing column '{missing_col}', retrying without it"
                    )
                    continue
                
                # Check for other specific error patterns if needed
                if "branch_code" in msg and ("schema cache" in msg or "Could not find" in msg):
                     self.known_missing_columns.add("branch_code")
                     for event in payload:
                        event.pop("branch_code", None)
                     self.logger.warning(
                        f"Supabase table '{self.events_table}' likely missing 'branch_code', retrying without it"
                     )
                     continue

                self.logger.warning(
                    f"Error submitting events (attempt {attempt + 1}/{retry_attempts}): {e}"
                )
                
                if attempt < retry_attempts - 1:
                    time.sleep(retry_delay)
        
        self.logger.error(f"Failed to submit events after {retry_attempts} attempts")
        return False
    
    def get_branch_config(self) -> Optional[Dict]:
        """Get branch configuration from Supabase"""
        if not self.client:
            return None
        
        try:
            response = self.client.table("branch_config").select("*").eq(
                "branch_code", self.branch_code
            ).execute()
            
            if response.data:
                return response.data[0]
            
            return None
        except Exception as e:
            self.logger.error(f"Error fetching branch config: {e}")
            return None


class SupabaseSync:
    """Background thread for Supabase synchronization"""
    
    def __init__(
        self,
        supabase_client: SupabaseClient,
        heartbeat_interval: int = 30,
        batch_size: int = 50,
        batch_timeout: int = 60,
        retry_attempts: int = 3,
        retry_delay: float = 2.0,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize Supabase sync
        
        Args:
            supabase_client: SupabaseClient instance
            heartbeat_interval: Heartbeat interval (seconds)
            batch_size: Events to batch before submitting
            batch_timeout: Max time to wait for batch (seconds)
            retry_attempts: Number of retry attempts in submit_events
            retry_delay: Delay between retries
            logger: Logger instance
        """
        self.client = supabase_client
        self.heartbeat_interval = heartbeat_interval
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.logger = logger or logging.getLogger(__name__)
        
        self.event_queue: List[Dict] = []
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        self.last_heartbeat = datetime.now()
        self.last_batch_flush = datetime.now()
    
    def start(self):
        """Start background sync thread"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._sync_loop, daemon=True)
        self.thread.start()
        self.logger.info("Supabase sync thread started")
    
    def stop(self):
        """Stop background sync thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        
        # Flush remaining events
        self._flush_events()
        self.logger.info("Supabase sync stopped")
    
    def add_event(self, event: Dict[str, Any]):
        """Add event to queue"""
        with self.lock:
            self.event_queue.append(event)
    
    def _sync_loop(self):
        """Background sync loop"""
        while self.running:
            try:
                # Check if should flush events
                should_flush = False
                
                with self.lock:
                    if len(self.event_queue) >= self.batch_size:
                        should_flush = True
                    elif len(self.event_queue) > 0:
                        time_since_flush = (
                            datetime.now() - self.last_batch_flush
                        ).total_seconds()
                        if time_since_flush > self.batch_timeout:
                            should_flush = True
                
                if should_flush:
                    self._flush_events()
                
                # Send heartbeat
                time_since_heartbeat = (
                    datetime.now() - self.last_heartbeat
                ).total_seconds()
                if time_since_heartbeat > self.heartbeat_interval:
                    self._send_heartbeat()
                
                time.sleep(1)
            
            except Exception as e:
                self.logger.error(f"Error in sync loop: {e}")
    
    def _flush_events(self):
        """Flush queued events"""
        with self.lock:
            if not self.event_queue:
                return
            
            events_to_submit = self.event_queue.copy()
            self.event_queue.clear()
        
        if self.client.submit_events(
            events_to_submit,
            retry_attempts=self.retry_attempts,
            retry_delay=self.retry_delay
        ):
            self.last_batch_flush = datetime.now()
    
    def _send_heartbeat(self):
        """Send heartbeat status"""
        # TODO: Implement with proper status info
        self.last_heartbeat = datetime.now()
    
    def get_queue_size(self) -> int:
        """Get number of queued events"""
        with self.lock:
            return len(self.event_queue)
