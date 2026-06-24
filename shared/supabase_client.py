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


class CCTVSupabaseRPCClient:
    """RPC-first client for CCTV ingestion functions."""

    def __init__(
        self,
        url: str,
        key: str,
        logger: Optional[logging.Logger] = None,
    ):
        self.url = (url or "").strip()
        self.key = (key or "").strip()
        self.logger = logger or logging.getLogger(__name__)
        self.client = None
        self.is_connected = False
        self._connect()

    def _connect(self) -> bool:
        if not self.url or not self.key:
            self.logger.warning("Supabase URL/Key is missing")
            return False
        if create_client is None:
            self.logger.error("supabase library not installed")
            return False
        try:
            self.client = create_client(self.url, self.key)
            self.is_connected = True
            return True
        except Exception as e:
            self.logger.error(f"Failed to create Supabase client: {e}")
            self.client = None
            self.is_connected = False
            return False

    def ensure_connected(self) -> bool:
        if self.is_connected and self.client is not None:
            return True
        return self._connect()

    def _rpc(self, fn_name: str, payload: Dict[str, Any]) -> Any:
        if not self.ensure_connected():
            raise RuntimeError("Supabase client is not connected")
        try:
            response = self.client.rpc(fn_name, payload).execute()
            return response.data
        except Exception as e:
            msg = str(e)
            self.logger.error(f"Supabase RPC failed ({fn_name}): {msg}")
            raise RuntimeError(msg) from e

    @staticmethod
    def _first_row(data: Any) -> Dict[str, Any]:
        if isinstance(data, list):
            if data and isinstance(data[0], dict):
                return data[0]
            return {}
        if isinstance(data, dict):
            return data
        return {}

    @staticmethod
    def mask_token(token: str) -> str:
        tok = (token or "").strip()
        if len(tok) <= 8:
            return "***"
        return f"{tok[:4]}...{tok[-4:]}"

    def test_device_token(self, device_token: str) -> bool:
        payload = {
            "p_device_token": (device_token or "").strip(),
            "p_status": "BOOT",
            "p_message": "token validation probe",
            "p_metrics": {"probe": True},
        }
        self._rpc("ingest_cctv_heartbeat", payload)
        return True

    def pair_cctv_device_with_enrollment(
        self,
        one_time_code: str,
        device_name: str,
        device_code: str,
        timezone_name: str = "Asia/Bangkok",
        metadata: Optional[Dict[str, Any]] = None,
        branch_name_reported: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = {
            "p_one_time_code": (one_time_code or "").strip(),
            "p_device_name": (device_name or "").strip(),
            "p_device_code": (device_code or "").strip(),
            "p_timezone": (timezone_name or "").strip() or "Asia/Bangkok",
            "p_metadata": metadata or {},
            "p_branch_name_reported": (branch_name_reported or "").strip() or None,
        }
        data = self._rpc("pair_cctv_device_with_enrollment", payload)
        return self._first_row(data)

    def admin_create_cctv_enrollment(
        self,
        branch_id: int,
        device_name_hint: Optional[str] = None,
        expires_in_minutes: int = 15,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "p_branch_id": int(branch_id),
            "p_device_name_hint": (device_name_hint or "").strip() or None,
            "p_expires_in_minutes": max(1, int(expires_in_minutes)),
            "p_metadata": metadata or {},
        }
        data = self._rpc("admin_create_cctv_enrollment", payload)
        return self._first_row(data)

    def ingest_cctv_heartbeat(
        self,
        device_token: str,
        status: str = "ONLINE",
        message: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        heartbeat_at: Optional[str] = None,
    ) -> Any:
        payload = {
            "p_device_token": (device_token or "").strip(),
            "p_status": status,
            "p_message": message,
            "p_metrics": metrics or {},
        }
        if heartbeat_at:
            payload["p_heartbeat_at"] = heartbeat_at
        return self._rpc("ingest_cctv_heartbeat", payload)

    def ingest_cctv_realtime(
        self,
        device_token: str,
        event_time: str,
        people_in: int,
        people_out: int,
        people_inside: int,
        people_passing: int,
        confidence: Optional[float] = None,
        raw_payload: Optional[Dict[str, Any]] = None,
        branch_name: Optional[str] = None,
    ) -> Any:
        payload = {
            "p_device_token": (device_token or "").strip(),
            "p_event_time": event_time,
            "p_people_in": max(0, int(people_in)),
            "p_people_out": max(0, int(people_out)),
            "p_people_inside": max(0, int(people_inside)),
            "p_people_passing": max(0, int(people_passing)),
            "p_confidence": confidence,
            "p_raw_payload": raw_payload or {},
            "p_branch_name": (branch_name or "").strip() or None,
        }
        return self._rpc("ingest_cctv_realtime", payload)

    def ingest_cctv_daily_summary(
        self,
        device_token: str,
        business_date: str,
        customers_total: int,
        peak_people_inside: int = 0,
        open_time: Optional[str] = None,
        close_time: Optional[str] = None,
        note: Optional[str] = None,
        raw_payload: Optional[Dict[str, Any]] = None,
        branch_name: Optional[str] = None,
    ) -> Any:
        payload = {
            "p_device_token": (device_token or "").strip(),
            "p_business_date": business_date,
            "p_customers_total": max(0, int(customers_total)),
            "p_peak_people_inside": max(0, int(peak_people_inside)),
            "p_open_time": open_time,
            "p_close_time": close_time,
            "p_note": note,
            "p_raw_payload": raw_payload or {},
            "p_branch_name": (branch_name or "").strip() or None,
        }
        return self._rpc("ingest_cctv_daily_summary", payload)

    # ------------------------------------------------------------------
    # Model / config OTA (device-token authenticated)
    # Requires the SECURITY DEFINER RPCs in
    # supabase/migrations/20260616_cctv_device_ota.sql
    # ------------------------------------------------------------------
    def get_cctv_runtime_bootstrap(self, device_token: str) -> Dict[str, Any]:
        """Return the active config + active model for this device's branch.

        Expected shape:
          {"config": {...} | null,
           "model": {"id","model_version","storage_bucket","storage_path",
                     "sha256","signed_url"?} | null}
        """
        data = self._rpc("get_cctv_runtime_bootstrap", {
            "p_device_token": (device_token or "").strip(),
        })
        return self._first_row(data) if not isinstance(data, dict) else data

    def log_cctv_model_download(self, device_token: str, model_id: Optional[str],
                                model_version: Optional[str], status: str,
                                message: Optional[str] = None,
                                sha256_ok: Optional[bool] = None) -> Any:
        """Record a model-download outcome (success/failed) via a device-token RPC."""
        return self._rpc("log_cctv_model_download", {
            "p_device_token": (device_token or "").strip(),
            "p_model_id": model_id,
            "p_model_version": model_version,
            "p_status": status,
            "p_message": message,
            "p_sha256_ok": sha256_ok,
        })

    def download_storage_object(self, bucket: str, path: str) -> bytes:
        """Download an object from Supabase Storage via the connected client.

        Works when the device has read access to the path (a device/anon read
        policy on the bucket, or a public bucket). If the bootstrap RPC returns a
        `signed_url` instead, fetch that URL directly rather than calling this.
        """
        if not self.ensure_connected():
            raise RuntimeError("Supabase client is not connected")
        return self.client.storage.from_(bucket).download(path)

    # ------------------------------------------------------------------
    # Remote device commands (reboot/restart_app/shutdown/power_cycle/wake)
    # Requires supabase/migrations/20260624_cctv_device_commands.sql
    # ------------------------------------------------------------------
    def get_cctv_device_commands(self, device_token: str) -> List[Dict[str, Any]]:
        """Pull this device's pending commands; the server marks them 'acked'."""
        data = self._rpc("get_cctv_device_commands", {
            "p_device_token": (device_token or "").strip(),
        })
        if isinstance(data, list):
            return [c for c in data if isinstance(c, dict)]
        return []

    def ack_cctv_device_command(self, device_token: str, command_id: str,
                                status: str, detail: Optional[str] = None) -> Any:
        """Report a command outcome ('done' or 'failed')."""
        return self._rpc("ack_cctv_device_command", {
            "p_device_token": (device_token or "").strip(),
            "p_command_id": command_id,
            "p_status": status,
            "p_detail": detail,
        })

    def admin_enqueue_cctv_command(self, device_id: int, command: str,
                                   args: Optional[Dict[str, Any]] = None,
                                   expires_in_minutes: int = 60) -> Any:
        """Queue a command for a device (authenticated caller only). Returns the command id."""
        return self._rpc("admin_enqueue_cctv_command", {
            "p_device_id": int(device_id),
            "p_command": command,
            "p_args": args or {},
            "p_expires_in_minutes": int(expires_in_minutes),
        })
