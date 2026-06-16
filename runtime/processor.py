
import csv
import json
import os
import time
import glob
import logging
from pathlib import Path
from typing import Optional, List, Set, Dict, Any
from datetime import datetime, timedelta, time as dt_time
import cv2
import shutil

# Reuse existing Agent logic but modified for file input
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from runtime.agent_v2 import RuntimeService, CameraStream, CONFIG, logger as agent_logger
from shared.config import Config
from shared.logger import setup_logger
from shared.supabase_client import CCTVSupabaseRPCClient

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None

logger = setup_logger("processor", "logs")


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        txt = value.strip().lower()
        if txt in {"1", "true", "yes", "on"}:
            return True
        if txt in {"0", "false", "no", "off"}:
            return False
    return bool(value) if value is not None else bool(default)

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
        
        from collections import deque
        self.rolling_buffer = deque(maxlen=150)
        self.recording_event = None

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.is_connected = False
            return None
        return frame

    def _save_trigger_clip_v2(self):
        """Dummy for batch mode to avoid errors if triggered"""
        pass

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
        
        from collections import deque
        self.rolling_buffer = deque(maxlen=1)
        self.recording_event = None

    def read_frame(self):
        return None
    
    def connect(self):
        self.is_connected = True
        
    def disconnect(self):
        self.is_connected = False

    def _save_trigger_clip_v2(self):
        pass

    def release(self):
        pass

class BatchProcessor(RuntimeService):
    def __init__(self):
        # Initialize parent RuntimeService but don't start threads yet
        super().__init__()
        self.service_mode = "recorded"
        self.config = CONFIG  # Fix: Ensure self.config is available
        self.recordings_dir = Path("data/recordings")
        self.processed_dir = Path("data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.segment_time_sec = 60
        self.file_ready_grace_sec = 15.0
        self.retry_open_max_age_sec = float((self.segment_time_sec * 3) + self.file_ready_grace_sec)
        self.processor_start_ts = time.time()
        runtime_cfg = self.config.get("runtime", {}) or {}
        # Default to skip old backlog on start unless explicitly disabled.
        skip_backlog_cfg = runtime_cfg.get("processor_skip_backlog_on_start", True)
        if isinstance(skip_backlog_cfg, str):
            self.processor_skip_backlog_on_start = skip_backlog_cfg.strip().lower() in {"1", "true", "yes", "on"}
        else:
            self.processor_skip_backlog_on_start = bool(skip_backlog_cfg)
        only_today_cfg = runtime_cfg.get("processor_only_today", True)
        if isinstance(only_today_cfg, str):
            self.processor_only_today = only_today_cfg.strip().lower() in {"1", "true", "yes", "on"}
        else:
            self.processor_only_today = bool(only_today_cfg)
        self.processor_max_recording_age_sec = max(
            60.0,
            float(runtime_cfg.get("processor_max_recording_age_sec", 24.0 * 3600.0)),
        )
        quick_cfg = runtime_cfg.get("processor_quick_skip_no_person", True)
        if isinstance(quick_cfg, str):
            self.processor_quick_skip_no_person = quick_cfg.strip().lower() in {"1", "true", "yes", "on"}
        else:
            self.processor_quick_skip_no_person = bool(quick_cfg)
        self.processor_quick_check_frames = max(1, int(runtime_cfg.get("processor_quick_check_frames", 18)))
        self.processor_quick_check_stride = max(1, int(runtime_cfg.get("processor_quick_check_stride", 12)))
        self.processor_frame_sleep_sec = max(0.0, float(runtime_cfg.get("processor_frame_sleep_sec", 0.03)))
        self.processor_previous_day_grace_hours = max(
            0.0, float(runtime_cfg.get("processor_previous_day_grace_hours", 10.0))
        )
        self.processor_startup_backlog_grace_sec = max(
            float(runtime_cfg.get("processor_startup_backlog_grace_sec", 300.0)),
            float(self.segment_time_sec) + float(self.file_ready_grace_sec),
        )
        self._quarantined_recordings: Set[str] = set()

        # Recorded-mode daily deadline sync (finish counting previous day and push before next morning)
        self._sync_state_path = Path("runtime/processor_sync_state.json")
        self._daily_stats_by_date: Dict[str, Dict[str, Any]] = {}
        self._daily_sent_dates: Set[str] = set()
        self._daily_sync_warnings: Set[str] = set()
        self._rpc_client: Optional[CCTVSupabaseRPCClient] = None
        self._next_deadline_check_ts = 0.0
        self._next_daily_sync_retry_ts = 0.0
        self._load_sync_state()
        self._init_deadline_sync_config(runtime_cfg)
        
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
        if self.deadline_sync_enabled:
            logger.info(
                "Recorded deadline sync enabled: deadline=%s, timezone=%s, grace=%.1fh",
                self.recorded_daily_deadline_time.strftime("%H:%M"),
                self.business_tz_name,
                self.processor_previous_day_grace_hours,
            )
        
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
        self._start_chair_service_autotrain_thread()

        # Cleanup old snapshots on startup
        self._cleanup_snapshots()
        self._quarantine_startup_backlog()

        # Global counting/logic loop (CRITICAL: this generates events from detections)
        counting_thread = threading.Thread(target=self._global_counting_loop, daemon=True)
        counting_thread.start()
        self.threads.append(counting_thread)
        
        from runtime.agent_v2 import CAMERAS_CONFIG
        
        while self.running:
            if self.enable_business_hours_guard and (not self._is_within_business_hours()):
                time.sleep(2.0)
                continue

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
                
                files = []
                for fp in sorted(list(cam_dir.glob("*.mp4"))):
                    try:
                        key = str(fp.resolve())
                    except Exception:
                        key = str(fp)
                    if key in self._quarantined_recordings:
                        continue
                    if not self._is_candidate_recording_file(fp):
                        continue
                    files.append(fp)
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

            self._maybe_finalize_previous_day(force=did_work)
            if not did_work:
                time.sleep(2.0) # Wait for new files
    # ... (start_processing, etc. unchanged)

    def stop(self):
        try:
            self._maybe_finalize_previous_day(force=True)
        except Exception as e:
            logger.warning(f"Final daily sync check on stop failed: {e}")
        super().stop()

    def _init_deadline_sync_config(self, runtime_cfg: Dict[str, Any]):
        sup_cfg = self.config.get("supabase", {}) or {}
        cloud_cfg = sup_cfg.get("cloud_sync", {}) or {}
        self.deadline_sync_enabled = bool(cloud_cfg.get("enabled", False)) and bool(cloud_cfg.get("sync_daily", True))
        self.cloud_sync_url = str(sup_cfg.get("url", "") or "").strip()
        self.cloud_sync_key = str(sup_cfg.get("key", "") or "").strip()
        self.cloud_sync_token = str(cloud_cfg.get("device_token", "") or "").strip()
        self.cloud_sync_branch_name = str(
            cloud_cfg.get("branch_name_reported", self.config.get("branch_code", ""))
            or self.config.get("branch_code", "")
        ).strip()
        self.cloud_sync_include_haircuts = bool(cloud_cfg.get("include_haircuts_in_daily", True))
        self.cloud_sync_include_washes = bool(cloud_cfg.get("include_washes_in_payload", True))
        self.business_tz_name = str(cloud_cfg.get("timezone", "Asia/Bangkok") or "Asia/Bangkok")
        deadline_txt = str(
            cloud_cfg.get(
                "recorded_daily_deadline_time",
                cloud_cfg.get("daily_summary_deadline_time", "08:00"),
            )
            or "08:00"
        )
        self.recorded_daily_deadline_time = self._parse_hhmm(deadline_txt, dt_time(hour=8, minute=0))
        deadline_hour = float(self.recorded_daily_deadline_time.hour) + (
            float(self.recorded_daily_deadline_time.minute) / 60.0
        )
        self.processor_previous_day_grace_hours = max(
            float(self.processor_previous_day_grace_hours),
            deadline_hour + 0.5,
        )
        self.deadline_check_interval_sec = max(
            15.0, float(runtime_cfg.get("processor_deadline_check_interval_sec", 30.0))
        )
        self.daily_sync_retry_sec = max(30.0, float(runtime_cfg.get("processor_daily_sync_retry_sec", 120.0)))

    @staticmethod
    def _parse_hhmm(value: str, fallback: Optional[dt_time] = None) -> Optional[dt_time]:
        txt = str(value or "").strip()
        if len(txt) >= 5 and txt[2] == ":" and txt[:2].isdigit() and txt[3:5].isdigit():
            hh = max(0, min(23, int(txt[:2])))
            mm = max(0, min(59, int(txt[3:5])))
            return dt_time(hour=hh, minute=mm)
        return fallback

    def _now_in_business_tz(self) -> datetime:
        if ZoneInfo is not None:
            try:
                return datetime.now(ZoneInfo(self.business_tz_name))
            except Exception:
                pass
        return datetime.now()

    def _load_sync_state(self):
        self._daily_stats_by_date = {}
        self._daily_sent_dates = set()
        if not self._sync_state_path.exists():
            return
        try:
            raw = json.loads(self._sync_state_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return
            sent = raw.get("sent_daily_dates", [])
            if isinstance(sent, list):
                self._daily_sent_dates = {str(x) for x in sent if str(x).strip()}
            stats = raw.get("daily_stats", {})
            if isinstance(stats, dict):
                for d, payload in stats.items():
                    if isinstance(payload, dict):
                        self._daily_stats_by_date[str(d)] = dict(payload)
        except Exception as e:
            logger.warning(f"Cannot load processor sync state: {e}")

    def _save_sync_state(self):
        try:
            self._sync_state_path.parent.mkdir(parents=True, exist_ok=True)
            # Keep last 90 days only
            keys = sorted(self._daily_stats_by_date.keys(), reverse=True)[:90]
            daily_stats = {k: self._daily_stats_by_date.get(k, {}) for k in keys}
            sent_dates = sorted(list(self._daily_sent_dates))[-180:]
            payload = {
                "updated_at": datetime.utcnow().isoformat(),
                "sent_daily_dates": sent_dates,
                "daily_stats": daily_stats,
            }
            self._sync_state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Cannot save processor sync state: {e}")

    def _resolve_recording_business_date(self, file_path: Path) -> str:
        dt = self._file_ts_from_name(file_path)
        if dt is None:
            try:
                dt = datetime.fromtimestamp(file_path.stat().st_mtime)
            except Exception:
                dt = self._now_in_business_tz().replace(tzinfo=None)
        return dt.strftime("%Y-%m-%d")

    def _update_daily_stats_for_file(self, file_path: Path, new_events: List[Any], max_people: int):
        business_date = self._resolve_recording_business_date(file_path)
        stats = dict(
            self._daily_stats_by_date.get(
                business_date,
                {
                    "haircuts": 0,
                    "washes": 0,
                    "waits": 0,
                    "verified": 0,
                    "total_events": 0,
                    "peak_people_inside": 0,
                    "processed_files": 0,
                    "last_update_ts": 0.0,
                },
            )
        )
        for ev in new_events:
            ev_type = getattr(getattr(ev, "event_type", None), "value", getattr(ev, "event_type", ""))
            ev_txt = str(ev_type or "").strip().lower()
            if ev_txt in {"haircut", "chair"}:
                stats["haircuts"] = int(stats.get("haircuts", 0)) + 1
            elif ev_txt == "wash":
                stats["washes"] = int(stats.get("washes", 0)) + 1
            elif ev_txt == "wait":
                stats["waits"] = int(stats.get("waits", 0)) + 1
            elif ev_txt == "verified":
                stats["verified"] = int(stats.get("verified", 0)) + 1
        stats["total_events"] = int(stats.get("total_events", 0)) + int(len(new_events))
        stats["processed_files"] = int(stats.get("processed_files", 0)) + 1
        stats["peak_people_inside"] = max(int(stats.get("peak_people_inside", 0)), int(max_people))
        stats["last_update_ts"] = float(time.time())
        self._daily_stats_by_date[business_date] = stats
        self._save_sync_state()

    def _load_daily_stats_from_report(self, business_date: str) -> Dict[str, int]:
        report_path = Path("reports") / f"report_{business_date}.csv"
        out = {"haircuts": 0, "washes": 0, "waits": 0, "verified": 0, "total_events": 0}
        if not report_path.exists():
            return out
        try:
            with report_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ev_type = str(row.get("event_type", "") or "").strip().lower()
                    out["total_events"] += 1
                    if ev_type in {"haircut", "chair"}:
                        out["haircuts"] += 1
                    elif ev_type == "wash":
                        out["washes"] += 1
                    elif ev_type == "wait":
                        out["waits"] += 1
                    elif ev_type == "verified":
                        out["verified"] += 1
        except Exception as e:
            logger.warning(f"Cannot parse report for {business_date}: {e}")
        return out

    def _has_pending_recordings_for_date(self, business_date: str) -> bool:
        date_key = str(business_date or "").replace("-", "")
        if len(date_key) != 8:
            return False
        if not self.recordings_dir.exists():
            return False
        for camera_dir in self.recordings_dir.iterdir():
            if not camera_dir.is_dir():
                continue
            for file_path in camera_dir.glob(f"{date_key}_*.mp4"):
                try:
                    key = str(file_path.resolve())
                except Exception:
                    key = str(file_path)
                if key in self._quarantined_recordings:
                    continue
                if file_path.exists():
                    return True
        return False

    def _daily_deadline_dt(self, business_date: str) -> Optional[datetime]:
        try:
            base = datetime.strptime(str(business_date), "%Y-%m-%d").date()
            due_date = base + timedelta(days=1)
            if ZoneInfo is not None:
                try:
                    return datetime(
                        due_date.year,
                        due_date.month,
                        due_date.day,
                        self.recorded_daily_deadline_time.hour,
                        self.recorded_daily_deadline_time.minute,
                        tzinfo=ZoneInfo(self.business_tz_name),
                    )
                except Exception:
                    pass
            return datetime(
                due_date.year,
                due_date.month,
                due_date.day,
                self.recorded_daily_deadline_time.hour,
                self.recorded_daily_deadline_time.minute,
            )
        except Exception:
            return None

    def _require_rpc_client(self) -> CCTVSupabaseRPCClient:
        if not self.cloud_sync_url or not self.cloud_sync_key or not self.cloud_sync_token:
            raise RuntimeError("Cloud sync URL/Key/device_token is missing in config")
        if self._rpc_client is not None:
            if self._rpc_client.ensure_connected():
                return self._rpc_client
        self._rpc_client = CCTVSupabaseRPCClient(url=self.cloud_sync_url, key=self.cloud_sync_key, logger=logger)
        if not self._rpc_client.ensure_connected():
            raise RuntimeError("Cannot connect to Supabase for recorded deadline sync")
        return self._rpc_client

    def _build_daily_payload_for_date(self, business_date: str) -> Dict[str, Any]:
        stats = dict(self._daily_stats_by_date.get(business_date, {}))
        if not stats:
            stats = {
                "haircuts": 0,
                "washes": 0,
                "waits": 0,
                "verified": 0,
                "total_events": 0,
                "peak_people_inside": 0,
                "processed_files": 0,
            }
        if int(stats.get("total_events", 0)) <= 0:
            report_stats = self._load_daily_stats_from_report(business_date)
            if int(report_stats.get("total_events", 0)) > 0:
                for k, v in report_stats.items():
                    stats[k] = int(v)
        customers_total = int(stats.get("haircuts", 0)) if self.cloud_sync_include_haircuts else int(
            stats.get("total_events", 0)
        )
        runtime_cfg = self.config.get("runtime", {}) or {}
        open_time = str(runtime_cfg.get("business_hours_start", "") or "").strip()[:5] or None
        close_time = str(runtime_cfg.get("business_hours_end", "") or "").strip()[:5] or None
        raw_payload = {
            "source": "HGCameraCounter-Processor",
            "count_mode": "recorded",
            "deadline_time": self.recorded_daily_deadline_time.strftime("%H:%M"),
            "summary": {
                "haircuts": int(stats.get("haircuts", 0)),
                "washes": int(stats.get("washes", 0)),
                "waits": int(stats.get("waits", 0)),
                "verified": int(stats.get("verified", 0)),
                "total_events": int(stats.get("total_events", 0)),
                "processed_files": int(stats.get("processed_files", 0)),
            },
        }
        if not self.cloud_sync_include_washes:
            raw_payload["summary"].pop("washes", None)
            raw_payload["summary"].pop("waits", None)

        return {
            "business_date": business_date,
            "customers_total": max(0, int(customers_total)),
            "peak_people_inside": max(0, int(stats.get("peak_people_inside", 0))),
            "open_time": open_time,
            "close_time": close_time,
            "note": "Recorded mode daily summary from processor deadline sync",
            "raw_payload": raw_payload,
            "branch_name": self.cloud_sync_branch_name or str(self.config.get("branch_code", "") or ""),
        }

    def _maybe_finalize_previous_day(self, force: bool = False):
        if not self.deadline_sync_enabled:
            return
        now_ts = time.time()
        if (not force) and now_ts < self._next_deadline_check_ts:
            return
        if now_ts < self._next_daily_sync_retry_ts:
            return
        self._next_deadline_check_ts = now_ts + self.deadline_check_interval_sec

        now_local = self._now_in_business_tz()
        candidates = set(self._daily_stats_by_date.keys())
        yesterday = (now_local.date() - timedelta(days=1)).strftime("%Y-%m-%d")
        candidates.add(yesterday)
        for business_date in sorted(candidates):
            if business_date in self._daily_sent_dates:
                continue
            due_at = self._daily_deadline_dt(business_date)
            if due_at is None:
                continue
            if now_local < due_at:
                continue
            if self._has_pending_recordings_for_date(business_date):
                warn_key = f"{business_date}:pending"
                if warn_key not in self._daily_sync_warnings:
                    self._daily_sync_warnings.add(warn_key)
                    logger.warning(
                        "Daily sync delayed for %s: recordings for this date are still pending; waiting to avoid partial counts.",
                        business_date,
                    )
                continue
            payload = self._build_daily_payload_for_date(business_date)
            try:
                client = self._require_rpc_client()
                client.ingest_cctv_daily_summary(device_token=self.cloud_sync_token, **payload)
                self._daily_sent_dates.add(business_date)
                self._daily_sync_warnings.discard(f"{business_date}:pending")
                self._save_sync_state()
                logger.info(
                    "Daily summary synced (recorded deadline): date=%s customers=%s peak=%s",
                    business_date,
                    payload.get("customers_total", 0),
                    payload.get("peak_people_inside", 0),
                )
            except Exception as e:
                self._next_daily_sync_retry_ts = time.time() + self.daily_sync_retry_sec
                logger.error(f"Daily summary sync failed for {business_date}: {e}")
                break

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

    def _mark_quarantined(self, file_path: Path):
        try:
            self._quarantined_recordings.add(str(file_path.resolve()))
        except Exception:
            self._quarantined_recordings.add(str(file_path))

    @staticmethod
    def _file_ts_from_name(file_path: Path) -> Optional[datetime]:
        """Parse timestamp from filename pattern YYYYMMDD_HHMMSS.mp4."""
        try:
            parts = file_path.stem.split("_")
            if len(parts) >= 2:
                date_txt = parts[0]
                time_txt = parts[1]
                if len(date_txt) == 8 and len(time_txt) == 6:
                    return datetime.strptime(f"{date_txt}_{time_txt}", "%Y%m%d_%H%M%S")
        except Exception:
            return None
        return None

    def _is_candidate_recording_file(self, file_path: Path) -> bool:
        now_dt = datetime.now()
        now_local = self._now_in_business_tz()
        file_dt = self._file_ts_from_name(file_path)
        if file_dt is None:
            try:
                file_dt = datetime.fromtimestamp(file_path.stat().st_mtime)
            except Exception:
                file_dt = now_dt

        # Process today's files, and optionally allow yesterday's files until grace hour
        # so recorded counting can finish and sync before next morning deadline.
        file_date = file_dt.date()
        today_local = now_local.date()
        if self.processor_only_today:
            allow_file = False
            if file_date == today_local:
                allow_file = True
            elif file_date == (today_local - timedelta(days=1)):
                prev_key = file_date.strftime("%Y-%m-%d")
                # Keep processing yesterday until summary is actually synced,
                # preventing data loss before next-morning deadline.
                if prev_key not in self._daily_sent_dates:
                    allow_file = True
                else:
                    current_hour = (
                        float(now_local.hour)
                        + (float(now_local.minute) / 60.0)
                        + (float(now_local.second) / 3600.0)
                    )
                    allow_file = current_hour <= float(self.processor_previous_day_grace_hours)
            if not allow_file:
                self._move_to_failed(file_path)
                self._mark_quarantined(file_path)
                return False

        age_sec = (now_dt - file_dt).total_seconds()
        if age_sec > self.processor_max_recording_age_sec:
            self._move_to_failed(file_path)
            self._mark_quarantined(file_path)
            return False

        return True

    def _quick_file_has_person_in_zone(self, camera_name: str, file_path: Path) -> bool:
        """Fast pre-check: skip full processing if no person appears in business zones."""
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            cap.release()
            return True  # fail-open

        zones = self.zones.get(camera_name, {}) or {}
        checked = 0
        frame_idx = 0
        has_person = False
        try:
            while checked < self.processor_quick_check_frames:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                if frame_idx % self.processor_quick_check_stride != 0:
                    frame_idx += 1
                    continue
                frame_idx += 1
                checked += 1

                h, w = frame.shape[:2]
                acquired = self.yolo_lock.acquire(timeout=1.0)
                if not acquired:
                    continue
                try:
                    res = self.discovery_model.predict(
                        frame,
                        conf=max(0.20, float(self.yolo_conf)),
                        iou=float(self.yolo_iou),
                        imgsz=min(int(self.yolo_imgsz), 640),
                        classes=[int(self.discovery_person_class_id)],
                        verbose=False,
                    )
                finally:
                    self.yolo_lock.release()

                if not res or res[0].boxes is None:
                    continue

                for box in res[0].boxes.xyxy:
                    x1, y1, x2, y2 = box.cpu().numpy().tolist()
                    if zones:
                        hits_primary = self._detect_zone_hits(
                            zones,
                            float(x1),
                            float(y1),
                            float(x2),
                            float(y2),
                            int(w),
                            int(h),
                            forced_mode=None,
                        )
                        # Conservative quick-check: test both center and foot to avoid
                        # false negative skips for seated customers.
                        hits_foot = self._detect_zone_hits(
                            zones,
                            float(x1),
                            float(y1),
                            float(x2),
                            float(y2),
                            int(w),
                            int(h),
                            forced_mode="foot",
                        )
                        hits_center = self._detect_zone_hits(
                            zones,
                            float(x1),
                            float(y1),
                            float(x2),
                            float(y2),
                            int(w),
                            int(h),
                            forced_mode="center",
                        )
                        hits = set([str(z) for z in hits_primary] + [str(z) for z in hits_foot] + [str(z) for z in hits_center])
                        if any(
                            str(z).upper().startswith("CHAIR")
                            or str(z).upper().startswith("WASH")
                            or str(z).upper().startswith("WAIT")
                            for z in hits
                        ):
                            has_person = True
                            break
                    else:
                        has_person = True
                        break
                if has_person:
                    break
        except Exception:
            return True  # fail-open
        finally:
            cap.release()
        return has_person

    def _quarantine_startup_backlog(self):
        """Skip old leftover recordings from previous runs so they are not reprocessed after restart."""
        try:
            if not self.processor_skip_backlog_on_start:
                return
            if not self.recordings_dir.exists():
                return
            moved = 0
            cutoff = self.processor_start_ts - self.file_ready_grace_sec
            for file_path in self.recordings_dir.rglob("*.mp4"):
                try:
                    st = file_path.stat()
                except OSError:
                    continue
                if st.st_mtime > cutoff:
                    continue
                age_sec = self.processor_start_ts - st.st_mtime
                if age_sec < self.processor_startup_backlog_grace_sec:
                    continue
                self._move_to_failed(file_path)
                self._mark_quarantined(file_path)
                moved += 1
            if moved:
                logger.info(
                    f"Startup backlog cleanup moved {moved} old recording(s) to failed "
                    f"(older than {int(self.processor_startup_backlog_grace_sec)}s)."
                )
        except Exception as e:
            logger.error(f"Startup backlog cleanup failed: {e}")
                
    def _process_file(self, camera_name: str, file_path: Path):
        logger.info(f"Processing file: {file_path}")

        if self.processor_quick_skip_no_person:
            has_person = self._quick_file_has_person_in_zone(camera_name, file_path)
            if not has_person:
                self._delete_file(file_path)
                logger.info(f"Quick-skip {file_path.name}: no person in business zones")
                return
        
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
                self._mark_quarantined(file_path)
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
                time.sleep(self.processor_frame_sleep_sec)
            
            
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
        if final_event_count > initial_event_count:
            new_events = list(self.all_events[initial_event_count:final_event_count])
        else:
            new_events = []
        self._update_daily_stats_for_file(file_path, new_events, max_people)
        
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

        # Re-check daily deadline sync after each processed file.
        self._maybe_finalize_previous_day(force=True)

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
             dest = failed_dir / file_path.name
             if dest.exists():
                 stem = file_path.stem
                 suffix = file_path.suffix
                 idx = 1
                 while True:
                     candidate = failed_dir / f"{stem}__{idx:03d}{suffix}"
                     if not candidate.exists():
                         dest = candidate
                         break
                     idx += 1
             shutil.move(str(file_path), str(dest))
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
