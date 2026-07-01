"""
Centralized configuration management
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

# Locked Supabase project URL — every device/branch uses this one project. The Setup
# Wizard shows it read-only so a branch can't accidentally point at the wrong backend.
LOCKED_SUPABASE_URL = "https://doafupjlqkydaoxmsqtc.supabase.co"


def _app_base() -> Path:
    """Base dir for relative paths: project root (source) or the exe folder (frozen)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


try:
    from shared.secure import decrypt_secret, encrypt_secret, is_encrypted
except Exception:  # fallback when imported outside the package
    try:
        from .secure import decrypt_secret, encrypt_secret, is_encrypted
    except Exception:
        decrypt_secret = encrypt_secret = is_encrypted = None

# Fields holding machine-bound secrets: stored DPAPI-encrypted at rest and
# decrypted transparently on load. See shared/secure.py.
_SECRET_PATHS = (
    ("supabase", "key"),
    ("supabase", "cloud_sync", "device_token"),
)


def _decrypt_secrets(obj):
    """Recursively decrypt any 'enc:dpapi:*' values; leave failures encrypted."""
    if is_encrypted is None:
        return obj
    if isinstance(obj, dict):
        return {k: _decrypt_secrets(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decrypt_secrets(v) for v in obj]
    if is_encrypted(obj):
        try:
            return decrypt_secret(obj)
        except Exception as e:
            print(f"[config] WARNING: secret decrypt failed (config from another machine?): {e}")
            return obj
    return obj


class Config:
    """Configuration loader and manager"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config from YAML or JSON file
        
        Args:
            config_path: path to config file (YAML or JSON)
                        if None, uses data/config/config.yaml
        """
        raw_path = Path(config_path or "data/config/config.yaml")
        if not raw_path.is_absolute():
            raw_path = _app_base() / raw_path
        self.config_path = str(raw_path)
        self.data: Dict[str, Any] = {}
        self.load()
        # Transparently decrypt machine-bound secrets (DPAPI); callers see plaintext.
        self.data = _decrypt_secrets(self.data)
    
    def load(self):
        """Load configuration from file"""
        defaults = self._get_default_config()
        if not os.path.exists(self.config_path):
            print(f"เนยย เนเธย  Config file not found: {self.config_path}")
            self.data = defaults
            return
        
        ext = Path(self.config_path).suffix.lower()
        
        try:
            if ext == '.yaml' or ext == '.yml':
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    loaded = yaml.safe_load(f)
                    if loaded is None:
                        self.data = defaults
                    elif isinstance(loaded, dict):
                        self.data = self._merge_defaults(defaults, loaded)
                    else:
                        raise ValueError(f"Config root must be a mapping/dict, got {type(loaded).__name__}")
            elif ext == '.json':
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        self.data = self._merge_defaults(defaults, loaded)
                    else:
                        raise ValueError(f"Config root must be a mapping/dict, got {type(loaded).__name__}")
            else:
                print(f"เนยย เนเธย  Unsupported config format: {ext}")
                self.data = defaults
        except Exception as e:
            print(f"เนยย Error loading config: {e}")
            self.data = defaults
    
    def save(self):
        """Save configuration to file"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        payload = self.data
        if isinstance(payload, Config):
            payload = payload.get_all()
        if not isinstance(payload, dict):
            raise TypeError(f"Config.save expected dict data, got {type(payload).__name__}")

        # Re-encrypt machine-bound secrets so they are never written to disk in plaintext.
        payload = self._encrypt_secret_paths(payload)
        
        try:
            if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    yaml.safe_dump(payload, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            else:
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"เนยย Error saving config: {e}")
            raise
    
    def _encrypt_secret_paths(self, payload):
        """Return a copy of payload with secret fields DPAPI-encrypted for disk."""
        if encrypt_secret is None or not isinstance(payload, dict):
            return payload
        import copy
        out = copy.deepcopy(payload)
        for path in _SECRET_PATHS:
            d = out
            for p in path[:-1]:
                d = d.get(p) if isinstance(d, dict) else None
                if not isinstance(d, dict):
                    break
            if isinstance(d, dict):
                leaf = path[-1]
                val = d.get(leaf)
                if isinstance(val, str) and val and not is_encrypted(val):
                    try:
                        d[leaf] = encrypt_secret(val)
                    except Exception as e:
                        print(f"[config] WARNING: could not encrypt {'.'.join(path)}: {e}")
        return out

    def get(self, key: str, default=None) -> Any:
        """Get config value"""
        return self.data.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set config value"""
        self.data[key] = value
    
    def get_all(self) -> Dict[str, Any]:
        """Get all config data"""
        return self.data
    
    def set_all(self, data: Dict[str, Any]):
        """Set all config data and save"""
        if isinstance(data, Config):
            data = data.get_all()
        if not isinstance(data, dict):
            raise TypeError(f"Config.set_all expected dict, got {type(data).__name__}")
        self.data = self._merge_defaults(self._get_default_config(), data)
        self.save()
    
    def __getitem__(self, key: str) -> Any:
        return self.data[key]
    
    def __setitem__(self, key: str, value: Any):
        self.data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def __iter__(self):
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)
    
    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """Get default configuration"""
        defaults = {
            "project_name": "HG Camera Counter",
            "version": "0.3.2",
            "branch_code": os.getenv("BRANCH_CODE", "DEMO"),
            # When true, the dashboard auto-starts counting shortly after it opens
            # (unattended boot recovery). The Startup launcher also passes --autostart
            # which both enables this and skips the PIN on the machine-bound device.
            "auto_start_service": False,
            # AI model updates pulled from a public manifest (the "อัปเดตโมเดล" tab).
            "models": {
                "manifest_url": os.getenv("MODELS_MANIFEST_URL", ""),
            },
            "supabase": {
                "url": LOCKED_SUPABASE_URL,
                "key": os.getenv("SUPABASE_ANON_KEY", ""),
            },
            "cameras": {},
            "yolo": {
                "model": "best.pt",
                "discovery_model": "yolov8n.pt",
                "worker_model": "best.pt",
                "auto_switch_discovery_on_schema_mismatch": False,
                "conf": 0.35,
                "iou": 0.5,
                "imgsz": 640,
            },
            "runtime": {
                "target_fps": 10,
                "max_workers": 4,
                "tier2_clip_duration_sec": 4.0,
                "service_vacant_grace_sec": 6.0,
                "haircut_count_zones": {},
                "enable_chair_service_classifier": False,
                "chair_service_classifier_model": "models/chair_service_cls.pt",
                "chair_service_classifier_positive_labels": ["haircut"],
                "chair_service_classifier_min_conf": 0.55,
                "chair_service_classifier_imgsz": 224,
                "chair_service_autotrain_enabled": True,
                "chair_service_autotrain_interval_hours": 6.0,
                "chair_service_autotrain_min_positive": 20,
                "chair_service_autotrain_min_negative": 20,
                "chair_service_autotrain_positive_dir": "data/performance_feedback/haircut",
                "chair_service_autotrain_negative_dirs": [
                    "data/performance_feedback/no haircut",
                ],
                "chair_service_autotrain_epochs": 30,
                "chair_service_autotrain_batch": 32,
                "chair_service_autotrain_imgsz": 224,
                "chair_service_autotrain_patience": 12,
                "chair_service_autotrain_workers": 2,
                "chair_service_autotrain_train_split": 0.85,
                "chair_service_autotrain_device": "auto",
                "chair_service_autotrain_timeout_min": 120,
                "recorder_enable_person_zone_gate": True,
                "recorder_probe_interval_sec": 1.0,
                "recorder_clip_duration_sec": 5.0,
                "recorder_trigger_cooldown_sec": 8.0,
                "recorder_same_person_dist_norm": 0.08,
                "recorder_same_person_memory_sec": 180.0,
                "recorder_required_stable_hits": 2,
                "recorder_person_conf": 0.30,
                "recorder_person_iou": 0.45,
                "recorder_person_imgsz": 384,
                "processor_skip_backlog_on_start": True,
                "processor_only_today": True,
                "processor_max_recording_age_sec": 86400.0,
                "processor_quick_skip_no_person": True,
                "processor_quick_check_frames": 18,
                "processor_quick_check_stride": 12,
                "processor_frame_sleep_sec": 0.03,
                "enable_reid": True,
                "staff_event_similarity_threshold": 0.72,
                "enable_role_db_reid": True,
                "role_db_barber_threshold": 0.78,
                "role_db_customer_threshold": 0.76,
                "role_db_customer_margin": 0.04,
                "role_db_customer_override_staff": True,
                "role_db_max_embeddings_per_role": 512,
                "enable_business_hours_guard": False,
                "business_hours_start": "09:00",
                "business_hours_end": "22:00",
                "restore_dashboard_state_on_startup": False,
                "restore_dashboard_state_max_age_sec": 900.0,
                "enable_event_reid_dedupe": True,
                "haircut_event_dedupe_window_sec": 900.0,
                "haircut_event_dedupe_similarity": 0.82,
                "event_snapshot_fallback_to_event_bbox": True,
                "event_same_seat_cooldown_sec": 1800.0,
                "enable_no_haircut_feedback_autocopy": True,
                "no_haircut_feedback_cooldown_sec": 20.0,
            },
            "paths": {
                "models": "models",
                "zones": "data/zones",
                "staff_gallery": "data/staff_gallery",
                "staff_db": "data/staff_gallery/staff_db.json",
                "role_db": "data/staff_gallery/barber_customer_db.json",
                "customer_by_admin": "data/customer_by_admin",
                "customer_wash_by_admin": "data/customer_wash_by_admin",
                "runtime_settings_override": "runtime/runtime_settings.override.json",
                "reports": "reports",
                "snapshots": "snapshots",
                "logs": "logs",
            },
            "watchdog": {
                "enabled": False,
                "check_interval_sec": 30.0,
                "liveness_file": "",
                "stale_after_sec": 300.0,
                "boot_grace_sec": 180.0,
                "reboot_on_missing_file": False,
                "require_seen_alive": True,
                "consecutive_stale_required": 2,
                "min_seconds_between_reboots": 900.0,
                "max_reboots_per_day": 5,
                "active_hours": "",
                "reboot_delay_sec": 30,
                "reboot_message": "HGCC watchdog: counting runtime unresponsive - auto restart",
                "dry_run": False,
                "state_file": "data/state/watchdog_state.json",
            },
        }
        try:
            template_path = _app_base() / "data" / "config" / "config.template.yaml"
            if template_path.exists():
                tmpl = yaml.safe_load(template_path.read_text(encoding="utf-8")) or {}
                if isinstance(tmpl, dict):
                    defaults = Config._merge_defaults(tmpl, defaults)
        except Exception:
            pass
        return defaults

    @staticmethod
    def _merge_defaults(defaults: Any, incoming: Any) -> Any:
        """Deep merge incoming over defaults; ignore None in incoming to preserve defaults."""
        if isinstance(defaults, dict):
            out = dict(defaults)
            if not isinstance(incoming, dict):
                return out
            for k, v in incoming.items():
                if v is None:
                    continue
                if isinstance(out.get(k), dict) and isinstance(v, dict):
                    out[k] = Config._merge_defaults(out[k], v)
                else:
                    out[k] = v
            return out
        return defaults if incoming is None else incoming

