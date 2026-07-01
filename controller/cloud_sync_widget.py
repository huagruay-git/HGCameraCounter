"""Cloud Sync Widget - Supabase CCTV RPC integration."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PySide6.QtCore import QTime, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QTimeEdit,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)

from shared.supabase_client import CCTVSupabaseRPCClient

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None


class CloudSyncWidget(QWidget):
    """Widget for schedule-based Supabase CCTV sync."""

    log_signal = Signal(str)
    status_signal = Signal(str, str)
    command_signal = Signal(dict)  # remote command -> executed on the main thread

    def __init__(self, host: Any):
        super().__init__()
        self.host = host
        self.command_signal.connect(self._execute_command)
        self.client: Optional[CCTVSupabaseRPCClient] = None
        self.client_lock = threading.Lock()
        self.task_lock = threading.Lock()
        self.active_tasks: Dict[str, bool] = {
            "heartbeat": False,
            "realtime": False,
            "daily": False,
            "validate": False,
            "pair": False,
        }

        self.sync_enabled = False
        self.next_heartbeat_at = 0.0
        self.next_realtime_at = 0.0
        self.last_daily_sent_date = ""
        self.daily_peak_by_date: Dict[str, int] = {}
        self.prev_inside_count: Optional[int] = None

        self._init_ui()
        self._load_from_config()

        self.scheduler_timer = QTimer(self)
        self.scheduler_timer.setInterval(1000)
        self.scheduler_timer.timeout.connect(self._on_scheduler_tick)
        self.scheduler_timer.start()

        self.log_signal.connect(self._log)
        self.status_signal.connect(self._set_status)
        self.status_signal.emit("Idle", "#616161")

    # ----------------------------- UI -----------------------------

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSizeConstraint(QLayout.SetMinimumSize)

        title = QLabel("Cloud Sync - Supabase CCTV RPC")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #1B5E20;")
        layout.addWidget(title)

        subtitle = QLabel(
            "Token-first security: app sends device_token, database validates device + branch in RPC."
        )
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        layout.addWidget(self._build_connection_group())
        layout.addWidget(self._build_rules_group())
        layout.addWidget(self._build_scheduler_group())
        layout.addWidget(self._build_manual_group())

        status_row = QHBoxLayout()
        status_row.addWidget(QLabel("Status:"))
        self.status_label = QLabel("-")
        self.status_label.setStyleSheet("font-weight: bold;")
        status_row.addWidget(self.status_label)
        status_row.addStretch()
        layout.addLayout(status_row)

        layout.addWidget(QLabel("Cloud Sync Log:"))
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMinimumHeight(180)
        layout.addWidget(self.log_output)

    def _build_connection_group(self) -> QGroupBox:
        box = QGroupBox("Connection")
        form = QFormLayout(box)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("https://<project>.supabase.co")
        form.addRow("Supabase URL:", self.url_input)

        self.key_input = QLineEdit()
        self.key_input.setEchoMode(QLineEdit.Password)
        self.key_input.setPlaceholderText("anon key")
        form.addRow("Supabase Key:", self.key_input)

        self.device_token_input = QLineEdit()
        self.device_token_input.setEchoMode(QLineEdit.Password)
        self.device_token_input.setPlaceholderText("device_token from cctv_devices")
        form.addRow("Device Token:", self.device_token_input)

        self.branch_name_input = QLineEdit()
        self.branch_name_input.setPlaceholderText("optional branch name/code to report")
        form.addRow("Branch Name Reported:", self.branch_name_input)

        self.timezone_input = QLineEdit("Asia/Bangkok")
        form.addRow("Business Timezone:", self.timezone_input)

        self.one_time_code_input = QLineEdit()
        self.one_time_code_input.setPlaceholderText("CTVP-XXXXXXXX...")
        form.addRow("One-time Pairing Code:", self.one_time_code_input)

        self.device_name_input = QLineEdit()
        self.device_name_input.setPlaceholderText("MiniPC-FrontDesk-01")
        form.addRow("Pair Device Name:", self.device_name_input)

        self.device_code_input = QLineEdit()
        self.device_code_input.setPlaceholderText("unique device_code")
        form.addRow("Pair Device Code:", self.device_code_input)

        row = QHBoxLayout()
        self.save_btn = QPushButton("Save Settings")
        self.save_btn.clicked.connect(self.save_settings)
        row.addWidget(self.save_btn)

        self.validate_btn = QPushButton("Validate Token")
        self.validate_btn.clicked.connect(self.validate_token)
        row.addWidget(self.validate_btn)
        row.addStretch()

        form.addRow(row)

        pair_row = QHBoxLayout()
        self.pair_btn = QPushButton("Pair Device (Get Token)")
        self.pair_btn.clicked.connect(self.pair_device_with_code)
        pair_row.addWidget(self.pair_btn)
        pair_row.addWidget(QLabel("Use one-time code from admin to receive token once."))
        pair_row.addStretch()
        form.addRow(pair_row)
        self._expand_inputs_in_group(box)
        return box

    def _build_rules_group(self) -> QGroupBox:
        box = QGroupBox("Payload Rules")
        lay = QVBoxLayout(box)

        self.sync_realtime_check = QCheckBox("Sync realtime counts (ingest_cctv_realtime)")
        self.sync_realtime_check.setChecked(True)
        lay.addWidget(self.sync_realtime_check)

        self.sync_heartbeat_check = QCheckBox("Sync heartbeat (ingest_cctv_heartbeat)")
        self.sync_heartbeat_check.setChecked(True)
        lay.addWidget(self.sync_heartbeat_check)

        self.sync_daily_check = QCheckBox("Sync daily summary (ingest_cctv_daily_summary)")
        self.sync_daily_check.setChecked(True)
        lay.addWidget(self.sync_daily_check)

        self.include_haircuts_check = QCheckBox("Use haircut count as customers_total")
        self.include_haircuts_check.setChecked(True)
        lay.addWidget(self.include_haircuts_check)

        self.include_washes_check = QCheckBox("Include wash/wait/staff in raw_payload")
        self.include_washes_check.setChecked(True)
        lay.addWidget(self.include_washes_check)

        return box

    def _build_scheduler_group(self) -> QGroupBox:
        box = QGroupBox("Scheduler")
        form = QFormLayout(box)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.realtime_interval_spin = QSpinBox()
        self.realtime_interval_spin.setRange(5, 3600)
        self.realtime_interval_spin.setValue(15)
        form.addRow("Realtime every (sec):", self.realtime_interval_spin)

        self.heartbeat_interval_spin = QSpinBox()
        self.heartbeat_interval_spin.setRange(10, 3600)
        self.heartbeat_interval_spin.setValue(60)
        form.addRow("Heartbeat every (sec):", self.heartbeat_interval_spin)

        self.daily_time_edit = QTimeEdit()
        self.daily_time_edit.setDisplayFormat("HH:mm")
        self.daily_time_edit.setTime(QTime(23, 55))
        form.addRow("Daily summary time:", self.daily_time_edit)

        self.recorded_deadline_edit = QTimeEdit()
        self.recorded_deadline_edit.setDisplayFormat("HH:mm")
        self.recorded_deadline_edit.setTime(QTime(8, 0))
        form.addRow("Recorded next-day deadline:", self.recorded_deadline_edit)

        row = QHBoxLayout()
        self.start_auto_btn = QPushButton("Start Auto Sync")
        self.start_auto_btn.clicked.connect(self.start_auto_sync)
        row.addWidget(self.start_auto_btn)

        self.stop_auto_btn = QPushButton("Stop Auto Sync")
        self.stop_auto_btn.clicked.connect(self.stop_auto_sync)
        self.stop_auto_btn.setEnabled(False)
        row.addWidget(self.stop_auto_btn)
        row.addStretch()
        form.addRow(row)
        self._expand_inputs_in_group(box)

        return box

    def _build_manual_group(self) -> QGroupBox:
        box = QGroupBox("Manual Push")
        row = QHBoxLayout(box)

        send_hb = QPushButton("Send Heartbeat")
        send_hb.clicked.connect(lambda: self._launch_task("heartbeat", self._send_heartbeat_once))
        row.addWidget(send_hb)

        send_rt = QPushButton("Send Realtime")
        send_rt.clicked.connect(lambda: self._launch_task("realtime", self._send_realtime_once))
        row.addWidget(send_rt)

        send_daily = QPushButton("Send Daily")
        send_daily.clicked.connect(lambda: self._launch_task("daily", self._send_daily_once))
        row.addWidget(send_daily)

        row.addStretch()
        return box

    def _expand_inputs_in_group(self, root: QWidget, min_width: int = 520, min_height: int = 50):
        for line in root.findChildren(QLineEdit):
            line.setMinimumWidth(max(line.minimumWidth(), min_width))
            line.setMinimumHeight(max(line.minimumHeight(), min_height))
            line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        for combo in root.findChildren(QTimeEdit):
            combo.setMinimumWidth(max(combo.minimumWidth(), min_width))
            combo.setMinimumHeight(max(combo.minimumHeight(), min_height))
            combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        for spin in root.findChildren(QSpinBox):
            spin.setMinimumWidth(max(spin.minimumWidth(), min_width))
            spin.setMinimumHeight(max(spin.minimumHeight(), min_height))
            spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    # ----------------------------- Config -----------------------------

    def _config_obj(self) -> Any:
        if hasattr(self.host, "config"):
            return self.host.config
        return self.host

    def _config_dict(self) -> Dict[str, Any]:
        cfg_obj = self._config_obj()
        if isinstance(cfg_obj, dict):
            return cfg_obj
        if hasattr(cfg_obj, "data") and isinstance(cfg_obj.data, dict):
            return cfg_obj.data
        if hasattr(cfg_obj, "get_all"):
            data = cfg_obj.get_all()
            if isinstance(data, dict):
                return data
        return {}

    def _save_config_dict(self, cfg: Dict[str, Any]):
        cfg_obj = self._config_obj()
        if isinstance(cfg_obj, dict):
            cfg_obj.clear()
            cfg_obj.update(cfg)
            return
        if hasattr(cfg_obj, "set_all"):
            cfg_obj.set_all(cfg)
            return
        if hasattr(cfg_obj, "data") and hasattr(cfg_obj, "save"):
            cfg_obj.data = cfg
            cfg_obj.save()

    def _load_from_config(self):
        cfg = self._config_dict()
        sup_cfg = dict(cfg.get("supabase", {}) or {})
        cloud_cfg = dict(sup_cfg.get("cloud_sync", {}) or {})

        self.url_input.setText(str(sup_cfg.get("url", "") or ""))
        self.key_input.setText(str(sup_cfg.get("key", "") or ""))
        self.device_token_input.setText(str(cloud_cfg.get("device_token", "") or ""))
        self.branch_name_input.setText(
            str(cloud_cfg.get("branch_name_reported", cfg.get("branch_code", "")) or "")
        )
        self.timezone_input.setText(str(cloud_cfg.get("timezone", "Asia/Bangkok") or "Asia/Bangkok"))
        self.one_time_code_input.setText(str(cloud_cfg.get("one_time_code", "") or ""))
        self.device_name_input.setText(str(cloud_cfg.get("pair_device_name", "") or ""))
        self.device_code_input.setText(str(cloud_cfg.get("pair_device_code", "") or ""))

        self.sync_realtime_check.setChecked(bool(cloud_cfg.get("sync_realtime", True)))
        self.sync_heartbeat_check.setChecked(bool(cloud_cfg.get("sync_heartbeat", True)))
        self.sync_daily_check.setChecked(bool(cloud_cfg.get("sync_daily", True)))
        self.include_haircuts_check.setChecked(bool(cloud_cfg.get("include_haircuts_in_daily", True)))
        self.include_washes_check.setChecked(bool(cloud_cfg.get("include_washes_in_payload", True)))

        self.realtime_interval_spin.setValue(max(5, int(cloud_cfg.get("realtime_interval_sec", 15))))
        self.heartbeat_interval_spin.setValue(max(10, int(cloud_cfg.get("heartbeat_interval_sec", 60))))

        daily_txt = str(cloud_cfg.get("daily_summary_time", "23:55") or "23:55")
        parts = daily_txt.split(":")
        hh = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 23
        mm = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 55
        self.daily_time_edit.setTime(QTime(max(0, min(23, hh)), max(0, min(59, mm))))

        deadline_txt = str(
            cloud_cfg.get(
                "recorded_daily_deadline_time",
                cloud_cfg.get("daily_summary_deadline_time", "08:00"),
            )
            or "08:00"
        )
        deadline_parts = deadline_txt.split(":")
        dh = int(deadline_parts[0]) if len(deadline_parts) > 0 and deadline_parts[0].isdigit() else 8
        dm = int(deadline_parts[1]) if len(deadline_parts) > 1 and deadline_parts[1].isdigit() else 0
        self.recorded_deadline_edit.setTime(QTime(max(0, min(23, dh)), max(0, min(59, dm))))

        self.sync_enabled = bool(cloud_cfg.get("enabled", False))
        self.start_auto_btn.setEnabled(not self.sync_enabled)
        self.stop_auto_btn.setEnabled(self.sync_enabled)
        if self.sync_enabled:
            self._reset_schedule()
            self.status_signal.emit("Auto sync enabled", "#1B5E20")

    def save_settings(self):
        cfg = dict(self._config_dict())
        sup_cfg = dict(cfg.get("supabase", {}) or {})
        cloud_cfg = dict(sup_cfg.get("cloud_sync", {}) or {})

        sup_cfg["url"] = self.url_input.text().strip()
        sup_cfg["key"] = self.key_input.text().strip()

        cloud_cfg.update(
            {
                "enabled": bool(self.sync_enabled),
                "device_token": self.device_token_input.text().strip(),
                "branch_name_reported": self.branch_name_input.text().strip(),
                "timezone": self.timezone_input.text().strip() or "Asia/Bangkok",
                "one_time_code": self.one_time_code_input.text().strip(),
                "pair_device_name": self.device_name_input.text().strip(),
                "pair_device_code": self.device_code_input.text().strip(),
                "sync_realtime": self.sync_realtime_check.isChecked(),
                "sync_heartbeat": self.sync_heartbeat_check.isChecked(),
                "sync_daily": self.sync_daily_check.isChecked(),
                "include_haircuts_in_daily": self.include_haircuts_check.isChecked(),
                "include_washes_in_payload": self.include_washes_check.isChecked(),
                "realtime_interval_sec": int(self.realtime_interval_spin.value()),
                "heartbeat_interval_sec": int(self.heartbeat_interval_spin.value()),
                "daily_summary_time": self.daily_time_edit.time().toString("HH:mm"),
                "recorded_daily_deadline_time": self.recorded_deadline_edit.time().toString("HH:mm"),
            }
        )
        sup_cfg["cloud_sync"] = cloud_cfg
        cfg["supabase"] = sup_cfg

        try:
            self._save_config_dict(cfg)
            self.log_signal.emit("Settings saved.")
            self.status_signal.emit("Settings saved", "#1565C0")
            # Mirror identity to %LOCALAPPDATA% (survives a reinstall) so a re-paired/
            # re-installed device keeps the SAME device code + token instead of a new one.
            try:
                _tok = str(cloud_cfg.get("device_token", "") or "").strip()
                if _tok:
                    from shared.device_identity import save_identity
                    save_identity({
                        "device_token": _tok,
                        "device_code": str(cloud_cfg.get("pair_device_code", "") or ""),
                        "device_name": str(cloud_cfg.get("pair_device_name", "") or ""),
                        "branch_name_reported": str(cloud_cfg.get("branch_name_reported", "") or ""),
                        "timezone": str(cloud_cfg.get("timezone", "") or ""),
                        "branch_code": str(cfg.get("branch_code", "") or ""),
                    })
            except Exception:
                pass
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save settings: {e}")
            self.log_signal.emit(f"Failed to save settings: {e}")
            self.status_signal.emit("Save failed", "#B71C1C")

    # ----------------------------- Scheduler -----------------------------

    def start_auto_sync(self):
        if not self._validate_required_inputs(show_dialog=True, require_token=True):
            return
        self.sync_enabled = True
        self._reset_schedule()
        self.start_auto_btn.setEnabled(False)
        self.stop_auto_btn.setEnabled(True)
        self.save_settings()
        self.status_signal.emit("Auto sync running", "#1B5E20")
        self.log_signal.emit("Auto sync started.")

    def stop_auto_sync(self):
        self.sync_enabled = False
        self.start_auto_btn.setEnabled(True)
        self.stop_auto_btn.setEnabled(False)
        self.save_settings()
        self.status_signal.emit("Auto sync stopped", "#6D4C41")
        self.log_signal.emit("Auto sync stopped.")

    def _reset_schedule(self):
        now = time.time()
        self.next_heartbeat_at = now
        self.next_realtime_at = now

    def _on_scheduler_tick(self):
        if not self.sync_enabled:
            return
        if not self._validate_required_inputs(show_dialog=False, require_token=True):
            return
        now_ts = time.time()
        if self.sync_heartbeat_check.isChecked() and now_ts >= self.next_heartbeat_at:
            self.next_heartbeat_at = now_ts + float(self.heartbeat_interval_spin.value())
            self._launch_task("heartbeat", self._send_heartbeat_once)

        if self.sync_realtime_check.isChecked() and now_ts >= self.next_realtime_at:
            self.next_realtime_at = now_ts + float(self.realtime_interval_spin.value())
            self._launch_task("realtime", self._send_realtime_once)

        if self.sync_daily_check.isChecked() and self._should_send_daily_now():
            self._launch_task("daily", self._send_daily_once)

        # Poll for remote commands (reboot/restart/etc.) on the heartbeat cadence (min 15s).
        if now_ts >= getattr(self, "next_command_at", 0.0):
            self.next_command_at = now_ts + max(15.0, float(self.heartbeat_interval_spin.value()))
            self._launch_task("commands", self._poll_commands_once, require_token=True)

    def _should_send_daily_now(self) -> bool:
        now_local = self._now_in_business_tz()
        business_date = now_local.strftime("%Y-%m-%d")
        if self.last_daily_sent_date == business_date:
            return False
        target = self.daily_time_edit.time()
        current_hm = now_local.strftime("%H:%M")
        target_hm = target.toString("HH:mm")
        return current_hm >= target_hm

    # ----------------------------- Actions -----------------------------

    def validate_token(self):
        self._launch_task("validate", self._validate_token_once, require_token=True)

    def pair_device_with_code(self):
        if not self._validate_pairing_inputs(show_dialog=True):
            return
        self._launch_task("pair", self._pair_device_once, require_token=False)

    def _validate_token_once(self):
        client = self._require_client()
        token = self.device_token_input.text().strip()
        masked = CCTVSupabaseRPCClient.mask_token(token)
        client.test_device_token(token)
        self.log_signal.emit(f"Token validation passed ({masked}).")
        self.status_signal.emit("Token valid", "#1B5E20")

    def _pair_device_once(self):
        if not self._validate_pairing_inputs(show_dialog=False):
            raise RuntimeError("Missing pairing fields: one-time code, device name, or device code.")
        client = self._require_client()
        one_time_code = self.one_time_code_input.text().strip()
        device_name = self.device_name_input.text().strip()
        device_code = self.device_code_input.text().strip()
        timezone_name = self.timezone_input.text().strip() or "Asia/Bangkok"
        pair_meta = {"source": "HGCameraCounter", "paired_via": "cloud_sync_widget"}
        row = client.pair_cctv_device_with_enrollment(
            one_time_code=one_time_code,
            device_name=device_name,
            device_code=device_code,
            timezone_name=timezone_name,
            metadata=pair_meta,
            branch_name_reported=self._branch_name_reported(),
        )
        token = str(row.get("device_token", "") or "").strip()
        if not token:
            raise RuntimeError("Pairing succeeded but no device token returned.")
        self.device_token_input.setText(token)
        self.one_time_code_input.clear()
        self.log_signal.emit(
            f"Pairing success: device_id={row.get('device_id')} branch_id={row.get('branch_id')} "
            f"token={CCTVSupabaseRPCClient.mask_token(token)}"
        )
        self.status_signal.emit("Pairing success", "#2E7D32")
        self.save_settings()

    def _send_heartbeat_once(self):
        client = self._require_client()
        token = self.device_token_input.text().strip()
        state = self._load_runtime_state()
        status, metrics, message = self._build_heartbeat_payload(state)
        client.ingest_cctv_heartbeat(
            device_token=token,
            status=status,
            message=message,
            metrics=metrics,
            heartbeat_at=datetime.now(timezone.utc).isoformat(),
        )
        self.log_signal.emit(
            f"Heartbeat synced: status={status}, active={metrics.get('active_people', 0)}"
        )
        self.status_signal.emit("Heartbeat synced", "#2E7D32")

    def _send_realtime_once(self):
        client = self._require_client()
        token = self.device_token_input.text().strip()
        state = self._load_runtime_state()
        payload = self._build_realtime_payload(state)
        client.ingest_cctv_realtime(device_token=token, **payload)
        self.log_signal.emit(
            "Realtime synced: "
            f"in={payload['people_in']} out={payload['people_out']} inside={payload['people_inside']}"
        )
        self.status_signal.emit("Realtime synced", "#2E7D32")

    def _send_daily_once(self):
        client = self._require_client()
        token = self.device_token_input.text().strip()
        state = self._load_runtime_state()
        payload = self._build_daily_payload(state)
        client.ingest_cctv_daily_summary(device_token=token, **payload)
        self.last_daily_sent_date = payload["business_date"]
        self.log_signal.emit(
            f"Daily summary synced: date={payload['business_date']} customers={payload['customers_total']}"
        )
        self.status_signal.emit("Daily summary synced", "#2E7D32")

    # ----------------------------- Builders -----------------------------

    def _build_heartbeat_payload(self, state: Dict[str, Any]) -> Tuple[str, Dict[str, Any], str]:
        status = state.get("status", {}) or {}
        summary = state.get("summary", {}) or {}
        cameras = status.get("cameras", {}) or {}
        online_count = sum(1 for cam in cameras.values() if bool((cam or {}).get("connected")))
        total_count = len(cameras)
        running = bool(status.get("running", False))
        heartbeat_status = "ONLINE" if running else "OFFLINE"
        message = f"Runtime {'running' if running else 'stopped'}"

        metrics = {
            "active_people": int(summary.get("active_people", status.get("active_tracks", 0)) or 0),
            "haircuts": int(summary.get("haircuts", 0) or 0),
            "washes": int(summary.get("washes", 0) or 0),
            "waits": int(summary.get("waits", 0) or 0),
            "verified": int(summary.get("verified", 0) or 0),
            "camera_online": int(online_count),
            "camera_total": int(total_count),
            "count_mode": str(status.get("count_mode", "")),
            "branch_code": str(self._config_dict().get("branch_code", "")),
            "app_version": self._app_version(),
        }
        return heartbeat_status, metrics, message

    def _app_version(self) -> str:
        """Installed app version (VERSION file) — reported to HQ in each heartbeat."""
        try:
            import sys
            from pathlib import Path
            from shared.updater import read_version_file
            root = (Path(sys.executable).resolve().parent if getattr(sys, "frozen", False)
                    else Path(__file__).resolve().parent.parent)
            return read_version_file(root) or str(self._config_dict().get("version", "") or "")
        except Exception:
            return str(self._config_dict().get("version", "") or "")

    def _build_realtime_payload(self, state: Dict[str, Any]) -> Dict[str, Any]:
        status = state.get("status", {}) or {}
        summary = state.get("summary", {}) or {}
        realtime_counts = status.get("realtime_counts", {}) or {}

        people_inside = int(summary.get("active_people", status.get("active_tracks", 0)) or 0)
        people_passing = int(realtime_counts.get("waits_total", summary.get("waits", 0)) or 0)

        if self.prev_inside_count is None:
            people_in = 0
            people_out = 0
        else:
            people_in = max(0, people_inside - self.prev_inside_count)
            people_out = max(0, self.prev_inside_count - people_inside)
        self.prev_inside_count = people_inside

        unknown_rate = summary.get("role_unknown_rate")
        confidence = None
        if unknown_rate is not None:
            try:
                confidence = round(max(0.0, min(1.0, 1.0 - float(unknown_rate))) * 100.0, 2)
            except Exception:
                confidence = None

        raw_payload = {
            "source": "HGCameraCounter",
            "count_mode": status.get("count_mode"),
            "active_people": people_inside,
            "haircuts": int(summary.get("haircuts", 0) or 0),
            "washes": int(summary.get("washes", 0) or 0),
            "waits": int(summary.get("waits", 0) or 0),
            "verified": int(summary.get("verified", 0) or 0),
            "chairs_total": int(realtime_counts.get("chairs_total", 0) or 0),
            "washes_total": int(realtime_counts.get("washes_total", 0) or 0),
            "waits_total": int(realtime_counts.get("waits_total", 0) or 0),
            "camera_people": status.get("camera_people", {}),
            "camera_staff": status.get("camera_staff", {}),
            "same_person_multi_cam": status.get("same_person_multi_cam", {}),
            "dashboard_timestamp": float(state.get("timestamp", 0.0) or 0.0),
        }
        if not self.include_washes_check.isChecked():
            raw_payload.pop("washes", None)
            raw_payload.pop("washes_total", None)

        return {
            "event_time": datetime.now(timezone.utc).isoformat(),
            "people_in": people_in,
            "people_out": people_out,
            "people_inside": people_inside,
            "people_passing": people_passing,
            "confidence": confidence,
            "raw_payload": raw_payload,
            "branch_name": self._branch_name_reported(),
        }

    def _build_daily_payload(self, state: Dict[str, Any]) -> Dict[str, Any]:
        status = state.get("status", {}) or {}
        summary = state.get("summary", {}) or {}
        realtime_counts = status.get("realtime_counts", {}) or {}

        now_local = self._now_in_business_tz()
        business_date = now_local.strftime("%Y-%m-%d")
        people_inside = int(summary.get("active_people", status.get("active_tracks", 0)) or 0)
        prev_peak = int(self.daily_peak_by_date.get(business_date, 0))
        peak_people_inside = max(prev_peak, people_inside)
        self.daily_peak_by_date[business_date] = peak_people_inside

        customers_total = int(summary.get("haircuts", 0) or 0)
        if not self.include_haircuts_check.isChecked():
            customers_total = int(summary.get("active_people", status.get("active_tracks", 0)) or 0)

        cfg = self._config_dict()
        runtime_cfg = cfg.get("runtime", {}) or {}
        open_time = runtime_cfg.get("business_hours_start")
        close_time = runtime_cfg.get("business_hours_end")
        if open_time and len(str(open_time)) > 5:
            open_time = str(open_time)[:5]
        if close_time and len(str(close_time)) > 5:
            close_time = str(close_time)[:5]

        raw_payload = {
            "source": "HGCameraCounter",
            "summary": {
                "haircuts": int(summary.get("haircuts", 0) or 0),
                "washes": int(summary.get("washes", 0) or 0),
                "waits": int(summary.get("waits", 0) or 0),
                "verified": int(summary.get("verified", 0) or 0),
                "total_events": int(summary.get("total_events", 0) or 0),
            },
            "realtime_counts": {
                "chairs_total": int(realtime_counts.get("chairs_total", 0) or 0),
                "washes_total": int(realtime_counts.get("washes_total", 0) or 0),
                "waits_total": int(realtime_counts.get("waits_total", 0) or 0),
            },
        }
        if not self.include_washes_check.isChecked():
            raw_payload["summary"].pop("washes", None)
            raw_payload["summary"].pop("waits", None)
            raw_payload["realtime_counts"].pop("washes_total", None)
            raw_payload["realtime_counts"].pop("waits_total", None)

        return {
            "business_date": business_date,
            "customers_total": max(0, customers_total),
            "peak_people_inside": max(0, peak_people_inside),
            "open_time": open_time,
            "close_time": close_time,
            "note": "Daily summary from HGCameraCounter Cloud Sync",
            "raw_payload": raw_payload,
            "branch_name": self._branch_name_reported(),
        }

    # ----------------------------- Runtime state -----------------------------

    def _project_root(self) -> Path:
        if hasattr(self.host, "_project_root") and callable(getattr(self.host, "_project_root")):
            try:
                return Path(self.host._project_root())
            except Exception:
                pass
        return Path(__file__).resolve().parent.parent

    def _dashboard_state_path(self) -> Path:
        if hasattr(self.host, "dashboard_state_file"):
            try:
                return Path(self.host.dashboard_state_file)
            except Exception:
                pass
        cfg = self._config_dict()
        rel = (cfg.get("paths", {}) or {}).get("dashboard_state", "runtime/dashboard_state.json")
        path = Path(rel)
        if path.is_absolute():
            return path
        return (self._project_root() / path).resolve()

    def _load_runtime_state(self) -> Dict[str, Any]:
        path = self._dashboard_state_path()
        if not path.exists():
            return {"status": {}, "summary": {}, "timestamp": 0.0}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            self.log_signal.emit(f"Failed reading dashboard_state: {e}")
            return {"status": {}, "summary": {}, "timestamp": 0.0}

        if isinstance(raw, dict) and "status" in raw and "summary" in raw:
            state = raw
        elif isinstance(raw, dict):
            # backward compatibility for flat payload
            state = {
                "timestamp": raw.get("timestamp", time.time()),
                "status": raw.get("status", {}),
                "summary": raw.get("summary", {}),
            }
        else:
            state = {"status": {}, "summary": {}, "timestamp": 0.0}

        status = state.get("status", {}) or {}
        summary = state.get("summary", {}) or {}
        inside = int(summary.get("active_people", status.get("active_tracks", 0)) or 0)
        biz_date = self._now_in_business_tz().strftime("%Y-%m-%d")
        self.daily_peak_by_date[biz_date] = max(int(self.daily_peak_by_date.get(biz_date, 0)), inside)
        return state

    # ----------------------------- Client / task helpers -----------------------------

    def _require_client(self) -> CCTVSupabaseRPCClient:
        url = self.url_input.text().strip()
        key = self.key_input.text().strip()
        with self.client_lock:
            if self.client is not None and self.client.url == url and self.client.key == key:
                if self.client.ensure_connected():
                    return self.client
            self.client = CCTVSupabaseRPCClient(url=url, key=key)
            if not self.client.ensure_connected():
                raise RuntimeError("Cannot connect to Supabase (check URL/Key).")
            return self.client

    def _poll_commands_once(self):
        """Pull pending remote commands (runs in a worker thread) and marshal each to
        the main thread for execution."""
        client = self._require_client()
        token = self.device_token_input.text().strip()
        if not token:
            return
        cmds = client.get_cctv_device_commands(token)
        if cmds:
            self.log_signal.emit(f"Remote command(s) received: {len(cmds)}")
        for cmd in cmds:
            self.command_signal.emit(cmd)

    def _execute_command(self, cmd: dict):
        """Execute one remote command on the main thread, then ack the outcome."""
        import subprocess
        import urllib.request
        token = self.device_token_input.text().strip()
        cmd_id = cmd.get("id")
        name = str(cmd.get("command", "")).strip().lower()
        args = cmd.get("args") if isinstance(cmd.get("args"), dict) else {}

        def ack(status: str, detail: str):
            try:
                self._require_client().ack_cctv_device_command(token, cmd_id, status, detail)
            except Exception as e:
                self.log_signal.emit(f"ack failed: {e}")

        self.log_signal.emit(f"Executing remote command: {name} ({cmd_id})")
        try:
            if name in ("reboot", "restart_pc"):
                delay = int(args.get("delay_sec", 15))
                ack("done", f"rebooting in {delay}s")
                subprocess.Popen(["shutdown", "/r", "/f", "/t", str(delay), "/c", "HGCC remote reboot"])
            elif name == "shutdown":
                delay = int(args.get("delay_sec", 15))
                ack("done", f"shutting down in {delay}s")
                subprocess.Popen(["shutdown", "/s", "/f", "/t", str(delay), "/c", "HGCC remote shutdown"])
            elif name == "restart_app":
                ack("done", "restarting app")
                if hasattr(self.host, "_relaunch_app"):
                    self.host._relaunch_app()
                else:
                    self.log_signal.emit("restart_app: relauncher unavailable")
            elif name in ("power_cycle", "wake"):
                # Drive a smart-plug / WOL helper via an HTTP URL carried in args.url
                # (e.g. a Shelly/Tasmota local endpoint). A fully-off PC cannot run this.
                url = str(args.get("url", "") or "").strip()
                if not url:
                    ack("failed", "no plug url in args (e.g. {\"url\":\"http://<plug-ip>/relay/0?turn=off\"})")
                else:
                    urllib.request.urlopen(url, timeout=10)
                    ack("done", f"plug/WOL triggered: {url[:48]}")
            elif name == "update_now":
                ack("done", "update check triggered")
                if hasattr(self.host, "_start_update_worker"):
                    self.host._start_update_worker(do_download=True)
            elif name == "ping":
                ack("done", "pong")
            else:
                ack("failed", f"unknown command: {name}")
        except Exception as e:
            self.log_signal.emit(f"command '{name}' failed: {e}")
            ack("failed", str(e))

    def _launch_task(self, name: str, fn, require_token: bool = True):
        with self.task_lock:
            if self.active_tasks.get(name, False):
                return
            self.active_tasks[name] = True

        def _runner():
            try:
                if not self._validate_required_inputs(show_dialog=False, require_token=require_token):
                    if require_token:
                        raise RuntimeError("Missing Supabase URL/Key/device_token.")
                    raise RuntimeError("Missing Supabase URL/Key.")
                fn()
            except Exception as e:
                self.log_signal.emit(f"{name} failed: {e}")
                self.status_signal.emit(f"{name} failed", "#B71C1C")
            finally:
                with self.task_lock:
                    self.active_tasks[name] = False

        threading.Thread(target=_runner, daemon=True).start()

    def _validate_required_inputs(self, show_dialog: bool, require_token: bool) -> bool:
        missing = []
        if not self.url_input.text().strip():
            missing.append("Supabase URL")
        if not self.key_input.text().strip():
            missing.append("Supabase Key")
        if require_token and not self.device_token_input.text().strip():
            missing.append("Device Token")
        if missing and show_dialog:
            QMessageBox.warning(self, "Missing Inputs", f"Please fill: {', '.join(missing)}")
        return not missing

    def _validate_pairing_inputs(self, show_dialog: bool) -> bool:
        missing = []
        if not self.one_time_code_input.text().strip():
            missing.append("One-time Pairing Code")
        if not self.device_name_input.text().strip():
            missing.append("Pair Device Name")
        if not self.device_code_input.text().strip():
            missing.append("Pair Device Code")
        if missing and show_dialog:
            QMessageBox.warning(self, "Missing Pairing Inputs", f"Please fill: {', '.join(missing)}")
        return not missing

    # ----------------------------- Utility -----------------------------

    def _branch_name_reported(self) -> str:
        txt = self.branch_name_input.text().strip()
        if txt:
            return txt
        cfg = self._config_dict()
        return str(cfg.get("branch_code", "") or "")

    def _now_in_business_tz(self) -> datetime:
        tz_name = self.timezone_input.text().strip() or "Asia/Bangkok"
        if ZoneInfo is not None:
            try:
                return datetime.now(ZoneInfo(tz_name))
            except Exception:
                pass
        return datetime.now()

    def _set_status(self, text: str, color: str):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"font-weight: bold; color: {color};")

    def _log(self, message: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_output.append(f"[{ts}] {message}")

    def shutdown(self):
        self.sync_enabled = False
        if hasattr(self, "scheduler_timer") and self.scheduler_timer is not None:
            self.scheduler_timer.stop()
