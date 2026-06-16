import os
import time
import json
import signal
import subprocess
import threading
import logging
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import cv2
from ultralytics import YOLO

CONFIG_PATH = "data/config/config.yaml"

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import Config
from shared.logger import setup_logger
from shared.ffmpeg_manager import find_ffmpeg_binary, ensure_ffmpeg_available


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUSINESS_ZONE_PREFIXES = ("CHAIR", "WASH", "WAIT")


def _cfg_bool(v, default=False) -> bool:
    if v is None:
        return bool(default)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "on"}
    return bool(v)


def _normalize_label(name: str) -> str:
    s = str(name or "").strip().lower()
    return "".join(ch if ch.isalnum() else "_" for ch in s)


def _label_contains_tokens(name: str, tokens: List[str]) -> bool:
    normalized = _normalize_label(name)
    return all(tok in normalized for tok in tokens)


def _point_in_polygon(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
    if not poly:
        return False
    inside = False
    p1x, p1y = poly[0]
    for i in range(1, len(poly) + 1):
        p2x, p2y = poly[i % len(poly)]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    xinters = None
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y + 1e-9) + p1x
                    if p1x == p2x or (xinters is not None and x <= xinters):
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


class PersonZoneGate:
    """Shared YOLO probe model to detect people quickly for recorder gating."""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.model: Optional[YOLO] = None
        self.model_lock = threading.Lock()
        self.enabled = False
        self.person_class_id = 0
        self.conf = 0.30
        self.iou = 0.45
        self.imgsz = 384
        self._init_from_config()

    def _resolve_model_path(self, model_name: str) -> str:
        model_name = str(model_name or "yolov8n.pt")
        p = Path(model_name)
        if p.is_absolute():
            return str(p)
        models_dir = Path(str((self.config.get("paths", {}) or {}).get("models", "models")))
        if not models_dir.is_absolute():
            models_dir = PROJECT_ROOT / models_dir
        candidate = models_dir / p.name
        if candidate.exists():
            return str(candidate)
        return model_name

    def _configure_class_schema(self):
        if self.model is None:
            return
        try:
            names = getattr(self.model, "names", None)
            names_dict: Dict[int, str] = {}
            if isinstance(names, dict):
                for k, v in names.items():
                    try:
                        names_dict[int(k)] = str(v)
                    except Exception:
                        continue
            elif isinstance(names, (list, tuple)):
                for i, v in enumerate(names):
                    names_dict[i] = str(v)
            pid_cfg = int((self.config.get("yolo", {}) or {}).get("person_class_id", 0))
            if pid_cfg in names_dict:
                self.person_class_id = pid_cfg
            else:
                fallback = None
                for idx, nm in names_dict.items():
                    if _label_contains_tokens(nm, ["person"]):
                        fallback = idx
                        break
                self.person_class_id = int(fallback) if fallback is not None else pid_cfg
        except Exception:
            self.person_class_id = int((self.config.get("yolo", {}) or {}).get("person_class_id", 0))

    def _init_from_config(self):
        runtime_cfg = self.config.get("runtime", {}) or {}
        yolo_cfg = self.config.get("yolo", {}) or {}
        self.enabled = _cfg_bool(runtime_cfg.get("recorder_enable_person_zone_gate", True), True)
        self.conf = float(runtime_cfg.get("recorder_person_conf", max(0.2, float(yolo_cfg.get("conf", 0.35)))))
        self.iou = float(runtime_cfg.get("recorder_person_iou", 0.45))
        self.imgsz = int(runtime_cfg.get("recorder_person_imgsz", 384))
        if not self.enabled:
            self.logger.info("Recorder person-zone gate disabled by config")
            return

        model_name = yolo_cfg.get("discovery_model", yolo_cfg.get("model", "yolov8n.pt"))
        model_path = self._resolve_model_path(str(model_name))
        try:
            self.model = YOLO(model_path)
            self._configure_class_schema()
            self.logger.info(
                f"Recorder probe model ready: {model_path} person_class_id={self.person_class_id} "
                f"conf={self.conf:.2f} imgsz={self.imgsz}"
            )
        except Exception as e:
            self.enabled = False
            self.model = None
            self.logger.error(f"Recorder probe model init failed: {e}")

    def is_ready(self) -> bool:
        return self.enabled and self.model is not None

    def detect_people(self, frame) -> List[Tuple[float, float, float, float, float]]:
        if (not self.is_ready()) or frame is None or getattr(frame, "size", 0) == 0:
            return []
        with self.model_lock:
            results = self.model.predict(
                frame,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                classes=[int(self.person_class_id)],
                verbose=False,
            )
        out: List[Tuple[float, float, float, float, float]] = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i, b in enumerate(boxes.xyxy):
                x1, y1, x2, y2 = b.cpu().numpy().tolist()
                conf = float(boxes.conf[i]) if boxes.conf is not None else 0.0
                out.append((float(x1), float(y1), float(x2), float(y2), conf))
        out.sort(key=lambda x: x[4], reverse=True)
        return out


class RTSPRecorder:
    def __init__(
        self,
        camera_name: str,
        rtsp_url: str,
        output_dir: str,
        segment_time: int = 60,
        ffmpeg_path: Optional[str] = None,
        zone_gate: Optional[PersonZoneGate] = None,
        runtime_cfg: Optional[Dict] = None,
    ):
        self.camera_name = camera_name
        self.rtsp_url = rtsp_url
        self.output_dir = Path(output_dir) / camera_name
        self.segment_time = int(segment_time)
        self.ffmpeg_path = ffmpeg_path
        self.process: Optional[subprocess.Popen] = None
        self.running = False
        self.logger = logging.getLogger(f"Recorder.{camera_name}")
        self.zone_gate = zone_gate
        self.runtime_cfg = runtime_cfg or {}

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_handle = None
        self.probe_cap = None

        self.enable_business_hours_guard = _cfg_bool(
            self.runtime_cfg.get("enable_business_hours_guard", False),
            False,
        )
        self.business_hours_start = str(self.runtime_cfg.get("business_hours_start", "09:00"))
        self.business_hours_end = str(self.runtime_cfg.get("business_hours_end", "22:00"))

        self.enable_person_zone_gate = (
            _cfg_bool(self.runtime_cfg.get("recorder_enable_person_zone_gate", True), True)
            and self.zone_gate is not None
            and self.zone_gate.is_ready()
        )
        self.probe_interval_sec = max(0.2, float(self.runtime_cfg.get("recorder_probe_interval_sec", 1.0)))
        self.clip_duration_sec = max(1.0, float(self.runtime_cfg.get("recorder_clip_duration_sec", 5.0)))
        self.trigger_cooldown_sec = max(0.5, float(self.runtime_cfg.get("recorder_trigger_cooldown_sec", 8.0)))
        self.same_person_dist_norm = min(1.0, max(0.01, float(self.runtime_cfg.get("recorder_same_person_dist_norm", 0.08))))
        self.same_person_memory_sec = max(1.0, float(self.runtime_cfg.get("recorder_same_person_memory_sec", 180.0)))
        self.required_stable_hits = max(1, int(self.runtime_cfg.get("recorder_required_stable_hits", 2)))
        self.clip_ffmpeg_timeout_sec = max(5.0, self.clip_duration_sec + 12.0)

        self._last_record_ts = 0.0
        self._last_identity: Optional[Tuple[float, float]] = None
        self._last_identity_ts = 0.0
        self._stable_hits = 0
        self._record_lock = threading.Lock()
        self._probe_fail_count = 0

        self.business_zones = self._load_business_zones()

    def _resolve_ffmpeg_binary(self) -> Optional[str]:
        cfg_like = {"runtime": {"ffmpeg_path": self.ffmpeg_path}} if self.ffmpeg_path else None
        return find_ffmpeg_binary(cfg_like, project_root=PROJECT_ROOT)

    @staticmethod
    def _parse_hhmm(value: str) -> Optional[dt_time]:
        try:
            txt = str(value or "").strip()
            if not txt:
                return None
            return datetime.strptime(txt, "%H:%M").time()
        except Exception:
            return None

    def _is_within_business_hours(self) -> bool:
        if not self.enable_business_hours_guard:
            return True
        now_dt = datetime.now()
        start_t = self._parse_hhmm(self.business_hours_start)
        end_t = self._parse_hhmm(self.business_hours_end)
        if start_t is None or end_t is None:
            return True
        now_t = now_dt.time()
        if start_t <= end_t:
            return start_t <= now_t <= end_t
        return now_t >= start_t or now_t <= end_t

    def _load_business_zones(self) -> Dict[str, List[Tuple[float, float]]]:
        path = PROJECT_ROOT / "data" / "zones" / f"zones_{self.camera_name}.json"
        zones: Dict[str, List[Tuple[float, float]]] = {}
        if not path.exists():
            self.logger.warning(f"Zones not found for recorder gate: {path}")
            return zones
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                return zones
            for item in data:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                points = item.get("polygon_json") or item.get("points") or []
                if not name or (not any(name.upper().startswith(p) for p in BUSINESS_ZONE_PREFIXES)):
                    continue
                poly: List[Tuple[float, float]] = []
                for p in points:
                    if isinstance(p, dict):
                        poly.append((float(p.get("x", 0.0)), float(p.get("y", 0.0)))
                                   )
                    elif isinstance(p, (list, tuple)) and len(p) >= 2:
                        poly.append((float(p[0]), float(p[1])))
                if len(poly) >= 3:
                    zones[name] = poly
            self.logger.info(f"Recorder gate zones for {self.camera_name}: {list(zones.keys())}")
        except Exception as e:
            self.logger.error(f"Failed loading zones for recorder gate {self.camera_name}: {e}")
        return zones

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        mode = "smart-gated" if self.enable_person_zone_gate else "continuous-segment"
        self.logger.info(f"Recorder started for {self.camera_name} mode={mode}")

    def stop(self):
        self.running = False
        self._stop_ffmpeg()
        self._release_probe_cap()
        if self.log_handle:
            try:
                self.log_handle.close()
            except Exception:
                pass
            self.log_handle = None
        self.logger.info(f"Recorder stopped for {self.camera_name}")

    def _release_probe_cap(self):
        if self.probe_cap is not None:
            try:
                self.probe_cap.release()
            except Exception:
                pass
            self.probe_cap = None

    def _stop_ffmpeg(self):
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None
        self._kill_orphans()

    def _kill_orphans(self):
        if os.name == "nt":
            return
        try:
            import shlex
            safe_url = shlex.quote(self.rtsp_url)
            cmd = f"pgrep -f {safe_url}"
            try:
                pids = subprocess.check_output(cmd, shell=True).decode().split()
            except subprocess.CalledProcessError:
                return
            my_pid = str(os.getpid())
            for pid in pids:
                if pid == my_pid:
                    continue
                try:
                    subprocess.call(["kill", "-9", pid])
                except Exception:
                    pass
        except Exception as e:
            self.logger.error(f"Error killing orphan ffmpeg: {e}")

    def _ensure_log_handle(self):
        if self.log_handle is None or self.log_handle.closed:
            log_file = self.output_dir / "ffmpeg.log"
            self.log_handle = open(log_file, "a", encoding="utf-8")

    def _start_continuous_ffmpeg_segment(self):
        self._stop_ffmpeg()
        ffmpeg_bin = self._resolve_ffmpeg_binary()
        if not ffmpeg_bin:
            self.logger.error("ffmpeg not found for recorder")
            return
        filename_pattern = str(self.output_dir / "%Y%m%d_%H%M%S.mp4")
        cmd = [
            ffmpeg_bin,
            "-y",
            "-rtsp_transport", "tcp",
            "-i", self.rtsp_url,
            "-c:v", "copy",
            "-an",
            "-f", "segment",
            "-segment_time", str(self.segment_time),
            "-segment_format", "mp4",
            "-reset_timestamps", "1",
            "-strftime", "1",
            filename_pattern,
        ]
        self._ensure_log_handle()
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=self.log_handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
            self.logger.info(f"Continuous recorder started: {self.camera_name}")
        except Exception as e:
            self.logger.error(f"Failed to start continuous recorder: {e}")
            self.process = None

    def _record_short_clip_once(self):
        ffmpeg_bin = self._resolve_ffmpeg_binary()
        if not ffmpeg_bin:
            self.logger.error("ffmpeg not found for short clip recording")
            return

        ts_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = self.output_dir / f"{ts_name}.mp4"
        cmd = [
            ffmpeg_bin,
            "-y",
            "-rtsp_transport", "tcp",
            "-i", self.rtsp_url,
            "-t", str(self.clip_duration_sec),
            "-c:v", "copy",
            "-an",
            "-movflags", "+faststart",
            str(out_file),
        ]
        self._ensure_log_handle()
        self.logger.info(
            f"Recording short clip for {self.camera_name}: {self.clip_duration_sec:.1f}s -> {out_file.name}"
        )
        try:
            with self._record_lock:
                proc = subprocess.Popen(
                    cmd,
                    stdout=self.log_handle,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                )
                self.process = proc
                proc.wait(timeout=self.clip_ffmpeg_timeout_sec)
        except subprocess.TimeoutExpired:
            self.logger.warning(f"Short clip timeout, killing ffmpeg: {self.camera_name}")
            try:
                if self.process:
                    self.process.kill()
            except Exception:
                pass
        except Exception as e:
            self.logger.error(f"Failed to record short clip: {e}")
        finally:
            self.process = None

    def _ensure_probe_cap(self) -> bool:
        if self.probe_cap is not None and self.probe_cap.isOpened():
            return True
        self._release_probe_cap()
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            return False
        self.probe_cap = cap
        return True

    def _same_identity(self, a: Optional[Tuple[float, float]], b: Optional[Tuple[float, float]]) -> bool:
        if a is None or b is None:
            return False
        dx = float(a[0] - b[0])
        dy = float(a[1] - b[1])
        return (dx * dx + dy * dy) <= (self.same_person_dist_norm * self.same_person_dist_norm)

    def _identity_in_business_zone(self, boxes: List[Tuple[float, float, float, float, float]], frame_w: int, frame_h: int) -> Optional[Tuple[float, float]]:
        if frame_w <= 0 or frame_h <= 0 or not boxes:
            return None
        for x1, y1, x2, y2, _conf in boxes:
            cx_n = ((x1 + x2) * 0.5) / float(frame_w)
            foot_n = max(y1, y2) / float(frame_h)
            center_n = ((y1 + y2) * 0.5) / float(frame_h)
            foot_pt = (cx_n, foot_n)
            center_pt = (cx_n, center_n)
            if not self.business_zones:
                return foot_pt
            for poly in self.business_zones.values():
                # Conservative gate: accept either foot or center in zone.
                # Seated customers are often missed when foot point leaves polygon.
                if _point_in_polygon(foot_pt[0], foot_pt[1], poly) or _point_in_polygon(center_pt[0], center_pt[1], poly):
                    return center_pt
        return None

    def _probe_person_in_zone(self) -> Tuple[bool, Optional[Tuple[float, float]], bool]:
        now_ts = time.time()
        if not self._ensure_probe_cap():
            self._probe_fail_count += 1
            if self._probe_fail_count % 5 == 0:
                self.logger.warning(f"Probe stream unavailable for {self.camera_name}")
            self._stable_hits = 0
            return False, None, False

        ret, frame = self.probe_cap.read()
        if not ret or frame is None:
            self._probe_fail_count += 1
            if self._probe_fail_count % 5 == 0:
                self.logger.warning(f"Probe read failed for {self.camera_name}")
            self._stable_hits = 0
            self._release_probe_cap()
            return False, None, False

        self._probe_fail_count = 0
        h, w = frame.shape[:2]
        boxes = self.zone_gate.detect_people(frame) if self.zone_gate is not None else []
        identity = self._identity_in_business_zone(boxes, w, h)
        prev_identity = self._last_identity
        prev_identity_ts = self._last_identity_ts
        same_prev = (
            self._same_identity(identity, prev_identity)
            and ((now_ts - prev_identity_ts) <= self.same_person_memory_sec)
        )
        if identity is None:
            self._stable_hits = 0
            return False, None, False

        if same_prev:
            self._stable_hits = min(100, self._stable_hits + 1)
        else:
            self._stable_hits = 1
        self._last_identity = identity
        self._last_identity_ts = now_ts
        return True, identity, same_prev

    def _monitor_loop(self):
        while self.running:
            if not self._is_within_business_hours():
                if self.process is not None:
                    self._stop_ffmpeg()
                self._stable_hits = 0
                time.sleep(max(1.0, self.probe_interval_sec))
                continue

            if not self.enable_person_zone_gate:
                if self.process is None or self.process.poll() is not None:
                    self._start_continuous_ffmpeg_segment()
                time.sleep(5.0)
                continue

            occupied, identity, same_prev = self._probe_person_in_zone()
            now_ts = time.time()
            if occupied and identity is not None:
                if (now_ts - self._last_record_ts) >= self.trigger_cooldown_sec:
                    should_record = same_prev or (self._stable_hits >= self.required_stable_hits)
                    if should_record:
                        self._record_short_clip_once()
                        self._last_record_ts = time.time()
            else:
                self._stable_hits = 0

            time.sleep(self.probe_interval_sec)


class ServiceRecorder:
    def __init__(self):
        self.config = Config(CONFIG_PATH)
        self.logger = setup_logger("recorder", "logs")
        self.recorders: Dict[str, RTSPRecorder] = {}
        runtime_cfg = self.config.get("runtime", {}) or {}
        paths_cfg = self.config.get("paths", {}) or {}
        self.runtime_cfg = runtime_cfg
        self.ffmpeg_path = runtime_cfg.get("ffmpeg_path") or paths_cfg.get("ffmpeg")
        self.zone_gate = PersonZoneGate(self.config, self.logger)

    def start(self):
        self.logger.info("Starting ServiceRecorder...")
        ffmpeg_path, ffmpeg_state = ensure_ffmpeg_available(
            config_obj=self.config,
            project_root=PROJECT_ROOT,
            logger=self.logger,
            auto_download=True,
        )
        self.ffmpeg_path = ffmpeg_path
        if ffmpeg_path:
            self.logger.info(f"FFmpeg ready ({ffmpeg_state}): {ffmpeg_path}")
        else:
            self.logger.warning(f"FFmpeg unavailable ({ffmpeg_state}); recorder workers may retry until available")

        cameras = self.config.get("cameras", {}) or {}
        for name, cfg in cameras.items():
            if not cfg.get("enabled", True):
                continue
            url = cfg.get("rtsp_url")
            if not url:
                continue
            rec = RTSPRecorder(
                camera_name=name,
                rtsp_url=url,
                output_dir="data/recordings",
                segment_time=int(self.runtime_cfg.get("recorder_segment_time_sec", 60)),
                ffmpeg_path=ffmpeg_path,
                zone_gate=self.zone_gate,
                runtime_cfg=self.runtime_cfg,
            )
            rec.start()
            self.recorders[name] = rec

    def stop(self):
        for rec in self.recorders.values():
            rec.stop()


if __name__ == "__main__":
    svc = ServiceRecorder()

    def handle_signal(signum, frame):
        print(f"Received signal {signum}, stopping...")
        svc.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    try:
        svc.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        svc.stop()
    except Exception as e:
        print(f"Recorder error: {e}")
        svc.stop()
