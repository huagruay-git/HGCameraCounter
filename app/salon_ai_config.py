"""Configuration models for the salon multi-camera AI pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


Point = Tuple[int, int]
Polygon = List[Point]


def _as_polygon(value: Any) -> Polygon:
    if not isinstance(value, list):
        return []
    polygon: Polygon = []
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                polygon.append((int(item[0]), int(item[1])))
            except Exception:
                continue
    return polygon


@dataclass
class ClassThresholds:
    customer_haircut: float = 0.45
    customer_wash: float = 0.40
    staff_barber: float = 0.50

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClassThresholds":
        src = data or {}
        return cls(
            customer_haircut=float(src.get("customer_haircut", 0.45)),
            customer_wash=float(src.get("customer_wash", 0.40)),
            staff_barber=float(src.get("staff_barber", 0.50)),
        )

    def as_dict(self) -> Dict[str, float]:
        return {
            "customer_haircut": self.customer_haircut,
            "customer_wash": self.customer_wash,
            "staff_barber": self.staff_barber,
        }


@dataclass
class ModelConfig:
    model_path: str = "models/best.pt"
    tracker_mode: str = "bytetrack"
    tracker_cfg: str = "bytetrack.yaml"
    deepsort_max_age: int = 30
    deepsort_n_init: int = 3
    deepsort_max_iou_distance: float = 0.7
    deepsort_nn_budget: int = 100
    imgsz: int = 640
    iou: float = 0.50
    base_conf: float = 0.25
    device: str = "auto"
    class_names: List[str] = field(
        default_factory=lambda: [
            "customer_haircut",
            "staff_barber",
            "customer_wash",
        ]
    )
    class_thresholds: ClassThresholds = field(default_factory=ClassThresholds)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        src = data or {}
        names = src.get("class_names", [])
        if not isinstance(names, list) or len(names) < 3:
            names = ["customer_haircut", "staff_barber", "customer_wash"]
        tracker_mode = str(src.get("tracker_mode", "bytetrack")).strip().lower()
        if tracker_mode not in {"bytetrack", "deepsort"}:
            tracker_mode = "bytetrack"
        return cls(
            model_path=str(src.get("model_path", "models/best.pt")),
            tracker_mode=tracker_mode,
            tracker_cfg=str(src.get("tracker_cfg", "bytetrack.yaml")),
            deepsort_max_age=int(src.get("deepsort_max_age", 30)),
            deepsort_n_init=int(src.get("deepsort_n_init", 3)),
            deepsort_max_iou_distance=float(src.get("deepsort_max_iou_distance", 0.7)),
            deepsort_nn_budget=int(src.get("deepsort_nn_budget", 100)),
            imgsz=int(src.get("imgsz", 640)),
            iou=float(src.get("iou", 0.50)),
            base_conf=float(src.get("base_conf", 0.25)),
            device=str(src.get("device", "auto")),
            class_names=list(names),
            class_thresholds=ClassThresholds.from_dict(src.get("class_thresholds", {})),
        )


@dataclass
class RuntimeConfig:
    target_fps: float = 8.0
    reconnect_sec: float = 2.0
    track_ttl_sec: float = 25.0
    zone_memory_sec: float = 8.0
    wash_return_window_sec: float = 3600.0
    wash_recount_cooldown_sec: float = 600.0
    haircut_recount_cooldown_sec: float = 7200.0
    staff_lock_sec: float = 1200.0
    reid_similarity_threshold: float = 0.86
    reid_ttl_sec: float = 3600.0
    smoothing_window: int = 12
    smoothing_min_votes: int = 4
    hard_negative_enabled: bool = True
    hard_negative_dir: str = "data/performance_feedback/hard_negative"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuntimeConfig":
        src = data or {}
        return cls(
            target_fps=float(src.get("target_fps", 8.0)),
            reconnect_sec=float(src.get("reconnect_sec", 2.0)),
            track_ttl_sec=float(src.get("track_ttl_sec", 25.0)),
            zone_memory_sec=float(src.get("zone_memory_sec", 8.0)),
            wash_return_window_sec=float(src.get("wash_return_window_sec", 3600.0)),
            wash_recount_cooldown_sec=float(src.get("wash_recount_cooldown_sec", 600.0)),
            haircut_recount_cooldown_sec=float(src.get("haircut_recount_cooldown_sec", 7200.0)),
            staff_lock_sec=float(src.get("staff_lock_sec", 1200.0)),
            reid_similarity_threshold=float(src.get("reid_similarity_threshold", 0.86)),
            reid_ttl_sec=float(src.get("reid_ttl_sec", 3600.0)),
            smoothing_window=int(src.get("smoothing_window", 12)),
            smoothing_min_votes=int(src.get("smoothing_min_votes", 4)),
            hard_negative_enabled=bool(src.get("hard_negative_enabled", True)),
            hard_negative_dir=str(src.get("hard_negative_dir", "data/performance_feedback/hard_negative")),
        )


@dataclass
class CameraConfig:
    camera_id: str
    rtsp_url: str
    enabled: bool = True
    zones: Dict[str, Polygon] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, camera_id: str, data: Dict[str, Any]) -> "CameraConfig":
        src = data or {}
        zone_src = src.get("zones", {})
        zone_map: Dict[str, Polygon] = {}
        if isinstance(zone_src, dict):
            for zone_name, poly in zone_src.items():
                zone_map[str(zone_name)] = _as_polygon(poly)
        return cls(
            camera_id=camera_id,
            rtsp_url=str(src.get("rtsp_url", "")),
            enabled=bool(src.get("enabled", True)),
            zones=zone_map,
        )


@dataclass
class SalonAIConfig:
    project_root: Path
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    cameras: List[CameraConfig] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SalonAIConfig":
        cfg_path = Path(path).expanduser()
        if not cfg_path.is_absolute():
            cfg_path = (Path(__file__).resolve().parent.parent / cfg_path).resolve()
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config root must be a dict, got {type(data).__name__}")

        cams: List[CameraConfig] = []
        for cam_id, cam_cfg in (data.get("cameras", {}) or {}).items():
            cams.append(CameraConfig.from_dict(str(cam_id), cam_cfg))

        project_root = cfg_path.parent
        # Typical location is <repo>/data/config/*.yaml -> repo root is parents[2].
        if len(cfg_path.parents) >= 3:
            project_root = cfg_path.parents[2]

        return cls(
            project_root=project_root.resolve(),
            model=ModelConfig.from_dict(data.get("model", {})),
            runtime=RuntimeConfig.from_dict(data.get("runtime", {})),
            cameras=cams,
        )
