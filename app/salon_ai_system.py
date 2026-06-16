"""High-level builder for the salon custom YOLO runtime."""

from __future__ import annotations

import logging
from pathlib import Path

from app.salon_ai_config import SalonAIConfig
from detectors.yolo_salon_detector import YOLOSalonDetector
from services.camera_runtime import MultiCameraRuntime
from services.counter_service import CountingService
from services.hard_negative_miner import HardNegativeConfig, HardNegativeMiner
from services.zone_manager import ZoneManager
from tracking.reid_memory import ReIDMemory
from tracking.temporal_smoother import TemporalSmoother


logger = logging.getLogger(__name__)


class SalonAISystem:
    def __init__(self, config_path: str = "data/config/salon_ai.runtime.yaml") -> None:
        self.config = SalonAIConfig.from_yaml(config_path)
        self.runtime = self._build_runtime()

    def _resolve(self, value: str) -> str:
        p = Path(value).expanduser()
        if p.is_absolute():
            return str(p)
        return str((self.config.project_root / p).resolve())

    def _build_runtime(self) -> MultiCameraRuntime:
        model_cfg = self.config.model
        runtime_cfg = self.config.runtime

        detector = YOLOSalonDetector(
            model_path=self._resolve(model_cfg.model_path),
            class_names=model_cfg.class_names,
            class_thresholds=model_cfg.class_thresholds.as_dict(),
            tracker_mode=model_cfg.tracker_mode,
            tracker_cfg=self._resolve(model_cfg.tracker_cfg),
            deepsort_max_age=model_cfg.deepsort_max_age,
            deepsort_n_init=model_cfg.deepsort_n_init,
            deepsort_max_iou_distance=model_cfg.deepsort_max_iou_distance,
            deepsort_nn_budget=model_cfg.deepsort_nn_budget,
            device=model_cfg.device,
            imgsz=model_cfg.imgsz,
            base_conf=model_cfg.base_conf,
            iou=model_cfg.iou,
        )

        zone_manager = ZoneManager()
        for cam in self.config.cameras:
            zone_manager.set_camera_zones(cam.camera_id, cam.zones)

        counter = CountingService(
            zone_memory_sec=runtime_cfg.zone_memory_sec,
            wash_return_window_sec=runtime_cfg.wash_return_window_sec,
            wash_recount_cooldown_sec=runtime_cfg.wash_recount_cooldown_sec,
            haircut_recount_cooldown_sec=runtime_cfg.haircut_recount_cooldown_sec,
            staff_lock_sec=runtime_cfg.staff_lock_sec,
            track_ttl_sec=runtime_cfg.track_ttl_sec,
        )
        reid = ReIDMemory(
            similarity_threshold=runtime_cfg.reid_similarity_threshold,
            ttl_sec=runtime_cfg.reid_ttl_sec,
        )
        smoother = TemporalSmoother(
            window_size=runtime_cfg.smoothing_window,
            min_votes=runtime_cfg.smoothing_min_votes,
        )

        miner = None
        if runtime_cfg.hard_negative_enabled:
            miner = HardNegativeMiner(
                HardNegativeConfig(
                    output_dir=self._resolve(runtime_cfg.hard_negative_dir),
                )
            )

        runtime = MultiCameraRuntime(
            detector=detector,
            zone_manager=zone_manager,
            counter_service=counter,
            reid_memory=reid,
            smoother=smoother,
            hard_negative_miner=miner,
            target_fps=runtime_cfg.target_fps,
            reconnect_sec=runtime_cfg.reconnect_sec,
        )

        for cam in self.config.cameras:
            if not cam.rtsp_url:
                logger.warning("Skipped camera '%s' because rtsp_url is empty", cam.camera_id)
                continue
            runtime.add_camera(cam.camera_id, cam.rtsp_url, cam.enabled)
        return runtime

    def start(self) -> None:
        self.runtime.start()

    def stop(self) -> None:
        self.runtime.stop()

    def status(self):
        return self.runtime.latest_packets()
