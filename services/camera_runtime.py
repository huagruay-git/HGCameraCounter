"""Realtime multi-camera runtime pipeline."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from detectors.yolo_salon_detector import Detection, YOLOSalonDetector
from services.counter_service import CounterSnapshot, CountingService
from services.hard_negative_miner import HardNegativeMiner
from services.zone_manager import ZoneManager
from tracking.reid_memory import ReIDMemory, extract_embedding
from tracking.temporal_smoother import TemporalSmoother


logger = logging.getLogger(__name__)


@dataclass
class FrameAnalyticsPacket:
    camera_id: str
    timestamp: float
    detections: List[Detection]
    counters: CounterSnapshot

    def to_dict(self) -> Dict:
        return {
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
            "detections": [
                {
                    "class_name": d.class_name,
                    "confidence": d.confidence,
                    "track_id": d.track_id,
                    "global_id": d.global_id,
                    "xyxy": list(d.xyxy),
                    "zone": d.zone,
                    "zones": list(d.zones),
                }
                for d in self.detections
            ],
            "counters": asdict(self.counters),
        }


class MultiCameraRuntime:
    def __init__(
        self,
        detector: YOLOSalonDetector,
        zone_manager: ZoneManager,
        counter_service: CountingService,
        reid_memory: ReIDMemory,
        smoother: TemporalSmoother,
        hard_negative_miner: Optional[HardNegativeMiner] = None,
        target_fps: float = 8.0,
        reconnect_sec: float = 2.0,
    ) -> None:
        self.detector = detector
        self.zone_manager = zone_manager
        self.counter_service = counter_service
        self.reid_memory = reid_memory
        self.smoother = smoother
        self.hard_negative_miner = hard_negative_miner
        self.target_fps = max(0.1, float(target_fps))
        self.reconnect_sec = max(0.5, float(reconnect_sec))

        self._camera_urls: Dict[str, str] = {}
        self._camera_enabled: Dict[str, bool] = {}
        self._threads: List[threading.Thread] = []
        self._callbacks: List[Callable[[FrameAnalyticsPacket, np.ndarray], None]] = []
        self._stop = threading.Event()
        self._latest_by_camera: Dict[str, FrameAnalyticsPacket] = {}
        self._lock = threading.Lock()

    def add_camera(self, camera_id: str, rtsp_url: str, enabled: bool = True) -> None:
        self._camera_urls[str(camera_id)] = str(rtsp_url)
        self._camera_enabled[str(camera_id)] = bool(enabled)

    def subscribe(self, callback: Callable[[FrameAnalyticsPacket, np.ndarray], None]) -> None:
        self._callbacks.append(callback)

    def start(self) -> None:
        self._stop.clear()
        self._threads.clear()
        for camera_id, url in self._camera_urls.items():
            if not self._camera_enabled.get(camera_id, True):
                continue
            th = threading.Thread(target=self._camera_loop, args=(camera_id, url), daemon=True)
            th.start()
            self._threads.append(th)
            logger.info("Started camera runtime thread: %s", camera_id)

    def stop(self) -> None:
        self._stop.set()
        for th in self._threads:
            th.join(timeout=2.0)
        self._threads.clear()

    def latest_packets(self) -> Dict[str, Dict]:
        with self._lock:
            return {cid: pkt.to_dict() for cid, pkt in self._latest_by_camera.items()}

    def _camera_loop(self, camera_id: str, rtsp_url: str) -> None:
        cap: Optional[cv2.VideoCapture] = None
        frame_period = 1.0 / self.target_fps
        last_frame_ts = 0.0

        while not self._stop.is_set():
            try:
                if cap is None or (not cap.isOpened()):
                    cap = cv2.VideoCapture(rtsp_url)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if not cap.isOpened():
                        logger.warning("RTSP open failed for %s; retrying...", camera_id)
                        time.sleep(self.reconnect_sec)
                        continue

                ok, frame = cap.read()
                if not ok or frame is None:
                    logger.warning("Frame read failed for %s; reconnecting...", camera_id)
                    cap.release()
                    cap = None
                    time.sleep(self.reconnect_sec)
                    continue

                now_ts = time.time()
                if (now_ts - last_frame_ts) < frame_period:
                    time.sleep(0.001)
                    continue
                last_frame_ts = now_ts

                packet = self._process_frame(camera_id, frame, now_ts)
                with self._lock:
                    self._latest_by_camera[camera_id] = packet

                for cb in self._callbacks:
                    try:
                        cb(packet, frame)
                    except Exception as callback_err:
                        logger.exception("Runtime callback error: %s", callback_err)
            except Exception as loop_err:
                logger.exception("Camera loop error (%s): %s", camera_id, loop_err)
                time.sleep(self.reconnect_sec)

        if cap is not None:
            cap.release()

    def _process_frame(self, camera_id: str, frame: np.ndarray, timestamp: float) -> FrameAnalyticsPacket:
        detections = self.detector.infer(frame, camera_id, timestamp)
        class_probs: Dict[int, Dict[str, float]] = {}

        for idx, det in enumerate(detections):
            det.zones = self.zone_manager.hit_zones_for_detection(camera_id, det, anchor="foot")
            det.zone = det.zones[0] if det.zones else None

            emb = extract_embedding(frame, det.xyxy)
            det.embedding = emb
            det.global_id = self.reid_memory.assign_global_id(
                camera_id=camera_id,
                track_id=det.track_id,
                embedding=emb,
                timestamp=timestamp,
            )

            stable_label, smoothed_prob = self.smoother.update(
                track_key=(camera_id, int(det.global_id)),
                class_name=det.class_name,
                confidence=det.confidence,
                timestamp=timestamp,
            )
            det.smoothed_class_name = stable_label
            det.smoothed_confidence = float(smoothed_prob.get(stable_label, det.confidence))
            det.class_name = stable_label
            det.confidence = max(float(det.confidence), float(det.smoothed_confidence))
            class_probs[idx] = smoothed_prob

        counters = self.counter_service.process(camera_id, detections, timestamp)

        if self.hard_negative_miner is not None:
            self.hard_negative_miner.maybe_save(frame, detections, class_probs, timestamp)

        return FrameAnalyticsPacket(
            camera_id=camera_id,
            timestamp=timestamp,
            detections=detections,
            counters=counters,
        )
