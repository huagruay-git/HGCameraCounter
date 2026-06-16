"""Collect uncertain detections for hard-negative retraining."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from detectors.yolo_salon_detector import Detection


@dataclass
class HardNegativeConfig:
    output_dir: str = "data/performance_feedback/hard_negative"
    min_conf: float = 0.20
    max_conf: float = 0.55
    top2_margin: float = 0.10
    per_camera_cooldown_sec: float = 4.0


class HardNegativeMiner:
    def __init__(self, cfg: HardNegativeConfig) -> None:
        self.cfg = cfg
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._last_saved_ts: Dict[str, float] = {}

    def maybe_save(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        class_probabilities: Dict[int, Dict[str, float]],
        timestamp: float,
    ) -> int:
        saved = 0
        for idx, det in enumerate(detections):
            probs = class_probabilities.get(idx, {})
            if not probs:
                continue
            ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
            if len(ranked) < 2:
                continue
            top1_name, top1_prob = ranked[0]
            _, top2_prob = ranked[1]
            ambiguous = (top1_prob <= self.cfg.max_conf) or ((top1_prob - top2_prob) <= self.cfg.top2_margin)
            if (not ambiguous) or (top1_prob < self.cfg.min_conf):
                continue

            last_ts = self._last_saved_ts.get(det.camera_id, 0.0)
            if (timestamp - last_ts) < self.cfg.per_camera_cooldown_sec:
                continue

            self._last_saved_ts[det.camera_id] = timestamp
            self._save_crop(frame, det, ranked, timestamp)
            saved += 1
        return saved

    def _save_crop(
        self,
        frame: np.ndarray,
        det: Detection,
        ranked_probs: List[tuple[str, float]],
        timestamp: float,
    ) -> None:
        dt = datetime.fromtimestamp(timestamp)
        day_dir = self.output_dir / dt.strftime("%Y%m%d")
        day_dir.mkdir(parents=True, exist_ok=True)

        x1, y1, x2, y2 = det.xyxy
        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(1, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(1, min(h, y2))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return

        stem = f"{dt.strftime('%H%M%S_%f')}_{det.camera_id}_gid{det.global_id or 0}_tid{det.track_id or 0}"
        img_path = day_dir / f"{stem}.jpg"
        meta_path = day_dir / f"{stem}.json"
        cv2.imwrite(str(img_path), crop)
        meta_path.write_text(
            json.dumps(
                {
                    "camera_id": det.camera_id,
                    "global_id": det.global_id,
                    "track_id": det.track_id,
                    "xyxy": list(det.xyxy),
                    "zone": det.zone,
                    "ranked_probabilities": ranked_probs,
                    "timestamp": timestamp,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
