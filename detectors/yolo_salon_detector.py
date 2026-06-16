"""YOLO detector wrapper for salon multi-class detection."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class Detection:
    camera_id: str
    class_id: int
    class_name: str
    confidence: float
    xyxy: Tuple[int, int, int, int]
    timestamp: float
    track_id: Optional[int] = None
    global_id: Optional[int] = None
    zone: Optional[str] = None
    zones: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    smoothed_class_name: Optional[str] = None
    smoothed_confidence: Optional[float] = None

    @property
    def foot_point(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) // 2, y2)

    @property
    def center_point(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) // 2, (y1 + y2) // 2)


class YOLOSalonDetector:
    def __init__(
        self,
        model_path: str,
        class_names: List[str],
        class_thresholds: Dict[str, float],
        tracker_mode: str = "bytetrack",
        tracker_cfg: str = "bytetrack.yaml",
        deepsort_max_age: int = 30,
        deepsort_n_init: int = 3,
        deepsort_max_iou_distance: float = 0.7,
        deepsort_nn_budget: int = 100,
        device: str = "auto",
        imgsz: int = 640,
        base_conf: float = 0.25,
        iou: float = 0.50,
    ) -> None:
        from ultralytics import YOLO

        self.model_path = str(model_path)
        self.class_names = list(class_names)
        self.class_thresholds = dict(class_thresholds)
        mode = str(tracker_mode or "bytetrack").strip().lower()
        self.tracker_mode = mode if mode in {"bytetrack", "deepsort"} else "bytetrack"
        self.tracker_cfg = str(tracker_cfg)
        self.deepsort_max_age = int(deepsort_max_age)
        self.deepsort_n_init = int(deepsort_n_init)
        self.deepsort_max_iou_distance = float(deepsort_max_iou_distance)
        self.deepsort_nn_budget = int(deepsort_nn_budget)
        self.device = self._select_device(device)
        self.imgsz = int(imgsz)
        self.base_conf = float(base_conf)
        self.iou = float(iou)
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"YOLO model not found: {self.model_path}")
        self.model = YOLO(self.model_path)
        self._deepsort = None
        if self.tracker_mode == "deepsort":
            self._init_deepsort()

    @staticmethod
    def _select_device(requested: str) -> str:
        req = str(requested or "auto").strip().lower()
        if req != "auto":
            return req
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def infer(self, frame: np.ndarray, camera_id: str, timestamp: float) -> List[Detection]:
        if frame is None:
            return []
        if self.tracker_mode == "deepsort":
            return self._infer_deepsort(frame, camera_id, timestamp)
        return self._infer_bytetrack(frame, camera_id, timestamp)

    def _infer_bytetrack(self, frame: np.ndarray, camera_id: str, timestamp: float) -> List[Detection]:
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_cfg,
            conf=self.base_conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        return self._parse_ultralytics_results(results, camera_id, timestamp)

    def _infer_deepsort(self, frame: np.ndarray, camera_id: str, timestamp: float) -> List[Detection]:
        if self._deepsort is None:
            return self._infer_bytetrack(frame, camera_id, timestamp)
        raw = self._predict_raw(frame)
        if not raw:
            return []

        ds_inputs: List[Tuple[List[float], float, str]] = []
        for det in raw:
            x1, y1, x2, y2 = det["xyxy"]
            ds_inputs.append(([float(x1), float(y1), float(max(1, x2 - x1)), float(max(1, y2 - y1))], float(det["confidence"]), str(det["class_name"])))

        tracks = self._deepsort.update_tracks(ds_inputs, frame=frame)
        detections: List[Detection] = []
        for trk in tracks:
            if hasattr(trk, "is_confirmed") and not trk.is_confirmed():
                continue
            if not hasattr(trk, "to_ltrb"):
                continue
            ltrb = trk.to_ltrb()
            if ltrb is None or len(ltrb) < 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in ltrb[:4]]
            if x2 <= x1 or y2 <= y1:
                continue

            best = self._best_match_raw((x1, y1, x2, y2), raw)
            if best is None:
                continue
            class_name = str(best["class_name"])
            conf = float(best["confidence"])
            threshold = float(self.class_thresholds.get(class_name, self.base_conf))
            if conf < threshold:
                continue
            track_id = int(getattr(trk, "track_id", 0)) if getattr(trk, "track_id", None) is not None else None
            detections.append(
                Detection(
                    camera_id=camera_id,
                    class_id=int(best["class_id"]),
                    class_name=class_name,
                    confidence=conf,
                    xyxy=(x1, y1, x2, y2),
                    timestamp=timestamp,
                    track_id=track_id,
                )
            )
        return detections

    def _init_deepsort(self) -> None:
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
        except Exception as exc:
            logger.warning("DeepSORT not available (%s). Fallback to ByteTrack.", exc)
            self.tracker_mode = "bytetrack"
            self._deepsort = None
            return
        self._deepsort = DeepSort(
            max_age=self.deepsort_max_age,
            n_init=self.deepsort_n_init,
            max_iou_distance=self.deepsort_max_iou_distance,
            nn_budget=self.deepsort_nn_budget,
        )
        logger.info("Initialized DeepSORT tracker")

    def _predict_raw(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        results = self.model.predict(
            source=frame,
            conf=self.base_conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        parsed = self._parse_ultralytics_results(results, camera_id="_", timestamp=0.0)
        out: List[Dict[str, Any]] = []
        for det in parsed:
            out.append(
                {
                    "class_id": det.class_id,
                    "class_name": det.class_name,
                    "confidence": det.confidence,
                    "xyxy": det.xyxy,
                }
            )
        return out

    def _parse_ultralytics_results(self, results: Any, camera_id: str, timestamp: float) -> List[Detection]:
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.cls is None or len(boxes.cls) == 0:
            return []

        names = getattr(result, "names", None) or {i: n for i, n in enumerate(self.class_names)}
        cls_ids = boxes.cls.int().cpu().tolist()
        confs = boxes.conf.float().cpu().tolist() if boxes.conf is not None else [0.0] * len(cls_ids)
        xyxy_list = boxes.xyxy.int().cpu().tolist()
        tid_list = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(cls_ids)

        detections: List[Detection] = []
        for idx, cls_id in enumerate(cls_ids):
            class_name = str(names.get(int(cls_id), f"class_{cls_id}"))
            conf = float(confs[idx])
            threshold = float(self.class_thresholds.get(class_name, self.base_conf))
            if conf < threshold:
                continue
            x1, y1, x2, y2 = [int(v) for v in xyxy_list[idx]]
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                Detection(
                    camera_id=camera_id,
                    class_id=int(cls_id),
                    class_name=class_name,
                    confidence=conf,
                    xyxy=(x1, y1, x2, y2),
                    timestamp=timestamp,
                    track_id=int(tid_list[idx]) if tid_list[idx] is not None else None,
                )
            )
        return detections

    @staticmethod
    def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        bb = max(0, bx2 - bx1) * max(0, by2 - by1)
        denom = aa + bb - inter
        if denom <= 0:
            return 0.0
        return float(inter / denom)

    def _best_match_raw(self, bbox: Tuple[int, int, int, int], raw: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        best = None
        best_iou = 0.0
        for item in raw:
            iou = self._bbox_iou(bbox, tuple(item["xyxy"]))
            if iou > best_iou:
                best_iou = iou
                best = item
        return best if best_iou > 0.05 else None
