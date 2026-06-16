"""Zone utilities for point-in-polygon checks."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

from detectors.yolo_salon_detector import Detection


Polygon = List[Tuple[int, int]]


class ZoneManager:
    def __init__(self, zones_by_camera: Dict[str, Dict[str, Polygon]] | None = None) -> None:
        self.zones_by_camera: Dict[str, Dict[str, Polygon]] = zones_by_camera or {}

    def set_camera_zones(self, camera_id: str, zones: Dict[str, Polygon]) -> None:
        self.zones_by_camera[str(camera_id)] = zones or {}

    @staticmethod
    def _contains(point: Tuple[int, int], polygon: Polygon) -> bool:
        if not polygon or len(polygon) < 3:
            return False
        arr = np.array(polygon, dtype=np.int32)
        return cv2.pointPolygonTest(arr, point, False) >= 0

    def hit_zones(self, camera_id: str, point: Tuple[int, int]) -> List[str]:
        zones = self.zones_by_camera.get(camera_id, {})
        hits: List[str] = []
        for zone_name, polygon in zones.items():
            if self._contains(point, polygon):
                hits.append(zone_name)
        return hits

    def hit_zones_for_detection(self, camera_id: str, det: Detection, anchor: str = "foot") -> List[str]:
        if anchor == "center":
            point = det.center_point
        else:
            point = det.foot_point
        return self.hit_zones(camera_id, point)
