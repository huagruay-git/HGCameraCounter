"""Simple appearance memory for cross-frame/cross-camera identity persistence."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-8 or nb <= 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def extract_embedding(frame: np.ndarray, xyxy: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(1, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(1, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((64,), dtype=np.float32)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((64,), dtype=np.float32)

    crop = cv2.resize(crop, (80, 120), interpolation=cv2.INTER_LINEAR)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256]).flatten()
    hist = hist.astype(np.float32)
    if float(hist.sum()) > 0:
        hist /= float(hist.sum())

    rgb_mean = crop.mean(axis=(0, 1)).astype(np.float32) / 255.0
    emb = np.concatenate([hist, rgb_mean], axis=0).astype(np.float32)
    norm = float(np.linalg.norm(emb))
    if norm > 1e-8:
        emb /= norm
    return emb


@dataclass
class ReIDRecord:
    global_id: int
    embedding: np.ndarray
    last_seen_ts: float
    last_camera: str
    last_track_id: Optional[int]


class ReIDMemory:
    def __init__(self, similarity_threshold: float = 0.86, ttl_sec: float = 3600.0) -> None:
        self.similarity_threshold = float(similarity_threshold)
        self.ttl_sec = float(ttl_sec)
        self._next_gid = 1
        self._records: Dict[int, ReIDRecord] = {}
        self._local_track_map: Dict[Tuple[str, int], int] = {}

    def assign_global_id(
        self,
        camera_id: str,
        track_id: Optional[int],
        embedding: np.ndarray,
        timestamp: float,
    ) -> int:
        self._cleanup(float(timestamp))

        if track_id is not None:
            key = (camera_id, int(track_id))
            gid = self._local_track_map.get(key)
            if gid is not None and gid in self._records:
                self._update_record(gid, camera_id, track_id, embedding, timestamp)
                return gid

        best_gid: Optional[int] = None
        best_sim = -math.inf
        for gid, rec in self._records.items():
            sim = _cosine_similarity(embedding, rec.embedding)
            if sim > best_sim:
                best_sim = sim
                best_gid = gid

        if best_gid is not None and best_sim >= self.similarity_threshold:
            self._update_record(best_gid, camera_id, track_id, embedding, timestamp)
            if track_id is not None:
                self._local_track_map[(camera_id, int(track_id))] = best_gid
            return best_gid

        gid = self._next_gid
        self._next_gid += 1
        self._records[gid] = ReIDRecord(
            global_id=gid,
            embedding=embedding,
            last_seen_ts=float(timestamp),
            last_camera=camera_id,
            last_track_id=int(track_id) if track_id is not None else None,
        )
        if track_id is not None:
            self._local_track_map[(camera_id, int(track_id))] = gid
        return gid

    def _update_record(
        self,
        gid: int,
        camera_id: str,
        track_id: Optional[int],
        embedding: np.ndarray,
        timestamp: float,
    ) -> None:
        rec = self._records[gid]
        rec.embedding = 0.7 * rec.embedding + 0.3 * embedding
        norm = float(np.linalg.norm(rec.embedding))
        if norm > 1e-8:
            rec.embedding = rec.embedding / norm
        rec.last_seen_ts = float(timestamp)
        rec.last_camera = camera_id
        rec.last_track_id = int(track_id) if track_id is not None else None

    def _cleanup(self, now_ts: float) -> None:
        drop = [gid for gid, rec in self._records.items() if (now_ts - rec.last_seen_ts) > self.ttl_sec]
        if not drop:
            return
        for gid in drop:
            self._records.pop(gid, None)
        keep_map: Dict[Tuple[str, int], int] = {}
        for key, gid in self._local_track_map.items():
            if gid in self._records:
                keep_map[key] = gid
        self._local_track_map = keep_map
