"""Temporal smoothing for unstable per-frame class predictions."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Tuple


@dataclass
class _Vote:
    timestamp: float
    class_name: str
    confidence: float


class TemporalSmoother:
    def __init__(self, window_size: int = 12, min_votes: int = 4, max_age_sec: float = 6.0) -> None:
        self.window_size = int(window_size)
        self.min_votes = int(min_votes)
        self.max_age_sec = float(max_age_sec)
        self._buffers: Dict[Tuple[str, int], Deque[_Vote]] = defaultdict(lambda: deque(maxlen=self.window_size))

    def update(
        self,
        track_key: Tuple[str, int],
        class_name: str,
        confidence: float,
        timestamp: float,
    ) -> Tuple[str, Dict[str, float]]:
        buf = self._buffers[track_key]
        buf.append(_Vote(timestamp=timestamp, class_name=class_name, confidence=float(confidence)))

        cutoff = float(timestamp) - self.max_age_sec
        while buf and buf[0].timestamp < cutoff:
            buf.popleft()

        score_by_class: Dict[str, float] = {}
        count_by_class: Dict[str, int] = {}
        for item in buf:
            score_by_class[item.class_name] = score_by_class.get(item.class_name, 0.0) + item.confidence
            count_by_class[item.class_name] = count_by_class.get(item.class_name, 0) + 1

        if not score_by_class:
            return class_name, {}

        stable = max(score_by_class.items(), key=lambda kv: kv[1])[0]
        vote_count = count_by_class.get(stable, 0)
        if vote_count < self.min_votes:
            stable = class_name

        total_score = sum(score_by_class.values()) or 1.0
        norm = {k: v / total_score for k, v in score_by_class.items()}
        return stable, norm
