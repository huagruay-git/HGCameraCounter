"""Business counting logic with anti-double-count and staff exclusion."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict, Dict, List, Optional

from detectors.yolo_salon_detector import Detection


@dataclass
class TrackBusinessState:
    global_id: int
    first_seen_ts: float
    last_seen_ts: float
    last_camera: str
    last_zone: Optional[str] = None
    last_zone_ts: float = 0.0
    chair_zone: Optional[str] = None
    haircut_counted: bool = False
    wash_counted: bool = False
    last_haircut_ts: float = 0.0
    last_wash_ts: float = 0.0
    wash_trip_started_ts: float = 0.0
    pending_wash_return: bool = False
    staff_lock_until_ts: float = 0.0


@dataclass
class CounterSnapshot:
    timestamp: float
    haircuts_total: int
    washes_total: int
    staff_seen_total: int
    haircuts_by_zone: Dict[str, int] = field(default_factory=dict)
    washes_by_zone: Dict[str, int] = field(default_factory=dict)
    active_customers: int = 0
    active_staff: int = 0


class CountingService:
    def __init__(
        self,
        zone_memory_sec: float = 8.0,
        wash_return_window_sec: float = 3600.0,
        wash_recount_cooldown_sec: float = 600.0,
        haircut_recount_cooldown_sec: float = 7200.0,
        staff_lock_sec: float = 1200.0,
        track_ttl_sec: float = 25.0,
    ) -> None:
        self.zone_memory_sec = float(zone_memory_sec)
        self.wash_return_window_sec = float(wash_return_window_sec)
        self.wash_recount_cooldown_sec = float(wash_recount_cooldown_sec)
        self.haircut_recount_cooldown_sec = float(haircut_recount_cooldown_sec)
        self.staff_lock_sec = float(staff_lock_sec)
        self.track_ttl_sec = float(track_ttl_sec)

        self.haircuts_total = 0
        self.washes_total = 0
        self.staff_seen_total = 0
        self.haircuts_by_zone: DefaultDict[str, int] = defaultdict(int)
        self.washes_by_zone: DefaultDict[str, int] = defaultdict(int)
        self._states: Dict[int, TrackBusinessState] = {}

    def _state_for(self, gid: int, camera_id: str, timestamp: float) -> TrackBusinessState:
        st = self._states.get(gid)
        if st is None:
            st = TrackBusinessState(
                global_id=gid,
                first_seen_ts=timestamp,
                last_seen_ts=timestamp,
                last_camera=camera_id,
            )
            self._states[gid] = st
        st.last_seen_ts = timestamp
        st.last_camera = camera_id
        return st

    @staticmethod
    def _is_zone(zone_name: Optional[str], prefix: str) -> bool:
        if not zone_name:
            return False
        return zone_name.lower().startswith(prefix.lower())

    def _cleanup(self, now_ts: float) -> None:
        drop = [gid for gid, st in self._states.items() if (now_ts - st.last_seen_ts) > self.track_ttl_sec]
        for gid in drop:
            self._states.pop(gid, None)

    def process(self, camera_id: str, detections: List[Detection], timestamp: float) -> CounterSnapshot:
        active_customers = 0
        active_staff = 0

        for det in detections:
            if det.global_id is None:
                continue
            state = self._state_for(det.global_id, camera_id, timestamp)
            zone = det.zone

            if zone:
                zone_changed = state.last_zone != zone
                enough_time = (timestamp - state.last_zone_ts) >= self.zone_memory_sec
                if zone_changed or enough_time:
                    state.last_zone = zone
                    state.last_zone_ts = timestamp

            is_staff_label = det.class_name == "staff_barber"
            if is_staff_label:
                state.staff_lock_until_ts = max(state.staff_lock_until_ts, timestamp + self.staff_lock_sec)
                self.staff_seen_total += 1
                active_staff += 1
                continue

            if timestamp <= state.staff_lock_until_ts:
                active_staff += 1
                continue

            active_customers += 1

            is_haircut = det.class_name == "customer_haircut"
            is_wash = det.class_name == "customer_wash"
            in_chair = self._is_zone(zone, "chair")
            in_wash = self._is_zone(zone, "wash")

            if is_haircut and in_chair:
                can_count_haircut = (not state.haircut_counted) or (
                    (timestamp - state.last_haircut_ts) >= self.haircut_recount_cooldown_sec
                )
                if can_count_haircut:
                    self.haircuts_total += 1
                    state.haircut_counted = True
                    state.last_haircut_ts = timestamp
                    state.chair_zone = zone
                    self.haircuts_by_zone[zone or "chair_unknown"] += 1

            if is_wash and state.haircut_counted:
                if in_wash:
                    state.wash_trip_started_ts = timestamp
                    state.pending_wash_return = True

                return_same_chair = in_chair and (state.chair_zone is None or state.chair_zone == zone)
                no_wash_zone_config = (not in_wash) and return_same_chair
                wash_window_open = (timestamp - state.last_haircut_ts) <= self.wash_return_window_sec
                cooldown_done = (timestamp - state.last_wash_ts) >= self.wash_recount_cooldown_sec
                pending_or_direct = state.pending_wash_return or no_wash_zone_config

                if pending_or_direct and return_same_chair and wash_window_open and cooldown_done:
                    self.washes_total += 1
                    state.wash_counted = True
                    state.last_wash_ts = timestamp
                    state.pending_wash_return = False
                    self.washes_by_zone[zone or "wash_unknown"] += 1

        self._cleanup(timestamp)
        return CounterSnapshot(
            timestamp=timestamp,
            haircuts_total=self.haircuts_total,
            washes_total=self.washes_total,
            staff_seen_total=self.staff_seen_total,
            haircuts_by_zone=dict(self.haircuts_by_zone),
            washes_by_zone=dict(self.washes_by_zone),
            active_customers=active_customers,
            active_staff=active_staff,
        )
