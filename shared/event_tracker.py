"""
Event Tracking System

Tracks people movements through zones and generates events:
- entrance: person enters shop
- chair: person sits in haircut zone
- wash: person goes to wash zone
- wait: person waits
- exit: person leaves shop
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional
from enum import Enum
import json


class EventType(Enum):
    """Event types"""
    ENTRANCE = "entrance"
    CHAIR = "haircut"
    WASH = "wash"
    WAIT = "wait"
    EXIT = "exit"


@dataclass
class Event:
    """Single event record"""
    timestamp: datetime
    camera: str
    event_type: EventType
    person_id: int
    zone_name: str
    dwell_seconds: float = 0.0
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON/CSV"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "camera": self.camera,
            "event_type": self.event_type.value,
            "person_id": self.person_id,
            "zone_name": self.zone_name,
            "dwell_seconds": round(self.dwell_seconds, 2),
            "metadata": json.dumps(self.metadata),
        }


@dataclass
class PersonSession:
    """Track person's current session"""
    person_id: int
    camera: str
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    
    # Zone tracking
    current_zones: Dict[str, datetime] = field(default_factory=dict)  # zone -> entry_time
    zone_counted: Set[str] = field(default_factory=set)  # zones already counted for current occupancy
    visited_zones: Set[str] = field(default_factory=set)
    
    # Event counters
    haircut_count: int = 0  # CHAIR zones
    wash_count: int = 0     # WASH zones
    wait_count: int = 0     # WAIT zones
    
    # Events
    events: List[Event] = field(default_factory=list)
    
    # Status
    is_active: bool = True
    exit_time: Optional[datetime] = None


class EventTracker:
    """Track events for multiple people"""
    
    def __init__(self, config: Dict):
        """
        Initialize tracker
        
        Args:
            config: Configuration with dwell_time thresholds
        """
        self.config = config
        self.sessions: Dict[int, PersonSession] = {}  # track_id -> PersonSession
        self.dwell_thresholds = config.get("dwell_time", {})
        self.lock = __import__("threading").Lock()
    
    def update_person_zone(
        self,
        person_id: int,
        camera: str,
        current_zones: Set[str],
        max_person_age: float = 30.0
    ):
        """
        Update person's zone presence
        
        Args:
            person_id: Track ID
            camera: Camera name
            current_zones: Set of zones person is currently in
            max_person_age: Max time (sec) before marking person as inactive
        """
        with self.lock:
            current_time = datetime.now()
            
            # Get or create session
            if person_id not in self.sessions:
                self.sessions[person_id] = PersonSession(person_id, camera)
            
            session = self.sessions[person_id]
            session.last_seen = current_time
            
            # Check for zone exits
            zones_exited = set(session.current_zones.keys()) - current_zones
            for zone_name in zones_exited:
                del session.current_zones[zone_name]
                session.zone_counted.discard(zone_name)
            
            # Check for zone entries
            zones_entered = current_zones - set(session.current_zones.keys())
            for zone_name in zones_entered:
                session.current_zones[zone_name] = current_time
                session.visited_zones.add(zone_name)

            # Edge-agent style: count as soon as dwell >= threshold while still occupying zone.
            for zone_name in list(session.current_zones.keys()):
                if zone_name in session.zone_counted:
                    continue
                entry_time = session.current_zones[zone_name]
                dwell_time = (current_time - entry_time).total_seconds()
                threshold = self._get_dwell_threshold(zone_name)
                if dwell_time < threshold:
                    continue
                generated = self._generate_zone_event(
                    session, zone_name, dwell_time, current_time
                )
                if generated:
                    session.zone_counted.add(zone_name)

            # Cleanup inactive sessions
            self._cleanup_inactive_sessions(max_person_age)
    
    def _generate_zone_event(
        self,
        session: PersonSession,
        zone_name: str,
        dwell_time: float,
        timestamp: datetime
    ) -> bool:
        """Generate event based on zone and dwell time"""
        # Determine event type and count
        event_type = self._get_event_type(zone_name)
        if event_type is None:
            return False
        
        # Check dwell threshold
        threshold = self._get_dwell_threshold(zone_name)
        if dwell_time < threshold:
            return False  # Below threshold, don't count
        
        # Create event
        event = Event(
            timestamp=timestamp,
            camera=session.camera,
            event_type=event_type,
            person_id=session.person_id,
            zone_name=zone_name,
            dwell_seconds=dwell_time,
            metadata={
                "threshold": threshold,
                "qualified": dwell_time >= threshold
            }
        )
        
        session.events.append(event)
        
        # Update counter
        if event_type == EventType.CHAIR:
            session.haircut_count += 1
        elif event_type == EventType.WASH:
            session.wash_count += 1
        elif event_type == EventType.WAIT:
            session.wait_count += 1

        return True
    
    def _get_event_type(self, zone_name: str) -> Optional[EventType]:
        """Determine event type from zone name"""
        zone_upper = zone_name.upper()
        
        if zone_upper.startswith("CHAIR"):
            return EventType.CHAIR
        elif zone_upper.startswith("WASH"):
            return EventType.WASH
        elif zone_upper in ("WAIT", "WAITING", "WAIT_AREA"):
            return EventType.WAIT
        
        return None
    
    def _get_dwell_threshold(self, zone_name: str) -> float:
        """Get dwell time threshold for zone"""
        zone_upper = zone_name.upper()
        
        if zone_upper.startswith("CHAIR"):
            return self.dwell_thresholds.get("chair", 120)
        elif zone_upper.startswith("WASH"):
            return self.dwell_thresholds.get("wash", 60)
        elif zone_upper in ("WAIT", "WAITING", "WAIT_AREA"):
            return self.dwell_thresholds.get("wait", 30)
        
        return float('inf')  # Don't count unknown zones
    
    def _cleanup_inactive_sessions(self, max_age: float):
        """Remove inactive sessions"""
        current_time = datetime.now()
        to_remove = []
        
        for person_id, session in self.sessions.items():
            age = (current_time - session.last_seen).total_seconds()
            if age > max_age:
                to_remove.append(person_id)
        
        for person_id in to_remove:
            del self.sessions[person_id]
    
    def get_events(self, flush: bool = False) -> List[Event]:
        """Get all events"""
        with self.lock:
            events = []
            for session in self.sessions.values():
                events.extend(session.events)
            
            if flush:
                for session in self.sessions.values():
                    session.events = []
            
            return events
    
    def get_summary(self) -> Dict:
        """Get session summary"""
        with self.lock:
            total_haircuts = sum(s.haircut_count for s in self.sessions.values())
            total_washes = sum(s.wash_count for s in self.sessions.values())
            total_waits = sum(s.wait_count for s in self.sessions.values())
            active_people = len(self.sessions)
            
            return {
                "active_people": active_people,
                "haircuts": total_haircuts,
                "washes": total_washes,
                "waits": total_waits,
                "total_events": sum(len(s.events) for s in self.sessions.values()),
            }
    
    def get_person_session(self, person_id: int) -> Optional[PersonSession]:
        """Get person's session info"""
        with self.lock:
            return self.sessions.get(person_id)

    def reset(self):
        """Reset all in-memory sessions and counters."""
        with self.lock:
            self.sessions = {}
