"""
Guardian Eye — In-Memory State Store
Holds detection logs, person registry, and alerts.
In production, swap with Redis or PostgreSQL.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import threading


@dataclass
class DetectedPerson:
    person_id: str           # e.g. "P-0001"
    track_id: int
    first_seen: str          # ISO timestamp
    last_seen: str
    frame_count: int         # how many frames detected
    confidence: float
    bbox: List[float]        # [x1, y1, x2, y2] in last frame
    gps_lat: float
    gps_lon: float
    thermal_score: float     # 0-1, heat signature intensity
    status: str              # "ACTIVE" | "LOST" | "RESCUED"
    thumbnail_path: Optional[str] = None


@dataclass
class Alert:
    alert_id: str
    timestamp: str
    level: str               # "INFO" | "WARNING" | "HIGH" | "CRITICAL"
    message: str
    person_ids: List[str]
    gps_lat: float
    gps_lon: float
    frame_path: Optional[str] = None
    acknowledged: bool = False


@dataclass
class LandingZone:
    lz_id: str
    timestamp: str
    center_x: int
    center_y: int
    area_px: int
    safety_score: float      # 0-1
    safe: bool
    gps_lat: float
    gps_lon: float
    depth_variance: float


class StateStore:
    def __init__(self):
        self._lock = threading.Lock()
        self.persons: Dict[str, DetectedPerson] = {}    # person_id -> DetectedPerson
        self.track_to_pid: Dict[int, str] = {}          # track_id -> person_id
        self.alerts: List[Alert] = []
        self.landing_zones: List[LandingZone] = []
        self._person_counter = 0
        self._alert_counter = 0
        self._lz_counter = 0

    # ── Person management ──────────────────────────────────────
    def get_or_create_person(self, track_id: int, confidence: float,
                              bbox: List[float], gps_lat: float, gps_lon: float,
                              thermal_score: float) -> DetectedPerson:
        with self._lock:
            now = datetime.utcnow().isoformat()
            if track_id in self.track_to_pid:
                pid = self.track_to_pid[track_id]
                p = self.persons[pid]
                p.last_seen = now
                p.frame_count += 1
                p.confidence = max(p.confidence, confidence)
                p.bbox = bbox
                p.gps_lat = gps_lat
                p.gps_lon = gps_lon
                p.thermal_score = thermal_score
                p.status = "ACTIVE"
                return p
            else:
                self._person_counter += 1
                pid = f"P-{self._person_counter:04d}"
                p = DetectedPerson(
                    person_id=pid,
                    track_id=track_id,
                    first_seen=now,
                    last_seen=now,
                    frame_count=1,
                    confidence=confidence,
                    bbox=bbox,
                    gps_lat=gps_lat,
                    gps_lon=gps_lon,
                    thermal_score=thermal_score,
                    status="ACTIVE",
                )
                self.persons[pid] = p
                self.track_to_pid[track_id] = pid
                return p

    def get_all_persons(self) -> List[DetectedPerson]:
        with self._lock:
            return list(self.persons.values())

    def mark_lost(self, track_id: int):
        with self._lock:
            if track_id in self.track_to_pid:
                pid = self.track_to_pid[track_id]
                self.persons[pid].status = "LOST"

    # ── Alert management ───────────────────────────────────────
    def add_alert(self, level: str, message: str, person_ids: List[str],
                  gps_lat: float, gps_lon: float,
                  frame_path: Optional[str] = None) -> Alert:
        with self._lock:
            self._alert_counter += 1
            alert = Alert(
                alert_id=f"ALT-{self._alert_counter:04d}",
                timestamp=datetime.utcnow().isoformat(),
                level=level,
                message=message,
                person_ids=person_ids,
                gps_lat=gps_lat,
                gps_lon=gps_lon,
                frame_path=frame_path,
            )
            self.alerts.append(alert)
            return alert

    def get_alerts(self, limit: int = 50, unacked_only: bool = False) -> List[Alert]:
        with self._lock:
            alerts = self.alerts[-limit:][::-1]
            if unacked_only:
                alerts = [a for a in alerts if not a.acknowledged]
            return alerts

    def acknowledge_alert(self, alert_id: str) -> bool:
        with self._lock:
            for a in self.alerts:
                if a.alert_id == alert_id:
                    a.acknowledged = True
                    return True
            return False

    # ── Landing zones ──────────────────────────────────────────
    def add_landing_zone(self, lz: LandingZone):
        with self._lock:
            self.landing_zones.append(lz)

    def get_latest_lzs(self, limit: int = 10) -> List[LandingZone]:
        with self._lock:
            return self.landing_zones[-limit:][::-1]

    # ── Reset ──────────────────────────────────────────────────
    def reset(self):
        with self._lock:
            self.persons.clear()
            self.track_to_pid.clear()
            self.alerts.clear()
            self.landing_zones.clear()
            self._person_counter = 0
            self._alert_counter = 0
            self._lz_counter = 0


# Singleton
store = StateStore()
