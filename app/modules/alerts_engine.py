"""
Guardian Eye — Alert Engine
Generates prioritized command alerts from detection and environmental data.
Implements cooldown to prevent alert flooding.
"""

from typing import List, Dict
from datetime import datetime
import time

from app.core.state import store
from app.core.config import settings
from app.modules.detection import Detection
from app.modules.environment import EnvironmentalReport
from app.core.logger import get_logger

logger = get_logger(__name__)

# Track last alert time per person to avoid flooding
_last_alert_time: Dict[str, float] = {}


def process_alerts(
    detections: List[Detection],
    env_report: EnvironmentalReport,
    frame_gps: tuple,
    person_ids: Dict[int, str],
    frame_path: str = None,
) -> List[dict]:
    """
    Evaluate detections + env conditions and fire alerts as needed.
    Returns list of newly created alert dicts.
    """
    now = time.time()
    new_alerts = []

    # ── New person detected ───────────────────────────────────
    for det in detections:
        pid = person_ids.get(det.track_id)
        if not pid:
            continue
        last_time = _last_alert_time.get(pid, 0)
        if now - last_time < settings.ALERT_COOLDOWN_SEC:
            continue

        person = store.persons.get(pid)
        is_new = person and person.frame_count <= 2

        if is_new:
            alert = store.add_alert(
                level="WARNING",
                message=f"NEW PERSON DETECTED: {pid} at ({frame_gps[0]:.5f}, {frame_gps[1]:.5f}) — Conf {int(det.confidence * 100)}%",
                person_ids=[pid],
                gps_lat=frame_gps[0],
                gps_lon=frame_gps[1],
                frame_path=frame_path,
            )
            _last_alert_time[pid] = now
            new_alerts.append(_alert_to_dict(alert))
            logger.info(f"Alert fired: NEW PERSON {pid}")

    # ── Multiple persons in frame ─────────────────────────────
    if len(detections) >= settings.HIGH_PRIORITY_MIN_PERSONS:
        pids = [person_ids[d.track_id] for d in detections if d.track_id in person_ids]
        key = "MULTI_" + "_".join(sorted(pids))
        if now - _last_alert_time.get(key, 0) > 30:
            alert = store.add_alert(
                level="HIGH",
                message=f"HIGH DENSITY ZONE: {len(detections)} persons detected in single frame. Possible mass casualty site.",
                person_ids=pids,
                gps_lat=frame_gps[0],
                gps_lon=frame_gps[1],
                frame_path=frame_path,
            )
            _last_alert_time[key] = now
            new_alerts.append(_alert_to_dict(alert))
            logger.warning(f"HIGH alert: {len(detections)} persons in frame")

    # ── Environmental danger ──────────────────────────────────
    # Safely get the safety level whether it's an object or a dictionary
    safety = env_report.get("safety_level", "SAFE") if isinstance(env_report, dict) else getattr(env_report, "safety_level", "SAFE")
    
    if safety in ("DANGER", "ABORT"):
        key = f"ENV_{env_report.safety_level}"
        if now - _last_alert_time.get(key, 0) > 60:
            alert = store.add_alert(
                level="CRITICAL" if env_report.safety_level == "ABORT" else "WARNING",
                message=f"ENVIRONMENTAL ALERT [{env_report.safety_level}]: {', '.join(env_report.conditions)}. {env_report.recommendations[0] if env_report.recommendations else ''}",
                person_ids=[],
                gps_lat=frame_gps[0],
                gps_lon=frame_gps[1],
            )
            _last_alert_time[key] = now
            new_alerts.append(_alert_to_dict(alert))
            logger.warning(f"ENV alert: {env_report.safety_level}")

    return new_alerts


def _alert_to_dict(alert) -> dict:
    return {
        "alert_id": alert.alert_id,
        "timestamp": alert.timestamp,
        "level": alert.level,
        "message": alert.message,
        "person_ids": alert.person_ids,
        "gps_lat": alert.gps_lat,
        "gps_lon": alert.gps_lon,
        "acknowledged": alert.acknowledged,
    }
