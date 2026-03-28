"""
Guardian Eye — Core Analysis Pipeline
Orchestrates all modules for a single frame.
"""

import cv2
import math
import numpy as np
import time
import os
import sqlite3
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from app.modules.detection import detector, Detection
from app.modules.depth import depth_analyzer, LandingZoneCandidate
from app.modules.environment import analyze_environment, annotate_env_frame
from app.modules.alerts_engine import process_alerts
from app.modules.vip_tracker import vip_tracker
from app.core.state import store, LandingZone
from app.utils.gps import get_dummy_gps, get_gps_from_frame_index
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────
# 📦 DB PATH
# ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "..", "db", "missions.db")

# prevent DB spam (important)
_last_saved_track = {}

# ─────────────────────────────────────────
# 💾 SAVE TO DATABASE
# ─────────────────────────────────────────
def save_detection_to_db(track_id, posture, score, sector="Nagpur-Main"):
    try:
        global _last_saved_track

        now = time.time()

        # cooldown (2 sec per track)
        if track_id in _last_saved_track:
            if now - _last_saved_track[track_id] < 2:
                return

        _last_saved_track[track_id] = now

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute(
            """
            INSERT INTO telemetry (timestamp, posture, score, sector)
            VALUES (?, ?, ?, ?)
            """,
            (timestamp, posture, score, sector)
        )

        conn.commit()
        conn.close()

    except Exception as e:
        logger.error(f"DB Save Error: {e}")


# ─────────────────────────────────────────
# 📊 FRAME RESULT
# ─────────────────────────────────────────
@dataclass
class FrameResult:
    frame_index: int
    timestamp: str
    persons: List[Dict]
    landing_zones: List[Dict]
    environment: Dict
    alerts_fired: List[Dict]
    gps_lat: float
    gps_lon: float
    person_count: int

    annotated_frame: Optional[np.ndarray] = None
    thermal_frame: Optional[np.ndarray] = None
    depth_frame: Optional[np.ndarray] = None


# ─────────────────────────────────────────
# 🧠 POSTURE ANALYSIS
# ─────────────────────────────────────────
def analyze_skeletal_posture(keypoints_array):
    try:
        if (keypoints_array[5][2] < 0.3 or keypoints_array[6][2] < 0.3 or
            keypoints_array[11][2] < 0.3 or keypoints_array[12][2] < 0.3):
            return None

        ls_x, ls_y = keypoints_array[5][:2]
        rs_x, rs_y = keypoints_array[6][:2]
        lh_x, lh_y = keypoints_array[11][:2]
        rh_x, rh_y = keypoints_array[12][:2]

        mid_shoulder_x = (ls_x + rs_x) / 2
        mid_shoulder_y = (ls_y + rs_y) / 2
        mid_hip_x = (lh_x + rh_x) / 2
        mid_hip_y = (lh_y + rh_y) / 2

        dx = abs(mid_shoulder_x - mid_hip_x)
        dy = abs(mid_shoulder_y - mid_hip_y)

        if dy == 0:
            return "LYING DOWN / INJURED"

        angle = math.degrees(math.atan(dx / dy))

        if angle < 35:
            return "STANDING"
        elif angle < 65:
            return "SLUMPED / INJURED"
        else:
            return "LYING DOWN / INJURED"

    except:
        return None


# ─────────────────────────────────────────
# 🚀 MAIN PIPELINE
# ─────────────────────────────────────────
def process_frame(
    frame: np.ndarray,
    frame_index: int = 0,
    total_frames: int = 1,
    job_id: str = "live",
    run_depth: bool = True,
    save_frames: bool = False,
) -> FrameResult:

    t0 = time.time()
    ts = datetime.utcnow().isoformat()

    gps_lat, gps_lon = get_gps_from_frame_index(frame_index, total_frames)

    detections: List[Detection] = detector.detect(frame, use_tracking=True)

    person_ids: Dict[int, str] = {}
    persons_out: List[Dict] = []

    # ─────────────────────────────
    # PERSON LOOP
    # ─────────────────────────────
    for det in detections:

        x1, y1, x2, y2 = det.bbox
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        p_lat, p_lon = get_dummy_gps(seed=det.track_id)

        is_vip = vip_tracker.check_vip_match(frame, det.bbox)

        w = x2 - x1
        h = y2 - y1

        posture_status = "LYING DOWN / INJURED" if w > (h * 1.2) else "STANDING"

        if hasattr(det, "keypoints") and det.keypoints is not None:
            sk = analyze_skeletal_posture(det.keypoints[0].cpu().numpy())
            if sk:
                posture_status = sk

        # VIP tag
        if is_vip:
            det.status = f"VIP | {posture_status}"
        else:
            det.status = posture_status

        priority_score = 10
        if "INJURED" in posture_status or "LYING" in posture_status:
            priority_score += 50

        # ─────────────────────────────
        # 💾 DB SAVE (FIXED + SAFE)
        # ─────────────────────────────
        save_detection_to_db(det.track_id, posture_status, priority_score)

        person = store.get_or_create_person(
            track_id=det.track_id,
            confidence=det.confidence,
            bbox=list(det.bbox),
            gps_lat=p_lat,
            gps_lon=p_lon,
            thermal_score=0.6,
        )

        person.status = posture_status
        person.x = center_x
        person.y = center_y
        person.priority_score = priority_score
        person.is_critical = priority_score >= 50

        person_ids[det.track_id] = person.person_id

        persons_out.append({
            "person_id": person.person_id,
            "track_id": det.track_id,
            "confidence": round(det.confidence, 3),
            "bbox": list(det.bbox),
            "center": [center_x, center_y],
            "gps_lat": p_lat,
            "gps_lon": p_lon,
            "priority_score": priority_score,
            "is_critical": person.is_critical,
            "status": person.status,
        })

    # ─────────────────────────────
    # FRAME RENDER
    # ─────────────────────────────
    annotated_frame = detector.annotate_frame(frame, detections, person_ids)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    high = clahe.apply(gray)
    thermal_frame = cv2.applyColorMap((high // 64) * 64, cv2.COLORMAP_JET)

    lz_out = []
    depth_frame = frame.copy()

    run_heavy = (frame_index % settings.FRAME_SKIP_RATE == 0)

    if run_depth and run_heavy:
        depth_map = depth_analyzer.estimate_depth(frame)

        if depth_map is not None:
            zones = depth_analyzer.find_landing_zones(depth_map, frame.shape)
            depth_frame = depth_analyzer.annotate_depth_frame(frame, depth_map, zones)

            for z in zones[:5]:
                lz_gps = get_dummy_gps(seed=z.center_x + z.center_y)

                lz_out.append({
                    "lz_id": f"LZ-{frame_index}-{z.center_x}",
                    "center_x": z.center_x,
                    "center_y": z.center_y,
                    "safety_score": z.safety_score,
                    "gps_lat": lz_gps[0],
                    "gps_lon": lz_gps[1],
                })

    env_out = {}

    if run_heavy:
        env = analyze_environment(frame)
        annotated_frame = annotate_env_frame(annotated_frame, env)

        env_out = {
            "visibility_score": env.visibility_score,
            "safety_level": env.safety_level,
            "conditions": env.conditions,
        }

    alerts = process_alerts(
        detections=detections,
        env_report=env if run_heavy else None,
        frame_gps=(gps_lat, gps_lon),
        person_ids=person_ids,
    )

    logger.debug(f"Frame {frame_index} processed | persons: {len(detections)}")

    return FrameResult(
        frame_index=frame_index,
        timestamp=ts,
        persons=persons_out,
        landing_zones=lz_out,
        environment=env_out,
        alerts_fired=alerts,
        gps_lat=gps_lat,
        gps_lon=gps_lon,
        person_count=len(detections),
        annotated_frame=annotated_frame,
        thermal_frame=thermal_frame,
        depth_frame=depth_frame,
    )