"""
Guardian Eye — Core Analysis Pipeline
Orchestrates all modules for a single frame.
"""

import cv2
import math
import numpy as np
import time
import os
import threading
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


def analyze_skeletal_posture(keypoints_array):
    """
    Calculates the spine angle using trigonometry on YOLOv8-Pose keypoints.
    COCO Keypoint Indices: 5=Left Shoulder, 6=Right Shoulder, 11=Left Hip, 12=Right Hip
    """
    try:
        if (keypoints_array[5][2] < 0.3 or keypoints_array[6][2] < 0.3 or 
            keypoints_array[11][2] < 0.3 or keypoints_array[12][2] < 0.3):
            return None

        ls_x, ls_y = keypoints_array[5][:2]
        rs_x, rs_y = keypoints_array[6][:2]
        lh_x, lh_y = keypoints_array[11][:2]
        rh_x, rh_y = keypoints_array[12][:2]

        mid_shoulder_x = (ls_x + rs_x) / 2.0
        mid_shoulder_y = (ls_y + rs_y) / 2.0
        mid_hip_x = (lh_x + rh_x) / 2.0
        mid_hip_y = (lh_y + rh_y) / 2.0

        dx = abs(mid_shoulder_x - mid_hip_x)
        dy = abs(mid_shoulder_y - mid_hip_y)

        if dy == 0:
            return "LYING DOWN / INJURED"

        angle_from_vertical = math.degrees(math.atan(dx / dy))

        if angle_from_vertical < 35:
            return "STANDING"
        elif angle_from_vertical < 65:
            return "SLUMPED / INJURED"
        else:
            return "LYING DOWN / INJURED"

    except Exception:
        return None


def process_frame(
    frame: np.ndarray,
    frame_index: int = 0,
    total_frames: int = 1,
    job_id: str = "live",
    run_depth: bool = True,
    save_frames: bool = False,
) -> FrameResult:

    t0 = time.time()

    from datetime import datetime
    ts = datetime.utcnow().isoformat()

    gps_lat, gps_lon = get_gps_from_frame_index(frame_index, total_frames)

    detections: List[Detection] = detector.detect(frame, use_tracking=True)

    person_ids: Dict[int, str] = {}
    persons_out: List[Dict] = []

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        p_lat, p_lon = get_dummy_gps(seed=det.track_id)

        is_vip = vip_tracker.check_vip_match(frame, det.bbox)

        # 1. Default fallback (The old way: Bounding Box Ratio)
        w = x2 - x1
        h = y2 - y1
        posture_status = "LYING DOWN / INJURED" if w > (h * 1.2) else "STANDING"

        # 2. 🚀 The Advanced Skeletal Override (The new way: Spine Angle)
        if hasattr(det, "keypoints") and det.keypoints is not None and len(det.keypoints) > 0:
            skeletal_status = analyze_skeletal_posture(det.keypoints[0].cpu().numpy())

            if skeletal_status is not None:
                posture_status = skeletal_status

        # 3. Apply VIP Status safely
        if is_vip:
            det.status = f"VIP | {posture_status}"
        else:
            det.status = posture_status

        priority_score = 10

        if "INJURED" in posture_status.upper() or "LYING" in posture_status.upper():
            priority_score += 50

        if det.track_id not in store.track_to_pid:
            store.log_timeline_event(
                f"New survivor tracked: ID P-{det.track_id}",
                "DETECTION"
            )

        person = store.get_or_create_person(
            track_id=det.track_id,
            confidence=det.confidence,
            bbox=list(det.bbox),
            gps_lat=p_lat,
            gps_lon=p_lon,
            thermal_score=0.6,
        )

        person.last_seen_epoch = time.time()

        old_posture = person.status

        if old_posture == "Standing" and posture_status == "Lying Down / Injured":
            store.log_timeline_event(
                f"CRITICAL: ID {person.person_id} has collapsed!",
                "CRITICAL"
            )

        if is_vip:
            person.status = f"VIP | {posture_status}"
        else:
            person.status = posture_status

        person.x = center_x
        person.y = center_y
        person.priority_score = priority_score
        person.is_critical = priority_score >= 50

        if person.is_critical and not getattr(person, "was_critical", False):
            store.log_timeline_event(
                f"CRITICAL: ID {person.person_id} flagged as high priority!",
                "CRITICAL"
            )
            person.was_critical = True

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
            "first_seen": person.first_seen,
            "last_seen": person.last_seen,
            "frame_count": person.frame_count,
            "status": person.status,
        })

    annotated_frame = detector.annotate_frame(frame, detections, person_ids)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    high_contrast = clahe.apply(gray_frame)

    quantized_gray = (high_contrast // 64) * 64
    thermal_frame = cv2.applyColorMap(quantized_gray, cv2.COLORMAP_JET)

    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        cv2.rectangle(thermal_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    lz_out: List[Dict] = []
    depth_frame = frame.copy()

    run_heavy_modules = (frame_index % settings.FRAME_SKIP_RATE == 0)

    if run_depth and run_heavy_modules:
        depth_map = depth_analyzer.estimate_depth(frame)

        if depth_map is not None:
            zones: List[LandingZoneCandidate] = depth_analyzer.find_landing_zones(
                depth_map, frame.shape
            )
            depth_frame = depth_analyzer.annotate_depth_frame(frame, depth_map, zones)

            for z in zones[:5]:
                lz_gps = get_dummy_gps(seed=z.center_x * 1000 + z.center_y)
                lz = LandingZone(
                    lz_id=f"LZ-{frame_index:04d}-{z.center_x}",
                    timestamp=ts,
                    center_x=z.center_x,
                    center_y=z.center_y,
                    area_px=z.area_px,
                    safety_score=z.safety_score,
                    safe=z.safe,
                    gps_lat=lz_gps[0],
                    gps_lon=lz_gps[1],
                    depth_variance=z.depth_variance,
                )
                store.add_landing_zone(lz)

                lz_out.append({
                    "lz_id": lz.lz_id,
                    "center_x": z.center_x,
                    "center_y": z.center_y,
                    "area_px": z.area_px,
                    "safety_score": z.safety_score,
                    "safe": z.safe,
                    "gps_lat": lz_gps[0],
                    "gps_lon": lz_gps[1],
                    "depth_variance": z.depth_variance,
                })

    env_out: Dict[str, Any] = {}
    raw_env_report = None

    if run_heavy_modules:
        raw_env_report = analyze_environment(frame)
        annotated_frame = annotate_env_frame(annotated_frame, raw_env_report)
        env_out = {
            "visibility_score": raw_env_report.visibility_score,
            "overall_safety_score": raw_env_report.overall_safety_score,
            "safety_level": raw_env_report.safety_level,
            "conditions": raw_env_report.conditions,
            "recommendations": raw_env_report.recommendations,
        }

    alerts_fired = process_alerts(
        detections=detections,
        env_report=raw_env_report,
        frame_gps=(gps_lat, gps_lon),
        person_ids=person_ids,
    )

    elapsed = round(time.time() - t0, 3)

    logger.debug(
        f"Frame {frame_index} processed in {elapsed}s — "
        f"{len(detections)} persons"
    )

    return FrameResult(
        frame_index=frame_index,
        timestamp=ts,
        persons=persons_out,
        landing_zones=lz_out,
        environment=env_out,
        alerts_fired=alerts_fired,
        gps_lat=gps_lat,
        gps_lon=gps_lon,
        person_count=len(detections),
        annotated_frame=annotated_frame,
        thermal_frame=thermal_frame,
        depth_frame=depth_frame,
    )