"""
Guardian Eye — Core Analysis Pipeline

Orchestrates all modules for a single frame.
"""

import cv2
import numpy as np
import time
import os
import threading  # <--- NEW: For background image saving
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from app.modules.detection import detector, Detection
from app.modules.thermal import rgb_to_thermal
from app.modules.depth import depth_analyzer, LandingZoneCandidate
from app.modules.environment import analyze_environment, annotate_env_frame
from app.modules.alerts_engine import process_alerts
from app.modules.vip_tracker import vip_tracker  # <--- NEW: VIP Tracker Import
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
    annotated_path: Optional[str] = None
    thermal_path: Optional[str] = None
    depth_path: Optional[str] = None


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

    # ── 1. GPS ────────────────────────────────────────────────
    gps_lat, gps_lon = get_gps_from_frame_index(frame_index, total_frames)

    # ── 2. Person detection + tracking ────────────────────────
    detections: List[Detection] = detector.detect(frame, use_tracking=True)

    # ── 3. Update state store & Check VIP Status ──────────────
    person_ids: Dict[int, str] = {}
    persons_out: List[Dict] = []

    for det in detections:
        p_lat, p_lon = get_dummy_gps(seed=det.track_id)

        # ---> NEW: Check if this person matches the VIP description
        is_vip = vip_tracker.check_vip_match(frame, det.bbox)
        if is_vip:
            det.status = "VIP TARGET ACQUIRED"

        person = store.get_or_create_person(
            track_id=det.track_id,
            confidence=det.confidence,
            bbox=list(det.bbox),
            gps_lat=p_lat,
            gps_lon=p_lon,
            thermal_score=0.6,
        )

        # If they were just flagged as VIP, update the store's status too
        if is_vip:
            person.status = "VIP TARGET ACQUIRED"

        person_ids[det.track_id] = person.person_id

        persons_out.append({
            "person_id": person.person_id,
            "track_id": det.track_id,
            "confidence": round(det.confidence, 3),
            "bbox": list(det.bbox),
            "center": list(det.center),
            "gps_lat": p_lat,
            "gps_lon": p_lon,
            "first_seen": person.first_seen,
            "last_seen": person.last_seen,
            "frame_count": person.frame_count,
            "status": person.status,
        })

    # ── 4. Annotated detection frame ──────────────────────────
    annotated_frame = detector.annotate_frame(frame, detections, person_ids)

    # ── 5. Thermal frame ──────────────────────────────────────
    thermal_frame, thermal_scores = rgb_to_thermal(frame, detections)

    for i, det in enumerate(detections):
        pid = person_ids.get(det.track_id)
        if pid and i < len(thermal_scores) and pid in store.persons:
            store.persons[pid].thermal_score = thermal_scores[i]
            persons_out[i]["thermal_score"] = thermal_scores[i]

    # ── 6. Depth + Landing Zones (GPU SAVER LOGIC) ────────────
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

    # ── 7. Environmental analysis (CPU SAVER) ─────────────────
    env_out: Dict[str, Any] = {}

    if run_heavy_modules:
        env_report = analyze_environment(frame)
        annotated_frame = annotate_env_frame(annotated_frame, env_report)

        env_out = {
            "laplacian_variance": env_report.laplacian_variance,
            "contrast_score": env_report.contrast_score,
            "brightness_mean": env_report.brightness_mean,
            "brightness_ok": env_report.brightness_ok,
            "noise_level": env_report.noise_level,
            "fog_probability": env_report.fog_probability,
            "rain_probability": env_report.rain_probability,
            "smoke_probability": env_report.smoke_probability,
            "visibility_score": env_report.visibility_score,
            "overall_safety_score": env_report.overall_safety_score,
            "safety_level": env_report.safety_level,
            "conditions": env_report.conditions,
            "recommendations": env_report.recommendations,
        }

    # ── 8. Alerts ─────────────────────────────────────────────
    alerts_fired = process_alerts(
        detections=detections,
        env_report=env_out if run_heavy_modules else None,
        frame_gps=(gps_lat, gps_lon),
        person_ids=person_ids,
    )

    # ── 9. Save frames (NON-BLOCKING) ─────────────────────────
    annotated_path = thermal_path = depth_path = None

    if save_frames:
        base = os.path.join(settings.OUTPUT_DIR, job_id)
        os.makedirs(base, exist_ok=True)

        annotated_path = f"{base}/frame_{frame_index:06d}_det.jpg"
        thermal_path   = f"{base}/frame_{frame_index:06d}_therm.jpg"
        depth_path     = f"{base}/frame_{frame_index:06d}_depth.jpg"

        # ---> STUTTER FIX: Offload file writing to a background thread
        def save_to_disk():
            cv2.imwrite(annotated_path, annotated_frame)
            cv2.imwrite(thermal_path, thermal_frame)
            cv2.imwrite(depth_path, depth_frame)
            
        threading.Thread(target=save_to_disk, daemon=True).start()

    elapsed = round(time.time() - t0, 3)
    logger.debug(
        f"Frame {frame_index} processed in {elapsed}s — "
        f"{len(detections)} persons — HeavyModules: {run_heavy_modules}"
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
        annotated_path=annotated_path,
        thermal_path=thermal_path,
        depth_path=depth_path,
    )