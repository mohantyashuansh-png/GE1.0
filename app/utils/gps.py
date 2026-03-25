"""
Guardian Eye — GPS Utility
Generates GPS coordinates for detections.
- Real mode: extract from drone video EXIF/telemetry metadata
- Demo mode: seed-based realistic jitter around a base coordinate
"""

import math
import random
from typing import Tuple, Optional
from app.core.config import settings


def _metres_to_degrees(metres: float) -> Tuple[float, float]:
    """Convert metre offset to approximate lat/lon delta."""
    lat_deg = metres / 111_320
    lon_deg = metres / (111_320 * math.cos(math.radians(settings.GPS_BASE_LAT)))
    return lat_deg, lon_deg


def get_dummy_gps(seed: Optional[int] = None) -> Tuple[float, float]:
    """
    Returns (lat, lon) jittered around the configured base coordinate.
    Use seed=track_id for stable per-person GPS (same person → same location).
    """
    rng = random.Random(seed)
    jitter = settings.GPS_JITTER_M
    dlat, dlon = _metres_to_degrees(rng.uniform(-jitter, jitter))
    lat = settings.GPS_BASE_LAT + dlat
    lon = settings.GPS_BASE_LON + _metres_to_degrees(rng.uniform(-jitter, jitter))[1]
    return round(lat, 6), round(lon, 6)


def get_gps_from_frame_index(frame_idx: int, total_frames: int) -> Tuple[float, float]:
    """
    Simulate drone flight path GPS.
    Drone moves in a sweeping grid pattern over the base zone.
    """
    # Simple linear sweep — drone flies N-S rows
    row_count = 5
    row = (frame_idx / max(total_frames, 1)) * row_count
    row_int = int(row)
    col_frac = (frame_idx % max(total_frames // row_count, 1)) / max(total_frames // row_count, 1)

    # Alternate row direction (boustrophedon / lawnmower pattern)
    if row_int % 2 == 1:
        col_frac = 1.0 - col_frac

    span_lat = 0.02   # ~2.2 km
    span_lon = 0.025  # ~2.2 km

    lat = settings.GPS_BASE_LAT + (row / row_count - 0.5) * span_lat
    lon = settings.GPS_BASE_LON + (col_frac - 0.5) * span_lon
    return round(lat, 6), round(lon, 6)


def extract_gps_from_exif(video_path: str) -> Optional[Tuple[float, float]]:
    """
    Try to extract GPS from video metadata using ffprobe.
    Returns None if not available (falls back to dummy).
    """
    try:
        import subprocess, json
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", video_path],
            capture_output=True, text=True, timeout=5
        )
        data = json.loads(result.stdout)
        tags = data.get("format", {}).get("tags", {})
        loc = tags.get("location") or tags.get("com.apple.quicktime.location.ISO6709")
        if loc:
            # Parse ISO 6709 format: +30.3165+078.0322/
            import re
            parts = re.findall(r"[+-]\d+\.?\d*", loc)
            if len(parts) >= 2:
                return float(parts[0]), float(parts[1])
    except Exception:
        pass
    return None
