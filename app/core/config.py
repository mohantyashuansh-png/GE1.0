"""
Guardian Eye — Configuration
All tunable parameters in one place.
"""

import os
from pathlib import Path
from typing import List


class Settings:
    # ── Paths ──────────────────────────────────────────────────
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    UPLOAD_DIR: str = str(BASE_DIR / "uploads")
    OUTPUT_DIR: str = str(BASE_DIR / "outputs")
    LOG_DIR: str    = str(BASE_DIR / "logs")

    # ── Models (GPU Targeting Enabled) ─────────────────────────
    YOLO_MODEL: str      = "yolov8n-pose.onnx"  # UPGRADED: Nano Pose model for injury detection
    DEVICE: str          = "cuda:0"           # FORCE GPU usage
    MIDAS_MODEL: str     = "MiDaS_small"      # small = fast on CPU/GPU
    MIDAS_TRANSFORM: str = "small_transform"

    # ── Performance / VRAM Protection ──────────────────────────
    FRAME_SKIP_RATE: int = 1  # Run heavy MiDaS/Env checks only once every 30 frames

    # ── Detection thresholds ───────────────────────────────────
    YOLO_CONF: float     = 0.35
    YOLO_IOU: float      = 0.45
    PERSON_CLASS_ID: int = 0

    # ── Tracking ───────────────────────────────────────────────
    TRACKER: str                  = "botsort.yaml"
    MAX_PERSON_LOG: int           = 500

    # ── Thermal simulation ─────────────────────────────────────
    THERMAL_COLORMAP: int         = 11   # cv2.COLORMAP_INFERNO
    THERMAL_PERSON_BOOST: float   = 1.4

    # ── MiDaS / Landing Zone ──────────────────────────────────
    DEPTH_FLAT_VARIANCE_THRESH: float = 0.015
    DEPTH_MIN_ZONE_AREA_PX: int       = 4000
    LZ_SAFE_SCORE_THRESH: float       = 0.65

    # ── Environmental analysis ────────────────────────────────
    LAPLACIAN_BLUR_THRESH: float  = 80.0
    CONTRAST_LOW_THRESH: float    = 40.0
    BRIGHTNESS_LOW: float         = 40.0
    BRIGHTNESS_HIGH: float        = 215.0
    WIND_SAFE_MAX_KMH: float      = 45.0
    RAIN_INTENSITY_THRESH: float  = 0.3

    # ── GPS (dummy seed — replace with real EXIF/telemetry) ──
    GPS_BASE_LAT: float           = 30.3165
    GPS_BASE_LON: float           = 78.0322
    GPS_JITTER_M: float           = 500.0

    # ── Alerts ────────────────────────────────────────────────
    ALERT_COOLDOWN_SEC: float      = 5.0
    HIGH_PRIORITY_MIN_PERSONS: int = 3

    # ── CORS ──────────────────────────────────────────────────
    ALLOWED_ORIGINS: List[str]     = ["*"]

    # ── Webcam ────────────────────────────────────────────────
    WEBCAM_INDEX: int = 0
    STREAM_FPS: int   = 15

    # "0" = Laptop Webcam | "video.mp4" = File | "http://..." = Phone Drone
    VIDEO_SOURCE: str = 0

    def __init__(self):
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)


settings = Settings()