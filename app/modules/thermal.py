"""
Guardian Eye — Thermal Imaging Simulation Module

Converts standard RGB frames into a tactical thermal-style view.
Cold terrain background + hot human heat signatures.
Outputs:
  - thermal frame (annotated heat map)
  - per-person thermal score (heat intensity 0-1)
"""

import cv2
import numpy as np
from typing import List, Tuple

from app.modules.detection import Detection
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


def rgb_to_thermal(
    frame: np.ndarray,
    detections: List[Detection]
) -> Tuple[np.ndarray, List[float]]:
    """
    Convert BGR frame to tactical thermal pseudo-color.

    Pipeline:
    1. Convert to grayscale
    2. Create cool terrain background (COLORMAP_BONE)
    3. Enhance contrast using CLAHE
    4. Apply hot inferno heat ONLY to detected persons
    5. Overlay posture + heat score

    Returns:
        thermal_frame: Thermal-styled BGR frame
        thermal_scores: List of heat scores [0–1]
    """

    # ─────────────────────────────────────────────
    # 1. Convert to grayscale
    # ─────────────────────────────────────────────
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ─────────────────────────────────────────────
    # 2. Cool tactical terrain background
    # ─────────────────────────────────────────────
    cool_bg = cv2.applyColorMap(gray, cv2.COLORMAP_BONE)
    thermal_colored = cool_bg.copy()

    # ─────────────────────────────────────────────
    # 3. Local contrast enhancement (heat extraction)
    # ─────────────────────────────────────────────
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray).astype(np.float32)

    thermal_scores: List[float] = []

    # ─────────────────────────────────────────────
    # 4. Apply heat only to persons
    # ─────────────────────────────────────────────
    for det in detections:
        x1, y1, x2, y2 = det.bbox

        # Clamp to frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            thermal_scores.append(0.0)
            continue

        # Extract detected person region
        region = enhanced[y1:y2, x1:x2]

        # Base heat score (normalized brightness)
        base_score = float(region.mean()) / 255.0

        # Boost heat intensity
        boosted = np.clip(
            region * settings.THERMAL_PERSON_BOOST + 40,
            0,
            255
        ).astype(np.uint8)

        # Apply inferno heat map to person only
        person_heat = cv2.applyColorMap(
            boosted,
            settings.THERMAL_COLORMAP
        )

        # Paste hot region back to cold background
        thermal_colored[y1:y2, x1:x2] = person_heat

        # Final normalized heat score
        thermal_score = float(
            np.clip(base_score * settings.THERMAL_PERSON_BOOST, 0, 1)
        )

        thermal_scores.append(round(thermal_score, 3))

        # ─────────────────────────────────────────
        # 5. Overlay status + heat %
        # ─────────────────────────────────────────
        status_color = (0, 0, 255) if "INJURED" in det.status else (255, 255, 255)

        cv2.putText(
            thermal_colored,
            f"{det.status} - HEAT {int(thermal_score * 100)}%",
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            status_color,
            2,
            cv2.LINE_AA
        )

    # Optional watermark
    cv2.putText(
        thermal_colored,
        "THERMAL IR MODE",
        (8, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 255),
        1,
        cv2.LINE_AA
    )

    return thermal_colored, thermal_scores