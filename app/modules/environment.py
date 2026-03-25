"""
Guardian Eye — Environmental Analysis Module

Analyzes frame quality and environmental conditions for rescue op safety.

Metrics computed:
  1. Blur / fog index          — Laplacian variance (low = blurry/foggy)
  2. Visibility score          — Contrast + brightness analysis
  3. Rain intensity estimate   — Noise pattern detection in frame
  4. Smoke/dust detection      — Grayish flat histogram signature
  5. Overall op safety score   — Composite 0-1 score
  6. Recommendations           — Actionable text for commanders
"""

import cv2
import numpy as np
from typing import Dict, Any, List
from dataclasses import dataclass, field

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EnvironmentalReport:
    # Raw metrics
    laplacian_variance: float       # Higher = sharper image
    contrast_score: float           # 0-255 range (RMS contrast)
    brightness_mean: float          # 0-255
    brightness_ok: bool
    noise_level: float              # 0-1 estimated noise ratio
    fog_probability: float          # 0-1
    rain_probability: float         # 0-1
    smoke_probability: float        # 0-1

    # Derived
    visibility_score: float         # 0-1
    overall_safety_score: float     # 0-1  (1 = fully safe for ops)
    safety_level: str               # "SAFE" | "CAUTION" | "DANGER" | "ABORT"
    conditions: List[str] = field(default_factory=list)   # detected conditions
    recommendations: List[str] = field(default_factory=list)


def analyze_environment(frame: np.ndarray) -> EnvironmentalReport:
    """
    Full environmental analysis on a single frame.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape

    # ── 1. Laplacian Variance (blur / fog detector) ─────────────
    laplacian = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F)
    lap_var = float(laplacian.var())

    # ── 2. Contrast (RMS contrast) ──────────────────────────────
    contrast = float(gray.std())

    # ── 3. Brightness ───────────────────────────────────────────
    brightness = float(gray.mean())
    brightness_ok = settings.BRIGHTNESS_LOW <= brightness <= settings.BRIGHTNESS_HIGH

    # ── 4. Noise estimation (high-frequency residual) ────────────
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = np.abs(gray - blurred)
    noise_level = float(np.clip(residual.mean() / 20.0, 0, 1))

    # ── 5. Fog probability ───────────────────────────────────────
    # Fog = low Laplacian + narrow histogram + elevated mean brightness
    hist = cv2.calcHist([gray.astype(np.uint8)], [0], None, [256], [0, 256])
    hist = hist.flatten() / (h * w)
    hist_spread = float(np.sum(hist > 0.001))   # number of "active" histogram bins
    fog_prob = 0.0
    if lap_var < settings.LAPLACIAN_BLUR_THRESH:
        fog_prob += 0.5
    if hist_spread < 100:
        fog_prob += 0.3
    if brightness > 160:
        fog_prob += 0.2
    fog_prob = float(np.clip(fog_prob, 0, 1))

    # ── 6. Rain probability ──────────────────────────────────────
    # Rain = high noise + vertical streak pattern
    # Use Sobel horizontal to detect vertical streaks
    sobel_x = cv2.Sobel(gray.astype(np.uint8), cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray.astype(np.uint8), cv2.CV_64F, 0, 1, ksize=3)
    streak_ratio = float(np.abs(sobel_y).mean() / (np.abs(sobel_x).mean() + 1e-6))
    rain_prob = float(np.clip((noise_level * 0.5 + min(streak_ratio / 3, 0.5)), 0, 1))

    # ── 7. Smoke/dust probability ────────────────────────────────
    # Smoke = flat grayish mid-tone histogram, low saturation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    saturation_mean = float(hsv[:, :, 1].mean())
    smoke_prob = float(np.clip(
        (1.0 - saturation_mean / 128.0) * 0.7 + (1.0 - min(lap_var / 200, 1.0)) * 0.3,
        0, 1
    ))
    # Only flag smoke if also low contrast
    if contrast > 60:
        smoke_prob *= 0.3

    # ── 8. Visibility score ──────────────────────────────────────
    blur_score  = float(np.clip(lap_var / 200.0, 0, 1))
    cont_score  = float(np.clip(contrast / 80.0, 0, 1))
    bright_score = 1.0 if brightness_ok else max(0.0, 1.0 - abs(brightness - 127) / 127)
    visibility_score = round(blur_score * 0.5 + cont_score * 0.3 + bright_score * 0.2, 3)

    # ── 9. Overall op safety score ───────────────────────────────
    hazard = (fog_prob * 0.3 + rain_prob * 0.25 + smoke_prob * 0.2 +
              (1 - visibility_score) * 0.25)
    overall_safety = round(float(np.clip(1.0 - hazard, 0, 1)), 3)

    # ── 10. Safety level ─────────────────────────────────────────
    if overall_safety >= 0.75:
        safety_level = "SAFE"
    elif overall_safety >= 0.50:
        safety_level = "CAUTION"
    elif overall_safety >= 0.30:
        safety_level = "DANGER"
    else:
        safety_level = "ABORT"

    # ── 11. Conditions & Recommendations ─────────────────────────
    conditions: List[str] = []
    recommendations: List[str] = []

    if fog_prob > 0.5:
        conditions.append("FOG / LOW VISIBILITY")
        recommendations.append("Switch to thermal imaging mode for better detection.")
    if rain_prob > 0.4:
        conditions.append("RAIN / PRECIPITATION")
        recommendations.append("Reduce drone altitude for better visibility. Use waterproof UAV config.")
    if smoke_prob > 0.5:
        conditions.append("SMOKE / DUST")
        recommendations.append("Deploy LIDAR-equipped drone if available. Mark zone as hazardous.")
    if lap_var < settings.LAPLACIAN_BLUR_THRESH:
        conditions.append("BLURRY FEED")
        recommendations.append("Check gimbal stability. Clean camera lens.")
    if not brightness_ok:
        if brightness < settings.BRIGHTNESS_LOW:
            conditions.append("LOW LIGHT / NIGHT CONDITIONS")
            recommendations.append("Activate night-vision / thermal IR sensors.")
        else:
            conditions.append("OVEREXPOSED FRAME")
            recommendations.append("Adjust camera exposure settings on UAV.")
    if not conditions:
        conditions.append("CLEAR CONDITIONS")
        recommendations.append("Optimal conditions for rescue operation. Proceed normally.")

    return EnvironmentalReport(
        laplacian_variance=round(lap_var, 2),
        contrast_score=round(contrast, 2),
        brightness_mean=round(brightness, 2),
        brightness_ok=brightness_ok,
        noise_level=round(noise_level, 3),
        fog_probability=round(fog_prob, 3),
        rain_probability=round(rain_prob, 3),
        smoke_probability=round(smoke_prob, 3),
        visibility_score=visibility_score,
        overall_safety_score=overall_safety,
        safety_level=safety_level,
        conditions=conditions,
        recommendations=recommendations,
    )


def annotate_env_frame(frame: np.ndarray, report: EnvironmentalReport) -> np.ndarray:
    """
    Overlay environmental HUD on frame.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # HUD panel (bottom-left)
    panel_h = 130
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, h - panel_h), (310, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0, annotated)

    color_map = {"SAFE": (0, 220, 80), "CAUTION": (0, 200, 255),
                 "DANGER": (0, 80, 255), "ABORT": (0, 0, 255)}
    lvl_color = color_map.get(report.safety_level, (255, 255, 255))

    rows = [
        (f"ENV SAFETY: {report.safety_level}", lvl_color, 0.65),
        (f"Visibility: {int(report.visibility_score * 100)}%  |  Blur: {report.laplacian_variance:.1f}", (200, 200, 200), 0.48),
        (f"Fog: {int(report.fog_probability * 100)}%  Rain: {int(report.rain_probability * 100)}%  Smoke: {int(report.smoke_probability * 100)}%", (200, 200, 200), 0.48),
        (f"Brightness: {report.brightness_mean:.0f}  Contrast: {report.contrast_score:.0f}  Noise: {int(report.noise_level * 100)}%", (200, 200, 200), 0.45),
    ]

    for i, (text, color, scale) in enumerate(rows):
        cv2.putText(annotated, text, (8, h - panel_h + 22 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    return annotated
