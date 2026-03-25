"""
Guardian Eye — Depth Estimation & Helicopter Landing Zone Module

Uses MiDaS (monocular depth estimation) to:
1. Build a depth map from a single frame
2. Find flat, low-variance regions suitable for helicopter landing
3. Score landing zones and return safe/unsafe recommendations

MiDaS outputs: higher value = closer to camera (inverted depth)
For LZ detection we want: large, FLAT regions = similar depth values across region
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LandingZoneCandidate:
    center_x: int
    center_y: int
    area_px: int
    safety_score: float    # 0-1
    safe: bool
    depth_variance: float
    bbox: Tuple[int, int, int, int]  # x1,y1,x2,y2


class DepthAnalyzer:
    def __init__(self):
        self._model = None
        self._transform = None

    def _load_model(self):
        if self._model is None:
            import torch
            logger.info(f"Loading MiDaS model: {settings.MIDAS_MODEL}")
            self._model = torch.hub.load(
                "intel-isl/MiDaS",
                settings.MIDAS_MODEL,
                pretrained=True,
                trust_repo=True,
            )
            self._model.eval()

            midas_transforms = torch.hub.load(
                "intel-isl/MiDaS",
                "transforms",
                trust_repo=True,
            )
            if settings.MIDAS_MODEL in ("DPT_Large", "DPT_Hybrid"):
                self._transform = midas_transforms.dpt_transform
            else:
                self._transform = midas_transforms.small_transform

            logger.info("MiDaS loaded.")

    def estimate_depth(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Returns normalized depth map (float32, 0-1, same H×W as input).
        Returns None on failure.
        """
        try:
            import torch
            self._load_model()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_batch = self._transform(rgb)

            with torch.no_grad():
                prediction = self._model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=frame.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            depth = prediction.numpy()
            # Normalize 0-1
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            return depth.astype(np.float32)

        except Exception as e:
            logger.error(f"MiDaS depth estimation failed: {e}")
            return None

    def find_landing_zones(
        self,
        depth_map: np.ndarray,
        frame_shape: Tuple[int, int],
    ) -> List[LandingZoneCandidate]:
        """
        Identifies flat regions in the depth map as potential helicopter LZs.

        Strategy:
        - Divide frame into a grid of patches
        - For each patch, compute depth variance
        - Low variance + sufficient area = flat ground = potential LZ
        - Score based on: flatness, size, proximity to frame center, not too close
        """
        h, w = frame_shape[:2]
        grid_rows, grid_cols = 6, 8
        patch_h = h // grid_rows
        patch_w = w // grid_cols

        candidates: List[LandingZoneCandidate] = []

        for r in range(grid_rows):
            for c in range(grid_cols):
                y1 = r * patch_h
                y2 = min((r + 1) * patch_h, h)
                x1 = c * patch_w
                x2 = min((c + 1) * patch_w, w)

                patch = depth_map[y1:y2, x1:x2]
                if patch.size == 0:
                    continue

                variance = float(np.var(patch))
                mean_depth = float(np.mean(patch))
                area_px = (y2 - y1) * (x2 - x1)

                if (variance < settings.DEPTH_FLAT_VARIANCE_THRESH and
                        area_px >= settings.DEPTH_MIN_ZONE_AREA_PX):

                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    # Safety scoring
                    # 1. Flatness score (lower variance = better)
                    flatness = 1.0 - min(variance / settings.DEPTH_FLAT_VARIANCE_THRESH, 1.0)

                    # 2. Not too close to camera (depth mean not too high = not elevated)
                    depth_ok = 1.0 - mean_depth   # prefer far/ground-level

                    # 3. Not on the very edge of frame
                    edge_margin = 0.1
                    edge_penalty = 1.0
                    if (cx < w * edge_margin or cx > w * (1 - edge_margin) or
                            cy < h * edge_margin or cy > h * (1 - edge_margin)):
                        edge_penalty = 0.5

                    safety_score = round((flatness * 0.6 + depth_ok * 0.4) * edge_penalty, 3)
                    safe = safety_score >= settings.LZ_SAFE_SCORE_THRESH

                    candidates.append(LandingZoneCandidate(
                        center_x=cx,
                        center_y=cy,
                        area_px=area_px,
                        safety_score=safety_score,
                        safe=safe,
                        depth_variance=round(variance, 6),
                        bbox=(x1, y1, x2, y2),
                    ))

        # Sort by safety score descending
        candidates.sort(key=lambda z: z.safety_score, reverse=True)
        return candidates

    def annotate_depth_frame(
        self,
        frame: np.ndarray,
        depth_map: np.ndarray,
        zones: List[LandingZoneCandidate],
    ) -> np.ndarray:
        """
        Overlays depth colormap + LZ annotations on a copy of the frame.
        """
        # Depth visualization
        depth_vis = (depth_map * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)

        # Blend with original
        annotated = cv2.addWeighted(frame, 0.45, depth_colored, 0.55, 0)

        for zone in zones:
            x1, y1, x2, y2 = zone.bbox
            color = (0, 220, 80) if zone.safe else (0, 80, 220)
            label = f"LZ {'SAFE' if zone.safe else 'UNSAFE'} {int(zone.safety_score * 100)}%"
            thickness = 2 if zone.safe else 1

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            if zone.safe:
                # Landing pad crosshair
                cx, cy = zone.center_x, zone.center_y
                cv2.line(annotated, (cx - 15, cy), (cx + 15, cy), (0, 255, 150), 2)
                cv2.line(annotated, (cx, cy - 15), (cx, cy + 15), (0, 255, 150), 2)
                cv2.circle(annotated, (cx, cy), 18, (0, 255, 150), 1)

            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

        # Mode label
        cv2.putText(annotated, "DEPTH / LZ ANALYSIS", (6, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 255), 1, cv2.LINE_AA)

        return annotated


# Singleton
depth_analyzer = DepthAnalyzer()
