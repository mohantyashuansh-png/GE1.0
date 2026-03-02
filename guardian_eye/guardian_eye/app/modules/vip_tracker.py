"""
Guardian Eye — Dynamic VIP Target Matcher
Accepts dynamic HSV arrays calculated by the Frontend Agentic LLM.
Zero-VRAM footprint (runs entirely on CPU via OpenCV).
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List

class VIPTracker:
    def __init__(self):
        # Now stores dynamic math bounds instead of string names
        self.active_target_top: Optional[Dict[str, np.ndarray]] = None
        self.active_target_bottom: Optional[Dict[str, np.ndarray]] = None

    def set_dynamic_target(self, top_hsv: Optional[Dict], bottom_hsv: Optional[Dict]):
        """
        Receives exact HSV bounds from the FastAPI router.
        Expected format for top_hsv: {"lower": [h, s, v], "upper": [h, s, v]}
        """
        if top_hsv:
            self.active_target_top = {
                "lower": np.array(top_hsv["lower"]),
                "upper": np.array(top_hsv["upper"])
            }
        else:
            self.active_target_top = None

        if bottom_hsv:
            self.active_target_bottom = {
                "lower": np.array(bottom_hsv["lower"]),
                "upper": np.array(bottom_hsv["upper"])
            }
        else:
            self.active_target_bottom = None

    def _check_color_match(self, img_crop: np.ndarray, bounds: Dict[str, np.ndarray]) -> bool:
        hsv_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_crop, bounds["lower"], bounds["upper"])
        
        # If more than 20% of the clothing area matches the dynamic color, it's a hit
        match_ratio = cv2.countNonZero(mask) / (img_crop.shape[0] * img_crop.shape[1] + 1e-6)
        return match_ratio > 0.20

    def check_vip_match(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """Slices the person in half and checks dynamic bounds."""
        if not self.active_target_top and not self.active_target_bottom:
            return False

        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if y2 <= y1 or x2 <= x1:
            return False

        person_crop = frame[y1:y2, x1:x2]
        h, w = person_crop.shape[:2]

        top_half = person_crop[0:h//2, :]
        bottom_half = person_crop[h//2:h, :]

        top_match = True
        bottom_match = True

        if self.active_target_top:
            top_match = self._check_color_match(top_half, self.active_target_top)
            
        if self.active_target_bottom:
            bottom_match = self._check_color_match(bottom_half, self.active_target_bottom)

        return top_match and bottom_match

# Singleton
vip_tracker = VIPTracker()