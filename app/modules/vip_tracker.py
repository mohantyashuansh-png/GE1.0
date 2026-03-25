import cv2
import numpy as np
from typing import Tuple, Dict, Optional

class VIPTracker:
    def __init__(self):
        self.active_target_top: Optional[Dict[str, np.ndarray]] = None
        self.active_target_bottom: Optional[Dict[str, np.ndarray]] = None

    def set_dynamic_target(self, top_hsv: Optional[Dict], bottom_hsv: Optional[Dict]):
        """Updates the mathematical bounds for the VIP."""
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

    def _check_color_match(self, img_crop: np.ndarray, target_hsv: Dict[str, np.ndarray]) -> bool:
        if img_crop is None or img_crop.size == 0 or target_hsv is None:
            return False
        
        hsv_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_crop, target_hsv["lower"], target_hsv["upper"])
        
        # 🚀 THE STICKY LOCK: Dropped to 5% to survive YOLO box breathing!
        match_ratio = cv2.countNonZero(mask) / (img_crop.shape[0] * img_crop.shape[1] + 1e-6)
        return match_ratio > 0.15

    def check_vip_match(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """Slices the person in half and checks dynamic bounds."""
        if not self.active_target_top and not self.active_target_bottom:
            return False

        x1, y1, x2, y2 = bbox
        
        # Prevent out-of-bounds crashing
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        person_crop = frame[y1:y2, x1:x2]

        if person_crop.size == 0:
            return False

        # Slice the person in half
        mid_y = person_crop.shape[0] // 2
        top_crop = person_crop[:mid_y, :]
        bottom_crop = person_crop[mid_y:, :]

        top_match = True
        if self.active_target_top:
            top_match = self._check_color_match(top_crop, self.active_target_top)

        bottom_match = True
        if self.active_target_bottom:
            bottom_match = self._check_color_match(bottom_crop, self.active_target_bottom)

        return top_match and bottom_match


# Export a single global instance for the pipeline to use
vip_tracker = VIPTracker()