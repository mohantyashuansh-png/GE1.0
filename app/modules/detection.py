"""
Guardian Eye — Person Detection Module
YOLOv8 Pose + ByteTrack for person detection and tracking.
Now includes posture / injury detection and VIP color targeting.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Detection:
    track_id: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    center: Tuple[int, int]
    status: str = "STANDING"


class PersonDetector:
    """
    Wraps YOLOv8 Pose + ByteTrack.
    Lazy-loads model on first use.
    """

    def __init__(self):
        self._model = None
        self._tracker_results = {}

    def _load_model(self):
        if self._model is None:
            from ultralytics import YOLO
            logger.info(f"Loading YOLO Pose model on {settings.DEVICE}")
            self._model = YOLO(settings.YOLO_MODEL, task="pose")
            logger.info("YOLO model loaded onto GPU.")

    def detect(self, frame: np.ndarray, use_tracking: bool = True) -> List[Detection]:
        self._load_model()
        
        # Keep this False so it accepts the standard 32-bit webcam frames
        use_half = False

        if use_tracking:
            results = self._model.track(
                frame,
                persist=True,
                conf=settings.YOLO_CONF,
                iou=settings.YOLO_IOU,
                classes=[settings.PERSON_CLASS_ID],
                verbose=False,
                tracker=settings.TRACKER,
                device=settings.DEVICE,
                half=use_half,
            )
        else:
            results = self._model.predict(
                frame,
                conf=settings.YOLO_CONF,
                iou=settings.YOLO_IOU,
                classes=[settings.PERSON_CLASS_ID],
                verbose=False,
                device=settings.DEVICE,
                half=use_half,
            )

        detections: List[Detection] = []

        for r in results:
            boxes = r.boxes
            keypoints = r.keypoints

            if boxes is None:
                continue

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])

                w = x2 - x1
                h = y2 - y1

                if use_tracking and box.id is not None:
                    track_id = int(box.id[0])
                else:
                    track_id = i

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                status = "STANDING"

                # ── INJURY DETECTION ──
                if w > (h * 1.2):
                    status = "LYING DOWN / INJURED"

                elif keypoints is not None and hasattr(keypoints, "data"):
                    if len(keypoints.data) > i:
                        kpts = keypoints.data[i]
                        if len(kpts) >= 17:
                            nose_y = float(kpts[0][1])
                            ankle_y = max(
                                float(kpts[15][1]),
                                float(kpts[16][1])
                            )
                            if abs(ankle_y - nose_y) < (h * 0.4):
                                status = "LYING DOWN / INJURED"

                detections.append(
                    Detection(
                        track_id=track_id,
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        center=(cx, cy),
                        status=status,
                    )
                )

        return detections

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        person_ids: Dict[int, str],
    ) -> np.ndarray:

        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # 🚀 PROPER ID AND CONFIDENCE SCORE RESTORED
            pid = person_ids.get(det.track_id, f"T-{det.track_id}")
            conf_pct = int(det.confidence * 100)

            # ───── UPDATED COLOR LOGIC ─────
            status_text = det.status.upper()

            if "VIP" in status_text:
                if "INJURED" in status_text or "LYING" in status_text:
                    color = (0, 165, 255)   # Bright Orange (VIP injured)
                else:
                    color = (255, 0, 255)   # Magenta (VIP safe)
            elif "INJURED" in status_text or "LYING" in status_text:
                color = (0, 0, 255)         # Red (Normal injured)
            else:
                color = (0, 255, 80)        # Green (Normal safe)
            # ──────────────────────────────

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # 🚀 FULL LABEL WITH ID, STATUS, AND %
            label = f"{pid} | {det.status} | {conf_pct}%"

            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(
                annotated,
                (x1, y1 - lh - 8),
                (x1 + lw + 6, y1),
                color,
                -1
            )

            cv2.putText(
                annotated,
                label,
                (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

            cv2.circle(annotated, det.center, 4, (255, 100, 0), -1)

        count = len(detections)

        cv2.rectangle(annotated, (0, 0), (260, 36), (0, 0, 0), -1)
        cv2.putText(
            annotated,
            f"PERSONS DETECTED: {count}",
            (6, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 80),
            2,
            cv2.LINE_AA,
        )

        return annotated


detector = PersonDetector()