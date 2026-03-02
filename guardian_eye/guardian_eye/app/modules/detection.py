"""
Guardian Eye — Person Detection Module
YOLOv8 Pose + ByteTrack for person detection and tracking.
Now includes posture / injury detection and FP16 VRAM optimization.
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
    status: str = "STANDING"  # NEW: Tracks if they are lying down


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
            self._model = YOLO(settings.YOLO_MODEL).to(settings.DEVICE)
            logger.info("YOLO model loaded onto GPU.")

    def detect(self, frame: np.ndarray, use_tracking: bool = True) -> List[Detection]:
        """
        Run detection (+ tracking if enabled) on a single BGR frame.
        Returns list of Detection objects for persons only.
        """
        self._load_model()
        
        # Enable FP16 inferencing if on GPU
        use_half = "cuda" in settings.DEVICE

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
                half=use_half,  # <--- VRAM HACK
            )
        else:
            results = self._model.predict(
                frame,
                conf=settings.YOLO_CONF,
                iou=settings.YOLO_IOU,
                classes=[settings.PERSON_CLASS_ID],
                verbose=False,
                device=settings.DEVICE,
                half=use_half,  # <--- VRAM HACK
            )

        detections: List[Detection] = []

        for r in results:
            boxes = r.boxes
            keypoints = r.keypoints  # Extract skeleton keypoints

            if boxes is None:
                continue

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                w, h = (x2 - x1), (y2 - y1)

                # Track ID
                if use_tracking and box.id is not None:
                    track_id = int(box.id[0])
                else:
                    track_id = i

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # ── INJURY DETECTION LOGIC ─────────────────────────
                status = "STANDING"

                # Check 1: Bounding box proportion (wider than tall)
                if w > (h * 1.2):
                    status = "LYING DOWN / INJURED"

                # Check 2: Skeleton-based posture validation
                elif keypoints is not None and hasattr(keypoints, "data"):
                    if len(keypoints.data) > i:
                        kpts = keypoints.data[i]

                        if len(kpts) >= 17:  # COCO keypoints
                            nose_y = float(kpts[0][1])
                            ankle_y = max(
                                float(kpts[15][1]),
                                float(kpts[16][1])
                            )

                            # If head is near same vertical level as feet
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
        """
        Draw bounding boxes, person IDs, posture status, and confidence.
        """
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            pid = person_ids.get(det.track_id, f"T-{det.track_id}")
            conf_pct = int(det.confidence * 100)

            # Color based on status
            if det.status == "LYING DOWN / INJURED":
                color = (0, 0, 255)  # Red alert
            else:
                color = (0, 255, 80)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

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

        # Person count overlay
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


# Singleton
detector = PersonDetector()