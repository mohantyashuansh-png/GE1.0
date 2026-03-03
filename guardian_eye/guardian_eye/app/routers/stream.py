"""
Guardian Eye — Live Stream Router (OPTIMIZED EDGE VERSION)
"""

import cv2
import asyncio
import json
import numpy as np
from typing import AsyncGenerator
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Form
from fastapi.responses import StreamingResponse

from app.modules.pipeline import process_frame
from app.core.config import settings
from app.core.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

# Shared global state
_last_annotated: np.ndarray = None
_last_thermal: np.ndarray = None
_last_depth: np.ndarray = None
_last_result = None
_cap: cv2.VideoCapture = None


def _open_camera() -> cv2.VideoCapture:
    global _cap
    source = int(settings.VIDEO_SOURCE) if settings.VIDEO_SOURCE.isdigit() else settings.VIDEO_SOURCE
    _cap = cv2.VideoCapture(source)
    _cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    _cap.set(cv2.CAP_PROP_FPS, settings.STREAM_FPS)
    return _cap


def _frame_to_jpeg(frame: np.ndarray) -> bytes:
    _, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return encoded.tobytes()


async def _webcam_generator() -> AsyncGenerator[bytes, None]:
    global _last_annotated, _last_thermal, _last_depth, _last_result

    cap = _open_camera()

    if not cap.isOpened():
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "NO CAMERA", (200, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        jpeg = _frame_to_jpeg(placeholder)
        while True:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            await asyncio.sleep(0.1)
        return

    frame_idx = 0

    try:
        while True:
            await asyncio.sleep(0.001)

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))

            # Run pipeline
            result = process_frame(
                frame=frame,
                frame_index=frame_idx,
                run_depth=True,
                job_id="live_stream",
            )

            _last_result = result
            _last_annotated = result.annotated_path
            _last_thermal = result.thermal_path
            _last_depth = result.depth_path

            # Stream annotated by default
            if _last_annotated is not None:
                jpeg = _frame_to_jpeg(_last_annotated)
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

            frame_idx += 1

    finally:
        cap.release()


@router.get("/webcam")
async def webcam_stream():
    return StreamingResponse(
        _webcam_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.get("/thermal")
async def video_feed_thermal():
    """Streams the Inferno Thermal map"""
    async def _gen():
        global _last_thermal
        while True:
            if _last_thermal is not None:
                _, buffer = cv2.imencode('.jpg', _last_thermal)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                       buffer.tobytes() + b'\r\n')
            await asyncio.sleep(0.05)

    return StreamingResponse(
        _gen(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.get("/depth")
async def video_feed_depth():
    """Streams the MiDaS Ghost Depth map"""
    async def _gen():
        global _last_depth
        while True:
            if _last_depth is not None:
                _, buffer = cv2.imencode('.jpg', _last_depth)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                       buffer.tobytes() + b'\r\n')
            await asyncio.sleep(0.05)

    return StreamingResponse(
        _gen(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.post("/source")
async def change_video_source(source: str = Form(...)):
    """Dynamically switch between Webcam, Phone, or Video file"""
    global _cap
    settings.VIDEO_SOURCE = source

    if _cap is not None:
        _cap.release()

    src = int(source) if source.isdigit() else source
    _cap = cv2.VideoCapture(src)
    _cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return {"status": "success", "new_source": source}


# ── WebSocket ─────────────────────────
@router.websocket("/ws")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected")

    global _last_result

    try:
        while True:
            if _last_result is not None:
                payload = {
                    "frame_index": _last_result.frame_index,
                    "timestamp": _last_result.timestamp,
                    "person_count": _last_result.person_count,
                    "persons": _last_result.persons,
                    "environment": _last_result.environment,
                    "alerts": _last_result.alerts_fired,
                    "gps_lat": _last_result.gps_lat,
                    "gps_lon": _last_result.gps_lon,
                    "landing_zones": _last_result.landing_zones,
                }
                await websocket.send_text(json.dumps(payload))

            await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
