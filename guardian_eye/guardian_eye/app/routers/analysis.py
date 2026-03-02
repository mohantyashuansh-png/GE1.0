"""
Guardian Eye — Analysis Router

POST /api/analyze/video  — Upload a video, run full pipeline, get results + annotated output
POST /api/analyze/frame  — Upload a single image frame, run full pipeline
GET  /api/analyze/jobs   — List processed job results

GET  /api/analyze/hazard — (NEW) VLM Hazard Scanner (Gemini)
GET  /api/analyze/triage — (NEW) VLM Medevac Triage (Gemini)
"""

import cv2
import uuid
import os
import json
import numpy as np
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse

from app.modules.pipeline import process_frame, FrameResult
from app.core.config import settings
from app.core.state import store
from app.core.logger import get_logger

# NEW: Import Gemini SDK for VLM features
import google.generativeai as genai

router = APIRouter()
logger = get_logger(__name__)

# In-memory job registry
_jobs: dict = {}


# ── Video upload & analysis ────────────────────────────────────
@router.post("/video")
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    run_depth: bool = Form(True),
    sample_every_n: int = Form(10),   # process every N frames (CPU friendly)
):
    """
    Upload a drone video file. Runs the full Guardian Eye pipeline.
    Returns job_id immediately; processing runs in background.
    Poll /api/analyze/jobs/{job_id} for results.
    """
    job_id = str(uuid.uuid4())[:8]

    # Save upload
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    ext = os.path.splitext(file.filename)[1] or ".mp4"
    upload_path = os.path.join(settings.UPLOAD_DIR, f"{job_id}{ext}")

    with open(upload_path, "wb") as f:
        content = await file.read()
        f.write(content)

    _jobs[job_id] = {"status": "queued", "progress": 0, "results": []}
    background_tasks.add_task(
        _process_video_job, job_id, upload_path, run_depth, sample_every_n
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "message": f"Processing started. Poll /api/analyze/jobs/{job_id} for results.",
        "file": file.filename,
    }


def _process_video_job(job_id: str, video_path: str, run_depth: bool, sample_every: int):
    """Background task: process video frame by frame."""
    _jobs[job_id]["status"] = "processing"
    results = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["error"] = "Cannot open video file."
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_idx = 0
    processed = 0

    # Output video writer (annotated)
    out_path = os.path.join(settings.OUTPUT_DIR, f"{job_id}_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_every == 0:
                result = process_frame(
                    frame=frame,
                    frame_index=frame_idx,
                    total_frames=total,
                    job_id=job_id,
                    run_depth=run_depth,
                    save_frames=False,
                )

                # Write annotated frame to output video
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(out_path, fourcc, fps / sample_every, (w, h))
                writer.write(_rebuild_annotated(frame, result))

                results.append({
                    "frame_index": result.frame_index,
                    "timestamp": result.timestamp,
                    "person_count": result.person_count,
                    "persons": result.persons,
                    "landing_zones": result.landing_zones,
                    "environment": result.environment,
                    "alerts_fired": result.alerts_fired,
                    "gps_lat": result.gps_lat,
                    "gps_lon": result.gps_lon,
                })
                processed += 1

            frame_idx += 1
            _jobs[job_id]["progress"] = round(frame_idx / total * 100, 1)

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["error"] = str(e)
    finally:
        cap.release()
        if writer:
            writer.release()

    _jobs[job_id]["status"] = "done"
    _jobs[job_id]["results"] = results
    _jobs[job_id]["frames_processed"] = processed
    _jobs[job_id]["output_video"] = f"/outputs/{job_id}_annotated.mp4"
    _jobs[job_id]["summary"] = _build_summary(results)
    logger.info(f"Job {job_id} complete: {processed} frames, {len(store.persons)} persons total")


def _rebuild_annotated(frame, result: FrameResult) -> np.ndarray:
    """Re-run quick annotation for video writer (no depth for speed)."""
    from app.modules.detection import detector, Detection
    from app.modules.environment import annotate_env_frame, analyze_environment
    detections = detector.detect(frame, use_tracking=False)
    pid_map = {d.track_id: f"P-{i+1:04d}" for i, d in enumerate(detections)}
    ann = detector.annotate_frame(frame, detections, pid_map)
    env = analyze_environment(frame)
    return annotate_env_frame(ann, env)


def _build_summary(results: list) -> dict:
    all_persons = list(store.persons.values())
    safe_lzs = [lz for lz in store.landing_zones if lz.safe]
    return {
        "total_persons_detected": len(all_persons),
        "total_frames_analyzed": len(results),
        "safe_landing_zones": len(safe_lzs),
        "total_alerts": len(store.alerts),
        "persons": [
            {
                "person_id": p.person_id,
                "confidence": p.confidence,
                "frame_count": p.frame_count,
                "gps_lat": p.gps_lat,
                "gps_lon": p.gps_lon,
                "thermal_score": p.thermal_score,
                "status": p.status,
                "first_seen": p.first_seen,
                "last_seen": p.last_seen,
            }
            for p in all_persons
        ],
    }


# ── Job status ─────────────────────────────────────────────────
@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/jobs")
def list_jobs():
    return [
        {"job_id": jid, "status": j["status"], "progress": j.get("progress", 0)}
        for jid, j in _jobs.items()
    ]


# ── Single frame analysis ──────────────────────────────────────
@router.post("/frame")
async def analyze_frame(
    file: UploadFile = File(...),
    run_depth: bool = Form(True),
):
    """
    Upload a single image (JPEG/PNG). Returns full analysis instantly.
    """
    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Cannot decode image.")

    result = process_frame(frame, run_depth=run_depth, save_frames=True, job_id="frame_api")

    return {
        "frame_index": result.frame_index,
        "timestamp": result.timestamp,
        "person_count": result.person_count,
        "persons": result.persons,
        "landing_zones": result.landing_zones,
        "environment": result.environment,
        "alerts_fired": result.alerts_fired,
        "gps_lat": result.gps_lat,
        "gps_lon": result.gps_lon,
        "outputs": {
            "annotated": result.annotated_path,
            "thermal": result.thermal_path,
            "depth": result.depth_path,
        },
    }

# ── Tactical VLM Analysis (Gemini Flash) ───────────────────────

def _get_gemini_model():
    """Helper to initialize Gemini securely."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Backend Error: GEMINI_API_KEY environment variable is not set.")
    genai.configure(api_key=api_key)
    # Using 1.5 Flash for the absolute lowest latency
    return genai.GenerativeModel('gemini-1.5-flash')

@router.get("/hazard")
async def get_hazard_analysis():
    """
    On-Demand Hazard Scanner.
    Grabs the very last frame from the live drone feed and analyzes environmental risks.
    """
    # Import inside the function to avoid circular import issues on startup
    from app.routers.stream import _last_annotated 
    
    if _last_annotated is None:
        raise HTTPException(status_code=400, detail="No active drone feed found. Start the stream first.")
        
    model = _get_gemini_model()
    
    # Convert OpenCV numpy array (BGR) to JPEG bytes
    _, buffer = cv2.imencode('.jpg', _last_annotated)
    image_bytes = buffer.tobytes()
    
    prompt = (
        "You are an NDRF tactical advisor. Analyze this disaster scene. "
        "Do not suggest rescue methods. ONLY list hidden environmental hazards "
        "for the rescue team (e.g., submerged powerlines, unstable rubble, toxic smoke direction) "
        "in 3 short bullet points."
    )
    
    try:
        response = model.generate_content([
            prompt, 
            {"mime_type": "image/jpeg", "data": image_bytes}
        ])
        return {"status": "success", "analysis": response.text}
    except Exception as e:
        logger.error(f"Gemini API Error (Hazard): {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/triage")
async def get_triage_analysis():
    """
    On-Demand Medevac Triage.
    Grabs the latest frame and estimates patient trauma for immediate dispatch.
    """
    from app.routers.stream import _last_annotated 
    
    if _last_annotated is None:
        raise HTTPException(status_code=400, detail="No active drone feed found. Start the stream first.")
        
    model = _get_gemini_model()
    
    _, buffer = cv2.imencode('.jpg', _last_annotated)
    image_bytes = buffer.tobytes()
    
    prompt = (
        "You are an AI Combat Medic. Analyze this rescued victim. "
        "Estimate burn surface area, visible trauma, or hypothermia risk. "
        "Generate a 20-word medical dispatch text for the receiving hospital."
    )
    
    try:
        response = model.generate_content([
            prompt, 
            {"mime_type": "image/jpeg", "data": image_bytes}
        ])
        return {"status": "success", "analysis": response.text}
    except Exception as e:
        logger.error(f"Gemini API Error (Triage): {e}")
        raise HTTPException(status_code=500, detail=str(e))