"""
GUARDIAN EYE — AI Missing Person Detection Backend
IAF / Indian Army SAR Operations
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import numpy as np
import os

from app.routers import analysis, stream, detections, alerts, health
from app.core.config import settings
from app.core.logger import get_logger

# Import detector for AI pre-warm
from app.modules.detection import detector

logger = get_logger(__name__)


app = FastAPI(
    title="Guardian Eye — SAR Backend",
    description="AI-powered Missing Person Detection for IAF & Indian Army Disaster Response",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# 🚀 SERVE THE FRONTEND DIRECTLY FROM FASTAPI
@app.get("/", response_class=HTMLResponse)
async def serve_command_deck():
    """
    Serves the Guardian Eye command deck UI directly.
    This bypasses CORS and local file security issues.
    """

    html_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "test_deck.html"
    )

    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# 🚀 FIX FOR 403 / CORS BLOCKS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Static files (processed outputs, annotated videos)
app.mount("/outputs", StaticFiles(directory=settings.OUTPUT_DIR), name="outputs")


# API Routers
app.include_router(health.router,      prefix="/api",            tags=["Health"])
app.include_router(analysis.router,    prefix="/api/analyze",    tags=["Analysis"])
app.include_router(stream.router,      prefix="/api/stream",     tags=["Live Stream"])
app.include_router(detections.router,  prefix="/api/detections", tags=["Detections"])
app.include_router(alerts.router,      prefix="/api/alerts",     tags=["Command Alerts"])


# 🚀 STARTUP EVENT — AI PRE-WARM
@app.on_event("startup")
async def startup_event():

    logger.info("[STARTUP] PRE-WARMING AI ENGINES...")

    # 1️⃣ Load model into memory
    detector._load_model()

    # 2️⃣ ONNX / CUDA warmup
    logger.info("[STARTUP] Compiling ONNX CUDA Graph (This takes 10-15 seconds)...")

    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)

    try:
        detector.detect(dummy_frame)
        logger.info("[STARTUP] AI model warmup completed.")
    except Exception as e:
        logger.warning(f"[STARTUP] Warmup failed: {e}")

    logger.info("[STARTUP] ALL AI ENGINES ARMED AND READY.")
    logger.info(f"Output dir: {settings.OUTPUT_DIR}")
    logger.info(f"Models: YOLOv8={settings.YOLO_MODEL}, MiDaS={settings.MIDAS_MODEL}")


# 🚀 SERVER ENTRY POINT
if __name__ == "__main__":

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )