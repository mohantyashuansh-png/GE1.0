"""
GUARDIAN EYE — AI Missing Person Detection Backend
IAF / Indian Army SAR Operations
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from app.routers import analysis, stream, detections, alerts, health
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Guardian Eye — SAR Backend",
    description="AI-powered Missing Person Detection for IAF & Indian Army Disaster Response",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files (processed outputs, annotated videos)
app.mount("/outputs", StaticFiles(directory=settings.OUTPUT_DIR), name="outputs")

# Routers
app.include_router(health.router,      prefix="/api",            tags=["Health"])
app.include_router(analysis.router,    prefix="/api/analyze",    tags=["Analysis"])
app.include_router(stream.router,      prefix="/api/stream",     tags=["Live Stream"])
app.include_router(detections.router,  prefix="/api/detections", tags=["Detections"])
app.include_router(alerts.router,      prefix="/api/alerts",     tags=["Command Alerts"])

@app.on_event("startup")
async def startup_event():
    logger.info("Guardian Eye backend starting up...")
    logger.info(f"Output dir: {settings.OUTPUT_DIR}")
    logger.info(f"Models: YOLOv8={settings.YOLO_MODEL}, MiDaS={settings.MIDAS_MODEL}")


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
