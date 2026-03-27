"""
GUARDIAN EYE — AI Missing Person Detection Backend
IAF / Indian Army SAR Operations
"""

import os
import uvicorn
import numpy as np
import sqlite3
from datetime import datetime
import sqlite3
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# ─────────────────────────────────────────
# 📦 CORE IMPORTS
# ─────────────────────────────────────────
from app.core.config import settings
from app.core.logger import get_logger

# ─────────────────────────────────────────
# 📡 ROUTERS
# ─────────────────────────────────────────
from app.routers import (
    analysis,
    stream,
    detections,
    alerts,
    health,
    history
)

# ─────────────────────────────────────────
# 🤖 AI MODULES
# ─────────────────────────────────────────
from app.modules.detection import detector

# ─────────────────────────────────────────
# 📝 LOGGER
# ─────────────────────────────────────────
logger = get_logger(__name__)

# ─────────────────────────────────────────
# 🚀 FASTAPI APP INIT
# ─────────────────────────────────────────
app = FastAPI(
    title="Guardian Eye — SAR Backend",
    description="AI-powered Missing Person Detection for IAF & Indian Army Disaster Response",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─────────────────────────────────────────
# 🌐 CORS
# ─────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# 📂 PATHS
# ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "db", "missions.db")

# ─────────────────────────────────────────
# 📂 STATIC FILES
# ─────────────────────────────────────────
app.mount(
    "/outputs",
    StaticFiles(directory=settings.OUTPUT_DIR),
    name="outputs"
)

# ─────────────────────────────────────────
# 📡 ROUTER REGISTRATION
# ─────────────────────────────────────────
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(analysis.router, prefix="/api/analyze", tags=["Analysis"])
app.include_router(stream.router, prefix="/api/stream", tags=["Live Stream"])
app.include_router(detections.router, prefix="/api/detections", tags=["Detections"])
app.include_router(alerts.router, prefix="/api/alerts", tags=["Alerts"])
app.include_router(history.router, prefix="/api/history", tags=["History"])

# ─────────────────────────────────────────
# 🖥️ FRONTEND UI
# ─────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_command_deck():
    html_path = os.path.join(BASE_DIR, "..", "test_deck.html")

    try:
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            "<h1>ERROR: test_deck.html not found</h1>",
            status_code=404
        )

# ─────────────────────────────────────────
# 💾 DATABASE FUNCTION (FIXED & CLEAN)
# ─────────────────────────────────────────
DB_PATH = "/Users/khushalika/Documents/RecentHackathon/GE1.0/app/db/missions.db"

def save_detection_to_db(posture, score, sector="Nagpur-Main"):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Naya data yahan se jayega
        cursor.execute(
            "INSERT INTO telemetry (timestamp, posture, score, sector) VALUES (?, ?, ?, ?)",
            (timestamp, posture, score, sector)
        )
        conn.commit()
        conn.close()
        print(f"✅ LIVE DATA SAVED: {posture}") # Terminal mein ye dikhna chahiye
    except Exception as e:
        print(f"❌ DB ERROR: {e}")

# ─────────────────────────────────────────
# ⚡ STARTUP EVENT
# ─────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("[STARTUP] ARMING GUARDIAN EYE SYSTEM...")

    try:
        detector._load_model()

        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        detector.detect(dummy_frame)

        logger.info("AI ENGINE READY")
    except Exception as e:
        logger.error(f"AI ENGINE ERROR: {e}")

    logger.info("SYSTEM READY")

# ─────────────────────────────────────────
# 🚀 ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )