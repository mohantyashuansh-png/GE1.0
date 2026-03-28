import os
from fastapi import APIRouter
import sqlite3

router = APIRouter()

# --- DYNAMIC PATH CALCULATION ---
# Ye code current file ki location se 'db/missions.db' ka rasta nikaal lega
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "db", "missions.db")

@router.get("/logs")
async def get_mission_logs():
    # Terminal mein debug print karega
    print(f"DEBUG: Looking for DB at {DB_PATH}")
    
    if not os.path.exists(DB_PATH):
        # Agar file nahi mili, toh ye error browser mein dikhega
        return {"error": f"Database NOT found at: {DB_PATH}"}
    
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM telemetry ORDER BY timestamp DESC LIMIT 50")
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        return {"error": str(e)}