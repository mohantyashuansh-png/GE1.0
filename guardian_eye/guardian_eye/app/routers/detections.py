"""
Guardian Eye — Detections Router

GET  /api/detections/persons          — All detected persons with IDs
GET  /api/detections/persons/{pid}    — Single person details
GET  /api/detections/map              — GPS points for map display
GET  /api/detections/landing-zones    — Latest safe landing zones
PATCH /api/detections/persons/{pid}  — Update status (RESCUED etc.)
"""

from fastapi import APIRouter, HTTPException
from typing import Optional

from app.core.state import store
from app.core.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get("/persons")
def get_all_persons(status: Optional[str] = None, limit: int = 100):
    """
    Returns all detected persons.
    Filter by status: ACTIVE | LOST | RESCUED
    """
    persons = store.get_all_persons()
    if status:
        persons = [p for p in persons if p.status == status.upper()]
    persons = persons[-limit:]

    return {
        "total": len(persons),
        "persons": [
            {
                "person_id": p.person_id,
                "track_id": p.track_id,
                "confidence": p.confidence,
                "bbox": p.bbox,
                "gps_lat": p.gps_lat,
                "gps_lon": p.gps_lon,
                "thermal_score": p.thermal_score,
                "first_seen": p.first_seen,
                "last_seen": p.last_seen,
                "frame_count": p.frame_count,
                "status": p.status,
            }
            for p in persons
        ],
    }


@router.get("/persons/{person_id}")
def get_person(person_id: str):
    p = store.persons.get(person_id)
    if not p:
        raise HTTPException(status_code=404, detail=f"Person {person_id} not found")
    return {
        "person_id": p.person_id,
        "track_id": p.track_id,
        "confidence": p.confidence,
        "bbox": p.bbox,
        "gps_lat": p.gps_lat,
        "gps_lon": p.gps_lon,
        "thermal_score": p.thermal_score,
        "first_seen": p.first_seen,
        "last_seen": p.last_seen,
        "frame_count": p.frame_count,
        "status": p.status,
        "thumbnail_path": p.thumbnail_path,
    }


@router.patch("/persons/{person_id}")
def update_person_status(person_id: str, status: str):
    """
    Update person status. Valid: ACTIVE, LOST, RESCUED.
    Called by field commanders after rescue.
    """
    valid = {"ACTIVE", "LOST", "RESCUED"}
    if status.upper() not in valid:
        raise HTTPException(status_code=400, detail=f"Status must be one of {valid}")

    p = store.persons.get(person_id)
    if not p:
        raise HTTPException(status_code=404, detail=f"Person {person_id} not found")

    p.status = status.upper()
    return {"person_id": person_id, "status": p.status}


@router.get("/map")
def get_map_points():
    """
    Returns GPS coordinates of all detected persons + safe LZs.
    Designed for Leaflet.js / map frontend.
    """
    persons = store.get_all_persons()
    lzs = store.get_latest_lzs(limit=20)

    return {
        "persons": [
            {
                "person_id": p.person_id,
                "lat": p.gps_lat,
                "lon": p.gps_lon,
                "status": p.status,
                "confidence": p.confidence,
                "thermal_score": p.thermal_score,
                "last_seen": p.last_seen,
            }
            for p in persons
        ],
        "landing_zones": [
            {
                "lz_id": lz.lz_id,
                "lat": lz.gps_lat,
                "lon": lz.gps_lon,
                "safe": lz.safe,
                "safety_score": lz.safety_score,
                "timestamp": lz.timestamp,
            }
            for lz in lzs
        ],
    }


@router.get("/landing-zones")
def get_landing_zones(limit: int = 10, safe_only: bool = False):
    lzs = store.get_latest_lzs(limit=limit)
    if safe_only:
        lzs = [lz for lz in lzs if lz.safe]
    return {
        "total": len(lzs),
        "landing_zones": [
            {
                "lz_id": lz.lz_id,
                "timestamp": lz.timestamp,
                "center_x": lz.center_x,
                "center_y": lz.center_y,
                "area_px": lz.area_px,
                "safety_score": lz.safety_score,
                "safe": lz.safe,
                "gps_lat": lz.gps_lat,
                "gps_lon": lz.gps_lon,
                "depth_variance": lz.depth_variance,
            }
            for lz in lzs
        ],
    }
