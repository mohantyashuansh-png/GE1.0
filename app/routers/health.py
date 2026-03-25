from fastapi import APIRouter
from datetime import datetime
from app.core.state import store

router = APIRouter()


@router.get("/health")
def health_check():
    return {
        "status": "operational",
        "service": "Guardian Eye SAR Backend",
        "timestamp": datetime.utcnow().isoformat(),
        "total_persons_logged": len(store.persons),
        "total_alerts": len(store.alerts),
        "total_landing_zones": len(store.landing_zones),
    }


@router.post("/reset")
def reset_state():
    """Reset all detection state (use between missions)."""
    store.reset()
    return {"status": "reset", "message": "State cleared for new mission."}
