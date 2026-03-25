# GUARDIAN EYE — Backend MVP
### AI-Based Missing Person Detection | IAF & Indian Army SAR Operations

---

## Project Structure

```
guardian_eye/
├── app/
│   ├── main.py                  # FastAPI entry point
│   ├── core/
│   │   ├── config.py            # All settings (thresholds, model names, GPS base)
│   │   ├── state.py             # In-memory person/alert/LZ store
│   │   └── logger.py            # Logging setup
│   ├── modules/
│   │   ├── detection.py         # YOLOv8 + ByteTrack person detector
│   │   ├── thermal.py           # Thermal IR simulation
│   │   ├── depth.py             # MiDaS depth + helicopter LZ finder
│   │   ├── environment.py       # Laplacian variance, fog, rain, safety score
│   │   ├── alerts_engine.py     # Command alert generation
│   │   └── pipeline.py          # Orchestrator — runs all modules per frame
│   ├── routers/
│   │   ├── health.py            # /api/health
│   │   ├── analysis.py          # /api/analyze/video  /api/analyze/frame
│   │   ├── stream.py            # /api/stream/webcam  /api/stream/ws
│   │   ├── detections.py        # /api/detections/*
│   │   └── alerts.py            # /api/alerts/*
│   └── utils/
│       └── gps.py               # GPS tagging (dummy + real EXIF)
├── uploads/                     # Incoming video files
├── outputs/                     # Annotated output videos + frames
├── logs/                        # guardian_eye.log
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# CPU-only PyTorch (recommended for laptop)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# All other dependencies
pip install -r requirements.txt
```

### 2. Run the server

```bash
cd guardian_eye
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Open API docs
```
http://localhost:8000/docs
```

---

## API Reference

### Health
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | System status |
| POST | `/api/reset` | Clear all detection state (new mission) |

### Analysis
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/analyze/video` | Upload drone video → full pipeline |
| POST | `/api/analyze/frame` | Upload single image → instant analysis |
| GET | `/api/analyze/jobs/{job_id}` | Poll video job status + results |
| GET | `/api/analyze/jobs` | List all jobs |

**Video upload params (form-data):**
- `file` — video file (mp4/avi/mov)
- `run_depth` — true/false (MiDaS depth, slower)
- `sample_every_n` — int (10 = every 10th frame, fast; 1 = every frame, slow)

### Live Stream
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/stream/webcam` | MJPEG stream (annotated, put in `<img src="">`) |
| GET | `/api/stream/thermal` | MJPEG thermal stream |
| WS | `/api/stream/ws` | WebSocket — JSON detection events per frame |

**WebSocket message format:**
```json
{
  "frame_index": 42,
  "timestamp": "2025-01-15T10:30:00",
  "person_count": 3,
  "persons": [
    {
      "person_id": "P-0001",
      "track_id": 1,
      "confidence": 0.92,
      "bbox": [120, 80, 200, 310],
      "center": [160, 195],
      "gps_lat": 30.3172,
      "gps_lon": 78.0341,
      "first_seen": "...",
      "status": "ACTIVE"
    }
  ],
  "environment": {
    "safety_level": "CAUTION",
    "fog_probability": 0.3,
    "visibility_score": 0.7,
    "overall_safety_score": 0.65,
    "conditions": ["FOG / LOW VISIBILITY"],
    "recommendations": ["Switch to thermal imaging mode."]
  },
  "alerts": [...],
  "landing_zones": [...]
}
```

### Detections
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/detections/persons` | All persons (filter: `?status=ACTIVE`) |
| GET | `/api/detections/persons/{id}` | Single person detail |
| PATCH | `/api/detections/persons/{id}` | Update status (`?status=RESCUED`) |
| GET | `/api/detections/map` | GPS points for Leaflet.js map |
| GET | `/api/detections/landing-zones` | Safe helicopter LZs |

### Alerts
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/alerts` | All alerts (newest first) |
| GET | `/api/alerts/unacked` | Unacknowledged alerts |
| PATCH | `/api/alerts/{id}/acknowledge` | Acknowledge alert |

---

## Configuration

Edit `app/core/config.py` to tune:

```python
YOLO_MODEL = "yolov8n.pt"          # n=fast, m=balanced, l/x=accurate
YOLO_CONF = 0.35                    # detection confidence threshold
MIDAS_MODEL = "MiDaS_small"        # small=fast CPU, DPT_Large=accurate GPU

GPS_BASE_LAT = 30.3165              # Change to actual op zone coordinates
GPS_BASE_LON = 78.0322

LAPLACIAN_BLUR_THRESH = 80.0       # below = fog/blur alert
LZ_SAFE_SCORE_THRESH = 0.65        # landing zone safety cutoff
HIGH_PRIORITY_MIN_PERSONS = 3      # alert when ≥ N persons in frame
```

---

## Kaggle Training Guide

To fine-tune YOLOv8 on aerial/disaster imagery:

### Datasets to use on Kaggle:
1. **VisDrone2023** — `https://kaggle.com/datasets/tiagosii/visdrone`
2. **HERIDAL** — Human detection in Alpine rescue (search Kaggle)
3. **Aerial Person Detection** — `ultralytics/arial-person-detection`

### Kaggle notebook setup:
```python
# Install
!pip install ultralytics -q

from ultralytics import YOLO

# Load pretrained
model = YOLO("yolov8m.pt")

# Train on VisDrone
model.train(
    data="/kaggle/input/visdrone/visdrone.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,          # GPU on Kaggle
    project="guardian_eye",
    name="visdrone_finetune",
    patience=10,
    cos_lr=True,
    mosaic=1.0,
    augment=True,
)

# Export trained weights
# Download: runs/detect/visdrone_finetune/weights/best.pt
```

### Use your trained model:
```python
# In config.py
YOLO_MODEL = "path/to/best.pt"
```

---

## Environmental Analysis — How It Works

The `/api/analyze/frame` and pipeline run these checks on every frame:

| Metric | Method | Threshold |
|--------|--------|-----------|
| **Blur / Fog** | Laplacian variance | < 80 = blurry |
| **Contrast** | RMS standard deviation | < 40 = low visibility |
| **Fog probability** | Low Laplacian + narrow histogram | > 0.5 = fog alert |
| **Rain probability** | Vertical Sobel streaks + noise | > 0.4 = rain alert |
| **Smoke probability** | Low saturation + flat histogram | > 0.5 = smoke alert |
| **Overall safety** | Composite score 0–1 | < 0.3 = ABORT |

Safety levels: `SAFE` → `CAUTION` → `DANGER` → `ABORT`

---

## MiDaS Landing Zone Detection

MiDaS estimates monocular depth from a single frame.

Landing zones are identified as regions where:
- Depth variance is low (flat ground)
- Area is large enough for helicopter (≥ 4000 px²)
- Not on frame edges
- Safety score ≥ 0.65

Each LZ gets GPS coordinates and appears in `/api/detections/landing-zones`.

---

## Frontend Integration Notes (for your team)

### Live video embed:
```html
<img src="http://localhost:8000/api/stream/webcam" />
<img src="http://localhost:8000/api/stream/thermal" />
```

### WebSocket connection:
```javascript
const ws = new WebSocket("ws://localhost:8000/api/stream/ws");
ws.onmessage = (e) => {
  const data = JSON.parse(e.data);
  updateMap(data.persons, data.landing_zones);
  updateAlerts(data.alerts);
  updateHUD(data.environment);
};
```

### Leaflet.js map markers:
```javascript
// GET /api/detections/map
fetch("/api/detections/map")
  .then(r => r.json())
  .then(data => {
    data.persons.forEach(p => {
      L.marker([p.lat, p.lon])
        .bindPopup(`${p.person_id} | ${p.status} | Conf: ${p.confidence}`)
        .addTo(map);
    });
    data.landing_zones.forEach(lz => {
      if (lz.safe) L.circle([lz.lat, lz.lon], {color: 'green', radius: 30}).addTo(map);
    });
  });
```

---

## Developed For
**Problem Statement 7 — AI-Based Missing Person Detection Using Drone Footage**
IAF & Indian Army Disaster Response Operations | Hackathon MVP
