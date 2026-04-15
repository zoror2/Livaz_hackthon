import os
import json
import asyncio
from pathlib import Path
import httpx
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from backend.sentinel_hub import fetch_ndwi_overlay, BBOXES
from backend.run_inference import load_test_ids, find_nearest_tile, run_prithvi_inference


app = FastAPI(title="Advaya Risk Engine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WORKSPACE_DIR  = Path("D:/AdvayaHakcathon")
STATIC_DIR     = WORKSPACE_DIR / "backend" / "static"
MANIFEST_PATH  = STATIC_DIR / "overlays" / "manifest.json"
SENTINEL_DIR   = STATIC_DIR / "sentinel"
SENTINEL_DIR.mkdir(parents=True, exist_ok=True)

# Serve PNGs at /static/...
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class PredictRequest(BaseModel):
    lat: float
    lon: float


async def fetch_live_weather(lat: float, lon: float):
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m"
    )
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(url, timeout=5)
            if r.status_code == 200:
                cur = r.json().get("current", {})
                return {
                    "temp":         f"{cur.get('temperature_2m', 0)}°C",
                    "humidity":     f"{cur.get('relative_humidity_2m', 0)}%",
                    "rainfall":     f"{cur.get('precipitation', 0)} mm",
                    "windSpeed":    f"{cur.get('wind_speed_10m', 0)} km/h",
                    "raw_rainfall": float(cur.get("precipitation", 0)),
                }
        except Exception as e:
            print(f"Weather API error: {e}")
    return {"temp": "28°C", "humidity": "85%", "rainfall": "0 mm",
            "windSpeed": "18 km/h", "raw_rainfall": 0.0}


def compute_risk(raw_rainfall: float, avg_flood_pct: float) -> dict:
    """
    Simple weighted fusion:
      - Weather contributes up to 50 points (20mm = max)
      - Satellite flood % contributes up to 50 points
    """
    weather_score   = min(50, int(raw_rainfall * 2.5))
    satellite_score = min(50, int(avg_flood_pct * 5))
    score = weather_score + satellite_score

    if score >= 75:
        level = "CRITICAL"
        alerts = {
            "en": "CRITICAL: Immediate evacuation required in low-lying coastal areas.",
            "ta": "முக்கியமான: தாழ்வான கடலோர பகுதிகளில் உடனடியாக வெளியேற்றம் தேவை.",
            "te": "క్రిటికల్: తక్కువ ఎత్తైన తీర ప్రాంతాల్లో వెంటనే తరలింపు అవసరం.",
        }
    elif score >= 50:
        level = "HIGH"
        alerts = {
            "en": "HIGH RISK: Severe flooding expected. Move to higher ground immediately.",
            "ta": "அதிக ஆபத்து: கடுமையான வெள்ளம் எதிர்பார்க்கப்படுகிறது. உயரமான இடத்திற்கு செல்லுங்கள்.",
            "te": "అధిక ప్రమాదం: తీవ్రమైన వరద నీరు వస్తుందని అంచనా. ఎత్తైన ప్రదేశానికి వెళ్ళండి.",
        }
    elif score >= 25:
        level = "MODERATE"
        alerts = {
            "en": "MODERATE: Flood watch in effect. Stay vigilant.",
            "ta": "மிதமான: வெள்ள கண்காணிப்பு அமலில் உள்ளது. விழிப்புடன் இருங்கள்.",
            "te": "మధ్యస్థంగా: వరద హెచ్చరిక అమలులో ఉంది. జాగ్రత్తగా ఉండండి.",
        }
    else:
        level = "LOW"
        alerts = {
            "en": "LOW RISK: No immediate threat. Normal conditions.",
            "ta": "குறைந்த ஆபத்து: உடனடி அச்சுறுத்தல் இல்லை. சாதாரண நிலைமைகள்.",
            "te": "తక్కువ ప్రమాదం: తక్షణ ముప్పు లేదు. సాధారణ పరిస్థితులు.",
        }

    return {"score": score, "level": level, "alerts": alerts}


@app.get("/api/health")
def health():
    return {"status": "Advaya Risk Engine alive"}


@app.get("/api/sentinel/status")
def sentinel_status():
    """Check which Sentinel NDWI overlays have been cached."""
    cached = [p.stem for p in SENTINEL_DIR.glob("*.png")]
    return {"cached_regions": cached, "available_regions": list(BBOXES.keys())}


@app.post("/api/sentinel/overlay/{region}")
async def get_sentinel_overlay(region: str, background_tasks: BackgroundTasks):
    """
    Fetch (or return cached) Sentinel-2 NDWI water overlay PNG for a named region.
    First call triggers the Sentinel Hub API (~5-10s). Subsequent calls return instantly.
    """
    if region not in BBOXES:
        return {"error": f"Unknown region. Choose from: {list(BBOXES.keys())}"}

    cache_path = SENTINEL_DIR / f"{region}.png"

    if cache_path.exists():
        return {
            "status": "cached",
            "url":    f"/static/sentinel/{region}.png",
            "region": region,
            "bbox":   BBOXES[region],
        }

    # Fetch from Sentinel Hub
    png_bytes = await fetch_ndwi_overlay(BBOXES[region])
    if png_bytes is None:
        return {"status": "error", "message": "Sentinel Hub API call failed. Check API key or try again."}

    cache_path.write_bytes(png_bytes)
    return {
        "status": "fresh",
        "url":    f"/static/sentinel/{region}.png",
        "region": region,
        "bbox":   BBOXES[region],
    }


@app.get("/api/overlays")
def get_overlays():
    """Returns the full manifest so the map can load all 67 pre-computed overlays."""
    if not MANIFEST_PATH.exists():
        return {"overlays": []}
    with open(MANIFEST_PATH) as f:
        return {"overlays": json.load(f)}


@app.post("/api/predict")
async def predict(req: PredictRequest):
    """
    1. Fetch real-time weather for clicked coordinates.
    2. Use the nearest pre-computed satellite overlay to get flood %.
    3. Fuse both into a composite risk score.
    """
    weather = await fetch_live_weather(req.lat, req.lon)

    # Find nearest overlay to the clicked point using simple Euclidean distance
    nearest = None
    min_dist = float("inf")
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            overlays = json.load(f)
        for o in overlays:
            # bounds = [[south, west], [north, east]]
            center_lat = (o["bounds"][0][0] + o["bounds"][1][0]) / 2
            center_lon = (o["bounds"][0][1] + o["bounds"][1][1]) / 2
            dist = ((center_lat - req.lat) ** 2 + (center_lon - req.lon) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                nearest = o

    avg_flood_pct = nearest["flood_pct"] if nearest else 0.0
    risk = compute_risk(weather["raw_rainfall"], avg_flood_pct)

    return {
        "status":          "success",
        "composite_score": risk["score"],
        "risk_level":      risk["level"],
        "weather":         weather,
        "alerts":          risk["alerts"],
        "nearest_overlay": nearest,
    }


@app.post("/api/predict_live")
async def predict_live(req: PredictRequest):
    """
    Runs actual Prithvi-EO-2.0 model on the nearest Sen1Floods11 test tile.
    Takes ~28s on CPU. Returns real flood prediction PNG + F1/IoU metrics.
    """
    test_ids  = load_test_ids()
    tile_info = find_nearest_tile(req.lat, req.lon, test_ids)
    if tile_info is None:
        return {"status": "error", "message": "No test tiles found."}

    try:
        result = await asyncio.to_thread(
            run_prithvi_inference,
            tile_info["image_id"],
            tile_info,
        )
    except Exception as e:
        return {"status": "error", "message": str(e)}

    weather = await fetch_live_weather(req.lat, req.lon)
    risk    = compute_risk(weather["raw_rainfall"], result["flood_pct"])

    return {
        "status":          "success",
        "composite_score": risk["score"],
        "risk_level":      risk["level"],
        "weather":         weather,
        "alerts":          risk["alerts"],
        "prediction":      result,
    }
