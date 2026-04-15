import os
import json
import math
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
from backend.twilio_alerts import trigger_emergency_call, send_shelter_sms
from backend.shelter_finder import find_nearby_shelters, format_shelter_sms


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
    """
    Fetches CURRENT weather + NEXT 12-HOUR rainfall forecast.
    The forecast rainfall is used for PREDICTIVE early warning —
    we warn people NOW before the flood hits.
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m"
        f"&hourly=precipitation"
        f"&forecast_hours=12"
    )
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(url, timeout=5)
            if r.status_code == 200:
                data = r.json()
                cur  = data.get("current", {})

                # Sum rainfall predicted over next 12 hours
                hourly_precip = data.get("hourly", {}).get("precipitation", [])
                forecast_12h  = sum(hourly_precip[:12])
                peak_hourly   = max(hourly_precip[:12]) if hourly_precip else 0

                return {
                    "temp":           f"{cur.get('temperature_2m', 0)}°C",
                    "humidity":       f"{cur.get('relative_humidity_2m', 0)}%",
                    "rainfall":       f"{cur.get('precipitation', 0)} mm",
                    "windSpeed":      f"{cur.get('wind_speed_10m', 0)} km/h",
                    "raw_rainfall":   float(cur.get("precipitation", 0)),
                    "forecast_12h":   round(forecast_12h, 1),
                    "peak_hourly":    round(peak_hourly, 1),
                }
        except Exception as e:
            print(f"Weather API error: {e}")
    return {"temp": "28°C", "humidity": "85%", "rainfall": "0 mm",
            "windSpeed": "18 km/h", "raw_rainfall": 0.0,
            "forecast_12h": 0.0, "peak_hourly": 0.0}

# Indian east + west coast approximate points for proximity calculation
_INDIA_COAST = [
    (8.1,77.5),(9.0,78.1),(10.0,79.5),(11.0,79.8),(11.5,79.9),
    (12.0,80.2),(13.1,80.3),(14.0,80.0),(15.0,80.3),(16.0,81.2),
    (17.0,82.3),(18.0,83.5),(19.0,84.7),(20.0,86.4),(21.5,87.5),
    # West coast
    (8.5,76.9),(10.0,76.2),(11.0,75.4),(12.0,74.8),(13.0,74.8),
    (14.0,74.5),(15.0,73.9),(16.0,73.5),(18.0,72.8),(19.0,72.8),
]


async def get_elevation(lat: float, lon: float) -> float | None:
    """Free Open-Elevation API — gives exact elevation for any point."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(
                f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
            )
            if r.status_code == 200:
                results = r.json().get("results", [])
                if results:
                    return float(results[0]["elevation"])
    except Exception:
        pass
    return None


def _coast_dist_deg(lat: float, lon: float) -> float:
    return min(math.sqrt((lat-c[0])**2+(lon-c[1])**2) for c in _INDIA_COAST)



def compute_risk_from_score(score: int) -> dict:
    """Convert a pre-computed score into level + bilingual alerts."""
    score = max(0, min(100, score))
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
            "en": "MODERATE: Flood watch in effect. Stay vigilant and monitor updates.",
            "ta": "மிதமான: வெள்ள கண்காணிப்பு அமலில் உள்ளது. விழிப்புடன் இருங்கள்.",
            "te": "మధ్యస్థంగా: వరద హెచ్చరిక అమలులో ఉంది. జాగ్రత్తగా ఉండండి.",
        }
    else:
        level = "LOW"
        alerts = {
            "en": "LOW RISK: No immediate threat detected. Normal conditions.",
            "ta": "குறைந்த ஆபத்து: உடனடி அச்சுறுத்தல் இல்லை. சாதாரண நிலைமைகள்.",
            "te": "తక్కువ ప్రమాదం: తక్షణ ముప్పు లేదు. సాధారణ పరిస్థితులు.",
        }
    return {"level": level, "alerts": alerts}


def compute_risk(raw_rainfall: float, flood_pct: float) -> dict:
    """
    Satellite flood detection is the PRIMARY risk signal (up to 60 pts).
    Live weather amplifies it — heavy rain adds up to 40 pts on top.

    flood_pct = 0%  + 0mm rain  → score ~0   (truly safe tile, no rain)
    flood_pct = 7%  + 0mm rain  → score ~42  (MODERATE baseline from model)
    flood_pct = 7%  + 20mm rain → score ~82  (HIGH — rain on flood-prone area)
    flood_pct = 30% + 20mm rain → score ~100 (CRITICAL)
    """
    satellite_score = min(60, int(flood_pct * 6))            # Prithvi model baseline
    weather_bonus   = min(40, int(raw_rainfall * 2.0))        # Live weather amplifier
    score = satellite_score + weather_bonus

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
            "en": "MODERATE: Flood watch in effect. Stay vigilant and monitor updates.",
            "ta": "மிதமான: வெள்ள கண்காணிப்பு அமலில் உள்ளது. விழிப்புடன் இருங்கள்.",
            "te": "మధ్యస్థంగా: వరద హెచ్చరిక అమలులో ఉంది. జాగ్రత్తగా ఉండండి.",
        }
    else:
        level = "LOW"
        alerts = {
            "en": "LOW RISK: No immediate threat detected. Normal conditions.",
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
    Location-specific risk for every point:
    - Elevation from Open-Elevation API (lower = more flood-prone)
    - Coastal proximity (closer to Indian coast = higher risk)
    - Nearest Prithvi model tile flood detection
    - Live weather from Open-Meteo
    """
    # Run weather + elevation in parallel
    weather, elevation = await asyncio.gather(
        fetch_live_weather(req.lat, req.lon),
        get_elevation(req.lat, req.lon),
    )

    # --- Elevation risk: 0-30 pts (sea level = 30, 300m+ = 0) ---
    if elevation is not None:
        elev_score = max(0, int(30 * (1 - min(elevation, 300) / 300)))
    else:
        elev_score = 10  # unknown → moderate

    # --- Coastal proximity: 0-20 pts (within 10km = 20, 200km+ = 0) ---
    coast_deg   = _coast_dist_deg(req.lat, req.lon)
    coast_km    = coast_deg * 111
    coast_score = max(0, int(20 * (1 - min(coast_km, 200) / 200)))

    # --- Nearest Prithvi tile: 0-30 pts ---
    nearest = None
    min_dist = float("inf")
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            overlays = json.load(f)
        for o in overlays:
            clat = (o["bounds"][0][0] + o["bounds"][1][0]) / 2
            clon = (o["bounds"][0][1] + o["bounds"][1][1]) / 2
            dist = math.sqrt((clat - req.lat)**2 + (clon - req.lon)**2)
            if dist < min_dist:
                min_dist = dist
                nearest = o
    satellite_score = min(30, int((nearest["flood_pct"] if nearest else 0) * 3))

    # --- Weather bonus: 0-20 pts ---
    weather_score = min(20, int(weather["raw_rainfall"] * 2.0))

    score = elev_score + coast_score + satellite_score + weather_score
    risk  = compute_risk_from_score(score)

    return {
        "status":          "success",
        "composite_score": score,
        "risk_level":      risk["level"],
        "weather":         weather,
        "alerts":          risk["alerts"],
        "nearest_overlay": nearest,
        "breakdown": {
            "elevation_m":    round(elevation, 1) if elevation is not None else None,
            "coast_km":       round(coast_km, 1),
            "elev_score":     elev_score,
            "coast_score":    coast_score,
            "satellite_score": satellite_score,
            "weather_score":  weather_score,
        },
    }


@app.post("/api/predict_live")
async def predict_live(req: PredictRequest):
    """
    Runs actual Prithvi-EO-2.0 model LIVE on the nearest tile (~28s).
    Also fetches elevation + weather for location-specific scoring.
    """
    test_ids  = load_test_ids()
    tile_info = find_nearest_tile(req.lat, req.lon, test_ids)
    if tile_info is None:
        return {"status": "error", "message": "No test tiles found."}

    # Run model + elevation + weather in parallel
    try:
        inference_future = asyncio.to_thread(
            run_prithvi_inference, tile_info["image_id"], tile_info,
        )
        weather_future  = fetch_live_weather(req.lat, req.lon)
        elev_future     = get_elevation(req.lat, req.lon)
        shelter_future  = find_nearby_shelters(req.lat, req.lon, radius_m=5000, limit=3)

        result, weather, elevation, nearby_shelters = await asyncio.gather(
            inference_future, weather_future, elev_future, shelter_future,
        )
    except Exception as e:
        nearby_shelters = []
        return {"status": "error", "message": str(e)}

    # --- Elevation risk: 0-25 pts ---
    if elevation is not None:
        elev_score = max(0, int(25 * (1 - min(elevation, 300) / 300)))
    else:
        elev_score = 8

    # --- Coastal proximity: 0-15 pts ---
    coast_deg   = _coast_dist_deg(req.lat, req.lon)
    coast_km    = coast_deg * 111
    coast_score = max(0, int(15 * (1 - min(coast_km, 200) / 200)))

    # --- Prithvi model: 0-40 pts (PRIMARY) ---
    satellite_score = min(40, int(result["flood_pct"] * 4))

    # --- Predictive Weather: 0-20 pts (uses 12-hour FORECAST) ---
    forecast_12h = weather.get("forecast_12h", 0.0)
    weather_score = min(20, int(forecast_12h * 0.5))  # 40mm in 12h = 20 pts

    score = elev_score + coast_score + satellite_score + weather_score
    risk  = compute_risk_from_score(score)

    # Auto-trigger emergency phone call + shelter WhatsApp if CRITICAL
    call_status = None
    sms_status  = None
    shelters    = nearby_shelters or []
    if score >= 75:
        call_status = trigger_emergency_call(
            lat=req.lat, lon=req.lon, score=score,
            risk_level=risk["level"],
            rainfall=weather["rainfall"],
            flood_pct=result["flood_pct"],
            forecast_48h=forecast_12h,
        )
        # Send SMS with nearest shelter locations
        if shelters:
            sms_body   = format_shelter_sms(score, shelters, forecast_12h)
            sms_status = send_shelter_sms(sms_body)

    return {
        "status":          "success",
        "composite_score": score,
        "risk_level":      risk["level"],
        "weather":         weather,
        "alerts":          risk["alerts"],
        "prediction":      result,
        "call_alert":      call_status,
        "sms_alert":       sms_status,
        "shelters":        shelters,
        "breakdown": {
            "elevation_m":     round(elevation, 1) if elevation is not None else None,
            "coast_km":        round(coast_km, 1),
            "elev_score":      elev_score,
            "coast_score":     coast_score,
            "satellite_score": satellite_score,
            "weather_score":   weather_score,
        },
    }
