import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import rasterio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

app = FastAPI(title="Coastal Climate Risk API", version="1.0.0")

BASE_DIR = Path(__file__).resolve().parents[1]
PRITHVI_DIR = BASE_DIR / "test" / "prithvi_sen1floods11"
INFERENCE_SCRIPT = PRITHVI_DIR / "inference.py"
CONFIG_PATH = PRITHVI_DIR / "config_local.yaml"
CHECKPOINT_PATH = PRITHVI_DIR / "Prithvi-EO-V2-300M-TL-Sen1Floods11.pt"
DEFAULT_DATA_FILE = PRITHVI_DIR / "examples" / "India_900498_S2Hand.tif"
OUTPUT_ROOT = BASE_DIR / "output" / "api_runs"

RESULTS_DB: Dict[str, Dict[str, Any]] = {}


class PredictRequest(BaseModel):
    district_name: str = Field(..., min_length=2, max_length=100)
    rainfall_30d: List[float]
    windspeed_30d: List[float]
    satellite_path: Optional[str] = Field(
        default=None,
        description="Optional local tif/tiff path. If omitted, a bundled sample image is used.",
    )

    @field_validator("rainfall_30d")
    @classmethod
    def validate_rainfall(cls, v: List[float]) -> List[float]:
        if len(v) != 30:
            raise ValueError("rainfall_30d must contain exactly 30 values")
        arr = np.asarray(v, dtype=np.float32)
        if not np.isfinite(arr).all():
            raise ValueError("rainfall_30d contains invalid numbers")
        return [float(x) for x in arr]

    @field_validator("windspeed_30d")
    @classmethod
    def validate_windspeed(cls, v: List[float]) -> List[float]:
        if len(v) != 30:
            raise ValueError("windspeed_30d must contain exactly 30 values")
        arr = np.asarray(v, dtype=np.float32)
        if not np.isfinite(arr).all():
            raise ValueError("windspeed_30d contains invalid numbers")
        return [float(x) for x in arr]


class PredictResponse(BaseModel):
    district: str
    risk_score: float
    confidence: float
    alerts: Dict[str, str]
    result_id: str


def _model_assets_ready() -> bool:
    return INFERENCE_SCRIPT.exists() and CONFIG_PATH.exists() and CHECKPOINT_PATH.exists()


def _weather_score(rainfall_30d: List[float], windspeed_30d: List[float]) -> float:
    rain = np.asarray(rainfall_30d, dtype=np.float32)
    wind = np.asarray(windspeed_30d, dtype=np.float32)

    rain_mean = float(rain.mean())
    wind_mean = float(wind.mean())
    rain_vol = float(rain.std())

    rain_norm = np.clip(rain_mean / 120.0, 0.0, 1.0)
    wind_norm = np.clip(wind_mean / 40.0, 0.0, 1.0)
    vol_norm = np.clip(rain_vol / 60.0, 0.0, 1.0)

    score = 0.55 * rain_norm + 0.35 * wind_norm + 0.10 * vol_norm
    return float(np.clip(score, 0.0, 1.0))


def _compute_flood_ratio(pred_tiff: Path) -> float:
    with rasterio.open(pred_tiff) as src:
        mask = src.read(1)

    flooded = mask > 127
    return float(flooded.sum() / max(mask.size, 1))


def _run_prithvi(data_file: Path, run_dir: Path) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(INFERENCE_SCRIPT),
        "--data_file",
        str(data_file),
        "--config",
        str(CONFIG_PATH),
        "--checkpoint",
        str(CHECKPOINT_PATH),
        "--output_dir",
        str(run_dir),
        "--rgb_outputs",
    ]

    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("HF_HUB_DISABLE_XET", "1")

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-4000:])

    pred_tiff = run_dir / f"pred_{data_file.stem}.tiff"
    if not pred_tiff.exists():
        raise RuntimeError(f"Prediction output not found: {pred_tiff}")

    flood_ratio = _compute_flood_ratio(pred_tiff)
    return {
        "flood_ratio": flood_ratio,
        "pred_tiff": str(pred_tiff),
        "overlay_tiff": str(run_dir / f"rgb_pred_{data_file.stem}.tiff"),
    }


def _build_alerts(district: str, risk_score: float) -> Dict[str, str]:
    if risk_score >= 0.75:
        english = (
            f"Critical flood risk in {district}. Move to safer ground and follow district emergency updates immediately."
        )
        tamil = (
            f"{district} பகுதியில் கடும் வெள்ள ஆபத்து. பாதுகாப்பான இடத்துக்கு உடனே செல்லவும்; மாவட்ட அவசர அறிவிப்புகளைப் பின்பற்றவும்."
        )
    elif risk_score >= 0.5:
        english = (
            f"High flood warning for {district}. Keep emergency kit ready and avoid low-lying roads."
        )
        tamil = (
            f"{district} பகுதியில் அதிக வெள்ள எச்சரிக்கை. அவசரப் பொருட்களைத் தயார் வைத்துக் கொள்ளவும்; தாழ்வான சாலைகளைத் தவிர்க்கவும்."
        )
    elif risk_score >= 0.2:
        english = (
            f"Moderate flood watch in {district}. Monitor weather updates and prepare local drainage measures."
        )
        tamil = (
            f"{district} பகுதியில் மிதமான வெள்ள கண்காணிப்பு. வானிலை புதுப்பிப்புகளை கவனித்து, உள்ளூர் வடிகால் முன்னெச்சரிக்கை எடுக்கவும்."
        )
    else:
        english = f"Low immediate flood risk in {district}. Continue regular monitoring."
        tamil = f"{district} பகுதியில் உடனடி வெள்ள ஆபத்து குறைவு. வழக்கமான கண்காணிப்பைத் தொடர்ந்து செய்யவும்."

    return {"english": english, "tamil": tamil}


def _fuse_scores(sat_score: float, weather_score: float) -> float:
    return float(np.clip(0.65 * sat_score + 0.35 * weather_score, 0.0, 1.0))


def _confidence(sat_score: float, weather_score: float, used_model: bool) -> float:
    agreement = 1.0 - abs(sat_score - weather_score)
    base = 0.68 if used_model else 0.54
    conf = base + 0.30 * np.clip(agreement, 0.0, 1.0)
    return float(np.clip(conf, 0.0, 0.99))


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_assets_ready": _model_assets_ready(),
        "inference_script": str(INFERENCE_SCRIPT),
        "checkpoint": str(CHECKPOINT_PATH),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    result_id = str(uuid.uuid4())
    started_at = time.time()
    run_dir = OUTPUT_ROOT / result_id
    run_dir.mkdir(parents=True, exist_ok=True)

    weather_score = _weather_score(payload.rainfall_30d, payload.windspeed_30d)
    sat_score = 0.0
    model_status = "fallback_weather_only"
    artifacts: Dict[str, Any] = {}

    if _model_assets_ready():
        data_file = Path(payload.satellite_path) if payload.satellite_path else DEFAULT_DATA_FILE
        if not data_file.exists():
            raise HTTPException(status_code=400, detail=f"Satellite file not found: {data_file}")

        try:
            sat_output = _run_prithvi(data_file=data_file, run_dir=run_dir)
            sat_score = float(sat_output["flood_ratio"])
            artifacts = sat_output
            model_status = "prithvi_success"
        except Exception as exc:
            model_status = f"prithvi_failed: {str(exc)[:180]}"

    risk_score = _fuse_scores(sat_score=sat_score, weather_score=weather_score)
    confidence = _confidence(sat_score=sat_score, weather_score=weather_score, used_model=model_status == "prithvi_success")
    alerts = _build_alerts(payload.district_name, risk_score)

    RESULTS_DB[result_id] = {
        "district": payload.district_name,
        "risk_score": risk_score,
        "confidence": confidence,
        "alerts": alerts,
        "model_status": model_status,
        "satellite_score": sat_score,
        "weather_score": weather_score,
        "artifacts": artifacts,
        "latency_sec": time.time() - started_at,
        "timestamp_unix": int(time.time()),
    }

    return PredictResponse(
        district=payload.district_name,
        risk_score=round(risk_score, 4),
        confidence=round(confidence, 4),
        alerts=alerts,
        result_id=result_id,
    )


@app.get("/results/{result_id}")
def get_result(result_id: str) -> Dict[str, Any]:
    result = RESULTS_DB.get(result_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    return result
