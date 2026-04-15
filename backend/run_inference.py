"""
run_inference.py
----------------
Runs the actual Prithvi-EO-2.0 model on a given test tile and returns
the flood prediction as both a binary tiff and a coloured PNG overlay.
Used by the FastAPI live inference endpoint.
"""

import subprocess
import json
import csv
import numpy as np
from pathlib import Path
from PIL import Image
import rasterio
from rasterio.warp import transform_bounds
from rasterio.crs import CRS

WORKSPACE  = Path("D:/AdvayaHakcathon")
PYTHON     = WORKSPACE / "venv" / "Scripts" / "python.exe"
INFER_SCR  = WORKSPACE / "test" / "prithvi_sen1floods11" / "inference.py"
CHECKPOINT = WORKSPACE / "test" / "prithvi_sen1floods11" / "Prithvi-EO-V2-300M-TL-Sen1Floods11.pt"
CONFIG     = WORKSPACE / "test" / "prithvi_sen1floods11" / "config_local.yaml"
DATA_DIR   = WORKSPACE / "dataset" / "dataset" / "Sen1Floods11_8Channel" / "image"
LABEL_DIR  = WORKSPACE / "dataset" / "dataset" / "Sen1Floods11_8Channel" / "label"
LIVE_OUT   = WORKSPACE / "backend" / "static" / "live"
LIVE_OUT.mkdir(parents=True, exist_ok=True)

# All 67 test image IDs with their geographic centres (from manifest)
import sys
sys.path.insert(0, str(WORKSPACE))


def load_test_ids() -> list[str]:
    split = WORKSPACE / "dataset" / "dataset" / "Sen1Floods11_8Channel" / "split" / "test.txt"
    if not split.exists():
        return []
    with open(split) as f:
        return [l.strip() for l in f if l.strip()]


def get_tile_bounds(image_id: str) -> dict | None:
    """Return WGS84 bounds and centre of a test tile."""
    tif = DATA_DIR / f"{image_id}_image.tif"
    if not tif.exists():
        return None
    try:
        with rasterio.open(tif) as src:
            b = src.bounds
            wgs84 = CRS.from_epsg(4326)
            if src.crs != wgs84:
                l, bot, r, top = transform_bounds(src.crs, wgs84,
                                                  b.left, b.bottom, b.right, b.top)
            else:
                l, bot, r, top = b.left, b.bottom, b.right, b.top
            return {
                "image_id": image_id,
                "bounds": [[bot, l], [top, r]],   # Leaflet [[S,W],[N,E]]
                "centre": [(bot + top) / 2, (l + r) / 2],
            }
    except Exception:
        return None


def find_nearest_tile(lat: float, lon: float, test_ids: list[str]) -> dict | None:
    """Find the test tile closest to the clicked coordinates."""
    best, best_dist = None, float("inf")
    for tid in test_ids:
        info = get_tile_bounds(tid)
        if info is None:
            continue
        clat, clon = info["centre"]
        dist = (clat - lat) ** 2 + (clon - lon) ** 2
        if dist < best_dist:
            best_dist = dist
            best = info
    return best


def pred_to_png(pred_tif: Path, out_png: Path) -> float:
    """
    Convert a prediction tiff (values 0=no flood, 255=flood) to a
    transparent blue PNG and return the flood percentage.
    """
    with rasterio.open(pred_tif) as src:
        data = src.read(1)

    H, W = data.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    flood_mask = (data == 255)
    rgba[flood_mask] = [30, 120, 255, 180]   # vivid blue, semi-transparent

    Image.fromarray(rgba, mode="RGBA").save(out_png)
    return float(flood_mask.sum()) / float(H * W) * 100


def compute_metrics(pred_tif: Path, image_id: str) -> dict:
    """Compute accuracy, F1, IoU against ground truth label tiff."""
    label_tif = LABEL_DIR / f"{image_id}_label.tif"
    if not label_tif.exists():
        return {"accuracy": None, "f1": None, "iou": None}

    with rasterio.open(pred_tif) as src:
        pred = src.read(1)
    with rasterio.open(label_tif) as src:
        label = src.read(1)

    # Ignore nodata (-1) pixels
    valid = label >= 0
    if valid.sum() == 0:
        return {"accuracy": 0.0, "f1": 0.0, "iou": 0.0}

    p = (pred[valid] == 255).astype(int)
    l = label[valid].astype(int)

    tp = ((p == 1) & (l == 1)).sum()
    fp = ((p == 1) & (l == 0)).sum()
    fn = ((p == 0) & (l == 1)).sum()
    tn = ((p == 0) & (l == 0)).sum()

    acc = (tp + tn) / (tp + fp + fn + tn + 1e-10)
    prec = tp / (tp + fp + 1e-10)
    rec  = tp / (tp + fn + 1e-10)
    f1   = 2 * prec * rec / (prec + rec + 1e-10)
    iou  = tp / (tp + fp + fn + 1e-10)

    return {
        "accuracy":  round(float(acc),  4),
        "precision": round(float(prec), 4),
        "recall":    round(float(rec),  4),
        "f1":        round(float(f1),   4),
        "iou":       round(float(iou),  4),
    }


def run_prithvi_inference(image_id: str, tile_info: dict) -> dict:
    """
    Run the full Prithvi model and return prediction results.
    Returns a dict with all metrics, PNG URL, and bounds.
    """
    input_tif = DATA_DIR / f"{image_id}_image.tif"
    pred_tif  = LIVE_OUT / f"pred_{image_id}.tiff"
    out_png   = LIVE_OUT / f"{image_id}.png"

    # Skip inference if already cached
    if not pred_tif.exists():
        cmd = [
            str(PYTHON), str(INFER_SCR),
            "--data_file",  str(input_tif),
            "--config",     str(CONFIG),
            "--checkpoint", str(CHECKPOINT),
            "--output_dir", str(LIVE_OUT),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"Inference failed:\n{result.stderr[-500:]}")

        # inference.py saves to output_dir/pred_<stem>.tiff
        stem = input_tif.stem  # e.g. "989230_image"
        saved = LIVE_OUT / f"pred_{stem}.tiff"
        if saved.exists() and saved != pred_tif:
            saved.rename(pred_tif)

    if not pred_tif.exists():
        raise FileNotFoundError(f"Prediction tiff not found after inference: {pred_tif}")

    flood_pct = pred_to_png(pred_tif, out_png)
    metrics   = compute_metrics(pred_tif, image_id)

    return {
        "image_id":   image_id,
        "bounds":     tile_info["bounds"],
        "png_url":    f"/static/live/{image_id}.png",
        "flood_pct":  round(flood_pct, 2),
        **metrics,
    }
