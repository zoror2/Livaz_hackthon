"""Test Prithvi inference on an 8-channel Sen1Floods11 image.

The 8-channel images have bands: [Blue, Green, Red, NIR, SAR_VV, SAR_VH, DEM, Slope]
Prithvi needs 6 bands: [Blue, Green, Red, NIR_NARROW, SWIR_1, SWIR_2]

Strategy: Select optical bands [0,1,2,3] and duplicate NIR for the missing SWIR slots.
This creates a 6-band input: [Blue, Green, Red, NIR, NIR, NIR]

We pick one image from the test split, run inference, then validate against label.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import rasterio

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET_DIR = Path("dataset/dataset/Sen1Floods11_8Channel")
IMAGE_DIR = DATASET_DIR / "image"
LABEL_DIR = DATASET_DIR / "label"
TEST_SPLIT = DATASET_DIR / "split" / "test.txt"

INFERENCE_PY = Path("test/prithvi_sen1floods11/inference.py")
CONFIG = Path("test/prithvi_sen1floods11/config_local.yaml")
CHECKPOINT = Path("test/prithvi_sen1floods11/Prithvi-EO-V2-300M-TL-Sen1Floods11.pt")
OUTPUT_DIR = Path("test/outputs_8ch_test")
TEMP_DIR = Path("test/temp_6band")

PYTHON = sys.executable


def load_test_ids() -> list[str]:
    """Read image IDs from test split."""
    return [line.strip() for line in TEST_SPLIT.read_text().splitlines() if line.strip()]


def convert_8ch_to_6band(src_path: Path, dst_path: Path) -> None:
    """Convert 8-band image to 6-band by selecting optical + duplicating NIR for SWIR."""
    with rasterio.open(src_path) as src:
        data = src.read()  # (8, 512, 512)
        meta = src.meta.copy()

    # Select: Blue(0), Green(1), Red(2), NIR(3), NIR(3), NIR(3)
    band_indices = [0, 1, 2, 3, 3, 3]
    new_data = data[band_indices]  # (6, 512, 512)

    meta.update(count=6)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_path, "w", **meta) as dst:
        dst.write(new_data)

    print(f"  Converted {src_path.name} -> {dst_path.name}  (8-band -> 6-band)")


def run_inference(image_path: Path) -> int:
    """Run Prithvi inference on a single image."""
    cmd = [
        PYTHON,
        str(INFERENCE_PY),
        "--data_file", str(image_path),
        "--config", str(CONFIG),
        "--checkpoint", str(CHECKPOINT),
        "--output_dir", str(OUTPUT_DIR),
        "--input_indices", "0", "1", "2", "3", "4", "5",  # all 6 bands, straight through
        "--rgb_outputs",
    ]
    print(f"  Running inference...")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  [FAIL] Return code: {proc.returncode}")
        print(f"  STDERR: {proc.stderr[:2000]}")
    else:
        print(f"  [OK] Inference complete")
    return proc.returncode


def validate(pred_path: Path, label_path: Path) -> dict:
    """Compute metrics comparing prediction to ground truth."""
    with rasterio.open(pred_path) as src:
        pred = (src.read(1) > 127).astype(np.int32)
    with rasterio.open(label_path) as src:
        label = src.read(1).astype(np.int32)

    valid = label != -1
    p, l = pred[valid], label[valid]
    total = int(valid.sum())

    tp = int(((p == 1) & (l == 1)).sum())
    tn = int(((p == 0) & (l == 0)).sum())
    fp = int(((p == 1) & (l == 0)).sum())
    fn = int(((p == 0) & (l == 1)).sum())

    acc = (tp + tn) / max(total, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    iou = tp / max(tp + fp + fn, 1)

    return {
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1": f1, "iou_flood": iou,
        "pred_flood_pct": float(p.sum()) / max(total, 1) * 100,
        "true_flood_pct": float(l.sum()) / max(total, 1) * 100,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def main() -> None:
    test_ids = load_test_ids()
    print(f"Test split has {len(test_ids)} images")

    # Pick first test image
    img_id = test_ids[0]
    print(f"\n--- Testing image: {img_id} ---")

    src_image = IMAGE_DIR / f"{img_id}_image.tif"
    label_file = LABEL_DIR / f"{img_id}_label.tif"

    if not src_image.exists():
        raise FileNotFoundError(f"Image not found: {src_image}")
    if not label_file.exists():
        raise FileNotFoundError(f"Label not found: {label_file}")

    # Step 1: Convert 8-band to 6-band
    temp_image = TEMP_DIR / f"{img_id}_6band.tif"
    convert_8ch_to_6band(src_image, temp_image)

    # Step 2: Run Prithvi inference
    rc = run_inference(temp_image)
    if rc != 0:
        print("Inference failed! See errors above.")
        return

    # Step 3: Validate
    pred_file = OUTPUT_DIR / f"pred_{img_id}_6band.tiff"
    if not pred_file.exists():
        print(f"  Prediction file not found: {pred_file}")
        # Check what was actually created
        for f in OUTPUT_DIR.iterdir():
            print(f"  Found: {f.name}")
        return

    metrics = validate(pred_file, label_file)

    print(f"\n{'='*50}")
    print(f"  RESULTS for image {img_id}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    print(f"  IoU Flood : {metrics['iou_flood']:.4f}")
    print(f"  Pred flood: {metrics['pred_flood_pct']:.2f}%")
    print(f"  True flood: {metrics['true_flood_pct']:.2f}%")
    print(f"  TP={metrics['tp']} TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
