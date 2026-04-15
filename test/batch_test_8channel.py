"""Batch test Prithvi on multiple 8-channel Sen1Floods11 test images.

Runs inference on N images from the test split, validates each against
ground truth, and produces a summary CSV + average metrics.
"""

import csv
import subprocess
import sys
import time
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
OUTPUT_DIR = Path("test/outputs_batch_test")
TEMP_DIR = Path("test/temp_6band")

PYTHON = sys.executable
NUM_IMAGES = 5  # Test on 5 images from test split


def load_test_ids() -> list[str]:
    return [line.strip() for line in TEST_SPLIT.read_text().splitlines() if line.strip()]


def convert_8ch_to_6band(src_path: Path, dst_path: Path) -> None:
    with rasterio.open(src_path) as src:
        data = src.read()
        meta = src.meta.copy()
    new_data = data[[0, 1, 2, 3, 3, 3]]
    meta.update(count=6)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_path, "w", **meta) as dst:
        dst.write(new_data)


def run_inference(image_path: Path, output_dir: Path) -> int:
    cmd = [
        PYTHON, str(INFERENCE_PY),
        "--data_file", str(image_path),
        "--config", str(CONFIG),
        "--checkpoint", str(CHECKPOINT),
        "--output_dir", str(output_dir),
        "--input_indices", "0", "1", "2", "3", "4", "5",
        "--rgb_outputs",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode


def validate(pred_path: Path, label_path: Path) -> dict:
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
    }


def main() -> None:
    test_ids = load_test_ids()
    selected = test_ids[:NUM_IMAGES]
    print(f"Testing {len(selected)} images from test split\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for i, img_id in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] Image: {img_id}")
        src_image = IMAGE_DIR / f"{img_id}_image.tif"
        label_file = LABEL_DIR / f"{img_id}_label.tif"

        if not src_image.exists() or not label_file.exists():
            print(f"  SKIP - file missing")
            continue

        # Convert
        temp_image = TEMP_DIR / f"{img_id}_6band.tif"
        convert_8ch_to_6band(src_image, temp_image)

        # Inference
        t0 = time.time()
        rc = run_inference(temp_image, OUTPUT_DIR)
        elapsed = time.time() - t0

        if rc != 0:
            print(f"  FAIL (took {elapsed:.1f}s)")
            continue

        # Validate
        pred_file = OUTPUT_DIR / f"pred_{img_id}_6band.tiff"
        if not pred_file.exists():
            print(f"  Prediction file not found")
            continue

        metrics = validate(pred_file, label_file)
        metrics["image_id"] = img_id
        metrics["time_s"] = elapsed
        results.append(metrics)

        print(f"  Acc={metrics['accuracy']:.4f}  F1={metrics['f1']:.4f}  IoU={metrics['iou_flood']:.4f}  ({elapsed:.1f}s)")

    # Summary
    if results:
        print(f"\n{'='*60}")
        print(f"  BATCH RESULTS — {len(results)} images")
        print(f"{'='*60}")

        avg_acc = np.mean([r["accuracy"] for r in results])
        avg_f1 = np.mean([r["f1"] for r in results])
        avg_iou = np.mean([r["iou_flood"] for r in results])
        avg_prec = np.mean([r["precision"] for r in results])
        avg_rec = np.mean([r["recall"] for r in results])
        avg_time = np.mean([r["time_s"] for r in results])

        print(f"  Avg Accuracy  : {avg_acc:.4f} ({avg_acc*100:.2f}%)")
        print(f"  Avg Precision : {avg_prec:.4f}")
        print(f"  Avg Recall    : {avg_rec:.4f}")
        print(f"  Avg F1 Score  : {avg_f1:.4f}")
        print(f"  Avg IoU Flood : {avg_iou:.4f}")
        print(f"  Avg Time/img  : {avg_time:.1f}s")
        print(f"{'='*60}")

        # Save CSV
        csv_path = OUTPUT_DIR / "batch_results.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "image_id", "accuracy", "precision", "recall", "f1",
                "iou_flood", "pred_flood_pct", "true_flood_pct", "time_s",
            ])
            writer.writeheader()
            writer.writerows(results)
        print(f"\n  Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
