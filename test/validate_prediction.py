"""Validate Prithvi flood prediction against Sen1Floods11 ground-truth label.

Computes: Accuracy, IoU, Precision, Recall, F1, Confusion Matrix, Flood Ratios.
Excludes no-data pixels (label == -1) from all metrics.
"""

from pathlib import Path

import numpy as np
import rasterio


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PRED_PATH = Path("test/prithvi_sen1floods11/outputs/pred_India_900498_S2Hand.tiff")
LABEL_PATH = Path("dataset/dataset/Sen1Floods11_8Channel/label/900498_label.tif")


def load_prediction(path: Path) -> np.ndarray:
    """Load prediction mask. Converts uint8 (0/255) to binary (0/1)."""
    with rasterio.open(path) as src:
        mask = src.read(1)
    return (mask > 127).astype(np.int32)


def load_label(path: Path) -> np.ndarray:
    """Load ground-truth label. Values: -1=nodata, 0=no-water, 1=flood."""
    with rasterio.open(path) as src:
        label = src.read(1)
    return label.astype(np.int32)


def compute_metrics(pred: np.ndarray, label: np.ndarray) -> dict:
    """Compute all classification metrics, ignoring no-data pixels (label == -1)."""
    # Mask out no-data pixels
    valid = label != -1
    pred_valid = pred[valid]
    label_valid = label[valid]

    total = valid.sum()

    # Confusion matrix components
    tp = int(((pred_valid == 1) & (label_valid == 1)).sum())
    tn = int(((pred_valid == 0) & (label_valid == 0)).sum())
    fp = int(((pred_valid == 1) & (label_valid == 0)).sum())
    fn = int(((pred_valid == 0) & (label_valid == 1)).sum())

    # Metrics
    accuracy = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    # IoU (Jaccard) for flood class
    iou_flood = tp / max(tp + fp + fn, 1)

    # IoU for no-water class
    iou_nowater = tn / max(tn + fp + fn, 1)

    # Mean IoU
    miou = (iou_flood + iou_nowater) / 2.0

    # Flood ratios
    pred_flood_ratio = float(pred_valid.sum()) / max(total, 1)
    label_flood_ratio = float(label_valid.sum()) / max(total, 1)

    return {
        "total_pixels": int(total),
        "nodata_pixels": int((~valid).sum()),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "iou_flood": iou_flood,
        "iou_nowater": iou_nowater,
        "mean_iou": miou,
        "pred_flood_ratio": pred_flood_ratio,
        "label_flood_ratio": label_flood_ratio,
    }


def print_report(metrics: dict) -> None:
    print("=" * 60)
    print("  PRITHVI FLOOD SEGMENTATION — VALIDATION REPORT")
    print("=" * 60)

    print(f"\n  Prediction : {PRED_PATH}")
    print(f"  Label      : {LABEL_PATH}")

    print(f"\n--- Pixel Counts ---")
    print(f"  Total valid pixels : {metrics['total_pixels']:,}")
    print(f"  No-data excluded   : {metrics['nodata_pixels']:,}")

    print(f"\n--- Confusion Matrix ---")
    print(f"                    Predicted")
    print(f"                  No-Water  Flood")
    print(f"  Actual No-Water  {metrics['tn']:>7,}  {metrics['fp']:>7,}")
    print(f"  Actual Flood     {metrics['fn']:>7,}  {metrics['tp']:>7,}")

    print(f"\n--- Classification Metrics ---")
    print(f"  Accuracy       : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision      : {metrics['precision']:.4f}")
    print(f"  Recall         : {metrics['recall']:.4f}")
    print(f"  F1 Score       : {metrics['f1_score']:.4f}")

    print(f"\n--- IoU (Jaccard Index) ---")
    print(f"  IoU (Flood)    : {metrics['iou_flood']:.4f}")
    print(f"  IoU (No-Water) : {metrics['iou_nowater']:.4f}")
    print(f"  Mean IoU       : {metrics['mean_iou']:.4f}")

    print(f"\n--- Flood Coverage ---")
    print(f"  Predicted flood ratio : {metrics['pred_flood_ratio']:.4f}  ({metrics['pred_flood_ratio']*100:.2f}%)")
    print(f"  Actual flood ratio    : {metrics['label_flood_ratio']:.4f}  ({metrics['label_flood_ratio']*100:.2f}%)")
    ratio_diff = abs(metrics['pred_flood_ratio'] - metrics['label_flood_ratio'])
    print(f"  Absolute difference   : {ratio_diff:.4f}  ({ratio_diff*100:.2f}%)")

    print("\n" + "=" * 60)

    # Quick quality judgement
    f1 = metrics["f1_score"]
    if f1 >= 0.8:
        verdict = "EXCELLENT — Model performs very well on this image."
    elif f1 >= 0.6:
        verdict = "GOOD — Reasonable performance, usable for demo."
    elif f1 >= 0.4:
        verdict = "FAIR — Some errors, but captures general flood patterns."
    else:
        verdict = "POOR — Significant mismatch, may need investigation."

    print(f"  Verdict: {verdict}")
    print("=" * 60)


def main() -> None:
    if not PRED_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {PRED_PATH}\n"
            "Run inference first to generate it."
        )
    if not LABEL_PATH.exists():
        raise FileNotFoundError(
            f"Label file not found: {LABEL_PATH}\n"
            "Ensure the Sen1Floods11_8Channel dataset is at the expected path."
        )

    pred = load_prediction(PRED_PATH)
    label = load_label(LABEL_PATH)

    print(f"Prediction shape: {pred.shape}, unique: {np.unique(pred)}")
    print(f"Label shape:      {label.shape}, unique: {np.unique(label)}")

    if pred.shape != label.shape:
        raise ValueError(
            f"Shape mismatch: pred={pred.shape}, label={label.shape}"
        )

    metrics = compute_metrics(pred, label)
    print_report(metrics)


if __name__ == "__main__":
    main()
