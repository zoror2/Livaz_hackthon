"""
Full GPU-accelerated evaluation on all test split images.
Generates:
  - outputs_full_eval/full_results.csv    — per-image metrics
  - outputs_full_eval/graphs/             — all plots
"""

import csv
import sys
import time
import subprocess
from pathlib import Path

import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET_DIR  = Path("dataset/dataset/Sen1Floods11_8Channel")
IMAGE_DIR    = DATASET_DIR / "image"
LABEL_DIR    = DATASET_DIR / "label"
TEST_SPLIT   = DATASET_DIR / "split" / "test.txt"

INFERENCE_PY = Path("test/prithvi_sen1floods11/inference.py")
CONFIG       = Path("test/prithvi_sen1floods11/config_local.yaml")
CHECKPOINT   = Path("test/prithvi_sen1floods11/Prithvi-EO-V2-300M-TL-Sen1Floods11.pt")
OUTPUT_DIR   = Path("test/outputs_full_eval")
TEMP_DIR     = Path("test/temp_6band_full")
GRAPH_DIR    = OUTPUT_DIR / "graphs"

PYTHON       = sys.executable
NUM_IMAGES   = None   # None = run ALL test images, or set e.g. 20

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_test_ids() -> list[str]:
    return [l.strip() for l in TEST_SPLIT.read_text().splitlines() if l.strip()]


def convert_8ch_to_6band(src: Path, dst: Path) -> None:
    with rasterio.open(src) as f:
        data = f.read()
        meta = f.meta.copy()
    new_data = data[[0, 1, 2, 3, 3, 3]]   # duplicate band-4 for NIR/SWIR slots
    meta.update(count=6)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst, "w", **meta) as f:
        f.write(new_data)


def run_inference(image_path: Path, out_dir: Path) -> int:
    cmd = [
        PYTHON, str(INFERENCE_PY),
        "--data_file",      str(image_path),
        "--config",         str(CONFIG),
        "--checkpoint",     str(CHECKPOINT),
        "--output_dir",     str(out_dir),
        "--input_indices",  "0", "1", "2", "3", "4", "5",
        "--rgb_outputs",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode


def validate(pred_path: Path, label_path: Path) -> dict:
    with rasterio.open(pred_path) as f:
        pred = (f.read(1) > 127).astype(np.int32)
    with rasterio.open(label_path) as f:
        label = f.read(1).astype(np.int32)

    valid = label != -1
    p, l  = pred[valid], label[valid]
    total = int(valid.sum())

    tp = int(((p == 1) & (l == 1)).sum())
    tn = int(((p == 0) & (l == 0)).sum())
    fp = int(((p == 1) & (l == 0)).sum())
    fn = int(((p == 0) & (l == 1)).sum())

    acc  = (tp + tn) / max(total, 1)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-8)
    iou  = tp / max(tp + fp + fn, 1)

    return {
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1": f1, "iou_flood": iou,
        "pred_flood_pct": float(p.sum()) / max(total, 1) * 100,
        "true_flood_pct": float(l.sum()) / max(total, 1) * 100,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_graphs(results: list[dict]) -> None:
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    ids    = [r["image_id"] for r in results]
    f1s    = [r["f1"]       for r in results]
    ious   = [r["iou_flood"]for r in results]
    accs   = [r["accuracy"] for r in results]
    precs  = [r["precision"]for r in results]
    recs   = [r["recall"]   for r in results]
    times  = [r["time_s"]   for r in results]
    pred_f = [r["pred_flood_pct"] for r in results]
    true_f = [r["true_flood_pct"] for r in results]

    style = {
        "figure.facecolor": "#0f0f1a",
        "axes.facecolor":   "#1a1a2e",
        "axes.edgecolor":   "#444",
        "axes.labelcolor":  "#ccc",
        "xtick.color":      "#aaa",
        "ytick.color":      "#aaa",
        "text.color":       "#eee",
        "grid.color":       "#333",
        "grid.linestyle":   "--",
        "grid.alpha":       0.4,
    }
    plt.rcParams.update(style)
    plt.rcParams["font.family"] = "DejaVu Sans"

    x = np.arange(len(ids))

    # ── 1. F1 / IoU / Accuracy bar chart ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(14, len(ids)*0.5), 6))
    w = 0.25
    b1 = ax.bar(x - w, f1s,   w, label="F1",       color="#6c63ff", alpha=0.85)
    b2 = ax.bar(x,     ious,  w, label="IoU",      color="#00d2ff", alpha=0.85)
    b3 = ax.bar(x + w, accs,  w, label="Accuracy", color="#43e97b", alpha=0.85)
    ax.axhline(np.mean(f1s),  color="#6c63ff", linewidth=1.2, linestyle="--", alpha=0.6)
    ax.axhline(np.mean(ious), color="#00d2ff", linewidth=1.2, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=90, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-Image Metrics: F1 / IoU / Accuracy", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    ax.grid(axis="y")
    plt.tight_layout()
    fig.savefig(GRAPH_DIR / "1_metrics_bar.png", dpi=150)
    plt.close(fig)

    # ── 2. Precision vs Recall scatter ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(precs, recs, c=f1s, cmap="plasma", s=80, alpha=0.85, edgecolors="none")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("F1 Score", color="#eee")
    cb.ax.yaxis.set_tick_params(color="#aaa")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#aaa")
    ax.set_xlabel("Precision")
    ax.set_ylabel("Recall")
    ax.set_title("Precision vs Recall (coloured by F1)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.plot([0, 1], [0, 1], "w--", alpha=0.2)
    ax.grid()
    plt.tight_layout()
    fig.savefig(GRAPH_DIR / "2_precision_recall_scatter.png", dpi=150)
    plt.close(fig)

    # ── 3. F1 distribution histogram ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    n, bins, patches = ax.hist(f1s, bins=15, color="#6c63ff", edgecolor="#333", alpha=0.85)
    ax.axvline(np.mean(f1s), color="#ff6584", linewidth=2, linestyle="--", label=f"Mean F1 = {np.mean(f1s):.3f}")
    ax.axvline(np.median(f1s), color="#43e97b", linewidth=2, linestyle="--", label=f"Median F1 = {np.median(f1s):.3f}")
    ax.set_xlabel("F1 Score")
    ax.set_ylabel("Count")
    ax.set_title("F1 Score Distribution", fontsize=13, fontweight="bold", pad=12)
    ax.legend()
    ax.grid(axis="y")
    plt.tight_layout()
    fig.savefig(GRAPH_DIR / "3_f1_distribution.png", dpi=150)
    plt.close(fig)

    # ── 4. Predicted vs True flood % ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(true_f, pred_f, c="#00d2ff", s=70, alpha=0.85, edgecolors="none")
    lim = max(max(true_f), max(pred_f)) * 1.1
    ax.plot([0, lim], [0, lim], "w--", alpha=0.4, label="Perfect prediction")
    ax.set_xlabel("True Flood %")
    ax.set_ylabel("Predicted Flood %")
    ax.set_title("Predicted vs True Flood Coverage (%)", fontsize=13, fontweight="bold", pad=12)
    ax.legend()
    ax.grid()
    plt.tight_layout()
    fig.savefig(GRAPH_DIR / "4_flood_pct_scatter.png", dpi=150)
    plt.close(fig)

    # ── 5. Inference time per image ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(14, len(ids)*0.5), 5))
    colors = ["#ff6584" if t > np.mean(times) else "#43e97b" for t in times]
    ax.bar(x, times, color=colors, alpha=0.85)
    ax.axhline(np.mean(times), color="white", linewidth=1.5, linestyle="--",
               label=f"Mean = {np.mean(times):.1f}s")
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=90, fontsize=7)
    ax.set_ylabel("Seconds")
    ax.set_title("Inference Time per Image", fontsize=13, fontweight="bold", pad=12)
    fast = mpatches.Patch(color="#43e97b", label="Below avg (fast)")
    slow = mpatches.Patch(color="#ff6584", label="Above avg (slow)")
    ax.legend(handles=[fast, slow, plt.Line2D([0],[0],color="white",linestyle="--",label=f"Mean={np.mean(times):.1f}s")])
    ax.grid(axis="y")
    plt.tight_layout()
    fig.savefig(GRAPH_DIR / "5_inference_time.png", dpi=150)
    plt.close(fig)

    # ── 6. Summary radar / metrics overview ──────────────────────────────────
    cats = ["Accuracy", "Precision", "Recall", "F1", "IoU"]
    vals = [np.mean(accs), np.mean(precs), np.mean(recs), np.mean(f1s), np.mean(ious)]
    angles = np.linspace(0, 2*np.pi, len(cats), endpoint=False).tolist()
    vals_r  = vals + [vals[0]]
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#1a1a2e")
    ax.plot(angles, vals_r, "o-", linewidth=2, color="#6c63ff")
    ax.fill(angles, vals_r, alpha=0.25, color="#6c63ff")
    ax.set_thetagrids(np.degrees(angles[:-1]), cats, color="#ccc", fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title("Average Performance Radar", fontsize=14, fontweight="bold",
                 pad=20, color="#eee")
    ax.tick_params(colors="#aaa")
    ax.grid(color="#444", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(GRAPH_DIR / "6_radar_chart.png", dpi=150)
    plt.close(fig)

    print(f"\n  [GRAPHS] Saved to: {GRAPH_DIR.resolve()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    test_ids = load_test_ids()
    selected = test_ids if NUM_IMAGES is None else test_ids[:NUM_IMAGES]
    print(f">> Running evaluation on {len(selected)} test images\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for i, img_id in enumerate(selected, 1):
        print(f"[{i:3d}/{len(selected)}] {img_id}", end="  ", flush=True)

        src   = IMAGE_DIR / f"{img_id}_image.tif"
        label = LABEL_DIR / f"{img_id}_label.tif"

        if not src.exists() or not label.exists():
            print("SKIP (file missing)")
            continue

        # Convert 8-ch → 6-band
        tmp = TEMP_DIR / f"{img_id}_6band.tif"
        convert_8ch_to_6band(src, tmp)

        # Inference
        t0 = time.time()
        rc = run_inference(tmp, OUTPUT_DIR)
        elapsed = time.time() - t0

        if rc != 0:
            print(f"FAIL ({elapsed:.1f}s)")
            continue

        pred_file = OUTPUT_DIR / f"pred_{img_id}_6band.tiff"
        if not pred_file.exists():
            print("FAIL (no pred file)")
            continue

        m = validate(pred_file, label)
        m["image_id"] = img_id
        m["time_s"]   = elapsed
        results.append(m)
        print(f"Acc={m['accuracy']:.3f}  F1={m['f1']:.3f}  IoU={m['iou_flood']:.3f}  ({elapsed:.1f}s)")

    if not results:
        print("No results — check your dataset paths.")
        return

    # Summary
    print(f"\n{'='*65}")
    print(f"  FULL EVALUATION — {len(results)} images")
    print(f"{'='*65}")
    for key, label in [
        ("accuracy",  "Avg Accuracy "),
        ("precision", "Avg Precision"),
        ("recall",    "Avg Recall   "),
        ("f1",        "Avg F1 Score "),
        ("iou_flood", "Avg IoU Flood"),
    ]:
        val = np.mean([r[key] for r in results])
        print(f"  {label}: {val:.4f}  ({val*100:.2f}%)")
    print(f"  Avg Time/img : {np.mean([r['time_s'] for r in results]):.1f}s")
    print(f"{'='*65}")

    # Save CSV
    csv_path = OUTPUT_DIR / "full_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image_id", "accuracy", "precision", "recall", "f1",
            "iou_flood", "pred_flood_pct", "true_flood_pct", "time_s",
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  [CSV] Saved to: {csv_path.resolve()}")

    # Generate all graphs
    save_graphs(results)


if __name__ == "__main__":
    main()
