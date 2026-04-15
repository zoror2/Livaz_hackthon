import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import rasterio


def compute_flood_ratio(pred_tiff: Path) -> float:
    with rasterio.open(pred_tiff) as src:
        mask = src.read(1)
    # Prediction is written as uint8 class map; flood pixels are high values.
    flood = (mask > 127)
    return float(flood.sum() / max(mask.size, 1))


def run_one(inference_py: Path, image_path: Path, config: Path, checkpoint: Path, output_dir: Path) -> int:
    cmd = [
        sys.executable,
        str(inference_py),
        "--data_file",
        str(image_path),
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--output_dir",
        str(output_dir),
        "--rgb_outputs",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[FAIL] {image_path.name}")
        print(proc.stderr[:2000])
    else:
        print(f"[OK] {image_path.name}")
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Prithvi flood inference on all TIFF files in a folder")
    parser.add_argument("--input_dir", required=True, help="Folder containing .tif/.tiff images")
    parser.add_argument("--output_dir", default="test/prithvi_sen1floods11/outputs_batch", help="Folder for outputs")
    parser.add_argument("--inference_py", default="test/prithvi_sen1floods11/inference.py")
    parser.add_argument("--config", default="test/prithvi_sen1floods11/config_local.yaml")
    parser.add_argument("--checkpoint", default="test/prithvi_sen1floods11/Prithvi-EO-V2-300M-TL-Sen1Floods11.pt")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    inference_py = Path(args.inference_py)
    config = Path(args.config)
    checkpoint = Path(args.checkpoint)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")
    if not inference_py.exists():
        raise FileNotFoundError(f"Inference script not found: {inference_py}")
    if not config.exists():
        raise FileNotFoundError(f"Config not found: {config}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    images = sorted(list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff")))
    if not images:
        raise RuntimeError(f"No TIFF files found in: {input_dir}")

    rows = []
    ok_count = 0
    for img in images:
        rc = run_one(inference_py, img, config, checkpoint, output_dir)
        pred_name = f"pred_{img.stem}.tiff"
        pred_file = output_dir / pred_name

        if rc == 0 and pred_file.exists():
            ratio = compute_flood_ratio(pred_file)
            rows.append({"image": img.name, "pred_file": pred_name, "flood_ratio": ratio})
            ok_count += 1
        else:
            rows.append({"image": img.name, "pred_file": "", "flood_ratio": ""})

    csv_path = output_dir / "flood_ratios.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "pred_file", "flood_ratio"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCompleted: {ok_count}/{len(images)} images")
    print(f"Ratios CSV: {csv_path}")


if __name__ == "__main__":
    main()
