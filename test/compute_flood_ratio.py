from pathlib import Path

import numpy as np
import rasterio

PRED_PATH = Path("test/prithvi_sen1floods11/outputs/pred_India_900498_S2Hand.tiff")


def main() -> None:
    if not PRED_PATH.exists():
        raise FileNotFoundError(f"Prediction file not found: {PRED_PATH}")

    with rasterio.open(PRED_PATH) as src:
        mask = src.read(1)

    # In this model output path, class map is saved as uint8 after multiplying by 255.
    # So non-flood is 0 and flood is typically 255.
    flood_pixels = (mask > 127)
    total_pixels = mask.size

    flood_ratio = float(flood_pixels.sum() / max(total_pixels, 1))
    flood_percent = flood_ratio * 100.0

    print(f"Prediction file: {PRED_PATH}")
    print(f"Shape: {mask.shape}")
    print(f"Unique values (sample): {np.unique(mask)[:10]}")
    print(f"Flood ratio: {flood_ratio:.6f}")
    print(f"Flood percent: {flood_percent:.2f}%")


if __name__ == "__main__":
    main()
