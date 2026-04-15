import argparse
import os
import time
from pathlib import Path

from huggingface_hub import hf_hub_download, model_info, snapshot_download

REPO_ID = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11"
TARGET_DIR = Path("test") / "prithvi_sen1floods11"


def with_retries(fn, attempts=5, delay_seconds=3):
    last_err = None
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # pragma: no cover
            last_err = exc
            print(f"Attempt {i}/{attempts} failed: {exc}")
            if i < attempts:
                sleep_for = delay_seconds * i
                print(f"Retrying in {sleep_for}s...")
                time.sleep(sleep_for)
    raise last_err


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Prithvi Sen1Floods11 model files")
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Download only smoke-test files (inference.py, requirements.txt, example tif).",
    )
    parser.add_argument(
        "--checkpoint-only",
        action="store_true",
        help="Download only the model checkpoint file.",
    )
    args = parser.parse_args()

    # Helps avoid flaky xet transport issues on some Windows networks.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fetching model metadata for: {REPO_ID}")
    try:
        info = with_retries(lambda: model_info(REPO_ID), attempts=5, delay_seconds=2)
        print("Model ID:", info.id)
        print("Library:", info.library_name)
        print("Tags:", info.tags)
    except Exception as exc:
        print(f"Metadata fetch failed, continuing to download step: {exc}")

    allow_patterns = None
    if args.minimal:
        print("Minimal mode enabled. Downloading smoke-test files only.")
        essential_files = [
            "inference.py",
            "requirements.txt",
            "config.yaml",
            "config.json",
            "README.md",
            "examples/India_900498_S2Hand.tif",
        ]
        for file_name in essential_files:
            print(f"Downloading: {file_name}")
            with_retries(
                lambda fn=file_name: hf_hub_download(
                    repo_id=REPO_ID,
                    filename=fn,
                    local_dir=str(TARGET_DIR),
                ),
                attempts=5,
                delay_seconds=2,
            )
        print("Minimal download complete.")
        print("Local path:", TARGET_DIR.resolve())
        return

    if args.checkpoint_only:
        ckpt_name = "Prithvi-EO-V2-300M-TL-Sen1Floods11.pt"
        print(f"Downloading checkpoint: {ckpt_name}")
        with_retries(
            lambda: hf_hub_download(
                repo_id=REPO_ID,
                filename=ckpt_name,
                local_dir=str(TARGET_DIR),
            ),
            attempts=5,
            delay_seconds=4,
        )
        print("Checkpoint download complete.")
        print("Local path:", TARGET_DIR.resolve())
        return

    print(f"Downloading snapshot into: {TARGET_DIR.resolve()}")
    local_dir = with_retries(
        lambda: snapshot_download(
            repo_id=REPO_ID,
            local_dir=str(TARGET_DIR),
            allow_patterns=allow_patterns,
        ),
        attempts=5,
        delay_seconds=3,
    )
    print("Download complete.")
    print("Local path:", local_dir)


if __name__ == "__main__":
    main()
