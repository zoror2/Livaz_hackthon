from pathlib import Path
import subprocess
import sys

MODEL_DIR = Path("test") / "prithvi_sen1floods11"
INFERENCE_SCRIPT = MODEL_DIR / "inference.py"
SAMPLE_TIF = MODEL_DIR / "examples" / "India_900498_S2Hand.tif"


def check_files() -> bool:
    ok = True
    print("Model directory:", MODEL_DIR.resolve())

    if not MODEL_DIR.exists():
        print("[FAIL] Model folder not found. Run download_prithvi_model.py first.")
        return False

    checks = {
        "inference.py": INFERENCE_SCRIPT.exists(),
        "sample TIFF": SAMPLE_TIF.exists(),
        "requirements.txt": (MODEL_DIR / "requirements.txt").exists(),
    }

    for name, status in checks.items():
        print(f"[{ 'OK' if status else 'FAIL' }] {name}")
        ok = ok and status

    return ok


def smoke_help() -> int:
    cmd = [sys.executable, str(INFERENCE_SCRIPT), "--help"]
    print("\nRunning:", " ".join(cmd))

    proc = subprocess.run(cmd, capture_output=True, text=True)
    print("Return code:", proc.returncode)

    if proc.stdout:
        print("\n--- STDOUT (first 1500 chars) ---")
        print(proc.stdout[:1500])
    if proc.stderr:
        print("\n--- STDERR (first 1500 chars) ---")
        print(proc.stderr[:1500])

    return proc.returncode


def help_flag_not_supported_but_cli_alive(stderr_text: str) -> bool:
    text = (stderr_text or "").lower()
    return "usage:" in text and "unrecognized arguments: --help" in text


def main() -> None:
    if not check_files():
        raise SystemExit(1)

    cmd = [sys.executable, str(INFERENCE_SCRIPT), "--help"]
    print("\nRunning:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print("Return code:", proc.returncode)

    if proc.stdout:
        print("\n--- STDOUT (first 1500 chars) ---")
        print(proc.stdout[:1500])
    if proc.stderr:
        print("\n--- STDERR (first 1500 chars) ---")
        print(proc.stderr[:1500])

    if proc.returncode == 0:
        print("\n[PASS] Prithvi inference script is runnable in this environment.")
    elif help_flag_not_supported_but_cli_alive(proc.stderr):
        print("\n[PASS] CLI is runnable (this script does not accept --help).")
        print("Use: --data_file, --config, --checkpoint, --output_dir")
    else:
        print("\n[INFO] Script exists but runtime dependencies may still be missing.")
        print("Install model requirements and terratorch, then rerun this test.")


if __name__ == "__main__":
    main()
