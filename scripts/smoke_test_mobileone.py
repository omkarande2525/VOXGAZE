"""
VoxGaze Milestone M1 / MobileOne S0 Integration
Isolated Smoke Test: Pretrained ONNX Inference Validation

Loads the downloaded official MobileOne S0 ONNX model, inspects session metadata,
and executes inference on a synthetic test input to verify runtime stability.

CRITICAL INVARIANT:
Synthetic inference results verify runtime tensor execution and decoding ONLY.
They DO NOT represent physical gaze accuracy, biological validity, or user calibration.
"""

from pathlib import Path
import sys
import time

import cv2
import numpy as np

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.perception.mobileone_estimator import MobileOneGazeEstimator

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "mobileone_s0_gaze.onnx"


def run_smoke_test(model_path: Path = DEFAULT_MODEL_PATH) -> bool:
    print("=" * 80)
    print(" VOXGAZE MOBILEONE S0 ONNX INFERENCE SMOKE TEST")
    print("=" * 80)

    model_path = Path(model_path).resolve()
    if not model_path.is_file():
        print(f"[FAIL] Model file not found at: {model_path}")
        print("       Please run: python scripts/download_mobileone.py")
        return False

    print(f"[INFO] Loading model from: {model_path}")
    print(f"       File size: {model_path.stat().st_size:,} bytes")

    try:
        estimator = MobileOneGazeEstimator(model_path=model_path)
    except Exception as exc:
        print(f"[FAIL] Failed to initialize ONNX Runtime session: {exc}")
        return False

    # Display session metadata
    inputs = estimator.session.get_inputs()
    outputs = estimator.session.get_outputs()

    print("[INFO] ONNX Session Metadata:")
    print(f"       Input  : name='{inputs[0].name}', shape={inputs[0].shape}, type={inputs[0].type}")
    for i, out in enumerate(outputs):
        print(f"       Output {i}: name='{out.name}', shape={out.shape}, type={out.type}")

    # Generate synthetic input (256x256 BGR image simulating neutral synthetic facial tone)
    print("\n[INFO] Generating synthetic test crop (256x256x3 BGR)...")
    synthetic_face = np.full((256, 256, 3), fill_value=128, dtype=np.uint8)
    # Add simple geometric gradient to avoid uniform degenerate feature map
    synthetic_face[64:192, 64:192, :] = 160

    # Warmup inference
    _ = estimator.predict(synthetic_face)

    # Timed inference trials
    num_trials = 10
    latencies = []
    print(f"[INFO] Running {num_trials} inference trials on synthetic input...")
    for _ in range(num_trials):
        res = estimator.predict(synthetic_face)
        if not res.is_valid:
            print(f"[FAIL] Inference trial failed: {res.error_message}")
            return False
        latencies.append(res.inference_latency_ms)

    median_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))

    last_res = estimator.predict(synthetic_face)

    print("\n[INFO] Inference Results on Synthetic Input:")
    print(f"       Status              : {'VALID' if last_res.is_valid else 'INVALID'}")
    print(f"       Decoded Yaw (H)     : {last_res.yaw_deg:.2f}° ({last_res.yaw_rad:+.4f} rad)")
    print(f"       Decoded Pitch (V)   : {last_res.pitch_deg:.2f}° ({last_res.pitch_rad:+.4f} rad)")
    print(f"       Preprocessing Time  : {last_res.preprocessing_latency_ms:.2f} ms")
    print(f"       Inference Median p50: {median_lat:.2f} ms")
    print(f"       Inference 95th p95  : {p95_lat:.2f} ms")

    print("\n" + "-" * 80)
    print(" [SCIENTIFIC INTEGRITY DISCLAIMER]")
    print(" This smoke test confirms ONNX Runtime environment compatibility, tensor")
    print(" dimension negotiation, and mathematical output decoding.")
    print(" It DOES NOT evaluate or imply physical gaze accuracy on human subjects.")
    print(" Real-world validation remains unverified until formal camera benchmarking.")
    print("-" * 80)

    print("\n[SUCCESS] MobileOne S0 ONNX isolated inference smoke test passed.")
    return True


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
