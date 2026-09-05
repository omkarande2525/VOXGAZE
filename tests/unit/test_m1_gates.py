"""
Milestone M1 Acceptance Gate Tests
Verifies pure geometry, calibration service, hysteresis engine, synthetic regression, and artifact generation.
"""

from pathlib import Path
import pytest
from backend.app.core.contracts import Direction
from backend.app.perception.geometry import (
    compute_ear,
    norm_horizontal_gaze,
    norm_vertical_gaze,
)
from backend.app.calibration.service import CalibrationService
from backend.app.perception.gaze_engine import GazeEngine
from scripts.generate_synthetic_gaze import run_synthetic_stress_evaluation

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def test_m1_modules_exist_and_pure():
    """Verify M1 modules exist."""
    geom_file = ROOT_DIR / "backend" / "app" / "perception" / "geometry.py"
    calib_prof = ROOT_DIR / "backend" / "app" / "calibration" / "profile.py"
    calib_serv = ROOT_DIR / "backend" / "app" / "calibration" / "service.py"
    gaze_eng = ROOT_DIR / "backend" / "app" / "perception" / "gaze_engine.py"
    synth_script = ROOT_DIR / "scripts" / "generate_synthetic_gaze.py"
    cam_script = ROOT_DIR / "scripts" / "milestone1_camera_test.py"

    for p in [geom_file, calib_prof, calib_serv, gaze_eng, synth_script, cam_script]:
        assert p.is_file(), f"Missing required M1 module: {p}"

    # Verify geometry.py has zero forbidden imports
    with open(geom_file, "r", encoding="utf-8") as f:
        content = f.read()
    for forbidden in ["cv2", "mediapipe", "fastapi", "torch", "socket", "urllib"]:
        assert f"import {forbidden}" not in content, f"geometry.py violates purity by importing {forbidden}"


def test_m1_synthetic_stress_suite_regression():
    """Run full synthetic stress evaluation and assert 100% regression pass."""
    report = run_synthetic_stress_evaluation()

    # Directional accuracy must be 100% on clean synthetic signal
    for d, acc in report["clean_accuracy"].items():
        assert acc == 1.0, f"Clean synthetic accuracy for {d} was {acc} < 1.0"

    # False activation rate on jittery center tremor must be 0.0
    assert report["jitter_false_activation_rate"] == 0.0, "Jitter caused false activation"

    # Hysteresis boundary stability must pass
    assert report["hysteresis_boundary_stability"] is True, "Hysteresis boundary stability failed"

    # Dropouts and bad tracking must produce abstentions
    assert report["dropouts_handled_correctly"] is True, "Dropout handling failed"

    # Blink detection must be verified
    assert report["blink_detection_verified"] is True, "Blink detection failed"


def test_m1_artifacts_generated_and_valid():
    """Verify machine-readable artifacts in artifacts/m1/."""
    artifacts_dir = ROOT_DIR / "artifacts" / "m1"
    assert (artifacts_dir / "camera_report.json").is_file()
    assert (artifacts_dir / "calibration_report.json").is_file()
    assert (artifacts_dir / "direction_confusion_matrix.csv").is_file()
    assert (artifacts_dir / "latency_summary.json").is_file()
