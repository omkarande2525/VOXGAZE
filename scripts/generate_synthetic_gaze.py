"""
VoxGaze Synthetic Gaze Data Generator & Stress-Testing Suite
Generates deterministic synthetic gaze feature streams (clean, jitter, drift,
boundary oscillation, blinks, dropouts) to rigorously stress-test the GazeEngine and Hysteresis.
"""

import math
from pathlib import Path
import random
import sys
from typing import Dict, List, Tuple
import numpy as np

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.core.contracts import (
    CalibrationProfile,
    CalibrationStats,
    CalibrationThresholds,
    Direction,
    DirectionThreshold,
    EyeLandmarks,
    GazeFeatures,
    GazeState,
    Point2D,
)
from backend.app.perception.gaze_engine import GazeEngine


def get_mock_calibration_profile() -> CalibrationProfile:
    """Create a standardized, realistic calibration profile for testing."""
    stats = {
        "CENTER": CalibrationStats(sample_count=45, median_h=0.50, median_v=0.50, mad_h=0.015, mad_v=0.015, min_h=0.47, max_h=0.53, min_v=0.47, max_v=0.53),
        "LEFT": CalibrationStats(sample_count=45, median_h=0.30, median_v=0.50, mad_h=0.018, mad_v=0.015, min_h=0.26, max_h=0.34, min_v=0.47, max_v=0.53),
        "RIGHT": CalibrationStats(sample_count=45, median_h=0.70, median_v=0.50, mad_h=0.018, mad_v=0.015, min_h=0.66, max_h=0.74, min_v=0.47, max_v=0.53),
        "UP": CalibrationStats(sample_count=45, median_h=0.50, median_v=0.30, mad_h=0.015, mad_v=0.018, min_h=0.47, max_h=0.53, min_v=0.26, max_v=0.34),
        "DOWN": CalibrationStats(sample_count=45, median_h=0.50, median_v=0.70, mad_h=0.015, mad_v=0.018, min_h=0.47, max_h=0.53, min_v=0.66, max_v=0.74),
        "CLOSED": CalibrationStats(sample_count=45, median_h=0.50, median_v=0.50, mad_h=0.015, mad_v=0.015, min_h=0.47, max_h=0.53, min_v=0.47, max_v=0.53),
    }

    # Center: 0.50. Left: 0.30 (delta 0.20). Right: 0.70 (delta 0.20). Up: 0.30 (delta 0.20). Down: 0.70 (delta 0.20)
    # enter = 0.50 - 0.60*0.20 = 0.38, exit = 0.50 - 0.35*0.20 = 0.43
    thresholds = CalibrationThresholds(
        left=DirectionThreshold(enter=0.38, exit=0.43),
        right=DirectionThreshold(enter=0.62, exit=0.57),
        up=DirectionThreshold(enter=0.38, exit=0.43),
        down=DirectionThreshold(enter=0.62, exit=0.57),
        blink_closed_ear=0.18,
        blink_open_ear=0.28,
    )

    return CalibrationProfile(
        profile_version=1,
        algorithm_version="synthetic-v1",
        user_id="synthetic_user",
        created_at_utc="2026-09-06T00:00:00Z",
        camera_index=0,
        camera_resolution=(1280, 720),
        stats=stats,
        thresholds=thresholds,
        is_valid=True,
    )


def make_sample(
    timestamp_ms: int,
    h: float,
    v: float,
    ear: float = 0.30,
    confidence: float = 0.95,
    is_valid: bool = True,
    no_face: bool = False,
) -> GazeFeatures:
    """Helper to synthesize a single GazeFeatures frame."""
    if no_face:
        return GazeFeatures(
            timestamp_ms=timestamp_ms,
            left_eye=None,
            tracking_confidence=confidence,
            is_tracking_valid=is_valid,
        )

    eye = EyeLandmarks(
        iris_center=Point2D(x=100.0, y=100.0),
        inner_canthus=Point2D(x=80.0, y=100.0),
        outer_canthus=Point2D(x=120.0, y=100.0),
        upper_lid=Point2D(x=100.0, y=90.0),
        lower_lid=Point2D(x=100.0, y=110.0),
        h_ratio=h,
        v_ratio=v,
        ear=ear,
    )
    return GazeFeatures(
        timestamp_ms=timestamp_ms,
        left_eye=eye,
        tracking_confidence=confidence,
        is_tracking_valid=is_valid,
    )


def generate_clean_trajectory(direction: Direction, num_frames: int = 30, fps: int = 30) -> List[GazeFeatures]:
    """Generate clean, stable fixation on target direction."""
    coords = {
        Direction.CENTER: (0.50, 0.50),
        Direction.LEFT: (0.28, 0.50),
        Direction.RIGHT: (0.72, 0.50),
        Direction.UP: (0.50, 0.28),
        Direction.DOWN: (0.50, 0.72),
    }
    h, v = coords[direction]
    dt_ms = int(1000 / fps)
    return [make_sample(i * dt_ms, h, v) for i in range(num_frames)]


def generate_saccade_roundtrip(target: Direction, num_frames: int = 60, fps: int = 30) -> List[GazeFeatures]:
    """Generate CENTER -> TARGET -> CENTER sequence."""
    dt_ms = int(1000 / fps)
    samples = []
    # 20 frames center, 20 frames target, 20 frames center
    c_samples = generate_clean_trajectory(Direction.CENTER, 20, fps)
    t_samples = generate_clean_trajectory(target, 20, fps)
    c2_samples = generate_clean_trajectory(Direction.CENTER, 20, fps)

    all_frames = c_samples + t_samples + c2_samples
    for i, s in enumerate(all_frames):
        samples.append(make_sample(i * dt_ms, s.left_eye.h_ratio, s.left_eye.v_ratio))
    return samples


def generate_noisy_center(num_frames: int = 100, noise_std: float = 0.02, seed: int = 42) -> List[GazeFeatures]:
    """Generate center fixation corrupted by Gaussian tremor / micro-saccades."""
    np.random.seed(seed)
    samples = []
    dt_ms = 33
    for i in range(num_frames):
        h = 0.50 + np.random.normal(0, noise_std)
        v = 0.50 + np.random.normal(0, noise_std)
        samples.append(make_sample(i * dt_ms, float(h), float(v)))
    return samples


def generate_hysteresis_boundary_jitter(num_frames: int = 40, seed: int = 42) -> List[GazeFeatures]:
    """
    Generate signal hovering around the LEFT enter/exit thresholds:
    Crosses LEFT enter (0.37), stays in LEFT state even when oscillating between 0.39 and 0.42 (< 0.43 exit).
    """
    random.seed(seed)
    samples = []
    dt_ms = 33
    # Step 1: Start Center (0.50)
    samples.append(make_sample(0, 0.50, 0.50))
    # Step 2: Cross Left Enter (0.36 < 0.38)
    samples.append(make_sample(dt_ms, 0.36, 0.50))
    # Step 3: Oscillate between 0.39 and 0.42 (all < 0.43 exit). Should strictly remain LEFT!
    for i in range(2, num_frames - 5):
        h = random.uniform(0.39, 0.42)
        samples.append(make_sample(i * dt_ms, h, 0.50))
    # Step 4: Cross Exit back to Center (0.45 > 0.43)
    for i in range(num_frames - 5, num_frames):
        samples.append(make_sample(i * dt_ms, 0.48, 0.50))
    return samples


def generate_dropouts_and_bad_tracking() -> List[GazeFeatures]:
    """Generate sequences with face loss, low confidence, and invalid flags."""
    dt_ms = 33
    return [
        make_sample(0, 0.50, 0.50),                           # Center
        make_sample(dt_ms, 0.50, 0.50, confidence=0.30),     # Low confidence -> ABSTAIN
        make_sample(2 * dt_ms, 0.50, 0.50, is_valid=False),  # Invalid tracking -> ABSTAIN
        make_sample(3 * dt_ms, 0.50, 0.50, no_face=True),    # No face -> NO_FACE
        make_sample(4 * dt_ms, 0.50, 0.50),                  # Recovered -> CENTER
    ]


def generate_blink_sequence() -> List[GazeFeatures]:
    """Generate natural short blink (100ms) vs deliberate closure (500ms)."""
    dt_ms = 33
    samples = []
    t = 0
    # 5 frames eyes open (EAR=0.30)
    for _ in range(5):
        samples.append(make_sample(t, 0.50, 0.50, ear=0.30))
        t += dt_ms
    # 3 frames quick natural blink (EAR=0.10, ~100ms)
    for _ in range(3):
        samples.append(make_sample(t, 0.50, 0.50, ear=0.10))
        t += dt_ms
    # 5 frames eyes open
    for _ in range(5):
        samples.append(make_sample(t, 0.50, 0.50, ear=0.30))
        t += dt_ms
    # 15 frames deliberate closure (EAR=0.10, ~500ms)
    for _ in range(15):
        samples.append(make_sample(t, 0.50, 0.50, ear=0.10))
        t += dt_ms
    # 5 frames reopen
    for _ in range(5):
        samples.append(make_sample(t, 0.50, 0.50, ear=0.30))
        t += dt_ms
    return samples


def run_synthetic_stress_evaluation() -> Dict[str, any]:
    """
    Run full synthetic test suite against GazeEngine and compute quantitative metrics.
    """
    profile = get_mock_calibration_profile()
    engine = GazeEngine(profile=profile)

    results = {
        "clean_accuracy": {},
        "jitter_false_activation_rate": 0.0,
        "hysteresis_boundary_stability": False,
        "dropouts_handled_correctly": False,
        "blink_detection_verified": False,
    }

    # 1. Clean directional accuracy
    for d in [Direction.CENTER, Direction.LEFT, Direction.RIGHT, Direction.UP, Direction.DOWN]:
        engine.current_direction = Direction.CENTER
        stream = generate_clean_trajectory(d, 30)
        states = [engine.process_features(s).direction for s in stream]
        correct = sum(1 for s in states if s == d)
        results["clean_accuracy"][d.value] = correct / len(states)

    # 2. Noisy center tremor (FAR test: 0 unintended directions)
    engine.current_direction = Direction.CENTER
    noisy_stream = generate_noisy_center(100, noise_std=0.02)
    noisy_states = [engine.process_features(s).direction for s in noisy_stream]
    false_activations = sum(1 for s in noisy_states if s != Direction.CENTER)
    results["jitter_false_activation_rate"] = false_activations / len(noisy_states)

    # 3. Hysteresis boundary stability
    engine.current_direction = Direction.CENTER
    hyst_stream = generate_hysteresis_boundary_jitter(40)
    hyst_states = [engine.process_features(s).direction for s in hyst_stream]
    # Verify: frame 1 entered LEFT, frames 2..34 remained LEFT during jitter, final frames returned to CENTER
    stayed_in_left = all(st == Direction.LEFT for st in hyst_states[1:-5])
    returned_to_center = (hyst_states[-1] == Direction.CENTER)
    results["hysteresis_boundary_stability"] = stayed_in_left and returned_to_center

    # 4. Dropout & bad tracking handling
    dropout_stream = generate_dropouts_and_bad_tracking()
    dropout_states = [engine.process_features(s).direction for s in dropout_stream]
    results["dropouts_handled_correctly"] = (
        dropout_states[1] == Direction.ABSTAIN and
        dropout_states[2] == Direction.ABSTAIN and
        dropout_states[3] == Direction.NO_FACE and
        dropout_states[4] == Direction.CENTER
    )

    # 5. Blink detection
    blink_stream = generate_blink_sequence()
    blink_states = [engine.process_features(s).direction for s in blink_stream]
    blink_detected = any(st == Direction.BLINK for st in blink_states)
    results["blink_detection_verified"] = blink_detected

    return results


if __name__ == "__main__":
    print("Running VoxGaze Synthetic Gaze Stress Evaluation...")
    report = run_synthetic_stress_evaluation()
    print("\n--- Synthetic Evaluation Report ---")
    for k, v in report.items():
        print(f"  {k}: {v}")
    all_clean_pass = all(acc == 1.0 for acc in report["clean_accuracy"].values())
    all_robust_pass = (
        report["jitter_false_activation_rate"] == 0.0 and
        report["hysteresis_boundary_stability"] and
        report["dropouts_handled_correctly"] and
        report["blink_detection_verified"]
    )
    if all_clean_pass and all_robust_pass:
        print("\n[PASS] SYNTHETIC REGRESSION: ALL CHECKS PASSED (100% STABILITY)")
    else:
        print("\n[FAIL] SYNTHETIC REGRESSION: FAILED")
        exit(1)
