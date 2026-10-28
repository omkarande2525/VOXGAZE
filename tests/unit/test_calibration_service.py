"""
Unit tests for calibration service, robust outlier rejection, quality gating,
EAR-only closed-eye validation, active eye baseline, and profile persistence.
"""

from pathlib import Path
import pytest
import numpy as np

from backend.app.calibration.profile import (
    load_calibration_profile,
    save_calibration_profile,
    validate_thresholds,
)
from backend.app.calibration.service import CalibrationService, CalibrationTargetBuffer
from backend.app.core.contracts import (
    CalibrationThresholds,
    DirectionThreshold,
    EyeLandmarks,
    GazeFeatures,
    Point2D,
)


def make_test_features(
    timestamp_ms: int,
    h: float,
    v: float,
    ear: float = 0.30,
    is_valid: bool = True,
) -> GazeFeatures:
    eye = EyeLandmarks(
        iris_center=Point2D(x=100, y=100),
        inner_canthus=Point2D(x=80, y=100),
        outer_canthus=Point2D(x=120, y=100),
        upper_lid=Point2D(x=100, y=90),
        lower_lid=Point2D(x=100, y=110),
        h_ratio=h,
        v_ratio=v,
        ear=ear,
    )
    return GazeFeatures(
        timestamp_ms=timestamp_ms,
        left_eye=eye,
        right_eye=eye,
        ear_left=ear,
        ear_right=ear,
        tracking_confidence=0.95,
        is_tracking_valid=is_valid,
    )


def test_target_buffer_warmup_and_collection():
    buf = CalibrationTargetBuffer("CENTER", warmup_sec=0.5, collection_sec=1.5, active_eye="right")
    # Feed 0ms to 400ms (warmup)
    for t in range(0, 500, 100):
        accepted = buf.add_sample(make_test_features(t, 0.50, 0.50))
        assert accepted is False

    # Feed 500ms to 2000ms (active collection)
    for t in range(500, 2100, 100):
        accepted = buf.add_sample(make_test_features(t, 0.50, 0.50))
        assert accepted is True

    # After 2000ms (window closed)
    accepted = buf.add_sample(make_test_features(2200, 0.50, 0.50))
    assert accepted is False
    assert buf.is_complete(2200) is True


def test_compute_stats_median_and_mad():
    buf = CalibrationTargetBuffer("CENTER", warmup_sec=0.0, collection_sec=2.0, active_eye="right")
    # Feed values with known median & spread: 0.48, 0.49, 0.50, 0.51, 0.52
    for i, val in enumerate([0.48, 0.49, 0.50, 0.51, 0.52]):
        buf.add_sample(make_test_features(i * 100, val, val, ear=0.30))

    stats = buf.compute_stats()
    assert stats.sample_count == 5
    assert stats.median_h == pytest.approx(0.50)
    assert stats.median_v == pytest.approx(0.50)
    assert stats.min_h == pytest.approx(0.48)
    assert stats.max_h == pytest.approx(0.52)
    assert stats.mad_h == pytest.approx(0.01)
    assert stats.median_ear == pytest.approx(0.30)
    assert stats.mad_ear == pytest.approx(0.0)


def test_calibration_service_compiles_valid_profile():
    service = CalibrationService(user_id="test_user", active_eye="right")

    target_data = {
        "CENTER": (0.50, 0.50, 0.30),
        "LEFT":   (0.25, 0.50, 0.30),
        "RIGHT":  (0.75, 0.50, 0.30),
        "UP":     (0.50, 0.25, 0.30),
        "DOWN":   (0.50, 0.75, 0.30),
        "CLOSED": (0.50, 0.50, 0.10),
    }

    for target, (h, v, ear) in target_data.items():
        for i in range(60):
            ts = i * 33
            service.feed_sample(target, make_test_features(ts, h, v, ear))

    profile = service.compile_profile()
    assert profile.user_id == "test_user"
    assert profile.is_valid is True
    assert profile.thresholds.left.enter < profile.thresholds.left.exit
    assert profile.thresholds.right.enter > profile.thresholds.right.exit
    assert profile.thresholds.up.enter < profile.thresholds.up.exit
    assert profile.thresholds.down.enter > profile.thresholds.down.exit


def test_closed_ear_only_validation():
    """
    Requirement 5: CLOSED calibration must be validated using EAR and sample quality only.
    Do NOT evaluate iris H/V MAD for CLOSED.
    Even with high H/V jitter during eye closure, calibration must succeed.
    """
    service = CalibrationService(user_id="closed_jitter_user", active_eye="right")
    np.random.seed(42)

    # Feed clean targets for CENTER, LEFT, RIGHT, UP, DOWN
    target_clean = {
        "CENTER": (0.50, 0.50, 0.30),
        "LEFT":   (0.25, 0.50, 0.30),
        "RIGHT":  (0.75, 0.50, 0.30),
        "UP":     (0.50, 0.25, 0.30),
        "DOWN":   (0.50, 0.75, 0.30),
    }
    for target, (h, v, ear) in target_clean.items():
        for i in range(60):
            service.feed_sample(target, make_test_features(i * 33, h, v, ear))

    # Feed CLOSED with extreme H and V noise (MAD > 0.15), but clean low EAR
    for i in range(60):
        h_noise = 0.50 + np.random.uniform(-0.30, 0.30)
        v_noise = 0.50 + np.random.uniform(-0.30, 0.30)
        service.feed_sample("CLOSED", make_test_features(i * 33, float(h_noise), float(v_noise), ear=0.08))

    # Must SUCCEED because H/V MAD is ignored for CLOSED
    profile = service.compile_profile()
    assert profile.is_valid is True
    assert profile.stats["CLOSED"].mad_h > 0.08  # High jitter was present
    assert profile.thresholds.blink_closed_ear < profile.thresholds.blink_open_ear


def test_calibration_quality_gate_rejects_high_variance_on_open_targets():
    """Jitter on open fixation targets (CENTER, etc.) must still trigger quality rejection."""
    service = CalibrationService(user_id="jitter_user", active_eye="right")
    np.random.seed(42)

    # Feed CENTER with huge jitter (MAD > 0.08)
    for i in range(60):
        ts = i * 33
        h = 0.50 + np.random.uniform(-0.25, 0.25)
        service.feed_sample("CENTER", make_test_features(ts, float(h), 0.50))

    # Feed others normally
    for target in ["LEFT", "RIGHT", "UP", "DOWN", "CLOSED"]:
        for i in range(60):
            ts = i * 33
            service.feed_sample(target, make_test_features(ts, 0.50, 0.50, ear=0.10 if target == "CLOSED" else 0.30))

    with pytest.raises(ValueError, match="Calibration quality validation failed"):
        service.compile_profile()


def test_calibration_diagnostics_generation():
    """Verify raw calibration statistics and separation metrics calculation."""
    service = CalibrationService(user_id="diag_user", active_eye="right")

    target_data = {
        "CENTER": (0.50, 0.50, 0.30),
        "LEFT":   (0.25, 0.50, 0.30),
        "RIGHT":  (0.75, 0.50, 0.30),
        "UP":     (0.50, 0.25, 0.30),
        "DOWN":   (0.50, 0.75, 0.30),
        "CLOSED": (0.50, 0.50, 0.10),
    }
    for target, (h, v, ear) in target_data.items():
        for i in range(60):
            service.feed_sample(target, make_test_features(i * 33, h, v, ear))

    profile = service.compile_profile()
    diagnostics = service.get_calibration_diagnostics(profile.stats)

    assert "CENTER" in diagnostics
    assert "LEFT" in diagnostics
    assert diagnostics["LEFT"]["separation_from_center"] == pytest.approx(-0.25)
    assert diagnostics["RIGHT"]["separation_from_center"] == pytest.approx(0.25)
    assert diagnostics["CLOSED"]["separation_from_center"] == pytest.approx(0.20)
    assert diagnostics["CENTER"]["samples"] >= 20


def test_profile_serialization_roundtrip(tmp_path: Path):
    thresholds = CalibrationThresholds(
        left=DirectionThreshold(enter=0.38, exit=0.43),
        right=DirectionThreshold(enter=0.62, exit=0.57),
        up=DirectionThreshold(enter=0.38, exit=0.43),
        down=DirectionThreshold(enter=0.62, exit=0.57),
        blink_closed_ear=0.18,
        blink_open_ear=0.28,
    )
    is_valid, msg = validate_thresholds(thresholds)
    assert is_valid is True

    file_path = tmp_path / "test_profile.json"
    from scripts.generate_synthetic_gaze import get_mock_calibration_profile
    profile = get_mock_calibration_profile()

    saved_path = save_calibration_profile(profile, file_path)
    assert saved_path.is_file()

    reloaded = load_calibration_profile(saved_path)
    assert reloaded.user_id == profile.user_id
    assert reloaded.thresholds.left.enter == profile.thresholds.left.enter
