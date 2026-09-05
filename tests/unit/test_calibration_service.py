"""
Unit tests for calibration service, robust outlier rejection, quality gating, and profile persistence.
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


def make_test_features(timestamp_ms: int, h: float, v: float, ear: float = 0.30, is_valid: bool = True) -> GazeFeatures:
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
        tracking_confidence=0.95,
        is_tracking_valid=is_valid,
    )


def test_target_buffer_warmup_and_collection():
    buf = CalibrationTargetBuffer("CENTER", warmup_sec=0.5, collection_sec=1.5)
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
    buf = CalibrationTargetBuffer("CENTER", warmup_sec=0.0, collection_sec=2.0)
    # Feed values with known median & spread: 0.48, 0.49, 0.50, 0.51, 0.52
    for i, val in enumerate([0.48, 0.49, 0.50, 0.51, 0.52]):
        buf.add_sample(make_test_features(i * 100, val, val))

    stats = buf.compute_stats()
    assert stats.sample_count == 5
    assert stats.median_h == pytest.approx(0.50)
    assert stats.median_v == pytest.approx(0.50)
    assert stats.min_h == pytest.approx(0.48)
    assert stats.max_h == pytest.approx(0.52)
    # MAD of [0.48, 0.49, 0.50, 0.51, 0.52] around 0.50 is median([0.02, 0.01, 0.0, 0.01, 0.02]) = 0.01
    assert stats.mad_h == pytest.approx(0.01)


def test_calibration_service_compiles_valid_profile():
    service = CalibrationService(user_id="test_user")

    # Feed clean synthetic sessions for all 6 targets (30 samples each)
    target_data = {
        "CENTER": (0.50, 0.50, 0.30),
        "LEFT":   (0.25, 0.50, 0.30),
        "RIGHT":  (0.75, 0.50, 0.30),
        "UP":     (0.50, 0.25, 0.30),
        "DOWN":   (0.50, 0.75, 0.30),
        "CLOSED": (0.50, 0.50, 0.10),
    }

    for target, (h, v, ear) in target_data.items():
        # Feed 0ms to 2000ms (30 samples in 33ms steps)
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


def test_calibration_quality_gate_rejects_high_variance():
    service = CalibrationService(user_id="jitter_user")
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
            service.feed_sample(target, make_test_features(ts, 0.50, 0.50))

    with pytest.raises(ValueError, match="Calibration quality validation failed"):
        service.compile_profile()


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
