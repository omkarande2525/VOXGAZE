"""
Unit tests for Personalized Gaze-to-Screen Regression Mapper.
Tests fitting, feature sets, mathematical regularization, prediction clamping,
non-finite rejection, and held-out validation metrics calculation.
"""

import math
import numpy as np
import pytest

from backend.app.calibration.personalized_mapper import (
    CalibrationSample,
    FeatureSetType,
    PersonalizedGazeMapper,
)


def _generate_synthetic_samples(n_per_target: int = 20) -> list:
    """Generates synthetic calibration samples with linear yaw/pitch response."""
    samples = []
    targets = {
        "CENTER": (0.50, 0.50, 0.0, 0.0),
        "LEFT": (0.15, 0.50, -15.0, 0.0),
        "RIGHT": (0.85, 0.50, 15.0, 0.0),
        "UP": (0.50, 0.15, 0.0, 10.0),
        "DOWN": (0.50, 0.85, 0.0, -10.0),
    }

    for direction, (tx, ty, base_yaw, base_pitch) in targets.items():
        for _ in range(n_per_target):
            yaw = base_yaw + np.random.normal(0, 0.5)
            pitch = base_pitch + np.random.normal(0, 0.5)
            rh = 0.50 + (yaw / 100.0)
            rv = 0.50 - (pitch / 100.0)
            lh = 0.50 + (yaw / 100.0)
            lv = 0.50 - (pitch / 100.0)
            cn = (pitch / 100.0)

            samples.append(
                CalibrationSample(
                    target_x=tx,
                    target_y=ty,
                    target_direction=direction,
                    yaw_deg=yaw,
                    pitch_deg=pitch,
                    right_eye_h=rh,
                    right_eye_v=rv,
                    left_eye_h=lh,
                    left_eye_v=lv,
                    canthal_normal=cn,
                )
            )
    return samples


def test_mapper_unfitted_returns_abstain():
    mapper = PersonalizedGazeMapper(FeatureSetType.ANGLES_ONLY)
    sample = CalibrationSample(0.5, 0.5, "CENTER", yaw_deg=0.0, pitch_deg=0.0)
    pred = mapper.predict(sample)

    assert pred.is_valid is False
    assert "not fitted" in pred.error_message
    assert pred.screen_x == 0.5
    assert pred.screen_y == 0.5


def test_mapper_fit_and_predict_angles_only():
    train_samples = _generate_synthetic_samples(30)
    mapper = PersonalizedGazeMapper(FeatureSetType.ANGLES_ONLY, alpha=1.0)
    success = mapper.fit(train_samples)

    assert success is True
    assert mapper.is_fitted is True

    # Test center prediction
    center_sample = CalibrationSample(0.5, 0.5, "CENTER", yaw_deg=0.0, pitch_deg=0.0)
    pred_center = mapper.predict(center_sample)
    assert pred_center.is_valid is True
    assert abs(pred_center.screen_x - 0.50) < 0.08
    assert abs(pred_center.screen_y - 0.50) < 0.08

    # Test left prediction
    left_sample = CalibrationSample(0.15, 0.5, "LEFT", yaw_deg=-15.0, pitch_deg=0.0)
    pred_left = mapper.predict(left_sample)
    assert pred_left.is_valid is True
    assert pred_left.screen_x < 0.30
    assert pred_left.predicted_direction == "LEFT"

    # Test right prediction
    right_sample = CalibrationSample(0.85, 0.5, "RIGHT", yaw_deg=15.0, pitch_deg=0.0)
    pred_right = mapper.predict(right_sample)
    assert pred_right.is_valid is True
    assert pred_right.screen_x > 0.70
    assert pred_right.predicted_direction == "RIGHT"


def test_mapper_feature_sets():
    train_samples = _generate_synthetic_samples(20)

    for fs in [
        FeatureSetType.ANGLES_ONLY,
        FeatureSetType.ANGLES_POLY,
        FeatureSetType.GEOMETRY_ONLY,
        FeatureSetType.HYBRID,
    ]:
        m = PersonalizedGazeMapper(fs, alpha=0.5)
        fitted = m.fit(train_samples)
        assert fitted is True
        sample = CalibrationSample(
            0.5, 0.5, "CENTER",
            yaw_deg=0.0, pitch_deg=0.0,
            right_eye_h=0.5, right_eye_v=0.5,
            left_eye_h=0.5, left_eye_v=0.5,
            canthal_normal=0.0,
        )
        pred = m.predict(sample)
        assert pred.is_valid is True
        assert 0.0 <= pred.screen_x <= 1.0
        assert 0.0 <= pred.screen_y <= 1.0


def test_mapper_rejects_non_finite_inputs():
    train_samples = _generate_synthetic_samples(20)
    m = PersonalizedGazeMapper(FeatureSetType.ANGLES_ONLY)
    m.fit(train_samples)

    nan_sample = CalibrationSample(0.5, 0.5, "CENTER", yaw_deg=float("nan"), pitch_deg=0.0)
    pred = m.predict(nan_sample)
    assert pred.is_valid is False
    assert "Missing or non-finite" in pred.error_message


def test_mapper_held_out_evaluation_metrics():
    train_samples = _generate_synthetic_samples(30)
    test_samples = _generate_synthetic_samples(20)

    m = PersonalizedGazeMapper(FeatureSetType.ANGLES_ONLY, alpha=1.0)
    m.fit(train_samples)

    report = m.evaluate_held_out(test_samples)
    assert report["valid_sample_rate"] == 1.0
    assert report["mean_target_error"] < 0.15
    assert report["selection_accuracy"] > 0.85
    assert report["center_false_activation_rate"] < 0.10
    assert report["latency_p50_us"] >= 0.0
