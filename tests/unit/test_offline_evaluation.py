"""
Unit tests for Offline Observation Evaluation and Model Benchmarking.
"""

from pathlib import Path
import pytest

from scripts.evaluate_observation_models import (
    compute_distribution_stats,
    safe_bool,
    safe_float,
    convert_records_to_calibration_samples,
    analyze_session_distributions,
)


def test_safe_conversions():
    assert safe_float("12.34") == 12.34
    assert safe_float("-5.0") == -5.0
    assert safe_float("") is None
    assert safe_float(None) is None
    assert safe_float("nan") is None

    assert safe_bool("true") is True
    assert safe_bool("True") is True
    assert safe_bool("1") is True
    assert safe_bool("false") is False
    assert safe_bool("0") is False
    assert safe_bool(None) is False


def test_compute_distribution_stats_empty():
    stats = compute_distribution_stats([])
    assert stats["count"] == 0
    assert stats["mean"] is None
    assert stats["median"] is None


def test_compute_distribution_stats_values():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    stats = compute_distribution_stats(vals)
    assert stats["count"] == 5
    assert stats["mean"] == 3.0
    assert stats["median"] == 3.0
    assert stats["min"] == 1.0
    assert stats["max"] == 5.0
    assert stats["p50"] == 3.0


def test_convert_records_to_samples():
    records = [
        {
            "target_direction": "CENTER",
            "mobileone_yaw_deg": "-5.2",
            "mobileone_pitch_deg": "-10.1",
            "right_eye_h_ratio": "0.48",
            "right_eye_v_ratio": "0.52",
        },
        {
            "target_direction": "UNKNOWN_TARGET",  # Should be skipped
            "mobileone_yaw_deg": "0.0",
        },
    ]
    samples = convert_records_to_calibration_samples(records)
    assert len(samples) == 1
    assert samples[0].target_direction == "CENTER"
    assert samples[0].target_x == 0.50
    assert samples[0].target_y == 0.50
    assert samples[0].yaw_deg == -5.2
    assert samples[0].pitch_deg == -10.1
