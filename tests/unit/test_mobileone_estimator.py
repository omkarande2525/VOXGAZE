"""
Unit tests for MobileOne S0 Gaze Estimator (Phase 02).
Tests preprocessing, numerical normalization, logit decoding, boundary conditions,
shape validation, non-finite rejection, and mock ONNX inference with zero webcam dependency.
"""

import math
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from backend.app.perception.mobileone_estimator import (
    MobileOneGazeEstimator,
    MobileOneOutput,
)


class MockNode:
    def __init__(self, name: str, shape: list):
        self.name = name
        self.shape = shape


def create_mock_session(
    input_name: str = "input",
    input_shape: list = [1, 3, 448, 448],
    output_names: list = ["yaw", "pitch"],
    output_shapes: list = [[1, 90], [1, 90]],
    mock_run_return: tuple = None,
):
    session = MagicMock()
    session.get_inputs.return_value = [MockNode(input_name, input_shape)]
    session.get_outputs.return_value = [
        MockNode(output_names[i], output_shapes[i]) for i in range(len(output_names))
    ]
    if mock_run_return is not None:
        session.run.return_value = mock_run_return
    return session


# ---------------------------------------------------------------------------
# Preprocessing Tests
# ---------------------------------------------------------------------------

def test_preprocessing_shape_and_dtype():
    session = create_mock_session()
    estimator = MobileOneGazeEstimator(session=session)

    # Arbitrary non-square crop
    raw_crop = np.zeros((200, 150, 3), dtype=np.uint8)
    tensor = estimator.preprocess(raw_crop)

    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (1, 3, 448, 448)
    assert tensor.dtype == np.float32


def test_preprocessing_rgb_conversion_and_standardization():
    session = create_mock_session()
    estimator = MobileOneGazeEstimator(session=session)

    # Pure Blue in BGR is (255, 0, 0) -> in RGB it should be (0, 0, 255)
    pure_blue_bgr = np.zeros((64, 64, 3), dtype=np.uint8)
    pure_blue_bgr[:, :, 0] = 255  # Blue channel

    tensor = estimator.preprocess(pure_blue_bgr)

    # In RGB:
    # R = 0.0 -> normalized = (0.0 - 0.485) / 0.229 = -2.1179
    # G = 0.0 -> normalized = (0.0 - 0.456) / 0.224 = -2.0357
    # B = 1.0 -> normalized = (1.0 - 0.406) / 0.225 = +2.6400
    expected_r = (0.0 - 0.485) / 0.229
    expected_g = (0.0 - 0.456) / 0.224
    expected_b = (1.0 - 0.406) / 0.225

    np.testing.assert_allclose(tensor[0, 0, 0, 0], expected_r, rtol=1e-4)
    np.testing.assert_allclose(tensor[0, 1, 0, 0], expected_g, rtol=1e-4)
    np.testing.assert_allclose(tensor[0, 2, 0, 0], expected_b, rtol=1e-4)


def test_preprocessing_rejects_invalid_inputs():
    session = create_mock_session()
    estimator = MobileOneGazeEstimator(session=session)

    with pytest.raises(ValueError, match="non-None"):
        estimator.preprocess(None)

    with pytest.raises(ValueError, match="non-empty 3-channel"):
        estimator.preprocess(np.array([], dtype=np.uint8))

    with pytest.raises(ValueError, match="non-empty 3-channel"):
        estimator.preprocess(np.zeros((100, 100), dtype=np.uint8))  # 2D grayscale

    with pytest.raises(ValueError, match="non-empty 3-channel"):
        estimator.preprocess(np.zeros((100, 100, 4), dtype=np.uint8))  # 4-channel BGRA


# ---------------------------------------------------------------------------
# Decoding Tests
# ---------------------------------------------------------------------------

def test_decode_logits_center_gaze():
    """Peak at bin 45 corresponds to 45 * 4 = 180 deg -> offset = 0 deg (0 rad)."""
    session = create_mock_session()
    estimator = MobileOneGazeEstimator(session=session)

    # Sharp one-hot-like peak at index 45
    center_logits = np.zeros(90, dtype=np.float32)
    center_logits[45] = 100.0  # Dominates softmax

    pitch_rad, yaw_rad, pitch_deg, yaw_deg = estimator.decode_logits(
        center_logits, center_logits
    )

    assert pytest.approx(0.0, abs=1e-3) == yaw_deg
    assert pytest.approx(0.0, abs=1e-3) == pitch_deg
    assert pytest.approx(0.0, abs=1e-3) == yaw_rad
    assert pytest.approx(0.0, abs=1e-3) == pitch_rad


def test_decode_logits_positive_and_negative_angles():
    session = create_mock_session()
    estimator = MobileOneGazeEstimator(session=session)

    # Bin 60 -> 60 * 4 = 240 deg -> 240 - 180 = +60 deg (+pi/3 rad)
    pos_logits = np.zeros(90, dtype=np.float32)
    pos_logits[60] = 100.0

    # Bin 30 -> 30 * 4 = 120 deg -> 120 - 180 = -60 deg (-pi/3 rad)
    neg_logits = np.zeros(90, dtype=np.float32)
    neg_logits[30] = 100.0

    pitch_rad, yaw_rad, pitch_deg, yaw_deg = estimator.decode_logits(
        yaw_logits=pos_logits, pitch_logits=neg_logits
    )

    assert pytest.approx(60.0, abs=1e-2) == yaw_deg
    assert pytest.approx(math.radians(60.0), abs=1e-3) == yaw_rad
    assert pytest.approx(-60.0, abs=1e-2) == pitch_deg
    assert pytest.approx(math.radians(-60.0), abs=1e-3) == pitch_rad


def test_decode_logits_rejects_non_finite():
    session = create_mock_session()
    estimator = MobileOneGazeEstimator(session=session)

    nan_logits = np.zeros(90, dtype=np.float32)
    nan_logits[10] = float("nan")

    with pytest.raises(ValueError, match="Non-finite"):
        estimator.decode_logits(nan_logits, np.zeros(90, dtype=np.float32))

    inf_logits = np.zeros(90, dtype=np.float32)
    inf_logits[10] = float("inf")

    with pytest.raises(ValueError, match="Non-finite"):
        estimator.decode_logits(np.zeros(90, dtype=np.float32), inf_logits)


def test_decode_logits_rejects_wrong_length():
    session = create_mock_session()
    estimator = MobileOneGazeEstimator(session=session)

    with pytest.raises(ValueError, match="length 90"):
        estimator.decode_logits(np.zeros(45, dtype=np.float32), np.zeros(90, dtype=np.float32))


# ---------------------------------------------------------------------------
# Session & End-to-End Prediction Tests
# ---------------------------------------------------------------------------

def test_session_validation_detects_structural_anomalies():
    # Missing input
    empty_input_session = MagicMock()
    empty_input_session.get_inputs.return_value = []
    with pytest.raises(ValueError, match="no input nodes"):
        MobileOneGazeEstimator(session=empty_input_session)

    # 1 output node instead of 2
    single_out_session = MagicMock()
    single_out_session.get_inputs.return_value = [MockNode("input", [1, 3, 448, 448])]
    single_out_session.get_outputs.return_value = [MockNode("yaw", [1, 90])]
    with pytest.raises(ValueError, match="exactly 2 output nodes"):
        MobileOneGazeEstimator(session=single_out_session)


def test_predict_end_to_end_mock():
    # Set up mock session returning center logits
    dummy_yaw = np.zeros((1, 90), dtype=np.float32)
    dummy_yaw[0, 45] = 50.0
    dummy_pitch = np.zeros((1, 90), dtype=np.float32)
    dummy_pitch[0, 45] = 50.0

    session = create_mock_session(mock_run_return=(dummy_yaw, dummy_pitch))
    estimator = MobileOneGazeEstimator(session=session)

    sample_crop = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    output = estimator.predict(sample_crop)

    assert output.is_valid is True
    assert output.error_message is None
    assert pytest.approx(0.0, abs=1e-2) == output.yaw_deg
    assert pytest.approx(0.0, abs=1e-2) == output.pitch_deg
    assert output.inference_latency_ms >= 0.0
    assert output.preprocessing_latency_ms >= 0.0


def test_predict_catches_inference_errors_gracefully():
    session = create_mock_session()
    session.run.side_effect = RuntimeError("Simulated ONNX Runtime device crash")
    estimator = MobileOneGazeEstimator(session=session)

    sample_crop = np.zeros((100, 100, 3), dtype=np.uint8)
    output = estimator.predict(sample_crop)

    assert output.is_valid is False
    assert "Simulated ONNX Runtime device crash" in output.error_message
    assert output.yaw_rad == 0.0
    assert output.pitch_rad == 0.0
