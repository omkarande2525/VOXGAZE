"""
Unit tests for ResNet-34 Gaze Estimator and Model Registry.
Tests preprocessing, numerical normalization, logit decoding, boundary conditions,
shape validation, non-finite rejection, mock ONNX inference, and polymorphic registry dispatch.
"""

import math
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from backend.app.perception.gaze_estimator_interface import (
    BaseGazeEstimator,
    GazeModelOutput,
)
from backend.app.perception.model_registry import (
    create_gaze_estimator,
    get_available_models,
)
from backend.app.perception.resnet34_estimator import (
    ResNet34GazeEstimator,
    ResNet34Output,
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


def test_resnet34_implements_interface():
    session = create_mock_session()
    estimator = ResNet34GazeEstimator(session=session)
    assert isinstance(estimator, BaseGazeEstimator)
    assert estimator.model_name == "resnet34"


def test_resnet34_preprocessing_shape_and_dtype():
    session = create_mock_session()
    estimator = ResNet34GazeEstimator(session=session)

    raw_crop = np.zeros((180, 220, 3), dtype=np.uint8)
    tensor = estimator.preprocess(raw_crop)

    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (1, 3, 448, 448)
    assert tensor.dtype == np.float32


def test_resnet34_preprocessing_rgb_and_standardization():
    session = create_mock_session()
    estimator = ResNet34GazeEstimator(session=session)

    pure_blue_bgr = np.zeros((64, 64, 3), dtype=np.uint8)
    pure_blue_bgr[:, :, 0] = 255  # B=255, G=0, R=0

    tensor = estimator.preprocess(pure_blue_bgr)

    # In RGB: R=0, G=0, B=1.0
    expected_r = (0.0 - 0.485) / 0.229
    expected_g = (0.0 - 0.456) / 0.224
    expected_b = (1.0 - 0.406) / 0.225

    np.testing.assert_allclose(tensor[0, 0, 0, 0], expected_r, rtol=1e-4)
    np.testing.assert_allclose(tensor[0, 1, 0, 0], expected_g, rtol=1e-4)
    np.testing.assert_allclose(tensor[0, 2, 0, 0], expected_b, rtol=1e-4)


def test_resnet34_preprocessing_rejects_invalid_inputs():
    session = create_mock_session()
    estimator = ResNet34GazeEstimator(session=session)

    with pytest.raises(ValueError, match="non-None"):
        estimator.preprocess(None)

    with pytest.raises(ValueError, match="non-empty 3-channel"):
        estimator.preprocess(np.array([], dtype=np.uint8))

    with pytest.raises(ValueError, match="non-empty 3-channel"):
        estimator.preprocess(np.zeros((100, 100), dtype=np.uint8))

    with pytest.raises(ValueError, match="non-empty 3-channel"):
        estimator.preprocess(np.zeros((100, 100, 4), dtype=np.uint8))


def test_resnet34_decode_logits_center():
    session = create_mock_session()
    estimator = ResNet34GazeEstimator(session=session)

    # Bin index 45 corresponds to 45 * 4 = 180 deg -> angle = 0 deg
    yaw_logits = np.full(90, -100.0, dtype=np.float32)
    yaw_logits[45] = 100.0

    pitch_logits = np.full(90, -100.0, dtype=np.float32)
    pitch_logits[45] = 100.0

    pitch_rad, yaw_rad, pitch_deg, yaw_deg = estimator.decode_logits(yaw_logits, pitch_logits)

    assert abs(yaw_deg) < 1e-4
    assert abs(pitch_deg) < 1e-4
    assert abs(yaw_rad) < 1e-4
    assert abs(pitch_rad) < 1e-4


def test_resnet34_decode_logits_rejects_non_finite_and_wrong_length():
    session = create_mock_session()
    estimator = ResNet34GazeEstimator(session=session)

    nan_logits = np.zeros(90, dtype=np.float32)
    nan_logits[10] = float("nan")
    valid_logits = np.zeros(90, dtype=np.float32)

    with pytest.raises(ValueError, match="Non-finite values"):
        estimator.decode_logits(nan_logits, valid_logits)

    with pytest.raises(ValueError, match="Logit vectors must have length 90"):
        estimator.decode_logits(np.zeros(45), np.zeros(90))


def test_resnet34_predict_end_to_end_mock():
    # Bin 45 on both -> 0 deg
    mock_yaw = np.full((1, 90), -100.0, dtype=np.float32)
    mock_yaw[0, 45] = 100.0
    mock_pitch = np.full((1, 90), -100.0, dtype=np.float32)
    mock_pitch[0, 45] = 100.0

    session = create_mock_session(mock_run_return=(mock_yaw, mock_pitch))
    estimator = ResNet34GazeEstimator(session=session)

    crop = np.zeros((120, 120, 3), dtype=np.uint8)
    output = estimator.predict(crop)

    assert isinstance(output, ResNet34Output)
    assert isinstance(output, GazeModelOutput)
    assert output.model_name == "resnet34"
    assert output.is_valid is True
    assert abs(output.yaw_deg) < 1e-4
    assert abs(output.pitch_deg) < 1e-4
    assert output.inference_latency_ms >= 0.0
    assert output.preprocessing_latency_ms >= 0.0


def test_resnet34_predict_catches_errors_gracefully():
    session = create_mock_session()
    session.run.side_effect = RuntimeError("Inference kernel crashed")
    estimator = ResNet34GazeEstimator(session=session)

    crop = np.zeros((64, 64, 3), dtype=np.uint8)
    output = estimator.predict(crop)

    assert output.is_valid is False
    assert "Inference kernel crashed" in output.error_message
    assert output.pitch_deg == 0.0
    assert output.yaw_deg == 0.0


def test_model_registry_dispatch():
    models = get_available_models()
    assert "mobileone_s0" in models
    assert "resnet34" in models

    mock_sess = create_mock_session()
    m1 = create_gaze_estimator("mobileone_s0", session=mock_sess)
    r34 = create_gaze_estimator("resnet34", session=mock_sess)

    assert m1.model_name == "mobileone_s0"
    assert r34.model_name == "resnet34"

    with pytest.raises(ValueError, match="Unknown model architecture"):
        create_gaze_estimator("transformer_v99")
