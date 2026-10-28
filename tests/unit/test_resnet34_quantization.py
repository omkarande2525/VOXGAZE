"""
Unit tests for ResNet-34 ONNX Quantization (Task T4).
Tests dynamic and static INT8 models on memory-safe fixtures, verifying:
- Non-empty output contract compliance
- Numerical bounds on synthetic face crops
- Finite float outputs with valid radians/degrees conversions
"""

from pathlib import Path
import pytest
import numpy as np

from backend.app.perception.resnet34_estimator import ResNet34GazeEstimator, ResNet34Output
from scripts.quantize_resnet34 import (
    generate_memory_safe_test_crops,
    DYNAMIC_INT8_PATH,
    STATIC_INT8_PATH,
    FP32_PATH,
)


@pytest.fixture(scope="module")
def fp32_estimator():
    if not FP32_PATH.is_file():
        pytest.skip(f"FP32 model not found at {FP32_PATH}")
    return ResNet34GazeEstimator(model_path=FP32_PATH)


@pytest.fixture(scope="module")
def dynamic_int8_estimator():
    if not DYNAMIC_INT8_PATH.is_file():
        pytest.skip(f"Dynamic INT8 model not found at {DYNAMIC_INT8_PATH}")
    return ResNet34GazeEstimator(model_path=DYNAMIC_INT8_PATH)


@pytest.fixture(scope="module")
def static_int8_estimator():
    if not STATIC_INT8_PATH.is_file():
        pytest.skip(f"Static INT8 model not found at {STATIC_INT8_PATH}")
    return ResNet34GazeEstimator(model_path=STATIC_INT8_PATH)


def test_quantized_models_exist():
    assert FP32_PATH.is_file(), "FP32 ResNet-34 model missing"
    assert DYNAMIC_INT8_PATH.is_file(), "Dynamic INT8 model missing"
    assert STATIC_INT8_PATH.is_file(), "Static INT8 model missing"


def test_dynamic_int8_numerical_fidelity_vs_fp32(fp32_estimator, dynamic_int8_estimator):
    """Dynamic INT8 quantizing Gemm/MatMul heads must maintain < 0.5 deg error vs FP32."""
    crops = generate_memory_safe_test_crops()
    assert len(crops) >= 3

    for name, crop in crops:
        out_fp = fp32_estimator.predict(crop)
        out_dyn = dynamic_int8_estimator.predict(crop)

        assert out_fp.is_valid is True
        assert out_dyn.is_valid is True

        yaw_err = abs(out_dyn.yaw_deg - out_fp.yaw_deg)
        pitch_err = abs(out_dyn.pitch_deg - out_fp.pitch_deg)

        assert yaw_err < 0.5, f"Dynamic INT8 yaw delta too large on {name}: {yaw_err} deg"
        assert pitch_err < 0.5, f"Dynamic INT8 pitch delta too large on {name}: {pitch_err} deg"


def test_static_int8_validity_and_bounds(static_int8_estimator):
    """Static QDQ INT8 model must execute without crash and return valid bounded angles."""
    crops = generate_memory_safe_test_crops()
    assert len(crops) >= 3

    for name, crop in crops:
        out_stat = static_int8_estimator.predict(crop)

        assert isinstance(out_stat, ResNet34Output)
        assert out_stat.is_valid is True
        assert -180.0 <= out_stat.yaw_deg <= 180.0
        assert -180.0 <= out_stat.pitch_deg <= 180.0
        assert out_stat.inference_latency_ms > 0.0


def test_static_int8_model_size_reduction():
    """Static INT8 model must achieve >= 60% compression vs FP32."""
    fp32_size = FP32_PATH.stat().st_size
    stat_size = STATIC_INT8_PATH.stat().st_size
    compression_ratio = 1.0 - (stat_size / fp32_size)
    assert compression_ratio >= 0.60, f"Static INT8 compression insufficient: {compression_ratio*100:.1f}%"
