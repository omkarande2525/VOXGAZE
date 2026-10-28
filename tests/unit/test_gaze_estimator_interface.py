"""
Unit tests for BaseGazeEstimator and GazeModelOutput data contract.
"""

from backend.app.perception.gaze_estimator_interface import (
    BaseGazeEstimator,
    GazeModelOutput,
)


def test_gaze_model_output_immutability():
    output = GazeModelOutput(
        pitch_rad=0.1,
        yaw_rad=0.2,
        pitch_deg=5.73,
        yaw_deg=11.46,
        inference_latency_ms=5.5,
        preprocessing_latency_ms=0.8,
        model_name="test_model",
        is_valid=True,
    )
    assert output.pitch_rad == 0.1
    assert output.yaw_rad == 0.2
    assert output.model_name == "test_model"
    assert output.is_valid is True
    assert output.error_message is None
