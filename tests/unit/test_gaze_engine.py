"""
Unit tests for GazeEngine hysteresis, quality abstention, and state transitions.
"""

import pytest
from backend.app.core.contracts import Direction
from backend.app.perception.gaze_engine import GazeEngine
from scripts.generate_synthetic_gaze import get_mock_calibration_profile, make_sample


def test_gaze_engine_uncalibrated_behavior():
    engine = GazeEngine(profile=None)
    sample = make_sample(1000, 0.50, 0.50)
    state = engine.process_features(sample)
    assert state.direction == Direction.UNCALIBRATED
    assert "! UNCALIBRATED" in state.status_flag


def test_gaze_engine_quality_gate_abstentions():
    profile = get_mock_calibration_profile()
    engine = GazeEngine(profile=profile, min_confidence=0.6)

    # 1. No face
    no_face_sample = make_sample(1000, 0.50, 0.50, no_face=True)
    assert engine.process_features(no_face_sample).direction == Direction.NO_FACE

    # 2. Low confidence
    low_conf_sample = make_sample(1033, 0.50, 0.50, confidence=0.4)
    assert engine.process_features(low_conf_sample).direction == Direction.ABSTAIN

    # 3. Explicit invalid flag
    invalid_sample = make_sample(1066, 0.50, 0.50, is_valid=False)
    assert engine.process_features(invalid_sample).direction == Direction.ABSTAIN


def test_gaze_engine_directional_hysteresis():
    profile = get_mock_calibration_profile()
    # Left: enter = 0.38, exit = 0.43
    # Right: enter = 0.62, exit = 0.57
    engine = GazeEngine(profile=profile)

    # Start Center
    assert engine.process_features(make_sample(0, 0.50, 0.50)).direction == Direction.CENTER

    # Looking slightly left, but not crossing enter threshold (0.40 > 0.38) -> Still CENTER
    assert engine.process_features(make_sample(33, 0.40, 0.50)).direction == Direction.CENTER

    # Cross LEFT enter threshold (0.35 < 0.38) -> Transition to LEFT
    assert engine.process_features(make_sample(66, 0.35, 0.50)).direction == Direction.LEFT

    # Oscillating near boundary (0.41 < 0.43 exit threshold) -> Must strictly stay LEFT!
    assert engine.process_features(make_sample(99, 0.41, 0.50)).direction == Direction.LEFT
    assert engine.process_features(make_sample(132, 0.42, 0.50)).direction == Direction.LEFT

    # Cross LEFT exit threshold back to center (0.46 > 0.43) -> Transition to CENTER
    assert engine.process_features(make_sample(165, 0.46, 0.50)).direction == Direction.CENTER


def test_gaze_engine_vertical_hysteresis():
    profile = get_mock_calibration_profile()
    # Up: enter = 0.38, exit = 0.43
    # Down: enter = 0.62, exit = 0.57
    engine = GazeEngine(profile=profile)

    # Look UP (0.30 < 0.38)
    assert engine.process_features(make_sample(0, 0.50, 0.30)).direction == Direction.UP
    # Jitter to 0.41 (< 0.43 exit) -> Stay UP
    assert engine.process_features(make_sample(33, 0.50, 0.41)).direction == Direction.UP
    # Cross exit (0.46 > 0.43) -> Return CENTER
    assert engine.process_features(make_sample(66, 0.50, 0.46)).direction == Direction.CENTER

    # Look DOWN (0.68 > 0.62)
    assert engine.process_features(make_sample(99, 0.50, 0.68)).direction == Direction.DOWN
    # Jitter to 0.59 (> 0.57 exit) -> Stay DOWN
    assert engine.process_features(make_sample(132, 0.50, 0.59)).direction == Direction.DOWN
    # Cross exit (0.54 < 0.57) -> Return CENTER
    assert engine.process_features(make_sample(165, 0.50, 0.54)).direction == Direction.CENTER


def test_gaze_engine_blink_measurement():
    profile = get_mock_calibration_profile()
    # blink_closed_ear = 0.18, blink_open_ear = 0.28
    engine = GazeEngine(profile=profile)

    # Eyes open
    assert engine.process_features(make_sample(0, 0.50, 0.50, ear=0.30)).direction == Direction.CENTER

    # Eyes closed for 300ms
    assert engine.process_features(make_sample(100, 0.50, 0.50, ear=0.10)).direction == Direction.BLINK
    assert engine.process_features(make_sample(200, 0.50, 0.50, ear=0.10)).direction == Direction.BLINK
    assert engine.process_features(make_sample(400, 0.50, 0.50, ear=0.10)).direction == Direction.BLINK

    # Reopen eyes
    state_open = engine.process_features(make_sample(500, 0.50, 0.50, ear=0.30))
    assert state_open.direction == Direction.CENTER
    assert engine.last_blink_duration_ms >= 300.0
