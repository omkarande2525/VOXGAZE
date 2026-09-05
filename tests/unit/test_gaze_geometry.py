"""
Unit tests for pure geometry calculations in backend.app.perception.geometry.
Zero external OpenCV/UI dependencies.
"""

import math
import pytest
from backend.app.core.contracts import Point2D
from backend.app.perception.geometry import (
    compute_ear,
    euclidean_distance,
    extract_bilateral_features,
    extract_eye_features,
    is_valid_point,
    norm_horizontal_gaze,
    norm_vertical_gaze,
)


def test_euclidean_distance():
    p1 = Point2D(x=0.0, y=0.0)
    p2 = Point2D(x=3.0, y=4.0)
    assert euclidean_distance(p1, p2) == pytest.approx(5.0)
    assert euclidean_distance((0, 0), (0, 10)) == pytest.approx(10.0)


def test_is_valid_point():
    assert is_valid_point(Point2D(x=10.0, y=20.0)) is True
    assert is_valid_point(Point2D(x=float("nan"), y=20.0)) is False
    assert is_valid_point(Point2D(x=10.0, y=float("inf"))) is False


def test_norm_horizontal_gaze():
    outer = Point2D(x=100.0, y=100.0)
    inner = Point2D(x=200.0, y=100.0)

    # Iris at center
    iris_center = Point2D(x=150.0, y=100.0)
    assert norm_horizontal_gaze(iris_center, outer, inner) == pytest.approx(0.50)

    # Iris at outer canthus (looking left)
    iris_left = Point2D(x=120.0, y=100.0)
    assert norm_horizontal_gaze(iris_left, outer, inner) == pytest.approx(0.20)

    # Iris at inner canthus (looking right)
    iris_right = Point2D(x=180.0, y=100.0)
    assert norm_horizontal_gaze(iris_right, outer, inner) == pytest.approx(0.80)


def test_norm_vertical_gaze():
    upper = Point2D(x=150.0, y=80.0)
    lower = Point2D(x=150.0, y=120.0)

    # Iris at vertical middle
    iris_center = Point2D(x=150.0, y=100.0)
    assert norm_vertical_gaze(iris_center, upper, lower) == pytest.approx(0.50)

    # Iris looking up
    iris_up = Point2D(x=150.0, y=90.0)
    assert norm_vertical_gaze(iris_up, upper, lower) == pytest.approx(0.25)

    # Iris looking down
    iris_down = Point2D(x=150.0, y=110.0)
    assert norm_vertical_gaze(iris_down, upper, lower) == pytest.approx(0.75)


def test_compute_ear():
    outer = Point2D(x=100.0, y=100.0)
    inner = Point2D(x=200.0, y=100.0)  # width = 100
    upper = Point2D(x=150.0, y=85.0)
    lower = Point2D(x=150.0, y=115.0)  # height = 30

    ear = compute_ear(upper, lower, outer, inner)
    assert ear == pytest.approx(0.30)

    # Closed eye (height = 5)
    upper_closed = Point2D(x=150.0, y=98.0)
    lower_closed = Point2D(x=150.0, y=102.0)
    ear_closed = compute_ear(upper_closed, lower_closed, outer, inner)
    assert ear_closed == pytest.approx(0.04)


def test_extract_eye_features_and_bilateral():
    outer = Point2D(x=100.0, y=100.0)
    inner = Point2D(x=200.0, y=100.0)
    upper = Point2D(x=150.0, y=85.0)
    lower = Point2D(x=150.0, y=115.0)
    iris = Point2D(x=150.0, y=100.0)

    eye_landmarks = extract_eye_features(iris, inner, outer, upper, lower)
    assert eye_landmarks.h_ratio == pytest.approx(0.50)
    assert eye_landmarks.v_ratio == pytest.approx(0.50)
    assert eye_landmarks.ear == pytest.approx(0.30)

    bilateral = extract_bilateral_features(
        timestamp_ms=1000,
        left_eye=eye_landmarks,
        right_eye=None,
        head_pose={"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
        tracking_confidence=0.95,
    )
    assert bilateral.timestamp_ms == 1000
    assert bilateral.ear_left == pytest.approx(0.30)
    assert bilateral.ear_right == 0.0
    assert bilateral.is_tracking_valid is True


def test_extract_eye_features_rejects_nan():
    outer = Point2D(x=100.0, y=100.0)
    inner = Point2D(x=200.0, y=100.0)
    upper = Point2D(x=150.0, y=float("nan"))
    lower = Point2D(x=150.0, y=115.0)
    iris = Point2D(x=150.0, y=100.0)

    with pytest.raises(ValueError, match="Non-finite"):
        extract_eye_features(iris, inner, outer, upper, lower)
