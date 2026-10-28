"""
Unit tests for deterministic face crop extraction helper (backend/app/perception/face_crop.py).
Validates boundary clamping, square-expansion geometry, minimum size gating, and malformed input handling.
All tests use synthetic inputs and require zero webcam hardware.
"""

from dataclasses import dataclass
import numpy as np
import pytest

from backend.app.perception.face_crop import extract_face_crop, FaceCropResult


@dataclass
class SyntheticLandmark:
    x: float
    y: float
    z: float = 0.0


def create_synthetic_face_landmarks(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    num_points: int = 468,
) -> list:
    """Generates synthetic landmarks simulating an elliptical face distribution."""
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    landmarks = []
    rx = width / 2.0
    ry = height / 2.0
    for a in angles:
        px = center_x + rx * np.cos(a)
        py = center_y + ry * np.sin(a)
        landmarks.append(SyntheticLandmark(x=px, y=py))
    return landmarks


def test_valid_landmarks_produce_valid_crop():
    # 720p frame: 1280x720
    frame = np.full((720, 1280, 3), 128, dtype=np.uint8)

    # Face centered at (640, 360) in normalized coordinates (0.5, 0.5), width 200px (0.156), height 240px (0.333)
    landmarks = create_synthetic_face_landmarks(0.5, 0.5, 200 / 1280, 240 / 720)

    result = extract_face_crop(frame, landmarks, padding_scale=1.20, min_size_px=64)

    assert result.is_valid is True
    assert result.crop_bgr is not None
    assert result.rejection_reason is None
    assert result.crop_bgr.ndim == 3
    assert result.crop_bgr.shape[2] == 3

    # Check square dimensions: max_side = 240, padded = 240 * 1.2 = 288 px
    # Expected width and height approximately 288 px
    h_crop, w_crop = result.crop_bgr.shape[:2]
    assert pytest.approx(288, abs=2) == w_crop
    assert pytest.approx(288, abs=2) == h_crop


def test_clamping_at_frame_edges():
    # Frame 640x480
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)

    # Face near the top-left boundary (center at 0.05, 0.05)
    landmarks = create_synthetic_face_landmarks(0.05, 0.05, 0.08, 0.08)

    result = extract_face_crop(frame, landmarks, padding_scale=1.20, min_size_px=32)

    assert result.is_valid is True
    x1, y1, x2, y2 = result.bbox_clamped

    # Coordinates must be strictly clamped to frame boundaries [0, W] and [0, H]
    assert x1 >= 0
    assert y1 >= 0
    assert x2 <= 640
    assert y2 <= 480
    assert result.crop_bgr.shape[0] == y2 - y1
    assert result.crop_bgr.shape[1] == x2 - x1


def test_rejects_crops_below_minimum_size_gate():
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)

    # Tiny face: 20x20 pixels (below 64px gate)
    landmarks = create_synthetic_face_landmarks(0.5, 0.5, 20 / 640, 20 / 480)

    result = extract_face_crop(frame, landmarks, padding_scale=1.20, min_size_px=64)

    assert result.is_valid is False
    assert result.crop_bgr is None
    assert "below minimum size gate" in result.rejection_reason


def test_malformed_and_empty_inputs_do_not_crash():
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)

    # None landmarks
    res1 = extract_face_crop(frame, None)
    assert res1.is_valid is False
    assert "No landmarks" in res1.rejection_reason

    # Empty landmark list
    res2 = extract_face_crop(frame, [])
    assert res2.is_valid is False
    assert "Insufficient" in res2.rejection_reason

    # Landmarks with NaN/Inf coordinates
    nan_landmarks = [SyntheticLandmark(x=float("nan"), y=0.5) for _ in range(50)]
    res3 = extract_face_crop(frame, nan_landmarks)
    assert res3.is_valid is False

    # None frame
    res4 = extract_face_crop(None, [SyntheticLandmark(0.5, 0.5) for _ in range(50)])
    assert res4.is_valid is False

    # 2D Grayscale frame instead of 3-channel BGR
    res5 = extract_face_crop(np.zeros((100, 100), dtype=np.uint8), [SyntheticLandmark(0.5, 0.5) for _ in range(50)])
    assert res5.is_valid is False


def test_padding_and_square_crop_are_deterministic():
    frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
    landmarks = create_synthetic_face_landmarks(0.5, 0.5, 150 / 1280, 200 / 720)

    # Run multiple times with same inputs
    res_a = extract_face_crop(frame, landmarks, padding_scale=1.20)
    res_b = extract_face_crop(frame, landmarks, padding_scale=1.20)

    assert res_a.is_valid is True
    assert res_b.is_valid is True
    assert res_a.bbox_clamped == res_b.bbox_clamped
    assert res_a.bbox_raw == res_b.bbox_raw
    np.testing.assert_array_equal(res_a.crop_bgr, res_b.crop_bgr)
