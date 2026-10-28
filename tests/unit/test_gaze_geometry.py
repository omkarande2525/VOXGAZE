"""
Unit tests for pure geometry calculations in backend.app.perception.geometry.
Zero external OpenCV/UI dependencies.
Verifies anatomical eye topology, directional normalization conventions,
mirrored display invariance, and degenerate geometry handling.
"""

import math
import pytest
from backend.app.core.contracts import Point2D
from backend.app.perception.geometry import (
    LEFT_INNER_CANTHUS,
    LEFT_IRIS_ALL,
    LEFT_IRIS_CENTER,
    LEFT_LOWER_LID,
    LEFT_OUTER_CANTHUS,
    LEFT_UPPER_LID,
    RIGHT_INNER_CANTHUS,
    RIGHT_IRIS_ALL,
    RIGHT_IRIS_CENTER,
    RIGHT_LOWER_LID,
    RIGHT_OUTER_CANTHUS,
    RIGHT_UPPER_LID,
    compute_ear,
    compute_canthal_normal_distance_px,
    euclidean_distance,
    extract_bilateral_features,
    extract_eye_features,
    fuse_bilateral_mean,
    fuse_bilateral_weighted,
    is_valid_point,
    norm_horizontal_gaze,
    norm_vertical_gaze,
    norm_vertical_gaze_canthal_normal_experimental,
    project_ratio_2d,
    BilateralQualityResult,
    EyeQualityResult,
    GeometryQualityReason,
    validate_bilateral_geometry,
    validate_eye_geometry,
    MIN_EYE_WIDTH_PX,
    MIN_LID_SEPARATION_PX,
    MIN_EAR_FOR_GAZE,
    MAX_PLAUSIBLE_EAR,
    PLAUSIBLE_H_MIN,
    PLAUSIBLE_H_MAX,
    PLAUSIBLE_V_MIN,
    PLAUSIBLE_V_MAX,
    PLAUSIBLE_CANTHAL_MIN,
    PLAUSIBLE_CANTHAL_MAX,
)


def test_landmark_topology_indices():
    """Verify official MediaPipe Face Mesh landmark indices for both eyes."""
    assert RIGHT_IRIS_CENTER == 468
    assert RIGHT_IRIS_ALL == [468, 469, 470, 471, 472]
    assert RIGHT_OUTER_CANTHUS == 33
    assert RIGHT_INNER_CANTHUS == 133
    assert RIGHT_UPPER_LID == 159
    assert RIGHT_LOWER_LID == 145

    assert LEFT_IRIS_CENTER == 473
    assert LEFT_IRIS_ALL == [473, 474, 475, 476, 477]
    assert LEFT_OUTER_CANTHUS == 263
    assert LEFT_INNER_CANTHUS == 362
    assert LEFT_UPPER_LID == 386
    assert LEFT_LOWER_LID == 374


def test_euclidean_distance():
    p1 = Point2D(x=0.0, y=0.0)
    p2 = Point2D(x=3.0, y=4.0)
    assert euclidean_distance(p1, p2) == pytest.approx(5.0)
    assert euclidean_distance((0, 0), (0, 10)) == pytest.approx(10.0)


def test_is_valid_point():
    assert is_valid_point(Point2D(x=10.0, y=20.0)) is True
    assert is_valid_point(Point2D(x=float("nan"), y=20.0)) is False
    assert is_valid_point(Point2D(x=10.0, y=float("inf"))) is False
    assert is_valid_point((10.0, 20.0)) is True
    assert is_valid_point("invalid") is False


def test_anatomical_right_eye_normalization():
    """
    Anatomical RIGHT eye normalization into common user-direction convention:
    User physical LEFT  -> iris near inner canthus (133, nasal)     -> H ≈ 0.0
    User physical RIGHT -> iris near outer canthus (33, temporal)   -> H ≈ 1.0
    """
    inner = Point2D(x=150.0, y=100.0)  # Nasal (user's left)
    outer = Point2D(x=50.0, y=100.0)   # Temporal (user's right)

    # Iris at exact inner canthus -> H = 0.0 (USER_LEFT)
    assert norm_horizontal_gaze(inner, inner, outer, is_right_eye=True) == pytest.approx(0.0)

    # Iris near inner canthus (looking left) -> H ≈ 0.10
    iris_left = Point2D(x=140.0, y=100.0)
    assert norm_horizontal_gaze(iris_left, inner, outer, is_right_eye=True) == pytest.approx(0.10)

    # Iris at center -> H = 0.50
    iris_center = Point2D(x=100.0, y=100.0)
    assert norm_horizontal_gaze(iris_center, inner, outer, is_right_eye=True) == pytest.approx(0.50)

    # Iris near outer canthus (looking right) -> H ≈ 0.90
    iris_right = Point2D(x=60.0, y=100.0)
    assert norm_horizontal_gaze(iris_right, inner, outer, is_right_eye=True) == pytest.approx(0.90)

    # Iris at exact outer canthus -> H = 1.0 (USER_RIGHT)
    assert norm_horizontal_gaze(outer, inner, outer, is_right_eye=True) == pytest.approx(1.0)


def test_anatomical_left_eye_normalization():
    """
    Anatomical LEFT eye normalization into common user-direction convention:
    User physical LEFT  -> iris near outer canthus (263, temporal)  -> H ≈ 0.0
    User physical RIGHT -> iris near inner canthus (362, nasal)     -> H ≈ 1.0
    """
    outer = Point2D(x=250.0, y=100.0)  # Temporal (user's left)
    inner = Point2D(x=150.0, y=100.0)  # Nasal (user's right)

    # Iris at exact outer canthus -> H = 0.0 (USER_LEFT)
    assert norm_horizontal_gaze(outer, inner, outer, is_right_eye=False) == pytest.approx(0.0)

    # Iris near outer canthus (looking left) -> H ≈ 0.10
    iris_left = Point2D(x=240.0, y=100.0)
    assert norm_horizontal_gaze(iris_left, inner, outer, is_right_eye=False) == pytest.approx(0.10)

    # Iris at center -> H = 0.50
    iris_center = Point2D(x=200.0, y=100.0)
    assert norm_horizontal_gaze(iris_center, inner, outer, is_right_eye=False) == pytest.approx(0.50)

    # Iris near inner canthus (looking right) -> H ≈ 0.90
    iris_right = Point2D(x=160.0, y=100.0)
    assert norm_horizontal_gaze(iris_right, inner, outer, is_right_eye=False) == pytest.approx(0.90)

    # Iris at exact inner canthus -> H = 1.0 (USER_RIGHT)
    assert norm_horizontal_gaze(inner, inner, outer, is_right_eye=False) == pytest.approx(1.0)


def test_vertical_normalization():
    """
    Vertical normalization:
    UP   -> iris near upper lid (159/386) -> V ≈ 0.0
    DOWN -> iris near lower lid (145/374) -> V ≈ 1.0
    """
    upper = Point2D(x=100.0, y=50.0)
    lower = Point2D(x=100.0, y=150.0)

    # Iris at exact upper lid -> V = 0.0 (UP)
    assert norm_vertical_gaze(upper, upper, lower) == pytest.approx(0.0)

    # Iris looking up -> V ≈ 0.10
    iris_up = Point2D(x=100.0, y=60.0)
    assert norm_vertical_gaze(iris_up, upper, lower) == pytest.approx(0.10)

    # Iris at center -> V = 0.50
    iris_center = Point2D(x=100.0, y=100.0)
    assert norm_vertical_gaze(iris_center, upper, lower) == pytest.approx(0.50)

    # Iris looking down -> V ≈ 0.90
    iris_down = Point2D(x=100.0, y=140.0)
    assert norm_vertical_gaze(iris_down, upper, lower) == pytest.approx(0.90)

    # Iris at exact lower lid -> V = 1.0 (DOWN)
    assert norm_vertical_gaze(lower, upper, lower) == pytest.approx(1.0)


def test_vertical_monotonicity_fixed_eyelids():
    """
    Synthetic verification of vertical normalization monotonicity with fixed eyelids:
    Moving iris strictly upward (decreasing y in image coordinates) must strictly decrease V.
    Moving iris strictly downward (increasing y in image coordinates) must strictly increase V.
    """
    upper = Point2D(x=100.0, y=50.0)
    lower = Point2D(x=100.0, y=150.0)

    # Sequence of y positions from top (y=60) to bottom (y=140)
    y_positions = [60.0, 75.0, 90.0, 100.0, 110.0, 125.0, 140.0]
    v_values = [norm_vertical_gaze(Point2D(x=100.0, y=y), upper, lower) for y in y_positions]

    # Strictly monotonically increasing as y increases (moving downward)
    for i in range(len(v_values) - 1):
        assert v_values[i + 1] > v_values[i], f"Expected V to increase as y increases (downward), but {v_values[i+1]} <= {v_values[i]}"

    # Conversely, moving upward (reversing the sequence) strictly decreases V
    reversed_v = list(reversed(v_values))
    for i in range(len(reversed_v) - 1):
        assert reversed_v[i + 1] < reversed_v[i], f"Expected V to decrease as y decreases (upward), but {reversed_v[i+1]} >= {reversed_v[i]}"

    # Verify with slanted eyelids (non-vertical axis)
    upper_slanted = Point2D(x=95.0, y=50.0)
    lower_slanted = Point2D(x=105.0, y=150.0)
    v_slanted = [norm_vertical_gaze(Point2D(x=100.0, y=y), upper_slanted, lower_slanted) for y in y_positions]
    for i in range(len(v_slanted) - 1):
        assert v_slanted[i + 1] > v_slanted[i]


def test_canthal_normal_vertical_monotonicity():
    """
    Synthetic verification of canthal-normal vertical projection:
    Moving iris upward (decreasing y) strictly decreases V_canthal and distance_px.
    Moving iris downward (increasing y) strictly increases V_canthal and distance_px.
    Sign convention: UP < CENTER < DOWN.
    """
    inner = Point2D(x=150.0, y=100.0)  # nasal (user left)
    outer = Point2D(x=50.0, y=100.0)   # temporal (user right), width = 100

    # Iris on the canthal line (y = 100) -> V = 0.0, dist = 0.0 px
    iris_center = Point2D(x=100.0, y=100.0)
    assert norm_vertical_gaze_canthal_normal_experimental(iris_center, inner, outer, is_right_eye=True) == pytest.approx(0.0)
    assert compute_canthal_normal_distance_px(iris_center, inner, outer, is_right_eye=True) == pytest.approx(0.0)

    # Sequence of y positions: up (y=70, 80, 90) -> center (y=100) -> down (y=110, 120, 130)
    y_positions = [70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0]
    v_values = [
        norm_vertical_gaze_canthal_normal_experimental(Point2D(x=100.0, y=y), inner, outer, is_right_eye=True)
        for y in y_positions
    ]
    dist_values = [
        compute_canthal_normal_distance_px(Point2D(x=100.0, y=y), inner, outer, is_right_eye=True)
        for y in y_positions
    ]

    # Strictly monotonically increasing as iris moves downward
    for i in range(len(v_values) - 1):
        assert v_values[i + 1] > v_values[i]
        assert dist_values[i + 1] > dist_values[i]

    # Upward positions have negative values (V < 0)
    for y, v in zip(y_positions[:3], v_values[:3]):
        assert v < 0.0

    # Downward positions have positive values (V > 0)
    for y, v in zip(y_positions[4:], v_values[4:]):
        assert v > 0.0

    # Explicit ordering check: UP < CENTER < DOWN
    v_up = norm_vertical_gaze_canthal_normal_experimental(Point2D(x=100.0, y=85.0), inner, outer, is_right_eye=True)
    v_center = norm_vertical_gaze_canthal_normal_experimental(Point2D(x=100.0, y=100.0), inner, outer, is_right_eye=True)
    v_down = norm_vertical_gaze_canthal_normal_experimental(Point2D(x=100.0, y=115.0), inner, outer, is_right_eye=True)
    assert v_up < v_center < v_down


def test_canthal_normal_both_anatomical_eyes():
    """
    Verify that the anatomical coordinate convention produces identical
    UP < CENTER < DOWN sign conventions for BOTH right and left eyes.
    """
    # RIGHT EYE: inner at (150, 100), outer at (50, 100) -> width = 100
    r_inner = Point2D(x=150.0, y=100.0)
    r_outer = Point2D(x=50.0, y=100.0)
    r_up = Point2D(x=100.0, y=85.0)     # 15 px up
    r_center = Point2D(x=100.0, y=100.0)
    r_down = Point2D(x=100.0, y=115.0)  # 15 px down

    rv_up = norm_vertical_gaze_canthal_normal_experimental(r_up, r_inner, r_outer, is_right_eye=True)
    rv_center = norm_vertical_gaze_canthal_normal_experimental(r_center, r_inner, r_outer, is_right_eye=True)
    rv_down = norm_vertical_gaze_canthal_normal_experimental(r_down, r_inner, r_outer, is_right_eye=True)

    assert rv_up == pytest.approx(-0.15)
    assert rv_center == pytest.approx(0.00)
    assert rv_down == pytest.approx(+0.15)
    assert rv_up < rv_center < rv_down

    # LEFT EYE: outer at (250, 100), inner at (150, 100) -> width = 100
    l_outer = Point2D(x=250.0, y=100.0)
    l_inner = Point2D(x=150.0, y=100.0)
    l_up = Point2D(x=200.0, y=85.0)     # 15 px up
    l_center = Point2D(x=200.0, y=100.0)
    l_down = Point2D(x=200.0, y=115.0)  # 15 px down

    lv_up = norm_vertical_gaze_canthal_normal_experimental(l_up, l_inner, l_outer, is_right_eye=False)
    lv_center = norm_vertical_gaze_canthal_normal_experimental(l_center, l_inner, l_outer, is_right_eye=False)
    lv_down = norm_vertical_gaze_canthal_normal_experimental(l_down, l_inner, l_outer, is_right_eye=False)

    assert lv_up == pytest.approx(-0.15)
    assert lv_center == pytest.approx(0.00)
    assert lv_down == pytest.approx(+0.15)
    assert lv_up < lv_center < lv_down


def test_canthal_normal_slanted_axes():
    """
    Verify that head roll / slanted canthal axes preserve UP < CENTER < DOWN
    in the face-aligned anatomical coordinate frame.
    """
    angles = [-30.0, -15.0, 0.0, 15.0, 30.0]
    for angle_deg in angles:
        rad = math.radians(angle_deg)
        dx = -100.0 * math.cos(rad)
        dy = -100.0 * math.sin(rad)
        inner = Point2D(x=200.0, y=200.0)
        outer = Point2D(x=200.0 + dx, y=200.0 + dy)
        mid_x = (inner.x + outer.x) / 2.0
        mid_y = (inner.y + outer.y) / 2.0

        # Normal unit vector pointing DOWN (+y in head frame): (dy, -dx) / 100
        u_down_x = dy / 100.0
        u_down_y = -dx / 100.0

        iris_center = Point2D(x=mid_x, y=mid_y)
        iris_up = Point2D(x=mid_x - 12.0 * u_down_x, y=mid_y - 12.0 * u_down_y)
        iris_down = Point2D(x=mid_x + 12.0 * u_down_x, y=mid_y + 12.0 * u_down_y)

        v_up = norm_vertical_gaze_canthal_normal_experimental(iris_up, inner, outer, is_right_eye=True)
        v_center = norm_vertical_gaze_canthal_normal_experimental(iris_center, inner, outer, is_right_eye=True)
        v_down = norm_vertical_gaze_canthal_normal_experimental(iris_down, inner, outer, is_right_eye=True)

        assert v_up == pytest.approx(-0.12, abs=1e-5)
        assert v_center == pytest.approx(0.00, abs=1e-5)
        assert v_down == pytest.approx(+0.12, abs=1e-5)
        assert v_up < v_center < v_down


def test_canthal_normal_degenerate_and_invalid():
    """Verify graceful handling of zero-width eyes and NaN rejection."""
    # Degenerate zero-width eye returns 0.0
    pt = Point2D(x=100.0, y=100.0)
    assert norm_vertical_gaze_canthal_normal_experimental(pt, pt, pt, is_right_eye=True) == 0.0
    assert compute_canthal_normal_distance_px(pt, pt, pt, is_right_eye=True) == 0.0

    # Non-finite coordinate rejection
    inner = Point2D(x=150.0, y=100.0)
    outer = Point2D(x=50.0, y=100.0)
    nan_pt = Point2D(x=100.0, y=float("nan"))
    with pytest.raises(ValueError, match="Non-finite"):
        norm_vertical_gaze_canthal_normal_experimental(nan_pt, inner, outer, is_right_eye=True)
    with pytest.raises(ValueError, match="Non-finite"):
        compute_canthal_normal_distance_px(nan_pt, inner, outer, is_right_eye=True)


def test_mirrored_display_semantics():
    """
    Verify that horizontal reflection (screen mirroring x -> W - 1 - x)
    preserves normalized H ratio because normalization is computed relative
    to anatomical landmark vectors.
    """
    w = 640
    # Right eye raw coords
    r_inner = Point2D(x=150.0, y=100.0)
    r_outer = Point2D(x=50.0, y=100.0)
    r_iris = Point2D(x=130.0, y=100.0)  # looking left

    h_raw = norm_horizontal_gaze(r_iris, r_inner, r_outer, is_right_eye=True)

    # Mirrored coords
    r_inner_mirrored = Point2D(x=float(w - 1 - r_inner.x), y=100.0)
    r_outer_mirrored = Point2D(x=float(w - 1 - r_outer.x), y=100.0)
    r_iris_mirrored = Point2D(x=float(w - 1 - r_iris.x), y=100.0)

    h_mirrored = norm_horizontal_gaze(r_iris_mirrored, r_inner_mirrored, r_outer_mirrored, is_right_eye=True)

    assert h_raw == pytest.approx(h_mirrored)
    assert h_raw == pytest.approx(0.20)


def test_compute_ear():
    outer = Point2D(x=50.0, y=100.0)
    inner = Point2D(x=150.0, y=100.0)  # width = 100
    upper = Point2D(x=100.0, y=85.0)
    lower = Point2D(x=100.0, y=115.0)  # height = 30

    ear = compute_ear(upper, lower, outer, inner)
    assert ear == pytest.approx(0.30)

    # Closed eye (height = 4)
    upper_closed = Point2D(x=100.0, y=98.0)
    lower_closed = Point2D(x=100.0, y=102.0)
    ear_closed = compute_ear(upper_closed, lower_closed, outer, inner)
    assert ear_closed == pytest.approx(0.04)


def test_invalid_and_degenerate_geometry():
    """Verify graceful handling of degenerate zero-width eyes and NaN rejection."""
    # Degenerate zero-width eye
    pt = Point2D(x=100.0, y=100.0)
    ratio = project_ratio_2d(pt, pt, pt)
    assert ratio == 0.50

    # Non-finite coordinate rejection
    inner = Point2D(x=150.0, y=100.0)
    outer = Point2D(x=50.0, y=100.0)
    upper = Point2D(x=100.0, y=float("nan"))
    lower = Point2D(x=100.0, y=115.0)
    iris = Point2D(x=100.0, y=100.0)

    with pytest.raises(ValueError, match="Non-finite"):
        extract_eye_features(iris, inner, outer, upper, lower, is_right_eye=True)


def test_extract_eye_features_and_bilateral():
    """Verify full extraction contract with both anatomical eyes."""
    r_inner = Point2D(x=150.0, y=100.0)
    r_outer = Point2D(x=50.0, y=100.0)
    r_upper = Point2D(x=100.0, y=85.0)
    r_lower = Point2D(x=100.0, y=115.0)
    r_iris = Point2D(x=100.0, y=100.0)

    right_eye = extract_eye_features(r_iris, r_inner, r_outer, r_upper, r_lower, is_right_eye=True)
    assert right_eye.h_ratio == pytest.approx(0.50)
    assert right_eye.v_ratio == pytest.approx(0.50)
    assert right_eye.ear == pytest.approx(0.30)

    l_inner = Point2D(x=150.0, y=100.0)
    l_outer = Point2D(x=250.0, y=100.0)
    l_upper = Point2D(x=200.0, y=85.0)
    l_lower = Point2D(x=200.0, y=115.0)
    l_iris = Point2D(x=200.0, y=100.0)

    left_eye = extract_eye_features(l_iris, l_inner, l_outer, l_upper, l_lower, is_right_eye=False)
    assert left_eye.h_ratio == pytest.approx(0.50)
    assert left_eye.v_ratio == pytest.approx(0.50)
    assert left_eye.ear == pytest.approx(0.30)

    bilateral = extract_bilateral_features(
        timestamp_ms=1000,
        left_eye=left_eye,
        right_eye=right_eye,
        tracking_confidence=0.95,
    )
    assert bilateral.timestamp_ms == 1000
    assert bilateral.ear_left == pytest.approx(0.30)
    assert bilateral.ear_right == pytest.approx(0.30)
    assert bilateral.is_tracking_valid is True


def test_bilateral_fusion_mean_and_weighted():
    """Verify arithmetic mean and confidence-weighted fusion math."""
    # Arithmetic mean: (0.40 + 0.60) / 2 = 0.50
    assert fuse_bilateral_mean(0.40, 0.60) == pytest.approx(0.50)
    assert fuse_bilateral_mean(-0.06, -0.04) == pytest.approx(-0.05)

    # Weighted fusion with explicit physical confidence
    # (0.40 * 0.8 + 0.60 * 0.2) / 1.0 = 0.32 + 0.12 = 0.44
    assert fuse_bilateral_weighted(0.40, 0.60, conf_right=0.8, conf_left=0.2) == pytest.approx(0.44)

    # Equal confidence matches arithmetic mean
    w_equal = fuse_bilateral_weighted(0.473, 0.495, conf_right=0.9, conf_left=0.9)
    m_equal = fuse_bilateral_mean(0.473, 0.495)
    assert w_equal == pytest.approx(m_equal)

    # Zero confidence fallback
    assert fuse_bilateral_weighted(0.30, 0.50, conf_right=0.0, conf_left=0.0) == pytest.approx(0.40)

    # Non-finite rejection
    with pytest.raises(ValueError, match="Non-finite"):
        fuse_bilateral_mean(float("nan"), 0.50)
    with pytest.raises(ValueError, match="Non-finite"):
        fuse_bilateral_weighted(0.50, float("inf"), 1.0, 1.0)


def test_bilateral_fusion_invariants():
    """Verify bilateral fusion preserves physical direction ordering."""
    # Horizontal: LEFT < CENTER < RIGHT
    # Right eye: 0.47, 0.50, 0.54
    # Left eye:  0.46, 0.51, 0.55
    h_b_left = fuse_bilateral_mean(0.47, 0.46)
    h_b_center = fuse_bilateral_mean(0.50, 0.51)
    h_b_right = fuse_bilateral_mean(0.54, 0.55)
    assert h_b_left < h_b_center < h_b_right

    # Vertical (Canthal Normal): UP < CENTER < DOWN
    # Right eye: -0.064, -0.060, -0.041
    # Left eye:  -0.066, -0.059, -0.040
    v_b_up = fuse_bilateral_mean(-0.064, -0.066)
    v_b_center = fuse_bilateral_mean(-0.060, -0.059)
    v_b_down = fuse_bilateral_mean(-0.041, -0.040)
    assert v_b_up < v_b_center < v_b_down


# =========================================================================
# Geometry Quality Validation Tests (M1.5-A Hardening)
# =========================================================================

def test_validate_eye_geometry_valid_right_and_left():
    """Verify clean, valid geometry produces is_valid=True with no rejection reasons."""
    # Right eye: width = 100, height = 30, EAR = 0.30, center gaze
    r_inner = Point2D(x=150.0, y=100.0)
    r_outer = Point2D(x=50.0, y=100.0)
    r_upper = Point2D(x=100.0, y=85.0)
    r_lower = Point2D(x=100.0, y=115.0)
    r_iris = Point2D(x=100.0, y=100.0)

    res_r = validate_eye_geometry(r_iris, r_inner, r_outer, r_upper, r_lower, is_right_eye=True)
    assert res_r.is_valid is True
    assert len(res_r.rejection_reasons) == 0
    assert res_r.eye_width_px == pytest.approx(100.0)
    assert res_r.lid_separation_px == pytest.approx(30.0)
    assert res_r.h_ratio == pytest.approx(0.50)
    assert res_r.v_ratio == pytest.approx(0.50)
    assert res_r.ear == pytest.approx(0.30)
    assert res_r.canthal_ratio == pytest.approx(0.0)

    # Left eye: width = 100, height = 30, EAR = 0.30, center gaze
    l_inner = Point2D(x=150.0, y=100.0)
    l_outer = Point2D(x=250.0, y=100.0)
    l_upper = Point2D(x=200.0, y=85.0)
    l_lower = Point2D(x=200.0, y=115.0)
    l_iris = Point2D(x=200.0, y=100.0)

    res_l = validate_eye_geometry(l_iris, l_inner, l_outer, l_upper, l_lower, is_right_eye=False)
    assert res_l.is_valid is True
    assert len(res_l.rejection_reasons) == 0
    assert res_l.eye_width_px == pytest.approx(100.0)
    assert res_l.h_ratio == pytest.approx(0.50)


def test_validate_eye_geometry_non_finite():
    """Verify NaN and Inf coordinates are rejected with NON_FINITE_COORDINATES."""
    inner = Point2D(x=150.0, y=100.0)
    outer = Point2D(x=50.0, y=100.0)
    upper = Point2D(x=100.0, y=float("nan"))
    lower = Point2D(x=100.0, y=115.0)
    iris = Point2D(x=100.0, y=100.0)

    res = validate_eye_geometry(iris, inner, outer, upper, lower, is_right_eye=True)
    assert res.is_valid is False
    assert GeometryQualityReason.NON_FINITE_COORDINATES in res.rejection_reasons

    # Infinity check
    iris_inf = Point2D(x=float("inf"), y=100.0)
    upper_finite = Point2D(x=100.0, y=85.0)
    res_inf = validate_eye_geometry(iris_inf, inner, outer, upper_finite, lower, is_right_eye=True)
    assert res_inf.is_valid is False
    assert GeometryQualityReason.NON_FINITE_COORDINATES in res_inf.rejection_reasons


def test_validate_eye_geometry_sub_threshold_and_zero_width():
    """Verify eyes with width < MIN_EYE_WIDTH_PX are rejected."""
    # Zero width
    pt = Point2D(x=100.0, y=100.0)
    res_zero = validate_eye_geometry(pt, pt, pt, pt, pt, is_right_eye=True)
    assert res_zero.is_valid is False
    assert GeometryQualityReason.INSUFFICIENT_EYE_WIDTH in res_zero.rejection_reasons
    assert GeometryQualityReason.INSUFFICIENT_LID_SEPARATION in res_zero.rejection_reasons

    # Sub-threshold width = 5.0 px (< 10.0 px threshold)
    inner = Point2D(x=105.0, y=100.0)
    outer = Point2D(x=100.0, y=100.0)
    upper = Point2D(x=102.5, y=95.0)
    lower = Point2D(x=102.5, y=105.0)
    iris = Point2D(x=102.5, y=100.0)

    res_sub = validate_eye_geometry(iris, inner, outer, upper, lower, is_right_eye=True)
    assert res_sub.is_valid is False
    assert GeometryQualityReason.INSUFFICIENT_EYE_WIDTH in res_sub.rejection_reasons
    assert res_sub.eye_width_px == pytest.approx(5.0)


def test_validate_eye_geometry_lid_separation_and_blink():
    """Verify closed/blinking eyes (lid separation < 2.0 or EAR < 0.08) are rejected."""
    inner = Point2D(x=150.0, y=100.0)
    outer = Point2D(x=50.0, y=100.0)  # width = 100
    # Closed lids: separation = 1.0 px, EAR = 0.01
    upper = Point2D(x=100.0, y=99.5)
    lower = Point2D(x=100.0, y=100.5)
    iris = Point2D(x=100.0, y=100.0)

    res = validate_eye_geometry(iris, inner, outer, upper, lower, is_right_eye=True)
    assert res.is_valid is False
    assert GeometryQualityReason.INSUFFICIENT_LID_SEPARATION in res.rejection_reasons
    assert GeometryQualityReason.EYELID_CLOSED_OR_BLINK in res.rejection_reasons
    assert res.lid_separation_px == pytest.approx(1.0)
    assert res.ear == pytest.approx(0.01)

    # Moderate closure: lid separation = 5.0 px (passes separation), but EAR = 0.05 (< 0.08)
    upper_mod = Point2D(x=100.0, y=97.5)
    lower_mod = Point2D(x=100.0, y=102.5)
    res_mod = validate_eye_geometry(iris, inner, outer, upper_mod, lower_mod, is_right_eye=True)
    assert res_mod.is_valid is False
    assert GeometryQualityReason.EYELID_CLOSED_OR_BLINK in res_mod.rejection_reasons
    assert GeometryQualityReason.INSUFFICIENT_LID_SEPARATION not in res_mod.rejection_reasons


def test_validate_eye_geometry_implausible_ear():
    """Verify degraded/distorted landmarks with EAR > 0.70 are rejected."""
    inner = Point2D(x=150.0, y=100.0)
    outer = Point2D(x=50.0, y=100.0)  # width = 100
    # Implausible vertical stretch: height = 85 px -> EAR = 0.85
    upper = Point2D(x=100.0, y=57.5)
    lower = Point2D(x=100.0, y=142.5)
    iris = Point2D(x=100.0, y=100.0)

    res = validate_eye_geometry(iris, inner, outer, upper, lower, is_right_eye=True)
    assert res.is_valid is False
    assert GeometryQualityReason.IMPLAUSIBLE_EAR in res.rejection_reasons
    assert res.ear == pytest.approx(0.85)


def test_validate_eye_geometry_plausible_boundary_ratios():
    """Verify boundary behavior for horizontal, vertical, and canthal ratios."""
    inner = Point2D(x=150.0, y=100.0)
    outer = Point2D(x=50.0, y=100.0)  # width = 100
    upper = Point2D(x=100.0, y=85.0)
    lower = Point2D(x=100.0, y=115.0) # height = 30

    # 1. Horizontal at boundary: H = 1.25 (boundary passes)
    iris_h_boundary = Point2D(x=25.0, y=100.0)  # dx = -100, px - sx = 25 - 150 = -125 -> H = 1.25
    res_h_bound = validate_eye_geometry(iris_h_boundary, inner, outer, upper, lower, is_right_eye=True)
    assert GeometryQualityReason.IMPLAUSIBLE_H_RATIO not in res_h_bound.rejection_reasons

    # 2. Horizontal out-of-band: H = 1.50
    iris_h_out = Point2D(x=0.0, y=100.0)  # px - sx = 0 - 150 = -150 -> H = 1.50
    res_h_out = validate_eye_geometry(iris_h_out, inner, outer, upper, lower, is_right_eye=True)
    assert res_h_out.is_valid is False
    assert GeometryQualityReason.IMPLAUSIBLE_H_RATIO in res_h_out.rejection_reasons
    assert res_h_out.h_ratio == pytest.approx(1.50)

    # 3. Canthal out-of-band: canthal_ratio = -0.40 (outside [-0.25, 0.25])
    iris_canthal_out = Point2D(x=100.0, y=60.0)  # dy=0, dx=-100 -> py-sy = -40, -(-40)*(-100) = -4000/10000 = -0.40
    res_c_out = validate_eye_geometry(iris_canthal_out, inner, outer, upper, lower, is_right_eye=True)
    assert res_c_out.is_valid is False
    assert GeometryQualityReason.IMPLAUSIBLE_CANTHAL_RATIO in res_c_out.rejection_reasons
    assert res_c_out.canthal_ratio == pytest.approx(-0.40)


def test_validate_eye_geometry_preserves_raw_unclamped_measurements():
    """Verify that extreme raw outlier ratios are NEVER clamped or overwritten."""
    inner = Point2D(x=150.0, y=100.0)
    outer = Point2D(x=50.0, y=100.0)
    # Severe blink configuration mimicking Run 2 Row 608
    upper = Point2D(x=100.0, y=99.5)
    lower = Point2D(x=100.0, y=100.5)
    iris = Point2D(x=100.0, y=94.0)  # 5.5 px above upper lid

    res = validate_eye_geometry(iris, inner, outer, upper, lower, is_right_eye=True)
    assert res.is_valid is False
    # Raw v_ratio is preserved as extreme negative value (~ -5.5), NOT clamped to 0.0
    assert res.v_ratio < -4.0
    assert res.v_ratio != 0.0


def test_validate_bilateral_geometry_asymmetric():
    """Verify bilateral handling when one eye is valid and the other is invalid."""
    # Right eye valid
    r_inner = Point2D(x=150.0, y=100.0)
    r_outer = Point2D(x=50.0, y=100.0)
    r_upper = Point2D(x=100.0, y=85.0)
    r_lower = Point2D(x=100.0, y=115.0)
    r_iris = Point2D(x=100.0, y=100.0)
    res_r = validate_eye_geometry(r_iris, r_inner, r_outer, r_upper, r_lower, is_right_eye=True)
    assert res_r.is_valid is True

    # Left eye closed (blink)
    l_inner = Point2D(x=150.0, y=100.0)
    l_outer = Point2D(x=250.0, y=100.0)
    l_upper = Point2D(x=200.0, y=99.5)
    l_lower = Point2D(x=200.0, y=100.5)
    l_iris = Point2D(x=200.0, y=100.0)
    res_l = validate_eye_geometry(l_iris, l_inner, l_outer, l_upper, l_lower, is_right_eye=False)
    assert res_l.is_valid is False

    # Strict bilateral (requires both) -> False
    bi_strict = validate_bilateral_geometry(res_r, res_l, require_both_eyes=True)
    assert bi_strict.is_valid is False
    assert any("LEFT_EYE_INVALID" in r for r in bi_strict.bilateral_reasons)

    # Permissive bilateral (either eye valid) -> True
    bi_permissive = validate_bilateral_geometry(res_r, res_l, require_both_eyes=False)
    assert bi_permissive.is_valid is True
    # Reason still reports that left eye was invalid for diagnostics
    assert any("LEFT_EYE_INVALID" in r for r in bi_permissive.bilateral_reasons)


# =========================================================================
# Regression tests: M1.5-B1 calibration harness point-type crash
# Verifies that extract_eye_features enforces its Point2D contract and
# that validate_eye_geometry continues to accept Union[Point2D, Tuple].
# =========================================================================


class TestCalibrationHarnessPointTypeRegression:
    """
    Regression tests for the AttributeError crash caused by passing raw tuples
    to extract_eye_features() which expects Point2D objects.

    Root cause: scripts/run_calibration_validation.py passed (float, float) tuples
    as landmarks, but extract_eye_features accesses pt.x / pt.y directly.
    Fix: Convert to Point2D at the harness boundary.
    """

    # Realistic eye landmarks (pixels) matching a typical 640x480 frame
    R_IRIS = Point2D(x=280.0, y=195.0)
    R_INNER = Point2D(x=310.0, y=195.0)
    R_OUTER = Point2D(x=250.0, y=195.0)
    R_UPPER = Point2D(x=280.0, y=188.0)
    R_LOWER = Point2D(x=280.0, y=202.0)

    L_IRIS = Point2D(x=360.0, y=195.0)
    L_INNER = Point2D(x=330.0, y=195.0)
    L_OUTER = Point2D(x=390.0, y=195.0)
    L_UPPER = Point2D(x=360.0, y=188.0)
    L_LOWER = Point2D(x=360.0, y=202.0)

    def test_extract_eye_features_accepts_point2d(self):
        """extract_eye_features should work with Point2D inputs (the correct type)."""
        r_feat = extract_eye_features(
            self.R_IRIS, self.R_INNER, self.R_OUTER, self.R_UPPER, self.R_LOWER,
            is_right_eye=True,
        )
        assert r_feat.iris_center == self.R_IRIS
        assert isinstance(r_feat.h_ratio, float)
        assert isinstance(r_feat.v_ratio, float)
        assert isinstance(r_feat.ear, float)

        l_feat = extract_eye_features(
            self.L_IRIS, self.L_INNER, self.L_OUTER, self.L_UPPER, self.L_LOWER,
            is_right_eye=False,
        )
        assert l_feat.iris_center == self.L_IRIS
        assert isinstance(l_feat.h_ratio, float)

    def test_extract_eye_features_rejects_raw_tuples(self):
        """
        extract_eye_features must fail on raw tuples because it accesses pt.x.
        This regression test ensures we catch a tuple being passed rather than
        silently producing garbage.
        """
        r_iris_tuple = (280.0, 195.0)
        r_inner_tuple = (310.0, 195.0)
        r_outer_tuple = (250.0, 195.0)
        r_upper_tuple = (280.0, 188.0)
        r_lower_tuple = (280.0, 202.0)

        with pytest.raises(AttributeError):
            extract_eye_features(
                r_iris_tuple, r_inner_tuple, r_outer_tuple,
                r_upper_tuple, r_lower_tuple, is_right_eye=True,
            )

    def test_validate_eye_geometry_still_accepts_tuples(self):
        """
        validate_eye_geometry uses Union[Point2D, Tuple] so tuples must still work.
        This ensures the fix didn't break Union-typed functions.
        """
        r_iris_tuple = (280.0, 195.0)
        r_inner_tuple = (310.0, 195.0)
        r_outer_tuple = (250.0, 195.0)
        r_upper_tuple = (280.0, 188.0)
        r_lower_tuple = (280.0, 202.0)

        result = validate_eye_geometry(
            r_iris_tuple, r_inner_tuple, r_outer_tuple,
            r_upper_tuple, r_lower_tuple, is_right_eye=True,
        )
        # Should produce a valid result (not crash)
        assert isinstance(result.is_valid, bool)
        assert isinstance(result.h_ratio, float)
        assert isinstance(result.ear, float)

    def test_point2d_from_numpy_coordinates(self):
        """
        Simulates the exact conversion pattern used in the calibration harness:
        extracting from a numpy pts array, wrapping in Point2D, then calling
        extract_eye_features.
        """
        import numpy as np

        # Simulate: pts = np.array([[p.x * w, p.y * h] for p in face])
        pts = np.array([
            [280.0, 195.0],  # idx 0 = iris
            [310.0, 195.0],  # idx 1 = inner
            [250.0, 195.0],  # idx 2 = outer
            [280.0, 188.0],  # idx 3 = upper
            [280.0, 202.0],  # idx 4 = lower
        ])

        iris = Point2D(x=float(pts[0, 0]), y=float(pts[0, 1]))
        inner = Point2D(x=float(pts[1, 0]), y=float(pts[1, 1]))
        outer = Point2D(x=float(pts[2, 0]), y=float(pts[2, 1]))
        upper = Point2D(x=float(pts[3, 0]), y=float(pts[3, 1]))
        lower = Point2D(x=float(pts[4, 0]), y=float(pts[4, 1]))

        feat = extract_eye_features(iris, inner, outer, upper, lower, is_right_eye=True)
        assert feat.iris_center.x == pytest.approx(280.0)
        assert feat.iris_center.y == pytest.approx(195.0)
        assert isinstance(feat.h_ratio, float)
        assert isinstance(feat.v_ratio, float)
        assert isinstance(feat.ear, float)


def test_validate_eye_geometry_closed_eyes_handling():
    """Verify that allow_closed_eyes permits low EAR while strictly rejecting non-finite and malformed samples."""
    # Typical closed eye geometry: inner to outer canthus is 30px, upper to lower lid is 1.0px (EAR ≈ 0.033)
    inner = (50.0, 100.0)
    outer = (20.0, 100.0)
    iris = (35.0, 100.0)
    upper = (35.0, 99.5)
    lower = (35.0, 100.5)

    # 1. Default (open-eye mode) rejects low EAR and narrow lid separation
    res_open = validate_eye_geometry(iris, inner, outer, upper, lower, is_right_eye=True, allow_closed_eyes=False)
    assert not res_open.is_valid
    assert GeometryQualityReason.EYELID_CLOSED_OR_BLINK in res_open.rejection_reasons
    assert GeometryQualityReason.INSUFFICIENT_LID_SEPARATION in res_open.rejection_reasons

    # 2. Closed-eye mode permits valid closed eye geometry
    res_closed = validate_eye_geometry(iris, inner, outer, upper, lower, is_right_eye=True, allow_closed_eyes=True)
    assert res_closed.is_valid
    assert len(res_closed.rejection_reasons) == 0
    assert res_closed.ear < 0.05

    # 3. Closed-eye mode STILL rejects non-finite coordinates
    res_nan = validate_eye_geometry((float("nan"), 100.0), inner, outer, upper, lower, is_right_eye=True, allow_closed_eyes=True)
    assert not res_nan.is_valid
    assert GeometryQualityReason.NON_FINITE_COORDINATES in res_nan.rejection_reasons

    # 4. Closed-eye mode STILL rejects collapsed eye width (zero/sub-threshold width)
    res_collapsed = validate_eye_geometry(iris, (50.0, 100.0), (50.0, 100.0), upper, lower, is_right_eye=True, allow_closed_eyes=True)
    assert not res_collapsed.is_valid
    assert GeometryQualityReason.INSUFFICIENT_EYE_WIDTH in res_collapsed.rejection_reasons

    # 5. Closed-eye mode STILL rejects implausible EAR (negative or extreme > 1.2)
    res_implausible = validate_eye_geometry(iris, inner, outer, (35.0, 50.0), (35.0, 150.0), is_right_eye=True, allow_closed_eyes=True)
    assert not res_implausible.is_valid
    assert GeometryQualityReason.IMPLAUSIBLE_EAR in res_implausible.rejection_reasons
