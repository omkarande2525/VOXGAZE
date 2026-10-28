"""
VoxGaze Pure Geometry & Eye Feature Normalization Module
Strictly pure mathematical functions with ZERO OpenCV, MediaPipe, or UI dependencies.
"""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Dict, List, Optional, Tuple, Union
from backend.app.core.contracts import EyeLandmarks, GazeFeatures, Point2D

# -------------------------------------------------------------------------
# MediaPipe Face Landmarker Topology (Anatomical Definitions)
# -------------------------------------------------------------------------
# Anatomical RIGHT eye (subject's right)
RIGHT_IRIS_CENTER = 468
RIGHT_IRIS_ALL = [468, 469, 470, 471, 472]
RIGHT_OUTER_CANTHUS = 33
RIGHT_INNER_CANTHUS = 133
RIGHT_UPPER_LID = 159
RIGHT_LOWER_LID = 145

# Anatomical LEFT eye (subject's left)
LEFT_IRIS_CENTER = 473
LEFT_IRIS_ALL = [473, 474, 475, 476, 477]
LEFT_OUTER_CANTHUS = 263
LEFT_INNER_CANTHUS = 362
LEFT_UPPER_LID = 386
LEFT_LOWER_LID = 374


def euclidean_distance(p1: Union[Point2D, Tuple[float, float]], p2: Union[Point2D, Tuple[float, float]]) -> float:
    """Calculate 2D Euclidean distance between two points."""
    x1, y1 = (p1.x, p1.y) if isinstance(p1, Point2D) else p1
    x2, y2 = (p2.x, p2.y) if isinstance(p2, Point2D) else p2
    return math.hypot(x2 - x1, y2 - y1)


def is_valid_point(p: Union[Point2D, Tuple[float, float]]) -> bool:
    """Validate that point coordinates are finite real numbers."""
    try:
        x, y = (p.x, p.y) if isinstance(p, Point2D) else (p[0], p[1])
        x = float(x)
        y = float(y)
        return math.isfinite(x) and math.isfinite(y)
    except (TypeError, ValueError, AttributeError, IndexError):
        return False


def project_ratio_2d(
    pt: Union[Point2D, Tuple[float, float]],
    p_start: Union[Point2D, Tuple[float, float]],
    p_end: Union[Point2D, Tuple[float, float]],
    eps: float = 1e-7,
) -> float:
    """
    Project point pt onto oriented line segment from p_start to p_end.
    Returns 0.0 when pt projects to p_start, and 1.0 when pt projects to p_end.
    Uses 2D vector projection: ((pt - p_start) . (p_end - p_start)) / (|p_end - p_start|^2 + eps).
    Handles degenerate collinear/zero-length segments gracefully.
    """
    px, py = (pt.x, pt.y) if isinstance(pt, Point2D) else pt
    sx, sy = (p_start.x, p_start.y) if isinstance(p_start, Point2D) else p_start
    ex, ey = (p_end.x, p_end.y) if isinstance(p_end, Point2D) else p_end

    dx = ex - sx
    dy = ey - sy
    seg_len_sq = dx * dx + dy * dy

    if seg_len_sq < eps:
        return 0.5

    vx = px - sx
    vy = py - sy

    proj = (vx * dx + vy * dy) / (seg_len_sq + eps)
    return float(proj)


def norm_horizontal_gaze(
    iris_pt: Union[Point2D, Tuple[float, float]],
    inner_canthus: Union[Point2D, Tuple[float, float]],
    outer_canthus: Union[Point2D, Tuple[float, float]],
    is_right_eye: bool = True,
    eps: float = 1e-7,
) -> float:
    """
    Calculate normalized horizontal iris position along the eye axis.
    Normalizes either anatomical eye into a common user-direction coordinate convention:
      H ≈ 0.0 -> USER_LEFT
      H ≈ 1.0 -> USER_RIGHT

    For anatomical RIGHT eye:
      User physical LEFT is toward inner canthus (nasal, 133).
      User physical RIGHT is toward outer canthus (temporal, 33).
      p_start = inner_canthus, p_end = outer_canthus.

    For anatomical LEFT eye:
      User physical LEFT is toward outer canthus (temporal, 263).
      User physical RIGHT is toward inner canthus (nasal, 362).
      p_start = outer_canthus, p_end = inner_canthus.
    """
    if is_right_eye:
        p_start = inner_canthus
        p_end = outer_canthus
    else:
        p_start = outer_canthus
        p_end = inner_canthus

    return project_ratio_2d(iris_pt, p_start, p_end, eps=eps)


def norm_vertical_gaze(
    iris_pt: Union[Point2D, Tuple[float, float]],
    upper_lid: Union[Point2D, Tuple[float, float]],
    lower_lid: Union[Point2D, Tuple[float, float]],
    eps: float = 1e-7,
) -> float:
    """
    Calculate normalized vertical iris position between upper and lower eyelids.
    Normalizes gaze into a common vertical convention:
      V ≈ 0.0 -> UP (iris near upper lid)
      V ≈ 1.0 -> DOWN (iris near lower lid)
    """
    return project_ratio_2d(iris_pt, upper_lid, lower_lid, eps=eps)


def norm_vertical_gaze_canthal_normal_experimental(
    iris_pt: Union[Point2D, Tuple[float, float]],
    inner_canthus: Union[Point2D, Tuple[float, float]],
    outer_canthus: Union[Point2D, Tuple[float, float]],
    is_right_eye: bool = True,
    eps: float = 1e-7,
) -> float:
    """
    EXPERIMENTAL: Calculate normalized vertical iris position relative to the
    perpendicular normal of the canthal eye axis (landmarks 33 and 133 / 263 and 362).

    Unlike eyelid-based vertical normalization, the canthi are anchored to facial
    skeletal anatomy (palpebral ligaments) and do not retract or droop during eye motion.

    Coordinate / Sign Convention (BOTH anatomical eyes):
      V < 0.0: UP (iris displaced superiorly / above the canthal line)
      V = 0.0: ON AXIS (iris center intersects the canthal axis)
      V > 0.0: DOWN (iris displaced inferiorly / below the canthal line)
      Monotonic ordering invariant: UP < CENTER < DOWN

    Anatomical Orientation Rationale:
      In standard camera image coordinates (+x right, +y down):
      - For anatomical RIGHT eye: p_start = inner_canthus (133), p_end = outer_canthus (33).
        Vector v = (p_end - p_start) has dx < 0 (pointing user-right / image-left).
      - For anatomical LEFT eye: p_start = outer_canthus (263), p_end = inner_canthus (362).
        Vector v = (p_end - p_start) has dx < 0 (pointing user-right / image-left).
      In both cases, rotating vector v by 90 degrees clockwise in image coordinates:
        n = (dy, -dx)
      Because dx < 0, the y-component -dx > 0, pointing strictly DOWNWARDS (+y)
      along the anatomical vertical axis of the eye.
      Therefore, for BOTH eyes, moving the iris UP reduces the projection along n,
      and moving the iris DOWN increases the projection along n.
    """
    for pt in (iris_pt, inner_canthus, outer_canthus):
        if not is_valid_point(pt):
            raise ValueError(f"Non-finite coordinates received in canthal normal calculation: {pt}")

    if is_right_eye:
        p_start = inner_canthus
        p_end = outer_canthus
    else:
        p_start = outer_canthus
        p_end = inner_canthus

    px, py = (iris_pt.x, iris_pt.y) if isinstance(iris_pt, Point2D) else iris_pt
    sx, sy = (p_start.x, p_start.y) if isinstance(p_start, Point2D) else p_start
    ex, ey = (p_end.x, p_end.y) if isinstance(p_end, Point2D) else p_end

    dx = ex - sx
    dy = ey - sy
    eye_width_sq = dx * dx + dy * dy

    if eye_width_sq < eps:
        return 0.0

    # Normal vector pointing downwards: (dy, -dx)
    # Dot product: (px - sx) * dy - (py - sy) * dx
    # Normalized by eye_width_sq (= dist_perp / eye_width)
    proj = ((px - sx) * dy - (py - sy) * dx) / (eye_width_sq + eps)
    return float(proj)


def compute_canthal_normal_distance_px(
    iris_pt: Union[Point2D, Tuple[float, float]],
    inner_canthus: Union[Point2D, Tuple[float, float]],
    outer_canthus: Union[Point2D, Tuple[float, float]],
    is_right_eye: bool = True,
    eps: float = 1e-7,
) -> float:
    """
    EXPERIMENTAL: Compute unnormalized signed perpendicular distance (in pixels)
    from the canthal axis to the iris center.
    UP < 0.0 px, DOWN > 0.0 px.
    """
    for pt in (iris_pt, inner_canthus, outer_canthus):
        if not is_valid_point(pt):
            raise ValueError(f"Non-finite coordinates received: {pt}")

    if is_right_eye:
        p_start = inner_canthus
        p_end = outer_canthus
    else:
        p_start = outer_canthus
        p_end = inner_canthus

    px, py = (iris_pt.x, iris_pt.y) if isinstance(iris_pt, Point2D) else iris_pt
    sx, sy = (p_start.x, p_start.y) if isinstance(p_start, Point2D) else p_start
    ex, ey = (p_end.x, p_end.y) if isinstance(p_end, Point2D) else p_end

    dx = ex - sx
    dy = ey - sy
    eye_width = math.hypot(dx, dy)

    if eye_width < eps:
        return 0.0

    # Signed distance: ((px - sx) * dy - (py - sy) * dx) / eye_width
    dist_px = ((px - sx) * dy - (py - sy) * dx) / (eye_width + eps)
    return float(dist_px)


def compute_ear(
    upper_lid: Union[Point2D, Tuple[float, float]],
    lower_lid: Union[Point2D, Tuple[float, float]],
    outer_canthus: Union[Point2D, Tuple[float, float]],
    inner_canthus: Union[Point2D, Tuple[float, float]],
    eps: float = 1e-7,
) -> float:
    """
    Compute Eye Aspect Ratio (EAR) as vertical opening / horizontal eye width.
    """
    height = euclidean_distance(upper_lid, lower_lid)
    width = euclidean_distance(outer_canthus, inner_canthus) + eps
    return float(height / width)


def extract_eye_features(
    iris_center: Point2D,
    inner_canthus: Point2D,
    outer_canthus: Point2D,
    upper_lid: Point2D,
    lower_lid: Point2D,
    is_right_eye: bool = True,
) -> EyeLandmarks:
    """
    Construct validated EyeLandmarks contract with computed H, V, and EAR.
    Normalizes either anatomical eye into a common user-direction coordinate convention.
    """
    for pt in [iris_center, inner_canthus, outer_canthus, upper_lid, lower_lid]:
        if not math.isfinite(pt.x) or not math.isfinite(pt.y):
            raise ValueError(f"Non-finite landmark coordinates received: ({pt.x}, {pt.y})")

    h = norm_horizontal_gaze(iris_center, inner_canthus, outer_canthus, is_right_eye=is_right_eye)
    v = norm_vertical_gaze(iris_center, upper_lid, lower_lid)
    ear = compute_ear(upper_lid, lower_lid, outer_canthus, inner_canthus)
    canthal = norm_vertical_gaze_canthal_normal_experimental(iris_center, inner_canthus, outer_canthus, is_right_eye=is_right_eye)

    return EyeLandmarks(
        iris_center=iris_center,
        inner_canthus=inner_canthus,
        outer_canthus=outer_canthus,
        upper_lid=upper_lid,
        lower_lid=lower_lid,
        h_ratio=h,
        v_ratio=v,
        ear=ear,
        canthal_ratio=canthal,
    )


def extract_bilateral_features(
    timestamp_ms: int,
    left_eye: Optional[EyeLandmarks] = None,
    right_eye: Optional[EyeLandmarks] = None,
    head_pose: Optional[dict] = None,
    tracking_confidence: float = 1.0,
    is_tracking_valid: bool = True,
    is_right_eye_valid: Optional[bool] = None,
    is_left_eye_valid: Optional[bool] = None,
) -> GazeFeatures:
    """
    Bilateral contract constructor supporting left eye, right eye, and head pose.
    Evaluates per-eye geometry validity and enforces that is_tracking_valid cannot
    remain True if both eyes are invalid or missing.
    """
    ear_left = left_eye.ear if left_eye else 0.0
    ear_right = right_eye.ear if right_eye else 0.0

    # Auto-evaluate right eye validity if not explicitly provided
    if is_right_eye_valid is None and right_eye is not None:
        try:
            r_res = validate_eye_geometry(
                right_eye.iris_center,
                right_eye.inner_canthus,
                right_eye.outer_canthus,
                right_eye.upper_lid,
                right_eye.lower_lid,
                is_right_eye=True,
            )
            is_right_eye_valid = r_res.is_valid
        except Exception:
            is_right_eye_valid = False

    # Auto-evaluate left eye validity if not explicitly provided
    if is_left_eye_valid is None and left_eye is not None:
        try:
            l_res = validate_eye_geometry(
                left_eye.iris_center,
                left_eye.inner_canthus,
                left_eye.outer_canthus,
                left_eye.upper_lid,
                left_eye.lower_lid,
                is_right_eye=False,
            )
            is_left_eye_valid = l_res.is_valid
        except Exception:
            is_left_eye_valid = False

    # If tracking was claimed to be valid, but both eyes are present and invalid, or no eyes present:
    # Tracking cannot remain valid if there is no valid eye signal.
    if is_tracking_valid:
        has_at_least_one_valid_eye = (is_right_eye_valid is True) or (is_left_eye_valid is True)
        # If neither eye has evaluated to True and at least one evaluated to False (or no eyes at all):
        if (right_eye is None and left_eye is None) or (is_right_eye_valid is False and is_left_eye_valid is False):
            is_tracking_valid = False
        elif (right_eye is not None and is_right_eye_valid is False and left_eye is None):
            is_tracking_valid = False
        elif (left_eye is not None and is_left_eye_valid is False and right_eye is None):
            is_tracking_valid = False

    return GazeFeatures(
        timestamp_ms=timestamp_ms,
        left_eye=left_eye,
        right_eye=right_eye,
        head_pose=head_pose,
        ear_left=ear_left,
        ear_right=ear_right,
        tracking_confidence=tracking_confidence,
        is_tracking_valid=is_tracking_valid,
        is_right_eye_valid=is_right_eye_valid,
        is_left_eye_valid=is_left_eye_valid,
    )


def fuse_bilateral_mean(val_right: float, val_left: float) -> float:
    """
    EXPERIMENTAL DIAGNOSTIC: Arithmetic mean fusion of normalized bilateral signals.
    Primary fusion: (val_right + val_left) / 2.0
    Requires both signals to be finite real numbers.
    """
    if not (math.isfinite(val_right) and math.isfinite(val_left)):
        raise ValueError(f"Non-finite values passed to fuse_bilateral_mean: ({val_right}, {val_left})")
    return float(0.5 * (val_right + val_left))


def fuse_bilateral_weighted(
    val_right: float,
    val_left: float,
    conf_right: float,
    conf_left: float,
    eps: float = 1e-7,
) -> float:
    """
    EXPERIMENTAL DIAGNOSTIC ONLY: Confidence-weighted fusion of bilateral normalized signals.
    Uses explicit, measurable physical quality signals (e.g. tracking confidence or valid geometry).
    Strictly zero learned or tuned weights.
    """
    if not (math.isfinite(val_right) and math.isfinite(val_left) and math.isfinite(conf_right) and math.isfinite(conf_left)):
        raise ValueError(f"Non-finite inputs to fuse_bilateral_weighted: ({val_right}, {val_left}, {conf_right}, {conf_left})")
    total_conf = conf_right + conf_left
    if total_conf < eps:
        return float(0.5 * (val_right + val_left))
    return float((val_right * conf_right + val_left * conf_left) / (total_conf + eps))


# -------------------------------------------------------------------------
# Provisional Geometry Quality Thresholds (M1.5-A Hardening)
# -------------------------------------------------------------------------
# Derived from Run 1 and Run 2 empirical distributions. These thresholds are
# provisional until validated across repeated multi-subject physical trials.
MIN_EYE_WIDTH_PX: float = 10.0
MIN_LID_SEPARATION_PX: float = 2.0
MIN_EAR_FOR_GAZE: float = 0.08
MAX_PLAUSIBLE_EAR: float = 0.70
PLAUSIBLE_H_MIN: float = -0.25
PLAUSIBLE_H_MAX: float = 1.25
PLAUSIBLE_V_MIN: float = -0.25
PLAUSIBLE_V_MAX: float = 1.25
PLAUSIBLE_CANTHAL_MIN: float = -0.25
PLAUSIBLE_CANTHAL_MAX: float = 0.25


class GeometryQualityReason(str, Enum):
    """Explicit reason codes for eye geometry quality rejection."""

    VALID = "VALID"
    NON_FINITE_COORDINATES = "NON_FINITE_COORDINATES"
    INSUFFICIENT_EYE_WIDTH = "INSUFFICIENT_EYE_WIDTH"
    INSUFFICIENT_LID_SEPARATION = "INSUFFICIENT_LID_SEPARATION"
    EYELID_CLOSED_OR_BLINK = "EYELID_CLOSED_OR_BLINK"
    IMPLAUSIBLE_EAR = "IMPLAUSIBLE_EAR"
    IMPLAUSIBLE_H_RATIO = "IMPLAUSIBLE_H_RATIO"
    IMPLAUSIBLE_V_RATIO = "IMPLAUSIBLE_V_RATIO"
    IMPLAUSIBLE_CANTHAL_RATIO = "IMPLAUSIBLE_CANTHAL_RATIO"


@dataclass(frozen=True)
class EyeQualityResult:
    """
    Quality audit result for a single eye's geometric features.

    Preserves raw measurements while providing explicit pass/fail classification
    and rejection reason codes.
    """

    is_valid: bool
    rejection_reasons: Tuple[GeometryQualityReason, ...] = ()
    eye_width_px: float = 0.0
    lid_separation_px: float = 0.0
    h_ratio: Optional[float] = None
    v_ratio: Optional[float] = None
    ear: Optional[float] = None
    canthal_ratio: Optional[float] = None


@dataclass(frozen=True)
class BilateralQualityResult:
    """
    Combined bilateral quality assessment.
    """

    is_valid: bool
    right_eye: EyeQualityResult
    left_eye: EyeQualityResult
    bilateral_reasons: Tuple[str, ...] = ()


def validate_eye_geometry(
    iris_center: Union[Point2D, Tuple[float, float]],
    inner_canthus: Union[Point2D, Tuple[float, float]],
    outer_canthus: Union[Point2D, Tuple[float, float]],
    upper_lid: Union[Point2D, Tuple[float, float]],
    lower_lid: Union[Point2D, Tuple[float, float]],
    is_right_eye: bool = True,
    min_eye_width_px: float = MIN_EYE_WIDTH_PX,
    min_lid_separation_px: float = MIN_LID_SEPARATION_PX,
    min_ear_for_gaze: float = MIN_EAR_FOR_GAZE,
    max_plausible_ear: float = MAX_PLAUSIBLE_EAR,
    plausible_h_bounds: Tuple[float, float] = (PLAUSIBLE_H_MIN, PLAUSIBLE_H_MAX),
    plausible_v_bounds: Tuple[float, float] = (PLAUSIBLE_V_MIN, PLAUSIBLE_V_MAX),
    plausible_canthal_bounds: Tuple[float, float] = (PLAUSIBLE_CANTHAL_MIN, PLAUSIBLE_CANTHAL_MAX),
    allow_closed_eyes: bool = False,
    eps: float = 1e-7,
) -> EyeQualityResult:
    """
    Pure deterministic validation of raw eye landmarks and derived gaze ratios.

    Invariants:
    1. Zero silent clamping: All raw ratios and distances are preserved in the result.
    2. Explicit reason codes: Every quality gate failure is reported.
    3. Mathematical undefinedness vs suspect bounds: Eyelid separation < min_lid_separation_px
       is flagged as INSUFFICIENT_LID_SEPARATION, distinguishing undefined division
       from out-of-bounds but computable gaze.
    4. Closed-eye calibration support: When allow_closed_eyes=True, low EAR (<0.18) and
       low lid separation are permitted for establishing baseline closed-eye EAR, while
       non-finite coordinates, insufficient eye width, and malformed geometry are strictly rejected.
    """
    # 1. Non-finite coordinate gate
    pts = [iris_center, inner_canthus, outer_canthus, upper_lid, lower_lid]
    if not all(is_valid_point(p) for p in pts):
        return EyeQualityResult(
            is_valid=False,
            rejection_reasons=(GeometryQualityReason.NON_FINITE_COORDINATES,),
        )

    # 2. Physical segment lengths (pixels)
    eye_width = euclidean_distance(inner_canthus, outer_canthus)
    lid_sep = euclidean_distance(upper_lid, lower_lid)

    # 3. Raw ratio computations (unclamped)
    h = norm_horizontal_gaze(iris_center, inner_canthus, outer_canthus, is_right_eye=is_right_eye, eps=eps)
    v = norm_vertical_gaze(iris_center, upper_lid, lower_lid, eps=eps)
    ear = compute_ear(upper_lid, lower_lid, outer_canthus, inner_canthus, eps=eps)
    canthal = norm_vertical_gaze_canthal_normal_experimental(
        iris_center, inner_canthus, outer_canthus, is_right_eye=is_right_eye, eps=eps
    )

    # 4. Sequential quality checks
    reasons: List[GeometryQualityReason] = []

    if eye_width < min_eye_width_px:
        reasons.append(GeometryQualityReason.INSUFFICIENT_EYE_WIDTH)

    if not allow_closed_eyes:
        if lid_sep < min_lid_separation_px:
            reasons.append(GeometryQualityReason.INSUFFICIENT_LID_SEPARATION)

        if ear < min_ear_for_gaze:
            reasons.append(GeometryQualityReason.EYELID_CLOSED_OR_BLINK)

    if ear < 0.0 or ear > max_plausible_ear:
        reasons.append(GeometryQualityReason.IMPLAUSIBLE_EAR)

    if not (plausible_h_bounds[0] <= h <= plausible_h_bounds[1]):
        reasons.append(GeometryQualityReason.IMPLAUSIBLE_H_RATIO)

    if not allow_closed_eyes and not (plausible_v_bounds[0] <= v <= plausible_v_bounds[1]):
        reasons.append(GeometryQualityReason.IMPLAUSIBLE_V_RATIO)

    if not (plausible_canthal_bounds[0] <= canthal <= plausible_canthal_bounds[1]):
        reasons.append(GeometryQualityReason.IMPLAUSIBLE_CANTHAL_RATIO)

    return EyeQualityResult(
        is_valid=(len(reasons) == 0),
        rejection_reasons=tuple(reasons),
        eye_width_px=float(eye_width),
        lid_separation_px=float(lid_sep),
        h_ratio=float(h),
        v_ratio=float(v),
        ear=float(ear),
        canthal_ratio=float(canthal),
    )


def validate_bilateral_geometry(
    right_result: EyeQualityResult,
    left_result: EyeQualityResult,
    require_both_eyes: bool = False,
) -> BilateralQualityResult:
    """
    Evaluates combined bilateral eye quality.

    Args:
        right_result: Quality result for anatomical right eye.
        left_result: Quality result for anatomical left eye.
        require_both_eyes: If True, requires both eyes to be valid. If False,
                           valid if at least one eye is valid.
    """
    reasons: List[str] = []
    if not right_result.is_valid:
        reasons.append(f"RIGHT_EYE_INVALID: {', '.join(r.value for r in right_result.rejection_reasons)}")
    if not left_result.is_valid:
        reasons.append(f"LEFT_EYE_INVALID: {', '.join(r.value for r in left_result.rejection_reasons)}")

    if require_both_eyes:
        is_valid = right_result.is_valid and left_result.is_valid
    else:
        is_valid = right_result.is_valid or left_result.is_valid

    return BilateralQualityResult(
        is_valid=is_valid,
        right_eye=right_result,
        left_eye=left_result,
        bilateral_reasons=tuple(reasons),
    )
