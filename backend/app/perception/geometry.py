"""
VoxGaze Pure Geometry & Eye Feature Normalization Module
Strictly pure mathematical functions with ZERO OpenCV, MediaPipe, or UI dependencies.
"""

import math
from typing import Optional, Tuple, Union
from backend.app.core.contracts import EyeLandmarks, GazeFeatures, Point2D


def euclidean_distance(p1: Union[Point2D, Tuple[float, float]], p2: Union[Point2D, Tuple[float, float]]) -> float:
    """Calculate 2D Euclidean distance between two points."""
    x1, y1 = (p1.x, p1.y) if isinstance(p1, Point2D) else p1
    x2, y2 = (p2.x, p2.y) if isinstance(p2, Point2D) else p2
    return math.hypot(x2 - x1, y2 - y1)


def is_valid_point(p: Union[Point2D, Tuple[float, float]]) -> bool:
    """Validate that point coordinates are finite real numbers."""
    x, y = (p.x, p.y) if isinstance(p.x if isinstance(p, Point2D) else p[0], float) else (float(p.x), float(p.y))
    return math.isfinite(x) and math.isfinite(y)


def norm_horizontal_gaze(
    iris_pt: Union[Point2D, Tuple[float, float]],
    outer_canthus: Union[Point2D, Tuple[float, float]],
    inner_canthus: Union[Point2D, Tuple[float, float]],
    eps: float = 1e-7,
) -> float:
    """
    Calculate normalized horizontal iris position along the eye axis.
    0.0 = Outer canthus, 1.0 = Inner canthus.
    For standard left eye orientation (from camera view / mirror view):
    looking left shifts iris towards outer/inner depending on eye.
    """
    ix, iy = (iris_pt.x, iris_pt.y) if isinstance(iris_pt, Point2D) else iris_pt
    ox, oy = (outer_canthus.x, outer_canthus.y) if isinstance(outer_canthus, Point2D) else outer_canthus
    nx, ny = (inner_canthus.x, inner_canthus.y) if isinstance(inner_canthus, Point2D) else inner_canthus

    width = euclidean_distance(outer_canthus, inner_canthus) + eps
    # Project iris displacement onto horizontal axis length
    h_ratio = (ix - ox) / width
    return float(h_ratio)


def norm_vertical_gaze(
    iris_pt: Union[Point2D, Tuple[float, float]],
    upper_lid: Union[Point2D, Tuple[float, float]],
    lower_lid: Union[Point2D, Tuple[float, float]],
    eps: float = 1e-7,
) -> float:
    """
    Calculate normalized vertical iris position between upper and lower eyelids.
    0.0 = Upper eyelid, 1.0 = Lower eyelid.
    """
    ix, iy = (iris_pt.x, iris_pt.y) if isinstance(iris_pt, Point2D) else iris_pt
    ux, uy = (upper_lid.x, upper_lid.y) if isinstance(upper_lid, Point2D) else upper_lid
    lx, ly = (lower_lid.x, lower_lid.y) if isinstance(lower_lid, Point2D) else lower_lid

    height = euclidean_distance(upper_lid, lower_lid) + eps
    v_ratio = (iy - uy) / height
    return float(v_ratio)


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
) -> EyeLandmarks:
    """
    Construct validated EyeLandmarks contract with computed H, V, and EAR.
    """
    for pt in [iris_center, inner_canthus, outer_canthus, upper_lid, lower_lid]:
        if not math.isfinite(pt.x) or not math.isfinite(pt.y):
            raise ValueError(f"Non-finite landmark coordinates received: ({pt.x}, {pt.y})")

    h = norm_horizontal_gaze(iris_center, outer_canthus, inner_canthus)
    v = norm_vertical_gaze(iris_center, upper_lid, lower_lid)
    ear = compute_ear(upper_lid, lower_lid, outer_canthus, inner_canthus)

    return EyeLandmarks(
        iris_center=iris_center,
        inner_canthus=inner_canthus,
        outer_canthus=outer_canthus,
        upper_lid=upper_lid,
        lower_lid=lower_lid,
        h_ratio=h,
        v_ratio=v,
        ear=ear,
    )


def extract_bilateral_features(
    timestamp_ms: int,
    left_eye: Optional[EyeLandmarks] = None,
    right_eye: Optional[EyeLandmarks] = None,
    head_pose: Optional[dict] = None,
    tracking_confidence: float = 1.0,
    is_tracking_valid: bool = True,
) -> GazeFeatures:
    """
    Bilateral contract constructor supporting left eye, right eye, and head pose.
    """
    ear_left = left_eye.ear if left_eye else 0.0
    ear_right = right_eye.ear if right_eye else 0.0

    return GazeFeatures(
        timestamp_ms=timestamp_ms,
        left_eye=left_eye,
        right_eye=right_eye,
        head_pose=head_pose,
        ear_left=ear_left,
        ear_right=ear_right,
        tracking_confidence=tracking_confidence,
        is_tracking_valid=is_tracking_valid,
    )
