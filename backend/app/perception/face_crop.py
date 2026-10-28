"""
VoxGaze Perception: Deterministic Face Crop Extraction for Auxiliary Estimators

Extracts a standardized, square-expanded face bounding box from MediaPipe FaceLandmarker
points suitable for auxiliary estimators like MobileOne S0.

Policy Specifications:
1. Landmark Extent: Bounding box derived from canonical 468 facial mesh landmarks.
2. Square Expansion: Max of width and height expanded by 20% (scale factor = 1.20) centered on face center.
3. Clamping: Crop boundaries strictly clamped to [0, W] and [0, H].
4. Minimum Dimension Gate: Rejects crops smaller than 64x64 pixels.
5. Experimental Note: This MediaPipe-derived crop is an experimental approximation of
   the original RetinaFace / Gaze360 training distribution.
"""

from dataclasses import dataclass
import math
from typing import Any, List, Optional, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class FaceCropResult:
    """Result of deterministic face crop extraction."""

    crop_bgr: Optional[np.ndarray]
    bbox_clamped: Optional[Tuple[int, int, int, int]]  # (x1, y1, x2, y2)
    bbox_raw: Optional[Tuple[int, int, int, int]]      # (x_min, y_min, x_max, y_max)
    is_valid: bool
    rejection_reason: Optional[str] = None


def extract_face_crop(
    frame_bgr: np.ndarray,
    landmarks: Any,
    padding_scale: float = 1.20,
    min_size_px: int = 64,
) -> FaceCropResult:
    """
    Extracts a padded, squared face crop from a raw BGR frame using MediaPipe landmarks.

    Args:
        frame_bgr: Full raw BGR image array from camera (H, W, 3).
        landmarks: MediaPipe NormalizedLandmarkList, sequence of NormalizedLandmark objects,
                   or an (N, 2) array of coordinates.
        padding_scale: Expansion multiplier around the face extent (default: 1.20 = 20% expansion).
        min_size_px: Minimum width and height required for a valid crop (default: 64 px).

    Returns:
        FaceCropResult containing the cropped BGR array (if valid) and bounding-box coordinates.
    """
    # 1. Validate input frame
    if frame_bgr is None or not isinstance(frame_bgr, np.ndarray):
        return FaceCropResult(
            crop_bgr=None, bbox_clamped=None, bbox_raw=None, is_valid=False,
            rejection_reason="Invalid or None frame array received."
        )

    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or frame_bgr.size == 0:
        return FaceCropResult(
            crop_bgr=None, bbox_clamped=None, bbox_raw=None, is_valid=False,
            rejection_reason=f"Frame must be non-empty 3-channel image (shape: {frame_bgr.shape if hasattr(frame_bgr, 'shape') else None})."
        )

    img_h, img_w = frame_bgr.shape[:2]
    if img_h < min_size_px or img_w < min_size_px:
        return FaceCropResult(
            crop_bgr=None, bbox_clamped=None, bbox_raw=None, is_valid=False,
            rejection_reason=f"Frame dimensions ({img_w}x{img_h}) are smaller than minimum crop gate ({min_size_px}px)."
        )

    # 2. Extract landmark coordinate list
    if landmarks is None:
        return FaceCropResult(
            crop_bgr=None, bbox_clamped=None, bbox_raw=None, is_valid=False,
            rejection_reason="No landmarks provided."
        )

    xs: List[float] = []
    ys: List[float] = []

    try:
        # Support both NormalizedLandmark objects (p.x, p.y) and coordinate tuples/arrays ([x, y])
        for p in landmarks:
            if hasattr(p, "x") and hasattr(p, "y"):
                px = float(p.x) * img_w if (0.0 <= p.x <= 1.0) else float(p.x)
                py = float(p.y) * img_h if (0.0 <= p.y <= 1.0) else float(p.y)
            elif isinstance(p, (tuple, list, np.ndarray)) and len(p) >= 2:
                px = float(p[0]) * img_w if (0.0 <= p[0] <= 1.0) else float(p[0])
                py = float(p[1]) * img_h if (0.0 <= p[1] <= 1.0) else float(p[1])
            else:
                continue

            if math.isfinite(px) and math.isfinite(py):
                xs.append(px)
                ys.append(py)

    except Exception as exc:
        return FaceCropResult(
            crop_bgr=None, bbox_clamped=None, bbox_raw=None, is_valid=False,
            rejection_reason=f"Malformed landmark list structure: {exc}"
        )

    # Require minimum face landmark density (at least 20 canonical points)
    if len(xs) < 20:
        return FaceCropResult(
            crop_bgr=None, bbox_clamped=None, bbox_raw=None, is_valid=False,
            rejection_reason=f"Insufficient valid landmark points detected: {len(xs)} < 20"
        )

    # 3. Compute raw landmark extents
    raw_x_min = float(min(xs))
    raw_x_max = float(max(xs))
    raw_y_min = float(min(ys))
    raw_y_max = float(max(ys))

    w_raw = raw_x_max - raw_x_min
    h_raw = raw_y_max - raw_y_min

    if w_raw <= 0 or h_raw <= 0:
        return FaceCropResult(
            crop_bgr=None, bbox_clamped=None, bbox_raw=None, is_valid=False,
            rejection_reason="Degenerate face landmark bounding box (zero width or height)."
        )

    bbox_raw = (int(round(raw_x_min)), int(round(raw_y_min)), int(round(raw_x_max)), int(round(raw_y_max)))

    # 4. Deterministic square expansion
    center_x = (raw_x_min + raw_x_max) / 2.0
    center_y = (raw_y_min + raw_y_max) / 2.0

    max_side = max(w_raw, h_raw)
    padded_side = max_side * padding_scale
    half_side = padded_side / 2.0

    unclamped_x1 = int(round(center_x - half_side))
    unclamped_y1 = int(round(center_y - half_side))
    unclamped_x2 = int(round(center_x + half_side))
    unclamped_y2 = int(round(center_y + half_side))

    # 5. Clamp coordinates to frame boundaries
    x1 = max(0, unclamped_x1)
    y1 = max(0, unclamped_y1)
    x2 = min(img_w, unclamped_x2)
    y2 = min(img_h, unclamped_y2)

    crop_w = x2 - x1
    crop_h = y2 - y1

    # 6. Validate resulting crop dimensions against minimum gate
    if crop_w < min_size_px or crop_h < min_size_px:
        return FaceCropResult(
            crop_bgr=None, bbox_clamped=(x1, y1, x2, y2), bbox_raw=bbox_raw, is_valid=False,
            rejection_reason=f"Padded face crop ({crop_w}x{crop_h}) is below minimum size gate ({min_size_px}x{min_size_px})."
        )

    # 7. Extract sub-array (contiguous copy to prevent memory leakage)
    crop = np.ascontiguousarray(frame_bgr[y1:y2, x1:x2])

    if crop.size == 0 or crop.shape[0] < min_size_px or crop.shape[1] < min_size_px:
        return FaceCropResult(
            crop_bgr=None, bbox_clamped=(x1, y1, x2, y2), bbox_raw=bbox_raw, is_valid=False,
            rejection_reason="Empty crop slice generated after clamping."
        )

    return FaceCropResult(
        crop_bgr=crop,
        bbox_clamped=(x1, y1, x2, y2),
        bbox_raw=bbox_raw,
        is_valid=True,
        rejection_reason=None,
    )
