"""
VoxGaze Physical Geometry Diagnostic Module
Pure mathematical and statistical helpers for M1 physical webcam diagnostic mode.
Evaluates single-eye (RIGHT, LEFT) and BILATERAL gaze signals, canthal-normal vertical projections,
inter-eye disagreement, and physical orderings. Strictly zero OpenCV, MediaPipe, or UI dependencies.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from backend.app.core.contracts import Point2D
from backend.app.perception.geometry import (
    compute_canthal_normal_distance_px,
    fuse_bilateral_mean,
    fuse_bilateral_weighted,
    norm_horizontal_gaze,
    norm_vertical_gaze,
    norm_vertical_gaze_canthal_normal_experimental,
)


@dataclass
class DiagnosticFrameData:
    """Individual frame measurements for right, left, and bilateral diagnostic tracking."""
    timestamp_ms: int
    h: float = 0.5  # Right eye H (backward compatibility)
    v: float = 0.5  # Right eye V_eyelid (backward compatibility)
    ear: float = 0.3 # Right eye EAR (backward compatibility)
    iris_x: float = 0.0
    iris_y: float = 0.0
    inner_x: float = 0.0
    inner_y: float = 0.0
    outer_x: float = 0.0
    outer_y: float = 0.0
    upper_x: float = 0.0
    upper_y: float = 0.0
    lower_x: float = 0.0
    lower_y: float = 0.0
    is_valid: bool = True
    v_eyelid: Optional[float] = None
    v_canthal: Optional[float] = None
    v_canthal_px: Optional[float] = None

    # Right eye explicit
    h_right: Optional[float] = None
    v_eyelid_right: Optional[float] = None
    v_canthal_right: Optional[float] = None
    v_canthal_right_px: Optional[float] = None
    ear_right: Optional[float] = None
    is_right_valid: bool = True

    # Left eye explicit
    h_left: Optional[float] = None
    v_eyelid_left: Optional[float] = None
    v_canthal_left: Optional[float] = None
    v_canthal_left_px: Optional[float] = None
    ear_left: Optional[float] = None
    iris_l_x: Optional[float] = None
    iris_l_y: Optional[float] = None
    inner_l_x: Optional[float] = None
    inner_l_y: Optional[float] = None
    outer_l_x: Optional[float] = None
    outer_l_y: Optional[float] = None
    upper_l_x: Optional[float] = None
    upper_l_y: Optional[float] = None
    lower_l_x: Optional[float] = None
    lower_l_y: Optional[float] = None
    is_left_valid: bool = False

    # Bilateral fusion (strictly valid only when both eyes valid)
    h_bilateral: Optional[float] = None
    v_canthal_bilateral: Optional[float] = None
    is_bilateral_valid: bool = False

    def __post_init__(self):
        # 1. Right Eye Sync
        if self.h_right is None:
            self.h_right = self.h
        if self.v_eyelid_right is None:
            self.v_eyelid_right = self.v_eyelid if self.v_eyelid is not None else self.v
        if self.ear_right is None:
            self.ear_right = self.ear

        has_right_coords = (
            math.isfinite(self.iris_x)
            and math.isfinite(self.inner_x)
            and math.isfinite(self.outer_x)
            and self.iris_x != 0.0
        )
        if self.v_canthal_right is None:
            if has_right_coords:
                self.v_canthal_right = norm_vertical_gaze_canthal_normal_experimental(
                    (self.iris_x, self.iris_y),
                    (self.inner_x, self.inner_y),
                    (self.outer_x, self.outer_y),
                    is_right_eye=True,
                )
            else:
                self.v_canthal_right = self.v_canthal if self.v_canthal is not None else 0.0

        if self.v_canthal_right_px is None:
            if has_right_coords:
                self.v_canthal_right_px = compute_canthal_normal_distance_px(
                    (self.iris_x, self.iris_y),
                    (self.inner_x, self.inner_y),
                    (self.outer_x, self.outer_y),
                    is_right_eye=True,
                )
            else:
                self.v_canthal_right_px = self.v_canthal_px if self.v_canthal_px is not None else 0.0

        self.v_canthal = self.v_canthal_right
        self.v_eyelid = self.v_eyelid_right
        self.v_canthal_px = self.v_canthal_right_px

        # 2. Left Eye Sync
        has_left_coords = (
            self.iris_l_x is not None
            and math.isfinite(self.iris_l_x)
            and self.inner_l_x is not None
            and math.isfinite(self.inner_l_x)
            and self.outer_l_x is not None
            and math.isfinite(self.outer_l_x)
        )
        if has_left_coords:
            self.is_left_valid = True
            if self.h_left is None:
                self.h_left = norm_horizontal_gaze(
                    (self.iris_l_x, self.iris_l_y),
                    (self.inner_l_x, self.inner_l_y),
                    (self.outer_l_x, self.outer_l_y),
                    is_right_eye=False,
                )
            if self.v_canthal_left is None:
                self.v_canthal_left = norm_vertical_gaze_canthal_normal_experimental(
                    (self.iris_l_x, self.iris_l_y),
                    (self.inner_l_x, self.inner_l_y),
                    (self.outer_l_x, self.outer_l_y),
                    is_right_eye=False,
                )
            if self.v_canthal_left_px is None:
                self.v_canthal_left_px = compute_canthal_normal_distance_px(
                    (self.iris_l_x, self.iris_l_y),
                    (self.inner_l_x, self.inner_l_y),
                    (self.outer_l_x, self.outer_l_y),
                    is_right_eye=False,
                )
            if self.v_eyelid_left is None and self.upper_l_x is not None and math.isfinite(self.upper_l_x):
                self.v_eyelid_left = norm_vertical_gaze(
                    (self.iris_l_x, self.iris_l_y),
                    (self.upper_l_x, self.upper_l_y),
                    (self.lower_l_x, self.lower_l_y),
                )
        else:
            if self.h_left is not None and math.isfinite(self.h_left):
                self.is_left_valid = True
            else:
                self.is_left_valid = False

        # 3. Bilateral Fusion (only performed when BOTH eye measurements are valid)
        if (
            self.is_valid
            and self.is_right_valid
            and self.is_left_valid
            and self.h_right is not None
            and self.h_left is not None
            and math.isfinite(self.h_right)
            and math.isfinite(self.h_left)
        ):
            self.is_bilateral_valid = True
            self.h_bilateral = fuse_bilateral_mean(self.h_right, self.h_left)
            if self.v_canthal_right is not None and self.v_canthal_left is not None:
                self.v_canthal_bilateral = fuse_bilateral_mean(self.v_canthal_right, self.v_canthal_left)
        else:
            self.is_bilateral_valid = False
            self.h_bilateral = None
            self.v_canthal_bilateral = None


@dataclass
class TargetDiagnosticStats:
    """Aggregated statistics for a diagnostic target fixation period."""
    target: str
    sample_count: int
    sample_count_right: int = 0
    sample_count_left: int = 0
    sample_count_bilateral: int = 0

    # Right eye
    median_h_right: float = 0.5
    mad_h_right: float = 0.0
    median_v_canthal_right: float = 0.0
    mad_v_canthal_right: float = 0.0
    median_v_eyelid_right: float = 0.5
    mad_v_eyelid_right: float = 0.0

    # Left eye
    median_h_left: float = 0.5
    mad_h_left: float = 0.0
    median_v_canthal_left: float = 0.0
    mad_v_canthal_left: float = 0.0
    median_v_eyelid_left: float = 0.5
    mad_v_eyelid_left: float = 0.0

    # Bilateral fusion (primary: arithmetic mean)
    median_h_bilateral: float = 0.5
    mad_h_bilateral: float = 0.0
    median_v_canthal_bilateral: float = 0.0
    mad_v_canthal_bilateral: float = 0.0

    # Inter-eye disagreement
    median_inter_eye_h: float = 0.0
    mad_inter_eye_h: float = 0.0
    median_inter_eye_v: float = 0.0
    mad_inter_eye_v: float = 0.0

    # Backward compatibility fields (mirroring right eye)
    median_h: float = 0.5
    mad_h: float = 0.0
    median_v: float = 0.5
    mad_v: float = 0.0
    median_ear: float = 0.3
    mad_ear: float = 0.0
    median_iris_x: float = 0.0
    median_iris_y: float = 0.0
    mad_iris_x: float = 0.0
    mad_iris_y: float = 0.0
    median_inner_x: float = 0.0
    median_inner_y: float = 0.0
    median_outer_x: float = 0.0
    median_outer_y: float = 0.0
    median_upper_x: float = 0.0
    median_upper_y: float = 0.0
    median_lower_x: float = 0.0
    median_lower_y: float = 0.0
    eye_width_px: float = 0.0
    eye_height_px: float = 0.0
    median_v_eyelid: float = 0.0
    mad_v_eyelid: float = 0.0
    median_v_canthal: float = 0.0
    mad_v_canthal: float = 0.0
    median_v_canthal_px: float = 0.0
    mad_v_canthal_px: float = 0.0

    def __post_init__(self):
        if self.sample_count_right == 0:
            self.sample_count_right = self.sample_count
        if self.median_h_right == 0.5 and self.median_h != 0.5:
            self.median_h_right = self.median_h
        if self.median_h == 0.5 and self.median_h_right != 0.5:
            self.median_h = self.median_h_right
        if self.mad_h_right == 0.0 and self.mad_h != 0.0:
            self.mad_h_right = self.mad_h
        if self.median_v_eyelid == 0.0 and self.median_v != 0.0:
            self.median_v_eyelid = self.median_v
            self.median_v_eyelid_right = self.median_v
        if self.mad_v_eyelid == 0.0 and self.mad_v != 0.0:
            self.mad_v_eyelid = self.mad_v
            self.mad_v_eyelid_right = self.mad_v

        if self.median_v_canthal_right == 0.0 and self.median_v_canthal != 0.0:
            self.median_v_canthal_right = self.median_v_canthal
        if self.median_v_canthal == 0.0 and self.median_v_canthal_right != 0.0:
            self.median_v_canthal = self.median_v_canthal_right

        has_right_coords = (
            math.isfinite(self.median_iris_x)
            and math.isfinite(self.median_inner_x)
            and math.isfinite(self.median_outer_x)
            and self.median_iris_x != 0.0
        )
        if self.median_v_canthal_right == 0.0 and self.median_v_canthal == 0.0 and has_right_coords:
            self.median_v_canthal = norm_vertical_gaze_canthal_normal_experimental(
                (self.median_iris_x, self.median_iris_y),
                (self.median_inner_x, self.median_inner_y),
                (self.median_outer_x, self.median_outer_y),
                is_right_eye=True,
            )
            self.median_v_canthal_right = self.median_v_canthal

        if self.median_v_canthal_px == 0.0 and has_right_coords:
            self.median_v_canthal_px = compute_canthal_normal_distance_px(
                (self.median_iris_x, self.median_iris_y),
                (self.median_inner_x, self.median_inner_y),
                (self.median_outer_x, self.median_outer_y),
                is_right_eye=True,
            )

        # Bilateral arithmetic mean only when both left and right are explicitly populated
        if self.sample_count_bilateral > 0 and self.median_h_bilateral == 0.5:
            self.median_h_bilateral = 0.5 * (self.median_h_right + self.median_h_left)
        if self.sample_count_bilateral > 0 and self.median_v_canthal_bilateral == 0.0:
            self.median_v_canthal_bilateral = 0.5 * (self.median_v_canthal_right + self.median_v_canthal_left)

    def to_dict(self) -> Dict[str, any]:
        return {
            "target": self.target,
            "sample_count": self.sample_count,
            "sample_count_right": self.sample_count_right,
            "sample_count_left": self.sample_count_left,
            "sample_count_bilateral": self.sample_count_bilateral,
            # Right eye
            "median_h_right": round(self.median_h_right, 4),
            "mad_h_right": round(self.mad_h_right, 4),
            "median_v_canthal_right": round(self.median_v_canthal_right, 4),
            "mad_v_canthal_right": round(self.mad_v_canthal_right, 4),
            "median_v_eyelid_right": round(self.median_v_eyelid_right, 3),
            "mad_v_eyelid_right": round(self.mad_v_eyelid_right, 3),
            # Left eye
            "median_h_left": round(self.median_h_left, 4),
            "mad_h_left": round(self.mad_h_left, 4),
            "median_v_canthal_left": round(self.median_v_canthal_left, 4),
            "mad_v_canthal_left": round(self.mad_v_canthal_left, 4),
            "median_v_eyelid_left": round(self.median_v_eyelid_left, 3),
            "mad_v_eyelid_left": round(self.mad_v_eyelid_left, 3),
            # Bilateral
            "median_h_bilateral": round(self.median_h_bilateral, 4),
            "mad_h_bilateral": round(self.mad_h_bilateral, 4),
            "median_v_canthal_bilateral": round(self.median_v_canthal_bilateral, 4),
            "mad_v_canthal_bilateral": round(self.mad_v_canthal_bilateral, 4),
            # Inter-eye disagreement
            "median_inter_eye_h": round(self.median_inter_eye_h, 4),
            "mad_inter_eye_h": round(self.mad_inter_eye_h, 4),
            "median_inter_eye_v": round(self.median_inter_eye_v, 4),
            "mad_inter_eye_v": round(self.mad_inter_eye_v, 4),
            # General / Legacy
            "median_h": round(self.median_h, 3),
            "mad_h": round(self.mad_h, 3),
            "median_v": round(self.median_v, 3),
            "mad_v": round(self.mad_v, 3),
            "median_v_eyelid": round(self.median_v_eyelid, 3),
            "mad_v_eyelid": round(self.mad_v_eyelid, 3),
            "median_v_canthal": round(self.median_v_canthal, 4),
            "mad_v_canthal": round(self.mad_v_canthal, 4),
            "median_v_canthal_px": round(self.median_v_canthal_px, 2),
            "mad_v_canthal_px": round(self.mad_v_canthal_px, 2),
            "median_ear": round(self.median_ear, 3),
            "mad_ear": round(self.mad_ear, 3),
            "median_iris_x": round(self.median_iris_x, 2),
            "median_iris_y": round(self.median_iris_y, 2),
            "mad_iris_x": round(self.mad_iris_x, 2),
            "mad_iris_y": round(self.mad_iris_y, 2),
            "median_inner_x": round(self.median_inner_x, 2),
            "median_inner_y": round(self.median_inner_y, 2),
            "median_outer_x": round(self.median_outer_x, 2),
            "median_outer_y": round(self.median_outer_y, 2),
            "median_upper_x": round(self.median_upper_x, 2),
            "median_upper_y": round(self.median_upper_y, 2),
            "median_lower_x": round(self.median_lower_x, 2),
            "median_lower_y": round(self.median_lower_y, 2),
            "eye_width_px": round(self.eye_width_px, 2),
            "eye_height_px": round(self.eye_height_px, 2),
        }


class DiagnosticTargetBuffer:
    """Collects and filters diagnostic frames for a single target."""

    def __init__(self, target_name: str, warmup_sec: float = 0.5, collection_sec: float = 2.0):
        self.target_name = target_name
        self.warmup_sec = warmup_sec
        self.collection_sec = collection_sec
        self.start_timestamp_ms: Optional[int] = None
        self.samples: List[DiagnosticFrameData] = []

    def add_frame(self, frame_data: DiagnosticFrameData) -> bool:
        if not frame_data.is_valid:
            return False

        # Validate coordinate finiteness
        for val in (
            frame_data.h, frame_data.v, frame_data.ear,
            frame_data.iris_x, frame_data.iris_y,
            frame_data.inner_x, frame_data.inner_y,
            frame_data.outer_x, frame_data.outer_y,
            frame_data.upper_x, frame_data.upper_y,
            frame_data.lower_x, frame_data.lower_y,
        ):
            if not math.isfinite(val):
                return False

        if self.start_timestamp_ms is None:
            self.start_timestamp_ms = frame_data.timestamp_ms

        elapsed_sec = (frame_data.timestamp_ms - self.start_timestamp_ms) / 1000.0

        # Discard warmup
        if elapsed_sec < self.warmup_sec:
            return False

        if elapsed_sec <= (self.warmup_sec + self.collection_sec):
            self.samples.append(frame_data)
            return True

        return False

    def is_complete(self, current_timestamp_ms: int) -> bool:
        if self.start_timestamp_ms is None:
            return False
        elapsed_sec = (current_timestamp_ms - self.start_timestamp_ms) / 1000.0
        return elapsed_sec >= (self.warmup_sec + self.collection_sec)

    def compute_stats(self) -> TargetDiagnosticStats:
        if not self.samples:
            raise ValueError(f"No valid diagnostic samples collected for target {self.target_name}")

        def med_mad(arr: np.ndarray) -> Tuple[float, float]:
            if len(arr) == 0:
                return 0.0, 0.0
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med)))
            return med, mad

        # 1. Right eye metrics
        right_samples = [s for s in self.samples if s.is_right_valid]
        cnt_r = len(right_samples)
        if cnt_r > 0:
            h_r_arr = np.array([s.h_right for s in right_samples])
            v_can_r_arr = np.array([s.v_canthal_right for s in right_samples])
            v_eye_r_arr = np.array([s.v_eyelid_right for s in right_samples])
            v_can_px_r_arr = np.array([s.v_canthal_right_px for s in right_samples])
            ear_r_arr = np.array([s.ear_right for s in right_samples])
            ix_arr = np.array([s.iris_x for s in right_samples])
            iy_arr = np.array([s.iris_y for s in right_samples])
            inn_x = np.array([s.inner_x for s in right_samples])
            inn_y = np.array([s.inner_y for s in right_samples])
            out_x = np.array([s.outer_x for s in right_samples])
            out_y = np.array([s.outer_y for s in right_samples])
            up_x = np.array([s.upper_x for s in right_samples])
            up_y = np.array([s.upper_y for s in right_samples])
            low_x = np.array([s.lower_x for s in right_samples])
            low_y = np.array([s.lower_y for s in right_samples])

            med_h_r, mad_h_r = med_mad(h_r_arr)
            med_v_can_r, mad_v_can_r = med_mad(v_can_r_arr)
            med_v_eye_r, mad_v_eye_r = med_mad(v_eye_r_arr)
            med_v_can_px_r, mad_v_can_px_r = med_mad(v_can_px_r_arr)
            med_ear_r, mad_ear_r = med_mad(ear_r_arr)
            med_ix, mad_ix = med_mad(ix_arr)
            med_iy, mad_iy = med_mad(iy_arr)

            med_inn_x = float(np.median(inn_x))
            med_inn_y = float(np.median(inn_y))
            med_out_x = float(np.median(out_x))
            med_out_y = float(np.median(out_y))
            med_up_x = float(np.median(up_x))
            med_up_y = float(np.median(up_y))
            med_low_x = float(np.median(low_x))
            med_low_y = float(np.median(low_y))

            eye_w = math.hypot(med_inn_x - med_out_x, med_inn_y - med_out_y)
            eye_h = math.hypot(med_up_x - med_low_x, med_up_y - med_low_y)
        else:
            med_h_r, mad_h_r = 0.5, 0.0
            med_v_can_r, mad_v_can_r = 0.0, 0.0
            med_v_eye_r, mad_v_eye_r = 0.5, 0.0
            med_v_can_px_r, mad_v_can_px_r = 0.0, 0.0
            med_ear_r, mad_ear_r = 0.3, 0.0
            med_ix, mad_ix = 0.0, 0.0
            med_iy, mad_iy = 0.0, 0.0
            med_inn_x, med_inn_y = 0.0, 0.0
            med_out_x, med_out_y = 0.0, 0.0
            med_up_x, med_up_y = 0.0, 0.0
            med_low_x, med_low_y = 0.0, 0.0
            eye_w, eye_h = 0.0, 0.0

        # 2. Left eye metrics
        left_samples = [s for s in self.samples if s.is_left_valid]
        cnt_l = len(left_samples)
        if cnt_l > 0:
            h_l_arr = np.array([s.h_left for s in left_samples])
            v_can_l_arr = np.array([s.v_canthal_left for s in left_samples])
            v_eye_l_arr = np.array([s.v_eyelid_left for s in left_samples if s.v_eyelid_left is not None])
            med_h_l, mad_h_l = med_mad(h_l_arr)
            med_v_can_l, mad_v_can_l = med_mad(v_can_l_arr)
            med_v_eye_l, mad_v_eye_l = med_mad(v_eye_l_arr)
        else:
            med_h_l, mad_h_l = 0.0, 0.0
            med_v_can_l, mad_v_can_l = 0.0, 0.0
            med_v_eye_l, mad_v_eye_l = 0.0, 0.0

        # 3. Bilateral metrics (only valid where both eyes valid)
        bi_samples = [s for s in self.samples if s.is_bilateral_valid]
        cnt_bi = len(bi_samples)
        if cnt_bi > 0:
            h_bi_arr = np.array([s.h_bilateral for s in bi_samples])
            v_can_bi_arr = np.array([s.v_canthal_bilateral for s in bi_samples])
            med_h_bi, mad_h_bi = med_mad(h_bi_arr)
            med_v_can_bi, mad_v_can_bi = med_mad(v_can_bi_arr)

            dis_h = np.abs(np.array([s.h_right for s in bi_samples]) - np.array([s.h_left for s in bi_samples]))
            dis_v = np.abs(np.array([s.v_canthal_right for s in bi_samples]) - np.array([s.v_canthal_left for s in bi_samples]))
            med_dis_h, mad_dis_h = med_mad(dis_h)
            med_dis_v, mad_dis_v = med_mad(dis_v)
        else:
            med_h_bi, mad_h_bi = 0.0, 0.0
            med_v_can_bi, mad_v_can_bi = 0.0, 0.0
            med_dis_h, mad_dis_h = 0.0, 0.0
            med_dis_v, mad_dis_v = 0.0, 0.0

        return TargetDiagnosticStats(
            target=self.target_name,
            sample_count=len(self.samples),
            sample_count_right=cnt_r,
            sample_count_left=cnt_l,
            sample_count_bilateral=cnt_bi,
            # Right eye
            median_h_right=med_h_r,
            mad_h_right=mad_h_r,
            median_v_canthal_right=med_v_can_r,
            mad_v_canthal_right=mad_v_can_r,
            median_v_eyelid_right=med_v_eye_r,
            mad_v_eyelid_right=mad_v_eye_r,
            # Left eye
            median_h_left=med_h_l,
            mad_h_left=mad_h_l,
            median_v_canthal_left=med_v_can_l,
            mad_v_canthal_left=mad_v_can_l,
            median_v_eyelid_left=med_v_eye_l,
            mad_v_eyelid_left=mad_v_eye_l,
            # Bilateral
            median_h_bilateral=med_h_bi,
            mad_h_bilateral=mad_h_bi,
            median_v_canthal_bilateral=med_v_can_bi,
            mad_v_canthal_bilateral=mad_v_can_bi,
            # Disagreement
            median_inter_eye_h=med_dis_h,
            mad_inter_eye_h=mad_dis_h,
            median_inter_eye_v=med_dis_v,
            mad_inter_eye_v=mad_dis_v,
            # General / Legacy
            median_h=med_h_r,
            mad_h=mad_h_r,
            median_v=med_v_eye_r,
            mad_v=mad_v_eye_r,
            median_v_eyelid=med_v_eye_r,
            mad_v_eyelid=mad_v_eye_r,
            median_v_canthal=med_v_can_r,
            mad_v_canthal=mad_v_can_r,
            median_v_canthal_px=med_v_can_px_r,
            mad_v_canthal_px=mad_v_can_px_r,
            median_ear=med_ear_r,
            mad_ear=mad_ear_r,
            median_iris_x=med_ix,
            median_iris_y=med_iy,
            mad_iris_x=mad_ix,
            mad_iris_y=mad_iy,
            median_inner_x=med_inn_x,
            median_inner_y=med_inn_y,
            median_outer_x=med_out_x,
            median_outer_y=med_out_y,
            median_upper_x=med_up_x,
            median_upper_y=med_up_y,
            median_lower_x=med_low_x,
            median_lower_y=med_low_y,
            eye_width_px=eye_w,
            eye_height_px=eye_h,
        )


def verify_physical_ordering(stats: Dict[str, TargetDiagnosticStats]) -> Dict[str, any]:
    """
    Explicitly verifies expected physical ordering independently for:
      - Right Eye: H_R, V_canthal_R
      - Left Eye: H_L, V_canthal_L
      - Bilateral: H_B, V_canthal_B
      - Eyelid: V_eyelid
    """
    req_targets = ["CENTER", "LEFT", "RIGHT", "UP", "DOWN"]
    for t in req_targets:
        if t not in stats:
            raise KeyError(f"Missing required target for physical ordering verification: {t}")

    c = stats["CENTER"]
    l = stats["LEFT"]
    r = stats["RIGHT"]
    u = stats["UP"]
    d = stats["DOWN"]

    # Right Eye Ordering
    hr_l_c = bool(l.median_h_right < c.median_h_right)
    hr_c_r = bool(c.median_h_right < r.median_h_right)
    hr_pass = bool(hr_l_c and hr_c_r)

    vr_u_c = bool(u.median_v_canthal_right < c.median_v_canthal_right)
    vr_c_d = bool(c.median_v_canthal_right < d.median_v_canthal_right)
    vr_pass = bool(vr_u_c and vr_c_d)

    # Left Eye Ordering
    hl_l_c = bool(l.median_h_left < c.median_h_left)
    hl_c_r = bool(c.median_h_left < r.median_h_left)
    hl_pass = bool(hl_l_c and hl_c_r)

    vl_u_c = bool(u.median_v_canthal_left < c.median_v_canthal_left)
    vl_c_d = bool(c.median_v_canthal_left < d.median_v_canthal_left)
    vl_pass = bool(vl_u_c and vl_c_d)

    # Bilateral Ordering
    hb_l_c = bool(l.median_h_bilateral < c.median_h_bilateral)
    hb_c_r = bool(c.median_h_bilateral < r.median_h_bilateral)
    hb_pass = bool(hb_l_c and hb_c_r)

    vb_u_c = bool(u.median_v_canthal_bilateral < c.median_v_canthal_bilateral)
    vb_c_d = bool(c.median_v_canthal_bilateral < d.median_v_canthal_bilateral)
    vb_pass = bool(vb_u_c and vb_c_d)

    # Eyelid-based V
    v_eye_u_c = bool(u.median_v_eyelid < c.median_v_eyelid)
    v_eye_c_d = bool(c.median_v_eyelid < d.median_v_eyelid)
    v_eye_pass = bool(v_eye_u_c and v_eye_c_d)

    return {
        # Right Eye
        "h_right_left_median": round(l.median_h_right, 4),
        "h_right_center_median": round(c.median_h_right, 4),
        "h_right_right_median": round(r.median_h_right, 4),
        "h_right_ordering_pass": hr_pass,
        "h_right_ordering_status": "PASS" if hr_pass else "FAIL",
        "v_canthal_right_up_median": round(u.median_v_canthal_right, 4),
        "v_canthal_right_center_median": round(c.median_v_canthal_right, 4),
        "v_canthal_right_down_median": round(d.median_v_canthal_right, 4),
        "v_canthal_right_ordering_pass": vr_pass,
        "v_canthal_right_ordering_status": "PASS" if vr_pass else "FAIL",

        # Left Eye
        "h_left_left_median": round(l.median_h_left, 4),
        "h_left_center_median": round(c.median_h_left, 4),
        "h_left_right_median": round(r.median_h_left, 4),
        "h_left_ordering_pass": hl_pass,
        "h_left_ordering_status": "PASS" if hl_pass else "FAIL",
        "v_canthal_left_up_median": round(u.median_v_canthal_left, 4),
        "v_canthal_left_center_median": round(c.median_v_canthal_left, 4),
        "v_canthal_left_down_median": round(d.median_v_canthal_left, 4),
        "v_canthal_left_ordering_pass": vl_pass,
        "v_canthal_left_ordering_status": "PASS" if vl_pass else "FAIL",

        # Bilateral
        "h_bilateral_left_median": round(l.median_h_bilateral, 4),
        "h_bilateral_center_median": round(c.median_h_bilateral, 4),
        "h_bilateral_right_median": round(r.median_h_bilateral, 4),
        "h_bilateral_ordering_pass": hb_pass,
        "h_bilateral_ordering_status": "PASS" if hb_pass else "FAIL",
        "v_canthal_bilateral_up_median": round(u.median_v_canthal_bilateral, 4),
        "v_canthal_bilateral_center_median": round(c.median_v_canthal_bilateral, 4),
        "v_canthal_bilateral_down_median": round(d.median_v_canthal_bilateral, 4),
        "v_canthal_bilateral_ordering_pass": vb_pass,
        "v_canthal_bilateral_ordering_status": "PASS" if vb_pass else "FAIL",

        # Backward compatibility
        "h_left_median": round(l.median_h, 3),
        "h_center_median": round(c.median_h, 3),
        "h_right_median": round(r.median_h, 3),
        "h_left_lt_center": hr_l_c,
        "h_center_lt_right": hr_c_r,
        "h_ordering_pass": hr_pass,
        "h_ordering_status": "PASS" if hr_pass else "FAIL",

        "v_eyelid_up_median": round(u.median_v_eyelid, 3),
        "v_eyelid_center_median": round(c.median_v_eyelid, 3),
        "v_eyelid_down_median": round(d.median_v_eyelid, 3),
        "v_eyelid_ordering_pass": v_eye_pass,
        "v_eyelid_ordering_status": "PASS" if v_eye_pass else "FAIL",

        "v_up_median": round(u.median_v, 3),
        "v_center_median": round(c.median_v, 3),
        "v_down_median": round(d.median_v, 3),
        "v_ordering_pass": v_eye_pass,
        "v_ordering_status": "PASS" if v_eye_pass else "FAIL",

        "v_canthal_up_median": round(u.median_v_canthal, 4),
        "v_canthal_center_median": round(c.median_v_canthal, 4),
        "v_canthal_down_median": round(d.median_v_canthal, 4),
        "v_canthal_up_px": round(u.median_v_canthal_px, 2),
        "v_canthal_center_px": round(c.median_v_canthal_px, 2),
        "v_canthal_down_px": round(d.median_v_canthal_px, 2),
        "v_canthal_ordering_pass": vr_pass,
        "v_canthal_ordering_status": "PASS" if vr_pass else "FAIL",

        "overall_physical_ordering_pass": bool(hr_pass and v_eye_pass),
    }


def compute_raw_coordinate_deltas(stats: Dict[str, TargetDiagnosticStats]) -> Dict[str, any]:
    """
    Computes real pixel displacements, head drift, normalized deltas, and bilateral comparisons
    between CENTER and each target.
    """
    if "CENTER" not in stats:
        raise KeyError("CENTER target statistics required to compute deltas")

    c = stats["CENTER"]
    deltas = {}

    for target in ["LEFT", "RIGHT", "UP", "DOWN"]:
        if target not in stats:
            continue
        s = stats[target]

        delta_iris_x = s.median_iris_x - c.median_iris_x
        delta_iris_y = s.median_iris_y - c.median_iris_y
        delta_head_x = s.median_inner_x - c.median_inner_x
        delta_head_y = s.median_inner_y - c.median_inner_y
        delta_ocular_x = delta_iris_x - delta_head_x
        delta_ocular_y = delta_iris_y - delta_head_y

        # Right eye separations
        if target == "LEFT":
            sep_h_r = c.median_h_right - s.median_h_right
            sep_h_l = c.median_h_left - s.median_h_left
            sep_h_bi = c.median_h_bilateral - s.median_h_bilateral
        elif target == "RIGHT":
            sep_h_r = s.median_h_right - c.median_h_right
            sep_h_l = s.median_h_left - c.median_h_left
            sep_h_bi = s.median_h_bilateral - c.median_h_bilateral
        else:
            sep_h_r = s.median_h_right - c.median_h_right
            sep_h_l = s.median_h_left - c.median_h_left
            sep_h_bi = s.median_h_bilateral - c.median_h_bilateral

        if target == "UP":
            sep_v_r = c.median_v_canthal_right - s.median_v_canthal_right
            sep_v_l = c.median_v_canthal_left - s.median_v_canthal_left
            sep_v_bi = c.median_v_canthal_bilateral - s.median_v_canthal_bilateral
        elif target == "DOWN":
            sep_v_r = s.median_v_canthal_right - c.median_v_canthal_right
            sep_v_l = s.median_v_canthal_left - c.median_v_canthal_left
            sep_v_bi = s.median_v_canthal_bilateral - c.median_v_canthal_bilateral
        else:
            sep_v_r = s.median_v_canthal_right - c.median_v_canthal_right
            sep_v_l = s.median_v_canthal_left - c.median_v_canthal_left
            sep_v_bi = s.median_v_canthal_bilateral - c.median_v_canthal_bilateral

        deltas[target] = {
            "delta_iris_x_px": round(delta_iris_x, 2),
            "delta_iris_y_px": round(delta_iris_y, 2),
            "delta_iris_total_displacement_px": round(math.hypot(delta_iris_x, delta_iris_y), 2),
            "head_inner_canthus_drift_x_px": round(delta_head_x, 2),
            "head_inner_canthus_drift_y_px": round(delta_head_y, 2),
            "head_inner_canthus_drift_px": round(math.hypot(delta_head_x, delta_head_y), 2),
            "head_outer_canthus_drift_px": round(
                math.hypot(s.median_outer_x - c.median_outer_x, s.median_outer_y - c.median_outer_y), 2
            ),
            "relative_ocular_displacement_x_px": round(delta_ocular_x, 2),
            "relative_ocular_displacement_y_px": round(delta_ocular_y, 2),
            # Separations
            "separation_h_right": round(sep_h_r, 4),
            "separation_h_left": round(sep_h_l, 4),
            "separation_h_bilateral": round(sep_h_bi, 4),
            "separation_v_right": round(sep_v_r, 4),
            "separation_v_left": round(sep_v_l, 4),
            "separation_v_bilateral": round(sep_v_bi, 4),
            # Inter-eye disagreement
            "inter_eye_h_disagreement": round(s.median_inter_eye_h, 4),
            "inter_eye_v_disagreement": round(s.median_inter_eye_v, 4),
            # Legacy
            "delta_h": round(s.median_h - c.median_h, 3),
            "delta_v": round(s.median_v - c.median_v, 3),
            "delta_v_eyelid": round(s.median_v_eyelid - c.median_v_eyelid, 3),
            "delta_v_canthal": round(s.median_v_canthal - c.median_v_canthal, 4),
            "delta_v_canthal_px": round(s.median_v_canthal_px - c.median_v_canthal_px, 2),
            "canthal_separation_norm": round(sep_v_r, 4),
            "canthal_separation_px": round(c.median_v_canthal_px - s.median_v_canthal_px if target == "UP" else s.median_v_canthal_px - c.median_v_canthal_px, 2),
            "eye_width_px": round(s.eye_width_px, 1),
            "eye_height_px": round(s.eye_height_px, 1),
        }

    return deltas


def format_diagnostic_summary(
    stats: Dict[str, TargetDiagnosticStats],
    ordering: Dict[str, any],
    deltas: Dict[str, any],
) -> str:
    """Builds an ASCII summary table of the physical diagnostic run."""
    lines = []
    lines.append("\n" + "=" * 106)
    lines.append("  VOXGAZE M1 PHYSICAL GEOMETRY DIAGNOSTIC REPORT (BILATERAL COMPARISON)")
    lines.append("=" * 106)

    # 1. Bilateral vs Single-Eye Feature Distribution
    lines.append(
        f"  {'TARGET':<8} {'N_BI':<5} {'H_RIGHT':<9} {'H_LEFT':<9} {'H_BILAT':<9} "
        f"{'V_RIGHT':<10} {'V_LEFT':<10} {'V_BILAT':<10} {'DIS_H':<8} {'DIS_V':<8}"
    )
    lines.append("  " + "-" * 102)
    for tgt in ["CENTER", "LEFT", "RIGHT", "UP", "DOWN"]:
        if tgt in stats:
            s = stats[tgt]
            lines.append(
                f"  {tgt:<8} {s.sample_count_bilateral:<5} {s.median_h_right:<9.4f} {s.median_h_left:<9.4f} "
                f"{s.median_h_bilateral:<9.4f} {s.median_v_canthal_right:<10.4f} {s.median_v_canthal_left:<10.4f} "
                f"{s.median_v_canthal_bilateral:<10.4f} {s.median_inter_eye_h:<8.4f} {s.median_inter_eye_v:<8.4f}"
            )
    lines.append("  " + "-" * 102)

    # 2. Variance Comparison (MAD)
    lines.append("\n  [VARIANCE COMPARISON: MAD (LOWER IS BETTER)]")
    lines.append(
        f"  {'TARGET':<8} {'MAD_H_R':<10} {'MAD_H_L':<10} {'MAD_H_BI':<10} "
        f"{'MAD_V_R':<10} {'MAD_V_L':<10} {'MAD_V_BI':<10}"
    )
    lines.append("  " + "-" * 72)
    for tgt in ["CENTER", "LEFT", "RIGHT", "UP", "DOWN"]:
        if tgt in stats:
            s = stats[tgt]
            lines.append(
                f"  {tgt:<8} {s.mad_h_right:<10.4f} {s.mad_h_left:<10.4f} {s.mad_h_bilateral:<10.4f} "
                f"{s.mad_v_canthal_right:<10.4f} {s.mad_v_canthal_left:<10.4f} {s.mad_v_canthal_bilateral:<10.4f}"
            )
    lines.append("  " + "-" * 72)

    # 3. Directional Separation Comparison against M1 Gate (0.060)
    lines.append("\n  [DIRECTIONAL SEPARATION VS UNCHANGED M1 GATE (>= 0.060)]")
    lines.append(
        f"  {'DIRECTION':<12} {'SEP_RIGHT':<12} {'SEP_LEFT':<12} {'SEP_BILATERAL':<15} {'GATE':<8} {'STATUS':<8}"
    )
    lines.append("  " + "-" * 72)
    for d_name, tgt, axis in [
        ("LEFT (H)", "LEFT", "h"),
        ("RIGHT (H)", "RIGHT", "h"),
        ("UP (V)", "UP", "v"),
        ("DOWN (V)", "DOWN", "v"),
    ]:
        if tgt in deltas:
            d = deltas[tgt]
            sr = d[f"separation_{axis}_right"]
            sl = d[f"separation_{axis}_left"]
            sb = d[f"separation_{axis}_bilateral"]
            stat = "PASS" if sb >= 0.060 else "FAIL"
            lines.append(f"  {d_name:<12} {sr:<+12.4f} {sl:<+12.4f} {sb:<+15.4f} {'>=0.060':<8} [{stat}]")
    lines.append("  " + "-" * 72)

    # 4. Physical Ordering Verification
    lines.append("\n  [PHYSICAL ORDERING VERIFICATION]")
    lines.append(
        f"  H Right     (LEFT < CENTER < RIGHT) : "
        f"{ordering['h_right_left_median']:.4f} < {ordering['h_right_center_median']:.4f} < {ordering['h_right_right_median']:.4f} "
        f"-> [{ordering['h_right_ordering_status']}]"
    )
    lines.append(
        f"  H Left      (LEFT < CENTER < RIGHT) : "
        f"{ordering['h_left_left_median']:.4f} < {ordering['h_left_center_median']:.4f} < {ordering['h_left_right_median']:.4f} "
        f"-> [{ordering['h_left_ordering_status']}]"
    )
    lines.append(
        f"  H Bilateral (LEFT < CENTER < RIGHT) : "
        f"{ordering['h_bilateral_left_median']:.4f} < {ordering['h_bilateral_center_median']:.4f} < {ordering['h_bilateral_right_median']:.4f} "
        f"-> [{ordering['h_bilateral_ordering_status']}]"
    )
    lines.append(
        f"  V Right     (UP < CENTER < DOWN)    : "
        f"{ordering['v_canthal_right_up_median']:.4f} < {ordering['v_canthal_right_center_median']:.4f} < {ordering['v_canthal_right_down_median']:.4f} "
        f"-> [{ordering['v_canthal_right_ordering_status']}]"
    )
    lines.append(
        f"  V Left      (UP < CENTER < DOWN)    : "
        f"{ordering['v_canthal_left_up_median']:.4f} < {ordering['v_canthal_left_center_median']:.4f} < {ordering['v_canthal_left_down_median']:.4f} "
        f"-> [{ordering['v_canthal_left_ordering_status']}]"
    )
    lines.append(
        f"  V Bilateral (UP < CENTER < DOWN)    : "
        f"{ordering['v_canthal_bilateral_up_median']:.4f} < {ordering['v_canthal_bilateral_center_median']:.4f} < {ordering['v_canthal_bilateral_down_median']:.4f} "
        f"-> [{ordering['v_canthal_bilateral_ordering_status']}]"
    )
    lines.append(
        f"  V Eyelid    (UP < CENTER < DOWN)    : "
        f"{ordering['v_eyelid_up_median']:.3f} < {ordering['v_eyelid_center_median']:.3f} < {ordering['v_eyelid_down_median']:.3f} "
        f"-> [{ordering['v_eyelid_ordering_status']} - Comparison Only]"
    )

    # 5. Tracking Availability & Sample Integrity
    lines.append("\n  [TRACKING AVAILABILITY PER TARGET]")
    total_r = sum(s.sample_count_right for s in stats.values())
    total_l = sum(s.sample_count_left for s in stats.values())
    total_bi = sum(s.sample_count_bilateral for s in stats.values())
    for tgt in ["CENTER", "LEFT", "RIGHT", "UP", "DOWN"]:
        if tgt in stats:
            s = stats[tgt]
            lines.append(
                f"  {tgt:<8} : Right={s.sample_count_right}, Left={s.sample_count_left}, "
                f"Bilateral={s.sample_count_bilateral} (Inter-eye Disagreement: H={s.median_inter_eye_h:.4f}, V={s.median_inter_eye_v:.4f})"
            )
    lines.append(f"  Total Valid Frames: Right={total_r}, Left={total_l}, Bilateral={total_bi}")

    lines.append("\n  [HEAD-FIXED CONTROL TRIAL INSTRUCTIONS]")
    lines.append("  1. Rest your chin on a stable support / surface.")
    lines.append("  2. Keep your face strictly stationary.")
    lines.append("  3. Move ONLY your eyes to the target.")
    lines.append("  4. Do not turn or pitch your head.")
    lines.append("=" * 106 + "\n")
    return "\n".join(lines)


def export_diagnostic_artifact(
    export_path: Path,
    stats: Dict[str, TargetDiagnosticStats],
    ordering: Dict[str, any],
    deltas: Dict[str, any],
    camera_metadata: Optional[Dict[str, any]] = None,
) -> Path:
    """Exports full physical geometry diagnostic report to JSON."""
    report = {
        "report_type": "PHYSICAL_GEOMETRY_DIAGNOSTIC",
        "active_eye": "BILATERAL_COMPARISON (Right + Left Anatomical)",
        "camera_metadata": camera_metadata or {},
        "target_stats": {k: v.to_dict() for k, v in stats.items()},
        "physical_ordering": ordering,
        "raw_deltas_from_center": deltas,
    }
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return export_path
