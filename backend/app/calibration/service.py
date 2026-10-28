"""
VoxGaze Calibration Service
Handles 5-point + blink calibration, robust outlier rejection (Median + MAD),
quality validation, threshold generation, and profile compilation.
Supports active eye baseline selection (RIGHT_EYE_BASELINE by default).
"""

from datetime import datetime, timezone
import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from backend.app.core.contracts import (
    CalibrationProfile,
    CalibrationStats,
    CalibrationThresholds,
    DirectionThreshold,
    EyeLandmarks,
    GazeFeatures,
)
from backend.app.perception.geometry import (
    GeometryQualityReason,
    MAX_PLAUSIBLE_EAR,
    MIN_EAR_FOR_GAZE,
    PLAUSIBLE_H_MAX,
    PLAUSIBLE_H_MIN,
    PLAUSIBLE_V_MAX,
    PLAUSIBLE_V_MIN,
    validate_eye_geometry,
)


class CalibrationTargetBuffer:
    """Collects and filters gaze samples for a specific calibration target."""

    def __init__(
        self,
        target_name: str,
        warmup_sec: float = 0.5,
        collection_sec: float = 1.5,
        active_eye: str = "right",
    ):
        self.target_name = target_name
        self.warmup_sec = warmup_sec
        self.collection_sec = collection_sec
        self.active_eye = active_eye
        self.start_timestamp_ms: Optional[int] = None
        self.h_samples: List[float] = []
        self.v_samples: List[float] = []
        self.ear_samples: List[float] = []

    def add_sample(self, features: GazeFeatures) -> bool:
        """
        Add sample if tracking is valid and warmup has passed.
        Extracts active eye (RIGHT_EYE_BASELINE by default, with left fallback for fixtures).
        Returns True if sample was collected in the active collection window.
        """
        if not features.is_tracking_valid:
            return False

        eye: Optional[EyeLandmarks] = features.right_eye if self.active_eye == "right" else features.left_eye
        active_valid = features.is_right_eye_valid if self.active_eye == "right" else features.is_left_eye_valid
        if eye is None:
            eye = features.left_eye if self.active_eye == "right" else features.right_eye
            active_valid = features.is_left_eye_valid if self.active_eye == "right" else features.is_right_eye_valid

        if eye is None:
            return False

        # Reject if active eye is explicitly marked invalid
        if active_valid is False:
            return False

        if not (math.isfinite(eye.h_ratio) and math.isfinite(eye.v_ratio) and math.isfinite(eye.ear)):
            return False

        is_right = (self.active_eye == "right")
        if self.target_name != "CLOSED":
            # For open gaze targets: reject blinks, insufficient lid separation, and out-of-band ratios
            if eye.ear < MIN_EAR_FOR_GAZE or eye.ear > MAX_PLAUSIBLE_EAR:
                return False
            if not (PLAUSIBLE_H_MIN <= eye.h_ratio <= PLAUSIBLE_H_MAX):
                return False
            if not (PLAUSIBLE_V_MIN <= eye.v_ratio <= PLAUSIBLE_V_MAX):
                return False

            # Validate physical landmarks if available
            try:
                quality = validate_eye_geometry(
                    iris_center=eye.iris_center,
                    inner_canthus=eye.inner_canthus,
                    outer_canthus=eye.outer_canthus,
                    upper_lid=eye.upper_lid,
                    lower_lid=eye.lower_lid,
                    is_right_eye=is_right,
                )
                critical_failures = {
                    GeometryQualityReason.NON_FINITE_COORDINATES,
                    GeometryQualityReason.INSUFFICIENT_EYE_WIDTH,
                    GeometryQualityReason.INSUFFICIENT_LID_SEPARATION,
                    GeometryQualityReason.EYELID_CLOSED_OR_BLINK,
                    GeometryQualityReason.IMPLAUSIBLE_EAR,
                }
                if any(r in critical_failures for r in quality.rejection_reasons):
                    return False
            except Exception:
                return False
        else:
            # For CLOSED target: eyelids are expected to be closed, but coordinates and eye width must be valid
            if eye.ear > MAX_PLAUSIBLE_EAR:
                return False
            try:
                quality = validate_eye_geometry(
                    iris_center=eye.iris_center,
                    inner_canthus=eye.inner_canthus,
                    outer_canthus=eye.outer_canthus,
                    upper_lid=eye.upper_lid,
                    lower_lid=eye.lower_lid,
                    is_right_eye=is_right,
                )
                if GeometryQualityReason.NON_FINITE_COORDINATES in quality.rejection_reasons:
                    return False
                if GeometryQualityReason.INSUFFICIENT_EYE_WIDTH in quality.rejection_reasons:
                    return False
            except Exception:
                return False

        if self.start_timestamp_ms is None:
            self.start_timestamp_ms = features.timestamp_ms

        elapsed_sec = (features.timestamp_ms - self.start_timestamp_ms) / 1000.0

        # Discard warm-up period
        if elapsed_sec < self.warmup_sec:
            return False

        # Collect within collection window
        if elapsed_sec <= (self.warmup_sec + self.collection_sec):
            self.h_samples.append(eye.h_ratio)
            self.v_samples.append(eye.v_ratio)
            self.ear_samples.append(eye.ear)
            return True

        return False

    def is_complete(self, current_timestamp_ms: int) -> bool:
        if self.start_timestamp_ms is None:
            return False
        elapsed_sec = (current_timestamp_ms - self.start_timestamp_ms) / 1000.0
        return elapsed_sec >= (self.warmup_sec + self.collection_sec)

    def compute_stats(self) -> CalibrationStats:
        """Calculate robust statistics (Median, MAD, min, max) for collected samples."""
        if len(self.h_samples) == 0:
            raise ValueError(f"No samples collected for target {self.target_name}")

        h_arr = np.array(self.h_samples)
        v_arr = np.array(self.v_samples)
        ear_arr = np.array(self.ear_samples)

        med_h = float(np.median(h_arr))
        med_v = float(np.median(v_arr))
        med_ear = float(np.median(ear_arr))

        # Median Absolute Deviation (MAD)
        mad_h = float(np.median(np.abs(h_arr - med_h)))
        mad_v = float(np.median(np.abs(v_arr - med_v)))
        mad_ear = float(np.median(np.abs(ear_arr - med_ear)))

        return CalibrationStats(
            sample_count=len(self.h_samples),
            median_h=med_h,
            median_v=med_v,
            mad_h=mad_h,
            mad_v=mad_v,
            min_h=float(np.min(h_arr)),
            max_h=float(np.max(h_arr)),
            min_v=float(np.min(v_arr)),
            max_v=float(np.max(v_arr)),
            median_ear=med_ear,
            mad_ear=mad_ear,
        )


class CalibrationService:
    """Manages the full 6-phase calibration procedure and threshold derivation."""

    MIN_SAMPLES_PER_TARGET = 20
    MAX_ALLOWED_MAD = 0.08
    MIN_DIRECTIONAL_SEPARATION = 0.06
    MIN_BLINK_SEPARATION = 0.05

    def __init__(
        self,
        user_id: str = "default_user",
        camera_index: int = 0,
        camera_resolution: Tuple[int, int] = (1280, 720),
        active_eye: str = "right",
    ):
        self.user_id = user_id
        self.camera_index = camera_index
        self.camera_resolution = camera_resolution
        self.active_eye = active_eye
        self.buffers: Dict[str, CalibrationTargetBuffer] = {
            t: CalibrationTargetBuffer(t, active_eye=active_eye)
            for t in ["CENTER", "LEFT", "RIGHT", "UP", "DOWN", "CLOSED"]
        }

    def feed_sample(self, target: str, features: GazeFeatures) -> bool:
        if target not in self.buffers:
            raise KeyError(f"Unknown calibration target: {target}")
        return self.buffers[target].add_sample(features)

    def validate_quality(self, stats: Dict[str, CalibrationStats]) -> Tuple[bool, List[str]]:
        """
        Audit all target stats against stability and separation criteria.
        Returns (is_passed, list_of_reasons_or_warnings).
        CLOSED target is validated using EAR and sample quality ONLY (never iris H/V MAD).
        """
        errors = []

        # 1. Sample count and MAD checks
        for target, s in stats.items():
            if s.sample_count < self.MIN_SAMPLES_PER_TARGET:
                errors.append(f"Insufficient samples for {target}: {s.sample_count} < {self.MIN_SAMPLES_PER_TARGET}")

            # Only evaluate H/V jitter for open gaze directions, NOT closed eyes
            if target != "CLOSED":
                if s.mad_h > self.MAX_ALLOWED_MAD:
                    errors.append(f"High horizontal jitter (MAD={s.mad_h:.3f}) for {target}")
                if s.mad_v > self.MAX_ALLOWED_MAD:
                    errors.append(f"High vertical jitter (MAD={s.mad_v:.3f}) for {target}")

        center = stats.get("CENTER")
        left = stats.get("LEFT")
        right = stats.get("RIGHT")
        up = stats.get("UP")
        down = stats.get("DOWN")
        closed = stats.get("CLOSED")

        if not (center and left and right and up and down and closed):
            errors.append("Missing one or more required calibration targets")
            return False, errors

        # 2. Directional separation checks
        # Left should have smaller H ratio than Center (H ≈ 0 for left)
        if (center.median_h - left.median_h) < self.MIN_DIRECTIONAL_SEPARATION:
            errors.append(
                f"Weak LEFT separation: center_h ({center.median_h:.3f}) - left_h ({left.median_h:.3f}) < {self.MIN_DIRECTIONAL_SEPARATION}"
            )

        # Right should have larger H ratio than Center (H ≈ 1 for right)
        if (right.median_h - center.median_h) < self.MIN_DIRECTIONAL_SEPARATION:
            errors.append(
                f"Weak RIGHT separation: right_h ({right.median_h:.3f}) - center_h ({center.median_h:.3f}) < {self.MIN_DIRECTIONAL_SEPARATION}"
            )

        # Up should have smaller V ratio than Center (V ≈ 0 for up)
        if (center.median_v - up.median_v) < self.MIN_DIRECTIONAL_SEPARATION:
            errors.append(
                f"Weak UP separation: center_v ({center.median_v:.3f}) - up_v ({up.median_v:.3f}) < {self.MIN_DIRECTIONAL_SEPARATION}"
            )

        # Down should have larger V ratio than Center (V ≈ 1 for down)
        if (down.median_v - center.median_v) < self.MIN_DIRECTIONAL_SEPARATION:
            errors.append(
                f"Weak DOWN separation: down_v ({down.median_v:.3f}) - center_v ({center.median_v:.3f}) < {self.MIN_DIRECTIONAL_SEPARATION}"
            )

        # 3. Blink EAR separation
        open_ear_med = center.median_ear if center.median_ear is not None else float(np.median(self.buffers["CENTER"].ear_samples))
        closed_ear_med = closed.median_ear if closed.median_ear is not None else float(np.median(self.buffers["CLOSED"].ear_samples))

        if (open_ear_med - closed_ear_med) < self.MIN_BLINK_SEPARATION:
            errors.append(
                f"Weak blink EAR separation: open ({open_ear_med:.3f}) - closed ({closed_ear_med:.3f}) < {self.MIN_BLINK_SEPARATION}"
            )

        return len(errors) == 0, errors

    def get_calibration_diagnostics(self, stats: Dict[str, CalibrationStats]) -> Dict[str, any]:
        """
        Format raw calibration statistics and separation metrics for logging/artifacts.
        """
        center = stats.get("CENTER")
        diagnostics = {}
        for target, s in stats.items():
            diag = {
                "target": target,
                "samples": s.sample_count,
                "median_h": round(s.median_h, 3),
                "mad_h": round(s.mad_h, 3),
                "median_v": round(s.median_v, 3),
                "mad_v": round(s.mad_v, 3),
                "median_ear": round(s.median_ear, 3) if s.median_ear is not None else None,
                "mad_ear": round(s.mad_ear, 3) if s.mad_ear is not None else None,
            }
            if center and target != "CENTER":
                if target in ("LEFT", "RIGHT"):
                    diag["separation_from_center"] = round(s.median_h - center.median_h, 3)
                elif target in ("UP", "DOWN"):
                    diag["separation_from_center"] = round(s.median_v - center.median_v, 3)
                elif target == "CLOSED":
                    open_ear = center.median_ear if center.median_ear is not None else 0.30
                    closed_ear = s.median_ear if s.median_ear is not None else 0.10
                    diag["separation_from_center"] = round(open_ear - closed_ear, 3)
            else:
                diag["separation_from_center"] = 0.0
            diagnostics[target] = diag
        return diagnostics

    def compile_profile(self) -> CalibrationProfile:
        """Calculate statistics, build thresholds, and generate CalibrationProfile."""
        stats = {target: buf.compute_stats() for target, buf in self.buffers.items()}
        is_valid, errors = self.validate_quality(stats)

        if not is_valid:
            raise ValueError(f"Calibration quality validation failed: {'; '.join(errors)}")

        center = stats["CENTER"]
        left = stats["LEFT"]
        right = stats["RIGHT"]
        up = stats["UP"]
        down = stats["DOWN"]
        closed = stats["CLOSED"]

        open_ear_med = center.median_ear if center.median_ear is not None else float(np.median(self.buffers["CENTER"].ear_samples))
        closed_ear_med = closed.median_ear if closed.median_ear is not None else float(np.median(self.buffers["CLOSED"].ear_samples))

        # Compute dual hysteresis thresholds (60% enter, 35% exit toward center)
        left_delta = center.median_h - left.median_h
        right_delta = right.median_h - center.median_h
        up_delta = center.median_v - up.median_v
        down_delta = down.median_v - center.median_v
        blink_delta = open_ear_med - closed_ear_med

        thresholds = CalibrationThresholds(
            left=DirectionThreshold(
                enter=center.median_h - 0.60 * left_delta,
                exit=center.median_h - 0.35 * left_delta,
            ),
            right=DirectionThreshold(
                enter=center.median_h + 0.60 * right_delta,
                exit=center.median_h + 0.35 * right_delta,
            ),
            up=DirectionThreshold(
                enter=center.median_v - 0.60 * up_delta,
                exit=center.median_v - 0.35 * up_delta,
            ),
            down=DirectionThreshold(
                enter=center.median_v + 0.60 * down_delta,
                exit=center.median_v + 0.35 * down_delta,
            ),
            blink_closed_ear=closed_ear_med + 0.40 * blink_delta,
            blink_open_ear=open_ear_med - 0.25 * blink_delta,
        )

        profile = CalibrationProfile(
            profile_version=1,
            algorithm_version="m1.0",
            user_id=self.user_id,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            camera_index=self.camera_index,
            camera_resolution=self.camera_resolution,
            stats=stats,
            thresholds=thresholds,
            is_valid=True,
        )
        return profile
