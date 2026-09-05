"""
VoxGaze Calibration Service
Handles 5-point + blink calibration, robust outlier rejection (Median + MAD),
quality validation, threshold generation, and profile compilation.
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


class CalibrationTargetBuffer:
    """Collects and filters gaze samples for a specific calibration target."""

    def __init__(self, target_name: str, warmup_sec: float = 0.5, collection_sec: float = 1.5):
        self.target_name = target_name
        self.warmup_sec = warmup_sec
        self.collection_sec = collection_sec
        self.start_timestamp_ms: Optional[int] = None
        self.h_samples: List[float] = []
        self.v_samples: List[float] = []
        self.ear_samples: List[float] = []

    def add_sample(self, features: GazeFeatures) -> bool:
        """
        Add sample if tracking is valid and warmup has passed.
        Returns True if sample was collected in the active collection window.
        """
        if not features.is_tracking_valid or features.left_eye is None:
            return False

        eye = features.left_eye
        if not (math.isfinite(eye.h_ratio) and math.isfinite(eye.v_ratio) and math.isfinite(eye.ear)):
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

        med_h = float(np.median(h_arr))
        med_v = float(np.median(v_arr))

        # Median Absolute Deviation (MAD)
        mad_h = float(np.median(np.abs(h_arr - med_h)))
        mad_v = float(np.median(np.abs(v_arr - med_v)))

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
        )


class CalibrationService:
    """Manages the full 6-phase calibration procedure and threshold derivation."""

    MIN_SAMPLES_PER_TARGET = 20
    MAX_ALLOWED_MAD = 0.08
    MIN_DIRECTIONAL_SEPARATION = 0.06
    MIN_BLINK_SEPARATION = 0.05

    def __init__(self, user_id: str = "default_user", camera_index: int = 0, camera_resolution: Tuple[int, int] = (1280, 720)):
        self.user_id = user_id
        self.camera_index = camera_index
        self.camera_resolution = camera_resolution
        self.buffers: Dict[str, CalibrationTargetBuffer] = {
            t: CalibrationTargetBuffer(t) for t in ["CENTER", "LEFT", "RIGHT", "UP", "DOWN", "CLOSED"]
        }

    def feed_sample(self, target: str, features: GazeFeatures) -> bool:
        if target not in self.buffers:
            raise KeyError(f"Unknown calibration target: {target}")
        return self.buffers[target].add_sample(features)

    def validate_quality(self, stats: Dict[str, CalibrationStats]) -> Tuple[bool, List[str]]:
        """
        Audit all target stats against stability and separation criteria.
        Returns (is_passed, list_of_reasons_or_warnings).
        """
        errors = []

        # 1. Sample count and MAD checks
        for target, s in stats.items():
            if s.sample_count < self.MIN_SAMPLES_PER_TARGET:
                errors.append(f"Insufficient samples for {target}: {s.sample_count} < {self.MIN_SAMPLES_PER_TARGET}")
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
        # Left should have smaller H ratio than Center
        if (center.median_h - left.median_h) < self.MIN_DIRECTIONAL_SEPARATION:
            errors.append(f"Weak LEFT separation: center_h ({center.median_h:.3f}) - left_h ({left.median_h:.3f}) < {self.MIN_DIRECTIONAL_SEPARATION}")

        # Right should have larger H ratio than Center
        if (right.median_h - center.median_h) < self.MIN_DIRECTIONAL_SEPARATION:
            errors.append(f"Weak RIGHT separation: right_h ({right.median_h:.3f}) - center_h ({center.median_h:.3f}) < {self.MIN_DIRECTIONAL_SEPARATION}")

        # Up should have smaller V ratio than Center
        if (center.median_v - up.median_v) < self.MIN_DIRECTIONAL_SEPARATION:
            errors.append(f"Weak UP separation: center_v ({center.median_v:.3f}) - up_v ({up.median_v:.3f}) < {self.MIN_DIRECTIONAL_SEPARATION}")

        # Down should have larger V ratio than Center
        if (down.median_v - center.median_v) < self.MIN_DIRECTIONAL_SEPARATION:
            errors.append(f"Weak DOWN separation: down_v ({down.median_v:.3f}) - center_v ({center.median_v:.3f}) < {self.MIN_DIRECTIONAL_SEPARATION}")

        # 3. Blink EAR separation
        open_ear = center.min_v  # EAR is captured in eye landmarks
        # EAR comparison from buffer
        open_ear_med = float(np.median(self.buffers["CENTER"].ear_samples)) if self.buffers["CENTER"].ear_samples else 0.30
        closed_ear_med = float(np.median(self.buffers["CLOSED"].ear_samples)) if self.buffers["CLOSED"].ear_samples else 0.12

        if (open_ear_med - closed_ear_med) < self.MIN_BLINK_SEPARATION:
            errors.append(f"Weak blink EAR separation: open ({open_ear_med:.3f}) - closed ({closed_ear_med:.3f}) < {self.MIN_BLINK_SEPARATION}")

        return len(errors) == 0, errors

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

        open_ear_med = float(np.median(self.buffers["CENTER"].ear_samples))
        closed_ear_med = float(np.median(self.buffers["CLOSED"].ear_samples))

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
