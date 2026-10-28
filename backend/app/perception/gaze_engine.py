"""
VoxGaze Gaze Engine & Directional Hysteresis Classifier
Consumes normalized GazeFeatures, applies user CalibrationProfile thresholds,
and outputs discrete GazeState while handling abstention and blink signals.
"""

import math
from typing import Optional, Tuple
from backend.app.core.contracts import (
    CalibrationProfile,
    Direction,
    GazeFeatures,
    GazeState,
)
from backend.app.perception.geometry import (
    PLAUSIBLE_H_MAX,
    PLAUSIBLE_H_MIN,
    PLAUSIBLE_V_MAX,
    PLAUSIBLE_V_MIN,
)


class GazeEngine:
    """Stateful directional classifier with dual-threshold hysteresis."""

    def __init__(
        self,
        profile: Optional[CalibrationProfile] = None,
        min_confidence: float = 0.5,
        active_eye: str = "right",
    ):
        self.profile: Optional[CalibrationProfile] = profile
        self.min_confidence: float = min_confidence
        self.active_eye: str = active_eye
        self.current_direction: Direction = Direction.CENTER
        self.blink_start_ms: Optional[int] = None
        self.last_blink_duration_ms: float = 0.0

    def set_profile(self, profile: CalibrationProfile) -> None:
        """Update active calibration profile and reset directional state to CENTER."""
        self.profile = profile
        self.current_direction = Direction.CENTER
        self.blink_start_ms = None
        self.last_blink_duration_ms = 0.0

    def clear_profile(self) -> None:
        """Clear active profile."""
        self.profile = None
        self.current_direction = Direction.UNCALIBRATED

    def process_features(self, features: GazeFeatures) -> GazeState:
        """
        Process a single GazeFeatures sample into a discrete GazeState.
        Enforces quality gating, blink detection, and hysteresis transitions.
        """
        ts = features.timestamp_ms
        conf = features.tracking_confidence

        # Select active eye (RIGHT_EYE_BASELINE by default, with fallback to left eye)
        eye = features.right_eye if self.active_eye == "right" else features.left_eye
        active_valid = features.is_right_eye_valid if self.active_eye == "right" else features.is_left_eye_valid
        if eye is None:
            eye = features.left_eye if self.active_eye == "right" else features.right_eye
            active_valid = features.is_left_eye_valid if self.active_eye == "right" else features.is_right_eye_valid

        # 1. Quality Gate: No face or eye detected
        if eye is None:
            self.blink_start_ms = None
            return GazeState(
                timestamp_ms=ts,
                direction=Direction.NO_FACE,
                raw_h=0.0,
                raw_v=0.0,
                ear=0.0,
                confidence=conf,
                status_flag="✕ NO_FACE",
            )

        h = eye.h_ratio
        v = eye.v_ratio
        ear = eye.ear

        # If profile calibrated using canthal normal, use active eye's canthal_ratio for vertical hysteresis
        if self.profile is not None and getattr(self.profile.thresholds, "vertical_metric", "eyelid") == "canthal_normal":
            if eye.canthal_ratio is not None and math.isfinite(eye.canthal_ratio):
                v = eye.canthal_ratio

        # 2. Quality Gate: Low tracking confidence, invalid tracking flag, or active eye marked invalid
        if not features.is_tracking_valid or conf < self.min_confidence or active_valid is False:
            self.blink_start_ms = None
            return GazeState(
                timestamp_ms=ts,
                direction=Direction.ABSTAIN,
                raw_h=h,
                raw_v=v,
                ear=ear,
                confidence=conf,
                status_flag="! ABSTAIN",
            )

        # 2b. Quality Gate: Non-finite coordinates or out-of-band ratios on active eye
        if not (math.isfinite(h) and math.isfinite(v) and math.isfinite(ear)):
            self.blink_start_ms = None
            return GazeState(
                timestamp_ms=ts,
                direction=Direction.ABSTAIN,
                raw_h=h,
                raw_v=v,
                ear=ear,
                confidence=conf,
                status_flag="! ABSTAIN (NON_FINITE)",
            )

        if not (PLAUSIBLE_H_MIN <= h <= PLAUSIBLE_H_MAX and PLAUSIBLE_V_MIN <= v <= PLAUSIBLE_V_MAX):
            self.blink_start_ms = None
            return GazeState(
                timestamp_ms=ts,
                direction=Direction.ABSTAIN,
                raw_h=h,
                raw_v=v,
                ear=ear,
                confidence=conf,
                status_flag="! ABSTAIN (OUT_OF_BOUNDS_RATIO)",
            )

        # 3. Quality Gate: Uncalibrated profile
        if self.profile is None:
            return GazeState(
                timestamp_ms=ts,
                direction=Direction.UNCALIBRATED,
                raw_h=h,
                raw_v=v,
                ear=ear,
                confidence=conf,
                status_flag="! UNCALIBRATED",
            )

        t = self.profile.thresholds

        # 4. Blink / Eye Closure Handling
        if ear <= t.blink_closed_ear:
            if self.blink_start_ms is None:
                self.blink_start_ms = ts
            self.last_blink_duration_ms = float(ts - self.blink_start_ms)
            return GazeState(
                timestamp_ms=ts,
                direction=Direction.BLINK,
                raw_h=h,
                raw_v=v,
                ear=ear,
                confidence=conf,
                status_flag="● BLINK",
            )
        else:
            if self.blink_start_ms is not None:
                self.last_blink_duration_ms = float(ts - self.blink_start_ms)
                self.blink_start_ms = None

        # 5. Dual-Threshold Hysteresis State Transition Logic
        new_direction = self._classify_with_hysteresis(h, v, t)
        self.current_direction = new_direction

        return GazeState(
            timestamp_ms=ts,
            direction=new_direction,
            raw_h=h,
            raw_v=v,
            ear=ear,
            confidence=conf,
            status_flag="✓ CALIBRATED",
        )

    def _classify_with_hysteresis(self, h: float, v: float, t) -> Direction:
        """Applies directional enter/exit hysteresis rules relative to current state."""
        state = self.current_direction

        # If currently in a directional state, check exit threshold back to CENTER
        # (A disabled direction immediately returns to CENTER)
        if state == Direction.LEFT:
            if not getattr(t.left, "is_enabled", True) or h > t.left.exit:
                return Direction.CENTER
            return Direction.LEFT

        elif state == Direction.RIGHT:
            if not getattr(t.right, "is_enabled", True) or h < t.right.exit:
                return Direction.CENTER
            return Direction.RIGHT

        elif state == Direction.UP:
            if not getattr(t.up, "is_enabled", True) or v > t.up.exit:
                return Direction.CENTER
            return Direction.UP

        elif state == Direction.DOWN:
            if not getattr(t.down, "is_enabled", True) or v < t.down.exit:
                return Direction.CENTER
            return Direction.DOWN

        # If currently CENTER (or recovered from BLINK/UNCALIBRATED), evaluate enter thresholds
        candidates = []

        if getattr(t.left, "is_enabled", True) and h < t.left.enter:
            # Measure relative entry depth beyond threshold
            depth = t.left.enter - h
            candidates.append((Direction.LEFT, depth))

        if getattr(t.right, "is_enabled", True) and h > t.right.enter:
            depth = h - t.right.enter
            candidates.append((Direction.RIGHT, depth))

        if getattr(t.up, "is_enabled", True) and v < t.up.enter:
            depth = t.up.enter - v
            candidates.append((Direction.UP, depth))

        if getattr(t.down, "is_enabled", True) and v > t.down.enter:
            depth = v - t.down.enter
            candidates.append((Direction.DOWN, depth))

        if not candidates:
            return Direction.CENTER

        # Pick candidate with greatest penetration depth beyond entry threshold
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
