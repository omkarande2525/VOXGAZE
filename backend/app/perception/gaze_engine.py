"""
VoxGaze Gaze Engine & Directional Hysteresis Classifier
Consumes normalized GazeFeatures, applies user CalibrationProfile thresholds,
and outputs discrete GazeState while handling abstention and blink signals.
"""

from typing import Optional, Tuple
from backend.app.core.contracts import (
    CalibrationProfile,
    Direction,
    GazeFeatures,
    GazeState,
)


class GazeEngine:
    """Stateful directional classifier with dual-threshold hysteresis."""

    def __init__(self, profile: Optional[CalibrationProfile] = None, min_confidence: float = 0.5):
        self.profile: Optional[CalibrationProfile] = profile
        self.min_confidence: float = min_confidence
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

        # 1. Quality Gate: No face detected
        if features.left_eye is None:
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

        eye = features.left_eye
        h = eye.h_ratio
        v = eye.v_ratio
        ear = eye.ear

        # 2. Quality Gate: Low tracking confidence or invalid tracking flag
        if not features.is_tracking_valid or conf < self.min_confidence:
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
        if state == Direction.LEFT:
            if h > t.left.exit:
                return Direction.CENTER
            return Direction.LEFT

        elif state == Direction.RIGHT:
            if h < t.right.exit:
                return Direction.CENTER
            return Direction.RIGHT

        elif state == Direction.UP:
            if v > t.up.exit:
                return Direction.CENTER
            return Direction.UP

        elif state == Direction.DOWN:
            if v < t.down.exit:
                return Direction.CENTER
            return Direction.DOWN

        # If currently CENTER (or recovered from BLINK/UNCALIBRATED), evaluate enter thresholds
        candidates = []

        if h < t.left.enter:
            # Measure relative entry depth beyond threshold
            depth = t.left.enter - h
            candidates.append((Direction.LEFT, depth))

        if h > t.right.enter:
            depth = h - t.right.enter
            candidates.append((Direction.RIGHT, depth))

        if v < t.up.enter:
            depth = t.up.enter - v
            candidates.append((Direction.UP, depth))

        if v > t.down.enter:
            depth = v - t.down.enter
            candidates.append((Direction.DOWN, depth))

        if not candidates:
            return Direction.CENTER

        # Pick candidate with greatest penetration depth beyond entry threshold
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
