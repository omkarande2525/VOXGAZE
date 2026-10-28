"""
VoxGaze AAC MVP: Gaze Input Providers

Defines the GazeInputProvider interface with two distinct, honest operating modes:
1. SimulatedGazeProvider (Demo Mode): Presenter moves visible gaze reticle via keyboard/WASD.
   Visibly labeled SIMULATED GAZE.
2. LiveGazeProvider (Live Research Mode): Real camera stream via MediaPipe & MobileOne.
   Enforces calibration separation gates; keeps live word activation DISABLED until gates pass.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.app.core.contracts import Point2D
from backend.app.calibration.personalized_mapper import (
    CalibrationSample,
    PersonalizedGazeMapper,
)
from backend.app.perception.gaze_estimator_interface import BaseGazeEstimator


@dataclass
class GazeTelemetry:
    """Comprehensive telemetry packet streamed to the UI."""
    mode: str  # "DEMO_SIMULATED" or "LIVE_RESEARCH"
    timestamp_ms: int
    pointer_x: Optional[float]  # Normalized [0, 1] screen X
    pointer_y: Optional[float]  # Normalized [0, 1] screen Y
    is_tracking_valid: bool
    face_detected: bool
    confidence: float
    status_label: str  # e.g. "● TRACKING", "! ABSTAIN", "✕ NO_FACE", "DEMO ACTIVE"
    is_selection_enabled: bool
    unsupported_reason: Optional[str]
    # Detailed live diagnostics
    h_ratio: Optional[float] = None
    v_ratio: Optional[float] = None
    ear: Optional[float] = None
    canthal_dist_px: Optional[float] = None
    head_yaw_deg: Optional[float] = None
    head_pitch_deg: Optional[float] = None
    is_simulated: bool = False


class GazeInputProvider(ABC):
    """Abstract base provider for gaze input."""

    @abstractmethod
    def get_telemetry(self) -> GazeTelemetry:
        """Returns the latest telemetry packet."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets provider state."""
        pass


class SimulatedGazeProvider(GazeInputProvider):
    """
    Demo / Simulated Gaze Mode:
    Provides keyboard-driven (Arrow keys / WASD) movement of a visible gaze indicator.
    Clearly and permanently labeled as simulated input.
    """

    def __init__(self, start_x: float = 0.50, start_y: float = 0.50, step_size: float = 0.08):
        self.x = start_x
        self.y = start_y
        self.step_size = step_size
        self._is_active = True

    def move(self, dx: float, dy: float) -> Tuple[float, float]:
        """Moves gaze reticle by delta, clamped to [0.05, 0.95]."""
        self.x = max(0.05, min(0.95, self.x + dx))
        self.y = max(0.05, min(0.95, self.y + dy))
        return self.x, self.y

    def jump_to(self, x: float, y: float) -> Tuple[float, float]:
        """Sets absolute gaze coordinates."""
        self.x = max(0.05, min(0.95, x))
        self.y = max(0.05, min(0.95, y))
        return self.x, self.y

    def get_telemetry(self) -> GazeTelemetry:
        ts = int(time.time() * 1000)
        return GazeTelemetry(
            mode="DEMO_SIMULATED",
            timestamp_ms=ts,
            pointer_x=round(self.x, 4),
            pointer_y=round(self.y, 4),
            is_tracking_valid=True,
            face_detected=True,
            confidence=1.0,
            status_label="● DEMO SIMULATION (KEYBOARD CONTROL)",
            is_selection_enabled=True,
            unsupported_reason=None,
            h_ratio=round(self.x, 3),
            v_ratio=round(self.y, 3),
            ear=0.30,
            is_simulated=True,
        )

    def reset(self) -> None:
        self.x = 0.50
        self.y = 0.50


class LiveGazeProvider(GazeInputProvider):
    """
    Live Research Mode:
    Executes MediaPipe FaceLandmarker and MobileOne on the physical camera stream.
    Strictly checks directional calibration separation gates:
    Because physical calibration has NOT yet passed horizontal or vertical separation gates,
    this provider honestly marks selection DISABLED and does NOT invent a fake mapping.
    """

    def __init__(
        self,
        camera_idx: int = 0,
        mapper: Optional[PersonalizedGazeMapper] = None,
        estimator: Optional[BaseGazeEstimator] = None,
    ):
        self.camera_idx = camera_idx
        self.mapper = mapper
        self.estimator = estimator
        self.face_detected = False
        self.is_tracking_valid = False
        self.confidence = 0.0
        self.h_ratio: Optional[float] = None
        self.v_ratio: Optional[float] = None
        self.ear: Optional[float] = None
        self.canthal_dist_px: Optional[float] = None
        self.head_yaw_deg: Optional[float] = None
        self.head_pitch_deg: Optional[float] = None
        self.status_label = "INITIALIZING_CAMERA"

        # Gate enforcement: Physical calibration separation has not passed
        self.is_selection_enabled = False
        self.unsupported_reason = (
            "LIVE_SELECTION_DISABLED: Calibration directional separation gates (H/V) "
            "have not passed in physical trials. Live word activation remains disabled."
        )

    def update_frame_measurements(
        self,
        face_detected: bool,
        geometry_valid: bool,
        confidence: float,
        h_ratio: Optional[float],
        v_ratio: Optional[float],
        ear: Optional[float],
        canthal_dist_px: Optional[float] = None,
        yaw_deg: Optional[float] = None,
        pitch_deg: Optional[float] = None,
    ) -> None:
        """Feeds fresh measurements from the camera pipeline."""
        self.face_detected = face_detected
        self.is_tracking_valid = face_detected and geometry_valid
        self.confidence = confidence
        self.h_ratio = h_ratio
        self.v_ratio = v_ratio
        self.ear = ear
        self.canthal_dist_px = canthal_dist_px
        self.head_yaw_deg = yaw_deg
        self.head_pitch_deg = pitch_deg

        if not face_detected:
            self.status_label = "✕ NO_FACE_DETECTED"
        elif not geometry_valid:
            self.status_label = "! ABSTAIN (GEOMETRY DEGRADED)"
        else:
            self.status_label = "● TRACKING ACTIVE (RESEARCH DIAGNOSTIC)"

    def get_telemetry(self) -> GazeTelemetry:
        ts = int(time.time() * 1000)
        pointer_x: Optional[float] = None
        pointer_y: Optional[float] = None

        if self.mapper is not None and getattr(self.mapper, "is_fitted", False) and self.is_tracking_valid:
            sample = CalibrationSample(
                target_x=0.5,
                target_y=0.5,
                target_direction="CENTER",
                yaw_deg=self.head_yaw_deg,
                pitch_deg=self.head_pitch_deg,
                right_eye_h=self.h_ratio,
                right_eye_v=self.v_ratio,
                left_eye_h=self.h_ratio,
                left_eye_v=self.v_ratio,
                canthal_normal=self.canthal_dist_px,
            )
            pred = self.mapper.predict(sample)
            if pred.is_valid:
                pointer_x = pred.screen_x
                pointer_y = pred.screen_y

        return GazeTelemetry(
            mode="LIVE_RESEARCH",
            timestamp_ms=ts,
            pointer_x=pointer_x,
            pointer_y=pointer_y,
            is_tracking_valid=self.is_tracking_valid,
            face_detected=self.face_detected,
            confidence=round(self.confidence, 3),
            status_label=self.status_label,
            is_selection_enabled=self.is_selection_enabled,
            unsupported_reason=self.unsupported_reason,
            h_ratio=round(self.h_ratio, 3) if self.h_ratio is not None else None,
            v_ratio=round(self.v_ratio, 3) if self.v_ratio is not None else None,
            ear=round(self.ear, 3) if self.ear is not None else None,
            canthal_dist_px=round(self.canthal_dist_px, 2) if self.canthal_dist_px is not None else None,
            head_yaw_deg=round(self.head_yaw_deg, 1) if self.head_yaw_deg is not None else None,
            head_pitch_deg=round(self.head_pitch_deg, 1) if self.head_pitch_deg is not None else None,
            is_simulated=False,
        )

    def reset(self) -> None:
        self.face_detected = False
        self.is_tracking_valid = False
        self.status_label = "RESET"
