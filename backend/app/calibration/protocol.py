"""
VoxGaze Calibration Validation Protocol (M1.5-B1)
Defines structured multi-phase calibration, independent evaluation, and head contrast sequences.

RESEARCH & SAFETY INVARIANTS:
1. Strict Phase Separation: Phase 1 (Calibration) samples derive candidate thresholds; Phase 2 (Evaluation) evaluates them independently.
2. Temporal Correlation Respect: Frames within a 4s trial are correlated; trial-level medians are reported alongside frame statistics.
3. Upward Gaze Feasibility: If Delta V < 0.05, UP gaze is marked UNSUPPORTED rather than forcing a classifier.
4. Comfortable Dynamics: Head contrast conditions use gentle (~10-15 deg) comfortable movements without extreme strain.
5. Privacy First: Logs bounded numerical metrics only; zero raw frames or face crops stored.
"""

from dataclasses import asdict, dataclass
from enum import Enum
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union


class ProtocolPhase(str, Enum):
    CALIBRATION_ESTIMATION = "CALIBRATION_ESTIMATION"
    INDEPENDENT_EVALUATION = "INDEPENDENT_EVALUATION"
    HEAD_MOVEMENT_CONTRAST = "HEAD_MOVEMENT_CONTRAST"


@dataclass(frozen=True)
class CalibrationValidationCondition:
    """Definition of a single condition in the M1.5-B1 protocol."""

    index: int
    name: str
    phase: ProtocolPhase
    target_direction: str  # "CENTER", "LEFT", "RIGHT", "UP", "DOWN", "CLOSED", "HEAD_LEFT", etc.
    instruction: str
    target_norm_pos: Optional[Tuple[float, float]] = None  # (x_frac, y_frac) on screen
    is_center: bool = False


# Screen target positions (comfortable, non-extreme)
TARGET_CENTER: Tuple[float, float] = (0.50, 0.50)
TARGET_LEFT: Tuple[float, float] = (0.15, 0.50)
TARGET_RIGHT: Tuple[float, float] = (0.85, 0.50)
TARGET_UP: Tuple[float, float] = (0.50, 0.15)
TARGET_DOWN: Tuple[float, float] = (0.50, 0.85)

CALIBRATION_VALIDATION_SEQUENCE: List[CalibrationValidationCondition] = [
    # -------------------------------------------------------------
    # PHASE 1: CALIBRATION ESTIMATION (5 targets + closed eyes, interleaved center returns)
    # -------------------------------------------------------------
    CalibrationValidationCondition(
        index=1,
        name="CALIB_CENTER_1",
        phase=ProtocolPhase.CALIBRATION_ESTIMATION,
        target_direction="CENTER",
        instruction="Look naturally at the CENTER target circle.",
        target_norm_pos=TARGET_CENTER,
        is_center=True,
    ),
    CalibrationValidationCondition(
        index=2,
        name="CALIB_LEFT",
        phase=ProtocolPhase.CALIBRATION_ESTIMATION,
        target_direction="LEFT",
        instruction="Shift eyes to the LEFT circle without moving your head.",
        target_norm_pos=TARGET_LEFT,
        is_center=False,
    ),
    CalibrationValidationCondition(
        index=3,
        name="CALIB_CENTER_2",
        phase=ProtocolPhase.CALIBRATION_ESTIMATION,
        target_direction="CENTER",
        instruction="Return eyes to the CENTER target circle.",
        target_norm_pos=TARGET_CENTER,
        is_center=True,
    ),
    CalibrationValidationCondition(
        index=4,
        name="CALIB_RIGHT",
        phase=ProtocolPhase.CALIBRATION_ESTIMATION,
        target_direction="RIGHT",
        instruction="Shift eyes to the RIGHT circle without moving your head.",
        target_norm_pos=TARGET_RIGHT,
        is_center=False,
    ),
    CalibrationValidationCondition(
        index=5,
        name="CALIB_CENTER_3",
        phase=ProtocolPhase.CALIBRATION_ESTIMATION,
        target_direction="CENTER",
        instruction="Return eyes to the CENTER target circle.",
        target_norm_pos=TARGET_CENTER,
        is_center=True,
    ),
    CalibrationValidationCondition(
        index=6,
        name="CALIB_UP",
        phase=ProtocolPhase.CALIBRATION_ESTIMATION,
        target_direction="UP",
        instruction="Shift eyes UP to the top circle without tilting your head.",
        target_norm_pos=TARGET_UP,
        is_center=False,
    ),
    CalibrationValidationCondition(
        index=7,
        name="CALIB_CENTER_4",
        phase=ProtocolPhase.CALIBRATION_ESTIMATION,
        target_direction="CENTER",
        instruction="Return eyes to the CENTER target circle.",
        target_norm_pos=TARGET_CENTER,
        is_center=True,
    ),
    CalibrationValidationCondition(
        index=8,
        name="CALIB_DOWN",
        phase=ProtocolPhase.CALIBRATION_ESTIMATION,
        target_direction="DOWN",
        instruction="Shift eyes DOWN to the bottom circle without tilting your head.",
        target_norm_pos=TARGET_DOWN,
        is_center=False,
    ),
    CalibrationValidationCondition(
        index=9,
        name="CALIB_CENTER_5",
        phase=ProtocolPhase.CALIBRATION_ESTIMATION,
        target_direction="CENTER",
        instruction="Return eyes to the CENTER target circle.",
        target_norm_pos=TARGET_CENTER,
        is_center=True,
    ),
    CalibrationValidationCondition(
        index=10,
        name="CALIB_CLOSED",
        phase=ProtocolPhase.CALIBRATION_ESTIMATION,
        target_direction="CLOSED",
        instruction="Gently close both eyes and hold until timer completes.",
        target_norm_pos=None,
        is_center=False,
    ),
    # -------------------------------------------------------------
    # PHASE 2: INDEPENDENT EVALUATION (Separate repeated test trials)
    # -------------------------------------------------------------
    CalibrationValidationCondition(
        index=11,
        name="EVAL_CENTER_1",
        phase=ProtocolPhase.INDEPENDENT_EVALUATION,
        target_direction="CENTER",
        instruction="[TEST TRIAL] Look at the CENTER target circle.",
        target_norm_pos=TARGET_CENTER,
        is_center=True,
    ),
    CalibrationValidationCondition(
        index=12,
        name="EVAL_LEFT",
        phase=ProtocolPhase.INDEPENDENT_EVALUATION,
        target_direction="LEFT",
        instruction="[TEST TRIAL] Shift eyes to the LEFT circle.",
        target_norm_pos=TARGET_LEFT,
        is_center=False,
    ),
    CalibrationValidationCondition(
        index=13,
        name="EVAL_CENTER_2",
        phase=ProtocolPhase.INDEPENDENT_EVALUATION,
        target_direction="CENTER",
        instruction="[TEST TRIAL] Return eyes to the CENTER circle.",
        target_norm_pos=TARGET_CENTER,
        is_center=True,
    ),
    CalibrationValidationCondition(
        index=14,
        name="EVAL_RIGHT",
        phase=ProtocolPhase.INDEPENDENT_EVALUATION,
        target_direction="RIGHT",
        instruction="[TEST TRIAL] Shift eyes to the RIGHT circle.",
        target_norm_pos=TARGET_RIGHT,
        is_center=False,
    ),
    CalibrationValidationCondition(
        index=15,
        name="EVAL_CENTER_3",
        phase=ProtocolPhase.INDEPENDENT_EVALUATION,
        target_direction="CENTER",
        instruction="[TEST TRIAL] Return eyes to the CENTER circle.",
        target_norm_pos=TARGET_CENTER,
        is_center=True,
    ),
    CalibrationValidationCondition(
        index=16,
        name="EVAL_UP",
        phase=ProtocolPhase.INDEPENDENT_EVALUATION,
        target_direction="UP",
        instruction="[TEST TRIAL] Shift eyes UP to the top circle.",
        target_norm_pos=TARGET_UP,
        is_center=False,
    ),
    CalibrationValidationCondition(
        index=17,
        name="EVAL_CENTER_4",
        phase=ProtocolPhase.INDEPENDENT_EVALUATION,
        target_direction="CENTER",
        instruction="[TEST TRIAL] Return eyes to the CENTER circle.",
        target_norm_pos=TARGET_CENTER,
        is_center=True,
    ),
    CalibrationValidationCondition(
        index=18,
        name="EVAL_DOWN",
        phase=ProtocolPhase.INDEPENDENT_EVALUATION,
        target_direction="DOWN",
        instruction="[TEST TRIAL] Shift eyes DOWN to the bottom circle.",
        target_norm_pos=TARGET_DOWN,
        is_center=False,
    ),
    CalibrationValidationCondition(
        index=19,
        name="EVAL_CENTER_5",
        phase=ProtocolPhase.INDEPENDENT_EVALUATION,
        target_direction="CENTER",
        instruction="[TEST TRIAL] Return eyes to the CENTER circle.",
        target_norm_pos=TARGET_CENTER,
        is_center=True,
    ),
    # -------------------------------------------------------------
    # PHASE 3: HEAD MOVEMENT CONTRAST (Exploratory head pose vs ocular gaze)
    # -------------------------------------------------------------
    CalibrationValidationCondition(
        index=20,
        name="HEAD_LEFT_COMFORTABLE",
        phase=ProtocolPhase.HEAD_MOVEMENT_CONTRAST,
        target_direction="HEAD_LEFT",
        instruction="Turn head gently (~10-15°) LEFT while keeping eyes comfortably on CENTER.",
        target_norm_pos=TARGET_CENTER,
        is_center=False,
    ),
    CalibrationValidationCondition(
        index=21,
        name="HEAD_CENTER_RETURN_1",
        phase=ProtocolPhase.HEAD_MOVEMENT_CONTRAST,
        target_direction="CENTER",
        instruction="Return head to CENTER position.",
        target_norm_pos=TARGET_CENTER,
        is_center=True,
    ),
    CalibrationValidationCondition(
        index=22,
        name="HEAD_RIGHT_COMFORTABLE",
        phase=ProtocolPhase.HEAD_MOVEMENT_CONTRAST,
        target_direction="HEAD_RIGHT",
        instruction="Turn head gently (~10-15°) RIGHT while keeping eyes comfortably on CENTER.",
        target_norm_pos=TARGET_CENTER,
        is_center=False,
    ),
    CalibrationValidationCondition(
        index=23,
        name="HEAD_CENTER_RETURN_2",
        phase=ProtocolPhase.HEAD_MOVEMENT_CONTRAST,
        target_direction="CENTER",
        instruction="Return head to CENTER position.",
        target_norm_pos=TARGET_CENTER,
        is_center=True,
    ),
    CalibrationValidationCondition(
        index=24,
        name="HEAD_UP_COMFORTABLE",
        phase=ProtocolPhase.HEAD_MOVEMENT_CONTRAST,
        target_direction="HEAD_UP",
        instruction="Tilt chin gently (~5-10°) UP while keeping eyes comfortably on CENTER.",
        target_norm_pos=TARGET_CENTER,
        is_center=False,
    ),
    CalibrationValidationCondition(
        index=25,
        name="HEAD_CENTER_RETURN_3",
        phase=ProtocolPhase.HEAD_MOVEMENT_CONTRAST,
        target_direction="CENTER",
        instruction="Return head to CENTER position.",
        target_norm_pos=TARGET_CENTER,
        is_center=True,
    ),
    CalibrationValidationCondition(
        index=26,
        name="HEAD_DOWN_COMFORTABLE",
        phase=ProtocolPhase.HEAD_MOVEMENT_CONTRAST,
        target_direction="HEAD_DOWN",
        instruction="Tilt chin gently (~5-10°) DOWN while keeping eyes comfortably on CENTER.",
        target_norm_pos=TARGET_CENTER,
        is_center=False,
    ),
    CalibrationValidationCondition(
        index=27,
        name="HEAD_CENTER_RETURN_FINAL",
        phase=ProtocolPhase.HEAD_MOVEMENT_CONTRAST,
        target_direction="CENTER",
        instruction="Final neutral center rest position.",
        target_norm_pos=TARGET_CENTER,
        is_center=True,
    ),
]


M15_CSV_HEADER: List[str] = [
    "session_id",
    "monotonic_time_s",
    "elapsed_session_s",
    "protocol_phase",
    "condition_index",
    "condition_name",
    "target_direction",
    "frame_index",
    "face_detected",
    "face_crop_valid",
    "geometry_valid",
    "active_eye",
    "right_eye_valid",
    "left_eye_valid",
    "right_eye_h_ratio",
    "right_eye_v_ratio",
    "right_eye_ear",
    "right_eye_canthal_ratio",
    "left_eye_h_ratio",
    "left_eye_v_ratio",
    "left_eye_ear",
    "left_eye_canthal_ratio",
    "canthal_normal_dist_px",
    "mobileone_valid",
    "mobileone_yaw_deg",
    "mobileone_pitch_deg",
    "mediapipe_latency_ms",
    "total_frame_latency_ms",
    "unavailable_reason",
]


@dataclass
class CalibrationValidationRecord:
    """Single numerical observation frame in the M1.5-B1 session."""

    session_id: str
    monotonic_time_s: float
    elapsed_session_s: float
    protocol_phase: str
    condition_index: int
    condition_name: str
    target_direction: str
    frame_index: int
    face_detected: bool
    face_crop_valid: bool
    geometry_valid: bool
    active_eye: str = "right"
    right_eye_valid: Optional[bool] = None
    left_eye_valid: Optional[bool] = None
    right_eye_h_ratio: Optional[float] = None
    right_eye_v_ratio: Optional[float] = None
    right_eye_ear: Optional[float] = None
    right_eye_canthal_ratio: Optional[float] = None
    left_eye_h_ratio: Optional[float] = None
    left_eye_v_ratio: Optional[float] = None
    left_eye_ear: Optional[float] = None
    left_eye_canthal_ratio: Optional[float] = None
    canthal_normal_dist_px: Optional[float] = None
    mobileone_valid: bool = False
    mobileone_yaw_deg: Optional[float] = None
    mobileone_pitch_deg: Optional[float] = None
    mediapipe_latency_ms: float = 0.0
    total_frame_latency_ms: float = 0.0
    unavailable_reason: Optional[str] = None

    def to_csv_row(self) -> List[str]:
        def _fmt_f(val: Optional[float], decimals: int = 4) -> str:
            if val is None or not math.isfinite(val):
                return ""
            return f"{val:.{decimals}f}"

        def _fmt_b(val: Optional[bool]) -> str:
            if val is None:
                return ""
            return "true" if val else "false"

        return [
            self.session_id,
            _fmt_f(self.monotonic_time_s, 6),
            _fmt_f(self.elapsed_session_s, 3),
            self.protocol_phase,
            str(self.condition_index),
            self.condition_name,
            self.target_direction,
            str(self.frame_index),
            _fmt_b(self.face_detected),
            _fmt_b(self.face_crop_valid),
            _fmt_b(self.geometry_valid),
            self.active_eye,
            _fmt_b(self.right_eye_valid),
            _fmt_b(self.left_eye_valid),
            _fmt_f(self.right_eye_h_ratio, 4),
            _fmt_f(self.right_eye_v_ratio, 4),
            _fmt_f(self.right_eye_ear, 4),
            _fmt_f(self.right_eye_canthal_ratio, 4),
            _fmt_f(self.left_eye_h_ratio, 4),
            _fmt_f(self.left_eye_v_ratio, 4),
            _fmt_f(self.left_eye_ear, 4),
            _fmt_f(self.left_eye_canthal_ratio, 4),
            _fmt_f(self.canthal_normal_dist_px, 2),
            _fmt_b(self.mobileone_valid),
            _fmt_f(self.mobileone_yaw_deg, 2),
            _fmt_f(self.mobileone_pitch_deg, 2),
            _fmt_f(self.mediapipe_latency_ms, 2),
            _fmt_f(self.total_frame_latency_ms, 2),
            self.unavailable_reason or "",
        ]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
