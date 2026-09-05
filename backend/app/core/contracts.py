"""
VoxGaze Core Data Contracts (Pydantic Models)
Defines schemas for bilateral gaze features, calibration, states, and telemetry.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


# ---------------------------------------------------------
# 1. Landmark & Geometry Contracts
# ---------------------------------------------------------

class Point2D(BaseModel):
    x: float
    y: float


class EyeLandmarks(BaseModel):
    iris_center: Point2D
    inner_canthus: Point2D
    outer_canthus: Point2D
    upper_lid: Point2D
    lower_lid: Point2D
    h_ratio: float = Field(..., description="Normalized horizontal ratio [0, 1]")
    v_ratio: float = Field(..., description="Normalized vertical ratio [0, 1]")
    ear: float = Field(..., description="Eye Aspect Ratio")


class GazeFeatures(BaseModel):
    timestamp_ms: int = Field(..., description="Monotonic timestamp in milliseconds")
    left_eye: Optional[EyeLandmarks] = None
    right_eye: Optional[EyeLandmarks] = None
    head_pose: Optional[Dict[str, float]] = Field(default=None, description="{'pitch': float, 'yaw': float, 'roll': float}")
    ear_left: float = 0.0
    ear_right: float = 0.0
    tracking_confidence: float = Field(..., ge=0.0, le=1.0)
    is_tracking_valid: bool = True


# ---------------------------------------------------------
# 2. Calibration Profile Contracts
# ---------------------------------------------------------

class DirectionThreshold(BaseModel):
    enter: float
    exit: float


class CalibrationThresholds(BaseModel):
    left: DirectionThreshold
    right: DirectionThreshold
    up: DirectionThreshold
    down: DirectionThreshold
    blink_closed_ear: float
    blink_open_ear: float


class CalibrationStats(BaseModel):
    sample_count: int
    median_h: float
    median_v: float
    mad_h: float
    mad_v: float
    min_h: float
    max_h: float
    min_v: float
    max_v: float


class CalibrationProfile(BaseModel):
    profile_version: int = 1
    algorithm_version: str = "m1.0"
    user_id: str
    created_at_utc: str
    camera_index: int
    camera_resolution: Tuple[int, int]
    stats: Dict[str, CalibrationStats]  # Keys: "CENTER", "LEFT", "RIGHT", "UP", "DOWN", "CLOSED"
    thresholds: CalibrationThresholds
    is_valid: bool = True


# ---------------------------------------------------------
# 3. Gaze & Interaction States
# ---------------------------------------------------------

class Direction(str, Enum):
    CENTER = "CENTER"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UP = "UP"
    DOWN = "DOWN"
    BLINK = "BLINK"
    ABSTAIN = "ABSTAIN"
    NO_FACE = "NO_FACE"
    UNCALIBRATED = "UNCALIBRATED"


class GazeState(BaseModel):
    timestamp_ms: int
    direction: Direction
    raw_h: float
    raw_v: float
    ear: float
    confidence: float
    status_flag: str  # "● TRACKING", "✓ CALIBRATED", "! ABSTAIN", "✕ NO_FACE"


# ---------------------------------------------------------
# 4. Semantic Intent & Preview Contracts
# ---------------------------------------------------------

class IntentCandidate(BaseModel):
    intent_id: str
    domain: str
    action: str
    target: Optional[str] = None
    display_label: str
    spoken_phrase: str


class InteractionStateEnum(str, Enum):
    IDLE = "IDLE"
    DOMAIN_SELECTED = "DOMAIN_SELECTED"
    ITEM_SELECTED = "ITEM_SELECTED"
    PREVIEW = "PREVIEW"
    CONFIRMED = "CONFIRMED"
    SPEAKING = "SPEAKING"
    ABSTAIN = "ABSTAIN"


class PreviewState(BaseModel):
    current_state: InteractionStateEnum
    active_domain: Optional[str] = None
    candidate_intent: Optional[IntentCandidate] = None
    prompt_text: str = ""
    can_cancel: bool = True


class InteractionEvent(BaseModel):
    session_id: str
    user_id: str
    timestamp_ms: int
    event_type: str
    direction: Optional[Direction] = None
    candidate_id: Optional[str] = None
    confirmed: bool = False
    cancelled: bool = False
    processing_latency_ms: float
