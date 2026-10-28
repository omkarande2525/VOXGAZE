"""
VoxGaze Perception: Common Gaze Estimator Interface & Contracts

Defines the abstract interface and standard output container for all deep-learning
and auxiliary gaze estimation models (e.g. MobileOne S0, ResNet-34).

DESIGN INVARIANTS:
1. Isolated from Core AAC FSM: Does NOT mutate GazeFeatures or bypass safety preview.
2. Uniform Coordinate & Angle System:
   - pitch_rad / pitch_deg: Vertical angle (positive = looking UP, negative = looking DOWN).
   - yaw_rad / yaw_deg: Horizontal angle (positive = looking RIGHT, negative = looking LEFT).
3. Explicit Latency & Validity Tracking: Every output tracks inference, preprocessing,
   and numerical validity flags.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class GazeModelOutput:
    """
    Standardized inference result across all model backbones.

    Angles follow official Gaze360 / L2CS / MobileGaze convention:
    - pitch_rad: Vertical rotation in radians (positive = looking UP, negative = looking DOWN).
    - yaw_rad: Horizontal rotation in radians (positive = looking RIGHT, negative = looking LEFT).
    - pitch_deg: Vertical rotation in degrees.
    - yaw_deg: Horizontal rotation in degrees.
    - inference_latency_ms: Core model inference duration in milliseconds.
    - preprocessing_latency_ms: Image preparation duration in milliseconds.
    - model_name: Identifier for the model architecture (e.g. 'mobileone_s0', 'resnet34').
    - is_valid: True if inference succeeded with finite outputs; False if aborted or degraded.
    - error_message: Descriptive reason when is_valid is False.
    """

    pitch_rad: float
    yaw_rad: float
    pitch_deg: float
    yaw_deg: float
    inference_latency_ms: float
    preprocessing_latency_ms: float
    model_name: str = "unknown"
    is_valid: bool = True
    error_message: Optional[str] = None


class BaseGazeEstimator(ABC):
    """
    Abstract base class for 3D gaze direction estimators.
    Guarantees consistent preprocessing contracts, decoding math, and execution profiling.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Unique model architecture identifier."""
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        Preprocesses a raw cropped face image (BGR) into an ONNX-compatible tensor batch.

        Args:
            face_bgr: NumPy array containing cropped face image in BGR format.

        Returns:
            Preprocessed tensor with shape (1, 3, H, W) as float32.
        """
        raise NotImplementedError

    @abstractmethod
    def decode_logits(
        self,
        yaw_logits: np.ndarray,
        pitch_logits: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """
        Decodes classification logits into continuous angles.

        Args:
            yaw_logits: Array of shape (1, NUM_BINS) or (NUM_BINS,)
            pitch_logits: Array of shape (1, NUM_BINS) or (NUM_BINS,)

        Returns:
            Tuple of (pitch_rad, yaw_rad, pitch_deg, yaw_deg)
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, face_bgr: np.ndarray) -> GazeModelOutput:
        """
        Executes end-to-end gaze estimation on a single cropped face image.

        Args:
            face_bgr: Crop of detected face bounding box in BGR format.

        Returns:
            GazeModelOutput containing decoded angles, timing metrics, and validity status.
        """
        raise NotImplementedError
