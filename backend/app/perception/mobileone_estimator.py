"""
VoxGaze Perception: Pretrained MobileOne S0 Gaze Estimator (ONNX Runtime)

Provides an isolated, non-invasive candidate gaze estimator implementing the exact
preprocessing, tensor formatting, and output decoding specifications from the official
MobileGaze / yakhyo/gaze-estimation repository.

Design Invariants:
1. Isolated from AAC FSM: Does NOT alter GazeFeatures, GazeEngine, Preview, or TTS.
2. Verified Preprocessing: RGB, resized to (448, 448), normalized to [0, 1], ImageNet mean/std, NCHW float32.
3. Official Decoding: 90-bin classification logits to continuous expectation in radians (bins=90, width=4, offset=180).
4. Safety & Robustness: Explicit validation against non-finite tensors, empty crops, and shape anomalies.
"""

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np

from backend.app.perception.gaze_estimator_interface import (
    BaseGazeEstimator,
    GazeModelOutput,
)


@dataclass(frozen=True)
class MobileOneOutput(GazeModelOutput):
    """
    Decoded inference result from MobileOne S0 Gaze Estimator.

    Angles follow official Gaze360 / L2CS convention:
    - pitch_rad: Vertical rotation in radians (positive = looking UP, negative = looking DOWN).
    - yaw_rad: Horizontal rotation in radians (positive = looking RIGHT, negative = looking LEFT).
    """

    model_name: str = "mobileone_s0"


class MobileOneGazeEstimator(BaseGazeEstimator):
    """
    Candidate 3D gaze direction estimator using pretrained MobileOne S0 ONNX runtime.
    """

    INPUT_SIZE: Tuple[int, int] = (448, 448)  # (width, height)
    IMAGENET_MEAN: np.ndarray = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD: np.ndarray = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    NUM_BINS: int = 90
    BIN_WIDTH: float = 4.0
    ANGLE_OFFSET: float = 180.0

    @property
    def model_name(self) -> str:
        return "mobileone_s0"

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        session: Optional[Any] = None,
    ) -> None:
        """
        Initializes the MobileOne Gaze Estimator.

        Args:
            model_path: Path to mobileone_s0_gaze.onnx.
            session: Optional pre-created onnxruntime.InferenceSession (useful for dependency injection and mocking).
        """
        self.idx_tensor = np.arange(self.NUM_BINS, dtype=np.float32)
        self.model_path = Path(model_path) if model_path is not None else None

        if session is not None:
            self.session = session
        elif model_path is not None:
            if not self.model_path.is_file():
                raise FileNotFoundError(f"MobileOne model file not found at: {self.model_path}")
            import onnxruntime as ort

            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=["CPUExecutionProvider"],
            )
        else:
            raise ValueError("Either model_path or session must be provided to MobileOneGazeEstimator.")

        self._validate_and_cache_io_metadata()

    def _validate_and_cache_io_metadata(self) -> None:
        """Inspect and validate ONNX session input/output bindings."""
        inputs = self.session.get_inputs()
        if not inputs:
            raise ValueError("ONNX model has no input nodes.")

        self.input_name: str = inputs[0].name
        input_shape = inputs[0].shape

        # Validate input spatial dimensions if static
        if len(input_shape) == 4:
            c, h, w = input_shape[1], input_shape[2], input_shape[3]
            if isinstance(c, int) and c != 3:
                raise ValueError(f"Expected 3 input channels, got {c}")
            if isinstance(h, int) and isinstance(w, int):
                if (w, h) != self.INPUT_SIZE:
                    raise ValueError(f"Expected input size {self.INPUT_SIZE}, got ({w}, {h})")

        outputs = self.session.get_outputs()
        self.output_names: List[str] = [out.name for out in outputs]
        if len(self.output_names) != 2:
            raise ValueError(
                f"Expected exactly 2 output nodes (yaw, pitch), found {len(self.output_names)}: {self.output_names}"
            )

        # Validate output shape if static
        for out in outputs:
            shape = out.shape
            if len(shape) == 2 and isinstance(shape[1], int) and shape[1] != self.NUM_BINS:
                raise ValueError(
                    f"Output node '{out.name}' has unexpected bin dimension: {shape[1]} (expected {self.NUM_BINS})"
                )

    def preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        Preprocesses a raw cropped face image (BGR) into an ONNX-compatible tensor batch.

        Steps:
        1. BGR -> RGB
        2. Resize to (448, 448)
        3. Scale to [0.0, 1.0]
        4. Standardize with ImageNet mean and std
        5. HWC -> CHW layout
        6. Expand batch dimension: (1, 3, 448, 448) float32

        Args:
            face_bgr: NumPy array containing cropped face image in BGR format.

        Returns:
            Preprocessed tensor with shape (1, 3, 448, 448) as float32.
        """
        if face_bgr is None or not isinstance(face_bgr, np.ndarray):
            raise ValueError("Input face crop must be a non-None numpy ndarray.")
        if face_bgr.size == 0 or face_bgr.ndim != 3 or face_bgr.shape[2] != 3:
            raise ValueError(
                f"Input face crop must be a non-empty 3-channel image (shape received: {face_bgr.shape if hasattr(face_bgr, 'shape') else None})."
            )

        # 1. Color conversion: BGR -> RGB
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

        # 2. Resize to exact input size
        resized = cv2.resize(rgb, self.INPUT_SIZE, interpolation=cv2.INTER_LINEAR)

        # 3. Scale to [0.0, 1.0]
        scaled = resized.astype(np.float32) / 255.0

        # 4. Standardize with ImageNet mean/std
        standardized = (scaled - self.IMAGENET_MEAN) / self.IMAGENET_STD

        # 5. Transpose HWC -> CHW
        chw = np.transpose(standardized, (2, 0, 1))

        # 6. Add batch dimension
        batch = np.expand_dims(chw, axis=0).astype(np.float32)

        return batch

    def decode_logits(
        self,
        yaw_logits: np.ndarray,
        pitch_logits: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """
        Decodes raw model classification logits into continuous angles.

        Decoding algorithm:
        1. Softmax to obtain probabilities over 90 discrete bins.
        2. Expectation: sum(prob * idx) * 4.0 degrees.
        3. Center offset: angle_deg - 180.0.
        4. Conversion to radians: deg2rad(angle_deg - 180.0).

        Args:
            yaw_logits: Array of shape (1, 90) or (90,)
            pitch_logits: Array of shape (1, 90) or (90,)

        Returns:
            Tuple of (pitch_rad, yaw_rad, pitch_deg, yaw_deg)
        """
        if not np.all(np.isfinite(yaw_logits)) or not np.all(np.isfinite(pitch_logits)):
            raise ValueError("Non-finite values (NaN/Inf) detected in model output logits.")

        yaw_flat = np.asarray(yaw_logits, dtype=np.float32).reshape(-1)
        pitch_flat = np.asarray(pitch_logits, dtype=np.float32).reshape(-1)

        if len(yaw_flat) != self.NUM_BINS or len(pitch_flat) != self.NUM_BINS:
            raise ValueError(
                f"Logit vectors must have length {self.NUM_BINS}. "
                f"Got yaw: {len(yaw_flat)}, pitch: {len(pitch_flat)}"
            )

        # Numerically stable softmax
        yaw_shift = yaw_flat - np.max(yaw_flat)
        yaw_exp = np.exp(yaw_shift)
        yaw_prob = yaw_exp / np.sum(yaw_exp)

        pitch_shift = pitch_flat - np.max(pitch_flat)
        pitch_exp = np.exp(pitch_shift)
        pitch_prob = pitch_exp / np.sum(pitch_exp)

        # Expected value calculation
        raw_yaw_deg = float(np.sum(yaw_prob * self.idx_tensor) * self.BIN_WIDTH)
        raw_pitch_deg = float(np.sum(pitch_prob * self.idx_tensor) * self.BIN_WIDTH)

        # Centering and conversion to radians
        yaw_deg = raw_yaw_deg - self.ANGLE_OFFSET
        pitch_deg = raw_pitch_deg - self.ANGLE_OFFSET

        yaw_rad = float(np.deg2rad(yaw_deg))
        pitch_rad = float(np.deg2rad(pitch_deg))

        return pitch_rad, yaw_rad, pitch_deg, yaw_deg

    def predict(self, face_bgr: np.ndarray) -> MobileOneOutput:
        """
        Executes end-to-end gaze estimation on a single cropped face image.

        Args:
            face_bgr: Crop of detected face bounding box in BGR format.

        Returns:
            MobileOneOutput containing angles and latency.
        """
        t0 = time.perf_counter()
        try:
            # Preprocessing
            input_tensor = self.preprocess(face_bgr)
            t_pre = time.perf_counter()
            pre_latency_ms = (t_pre - t0) * 1000.0

            # ONNX Inference
            t_infer_start = time.perf_counter()
            outputs = self.session.run(
                self.output_names,
                {self.input_name: input_tensor},
            )
            t_infer_end = time.perf_counter()
            infer_latency_ms = (t_infer_end - t_infer_start) * 1000.0

            if len(outputs) != 2:
                raise ValueError(f"Inference returned {len(outputs)} tensors, expected 2.")

            # Identify yaw and pitch outputs by binding names if present
            if self.output_names == ["yaw", "pitch"]:
                yaw_logits, pitch_logits = outputs[0], outputs[1]
            elif self.output_names == ["pitch", "yaw"]:
                pitch_logits, yaw_logits = outputs[0], outputs[1]
            else:
                # Default order from onnx_inference.py: outputs[0] = yaw, outputs[1] = pitch
                yaw_logits, pitch_logits = outputs[0], outputs[1]

            pitch_rad, yaw_rad, pitch_deg, yaw_deg = self.decode_logits(
                yaw_logits, pitch_logits
            )

            return MobileOneOutput(
                pitch_rad=pitch_rad,
                yaw_rad=yaw_rad,
                pitch_deg=pitch_deg,
                yaw_deg=yaw_deg,
                inference_latency_ms=infer_latency_ms,
                preprocessing_latency_ms=pre_latency_ms,
                is_valid=True,
            )

        except Exception as exc:
            total_elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return MobileOneOutput(
                pitch_rad=0.0,
                yaw_rad=0.0,
                pitch_deg=0.0,
                yaw_deg=0.0,
                inference_latency_ms=0.0,
                preprocessing_latency_ms=total_elapsed_ms,
                is_valid=False,
                error_message=str(exc),
            )
