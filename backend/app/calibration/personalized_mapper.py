"""
VoxGaze Calibration: Personalized Gaze-to-Screen Coordinate Mapper (M1.5 Experimental)

Learns a personalized screen-coordinate mapping (x, y in [0, 1]) from labelled
calibration targets using regularized linear and polynomial Ridge regression.

RESEARCH & SAFETY INVARIANTS:
1. Experimental Gate: This mapper is an experimental research module; live AAC selection
   remains STRICTLY disabled until held-out validation gates pass.
2. Strict Phase Separation: Must train on Phase 1 (CALIBRATION_ESTIMATION) ONLY and evaluate
   on independent held-out Phase 2 (INDEPENDENT_EVALUATION). Zero data leakage permitted.
3. Regularization & Numerical Stability: Ridge L2 regularization prevents overfitting on small
   calibration sets and guarantees invertible normal equations.
4. Abstention on Degradation: Returns is_valid=False upon non-finite or missing input features.
5. Latency Profiling: Microsecond-scale mapping latency measured on every prediction.
"""

from dataclasses import dataclass
from enum import Enum
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class FeatureSetType(str, Enum):
    ANGLES_ONLY = "ANGLES_ONLY"          # [yaw, pitch]
    ANGLES_POLY = "ANGLES_POLY"          # [yaw, pitch, yaw^2, pitch^2, yaw*pitch]
    GEOMETRY_ONLY = "GEOMETRY_ONLY"      # [right_h, right_v, left_h, left_v, canthal_normal]
    HYBRID = "HYBRID"                    # [yaw, pitch, right_h, right_v, left_h, left_v, canthal_normal]


@dataclass
class CalibrationSample:
    """Individual observation sample with screen target coordinates and input features."""

    target_x: float
    target_y: float
    target_direction: str  # "CENTER", "LEFT", "RIGHT", "UP", "DOWN"
    yaw_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    right_eye_h: Optional[float] = None
    right_eye_v: Optional[float] = None
    left_eye_h: Optional[float] = None
    left_eye_v: Optional[float] = None
    canthal_normal: Optional[float] = None
    condition_name: Optional[str] = None
    trial_index: Optional[int] = None
    is_settling: bool = False


@dataclass(frozen=True)
class MappingPrediction:
    """Result of mapping a gaze feature sample to screen space."""

    screen_x: float
    screen_y: float
    is_valid: bool
    mapping_latency_us: float
    predicted_direction: str  # Closest discrete target direction
    error_message: Optional[str] = None


class PersonalizedGazeMapper:
    """
    Experimental personalized regression mapper converting gaze signals to screen coordinates.
    """

    TARGET_POSITIONS: Dict[str, Tuple[float, float]] = {
        "CENTER": (0.50, 0.50),
        "LEFT": (0.15, 0.50),
        "RIGHT": (0.85, 0.50),
        "UP": (0.50, 0.15),
        "DOWN": (0.50, 0.85),
    }

    # Standard directional AAC bounding boxes matching actual UI touch/dwell target buttons
    TARGET_BOUNDING_BOXES: Dict[str, Tuple[float, float, float, float]] = {
        "CENTER": (0.35, 0.35, 0.65, 0.65),  # Resting center deadzone (no selection)
        "LEFT": (0.00, 0.30, 0.35, 0.70),    # Left button area
        "RIGHT": (0.65, 0.30, 1.00, 0.70),   # Right button area
        "UP": (0.30, 0.00, 0.70, 0.35),      # Top button area
        "DOWN": (0.30, 0.65, 0.70, 1.00),    # Bottom button area
    }

    def __init__(
        self,
        feature_set: FeatureSetType = FeatureSetType.ANGLES_ONLY,
        alpha: float = 1.0,
    ) -> None:
        """
        Args:
            feature_set: Which subset of features to construct.
            alpha: L2 Ridge regularization multiplier (higher = stronger regularization).
        """
        self.feature_set = feature_set
        self.alpha = float(alpha)
        self.is_fitted: bool = False

        # Learned regression parameters
        self.weights: Optional[np.ndarray] = None  # Shape (D, 2)
        self.bias: Optional[np.ndarray] = None     # Shape (2,)
        self.mean_x: Optional[np.ndarray] = None   # Shape (D,)
        self.std_x: Optional[np.ndarray] = None    # Shape (D,)

    def extract_features(self, sample: CalibrationSample) -> Optional[np.ndarray]:
        """
        Extracts and formats the feature vector for a single sample according to self.feature_set.
        Returns None if any required feature is missing or non-finite.
        """
        yaw = sample.yaw_deg
        pitch = sample.pitch_deg
        rh = sample.right_eye_h
        rv = sample.right_eye_v
        lh = sample.left_eye_h
        lv = sample.left_eye_v
        cn = sample.canthal_normal if sample.canthal_normal is not None else 0.0

        if self.feature_set == FeatureSetType.ANGLES_ONLY:
            if yaw is None or pitch is None or not (math.isfinite(yaw) and math.isfinite(pitch)):
                return None
            return np.array([yaw, pitch], dtype=np.float64)

        elif self.feature_set == FeatureSetType.ANGLES_POLY:
            if yaw is None or pitch is None or not (math.isfinite(yaw) and math.isfinite(pitch)):
                return None
            return np.array([yaw, pitch, yaw ** 2, pitch ** 2, yaw * pitch], dtype=np.float64)

        elif self.feature_set == FeatureSetType.GEOMETRY_ONLY:
            vals = [rh, rv, lh, lv, cn]
            if any(v is None or not math.isfinite(v) for v in vals):
                return None
            return np.array(vals, dtype=np.float64)

        elif self.feature_set == FeatureSetType.HYBRID:
            vals = [yaw, pitch, rh, rv, lh, lv, cn]
            if any(v is None or not math.isfinite(v) for v in vals):
                return None
            return np.array(vals, dtype=np.float64)

        return None

    def fit(self, training_samples: List[CalibrationSample]) -> bool:
        """
        Fits regularized Ridge regression weights on training samples (Phase 1).

        Args:
            training_samples: Calibration observations collected during Phase 1.

        Returns:
            True if fitting succeeded with >= 5 valid samples; False otherwise.
        """
        x_list: List[np.ndarray] = []
        y_list: List[Tuple[float, float]] = []

        for s in training_samples:
            feat = self.extract_features(s)
            if feat is not None and math.isfinite(s.target_x) and math.isfinite(s.target_y):
                x_list.append(feat)
                y_list.append((float(s.target_x), float(s.target_y)))

        if len(x_list) < 5:
            self.is_fitted = False
            return False

        X = np.array(x_list, dtype=np.float64)  # Shape (N, D)
        Y = np.array(y_list, dtype=np.float64)  # Shape (N, 2)

        # Standardize features
        mean_x = np.mean(X, axis=0)
        std_x = np.std(X, axis=0)
        std_x[std_x < 1e-6] = 1.0  # Prevent division by zero for invariant features

        X_norm = (X - mean_x) / std_x

        # Center targets to solve for weights without penalizing intercept
        mean_y = np.mean(Y, axis=0)
        Y_centered = Y - mean_y

        N, D = X_norm.shape
        # Ridge normal equations: (X_norm^T X_norm + alpha * I) W = X_norm^T Y_centered
        A = X_norm.T @ X_norm + self.alpha * np.eye(D, dtype=np.float64)
        B = X_norm.T @ Y_centered

        try:
            W = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            self.is_fitted = False
            return False

        self.weights = W
        self.bias = mean_y
        self.mean_x = mean_x
        self.std_x = std_x
        self.is_fitted = True
        return True

    def predict(self, sample: CalibrationSample) -> MappingPrediction:
        """
        Maps a gaze sample to continuous screen coordinates [0, 1] x [0, 1].
        """
        t0 = time.perf_counter()

        if not self.is_fitted or self.weights is None or self.bias is None:
            return MappingPrediction(
                screen_x=0.50,
                screen_y=0.50,
                is_valid=False,
                mapping_latency_us=0.0,
                predicted_direction="CENTER",
                error_message="Mapper is not fitted.",
            )

        feat = self.extract_features(sample)
        if feat is None:
            t_elapsed_us = (time.perf_counter() - t0) * 1e6
            return MappingPrediction(
                screen_x=0.50,
                screen_y=0.50,
                is_valid=False,
                mapping_latency_us=t_elapsed_us,
                predicted_direction="CENTER",
                error_message="Missing or non-finite features.",
            )

        # Normalize and predict
        feat_norm = (feat - self.mean_x) / self.std_x
        pred = feat_norm @ self.weights + self.bias

        # Clamp to screen boundary [0, 1]
        clamped_x = float(np.clip(pred[0], 0.0, 1.0))
        clamped_y = float(np.clip(pred[1], 0.0, 1.0))

        # Determine closest discrete direction
        closest_dir = self._classify_closest_direction(clamped_x, clamped_y)

        t_elapsed_us = (time.perf_counter() - t0) * 1e6

        return MappingPrediction(
            screen_x=round(clamped_x, 4),
            screen_y=round(clamped_y, 4),
            is_valid=True,
            mapping_latency_us=round(t_elapsed_us, 2),
            predicted_direction=closest_dir,
        )

    def _classify_closest_direction(self, x: float, y: float) -> str:
        """Finds closest canonical target direction to (x, y)."""
        min_dist = float("inf")
        best_dir = "CENTER"
        for name, (tx, ty) in self.TARGET_POSITIONS.items():
            dist = math.hypot(x - tx, y - ty)
            if dist < min_dist:
                min_dist = dist
                best_dir = name
        return best_dir

    def is_inside_bbox(self, x: float, y: float, bbox: Tuple[float, float, float, float]) -> bool:
        """Determines if normalized point (x, y) is inside [x_min, y_min, x_max, y_max]."""
        x_min, y_min, x_max, y_max = bbox
        return (x_min <= x <= x_max) and (y_min <= y <= y_max)

    def evaluate_held_out(
        self,
        test_samples: List[CalibrationSample],
        target_hit_radius: float = 0.22,
        center_zone_radius: float = 0.20,
    ) -> Dict[str, Any]:
        """
        Evaluates the fitted mapper on held-out test trials (Phase 2).
        Reports BOTH:
        1. Trial-Level Metrics: Groups frames by independent trial condition and tests the trial median
           against realistic AAC target bounding boxes (respecting temporal correlation).
        2. Frame-Level Metrics: Instantaneous jitter, error distributions, and per-frame hit rates.

        Args:
            test_samples: Held-out samples from Phase 2.
            target_hit_radius: Maximum Euclidean distance for frame hit heuristic.
            center_zone_radius: Maximum Euclidean distance from center to count as center deadzone.

        Returns:
            Dictionary of empirical metrics: trial-level accuracy, frame-level distributions,
            center-return false activation rates, and latency.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot evaluate unfitted mapper.")

        # Filter out settling frames if marked
        eval_samples = [s for s in test_samples if not getattr(s, "is_settling", False)]
        if not eval_samples:
            eval_samples = test_samples

        errors: List[float] = []
        x_errors: List[float] = []
        y_errors: List[float] = []
        latencies_us: List[float] = []

        # Frame-level accumulators
        total_peripheral_frames = 0
        correct_peripheral_frames = 0
        total_center_frames = 0
        center_false_activation_frames = 0

        # Grouping for trial-level evaluation
        trials: Dict[str, List[Tuple[float, float, str]]] = {}
        trial_targets: Dict[str, str] = {}

        valid_samples = 0
        total_samples = len(eval_samples)

        for s in eval_samples:
            pred = self.predict(s)
            if not pred.is_valid:
                continue

            valid_samples += 1
            latencies_us.append(pred.mapping_latency_us)

            # Euclidean error against continuous target
            err = math.hypot(pred.screen_x - s.target_x, pred.screen_y - s.target_y)
            errors.append(err)
            x_errors.append(abs(pred.screen_x - s.target_x))
            y_errors.append(abs(pred.screen_y - s.target_y))

            # Frame-level selection accuracy evaluation
            target_bbox = self.TARGET_BOUNDING_BOXES.get(s.target_direction)
            if s.target_direction in ("LEFT", "RIGHT", "UP", "DOWN"):
                total_peripheral_frames += 1
                in_box = self.is_inside_bbox(pred.screen_x, pred.screen_y, target_bbox) if target_bbox else False
                if in_box or (err <= target_hit_radius and pred.predicted_direction == s.target_direction):
                    correct_peripheral_frames += 1

            elif s.target_direction == "CENTER":
                total_center_frames += 1
                # If during center gaze, predicted point escapes center zone into a peripheral button
                center_dist = math.hypot(pred.screen_x - 0.50, pred.screen_y - 0.50)
                escaped_radius = center_dist > center_zone_radius
                escaped_box = not self.is_inside_bbox(pred.screen_x, pred.screen_y, self.TARGET_BOUNDING_BOXES["CENTER"])
                if (escaped_radius or escaped_box) and pred.predicted_direction != "CENTER":
                    center_false_activation_frames += 1

            # Trial grouping by condition_name or trial_index
            trial_key = s.condition_name if s.condition_name else (f"trial_{s.trial_index}" if s.trial_index is not None else s.target_direction)
            if trial_key not in trials:
                trials[trial_key] = []
                trial_targets[trial_key] = s.target_direction
            trials[trial_key].append((pred.screen_x, pred.screen_y, pred.predicted_direction))

        if not errors:
            return {
                "valid_sample_rate": 0.0,
                "error": "No valid predictions on test set.",
            }

        # -------------------------------------------------------------------
        # Trial-Level Aggregation (Accounts for temporal correlation)
        # -------------------------------------------------------------------
        trial_results = []
        peripheral_trials_total = 0
        peripheral_trials_passed = 0
        center_trials_total = 0
        center_trials_false_act = 0

        for t_name, pts in trials.items():
            if not pts:
                continue
            t_dir = trial_targets[t_name]
            med_x = float(np.median([p[0] for p in pts]))
            med_y = float(np.median([p[1] for p in pts]))
            target_bbox = self.TARGET_BOUNDING_BOXES.get(t_dir, (0.0, 0.0, 1.0, 1.0))

            if t_dir in ("LEFT", "RIGHT", "UP", "DOWN"):
                peripheral_trials_total += 1
                hit = self.is_inside_bbox(med_x, med_y, target_bbox)
                if hit:
                    peripheral_trials_passed += 1
                trial_results.append({
                    "trial_name": t_name,
                    "target_direction": t_dir,
                    "median_x": round(med_x, 4),
                    "median_y": round(med_y, 4),
                    "hit": hit,
                    "frames_in_trial": len(pts),
                })

            elif t_dir == "CENTER":
                center_trials_total += 1
                # Center trial passes if median remains inside resting deadzone
                stayed_center = self.is_inside_bbox(med_x, med_y, self.TARGET_BOUNDING_BOXES["CENTER"])
                if not stayed_center:
                    center_trials_false_act += 1
                trial_results.append({
                    "trial_name": t_name,
                    "target_direction": t_dir,
                    "median_x": round(med_x, 4),
                    "median_y": round(med_y, 4),
                    "hit": stayed_center,
                    "frames_in_trial": len(pts),
                })

        trial_selection_accuracy = (
            float(peripheral_trials_passed / peripheral_trials_total)
            if peripheral_trials_total > 0
            else 0.0
        )
        trial_center_false_rate = (
            float(center_trials_false_act / center_trials_total)
            if center_trials_total > 0
            else 0.0
        )

        frame_selection_accuracy = (
            float(correct_peripheral_frames / total_peripheral_frames)
            if total_peripheral_frames > 0
            else 0.0
        )
        frame_center_false_rate = (
            float(center_false_activation_frames / total_center_frames)
            if total_center_frames > 0
            else 0.0
        )

        return {
            "feature_set": self.feature_set.value,
            "alpha": self.alpha,
            "total_test_samples": total_samples,
            "valid_test_samples": valid_samples,
            "valid_sample_rate": round(valid_samples / total_samples, 4) if total_samples > 0 else 0.0,
            # Continuous error distribution
            "mean_target_error": round(float(np.mean(errors)), 4),
            "median_target_error": round(float(np.median(errors)), 4),
            "p95_target_error": round(float(np.percentile(errors, 95)), 4),
            "x_mae": round(float(np.mean(x_errors)), 4),
            "y_mae": round(float(np.mean(y_errors)), 4),
            # Trial-level metrics (Primary AAC Command Decision Unit)
            "trial_selection_accuracy": round(trial_selection_accuracy, 4),
            "trial_center_false_activation_rate": round(trial_center_false_rate, 4),
            "peripheral_trials_total": peripheral_trials_total,
            "peripheral_trials_passed": peripheral_trials_passed,
            "center_trials_total": center_trials_total,
            "center_trials_false_act": center_trials_false_act,
            "trial_results": trial_results,
            # Frame-level metrics (Jitter / Stability)
            "frame_selection_accuracy": round(frame_selection_accuracy, 4),
            "frame_center_false_activation_rate": round(frame_center_false_rate, 4),
            "selection_accuracy": round(trial_selection_accuracy, 4),  # Default alias to trial accuracy
            "center_false_activation_rate": round(trial_center_false_rate, 4),
            # Latency profiling
            "latency_p50_us": round(float(np.percentile(latencies_us, 50)), 2),
            "latency_p95_us": round(float(np.percentile(latencies_us, 95)), 2),
        }
