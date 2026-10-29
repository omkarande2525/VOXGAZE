"""
VoxGaze Milestone M1 / Controlled Sequential Model Comparison Protocol

Runs a compact, balanced, 3-block 5-point gaze trial sequentially comparing
MobileOne S0 vs ResNet-34 on identical in-memory frames.

RESEARCH & SCIENTIFIC INVARIANTS:
1. Sequential Execution: Runs MobileOne S0 and ResNet-34 sequentially on the same in-memory crop,
   preventing CPU starvation and ensuring zero thread-interference in the capture loop.
2. Balanced Order: Pseudo-randomizes peripheral target sequences across blocks to eliminate directional bias.
3. Strict Train/Test Separation: Fits personalized mappers on Blocks 1-2 ONLY and evaluates on Block 3.
4. Comprehensive Error Profiling:
   - Trial-level median selection accuracy against AAC button bounding boxes
   - Sustained dwell success (>= 400 ms continuous fixation in target box)
   - Transient false activations during center fixation
   - End-to-end latency (camera -> landmarks -> crop -> inference -> mapper)
5. Zero Frame Storage: Streams numerical records only; zero raw frames written to disk (Privacy Invariant #7).
6. 4-Direction Research Mode: Explicitly isolates and reports 4-direction mode (omitting UP) as a labeled research contrast.
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.calibration.personalized_mapper import (
    CalibrationSample,
    FeatureSetType,
    PersonalizedGazeMapper,
)
from backend.app.core.contracts import Point2D
from backend.app.perception.face_crop import extract_face_crop
from backend.app.perception.geometry import (
    compute_canthal_normal_distance_px,
    extract_eye_features,
)
from backend.app.perception.mobileone_estimator import MobileOneGazeEstimator
from backend.app.perception.model_registry import create_gaze_estimator
from backend.app.perception.resnet34_estimator import ResNet34GazeEstimator

# Landmark indices from canonical MediaPipe mesh
RIGHT_IRIS = 468
LEFT_IRIS = 473
RIGHT_INNER_CANTHUS = 133
RIGHT_OUTER_CANTHUS = 33
RIGHT_UPPER_LID = 159
RIGHT_LOWER_LID = 145
LEFT_INNER_CANTHUS = 362
LEFT_OUTER_CANTHUS = 263
LEFT_UPPER_LID = 386
LEFT_LOWER_LID = 374

# Target screen coordinates (normalized [0, 1])
TARGET_COORDS: Dict[str, Tuple[float, float]] = {
    "CENTER": (0.50, 0.50),
    "LEFT": (0.15, 0.50),
    "RIGHT": (0.85, 0.50),
    "UP": (0.50, 0.15),
    "DOWN": (0.50, 0.85),
}

# Standard AAC button bounding boxes
TARGET_BBOXES: Dict[str, Tuple[float, float, float, float]] = {
    "CENTER": (0.35, 0.35, 0.65, 0.65),  # Resting deadzone
    "LEFT": (0.00, 0.30, 0.35, 0.70),    # Left button
    "RIGHT": (0.65, 0.30, 1.00, 0.70),   # Right button
    "UP": (0.30, 0.00, 0.70, 0.35),      # Top button
    "DOWN": (0.30, 0.65, 0.70, 1.00),    # Bottom button
}

# Balanced sequences for the 5 blocks (interleaved center returns)
BLOCK_1_TARGETS: List[str] = ["CENTER", "LEFT", "CENTER", "RIGHT", "CENTER", "UP", "CENTER", "DOWN", "CENTER"]
BLOCK_2_TARGETS: List[str] = ["CENTER", "RIGHT", "CENTER", "DOWN", "CENTER", "LEFT", "CENTER", "UP", "CENTER"]
BLOCK_3_TARGETS: List[str] = ["CENTER", "DOWN", "CENTER", "LEFT", "CENTER", "UP", "CENTER", "RIGHT", "CENTER"]
BLOCK_4_TARGETS: List[str] = ["CENTER", "UP", "CENTER", "RIGHT", "CENTER", "DOWN", "CENTER", "LEFT", "CENTER"]
BLOCK_5_TARGETS: List[str] = ["CENTER", "LEFT", "CENTER", "UP", "CENTER", "DOWN", "CENTER", "RIGHT", "CENTER"]

BLOCK_TARGETS_MAP: Dict[int, List[str]] = {
    1: BLOCK_1_TARGETS,
    2: BLOCK_2_TARGETS,
    3: BLOCK_3_TARGETS,
    4: BLOCK_4_TARGETS,
    5: BLOCK_5_TARGETS,
}


def emit_beep(freq: int = 880, dur_ms: int = 120) -> None:
    """Emits short non-blocking notification tone on Windows."""
    try:
        import winsound
        winsound.Beep(freq, dur_ms)
    except Exception:
        pass


@dataclass
class FrameComparisonRecord:
    """Synchronous frame record logging both models on the exact same video frame."""

    session_id: str
    block_index: int
    condition_index: int
    trial_name: str
    target_direction: str
    frame_index: int
    monotonic_time_s: float
    face_detected: bool
    face_crop_valid: bool
    # Ocular geometry
    right_eye_h: Optional[float] = None
    right_eye_v: Optional[float] = None
    right_eye_ear: Optional[float] = None
    left_eye_h: Optional[float] = None
    left_eye_v: Optional[float] = None
    left_eye_ear: Optional[float] = None
    canthal_normal_px: Optional[float] = None
    # MobileOne S0 outputs & latencies
    m1_valid: bool = False
    m1_yaw_deg: Optional[float] = None
    m1_pitch_deg: Optional[float] = None
    m1_pre_ms: float = 0.0
    m1_infer_ms: float = 0.0
    m1_end_to_end_ms: float = 0.0
    # ResNet-34 outputs & latencies
    r34_valid: bool = False
    r34_yaw_deg: Optional[float] = None
    r34_pitch_deg: Optional[float] = None
    r34_pre_ms: float = 0.0
    r34_infer_ms: float = 0.0
    r34_end_to_end_ms: float = 0.0
    # System latencies
    mediapipe_ms: float = 0.0
    crop_ms: float = 0.0
    total_pipeline_ms: float = 0.0


def is_inside_box(x: float, y: float, bbox: Tuple[float, float, float, float]) -> bool:
    return (bbox[0] <= x <= bbox[2]) and (bbox[1] <= y <= bbox[3])


def evaluate_trial_dwell(
    predictions: List[Any],  # List of (x, y, t_s) or (x, y)
    target_bbox: Tuple[float, float, float, float],
    min_dwell_s: float = 0.400,
    min_dwell_frames: int = 12,
) -> bool:
    """
    Checks whether predictions maintain continuous dwell in target box.
    Uses real monotonic timestamps (>= min_dwell_s) when available in tuples of length >= 3,
    falling back to frame count (>= min_dwell_frames) for 2-tuples.
    """
    if not predictions:
        return False

    has_timestamps = len(predictions[0]) >= 3 if isinstance(predictions[0], (tuple, list)) else False

    if has_timestamps:
        dwell_start_s: Optional[float] = None
        for item in predictions:
            x, y, t_s = item[0], item[1], item[2]
            if is_inside_box(x, y, target_bbox):
                if dwell_start_s is None:
                    dwell_start_s = t_s
                elif (t_s - dwell_start_s) >= min_dwell_s:
                    return True
            else:
                dwell_start_s = None
        return False
    else:
        consecutive = 0
        for item in predictions:
            x, y = item[0], item[1]
            if is_inside_box(x, y, target_bbox):
                consecutive += 1
                if consecutive >= min_dwell_frames:
                    return True
            else:
                consecutive = 0
        return False


MIN_HELD_OUT_PERIPHERAL_TRIALS = 12
MIN_HELD_OUT_CENTER_TRIALS = 15
MIN_INDEPENDENT_SESSIONS = 2
MIN_PERIPHERAL_ACCURACY_THRESHOLD = 0.85
MAX_CENTER_FALSE_ACTIVATION_THRESHOLD = 0.05


def evaluate_model_adoption(
    eval_5dir: Optional[Dict[str, Any]],
    latency_info: Optional[Dict[str, Any]],
    sessions_passed: int = 1,
    is_mock: bool = False,
    model_name: str = "",
) -> Dict[str, Any]:
    """
    Evaluates strict non-negotiable adoption criteria:
    1. Not mock (mock reports can never confer adoption).
    2. Valid end-to-end latency (p50 > 0, finite, not None).
    3. Peripheral accuracy >= 85.0% on >= 12 held-out trials.
    4. Center false activations <= 5.0% on >= 15 held-out trials.
    5. Both passing across >= 2 independent sessions.

    Produces explicit INVALID/INCOMPLETE status on missing fields, empty trials, or NaN.
    """
    reasons: List[str] = []
    validation_label = "MOCK, not physical validation" if is_mock else "PHYSICAL"
    if is_mock:
        reasons.append("MOCK report: Synthetic/mock data can never confer adoption or physical validation.")

    # Check for empty, missing, or error evaluation dictionary
    if (
        eval_5dir is None
        or not isinstance(eval_5dir, dict)
        or eval_5dir.get("status") in ("INVALID_EMPTY_DATA", "INSUFFICIENT_DATA", "FIT_FAILED", "INSUFFICIENT_VALID_SAMPLES")
    ):
        status_val = eval_5dir.get("status", "INVALID_OR_INCOMPLETE_DATA") if isinstance(eval_5dir, dict) else "INVALID_OR_INCOMPLETE_DATA"
        reasons.append(f"Incomplete or invalid evaluation data (status={status_val}).")
        return {
            "model_name": model_name,
            "adopted": False,
            "status": status_val if status_val != "VALID" else "INVALID_OR_INCOMPLETE_DATA",
            "is_preliminary": True,
            "sample_size_status": "PRELIMINARY",
            "validation_label": validation_label,
            "reasons": reasons,
            "metrics": {
                "peripheral_trials_passed": 0,
                "peripheral_trials_total": 0,
                "peripheral_trial_accuracy": 0.0,
                "peripheral_trial_accuracy_display": "0/0 (0.0%)",
                "center_trials_false_act": 0,
                "center_trials_stayed": 0,
                "center_trials_total": 0,
                "center_return_false_activation_rate": 0.0,
                "center_return_false_activation_display": "0/0 (0.0%)",
                "overall_trials_correct": 0,
                "overall_trials_total": 0,
                "overall_trial_correctness_rate": 0.0,
                "overall_trial_correctness_display": "0/0 (0.0%)",
                "latency_p50_ms": None,
                "independent_sessions_passed": sessions_passed,
                "independent_sessions_required": MIN_INDEPENDENT_SESSIONS,
            },
            "criteria_gates": {
                "non_mock": not is_mock,
                "valid_latency": False,
                "peripheral_accuracy": False,
                "peripheral_sample_size": False,
                "center_false_activation": False,
                "center_sample_size": False,
                "cross_session_generalization": False,
            },
        }

    # Extract metrics safely
    peri_pass = eval_5dir.get("peripheral_trials_passed")
    peri_tot = eval_5dir.get("peripheral_trials_total")
    peri_acc = eval_5dir.get("peripheral_trial_accuracy")
    if peri_acc is None:
        peri_acc = eval_5dir.get("trial_selection_accuracy")

    center_fa = eval_5dir.get("center_trials_false_act")
    center_tot = eval_5dir.get("center_trials_total")
    center_fa_rate = eval_5dir.get("center_return_false_activation_rate")
    if center_fa_rate is None:
        center_fa_rate = eval_5dir.get("trial_center_false_activation_rate")

    overall_correct = eval_5dir.get("overall_trials_correct")
    overall_tot = eval_5dir.get("overall_trials_total")
    overall_rate = eval_5dir.get("overall_trial_correctness_rate")

    # Check for NaN / None / non-finite in numerical metrics
    has_nan = False
    for v in (peri_acc, center_fa_rate):
        if v is None or not math.isfinite(v):
            has_nan = True
            break

    if has_nan or peri_tot is None or center_tot is None or peri_pass is None or center_fa is None:
        reasons.append("Non-finite, NaN, or missing metric values detected in trial results.")
        return {
            "model_name": model_name,
            "adopted": False,
            "status": "INVALID_NAN_OR_MISSING_METRICS",
            "is_preliminary": True,
            "sample_size_status": "PRELIMINARY",
            "validation_label": validation_label,
            "reasons": reasons,
            "metrics": {
                "peripheral_trials_passed": peri_pass or 0,
                "peripheral_trials_total": peri_tot or 0,
                "peripheral_trial_accuracy": 0.0 if (peri_acc is None or not math.isfinite(peri_acc)) else peri_acc,
                "peripheral_trial_accuracy_display": f"{peri_pass or 0}/{peri_tot or 0}",
                "center_trials_false_act": center_fa or 0,
                "center_trials_stayed": (center_tot or 0) - (center_fa or 0),
                "center_trials_total": center_tot or 0,
                "center_return_false_activation_rate": 0.0 if (center_fa_rate is None or not math.isfinite(center_fa_rate)) else center_fa_rate,
                "center_return_false_activation_display": f"{center_fa or 0}/{center_tot or 0}",
                "overall_trials_correct": overall_correct or 0,
                "overall_trials_total": overall_tot or 0,
                "overall_trial_correctness_rate": 0.0 if (overall_rate is None or not math.isfinite(overall_rate)) else overall_rate,
                "overall_trial_correctness_display": f"{overall_correct or 0}/{overall_tot or 0}",
                "latency_p50_ms": None,
                "independent_sessions_passed": sessions_passed,
                "independent_sessions_required": MIN_INDEPENDENT_SESSIONS,
            },
            "criteria_gates": {
                "non_mock": not is_mock,
                "valid_latency": False,
                "peripheral_accuracy": False,
                "peripheral_sample_size": False,
                "center_false_activation": False,
                "center_sample_size": False,
                "cross_session_generalization": False,
            },
        }

    center_stayed = center_tot - center_fa
    if overall_tot is None:
        overall_tot = peri_tot + center_tot
    if overall_correct is None:
        overall_correct = peri_pass + center_stayed
    if overall_rate is None:
        overall_rate = float(overall_correct / overall_tot) if overall_tot > 0 else 0.0

    # Latency check
    lat_p50 = None
    valid_latency = False
    if latency_info and isinstance(latency_info, dict):
        lat_status = latency_info.get("status", "VALID")
        lat_val = latency_info.get("p50")
        if lat_status != "INVALID_MISSING_LATENCY" and lat_val is not None and math.isfinite(lat_val) and lat_val > 0:
            lat_p50 = lat_val
            valid_latency = True
        else:
            reasons.append("End-to-end latency is missing, zero, or non-finite.")
    else:
        reasons.append("Latency info dictionary is missing or invalid.")

    # Check adoption gates
    non_mock_pass = not is_mock
    latency_pass = valid_latency
    peri_sample_pass = (peri_tot >= MIN_HELD_OUT_PERIPHERAL_TRIALS)
    center_sample_pass = (center_tot >= MIN_HELD_OUT_CENTER_TRIALS)
    peri_acc_pass = (peri_acc >= MIN_PERIPHERAL_ACCURACY_THRESHOLD)
    center_fa_pass = (center_fa_rate <= MAX_CENTER_FALSE_ACTIVATION_THRESHOLD)
    session_pass = (sessions_passed >= MIN_INDEPENDENT_SESSIONS)

    is_preliminary = (not peri_sample_pass) or (not center_sample_pass) or (not session_pass)
    sample_size_status = "PRELIMINARY" if is_preliminary else "CONCLUSIVE"

    if not peri_sample_pass:
        reasons.append(f"Insufficient peripheral trials: {peri_tot}/{MIN_HELD_OUT_PERIPHERAL_TRIALS} held-out trials (PRELIMINARY).")
    if not center_sample_pass:
        reasons.append(f"Insufficient center trials: {center_tot}/{MIN_HELD_OUT_CENTER_TRIALS} held-out trials (PRELIMINARY).")
    if not session_pass:
        reasons.append(f"Insufficient independent sessions: {sessions_passed}/{MIN_INDEPENDENT_SESSIONS} required across independent sessions.")
    if not peri_acc_pass:
        reasons.append(f"Peripheral accuracy {peri_acc*100:.1f}% ({peri_pass}/{peri_tot}) is below required {MIN_PERIPHERAL_ACCURACY_THRESHOLD*100:.1f}%.")
    if not center_fa_pass:
        reasons.append(f"Center false activation rate {center_fa_rate*100:.1f}% ({center_fa}/{center_tot}) exceeds allowable {MAX_CENTER_FALSE_ACTIVATION_THRESHOLD*100:.1f}%.")

    all_gates_passed = (
        non_mock_pass
        and latency_pass
        and peri_sample_pass
        and center_sample_pass
        and peri_acc_pass
        and center_fa_pass
        and session_pass
    )

    if all_gates_passed:
        adopted = True
        status = "ADOPTED_VALIDATED"
    else:
        adopted = False
        if is_mock:
            status = "MOCK_REJECTED"
        elif not valid_latency:
            status = "INVALID_MISSING_LATENCY"
        elif not peri_sample_pass or not center_sample_pass:
            status = "PRELIMINARY_SAMPLE_SIZE"
        elif not session_pass:
            status = "PRELIMINARY_INSUFFICIENT_SESSIONS"
        elif not peri_acc_pass:
            status = "REJECTED_PERIPHERAL_ACCURACY_LOW"
        elif not center_fa_pass:
            status = "REJECTED_CENTER_FALSE_ACTIVATION_HIGH"
        else:
            status = "REJECTED"

    return {
        "model_name": model_name,
        "adopted": adopted,
        "status": status,
        "is_preliminary": is_preliminary,
        "sample_size_status": sample_size_status,
        "validation_label": validation_label,
        "reasons": reasons,
        "metrics": {
            "peripheral_trials_passed": peri_pass,
            "peripheral_trials_total": peri_tot,
            "peripheral_trial_accuracy": round(peri_acc, 4),
            "peripheral_trial_accuracy_display": f"{peri_pass}/{peri_tot} ({peri_acc*100:.1f}%)",
            "center_trials_false_act": center_fa,
            "center_trials_stayed": center_stayed,
            "center_trials_total": center_tot,
            "center_return_false_activation_rate": round(center_fa_rate, 4),
            "center_return_false_activation_display": f"{center_fa}/{center_tot} ({center_fa_rate*100:.1f}%)",
            "overall_trials_correct": overall_correct,
            "overall_trials_total": overall_tot,
            "overall_trial_correctness_rate": round(overall_rate, 4),
            "overall_trial_correctness_display": f"{overall_correct}/{overall_tot} ({overall_rate*100:.1f}%)",
            "latency_p50_ms": lat_p50,
            "independent_sessions_passed": sessions_passed,
            "independent_sessions_required": MIN_INDEPENDENT_SESSIONS,
        },
        "criteria_gates": {
            "non_mock": non_mock_pass,
            "valid_latency": latency_pass,
            "peripheral_accuracy": peri_acc_pass,
            "peripheral_sample_size": peri_sample_pass,
            "center_false_activation": center_fa_pass,
            "center_sample_size": center_sample_pass,
            "cross_session_generalization": session_pass,
        },
    }


def analyze_block_evaluation(
    records: List[FrameComparisonRecord],
    model_key: str,  # "m1" or "r34"
    feature_set: FeatureSetType = FeatureSetType.HYBRID,
) -> Dict[str, Any]:
    """
    Fits PersonalizedGazeMapper on calibration Blocks 1 & 2 and evaluates on
    held-out test Blocks 3, 4, 5 (12 peripheral trials, 15 center trials).

    Reports:
    - 5-direction and 4-direction (omitting UP) modes
    - Real timestamp-based continuous dwell (>= 400 ms)
    - True end-to-end target acquisition latency (onset to first box entry)
    - Transient false directional entries during center fixation (any frame vs median)
    """
    train_records = [r for r in records if r.block_index in (1, 2)]
    test_records = [r for r in records if r.block_index in (3, 4, 5)]

    empty_eval_dict = {
        "status": "INSUFFICIENT_DATA",
        "peripheral_trial_accuracy": 0.0,
        "peripheral_trials_passed": 0,
        "peripheral_trials_total": 0,
        "peripheral_trial_accuracy_display": "0/0 (0.0%)",
        "center_return_false_activation_rate": 0.0,
        "center_trials_false_act": 0,
        "center_trials_stayed": 0,
        "center_trials_total": 0,
        "center_return_false_activation_display": "0/0 (0.0%)",
        "overall_trial_correctness_rate": 0.0,
        "overall_trials_correct": 0,
        "overall_trials_total": 0,
        "overall_trial_correctness_display": "0/0 (0.0%)",
        "is_preliminary": True,
        "sample_size_status": "PRELIMINARY",
        "trial_selection_accuracy": 0.0,
        "trial_center_false_activation_rate": 0.0,
        "dwell_selection_accuracy": 0.0,
        "peripheral_dwell_accuracy": 0.0,
        "peripheral_dwell_passed": 0,
        "peripheral_dwell_total": 0,
        "peripheral_dwell_display": "0/0 (0.0%)",
        "trial_results": [],
    }

    if not train_records or not test_records:
        return {
            "status": "INSUFFICIENT_DATA",
            "model": "MobileOne S0" if model_key == "m1" else "ResNet-34",
            "feature_set": feature_set.value,
            "evaluation_5dir": empty_eval_dict,
            "target_acquisition_latency": {
                "p50_ms": None,
                "p95_ms": None,
                "mean_ms": None,
                "successful_acquisitions_count": 0,
                "total_peripheral_trials": 0,
                "acquisition_rate": 0.0,
                "per_trial_acquisitions": [],
            },
            "transient_center_escapes": {
                "total_center_trials": 0,
                "trials_with_transient_escapes": 0,
                "transient_escape_trial_rate": 0.0,
                "trials_with_transient_button_entries": 0,
                "transient_button_entry_trial_rate": 0.0,
                "total_center_frames": 0,
                "transient_escape_frames_count": 0,
                "transient_escape_frame_rate": 0.0,
                "transient_button_entry_frames_count": 0,
                "transient_button_entry_frame_rate": 0.0,
                "per_trial_breakdown": [],
            },
            "end_to_end_latency_ms": {"p50": None, "p95": None, "sample_count": 0, "status": "INVALID_MISSING_LATENCY"},
        }

    def _to_samples(recs: List[FrameComparisonRecord]) -> List[CalibrationSample]:
        samples = []
        for r in recs:
            if not r.face_crop_valid:
                continue
            yaw = r.m1_yaw_deg if model_key == "m1" else r.r34_yaw_deg
            pitch = r.m1_pitch_deg if model_key == "m1" else r.r34_pitch_deg
            valid = r.m1_valid if model_key == "m1" else r.r34_valid
            if not valid or yaw is None or pitch is None:
                continue

            tx, ty = TARGET_COORDS.get(r.target_direction, (0.5, 0.5))
            samples.append(
                CalibrationSample(
                    target_x=tx,
                    target_y=ty,
                    target_direction=r.target_direction,
                    yaw_deg=yaw,
                    pitch_deg=pitch,
                    right_eye_h=r.right_eye_h,
                    right_eye_v=r.right_eye_v,
                    left_eye_h=r.left_eye_h,
                    left_eye_v=r.left_eye_v,
                    canthal_normal=r.canthal_normal_px,
                    condition_name=r.trial_name,
                    trial_index=r.condition_index,
                )
            )
        return samples

    train_samples = _to_samples(train_records)
    test_samples = _to_samples(test_records)

    if len(train_samples) < 5 or len(test_samples) < 5:
        return {
            "status": "INSUFFICIENT_VALID_SAMPLES",
            "model": "MobileOne S0" if model_key == "m1" else "ResNet-34",
            "feature_set": feature_set.value,
            "evaluation_5dir": empty_eval_dict,
            "target_acquisition_latency": {
                "p50_ms": None,
                "p95_ms": None,
                "mean_ms": None,
                "successful_acquisitions_count": 0,
                "total_peripheral_trials": 0,
                "acquisition_rate": 0.0,
                "per_trial_acquisitions": [],
            },
            "transient_center_escapes": {
                "total_center_trials": 0,
                "trials_with_transient_escapes": 0,
                "transient_escape_trial_rate": 0.0,
                "trials_with_transient_button_entries": 0,
                "transient_button_entry_trial_rate": 0.0,
                "total_center_frames": 0,
                "transient_escape_frames_count": 0,
                "transient_escape_frame_rate": 0.0,
                "transient_button_entry_frames_count": 0,
                "transient_button_entry_frame_rate": 0.0,
                "per_trial_breakdown": [],
            },
            "end_to_end_latency_ms": {"p50": None, "p95": None, "sample_count": 0, "status": "INVALID_MISSING_LATENCY"},
        }

    # Separate mapper fitting per model on identical calibration trials
    mapper = PersonalizedGazeMapper(feature_set=feature_set, alpha=1.0)
    fit_ok = mapper.fit(train_samples)
    if not fit_ok:
        return {
            "status": "FIT_FAILED",
            "model": "MobileOne S0" if model_key == "m1" else "ResNet-34",
            "feature_set": feature_set.value,
            "evaluation_5dir": empty_eval_dict,
            "target_acquisition_latency": {
                "p50_ms": None,
                "p95_ms": None,
                "mean_ms": None,
                "successful_acquisitions_count": 0,
                "total_peripheral_trials": 0,
                "acquisition_rate": 0.0,
                "per_trial_acquisitions": [],
            },
            "transient_center_escapes": {
                "total_center_trials": 0,
                "trials_with_transient_escapes": 0,
                "transient_escape_trial_rate": 0.0,
                "trials_with_transient_button_entries": 0,
                "transient_button_entry_trial_rate": 0.0,
                "total_center_frames": 0,
                "transient_escape_frames_count": 0,
                "transient_escape_frame_rate": 0.0,
                "transient_button_entry_frames_count": 0,
                "transient_button_entry_frame_rate": 0.0,
                "per_trial_breakdown": [],
            },
            "end_to_end_latency_ms": {"p50": None, "p95": None, "sample_count": 0, "status": "INVALID_MISSING_LATENCY"},
        }

    # Base held-out evaluation (5-direction)
    eval_5dir = mapper.evaluate_held_out(test_samples)

    # 4-Direction Research Mode (Filter out UP trials)
    test_samples_4dir = [s for s in test_samples if s.target_direction != "UP"]
    eval_4dir = mapper.evaluate_held_out(test_samples_4dir)

    # Group held-out records with real timestamps for dwell and latency analysis
    trials_by_name: Dict[str, Dict[str, Any]] = {}
    for r in test_records:
        if not r.face_crop_valid:
            continue
        yaw = r.m1_yaw_deg if model_key == "m1" else r.r34_yaw_deg
        pitch = r.m1_pitch_deg if model_key == "m1" else r.r34_pitch_deg
        valid = r.m1_valid if model_key == "m1" else r.r34_valid
        if not valid or yaw is None or pitch is None:
            continue

        sample = CalibrationSample(
            target_x=TARGET_COORDS.get(r.target_direction, (0.5, 0.5))[0],
            target_y=TARGET_COORDS.get(r.target_direction, (0.5, 0.5))[1],
            target_direction=r.target_direction,
            yaw_deg=yaw,
            pitch_deg=pitch,
            right_eye_h=r.right_eye_h,
            right_eye_v=r.right_eye_v,
            left_eye_h=r.left_eye_h,
            left_eye_v=r.left_eye_v,
            canthal_normal=r.canthal_normal_px,
            condition_name=r.trial_name,
            trial_index=r.condition_index,
        )
        pred = mapper.predict(sample)
        if pred.is_valid and math.isfinite(pred.screen_x) and math.isfinite(pred.screen_y):
            t_name = r.trial_name
            if t_name not in trials_by_name:
                targets_seq = BLOCK_TARGETS_MAP.get(r.block_index, [])
                # r.condition_index is 1-based (1..len(targets_seq))
                prec_target = (
                    targets_seq[r.condition_index - 2]
                    if (r.condition_index >= 2 and (r.condition_index - 2) < len(targets_seq))
                    else "NONE_START_OF_BLOCK"
                )
                trials_by_name[t_name] = {
                    "target": r.target_direction,
                    "block_index": r.block_index,
                    "condition_index": r.condition_index,
                    "preceding_target": prec_target,
                    "preds_with_time": [],  # (x, y, t_s)
                }
            trials_by_name[t_name]["preds_with_time"].append((pred.screen_x, pred.screen_y, r.monotonic_time_s))

    # Real timestamp-based 400 ms continuous dwell evaluation
    dwell_passed = 0
    dwell_total_peripheral = 0
    for t_name, data in trials_by_name.items():
        t_dir = data["target"]
        if t_dir in ("LEFT", "RIGHT", "UP", "DOWN"):
            dwell_total_peripheral += 1
            bbox = TARGET_BBOXES[t_dir]
            if evaluate_trial_dwell(data["preds_with_time"], bbox, min_dwell_s=0.400):
                dwell_passed += 1

    dwell_accuracy = float(dwell_passed / dwell_total_peripheral) if dwell_total_peripheral else 0.0

    # True end-to-end target acquisition latency (time from target change to pointer first entering target box)
    acquisition_latencies_ms: List[float] = []
    per_trial_acquisitions: List[Dict[str, Any]] = []

    for t_name, data in trials_by_name.items():
        t_dir = data["target"]
        if t_dir in ("LEFT", "RIGHT", "UP", "DOWN"):
            bbox = TARGET_BBOXES[t_dir]
            preds = data["preds_with_time"]
            if not preds:
                continue
            t_onset_s = preds[0][2]
            entry_latency_ms: Optional[float] = None
            for x, y, t_s in preds:
                if is_inside_box(x, y, bbox):
                    entry_latency_ms = round((t_s - t_onset_s) * 1000.0, 2)
                    break

            per_trial_acquisitions.append({
                "trial_name": t_name,
                "target_direction": t_dir,
                "block_index": data.get("block_index"),
                "condition_index": data.get("condition_index"),
                "preceding_target": data.get("preceding_target"),
                "acquired": entry_latency_ms is not None,
                "latency_to_first_entry_ms": entry_latency_ms,
            })
            if entry_latency_ms is not None:
                acquisition_latencies_ms.append(entry_latency_ms)

    acq_p50 = float(np.percentile(acquisition_latencies_ms, 50)) if acquisition_latencies_ms else None
    acq_p95 = float(np.percentile(acquisition_latencies_ms, 95)) if acquisition_latencies_ms else None
    acq_mean = float(np.mean(acquisition_latencies_ms)) if acquisition_latencies_ms else None
    target_acq_stats = {
        "p50_ms": round(acq_p50, 2) if acq_p50 is not None else None,
        "p95_ms": round(acq_p95, 2) if acq_p95 is not None else None,
        "mean_ms": round(acq_mean, 2) if acq_mean is not None else None,
        "successful_acquisitions_count": len(acquisition_latencies_ms),
        "total_peripheral_trials": len(per_trial_acquisitions),
        "acquisition_rate": round(len(acquisition_latencies_ms) / len(per_trial_acquisitions), 4) if per_trial_acquisitions else 0.0,
        "per_trial_acquisitions": per_trial_acquisitions,
    }

    # Transient false directional entries during center fixation (any frame vs median)
    center_transient_trials: List[Dict[str, Any]] = []
    total_center_frames = 0
    total_frames_escaped_deadzone = 0
    total_frames_in_directional_buttons = 0
    trials_with_any_escape = 0
    trials_with_any_button_entry = 0

    center_bbox = TARGET_BBOXES["CENTER"]
    directional_bboxes = {d: TARGET_BBOXES[d] for d in ("LEFT", "RIGHT", "UP", "DOWN")}

    for t_name, data in trials_by_name.items():
        if data["target"] == "CENTER":
            preds = data["preds_with_time"]
            if not preds:
                continue
            n_frames = len(preds)
            total_center_frames += n_frames
            escaped_frames = 0
            button_frames = 0
            entered_buttons_by_dir: Dict[str, int] = {"LEFT": 0, "RIGHT": 0, "UP": 0, "DOWN": 0}

            for x, y, _ in preds:
                outside = not is_inside_box(x, y, center_bbox)
                if outside:
                    escaped_frames += 1
                for d_name, d_bbox in directional_bboxes.items():
                    if is_inside_box(x, y, d_bbox):
                        button_frames += 1
                        entered_buttons_by_dir[d_name] += 1
                        break

            med_x = float(np.median([p[0] for p in preds]))
            med_y = float(np.median([p[1] for p in preds]))
            median_stayed = is_inside_box(med_x, med_y, center_bbox)

            if escaped_frames > 0:
                trials_with_any_escape += 1
            if button_frames > 0:
                trials_with_any_button_entry += 1

            total_frames_escaped_deadzone += escaped_frames
            total_frames_in_directional_buttons += button_frames

            center_transient_trials.append({
                "trial_name": t_name,
                "block_index": data.get("block_index"),
                "condition_index": data.get("condition_index"),
                "preceding_target": data.get("preceding_target"),
                "total_frames": n_frames,
                "median_x": round(med_x, 4),
                "median_y": round(med_y, 4),
                "median_hit_deadzone": median_stayed,
                "frames_escaped_deadzone": escaped_frames,
                "frames_in_buttons": button_frames,
                "transient_clean": (escaped_frames == 0),
                "button_entries_by_dir": entered_buttons_by_dir,
            })

    total_center_trials = len(center_transient_trials)
    center_transient_summary = {
        "total_center_trials": total_center_trials,
        "trials_with_transient_escapes": trials_with_any_escape,
        "transient_escape_trial_rate": round(trials_with_any_escape / total_center_trials, 4) if total_center_trials else 0.0,
        "trials_with_transient_button_entries": trials_with_any_button_entry,
        "transient_button_entry_trial_rate": round(trials_with_any_button_entry / total_center_trials, 4) if total_center_trials else 0.0,
        "total_center_frames": total_center_frames,
        "transient_escape_frames_count": total_frames_escaped_deadzone,
        "transient_escape_frame_rate": round(total_frames_escaped_deadzone / total_center_frames, 4) if total_center_frames else 0.0,
        "transient_button_entry_frames_count": total_frames_in_directional_buttons,
        "transient_button_entry_frame_rate": round(total_frames_in_directional_buttons / total_center_frames, 4) if total_center_frames else 0.0,
        "per_trial_breakdown": center_transient_trials,
    }

    # Single-frame pipeline latency stats
    e2e_latencies = [
        (r.m1_end_to_end_ms if model_key == "m1" else r.r34_end_to_end_ms)
        for r in test_records
        if (r.m1_valid if model_key == "m1" else r.r34_valid)
    ]
    valid_latencies = [
        l for l in e2e_latencies
        if l is not None and math.isfinite(l) and l > 0
    ]
    if valid_latencies:
        e2e_p50 = round(float(np.percentile(valid_latencies, 50)), 2)
        e2e_p95 = round(float(np.percentile(valid_latencies, 95)), 2)
        lat_status = "VALID"
    else:
        e2e_p50 = None
        e2e_p95 = None
        lat_status = "INVALID_MISSING_LATENCY"

    return {
        "status": eval_5dir.get("status", "VALID"),
        "model": "MobileOne S0" if model_key == "m1" else "ResNet-34",
        "feature_set": feature_set.value,
        "evaluation_5dir": {
            "status": eval_5dir.get("status", "VALID"),
            "peripheral_trial_accuracy": eval_5dir.get("peripheral_trial_accuracy", 0.0),
            "peripheral_trials_passed": eval_5dir.get("peripheral_trials_passed", 0),
            "peripheral_trials_total": eval_5dir.get("peripheral_trials_total", 0),
            "peripheral_trial_accuracy_display": eval_5dir.get("peripheral_trial_accuracy_display", "0/0 (0.0%)"),
            "center_return_false_activation_rate": eval_5dir.get("center_return_false_activation_rate", 0.0),
            "center_trials_false_act": eval_5dir.get("center_trials_false_act", 0),
            "center_trials_stayed": eval_5dir.get("center_trials_stayed", 0),
            "center_trials_total": eval_5dir.get("center_trials_total", 0),
            "center_return_false_activation_display": eval_5dir.get("center_return_false_activation_display", "0/0 (0.0%)"),
            "overall_trial_correctness_rate": eval_5dir.get("overall_trial_correctness_rate", 0.0),
            "overall_trials_correct": eval_5dir.get("overall_trials_correct", 0),
            "overall_trials_total": eval_5dir.get("overall_trials_total", 0),
            "overall_trial_correctness_display": eval_5dir.get("overall_trial_correctness_display", "0/0 (0.0%)"),
            "is_preliminary": eval_5dir.get("is_preliminary", True),
            "sample_size_status": eval_5dir.get("sample_size_status", "PRELIMINARY"),
            "mean_target_error": eval_5dir.get("mean_target_error", 0.0),
            "peripheral_dwell_accuracy": round(dwell_accuracy, 4),
            "peripheral_dwell_passed": dwell_passed,
            "peripheral_dwell_total": dwell_total_peripheral,
            "peripheral_dwell_display": f"{dwell_passed}/{dwell_total_peripheral} ({dwell_accuracy*100:.1f}%)" if dwell_total_peripheral else "0/0 (0.0%)",
            "target_acquisition_latency": target_acq_stats,
            "transient_center_escapes": center_transient_summary,
            "trial_results": eval_5dir.get("trial_results", []),
            # Backward-compatible aliases
            "trial_selection_accuracy": eval_5dir.get("peripheral_trial_accuracy", 0.0),
            "selection_accuracy": eval_5dir.get("peripheral_trial_accuracy", 0.0),
            "trial_center_false_activation_rate": eval_5dir.get("center_return_false_activation_rate", 0.0),
            "dwell_selection_accuracy": round(dwell_accuracy, 4),
        },
        "evaluation_4dir_research_mode": {
            "status": eval_4dir.get("status", "VALID"),
            "peripheral_trial_accuracy": eval_4dir.get("peripheral_trial_accuracy", 0.0),
            "peripheral_trials_passed": eval_4dir.get("peripheral_trials_passed", 0),
            "peripheral_trials_total": eval_4dir.get("peripheral_trials_total", 0),
            "peripheral_trial_accuracy_display": eval_4dir.get("peripheral_trial_accuracy_display", "0/0 (0.0%)"),
            "center_return_false_activation_rate": eval_4dir.get("center_return_false_activation_rate", 0.0),
            "center_trials_false_act": eval_4dir.get("center_trials_false_act", 0),
            "center_trials_stayed": eval_4dir.get("center_trials_stayed", 0),
            "center_trials_total": eval_4dir.get("center_trials_total", 0),
            "overall_trial_correctness_rate": eval_4dir.get("overall_trial_correctness_rate", 0.0),
            "overall_trials_correct": eval_4dir.get("overall_trials_correct", 0),
            "overall_trials_total": eval_4dir.get("overall_trials_total", 0),
            "is_preliminary": eval_4dir.get("is_preliminary", True),
            "sample_size_status": eval_4dir.get("sample_size_status", "PRELIMINARY"),
            "mean_target_error": eval_4dir.get("mean_target_error", 0.0),
            "notice": "PROVISIONAL 4-DIRECTION RESEARCH MODE (OMITS UPWARD GAZE).",
            # Backward-compatible aliases
            "trial_selection_accuracy": eval_4dir.get("peripheral_trial_accuracy", 0.0),
            "trial_center_false_activation_rate": eval_4dir.get("center_return_false_activation_rate", 0.0),
        },
        "target_acquisition_latency": target_acq_stats,
        "transient_center_escapes": center_transient_summary,
        "end_to_end_latency_ms": {
            "p50": e2e_p50,
            "p95": e2e_p95,
            "sample_count": len(valid_latencies),
            "status": lat_status,
        },
    }


def draw_centered_text(
    img: np.ndarray,
    text: str,
    cx: int,
    cy: int,
    font_scale: float,
    color: Tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """Helper to draw text centered horizontally and vertically at (cx, cy)."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    tx = cx - tw // 2
    ty = cy + th // 2
    cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def draw_target_canvas(
    canvas_w: int,
    canvas_h: int,
    target_name: str,
    phase: str,  # "SETTLING" or "RECORDING"
    time_left_s: float,
    block_num: int,
    trial_num: int,
    total_trials: int,
    camera_preview: Optional[np.ndarray] = None,
    face_detected: bool = False,
) -> np.ndarray:
    """
    Renders visual fixation targets with full-screen black background and
    5-button assistive layout matching the evaluator's TARGET_BBOXES.
    """
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)  # Full black background

    # 1. Draw 5-button layout using identical bounding boxes as the evaluator
    for btn_name, (bx1, by1, bx2, by2) in TARGET_BBOXES.items():
        x1 = int(bx1 * canvas_w)
        y1 = int(by1 * canvas_h)
        x2 = int(bx2 * canvas_w)
        y2 = int(by2 * canvas_h)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        is_target = (btn_name == target_name)

        if is_target:
            if phase == "SETTLING":
                border_col = (0, 165, 255)  # Amber / Orange for settling
                fill_col = (0, 45, 75)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), fill_col, -1)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), border_col, 3)
                # Bullseye fixation center
                cv2.circle(canvas, (cx, cy), 20, border_col, 2, cv2.LINE_AA)
                cv2.circle(canvas, (cx, cy), 6, (0, 200, 255), -1, cv2.LINE_AA)
                draw_centered_text(canvas, f"{btn_name} [SETTLE]", cx, cy - 36, 0.7, border_col, 2)
            else:
                border_col = (0, 255, 0)  # Bright green for active fixation recording
                fill_col = (0, 60, 15)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), fill_col, -1)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), border_col, 4)
                # Bullseye fixation center
                cv2.circle(canvas, (cx, cy), 22, border_col, 3, cv2.LINE_AA)
                cv2.circle(canvas, (cx, cy), 6, (50, 255, 50), -1, cv2.LINE_AA)
                draw_centered_text(canvas, f"{btn_name} [FIXATE]", cx, cy - 36, 0.7, border_col, 2)
        else:
            border_col = (60, 60, 60)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), border_col, 1)
            draw_centered_text(canvas, btn_name, cx, cy, 0.65, (110, 110, 110), 1)

    # 2. Top status line
    if phase == "SETTLING":
        status_line = f"Block {block_num}/5, Trial {trial_num}/{total_trials}, SETTLE ({time_left_s:.1f}s)"
        status_col = (0, 180, 255)
    else:
        status_line = f"Block {block_num}/5, Trial {trial_num}/{total_trials}, fixate now ({time_left_s:.1f}s)"
        status_col = (0, 255, 0)

    draw_centered_text(canvas, status_line, canvas_w // 2, 40, 0.85, status_col, 2)

    # 3. Bottom instructions
    cv2.putText(
        canvas,
        "Press ESC or Q to abort (saves partial report)",
        (30, canvas_h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (120, 120, 120),
        1,
        cv2.LINE_AA,
    )

    # 4. Camera preview (bottom right corner, outside buttons)
    if camera_preview is not None:
        p_w = int(canvas_w * 0.15)
        p_h = int(canvas_h * 0.15)
        small_cam = cv2.resize(camera_preview, (p_w, p_h))
        mirrored = cv2.flip(small_cam, 1)
        bx1 = canvas_w - p_w - 20
        by1 = canvas_h - p_h - 20
        bx2 = bx1 + p_w
        by2 = by1 + p_h
        canvas[by1:by2, bx1:bx2] = mirrored

        stat_col = (0, 255, 0) if face_detected else (0, 0, 255)
        stat_txt = "FACE OK" if face_detected else "NO FACE"
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), stat_col, 2)
        cv2.putText(canvas, stat_txt, (bx1 + 8, by1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, stat_col, 1, cv2.LINE_AA)

    return canvas


def execute_comparison_session(
    mock_mode: bool = False,
    trial_duration_s: float = 3.0,
    settling_duration_s: float = 1.5,
    camera_idx: int = 0,
) -> Dict[str, Any]:
    """
    Executes the controlled sequential comparison trial.
    Runs headless/mock mode for automated testing or live OpenCV/camera mode.
    """
    session_id = f"cmp_v16_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1000000:06x}"
    output_dir = ROOT_DIR / "artifacts" / "observations"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{session_id}_comparison_report.json"

    print("=" * 80)
    print(" VOXGAZE CONTROLLED SEQUENTIAL MODEL COMPARISON (M1 / PIVOT)")
    print(f" Session ID: {session_id} | Mock Mode: {mock_mode}")
    print("=" * 80)

    # Initialize models
    try:
        m1_estimator = create_gaze_estimator("mobileone_s0")
        r34_estimator = create_gaze_estimator("resnet34")
    except Exception as exc:
        print(f"[ERROR] Failed to load models: {exc}")
        return {"error": str(exc)}

    records: List[FrameComparisonRecord] = []
    blocks = [
        (1, BLOCK_1_TARGETS),
        (2, BLOCK_2_TARGETS),
        (3, BLOCK_3_TARGETS),
        (4, BLOCK_4_TARGETS),
        (5, BLOCK_5_TARGETS),
    ]

    frame_counter = 0
    aborted = False

    if mock_mode:
        print("\n[INFO] Executing in mock/synthetic mode (15 frames per trial across 5 blocks)...")
        base_time = time.monotonic()
        for block_idx, targets in blocks:
            for cond_idx, target in enumerate(targets):
                trial_name = f"B{block_idx}_{cond_idx+1}_{target}"
                for _ in range(15):
                    frame_counter += 1
                    t_frame = base_time + (frame_counter * 0.0333)
                    if target == "CENTER":
                        base_yaw, base_pitch = -4.0, -10.0
                        rh, rv = 0.50, 0.50
                    elif target == "LEFT":
                        base_yaw, base_pitch = -18.0, -10.0
                        rh, rv = 0.41, 0.52
                    elif target == "RIGHT":
                        base_yaw, base_pitch = +12.0, -10.0
                        rh, rv = 0.59, 0.50
                    elif target == "UP":
                        base_yaw, base_pitch = -4.0, -6.0
                        rh, rv = 0.50, 0.48
                    elif target == "DOWN":
                        base_yaw, base_pitch = -4.0, -22.0
                        rh, rv = 0.50, 0.42

                    m1_yaw = base_yaw + np.random.normal(0, 0.5)
                    m1_pitch = base_pitch + np.random.normal(0, 0.5)
                    r34_yaw = base_yaw * 1.05 + np.random.normal(0, 0.4)
                    r34_pitch = base_pitch * 1.05 + np.random.normal(0, 0.4)

                    t_mp = 8.0
                    t_crop = 0.2
                    t_m1_pre = 1.0
                    t_m1_infer = 7.0
                    t_r34_pre = 1.0
                    t_r34_infer = 65.0

                    rec = FrameComparisonRecord(
                        session_id=session_id,
                        block_index=block_idx,
                        condition_index=cond_idx + 1,
                        trial_name=trial_name,
                        target_direction=target,
                        frame_index=frame_counter,
                        monotonic_time_s=t_frame,
                        face_detected=True,
                        face_crop_valid=True,
                        right_eye_h=rh + np.random.normal(0, 0.005),
                        right_eye_v=rv + np.random.normal(0, 0.005),
                        right_eye_ear=0.32,
                        left_eye_h=rh,
                        left_eye_v=rv,
                        left_eye_ear=0.32,
                        canthal_normal_px=-1.5,
                        m1_valid=True,
                        m1_yaw_deg=round(m1_yaw, 2),
                        m1_pitch_deg=round(m1_pitch, 2),
                        m1_pre_ms=t_m1_pre,
                        m1_infer_ms=t_m1_infer,
                        m1_end_to_end_ms=round(t_mp + t_crop + t_m1_pre + t_m1_infer + 0.01, 2),
                        r34_valid=True,
                        r34_yaw_deg=round(r34_yaw, 2),
                        r34_pitch_deg=round(r34_pitch, 2),
                        r34_pre_ms=t_r34_pre,
                        r34_infer_ms=t_r34_infer,
                        r34_end_to_end_ms=round(t_mp + t_crop + t_r34_pre + t_r34_infer + 0.01, 2),
                        mediapipe_ms=t_mp,
                        crop_ms=t_crop,
                        total_pipeline_ms=round(t_mp + t_crop + t_m1_infer + t_r34_infer, 2),
                    )
                    records.append(rec)

    else:
        # LIVE WEBCAM CAPTURE
        print("\n[INFO] Initializing webcam and MediaPipe FaceLandmarker...")
        cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_idx)
        if not cap.isOpened():
            print(f"[ERROR] Could not open camera device at index {camera_idx}.")
            return {"error": f"Camera index {camera_idx} could not be opened."}

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        landmarker_path = ROOT_DIR / "models" / "face_landmarker.task"
        if not landmarker_path.is_file():
            print(f"[ERROR] MediaPipe task asset not found at {landmarker_path}.")
            cap.release()
            return {"error": "Missing face_landmarker.task"}

        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        base_opts = mp_python.BaseOptions(model_asset_path=str(landmarker_path))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        detector = mp_vision.FaceLandmarker.create_from_options(options)

        win_name = "VoxGaze Controlled Model Comparison (Press ESC or Q to Abort)"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        aborted = False

        try:
            total_trials_per_block = len(BLOCK_1_TARGETS)
            for block_idx, targets in blocks:
                if aborted:
                    break

                print(f"\n>>> Starting Block {block_idx}/5 ({len(targets)} targets)...")
                emit_beep(freq=600, dur_ms=250)

                for cond_idx, target in enumerate(targets):
                    if aborted:
                        break

                    trial_name = f"B{block_idx}_{cond_idx+1}_{target}"
                    print(f"  Trial {cond_idx+1}/{total_trials_per_block}: {target} ({trial_name})")

                    # Phase 1: SETTLING
                    emit_beep(freq=880, dur_ms=100)
                    t_settle_start = time.monotonic()
                    while (time.monotonic() - t_settle_start) < settling_duration_s:
                        ret, raw_frame = cap.read()
                        if not ret or raw_frame is None:
                            continue

                        rem_s = max(0.0, settling_duration_s - (time.monotonic() - t_settle_start))
                        canvas = draw_target_canvas(
                            1280, 720, target, "SETTLING", rem_s,
                            block_idx, cond_idx + 1, total_trials_per_block,
                            camera_preview=raw_frame, face_detected=True,
                        )
                        cv2.imshow(win_name, canvas)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (27, ord("q"), ord("Q")):
                            aborted = True
                            break

                    if aborted:
                        break

                    # Phase 2: RECORDING
                    emit_beep(freq=1200, dur_ms=150)
                    t_rec_start = time.monotonic()
                    while (time.monotonic() - t_rec_start) < trial_duration_s:
                        frame_t0 = time.perf_counter()
                        ret, raw_frame = cap.read()
                        if not ret or raw_frame is None:
                            continue

                        frame_counter += 1
                        ts_ms = int(time.perf_counter() * 1000)

                        # 1. MediaPipe Inference
                        rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        t_mp_s = time.perf_counter()
                        detection_result = detector.detect_for_video(mp_img, ts_ms)
                        t_mp = (time.perf_counter() - t_mp_s) * 1000.0

                        face_detected = bool(detection_result.face_landmarks)
                        face_crop_valid = False
                        r_h, r_v, r_ear = None, None, None
                        l_h, l_v, l_ear = None, None, None
                        cn_dist = None

                        m1_valid = False
                        m1_yaw, m1_pitch = None, None
                        m1_pre_ms, m1_infer_ms = 0.0, 0.0

                        r34_valid = False
                        r34_yaw, r34_pitch = None, None
                        r34_pre_ms, r34_infer_ms = 0.0, 0.0

                        t_crop = 0.0

                        if face_detected:
                            face = detection_result.face_landmarks[0]
                            h_f, w_f = raw_frame.shape[:2]
                            pts = np.array([[p.x * w_f, p.y * h_f] for p in face])

                            # Geometry extraction
                            r_iris = Point2D(x=float(pts[RIGHT_IRIS, 0]), y=float(pts[RIGHT_IRIS, 1]))
                            r_inner = Point2D(x=float(pts[RIGHT_INNER_CANTHUS, 0]), y=float(pts[RIGHT_INNER_CANTHUS, 1]))
                            r_outer = Point2D(x=float(pts[RIGHT_OUTER_CANTHUS, 0]), y=float(pts[RIGHT_OUTER_CANTHUS, 1]))
                            r_up = Point2D(x=float(pts[RIGHT_UPPER_LID, 0]), y=float(pts[RIGHT_UPPER_LID, 1]))
                            r_low = Point2D(x=float(pts[RIGHT_LOWER_LID, 0]), y=float(pts[RIGHT_LOWER_LID, 1]))

                            l_iris = Point2D(x=float(pts[LEFT_IRIS, 0]), y=float(pts[LEFT_IRIS, 1]))
                            l_inner = Point2D(x=float(pts[LEFT_INNER_CANTHUS, 0]), y=float(pts[LEFT_INNER_CANTHUS, 1]))
                            l_outer = Point2D(x=float(pts[LEFT_OUTER_CANTHUS, 0]), y=float(pts[LEFT_OUTER_CANTHUS, 1]))
                            l_up = Point2D(x=float(pts[LEFT_UPPER_LID, 0]), y=float(pts[LEFT_UPPER_LID, 1]))
                            l_low = Point2D(x=float(pts[LEFT_LOWER_LID, 0]), y=float(pts[LEFT_LOWER_LID, 1]))

                            r_f = extract_eye_features(r_iris, r_inner, r_outer, r_up, r_low, is_right_eye=True)
                            l_f = extract_eye_features(l_iris, l_inner, l_outer, l_up, l_low, is_right_eye=False)
                            r_h, r_v, r_ear = r_f.h_ratio, r_f.v_ratio, r_f.ear
                            l_h, l_v, l_ear = l_f.h_ratio, l_f.v_ratio, l_f.ear
                            cn_dist = compute_canthal_normal_distance_px(r_iris, r_inner, r_outer, is_right_eye=True)

                            # 2. Face crop extraction
                            t_crop_s = time.perf_counter()
                            crop_res = extract_face_crop(raw_frame, face, padding_scale=1.20)
                            t_crop = (time.perf_counter() - t_crop_s) * 1000.0

                            if crop_res.is_valid and crop_res.crop_bgr is not None:
                                face_crop_valid = True
                                crop_bgr = crop_res.crop_bgr

                                # 3. Sequential Inference: Model 1 (MobileOne S0)
                                m1_out = m1_estimator.predict(crop_bgr)
                                if m1_out.is_valid:
                                    m1_valid = True
                                    m1_yaw = m1_out.yaw_deg
                                    m1_pitch = m1_out.pitch_deg
                                    m1_pre_ms = m1_out.preprocessing_latency_ms
                                    m1_infer_ms = m1_out.inference_latency_ms

                                # 4. Sequential Inference: Model 2 (ResNet-34) on identical crop
                                r34_out = r34_estimator.predict(crop_bgr)
                                if r34_out.is_valid:
                                    r34_valid = True
                                    r34_yaw = r34_out.yaw_deg
                                    r34_pitch = r34_out.pitch_deg
                                    r34_pre_ms = r34_out.preprocessing_latency_ms
                                    r34_infer_ms = r34_out.inference_latency_ms

                        t_pipe = (time.perf_counter() - frame_t0) * 1000.0
                        m1_e2e = t_mp + t_crop + m1_pre_ms + m1_infer_ms
                        r34_e2e = t_mp + t_crop + r34_pre_ms + r34_infer_ms

                        rec = FrameComparisonRecord(
                            session_id=session_id,
                            block_index=block_idx,
                            condition_index=cond_idx + 1,
                            trial_name=trial_name,
                            target_direction=target,
                            frame_index=frame_counter,
                            monotonic_time_s=time.monotonic(),
                            face_detected=face_detected,
                            face_crop_valid=face_crop_valid,
                            right_eye_h=round(r_h, 4) if r_h is not None else None,
                            right_eye_v=round(r_v, 4) if r_v is not None else None,
                            right_eye_ear=round(r_ear, 4) if r_ear is not None else None,
                            left_eye_h=round(l_h, 4) if l_h is not None else None,
                            left_eye_v=round(l_v, 4) if l_v is not None else None,
                            left_eye_ear=round(l_ear, 4) if l_ear is not None else None,
                            canthal_normal_px=round(cn_dist, 2) if cn_dist is not None else None,
                            m1_valid=m1_valid,
                            m1_yaw_deg=round(m1_yaw, 2) if m1_yaw is not None else None,
                            m1_pitch_deg=round(m1_pitch, 2) if m1_pitch is not None else None,
                            m1_pre_ms=round(m1_pre_ms, 2),
                            m1_infer_ms=round(m1_infer_ms, 2),
                            m1_end_to_end_ms=round(m1_e2e, 2),
                            r34_valid=r34_valid,
                            r34_yaw_deg=round(r34_yaw, 2) if r34_yaw is not None else None,
                            r34_pitch_deg=round(r34_pitch, 2) if r34_pitch is not None else None,
                            r34_pre_ms=round(r34_pre_ms, 2),
                            r34_infer_ms=round(r34_infer_ms, 2),
                            r34_end_to_end_ms=round(r34_e2e, 2),
                            mediapipe_ms=round(t_mp, 2),
                            crop_ms=round(t_crop, 2),
                            total_pipeline_ms=round(t_pipe, 2),
                        )
                        records.append(rec)

                        # Visual UI update
                        rem_s = max(0.0, trial_duration_s - (time.monotonic() - t_rec_start))
                        canvas = draw_target_canvas(
                            1280, 720, target, "RECORDING", rem_s,
                            block_idx, cond_idx + 1, total_trials_per_block,
                            camera_preview=raw_frame, face_detected=face_detected,
                        )
                        cv2.imshow(win_name, canvas)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (27, ord("q"), ord("Q")):
                            aborted = True
                            break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            if aborted:
                print("\n[INFO] Session aborted by user via ESC/Q.")

    # Guard against early abort: Esc aborts and still writes a partial report marked INCOMPLETE
    if aborted:
        print(f"\n[INFO] Writing partial report marked INCOMPLETE ({len(records)} frames collected)...")
        partial_summary = {
            "session_id": session_id,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "mock_mode": mock_mode,
            "status": "INCOMPLETE",
            "is_complete": False,
            "aborted": True,
            "validation_notice": "INCOMPLETE SESSION (ABORTED BY USER)",
            "total_frames_recorded": len(records),
            "reasons": ["Session aborted by user via ESC or Q key press."],
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(partial_summary, f, indent=2)
        print(f"[SUCCESS] Partial comparison report saved to: {report_path}")
        return partial_summary

    if len(records) < 15:
        print(f"\n[WARNING] Insufficient frames collected ({len(records)}). Session incomplete.")
        incomplete_summary = {
            "session_id": session_id,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "mock_mode": mock_mode,
            "status": "INCOMPLETE",
            "is_complete": False,
            "aborted": False,
            "validation_notice": "INCOMPLETE SESSION (INSUFFICIENT FRAMES)",
            "total_frames_recorded": len(records),
            "reasons": [f"Insufficient frames collected ({len(records)} < 15)."],
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(incomplete_summary, f, indent=2)
        print(f"[SUCCESS] Incomplete report saved to: {report_path}")
        return incomplete_summary

    # Perform Blocks 3, 4, 5 Held-Out Evaluation on both models (12 peripheral, 15 center trials)
    print("\n[STEP 1] Evaluating MobileOne S0 (Hybrid & Angles on Blocks 3-5)...")
    m1_hybrid = analyze_block_evaluation(records, model_key="m1", feature_set=FeatureSetType.HYBRID)
    m1_angles = analyze_block_evaluation(records, model_key="m1", feature_set=FeatureSetType.ANGLES_ONLY)

    print("\n[STEP 2] Evaluating ResNet-34 (Hybrid & Angles on Blocks 3-5)...")
    r34_hybrid = analyze_block_evaluation(records, model_key="r34", feature_set=FeatureSetType.HYBRID)
    r34_angles = analyze_block_evaluation(records, model_key="r34", feature_set=FeatureSetType.ANGLES_ONLY)

    # Rigorous non-negotiable adoption evaluations
    m1_eval = evaluate_model_adoption(
        m1_hybrid.get("evaluation_5dir"),
        m1_hybrid.get("end_to_end_latency_ms"),
        sessions_passed=1,  # Single execution is 1 session
        is_mock=mock_mode,
        model_name="MobileOne S0",
    )
    r34_eval = evaluate_model_adoption(
        r34_hybrid.get("evaluation_5dir"),
        r34_hybrid.get("end_to_end_latency_ms"),
        sessions_passed=1,
        is_mock=mock_mode,
        model_name="ResNet-34",
    )

    m1_acc = m1_eval["metrics"]["peripheral_trial_accuracy"]
    r34_acc = r34_eval["metrics"]["peripheral_trial_accuracy"]
    acc_diff_pp = round((r34_acc - m1_acc) * 100.0, 1)

    meets_relative_delta = acc_diff_pp >= 15.0
    # Adoption requires meeting ALL absolute gates AND relative advantage
    resnet34_adopted = bool(r34_eval["adopted"] and meets_relative_delta)

    if mock_mode:
        overall_status = "MOCK_NOT_PHYSICAL_VALIDATION"
    elif r34_eval["is_preliminary"]:
        overall_status = "PRELIMINARY_SAMPLE_SIZE"
    elif resnet34_adopted:
        overall_status = "ADOPTED_RESNET34"
    else:
        overall_status = "RETAIN_MOBILEONE_BASELINE"

    validation_label = "MOCK, not physical validation" if mock_mode else "PHYSICAL ACQUISITION"
    m1_e2e_p50 = m1_eval["metrics"]["latency_p50_ms"]
    r34_e2e_p50 = r34_eval["metrics"]["latency_p50_ms"]

    print("\n" + "=" * 80)
    print(" COMPARISON SUMMARY & RESNET-34 DECISION RULE (5-BLOCK PROTOCOL)")
    print("=" * 80)
    print(f" [VALIDATION STATUS]: {validation_label}")
    print(" MobileOne S0:")
    print(f"   - Peripheral Accuracy: {m1_eval['metrics']['peripheral_trial_accuracy_display']}")
    print(f"   - Center False Activation (Median): {m1_eval['metrics']['center_return_false_activation_display']}")
    print(f"   - Center Transient Escapes (Any Frame): {m1_hybrid.get('transient_center_escapes', {}).get('trials_with_transient_escapes', 0)}/{m1_hybrid.get('transient_center_escapes', {}).get('total_center_trials', 0)} trials")
    print(f"   - Continuous Dwell (>= 400 ms): {m1_hybrid.get('evaluation_5dir', {}).get('peripheral_dwell_display', 'N/A')}")
    print(f"   - Target Acquisition Latency (p50): {m1_hybrid.get('target_acquisition_latency', {}).get('p50_ms')} ms")
    print(f"   - Pipeline Latency (p50): {m1_e2e_p50} ms")
    print(f"   - Adoption Gate: {m1_eval['status']}")
    print(" ResNet-34:")
    print(f"   - Peripheral Accuracy: {r34_eval['metrics']['peripheral_trial_accuracy_display']}")
    print(f"   - Center False Activation (Median): {r34_eval['metrics']['center_return_false_activation_display']}")
    print(f"   - Center Transient Escapes (Any Frame): {r34_hybrid.get('transient_center_escapes', {}).get('trials_with_transient_escapes', 0)}/{r34_hybrid.get('transient_center_escapes', {}).get('total_center_trials', 0)} trials")
    print(f"   - Continuous Dwell (>= 400 ms): {r34_hybrid.get('evaluation_5dir', {}).get('peripheral_dwell_display', 'N/A')}")
    print(f"   - Target Acquisition Latency (p50): {r34_hybrid.get('target_acquisition_latency', {}).get('p50_ms')} ms")
    print(f"   - Pipeline Latency (p50): {r34_e2e_p50} ms")
    print(f"   - Adoption Gate: {r34_eval['status']}")
    print(f" ResNet-34 vs MobileOne Delta: {acc_diff_pp:+5.1f} percentage points (Rule requires: >= +15.0 pp)")
    print(f" Decision Rule Result: {'PASS (Adopt ResNet-34)' if resnet34_adopted else 'REJECT / PRELIMINARY (Retain MobileOne S0 Baseline)'}")
    print(f" Overall Decision Status: {overall_status}")

    summary = {
        "session_id": session_id,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mock_mode": mock_mode,
        "validation_notice": "MOCK, not physical validation" if mock_mode else "PHYSICAL ACQUISITION",
        "is_physical_validation": not mock_mode,
        "total_frames_recorded": len(records),
        "protocol": {
            "blocks_count": len(blocks),
            "train_blocks": [1, 2],
            "held_out_test_blocks": [3, 4, 5],
            "targets_per_block": len(BLOCK_1_TARGETS),
            "total_trials": len(blocks) * len(BLOCK_1_TARGETS),
            "held_out_trials_count": 3 * len(BLOCK_1_TARGETS),
            "held_out_peripheral_trials": 12,
            "held_out_center_trials": 15,
            "balanced_orders": {
                "block_1": BLOCK_1_TARGETS,
                "block_2": BLOCK_2_TARGETS,
                "block_3": BLOCK_3_TARGETS,
                "block_4": BLOCK_4_TARGETS,
                "block_5": BLOCK_5_TARGETS,
            },
        },
        "decision_rule": {
            "rule_statement": "Adoption requires: peripheral accuracy >= 85% on >= 12 held-out trials, center false activations <= 5% on >= 15 held-out trials, and both passing across >= 2 independent sessions.",
            "relative_advantage_requirement": "ResNet-34 requires exceeding MobileOne S0 by >= 15.0 percentage points in addition to meeting all absolute safety gates.",
            "validation_label": validation_label,
            "is_physical_validation": not mock_mode,
            "mobileone_accuracy_pct": round(m1_acc * 100.0, 2),
            "resnet34_accuracy_pct": round(r34_acc * 100.0, 2),
            "delta_percentage_points": acc_diff_pp,
            "meets_relative_delta": meets_relative_delta,
            "adopted": resnet34_adopted,
            "status": overall_status,
            "model_evaluations": {
                "mobileone_s0": m1_eval,
                "resnet34": r34_eval,
            },
        },
        "models": {
            "mobileone_s0_hybrid": m1_hybrid,
            "mobileone_s0_angles": m1_angles,
            "resnet34_hybrid": r34_hybrid,
            "resnet34_angles": r34_angles,
        },
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[SUCCESS] Comparison report saved to: {report_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run controlled sequential model comparison.")
    parser.add_argument("--mock", action="store_true", help="Run in mock/synthetic mode for verification.")
    parser.add_argument("--trial-duration", type=float, default=3.0, help="Trial fixation duration in seconds.")
    parser.add_argument("--settling-duration", type=float, default=1.5, help="Visual settling duration in seconds.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index.")
    args = parser.parse_args()

    execute_comparison_session(
        mock_mode=args.mock,
        trial_duration_s=args.trial_duration,
        settling_duration_s=args.settling_duration,
        camera_idx=args.camera,
    )
