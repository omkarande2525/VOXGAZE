"""
VoxGaze Perception: Structured Gaze-Observation Measurement Module

Provides deterministic schemas, condition sequence definitions, streamed CSV observation
logging, and quantitative summary metrics for empirical gaze diagnostics.

RESEARCH & SAFETY INVARIANTS:
1. Diagnostic Observation Only: Zero connection to AAC command FSM, preview, or Piper TTS.
2. User-Directed Conditions: Condition labels reflect requested tasks, not independently verified ground truth.
3. Honest Missing Data: Missing observations are represented as null/empty, never filled with zeros.
4. Privacy First: Logs bounded numerical features only. Zero raw images, face crops, or videos saved.
5. Non-Destructive Storage: Always generates unique output files; never silently overwrites existing data.
"""

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import numpy as np


@dataclass(frozen=True)
class ObservationCondition:
    """Definition of a single guided observation condition."""

    index: int
    name: str
    instruction: str
    is_center: bool = False


# Canonical 20-condition guided sequence
OBSERVATION_CONDITIONS: List[ObservationCondition] = [
    ObservationCondition(1, "CENTER_BASELINE", "Look naturally at the center of the screen.", is_center=True),
    ObservationCondition(2, "EYES_LEFT", "Shift gaze to the LEFT without turning your head.", is_center=False),
    ObservationCondition(3, "EYES_CENTER_RETURN", "Return gaze to the CENTER.", is_center=True),
    ObservationCondition(4, "EYES_RIGHT", "Shift gaze to the RIGHT without turning your head.", is_center=False),
    ObservationCondition(5, "EYES_CENTER_RETURN", "Return gaze to the CENTER.", is_center=True),
    ObservationCondition(6, "EYES_UP", "Shift gaze UP without tilting your head.", is_center=False),
    ObservationCondition(7, "EYES_CENTER_RETURN", "Return gaze to the CENTER.", is_center=True),
    ObservationCondition(8, "EYES_DOWN", "Shift gaze DOWN without tilting your head.", is_center=False),
    ObservationCondition(9, "EYES_CENTER_RETURN", "Return gaze to the CENTER.", is_center=True),
    ObservationCondition(10, "HEAD_LEFT_EYES_FORWARD", "Turn head slightly LEFT while keeping eyes forward on screen.", is_center=False),
    ObservationCondition(11, "HEAD_CENTER_RETURN", "Return head to CENTER.", is_center=True),
    ObservationCondition(12, "HEAD_RIGHT_EYES_FORWARD", "Turn head slightly RIGHT while keeping eyes forward on screen.", is_center=False),
    ObservationCondition(13, "HEAD_CENTER_RETURN", "Return head to CENTER.", is_center=True),
    ObservationCondition(14, "HEAD_UP_EYES_FORWARD", "Tilt head slightly UP while keeping eyes forward on screen.", is_center=False),
    ObservationCondition(15, "HEAD_CENTER_RETURN", "Return head to CENTER.", is_center=True),
    ObservationCondition(16, "HEAD_DOWN_EYES_FORWARD", "Tilt head slightly DOWN while keeping eyes forward on screen.", is_center=False),
    ObservationCondition(17, "HEAD_CENTER_RETURN", "Return head to CENTER.", is_center=True),
    ObservationCondition(18, "COMBINED_MOVEMENT", "Perform natural combined head and eye movement.", is_center=False),
    ObservationCondition(19, "TRACKING_RECOVERY", "Cover camera or look away briefly, then return to center.", is_center=False),
    ObservationCondition(20, "FINAL_CENTER_BASELINE", "Final neutral center gaze baseline.", is_center=True),
]


CSV_HEADER: List[str] = [
    "session_id",
    "monotonic_time_s",
    "elapsed_session_s",
    "condition_index",
    "condition_name",
    "frame_index",
    "face_detected",
    "face_crop_valid",
    "geometry_valid",
    "right_eye_h_ratio",
    "right_eye_v_ratio",
    "right_eye_ear",
    "left_eye_h_ratio",
    "left_eye_v_ratio",
    "left_eye_ear",
    "canthal_normal_dist_px",
    "mobileone_valid",
    "mobileone_yaw_deg",
    "mobileone_pitch_deg",
    "mediapipe_latency_ms",
    "crop_latency_ms",
    "mobileone_preprocess_ms",
    "mobileone_infer_ms",
    "total_frame_latency_ms",
    "unavailable_reason",
]


@dataclass
class FrameObservationRecord:
    """Single frame numerical measurement record."""

    session_id: str
    monotonic_time_s: float
    elapsed_session_s: float
    condition_index: int
    condition_name: str
    frame_index: int
    face_detected: bool
    face_crop_valid: bool
    geometry_valid: bool
    right_eye_h_ratio: Optional[float] = None
    right_eye_v_ratio: Optional[float] = None
    right_eye_ear: Optional[float] = None
    left_eye_h_ratio: Optional[float] = None
    left_eye_v_ratio: Optional[float] = None
    left_eye_ear: Optional[float] = None
    canthal_normal_dist_px: Optional[float] = None
    mobileone_valid: bool = False
    mobileone_yaw_deg: Optional[float] = None
    mobileone_pitch_deg: Optional[float] = None
    mediapipe_latency_ms: float = 0.0
    crop_latency_ms: float = 0.0
    mobileone_preprocess_ms: Optional[float] = None
    mobileone_infer_ms: Optional[float] = None
    total_frame_latency_ms: float = 0.0
    unavailable_reason: Optional[str] = None

    def to_csv_row(self) -> List[str]:
        """Formats record into CSV string cells, using empty string for missing values."""
        def _fmt_float(val: Optional[float], decimals: int = 4) -> str:
            if val is None or not math.isfinite(val):
                return ""
            return f"{val:.{decimals}f}"

        def _fmt_bool(val: bool) -> str:
            return "true" if val else "false"

        return [
            self.session_id,
            _fmt_float(self.monotonic_time_s, 6),
            _fmt_float(self.elapsed_session_s, 3),
            str(self.condition_index),
            self.condition_name,
            str(self.frame_index),
            _fmt_bool(self.face_detected),
            _fmt_bool(self.face_crop_valid),
            _fmt_bool(self.geometry_valid),
            _fmt_float(self.right_eye_h_ratio, 4),
            _fmt_float(self.right_eye_v_ratio, 4),
            _fmt_float(self.right_eye_ear, 4),
            _fmt_float(self.left_eye_h_ratio, 4),
            _fmt_float(self.left_eye_v_ratio, 4),
            _fmt_float(self.left_eye_ear, 4),
            _fmt_float(self.canthal_normal_dist_px, 2),
            _fmt_bool(self.mobileone_valid),
            _fmt_float(self.mobileone_yaw_deg, 2),
            _fmt_float(self.mobileone_pitch_deg, 2),
            _fmt_float(self.mediapipe_latency_ms, 2),
            _fmt_float(self.crop_latency_ms, 2),
            _fmt_float(self.mobileone_preprocess_ms, 2),
            _fmt_float(self.mobileone_infer_ms, 2),
            _fmt_float(self.total_frame_latency_ms, 2),
            self.unavailable_reason or "",
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Returns JSON-serializable dictionary with None for missing values."""
        return asdict(self)


def get_unique_output_path(
    target_dir: Union[str, Path],
    prefix: str = "obs",
    ext: str = "csv",
) -> Path:
    """
    Constructs a collision-safe, non-destructive output filepath.

    Args:
        target_dir: Target directory.
        prefix: Filename prefix (default: 'obs').
        ext: File extension without leading dot (default: 'csv').

    Returns:
        Guaranteed unique Path that does not already exist.
    """
    out_dir = Path(target_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    token = uuid.uuid4().hex[:6]
    candidate = out_dir / f"{prefix}_{timestamp}_{token}.{ext}"

    counter = 1
    while candidate.exists():
        candidate = out_dir / f"{prefix}_{timestamp}_{token}_{counter}.{ext}"
        counter += 1

    return candidate


def calculate_circular_stats(angles_deg: List[float]) -> Dict[str, Optional[float]]:
    """
    Computes circular mean, circular variance, and circular standard deviation for angles in degrees.

    For angles centered near 0°, circular and linear standard deviations are essentially identical.
    Circular formulation avoids wrap-around distortion near boundaries.

    Args:
        angles_deg: List of angles in degrees.

    Returns:
        Dictionary containing circular statistics.
    """
    valid_angles = [a for a in angles_deg if a is not None and math.isfinite(a)]
    if not valid_angles:
        return {
            "sample_count": 0,
            "circular_mean_deg": None,
            "circular_variance": None,
            "circular_std_deg": None,
        }

    rads = np.deg2rad(valid_angles)
    c_sum = float(np.sum(np.cos(rads)))
    s_sum = float(np.sum(np.sin(rads)))
    n = len(valid_angles)

    c_mean = c_sum / n
    s_mean = s_sum / n
    r = math.hypot(c_mean, s_mean)

    mean_rad = math.atan2(s_mean, c_mean)
    mean_deg = float(np.rad2deg(mean_rad))

    # Circular variance in [0, 1]
    circ_var = 1.0 - r

    # Circular standard deviation (Fisher definition)
    # std = sqrt(-2 * ln(r)) for r in (0, 1]
    if r > 1e-7:
        circ_std_rad = math.sqrt(max(0.0, -2.0 * math.log(min(1.0, r))))
        circ_std_deg = float(np.rad2deg(circ_std_rad))
    else:
        circ_std_deg = 180.0  # Max angular dispersion

    return {
        "sample_count": n,
        "circular_mean_deg": round(mean_deg, 3),
        "circular_variance": round(circ_var, 5),
        "circular_std_deg": round(circ_std_deg, 3),
    }


def compute_distribution_stats(values: List[float]) -> Dict[str, Optional[float]]:
    """Computes median, IQR, min, max, and standard deviation for real-valued samples."""
    valid_vals = [v for v in values if v is not None and math.isfinite(v)]
    if not valid_vals:
        return {
            "sample_count": 0,
            "median": None,
            "iqr": None,
            "min": None,
            "max": None,
            "std": None,
        }

    arr = np.array(valid_vals, dtype=np.float64)
    p25, p50, p75 = np.percentile(arr, [25, 50, 75])
    return {
        "sample_count": len(valid_vals),
        "median": round(float(p50), 4),
        "iqr": round(float(p75 - p25), 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
        "std": round(float(np.std(arr, ddof=1 if len(valid_vals) > 1 else 0)), 4),
    }


def compute_observation_summary(
    records: List[FrameObservationRecord],
    session_id: str,
    session_status: str,
    condition_states: Dict[int, str],
    start_time_iso: str,
    end_time_iso: str,
    elapsed_duration_s: float,
) -> Dict[str, Any]:
    """
    Computes aggregated quantitative summary metrics over an observation session.

    Distinguishes:
    - Frame-level signal variability.
    - Sensor / model availability rates.
    - Inference latency distribution (p50 and p95).
    - Note: Does NOT compute ground-truth accuracy.
    """
    total_frames = len(records)

    # Group records by condition index
    condition_records: Dict[int, List[FrameObservationRecord]] = {
        cond.index: [] for cond in OBSERVATION_CONDITIONS
    }
    for rec in records:
        if rec.condition_index in condition_records:
            condition_records[rec.condition_index].append(rec)

    # Condition summaries
    condition_summaries: List[Dict[str, Any]] = []
    completed_conditions: List[str] = []
    skipped_conditions: List[str] = []
    interrupted_conditions: List[str] = []

    for cond in OBSERVATION_CONDITIONS:
        c_recs = condition_records[cond.index]
        c_total = len(c_recs)
        status = condition_states.get(cond.index, "NOT_STARTED")

        if status == "COMPLETED":
            completed_conditions.append(cond.name)
        elif status == "SKIPPED":
            skipped_conditions.append(cond.name)
        elif status in ("RECORDING", "INTERRUPTED", "ABORTED"):
            interrupted_conditions.append(cond.name)

        face_detected_count = sum(1 for r in c_recs if r.face_detected)
        crop_valid_count = sum(1 for r in c_recs if r.face_crop_valid)
        geom_valid_count = sum(1 for r in c_recs if r.geometry_valid)
        model_valid_count = sum(1 for r in c_recs if r.mobileone_valid)

        face_rate = face_detected_count / c_total if c_total > 0 else 0.0
        geom_rate = geom_valid_count / c_total if c_total > 0 else 0.0
        model_rate = model_valid_count / c_total if c_total > 0 else 0.0

        condition_summaries.append({
            "condition_index": cond.index,
            "condition_name": cond.name,
            "instruction": cond.instruction,
            "status": status,
            "frames_collected": c_total,
            "face_detection_availability_rate": round(face_rate, 4),
            "valid_geometry_estimate_rate": round(geom_rate, 4),
            "valid_mobileone_estimate_rate": round(model_rate, 4),
            "counts": {
                "face_detected": face_detected_count,
                "face_lost": c_total - face_detected_count,
                "crop_valid": crop_valid_count,
                "crop_invalid": c_total - crop_valid_count,
                "geometry_valid": geom_valid_count,
                "mobileone_valid": model_valid_count,
                "mobileone_failed": c_total - model_valid_count,
            },
        })

    # Overall error/tracking loss counts
    tracking_lost_count = sum(1 for r in records if not r.face_detected)
    invalid_crop_count = sum(1 for r in records if r.face_detected and not r.face_crop_valid)
    inference_error_count = sum(1 for r in records if r.face_crop_valid and not r.mobileone_valid)

    # Latency percentiles
    model_latencies = [
        r.mobileone_infer_ms for r in records
        if r.mobileone_valid and r.mobileone_infer_ms is not None and math.isfinite(r.mobileone_infer_ms)
    ]
    total_latencies = [
        r.total_frame_latency_ms for r in records
        if math.isfinite(r.total_frame_latency_ms) and r.total_frame_latency_ms > 0
    ]

    def _calc_p50_p95(vals: List[float]) -> Dict[str, Any]:
        if not vals:
            return {"sample_count": 0, "median_p50_ms": None, "p95_ms": None}
        p50, p95 = np.percentile(vals, [50, 95])
        return {
            "sample_count": len(vals),
            "median_p50_ms": round(float(p50), 2),
            "p95_ms": round(float(p95), 2),
        }

    # Center observations variability
    center_indices = {cond.index for cond in OBSERVATION_CONDITIONS if cond.is_center}
    center_records = [r for r in records if r.condition_index in center_indices]

    center_yaw = [r.mobileone_yaw_deg for r in center_records if r.mobileone_valid and r.mobileone_yaw_deg is not None]
    center_pitch = [r.mobileone_pitch_deg for r in center_records if r.mobileone_valid and r.mobileone_pitch_deg is not None]
    center_r_h = [r.right_eye_h_ratio for r in center_records if r.geometry_valid and r.right_eye_h_ratio is not None]
    center_r_v = [r.right_eye_v_ratio for r in center_records if r.geometry_valid and r.right_eye_v_ratio is not None]

    center_variability = {
        "conditions_included": [cond.name for cond in OBSERVATION_CONDITIONS if cond.is_center],
        "total_center_frames": len(center_records),
        "disclaimer": "These statistics measure empirical frame-level dispersion during repeated center fixations, NOT ground-truth accuracy.",
        "mobileone_yaw_deg": {
            "linear_stats": compute_distribution_stats(center_yaw),
            "circular_stats": calculate_circular_stats(center_yaw),
        },
        "mobileone_pitch_deg": {
            "linear_stats": compute_distribution_stats(center_pitch),
            "circular_stats": calculate_circular_stats(center_pitch),
        },
        "geometry_right_eye_h_ratio": compute_distribution_stats(center_r_h),
        "geometry_right_eye_v_ratio": compute_distribution_stats(center_r_v),
    }

    return {
        "metadata": {
            "session_id": session_id,
            "session_status": session_status,
            "start_time_iso": start_time_iso,
            "end_time_iso": end_time_iso,
            "elapsed_duration_s": round(elapsed_duration_s, 2),
            "total_frames_recorded": total_frames,
            "disclaimer": (
                "Observation condition names reflect experimental instructions provided to the user, "
                "not independently verified ground truth. Frame-level variability measures signal stability "
                "rather than accuracy."
            ),
        },
        "conditions_overview": {
            "total_conditions": len(OBSERVATION_CONDITIONS),
            "completed_count": len(completed_conditions),
            "skipped_count": len(skipped_conditions),
            "interrupted_count": len(interrupted_conditions),
            "completed": completed_conditions,
            "skipped": skipped_conditions,
            "interrupted": interrupted_conditions,
        },
        "overall_counts": {
            "total_frames": total_frames,
            "tracking_loss_frames": tracking_lost_count,
            "invalid_crop_frames": invalid_crop_count,
            "mobileone_inference_error_frames": inference_error_count,
        },
        "latency_metrics": {
            "mobileone_inference_ms": _calc_p50_p95(model_latencies),
            "total_frame_latency_ms": _calc_p50_p95(total_latencies),
        },
        "center_observations_variability": center_variability,
        "condition_summaries": condition_summaries,
    }


class ObservationSession:
    """
    Manages interactive state, CSV streaming, and summary generation for a structured observation session.
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "artifacts/observations",
        output_file: Optional[Union[str, Path]] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if session_id is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            token = uuid.uuid4().hex[:6]
            self.session_id = f"obs_{ts}_{token}"
        else:
            self.session_id = session_id

        # Determine CSV filepath
        if output_file is not None:
            target_csv = Path(output_file).resolve()
            if target_csv.exists():
                # Avoid silent overwrite: generate non-destructive unique name
                target_csv = get_unique_output_path(
                    target_csv.parent,
                    prefix=target_csv.stem,
                    ext=target_csv.suffix.lstrip(".") or "csv",
                )
            self.csv_path = target_csv
        else:
            self.csv_path = self.output_dir / f"{self.session_id}.csv"

        self.json_summary_path = self.csv_path.with_name(f"{self.csv_path.stem}_summary.json")

        self.start_time_iso = datetime.now().isoformat()
        self.start_mono_time = time.monotonic()
        self.end_time_iso: Optional[str] = None

        self.status = "INITIALIZED"  # INITIALIZED, ACTIVE, COMPLETED, ABORTED
        self.current_condition_idx: int = 1
        self.is_recording_active: bool = False
        self.condition_states: Dict[int, str] = {c.index: "NOT_STARTED" for c in OBSERVATION_CONDITIONS}

        self.records: List[FrameObservationRecord] = []
        self._csv_file = None
        self._csv_writer = None

        # Initialize CSV file with header
        self._init_csv()

    def _init_csv(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_file = open(self.csv_path, mode="w", newline="", encoding="utf-8")
        import csv
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(CSV_HEADER)
        self._csv_file.flush()

    @property
    def current_condition(self) -> ObservationCondition:
        idx = min(self.current_condition_idx, len(OBSERVATION_CONDITIONS))
        return OBSERVATION_CONDITIONS[idx - 1]

    @property
    def is_finished(self) -> bool:
        return self.status in ("COMPLETED", "ABORTED")

    def start_recording_current_condition(self) -> None:
        """Starts collecting data for the current condition."""
        if self.is_finished:
            return
        self.status = "ACTIVE"
        self.is_recording_active = True
        self.condition_states[self.current_condition_idx] = "RECORDING"

    def finish_current_condition(self) -> bool:
        """
        Finishes collecting data for current condition and advances to the next condition.
        Returns True if more conditions remain, False if all conditions completed.
        """
        if self.is_finished:
            return False

        self.is_recording_active = False
        self.condition_states[self.current_condition_idx] = "COMPLETED"

        if self.current_condition_idx >= len(OBSERVATION_CONDITIONS):
            self.complete_session()
            return False

        self.current_condition_idx += 1
        self.condition_states[self.current_condition_idx] = "NOT_STARTED"
        return True

    def skip_current_condition(self) -> bool:
        """
        Skips the current condition without collecting (or discards active collection)
        and advances to the next condition.
        """
        if self.is_finished:
            return False

        self.is_recording_active = False
        self.condition_states[self.current_condition_idx] = "SKIPPED"

        if self.current_condition_idx >= len(OBSERVATION_CONDITIONS):
            self.complete_session()
            return False

        self.current_condition_idx += 1
        self.condition_states[self.current_condition_idx] = "NOT_STARTED"
        return True

    def record_frame(
        self,
        frame_idx: int,
        face_detected: bool,
        face_crop_valid: bool,
        geometry_valid: bool,
        right_eye_h: Optional[float] = None,
        right_eye_v: Optional[float] = None,
        right_eye_ear: Optional[float] = None,
        left_eye_h: Optional[float] = None,
        left_eye_v: Optional[float] = None,
        left_eye_ear: Optional[float] = None,
        canthal_normal_dist_px: Optional[float] = None,
        mobileone_valid: bool = False,
        mobileone_yaw_deg: Optional[float] = None,
        mobileone_pitch_deg: Optional[float] = None,
        mediapipe_latency_ms: float = 0.0,
        crop_latency_ms: float = 0.0,
        mobileone_preprocess_ms: Optional[float] = None,
        mobileone_infer_ms: Optional[float] = None,
        total_frame_latency_ms: float = 0.0,
        unavailable_reason: Optional[str] = None,
    ) -> Optional[FrameObservationRecord]:
        """
        Records a single frame observation if collection is active.
        Immediately streams the row to CSV.
        """
        if not self.is_recording_active or self.is_finished:
            return None

        now_mono = time.monotonic()
        elapsed_s = now_mono - self.start_mono_time

        record = FrameObservationRecord(
            session_id=self.session_id,
            monotonic_time_s=now_mono,
            elapsed_session_s=elapsed_s,
            condition_index=self.current_condition.index,
            condition_name=self.current_condition.name,
            frame_index=frame_idx,
            face_detected=face_detected,
            face_crop_valid=face_crop_valid,
            geometry_valid=geometry_valid,
            right_eye_h_ratio=right_eye_h,
            right_eye_v_ratio=right_eye_v,
            right_eye_ear=right_eye_ear,
            left_eye_h_ratio=left_eye_h,
            left_eye_v_ratio=left_eye_v,
            left_eye_ear=left_eye_ear,
            canthal_normal_dist_px=canthal_normal_dist_px,
            mobileone_valid=mobileone_valid,
            mobileone_yaw_deg=mobileone_yaw_deg,
            mobileone_pitch_deg=mobileone_pitch_deg,
            mediapipe_latency_ms=mediapipe_latency_ms,
            crop_latency_ms=crop_latency_ms,
            mobileone_preprocess_ms=mobileone_preprocess_ms,
            mobileone_infer_ms=mobileone_infer_ms,
            total_frame_latency_ms=total_frame_latency_ms,
            unavailable_reason=unavailable_reason,
        )

        self.records.append(record)

        # Stream row immediately to CSV
        if self._csv_writer and self._csv_file:
            self._csv_writer.writerow(record.to_csv_row())
            self._csv_file.flush()

        return record

    def complete_session(self) -> Dict[str, Any]:
        """Marks session as cleanly COMPLETED and saves JSON summary."""
        self.is_recording_active = False
        self.status = "COMPLETED"
        return self._finalize()

    def abort_session(self) -> Dict[str, Any]:
        """Marks session as ABORTED, preserves collected data, and saves JSON summary."""
        self.is_recording_active = False
        self.status = "ABORTED"
        if self.condition_states.get(self.current_condition_idx) == "RECORDING":
            self.condition_states[self.current_condition_idx] = "INTERRUPTED"
        return self._finalize()

    def close(self) -> None:
        """Closes open CSV file handle cleanly."""
        if self._csv_file:
            try:
                self._csv_file.flush()
                self._csv_file.close()
            except Exception:
                pass
            self._csv_file = None

    def __del__(self) -> None:
        self.close()

    def _finalize(self) -> Dict[str, Any]:
        if self.end_time_iso is None:
            self.end_time_iso = datetime.now().isoformat()
        elapsed_total = time.monotonic() - self.start_mono_time

        # Close CSV handle cleanly
        self.close()

        summary = compute_observation_summary(
            records=self.records,
            session_id=self.session_id,
            session_status=self.status,
            condition_states=self.condition_states,
            start_time_iso=self.start_time_iso,
            end_time_iso=self.end_time_iso,
            elapsed_duration_s=elapsed_total,
        )

        # Write summary JSON
        with open(self.json_summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary


# =====================================================================
# Run 2 Protocol: Auto-Timed Settling + Fixed Recording Window
# =====================================================================

# Run 2 timing constants (seconds)
V2_SETTLING_DURATION_S: float = 2.0
V2_RECORDING_DURATION_S: float = 5.0

# Run 2 condition phase states
V2_PHASE_WAITING = "WAITING"       # Waiting for user to start condition
V2_PHASE_SETTLING = "SETTLING"     # 2s countdown, frames NOT recorded
V2_PHASE_RECORDING = "RECORDING"   # 5s fixed recording window, frames streamed to CSV
V2_PHASE_DONE = "DONE"             # Condition finished


# Extended CSV schema for Run 2 — adds bilateral normalized canthal ratios
CSV_HEADER_V2: List[str] = [
    "session_id",
    "monotonic_time_s",
    "elapsed_session_s",
    "condition_index",
    "condition_name",
    "frame_index",
    "face_detected",
    "face_crop_valid",
    "geometry_valid",
    "right_eye_h_ratio",
    "right_eye_v_ratio",
    "right_eye_ear",
    "left_eye_h_ratio",
    "left_eye_v_ratio",
    "left_eye_ear",
    "canthal_normal_dist_px",
    "right_eye_canthal_ratio",
    "left_eye_canthal_ratio",
    "mobileone_valid",
    "mobileone_yaw_deg",
    "mobileone_pitch_deg",
    "mediapipe_latency_ms",
    "crop_latency_ms",
    "mobileone_preprocess_ms",
    "mobileone_infer_ms",
    "total_frame_latency_ms",
    "unavailable_reason",
]


@dataclass
class FrameObservationRecordV2:
    """Single frame numerical measurement record for Run 2 protocol.

    Extends the Run 1 schema with bilateral normalized canthal ratios.
    """

    session_id: str
    monotonic_time_s: float
    elapsed_session_s: float
    condition_index: int
    condition_name: str
    frame_index: int
    face_detected: bool
    face_crop_valid: bool
    geometry_valid: bool
    right_eye_h_ratio: Optional[float] = None
    right_eye_v_ratio: Optional[float] = None
    right_eye_ear: Optional[float] = None
    left_eye_h_ratio: Optional[float] = None
    left_eye_v_ratio: Optional[float] = None
    left_eye_ear: Optional[float] = None
    canthal_normal_dist_px: Optional[float] = None
    right_eye_canthal_ratio: Optional[float] = None
    left_eye_canthal_ratio: Optional[float] = None
    mobileone_valid: bool = False
    mobileone_yaw_deg: Optional[float] = None
    mobileone_pitch_deg: Optional[float] = None
    mediapipe_latency_ms: float = 0.0
    crop_latency_ms: float = 0.0
    mobileone_preprocess_ms: Optional[float] = None
    mobileone_infer_ms: Optional[float] = None
    total_frame_latency_ms: float = 0.0
    unavailable_reason: Optional[str] = None

    def to_csv_row(self) -> List[str]:
        """Formats record into CSV string cells, using empty string for missing values."""
        def _fmt_float(val: Optional[float], decimals: int = 4) -> str:
            if val is None or not math.isfinite(val):
                return ""
            return f"{val:.{decimals}f}"

        def _fmt_bool(val: bool) -> str:
            return "true" if val else "false"

        return [
            self.session_id,
            _fmt_float(self.monotonic_time_s, 6),
            _fmt_float(self.elapsed_session_s, 3),
            str(self.condition_index),
            self.condition_name,
            str(self.frame_index),
            _fmt_bool(self.face_detected),
            _fmt_bool(self.face_crop_valid),
            _fmt_bool(self.geometry_valid),
            _fmt_float(self.right_eye_h_ratio, 4),
            _fmt_float(self.right_eye_v_ratio, 4),
            _fmt_float(self.right_eye_ear, 4),
            _fmt_float(self.left_eye_h_ratio, 4),
            _fmt_float(self.left_eye_v_ratio, 4),
            _fmt_float(self.left_eye_ear, 4),
            _fmt_float(self.canthal_normal_dist_px, 2),
            _fmt_float(self.right_eye_canthal_ratio, 4),
            _fmt_float(self.left_eye_canthal_ratio, 4),
            _fmt_bool(self.mobileone_valid),
            _fmt_float(self.mobileone_yaw_deg, 2),
            _fmt_float(self.mobileone_pitch_deg, 2),
            _fmt_float(self.mediapipe_latency_ms, 2),
            _fmt_float(self.crop_latency_ms, 2),
            _fmt_float(self.mobileone_preprocess_ms, 2),
            _fmt_float(self.mobileone_infer_ms, 2),
            _fmt_float(self.total_frame_latency_ms, 2),
            self.unavailable_reason or "",
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Returns JSON-serializable dictionary with None for missing values."""
        return asdict(self)


@dataclass
class V2ConditionTiming:
    """Tracks timing metadata for a single Run 2 condition."""
    settling_start_mono: Optional[float] = None
    recording_start_mono: Optional[float] = None
    recording_end_mono: Optional[float] = None
    frames_during_settling: int = 0
    frames_during_recording: int = 0


class ObservationSessionV2:
    """
    Run 2 Protocol session manager with auto-timed phases.

    Each condition proceeds through:
      WAITING → SETTLING (2s, frames displayed but NOT logged) → RECORDING (5s, frames logged) → DONE

    Transitions from SETTLING to RECORDING and from RECORDING to DONE are automatic
    based on elapsed monotonic time. The user presses [SPACE] to initiate the
    WAITING → SETTLING transition.

    Differences from Run 1 (ObservationSession):
    - Fixed 2s settling + 5s recording per condition (no manual stop needed).
    - Extended CSV schema with bilateral normalized canthal ratios.
    - Per-condition timing metadata in the summary JSON.
    - Fully backward-compatible: Run 1 mode still uses ObservationSession.
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "artifacts/observations",
        output_file: Optional[Union[str, Path]] = None,
        session_id: Optional[str] = None,
        settling_duration_s: float = V2_SETTLING_DURATION_S,
        recording_duration_s: float = V2_RECORDING_DURATION_S,
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if session_id is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            token = uuid.uuid4().hex[:6]
            self.session_id = f"obs_v2_{ts}_{token}"
        else:
            self.session_id = session_id

        self.settling_duration_s = settling_duration_s
        self.recording_duration_s = recording_duration_s

        # Determine CSV filepath
        if output_file is not None:
            target_csv = Path(output_file).resolve()
            if target_csv.exists():
                target_csv = get_unique_output_path(
                    target_csv.parent,
                    prefix=target_csv.stem,
                    ext=target_csv.suffix.lstrip(".") or "csv",
                )
            self.csv_path = target_csv
        else:
            self.csv_path = self.output_dir / f"{self.session_id}.csv"

        self.json_summary_path = self.csv_path.with_name(f"{self.csv_path.stem}_summary.json")

        self.start_time_iso = datetime.now().isoformat()
        self.start_mono_time = time.monotonic()
        self.end_time_iso: Optional[str] = None

        self.status = "INITIALIZED"
        self.current_condition_idx: int = 1
        self.condition_states: Dict[int, str] = {c.index: "NOT_STARTED" for c in OBSERVATION_CONDITIONS}

        # Per-condition phase and timing
        self._condition_phase: Dict[int, str] = {c.index: V2_PHASE_WAITING for c in OBSERVATION_CONDITIONS}
        self._condition_timing: Dict[int, V2ConditionTiming] = {c.index: V2ConditionTiming() for c in OBSERVATION_CONDITIONS}

        self.records: List[FrameObservationRecordV2] = []
        self._csv_file = None
        self._csv_writer = None

        # Protocol version metadata
        self.protocol_version = "v2"

        self._init_csv()

    def _init_csv(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_file = open(self.csv_path, mode="w", newline="", encoding="utf-8")
        import csv
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(CSV_HEADER_V2)
        self._csv_file.flush()

    @property
    def current_condition(self) -> ObservationCondition:
        idx = min(self.current_condition_idx, len(OBSERVATION_CONDITIONS))
        return OBSERVATION_CONDITIONS[idx - 1]

    @property
    def current_phase(self) -> str:
        return self._condition_phase.get(self.current_condition_idx, V2_PHASE_WAITING)

    @property
    def is_settling(self) -> bool:
        return self.current_phase == V2_PHASE_SETTLING

    @property
    def is_recording_active(self) -> bool:
        return self.current_phase == V2_PHASE_RECORDING

    @property
    def is_finished(self) -> bool:
        return self.status in ("COMPLETED", "ABORTED")

    def get_settling_remaining_s(self, now_mono: Optional[float] = None) -> float:
        """Returns seconds remaining in the settling phase, or 0.0 if not settling."""
        if self.current_phase != V2_PHASE_SETTLING:
            return 0.0
        timing = self._condition_timing[self.current_condition_idx]
        if timing.settling_start_mono is None:
            return self.settling_duration_s
        if now_mono is None:
            now_mono = time.monotonic()
        elapsed = now_mono - timing.settling_start_mono
        return max(0.0, self.settling_duration_s - elapsed)

    def get_recording_remaining_s(self, now_mono: Optional[float] = None) -> float:
        """Returns seconds remaining in the recording phase, or 0.0 if not recording."""
        if self.current_phase != V2_PHASE_RECORDING:
            return 0.0
        timing = self._condition_timing[self.current_condition_idx]
        if timing.recording_start_mono is None:
            return self.recording_duration_s
        if now_mono is None:
            now_mono = time.monotonic()
        elapsed = now_mono - timing.recording_start_mono
        return max(0.0, self.recording_duration_s - elapsed)

    def start_condition(self, now_mono: Optional[float] = None) -> None:
        """Begin the settling phase for the current condition (called when user presses SPACE)."""
        if self.is_finished:
            return
        if now_mono is None:
            now_mono = time.monotonic()

        self.status = "ACTIVE"
        self._condition_phase[self.current_condition_idx] = V2_PHASE_SETTLING
        self.condition_states[self.current_condition_idx] = "SETTLING"
        self._condition_timing[self.current_condition_idx].settling_start_mono = now_mono

    def tick(self, now_mono: Optional[float] = None) -> str:
        """
        Called every frame. Checks phase timing and auto-transitions.
        Returns the current phase after any transitions.
        """
        if self.is_finished:
            return V2_PHASE_DONE
        if now_mono is None:
            now_mono = time.monotonic()

        phase = self._condition_phase[self.current_condition_idx]
        timing = self._condition_timing[self.current_condition_idx]

        if phase == V2_PHASE_SETTLING and timing.settling_start_mono is not None:
            elapsed = now_mono - timing.settling_start_mono
            if elapsed >= self.settling_duration_s:
                # Auto-transition to RECORDING
                self._condition_phase[self.current_condition_idx] = V2_PHASE_RECORDING
                self.condition_states[self.current_condition_idx] = "RECORDING"
                timing.recording_start_mono = now_mono
                phase = V2_PHASE_RECORDING

        if phase == V2_PHASE_RECORDING and timing.recording_start_mono is not None:
            elapsed = now_mono - timing.recording_start_mono
            if elapsed >= self.recording_duration_s:
                # Auto-finish condition
                timing.recording_end_mono = now_mono
                self._auto_finish_condition()
                return self.current_phase

        return phase

    def _auto_finish_condition(self) -> None:
        """Auto-finishes the current condition and advances to the next."""
        self._condition_phase[self.current_condition_idx] = V2_PHASE_DONE
        self.condition_states[self.current_condition_idx] = "COMPLETED"

        if self.current_condition_idx >= len(OBSERVATION_CONDITIONS):
            self.complete_session()
            return

        self.current_condition_idx += 1
        self.condition_states[self.current_condition_idx] = "NOT_STARTED"
        self._condition_phase[self.current_condition_idx] = V2_PHASE_WAITING

    def skip_current_condition(self) -> bool:
        """Skips the current condition and advances."""
        if self.is_finished:
            return False

        self._condition_phase[self.current_condition_idx] = V2_PHASE_DONE
        self.condition_states[self.current_condition_idx] = "SKIPPED"

        if self.current_condition_idx >= len(OBSERVATION_CONDITIONS):
            self.complete_session()
            return False

        self.current_condition_idx += 1
        self.condition_states[self.current_condition_idx] = "NOT_STARTED"
        self._condition_phase[self.current_condition_idx] = V2_PHASE_WAITING
        return True

    def record_frame(
        self,
        frame_idx: int,
        face_detected: bool,
        face_crop_valid: bool,
        geometry_valid: bool,
        right_eye_h: Optional[float] = None,
        right_eye_v: Optional[float] = None,
        right_eye_ear: Optional[float] = None,
        left_eye_h: Optional[float] = None,
        left_eye_v: Optional[float] = None,
        left_eye_ear: Optional[float] = None,
        canthal_normal_dist_px: Optional[float] = None,
        right_eye_canthal_ratio: Optional[float] = None,
        left_eye_canthal_ratio: Optional[float] = None,
        mobileone_valid: bool = False,
        mobileone_yaw_deg: Optional[float] = None,
        mobileone_pitch_deg: Optional[float] = None,
        mediapipe_latency_ms: float = 0.0,
        crop_latency_ms: float = 0.0,
        mobileone_preprocess_ms: Optional[float] = None,
        mobileone_infer_ms: Optional[float] = None,
        total_frame_latency_ms: float = 0.0,
        unavailable_reason: Optional[str] = None,
    ) -> Optional[FrameObservationRecordV2]:
        """
        Records a single frame observation if the current condition is in RECORDING phase.
        Frames during SETTLING are silently discarded (not written to CSV).
        """
        phase = self._condition_phase.get(self.current_condition_idx, V2_PHASE_WAITING)

        # Track settling frames for diagnostics, but do NOT record them
        if phase == V2_PHASE_SETTLING:
            self._condition_timing[self.current_condition_idx].frames_during_settling += 1
            return None

        if phase != V2_PHASE_RECORDING or self.is_finished:
            return None

        now_mono = time.monotonic()
        elapsed_s = now_mono - self.start_mono_time

        record = FrameObservationRecordV2(
            session_id=self.session_id,
            monotonic_time_s=now_mono,
            elapsed_session_s=elapsed_s,
            condition_index=self.current_condition.index,
            condition_name=self.current_condition.name,
            frame_index=frame_idx,
            face_detected=face_detected,
            face_crop_valid=face_crop_valid,
            geometry_valid=geometry_valid,
            right_eye_h_ratio=right_eye_h,
            right_eye_v_ratio=right_eye_v,
            right_eye_ear=right_eye_ear,
            left_eye_h_ratio=left_eye_h,
            left_eye_v_ratio=left_eye_v,
            left_eye_ear=left_eye_ear,
            canthal_normal_dist_px=canthal_normal_dist_px,
            right_eye_canthal_ratio=right_eye_canthal_ratio,
            left_eye_canthal_ratio=left_eye_canthal_ratio,
            mobileone_valid=mobileone_valid,
            mobileone_yaw_deg=mobileone_yaw_deg,
            mobileone_pitch_deg=mobileone_pitch_deg,
            mediapipe_latency_ms=mediapipe_latency_ms,
            crop_latency_ms=crop_latency_ms,
            mobileone_preprocess_ms=mobileone_preprocess_ms,
            mobileone_infer_ms=mobileone_infer_ms,
            total_frame_latency_ms=total_frame_latency_ms,
            unavailable_reason=unavailable_reason,
        )

        self.records.append(record)
        self._condition_timing[self.current_condition_idx].frames_during_recording += 1

        if self._csv_writer and self._csv_file:
            self._csv_writer.writerow(record.to_csv_row())
            self._csv_file.flush()

        return record

    def complete_session(self) -> Dict[str, Any]:
        """Marks session as cleanly COMPLETED and saves JSON summary."""
        self.status = "COMPLETED"
        return self._finalize()

    def abort_session(self) -> Dict[str, Any]:
        """Marks session as ABORTED, preserves collected data, and saves JSON summary."""
        self.status = "ABORTED"
        phase = self._condition_phase.get(self.current_condition_idx)
        if phase in (V2_PHASE_SETTLING, V2_PHASE_RECORDING):
            self.condition_states[self.current_condition_idx] = "INTERRUPTED"
            self._condition_phase[self.current_condition_idx] = V2_PHASE_DONE
        return self._finalize()

    def close(self) -> None:
        """Closes open CSV file handle cleanly."""
        if self._csv_file:
            try:
                self._csv_file.flush()
                self._csv_file.close()
            except Exception:
                pass
            self._csv_file = None

    def __del__(self) -> None:
        self.close()

    def _finalize(self) -> Dict[str, Any]:
        if self.end_time_iso is None:
            self.end_time_iso = datetime.now().isoformat()
        elapsed_total = time.monotonic() - self.start_mono_time

        self.close()

        # Build V2-specific per-condition timing metadata
        condition_timing_meta = {}
        for cond in OBSERVATION_CONDITIONS:
            t = self._condition_timing[cond.index]
            settling_s = None
            recording_s = None
            if t.settling_start_mono is not None and t.recording_start_mono is not None:
                settling_s = round(t.recording_start_mono - t.settling_start_mono, 3)
            if t.recording_start_mono is not None and t.recording_end_mono is not None:
                recording_s = round(t.recording_end_mono - t.recording_start_mono, 3)
            condition_timing_meta[cond.index] = {
                "settling_duration_actual_s": settling_s,
                "recording_duration_actual_s": recording_s,
                "frames_during_settling": t.frames_during_settling,
                "frames_during_recording": t.frames_during_recording,
            }

        # Reuse V1 summary computation with V2 records converted to V1-compatible format
        v1_records = []
        for r in self.records:
            v1_records.append(FrameObservationRecord(
                session_id=r.session_id,
                monotonic_time_s=r.monotonic_time_s,
                elapsed_session_s=r.elapsed_session_s,
                condition_index=r.condition_index,
                condition_name=r.condition_name,
                frame_index=r.frame_index,
                face_detected=r.face_detected,
                face_crop_valid=r.face_crop_valid,
                geometry_valid=r.geometry_valid,
                right_eye_h_ratio=r.right_eye_h_ratio,
                right_eye_v_ratio=r.right_eye_v_ratio,
                right_eye_ear=r.right_eye_ear,
                left_eye_h_ratio=r.left_eye_h_ratio,
                left_eye_v_ratio=r.left_eye_v_ratio,
                left_eye_ear=r.left_eye_ear,
                canthal_normal_dist_px=r.canthal_normal_dist_px,
                mobileone_valid=r.mobileone_valid,
                mobileone_yaw_deg=r.mobileone_yaw_deg,
                mobileone_pitch_deg=r.mobileone_pitch_deg,
                mediapipe_latency_ms=r.mediapipe_latency_ms,
                crop_latency_ms=r.crop_latency_ms,
                mobileone_preprocess_ms=r.mobileone_preprocess_ms,
                mobileone_infer_ms=r.mobileone_infer_ms,
                total_frame_latency_ms=r.total_frame_latency_ms,
                unavailable_reason=r.unavailable_reason,
            ))

        summary = compute_observation_summary(
            records=v1_records,
            session_id=self.session_id,
            session_status=self.status,
            condition_states=self.condition_states,
            start_time_iso=self.start_time_iso,
            end_time_iso=self.end_time_iso,
            elapsed_duration_s=elapsed_total,
        )

        # Inject V2-specific fields
        summary["metadata"]["protocol_version"] = self.protocol_version
        summary["metadata"]["settling_duration_config_s"] = self.settling_duration_s
        summary["metadata"]["recording_duration_config_s"] = self.recording_duration_s
        summary["condition_timing_v2"] = condition_timing_meta

        with open(self.json_summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary


# Visual fixation target definitions for Run 2 HUD
# Each condition maps to a target position as a fraction of screen dimensions:
#   (x_frac, y_frac) where (0.0, 0.0) = top-left, (1.0, 1.0) = bottom-right
# None means no special target (use the main camera view).
V2_FIXATION_TARGETS: Dict[str, Optional[Tuple[float, float]]] = {
    "CENTER_BASELINE": (0.5, 0.5),
    "EYES_LEFT": (0.08, 0.5),
    "EYES_CENTER_RETURN": (0.5, 0.5),
    "EYES_RIGHT": (0.92, 0.5),
    "EYES_UP": (0.5, 0.08),
    "EYES_DOWN": (0.5, 0.92),
    "HEAD_LEFT_EYES_FORWARD": (0.5, 0.5),
    "HEAD_CENTER_RETURN": (0.5, 0.5),
    "HEAD_RIGHT_EYES_FORWARD": (0.5, 0.5),
    "HEAD_UP_EYES_FORWARD": (0.5, 0.5),
    "HEAD_DOWN_EYES_FORWARD": (0.5, 0.5),
    "COMBINED_MOVEMENT": None,
    "TRACKING_RECOVERY": None,
    "FINAL_CENTER_BASELINE": (0.5, 0.5),
}
