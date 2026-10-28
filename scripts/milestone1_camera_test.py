"""
VoxGaze Milestone M1: Live Camera & Gaze Sensor Validation Harness
Integrates MediaPipe Face Landmarker Tasks API in VIDEO mode, camera capability profiler,
interactive 5-point calibration, directional hysteresis HUD, guided multi-trial evaluation,
granular latency profiling (capture, MediaPipe, geometry, classification, end-to-end),
physical geometry diagnostic mode, and machine-readable artifact export (artifacts/m1/).
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time
import urllib.request
from typing import Dict, List, Optional, Tuple

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from backend.app.core.contracts import (
    CalibrationProfile,
    Direction,
    EyeLandmarks,
    GazeFeatures,
    GazeState,
    Point2D,
)
from backend.app.calibration.profile import load_calibration_profile, save_calibration_profile
from backend.app.calibration.service import CalibrationService
from backend.app.perception.diagnostic import (
    DiagnosticFrameData,
    DiagnosticTargetBuffer,
    TargetDiagnosticStats,
    compute_raw_coordinate_deltas,
    export_diagnostic_artifact,
    format_diagnostic_summary,
    verify_physical_ordering,
)
from backend.app.perception.gaze_engine import GazeEngine
from backend.app.perception.geometry import (
    LEFT_INNER_CANTHUS,
    LEFT_IRIS_ALL,
    LEFT_IRIS_CENTER,
    LEFT_LOWER_LID,
    LEFT_OUTER_CANTHUS,
    LEFT_UPPER_LID,
    RIGHT_INNER_CANTHUS,
    RIGHT_IRIS_ALL,
    RIGHT_IRIS_CENTER,
    RIGHT_LOWER_LID,
    RIGHT_OUTER_CANTHUS,
    RIGHT_UPPER_LID,
    compute_ear,
    compute_canthal_normal_distance_px,
    extract_bilateral_features,
    extract_eye_features,
    norm_vertical_gaze_canthal_normal_experimental,
)

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "face_landmarker.task"
ARTIFACTS_DIR = Path("artifacts/m1")


def ensure_model_asset() -> Path:
    """Download the official MediaPipe Face Landmarker task model bundle if missing."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.is_file():
        print(f"Downloading MediaPipe Face Landmarker model from {MODEL_URL}...")
        urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
        print(f"Model saved to {MODEL_PATH}")
    return MODEL_PATH


class GranularLatencyProfiler:
    """Tracks latency breakdown across all pipeline stages with p50 and p95 calculations."""

    def __init__(self):
        self.capture_latencies: List[float] = []
        self.mediapipe_latencies: List[float] = []
        self.geometry_latencies: List[float] = []
        self.classification_latencies: List[float] = []
        self.end_to_end_latencies: List[float] = []

    def record_frame(self, t_cap: float, t_mp: float, t_geom: float, t_class: float, t_e2e: float):
        self.capture_latencies.append(t_cap)
        self.mediapipe_latencies.append(t_mp)
        self.geometry_latencies.append(t_geom)
        self.classification_latencies.append(t_class)
        self.end_to_end_latencies.append(t_e2e)

    def _get_stats(self, arr_list: List[float]) -> Dict[str, float]:
        if not arr_list:
            return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
        arr = np.array(arr_list)
        return {
            "mean": float(np.mean(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
        }

    def get_summary(self) -> Dict[str, any]:
        return {
            "camera_capture_latency_ms": self._get_stats(self.capture_latencies),
            "mediapipe_inference_latency_ms": self._get_stats(self.mediapipe_latencies),
            "geometry_normalization_latency_ms": self._get_stats(self.geometry_latencies),
            "gaze_classification_latency_ms": self._get_stats(self.classification_latencies),
            "end_to_end_processing_latency_ms": self._get_stats(self.end_to_end_latencies),
        }


class BenchmarkCollector:
    """Records event-level trials, confusion matrix, and tracking rates."""

    def __init__(self):
        self.total_frames = 0
        self.face_frames = 0
        self.iris_frames = 0
        self.trial_records: List[Tuple[str, str, bool]] = []  # (expected, predicted, is_correct)
        self.false_activations = 0
        self.passive_test_duration_sec = 0.0

    def record_frame(self, face_detected: bool, iris_detected: bool):
        self.total_frames += 1
        if face_detected:
            self.face_frames += 1
        if iris_detected:
            self.iris_frames += 1

    def record_trial(self, expected: str, predicted: str):
        is_correct = (expected == predicted)
        self.trial_records.append((expected, predicted, is_correct))

    def get_tracking_rates(self) -> Dict[str, float]:
        if self.total_frames == 0:
            return {"face_tracking_rate": 0.0, "iris_tracking_rate": 0.0}
        return {
            "face_tracking_rate": float(self.face_frames / self.total_frames),
            "iris_tracking_rate": float(self.iris_frames / self.total_frames),
        }

    def get_event_accuracy(self) -> float:
        if not self.trial_records:
            return 0.0
        correct = sum(1 for _, _, ok in self.trial_records if ok)
        return float(correct / len(self.trial_records))

    def export_confusion_matrix_csv(self, output_path: Path):
        directions = ["CENTER", "LEFT", "RIGHT", "UP", "DOWN", "ABSTAIN"]
        matrix = {exp: {pred: 0 for pred in directions} for exp in directions[:5]}

        for exp, pred, _ in self.trial_records:
            if exp in matrix and pred in matrix[exp]:
                matrix[exp][pred] += 1
            elif exp in matrix:
                matrix[exp]["ABSTAIN"] += 1

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["actual_direction"] + directions)
            for exp in directions[:5]:
                row = [exp] + [matrix[exp][pred] for pred in directions]
                writer.writerow(row)


def draw_eye_geometry_overlay(
    raw_frame: np.ndarray,
    r_iris: Point2D,
    r_inner: Point2D,
    r_outer: Point2D,
    r_upper: Point2D,
    r_lower: Point2D,
    h_ratio: float,
    v_ratio: float,
):
    """
    Renders anatomical landmarks and projected horizontal/vertical eye axes
    directly onto the raw camera frame before mirroring.
    """
    ix, iy = int(r_iris.x), int(r_iris.y)
    inn_x, inn_y = int(r_inner.x), int(r_inner.y)
    out_x, out_y = int(r_outer.x), int(r_outer.y)
    up_x, up_y = int(r_upper.x), int(r_upper.y)
    low_x, low_y = int(r_lower.x), int(r_lower.y)

    # 1. Horizontal Axis: Line between inner canthus (133) and outer canthus (33)
    cv2.line(raw_frame, (inn_x, inn_y), (out_x, out_y), (255, 255, 0), 2)

    # 2. Vertical Axis: Line between upper lid (159) and lower lid (145)
    cv2.line(raw_frame, (up_x, up_y), (low_x, low_y), (255, 0, 255), 2)

    # 3. Projected iris point on horizontal axis:
    proj_hx = int(r_inner.x + h_ratio * (r_outer.x - r_inner.x))
    proj_hy = int(r_inner.y + h_ratio * (r_outer.y - r_inner.y))
    cv2.circle(raw_frame, (proj_hx, proj_hy), 4, (0, 255, 255), -1)
    cv2.line(raw_frame, (ix, iy), (proj_hx, proj_hy), (0, 255, 255), 1)

    # 4. Projected iris point on vertical axis:
    proj_vx = int(r_upper.x + v_ratio * (r_lower.x - r_upper.x))
    proj_vy = int(r_upper.y + v_ratio * (r_lower.y - r_upper.y))
    cv2.circle(raw_frame, (proj_vx, proj_vy), 4, (255, 0, 255), -1)
    cv2.line(raw_frame, (ix, iy), (proj_vx, proj_vy), (255, 0, 255), 1)

    # 5. Landmark points
    cv2.circle(raw_frame, (inn_x, inn_y), 4, (0, 255, 255), -1)   # Yellow: Inner Canthus (133)
    cv2.circle(raw_frame, (out_x, out_y), 4, (0, 140, 255), -1)   # Orange: Outer Canthus (33)
    cv2.circle(raw_frame, (up_x, up_y), 4, (255, 200, 0), -1)     # Cyan: Upper Lid (159)
    cv2.circle(raw_frame, (low_x, low_y), 4, (200, 0, 255), -1)   # Magenta: Lower Lid (145)
    cv2.circle(raw_frame, (ix, iy), 4, (0, 255, 0), -1)           # Green: Iris Center (468)


def draw_eye_pip_magnifier(
    display_frame: np.ndarray,
    raw_frame: np.ndarray,
    r_inner: Point2D,
    r_outer: Point2D,
    r_upper: Point2D,
    r_lower: Point2D,
):
    """
    Renders an in-preview Picture-In-Picture (PIP) zoomed crop of the right eye
    in the bottom-right corner for magnified iris inspection.
    """
    h_disp, w_disp = display_frame.shape[:2]
    all_x = [r_inner.x, r_outer.x, r_upper.x, r_lower.x]
    all_y = [r_inner.y, r_outer.y, r_upper.y, r_lower.y]

    cx = int(np.mean(all_x))
    cy = int(np.mean(all_y))
    pad_w = 40
    pad_h = 30

    y1 = max(0, cy - pad_h)
    y2 = min(raw_frame.shape[0], cy + pad_h)
    x1 = max(0, cx - pad_w)
    x2 = min(raw_frame.shape[1], cx + pad_w)

    crop = raw_frame[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
        return

    # Mirror the crop to match preview perspective
    crop_mirrored = cv2.flip(crop, 1)

    # Scale 2.5x
    pip_w, pip_h = 200, 150
    crop_resized = cv2.resize(crop_mirrored, (pip_w, pip_h), interpolation=cv2.INTER_LINEAR)

    # Place in bottom-right corner
    px1 = w_disp - pip_w - 20
    py1 = h_disp - pip_h - 20
    px2 = px1 + pip_w
    py2 = py1 + pip_h

    display_frame[py1:py2, px1:px2] = crop_resized
    cv2.rectangle(display_frame, (px1, py1), (px2, py2), (0, 255, 255), 2)
    cv2.putText(display_frame, "RIGHT EYE 2.5X ZOOM", (px1 + 10, py1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)


def draw_high_contrast_target(
    img: np.ndarray,
    target: str,
    w_px: int,
    h_px: int,
    collected_samples: int = 0,
    required_samples: int = 45,
    mode_label: str = "CALIBRATION",
):
    """
    Renders high-contrast visual calibration/diagnostic targets on the mirrored display preview.
    Target coordinates map directly to user screen gaze directions.
    """
    if target == "CLOSED":
        cx, cy = w_px // 2, h_px // 2
        overlay = img.copy()
        cv2.rectangle(overlay, (cx - 300, cy - 80), (cx + 300, cy + 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
        cv2.rectangle(img, (cx - 300, cy - 80), (cx + 300, cy + 80), (0, 200, 255), 3)
        cv2.putText(
            img,
            "*** KEEP EYES CLOSED (BLINK HOLD) ***",
            (cx - 275, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            img,
            f"Measuring baseline EAR... ({collected_samples}/{required_samples} samples)",
            (cx - 240, cy + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        return

    # Screen coordinates on mirrored display
    coords = {
        "CENTER": (w_px // 2, h_px // 2),
        "LEFT": (int(w_px * 0.12), h_px // 2),
        "RIGHT": (int(w_px * 0.88), h_px // 2),
        "UP": (w_px // 2, int(h_px * 0.15)),
        "DOWN": (w_px // 2, int(h_px * 0.85)),
    }
    cx, cy = coords.get(target, (w_px // 2, h_px // 2))

    # Outer crosshairs
    cv2.line(img, (cx - 60, cy), (cx + 60, cy), (0, 0, 0), 6)
    cv2.line(img, (cx, cy - 60), (cx, cy + 60), (0, 0, 0), 6)
    cv2.line(img, (cx - 60, cy), (cx + 60, cy), (0, 255, 255), 2)
    cv2.line(img, (cx, cy - 60), (cx, cy + 60), (0, 255, 255), 2)

    # High-contrast concentric bullseye rings
    cv2.circle(img, (cx, cy), 38, (0, 0, 0), 6)
    cv2.circle(img, (cx, cy), 35, (0, 255, 255), 3)
    cv2.circle(img, (cx, cy), 22, (0, 0, 200), -1)
    cv2.circle(img, (cx, cy), 12, (0, 255, 255), -1)
    cv2.circle(img, (cx, cy), 4, (255, 255, 255), -1)

    # High-contrast target text banner
    label = f"[{mode_label}] FIXATE HERE: {target} ({collected_samples}/{required_samples})"
    label_y = cy - 50 if cy > 100 else cy + 65
    label_x = max(20, min(w_px - 440, cx - 180))
    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4)
    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)

    # Prominent Head-Fixed Control Trial overlay banner
    if mode_label == "DIAGNOSTIC":
        banner_w, banner_h = 760, 100
        bx1 = (w_px - banner_w) // 2
        by1 = 12
        bx2 = bx1 + banner_w
        by2 = by1 + banner_h

        overlay = img.copy()
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.82, img, 0.18, 0, img)
        cv2.rectangle(img, (bx1, by1), (bx2, by2), (0, 165, 255), 3)

        cv2.putText(
            img,
            "*** HEAD-FIXED CONTROL TRIAL ***",
            (bx1 + 180, by1 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            img,
            "Rest your chin on a stable support. Keep your face stationary.",
            (bx1 + 45, by1 + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            img,
            "Move ONLY your eyes to the target. Do not turn or pitch your head.",
            (bx1 + 35, by1 + 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (0, 200, 255),
            2,
        )


def draw_diagnostic_hud(
    img: np.ndarray,
    gaze_state: GazeState,
    fps: float,
    t_mp: float,
    t_e2e: float,
    r_iris: Optional[Point2D] = None,
    r_inner: Optional[Point2D] = None,
    r_outer: Optional[Point2D] = None,
    r_upper: Optional[Point2D] = None,
    r_lower: Optional[Point2D] = None,
    active_eye_name: str = "RIGHT_EYE_BASELINE",
    calib_service: Optional[CalibrationService] = None,
    in_calibration: bool = False,
    curr_calib_target: str = "",
    in_diagnostic: bool = False,
    curr_diag_target: str = "",
    diag_samples_count: int = 0,
):
    """
    Renders live diagnostic HUD showing raw pixel coordinates, normalized ratios,
    directional state, FPS/latency, eye dimensions, and calibration/diagnostic progression.
    """
    h_box = 235 if (in_calibration or in_diagnostic) else 170
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (560, h_box), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.70, img, 0.30, 0, img)
    cv2.rectangle(img, (10, 10), (560, h_box), (80, 80, 80), 1)

    # 1. Anatomical Eye Identity & Landmarks
    cv2.putText(
        img,
        f"ACTIVE EYE: {active_eye_name}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (180, 255, 180),
        1,
    )

    if r_iris and r_inner and r_outer and r_upper and r_lower:
        eye_w = math.hypot(r_inner.x - r_outer.x, r_inner.y - r_outer.y)
        eye_h = math.hypot(r_upper.x - r_lower.x, r_upper.y - r_lower.y)
        cv2.putText(
            img,
            f"IRIS(468): ({r_iris.x:.1f}, {r_iris.y:.1f}) | WIDTH: {eye_w:.1f}px | OPEN: {eye_h:.1f}px",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (220, 220, 220),
            1,
        )
        cv2.putText(
            img,
            f"OUTER(33): ({r_outer.x:.0f},{r_outer.y:.0f}) | INNER(133): ({r_inner.x:.0f},{r_inner.y:.0f})",
            (20, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (180, 180, 180),
            1,
        )
        cv2.putText(
            img,
            f"UPPER(159): ({r_upper.x:.0f},{r_upper.y:.0f}) | LOWER(145): ({r_lower.x:.0f},{r_lower.y:.0f})",
            (20, 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (180, 180, 180),
            1,
        )

    # 2. State & Confidence
    status_color = (0, 255, 0) if gaze_state.direction in (
        Direction.CENTER, Direction.LEFT, Direction.RIGHT, Direction.UP, Direction.DOWN
    ) else (0, 165, 255)
    if gaze_state.direction == Direction.NO_FACE:
        status_color = (0, 0, 255)

    cv2.putText(img, f"DIR: {gaze_state.direction.value}", (20, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(img, f"[{gaze_state.status_flag}]", (240, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.60, status_color, 2)

    # 3. Normalized Ratios & EAR
    cv2.putText(
        img,
        f"NORM: H={gaze_state.raw_h:.3f} | V={gaze_state.raw_v:.3f} | EAR={gaze_state.ear:.3f}",
        (20, 138),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 0),
        2,
    )

    # 4. Latency / Telemetry
    cv2.putText(
        img,
        f"FPS: {fps:.1f} | MediaPipe: {t_mp:.1f}ms | E2E: {t_e2e:.1f}ms",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (200, 200, 200),
        1,
    )

    # 5. Live Calibration / Diagnostic Diagnostics
    if in_calibration and calib_service and curr_calib_target in calib_service.buffers:
        buf = calib_service.buffers[curr_calib_target]
        cnt = len(buf.h_samples)
        med_h = float(np.median(buf.h_samples)) if cnt > 0 else 0.0
        mad_h = float(np.median(np.abs(np.array(buf.h_samples) - med_h))) if cnt > 0 else 0.0
        med_v = float(np.median(buf.v_samples)) if cnt > 0 else 0.0
        mad_v = float(np.median(np.abs(np.array(buf.v_samples) - med_v))) if cnt > 0 else 0.0
        med_ear = float(np.median(buf.ear_samples)) if len(buf.ear_samples) > 0 else 0.0

        center_buf = calib_service.buffers["CENTER"]
        center_med_h = float(np.median(center_buf.h_samples)) if len(center_buf.h_samples) > 0 else 0.50
        center_med_v = float(np.median(center_buf.v_samples)) if len(center_buf.v_samples) > 0 else 0.50

        sep_text = ""
        if curr_calib_target == "LEFT":
            sep = center_med_h - med_h
            sep_text = f"Sep: {sep:+.3f} (>= +0.06)"
        elif curr_calib_target == "RIGHT":
            sep = med_h - center_med_h
            sep_text = f"Sep: {sep:+.3f} (>= +0.06)"
        elif curr_calib_target == "UP":
            sep = center_med_v - med_v
            sep_text = f"Sep: {sep:+.3f} (>= +0.06)"
        elif curr_calib_target == "DOWN":
            sep = med_v - center_med_v
            sep_text = f"Sep: {sep:+.3f} (>= +0.06)"
        elif curr_calib_target == "CLOSED":
            open_ear = float(np.median(center_buf.ear_samples)) if len(center_buf.ear_samples) > 0 else 0.30
            sep = open_ear - med_ear
            sep_text = f"EAR Sep: {sep:.3f} (>= +0.05)"

        cv2.putText(
            img,
            f"CALIB TARGET: {curr_calib_target} | N={cnt} | Med H={med_h:.2f} MAD={mad_h:.3f}",
            (20, 188),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 255),
            1,
        )
        cv2.putText(
            img,
            f"Med V={med_v:.2f} MAD={mad_v:.3f} | EAR={med_ear:.2f} | {sep_text}",
            (20, 214),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 0) if "+" in sep_text or "EAR Sep" in sep_text else (200, 200, 200),
            1,
        )

    elif in_diagnostic:
        cv2.putText(
            img,
            f"*** DIAGNOSTIC MODE: {curr_diag_target} ({diag_samples_count}/50 samples) ***",
            (20, 192),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            img,
            "CONTROLLED TRIAL: Keep head RIGID; rotate ONLY eyes to target!",
            (20, 216),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 200, 255),
            2,
        )


def run_headless_simulation(artifacts_dir: Path = ARTIFACTS_DIR) -> Dict[str, any]:
    """
    Runs automated benchmark verification using SYNTHETIC streams for CI regression.
    Explicitly tags provenance as SYNTHETIC.
    """
    from scripts.generate_synthetic_gaze import (
        generate_clean_trajectory,
        generate_noisy_center,
        get_mock_calibration_profile,
    )

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    profiler = GranularLatencyProfiler()
    collector = BenchmarkCollector()
    profile = get_mock_calibration_profile()
    engine = GazeEngine(profile=profile, active_eye="right")

    trial_targets = [Direction.CENTER, Direction.LEFT, Direction.RIGHT, Direction.UP, Direction.DOWN]
    for target in trial_targets:
        for _ in range(20):
            stream = generate_clean_trajectory(target, 15)
            detected_states = []
            for s in stream:
                collector.record_frame(face_detected=True, iris_detected=True)
                t0 = time.perf_counter()
                st = engine.process_features(s)
                t_class = (time.perf_counter() - t0) * 1000.0
                profiler.record_frame(t_cap=0.0, t_mp=0.0, t_geom=0.0, t_class=t_class, t_e2e=t_class)
                detected_states.append(st.direction.value)

            mid_state = max(set(detected_states[5:]), key=detected_states[5:].count)
            collector.record_trial(target.value, mid_state)

    rates = collector.get_tracking_rates()
    lat_summary = profiler.get_summary()
    acc = collector.get_event_accuracy()

    camera_report = {
        "provenance": "SYNTHETIC_SIMULATOR (Automated CI)",
        "hardware_test_completed": False,
        "camera_index": 0,
        "requested_width": 1280,
        "requested_height": 720,
        "actual_width": 1280,
        "actual_height": 720,
        "observed_fps": 30.0,
        "backend": "SYNTHETIC_SIMULATOR",
        "face_tracking_rate": rates["face_tracking_rate"],
        "iris_tracking_rate": rates["iris_tracking_rate"],
        "test_duration_seconds": 60.0,
    }
    with open(artifacts_dir / "camera_report.json", "w", encoding="utf-8") as f:
        json.dump(camera_report, f, indent=2)

    with open(artifacts_dir / "latency_summary.json", "w", encoding="utf-8") as f:
        json.dump(lat_summary, f, indent=2)

    calib_report = {
        "provenance": "SYNTHETIC_CALIBRATION",
        "profile_version": profile.profile_version,
        "algorithm_version": profile.algorithm_version,
        "active_eye": "RIGHT_EYE_BASELINE",
        "stats": {k: v.model_dump() for k, v in profile.stats.items()},
        "thresholds": profile.thresholds.model_dump(),
        "quality_status": "PASSED",
        "quality_warnings": [],
        "quality_errors": [],
    }
    with open(artifacts_dir / "calibration_report.json", "w", encoding="utf-8") as f:
        json.dump(calib_report, f, indent=2)

    collector.export_confusion_matrix_csv(artifacts_dir / "direction_confusion_matrix.csv")

    return {
        "camera_report": camera_report,
        "latency_summary": lat_summary,
        "event_accuracy": acc,
        "trials_recorded": len(collector.trial_records),
    }


def main():
    parser = argparse.ArgumentParser(description="VoxGaze Milestone M1 Validation Harness")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--headless", action="store_true", help="Run automated synthetic benchmark simulation")
    parser.add_argument("--diagnostic", action="store_true", help="Launch directly into Physical Geometry Diagnostic Mode")
    parser.add_argument("--export-dir", type=str, default="artifacts/m1", help="Path for benchmark artifacts")
    args = parser.parse_args()

    export_path = Path(args.export_dir)

    if args.headless:
        print("[INFO] Running M1 synthetic benchmark simulation (CI Mode)...")
        res = run_headless_simulation(export_path)
        print(f"Synthetic artifacts exported to {export_path}/")
        print(f"Synthetic Event-Level Accuracy: {res['event_accuracy'] * 100:.1f}%")
        return

    model_path = ensure_model_asset()

    # Initialize MediaPipe Face Landmarker Tasks API in VIDEO mode
    base_opts = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.FaceLandmarkerOptions(
        base_options=base_opts,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.6,
        min_face_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    profiler = GranularLatencyProfiler()
    collector = BenchmarkCollector()
    gaze_engine = GazeEngine(profile=None, active_eye="right")

    # Try loading existing calibration profile
    profile_path = Path("data/local/calibration_user.json")
    if profile_path.is_file():
        try:
            profile = load_calibration_profile(profile_path)
            gaze_engine.set_profile(profile)
            print(f"[INFO] Loaded calibration profile for user: {profile.user_id}")
        except Exception as e:
            print(f"[WARN] Could not load local profile: {e}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Camera index {args.camera} could not be opened.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    act_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    act_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    backend_name = cap.getBackendName() if hasattr(cap, "getBackendName") else "OPENCV_DEFAULT"

    print("\n========================================")
    print("  VOXGAZE CAMERA CAPABILITY AUDIT")
    print(f"  Camera Index       : {args.camera}")
    print(f"  Active Baseline Eye: RIGHT_EYE_BASELINE (Anatomical)")
    print(f"  Requested Format   : 1280x720")
    print(f"  Actual Resolution  : {act_w}x{act_h}")
    print(f"  Backend Driver     : {backend_name}")
    print("========================================\n")
    print("Controls:")
    print("  [D] - Start Physical Geometry Diagnostic (CENTER -> LEFT -> RIGHT -> UP -> DOWN)")
    print("  [C] - Start 5-Point Calibration (CENTER -> LEFT -> RIGHT -> UP -> DOWN -> CLOSED)")
    print("  [E] - Start Guided 4-Direction Evaluation Trials (20 trials x 5 directions)")
    print("  [F] - Start Passive False-Activation Test (60 seconds)")
    print("  [S] - Save Real Benchmark Artifacts to artifacts/m1/")
    print("  [ESC] - Exit\n")

    calib_targets = ["CENTER", "LEFT", "RIGHT", "UP", "DOWN", "CLOSED"]
    calib_service: Optional[CalibrationService] = None
    calib_target_idx = 0
    in_calibration = False

    # Guided Diagnostic Mode State
    diag_targets = ["CENTER", "LEFT", "RIGHT", "UP", "DOWN"]
    diag_buffers: Dict[str, DiagnosticTargetBuffer] = {}
    diag_target_idx = 0
    in_diagnostic = bool(args.diagnostic)
    if in_diagnostic:
        diag_buffers = {t: DiagnosticTargetBuffer(t, warmup_sec=0.5, collection_sec=2.0) for t in diag_targets}
        diag_target_idx = 0
        print("[INFO] Launching directly into Physical Geometry Diagnostic Sequence...")

    # Guided Evaluation State
    eval_targets = [Direction.CENTER, Direction.LEFT, Direction.RIGHT, Direction.UP, Direction.DOWN]
    in_evaluation = False
    eval_target_idx = 0
    eval_trial_count = 0
    max_trials_per_target = 20
    eval_step_start_ms = 0
    eval_collected_states: List[Direction] = []

    # False Activation State
    in_false_act_test = False
    false_act_start_ms = 0
    false_act_duration_sec = 60.0
    false_act_triggers = 0

    prev_loop_time = time.perf_counter()

    while cap.isOpened():
        t_loop_start = time.perf_counter()

        # 1. Capture RAW unmirrored camera frame
        t_cap_start = time.perf_counter()
        ok, raw_frame = cap.read()
        t_cap = (time.perf_counter() - t_cap_start) * 1000.0
        if not ok:
            break

        h_px, w_px = raw_frame.shape[:2]

        # 2. MediaPipe inference on RAW native coordinates (unmirrored)
        rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.perf_counter() * 1000)

        t_mp_start = time.perf_counter()
        result = detector.detect_for_video(mp_image, timestamp_ms)
        t_mp = (time.perf_counter() - t_mp_start) * 1000.0

        face_detected = bool(result.face_landmarks)
        iris_detected = False
        features: Optional[GazeFeatures] = None

        r_iris = None
        r_inner = None
        r_outer = None
        r_upper = None
        r_lower = None

        l_iris = None
        l_inner = None
        l_outer = None
        l_upper = None
        l_lower = None

        # 3. Pure Geometry Feature Extraction on Anatomical Landmarks
        t_geom_start = time.perf_counter()
        if face_detected:
            face = result.face_landmarks[0]
            pts = np.array([[p.x * w_px, p.y * h_px] for p in face])

            # Anatomical RIGHT eye (subject's right eye, active M1 baseline)
            r_iris = Point2D(x=float(np.mean(pts[RIGHT_IRIS_ALL, 0])), y=float(np.mean(pts[RIGHT_IRIS_ALL, 1])))
            r_inner = Point2D(x=float(pts[RIGHT_INNER_CANTHUS, 0]), y=float(pts[RIGHT_INNER_CANTHUS, 1]))
            r_outer = Point2D(x=float(pts[RIGHT_OUTER_CANTHUS, 0]), y=float(pts[RIGHT_OUTER_CANTHUS, 1]))
            r_upper = Point2D(x=float(pts[RIGHT_UPPER_LID, 0]), y=float(pts[RIGHT_UPPER_LID, 1]))
            r_lower = Point2D(x=float(pts[RIGHT_LOWER_LID, 0]), y=float(pts[RIGHT_LOWER_LID, 1]))
            right_eye_feat = extract_eye_features(r_iris, r_inner, r_outer, r_upper, r_lower, is_right_eye=True)

            # Anatomical LEFT eye (subject's left eye)
            l_iris = Point2D(x=float(np.mean(pts[LEFT_IRIS_ALL, 0])), y=float(np.mean(pts[LEFT_IRIS_ALL, 1])))
            l_inner = Point2D(x=float(pts[LEFT_INNER_CANTHUS, 0]), y=float(pts[LEFT_INNER_CANTHUS, 1]))
            l_outer = Point2D(x=float(pts[LEFT_OUTER_CANTHUS, 0]), y=float(pts[LEFT_OUTER_CANTHUS, 1]))
            l_upper = Point2D(x=float(pts[LEFT_UPPER_LID, 0]), y=float(pts[LEFT_UPPER_LID, 1]))
            l_lower = Point2D(x=float(pts[LEFT_LOWER_LID, 0]), y=float(pts[LEFT_LOWER_LID, 1]))
            left_eye_feat = extract_eye_features(l_iris, l_inner, l_outer, l_upper, l_lower, is_right_eye=False)

            features = extract_bilateral_features(
                timestamp_ms=timestamp_ms,
                left_eye=left_eye_feat,
                right_eye=right_eye_feat,
            )
            iris_detected = True

            # Draw anatomical landmarks and projected axes directly onto raw frame
            draw_eye_geometry_overlay(
                raw_frame,
                r_iris,
                r_inner,
                r_outer,
                r_upper,
                r_lower,
                right_eye_feat.h_ratio,
                right_eye_feat.v_ratio,
            )

        t_geom = (time.perf_counter() - t_geom_start) * 1000.0

        collector.record_frame(face_detected, iris_detected)

        # 4. Gaze State Classification Latency
        t_class_start = time.perf_counter()
        if features:
            gaze_state = gaze_engine.process_features(features)
        else:
            gaze_state = GazeState(
                timestamp_ms=timestamp_ms,
                direction=Direction.NO_FACE,
                raw_h=0.0,
                raw_v=0.0,
                ear=0.0,
                confidence=0.0,
                status_flag="✕ NO_FACE",
            )
        t_class = (time.perf_counter() - t_class_start) * 1000.0

        t_e2e = (time.perf_counter() - t_loop_start) * 1000.0
        profiler.record_frame(t_cap, t_mp, t_geom, t_class, t_e2e)

        # 5. Physical Diagnostic Sequence Progression
        if in_diagnostic and features and r_iris and r_inner and r_outer and r_upper and r_lower:
            curr_target = diag_targets[diag_target_idx]
            v_can_val = norm_vertical_gaze_canthal_normal_experimental(r_iris, r_inner, r_outer, is_right_eye=True)
            v_can_px = compute_canthal_normal_distance_px(r_iris, r_inner, r_outer, is_right_eye=True)

            v_can_l_val = norm_vertical_gaze_canthal_normal_experimental(l_iris, l_inner, l_outer, is_right_eye=False) if (l_iris and l_inner and l_outer) else None
            v_can_l_px = compute_canthal_normal_distance_px(l_iris, l_inner, l_outer, is_right_eye=False) if (l_iris and l_inner and l_outer) else None

            frame_data = DiagnosticFrameData(
                timestamp_ms=timestamp_ms,
                h=features.right_eye.h_ratio,
                v=features.right_eye.v_ratio,
                ear=features.right_eye.ear,
                iris_x=r_iris.x,
                iris_y=r_iris.y,
                inner_x=r_inner.x,
                inner_y=r_inner.y,
                outer_x=r_outer.x,
                outer_y=r_outer.y,
                upper_x=r_upper.x,
                upper_y=r_upper.y,
                lower_x=r_lower.x,
                lower_y=r_lower.y,
                is_valid=features.is_tracking_valid,
                v_eyelid=features.right_eye.v_ratio,
                v_canthal=v_can_val,
                v_canthal_px=v_can_px,
                # Left eye:
                h_left=features.left_eye.h_ratio if features.left_eye else None,
                v_eyelid_left=features.left_eye.v_ratio if features.left_eye else None,
                v_canthal_left=v_can_l_val,
                v_canthal_left_px=v_can_l_px,
                ear_left=features.left_eye.ear if features.left_eye else None,
                iris_l_x=l_iris.x if l_iris else None,
                iris_l_y=l_iris.y if l_iris else None,
                inner_l_x=l_inner.x if l_inner else None,
                inner_l_y=l_inner.y if l_inner else None,
                outer_l_x=l_outer.x if l_outer else None,
                outer_l_y=l_outer.y if l_outer else None,
                upper_l_x=l_upper.x if l_upper else None,
                upper_l_y=l_upper.y if l_upper else None,
                lower_l_x=l_lower.x if l_lower else None,
                lower_l_y=l_lower.y if l_lower else None,
            )
            buf = diag_buffers[curr_target]
            buf.add_frame(frame_data)
            if buf.is_complete(timestamp_ms):
                diag_target_idx += 1
                if diag_target_idx >= len(diag_targets):
                    in_diagnostic = False
                    try:
                        diag_stats = {t: b.compute_stats() for t, b in diag_buffers.items()}
                        ordering = verify_physical_ordering(diag_stats)
                        deltas = compute_raw_coordinate_deltas(diag_stats)
                        summary_str = format_diagnostic_summary(diag_stats, ordering, deltas)
                        print(summary_str)

                        out_file = export_path / "physical_geometry_diagnostic.json"
                        cam_meta = {
                            "camera_index": args.camera,
                            "resolution": (w_px, h_px),
                            "fps": round(fps, 1),
                            "backend": backend_name,
                        }
                        export_diagnostic_artifact(out_file, diag_stats, ordering, deltas, cam_meta)
                        print(f"[INFO] Physical geometry diagnostic report exported to: {out_file}\n")
                        if args.diagnostic:
                            break
                    except Exception as diag_err:
                        print(f"[WARN] Diagnostic processing failed: {diag_err}")

        # 6. Calibration State Progression
        if in_calibration and calib_service and features:
            curr_target = calib_targets[calib_target_idx]
            calib_service.feed_sample(curr_target, features)
            buf = calib_service.buffers[curr_target]
            if buf.is_complete(timestamp_ms):
                calib_target_idx += 1
                if calib_target_idx >= len(calib_targets):
                    in_calibration = False
                    try:
                        new_profile = calib_service.compile_profile()
                        gaze_engine.set_profile(new_profile)
                        save_calibration_profile(new_profile, profile_path)

                        stats = new_profile.stats
                        diagnostics = calib_service.get_calibration_diagnostics(stats)

                        print("\n========================================")
                        print("  CALIBRATION REPORT (RIGHT_EYE_BASELINE)")
                        print("========================================")
                        print(f"  {'TARGET':<10} {'SAMPLES':<8} {'MED_H':<8} {'MAD_H':<8} {'MED_V':<8} {'MAD_V':<8} {'EAR':<8} {'MAD_EAR':<8} {'SEP':<8}")
                        print("  " + "-" * 76)
                        for tgt in calib_targets:
                            d = diagnostics[tgt]
                            ear_str = f"{d['median_ear']:.3f}" if d['median_ear'] is not None else "N/A"
                            mad_ear_str = f"{d['mad_ear']:.3f}" if d['mad_ear'] is not None else "N/A"
                            print(f"  {tgt:<10} {d['samples']:<8} {d['median_h']:<8.3f} {d['mad_h']:<8.3f} {d['median_v']:<8.3f} {d['mad_v']:<8.3f} {ear_str:<8} {mad_ear_str:<8} {d['separation_from_center']:<+8.3f}")
                        print("========================================")
                        print(f"[INFO] Calibration Profile saved to {profile_path}!\n")

                        export_path.mkdir(parents=True, exist_ok=True)
                        raw_calib_report = {
                            "provenance": "REAL_HARDWARE_WEBCAM",
                            "profile_version": new_profile.profile_version,
                            "algorithm_version": new_profile.algorithm_version,
                            "active_eye": "RIGHT_EYE_BASELINE",
                            "user_id": new_profile.user_id,
                            "camera_index": new_profile.camera_index,
                            "camera_resolution": new_profile.camera_resolution,
                            "stats": {k: v.model_dump() for k, v in new_profile.stats.items()},
                            "raw_diagnostics": diagnostics,
                            "thresholds": new_profile.thresholds.model_dump(),
                            "quality_status": "PASSED",
                        }
                        with open(export_path / "calibration_report.json", "w", encoding="utf-8") as f:
                            json.dump(raw_calib_report, f, indent=2)

                    except Exception as err:
                        print(f"\n[WARN] Calibration quality check rejected profile: {err}")

        # 7. Guided Evaluation Trial Progression
        if in_evaluation:
            curr_exp_dir = eval_targets[eval_target_idx]
            eval_collected_states.append(gaze_state.direction)
            elapsed_trial_sec = (timestamp_ms - eval_step_start_ms) / 1000.0

            if elapsed_trial_sec >= 2.0:
                mid_samples = eval_collected_states[len(eval_collected_states)//3:]
                classified_dir = max(set(mid_samples), key=mid_samples.count).value if mid_samples else Direction.ABSTAIN.value
                collector.record_trial(curr_exp_dir.value, classified_dir)
                eval_trial_count += 1
                eval_collected_states = []
                eval_step_start_ms = timestamp_ms

                if eval_trial_count >= max_trials_per_target:
                    eval_trial_count = 0
                    eval_target_idx += 1
                    if eval_target_idx >= len(eval_targets):
                        in_evaluation = False
                        print("\n========================================")
                        print("  EVALUATION TRIALS COMPLETE")
                        print(f"  Event-Level Accuracy: {collector.get_event_accuracy() * 100:.1f}%")
                        print("========================================\n")

        # 8. Passive False-Activation Test Progression
        if in_false_act_test:
            elapsed_fa_sec = (timestamp_ms - false_act_start_ms) / 1000.0
            if gaze_state.direction not in (Direction.CENTER, Direction.BLINK):
                false_act_triggers += 1
            if elapsed_fa_sec >= false_act_duration_sec:
                in_false_act_test = False
                collector.false_activations = false_act_triggers
                collector.passive_test_duration_sec = false_act_duration_sec
                print(f"\n[INFO] False-Activation Test Complete: {false_act_triggers} triggers in {false_act_duration_sec:.0f}s")

        fps = 1.0 / (time.perf_counter() - prev_loop_time + 1e-6)
        prev_loop_time = time.perf_counter()

        # 9. Create Mirrored Display Frame for User Preview Only
        # Mirroring only affects the preview screen, strictly decoupled from anatomical landmarks
        display_frame = cv2.flip(raw_frame, 1)

        # 10. Draw Magnified PIP Eye Crop in Corner
        if r_inner and r_outer and r_upper and r_lower:
            draw_eye_pip_magnifier(display_frame, raw_frame, r_inner, r_outer, r_upper, r_lower)

        # 11. Draw Targets (Calibration, Diagnostic, or Evaluation)
        if in_diagnostic:
            curr_target = diag_targets[diag_target_idx]
            buf = diag_buffers[curr_target]
            draw_high_contrast_target(
                display_frame,
                curr_target,
                w_px,
                h_px,
                collected_samples=len(buf.samples),
                required_samples=50,
                mode_label="DIAGNOSTIC",
            )
        elif in_calibration and calib_service:
            curr_target = calib_targets[calib_target_idx]
            buf = calib_service.buffers[curr_target]
            draw_high_contrast_target(
                display_frame,
                curr_target,
                w_px,
                h_px,
                collected_samples=len(buf.h_samples),
                required_samples=45,
                mode_label="CALIBRATION",
            )
        elif in_evaluation:
            curr_target = eval_targets[eval_target_idx].value
            draw_high_contrast_target(
                display_frame,
                curr_target,
                w_px,
                h_px,
                collected_samples=eval_trial_count + 1,
                required_samples=max_trials_per_target,
                mode_label="EVALUATION",
            )
        elif in_false_act_test:
            cv2.putText(
                display_frame,
                "*** PASSIVE TEST: READ / SPEAK CASUALLY ***",
                (w_px // 2 - 280, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.80,
                (0, 165, 255),
                2,
            )

        # 12. Multi-Modal Diagnostic HUD
        draw_diagnostic_hud(
            display_frame,
            gaze_state,
            fps,
            t_mp,
            t_e2e,
            r_iris=r_iris,
            r_inner=r_inner,
            r_outer=r_outer,
            r_upper=r_upper,
            r_lower=r_lower,
            active_eye_name="RIGHT_EYE_BASELINE",
            calib_service=calib_service,
            in_calibration=in_calibration,
            curr_calib_target=calib_targets[calib_target_idx] if in_calibration else "",
            in_diagnostic=in_diagnostic,
            curr_diag_target=diag_targets[diag_target_idx] if in_diagnostic else "",
            diag_samples_count=len(diag_buffers[diag_targets[diag_target_idx]].samples) if (in_diagnostic and diag_targets[diag_target_idx] in diag_buffers) else 0,
        )

        cv2.imshow("VoxGaze - Milestone M1 Live Validation Harness", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key in (ord('d'), ord('D')) and not in_diagnostic and not in_calibration and not in_evaluation:
            print("\n" + "=" * 80)
            print("  [INFO] Starting Physical Geometry Diagnostic Sequence...")
            print("  [CONTROLLED TRIAL REQUIREMENT]")
            print("    1. Keep head strictly stationary and rigid against a headrest / chin-stop.")
            print("    2. Rotate ONLY your eyes toward the target dot (do not turn or pitch head).")
            print("    3. Maintain calm fixation for the full 2 seconds per target.")
            print("=" * 80 + "\n")
            diag_buffers = {t: DiagnosticTargetBuffer(t, warmup_sec=0.5, collection_sec=2.0) for t in diag_targets}
            diag_target_idx = 0
            in_diagnostic = True
        elif key in (ord('c'), ord('C')) and not in_diagnostic and not in_calibration and not in_evaluation:
            print("\n[INFO] Starting 5-Point + Blink Calibration (RIGHT_EYE_BASELINE)...")
            calib_service = CalibrationService(
                user_id="user_01",
                camera_index=args.camera,
                camera_resolution=(w_px, h_px),
                active_eye="right",
            )
            calib_target_idx = 0
            in_calibration = True
        elif key in (ord('e'), ord('E')) and not in_diagnostic and not in_calibration and not in_evaluation:
            if gaze_engine.profile is None:
                print("\n[WARN] Please calibrate first before running evaluation trials!")
            else:
                print("\n[INFO] Starting Guided Evaluation Trials (20 per direction)...")
                in_evaluation = True
                eval_target_idx = 0
                eval_trial_count = 0
                eval_step_start_ms = int(time.perf_counter() * 1000)
                eval_collected_states = []
        elif key in (ord('f'), ord('F')) and not in_diagnostic and not in_calibration and not in_false_act_test:
            print("\n[INFO] Starting 60s Passive False-Activation Test...")
            in_false_act_test = True
            false_act_start_ms = int(time.perf_counter() * 1000)
            false_act_triggers = 0
        elif key in (ord('s'), ord('S')):
            print("[INFO] Saving live hardware benchmark artifacts...")
            rates = collector.get_tracking_rates()
            lat_summary = profiler.get_summary()
            cam_rep = {
                "provenance": "REAL_HARDWARE_WEBCAM",
                "hardware_test_completed": True,
                "camera_index": args.camera,
                "requested_width": 1280,
                "requested_height": 720,
                "actual_width": w_px,
                "actual_height": h_px,
                "observed_fps": round(fps, 1),
                "backend": backend_name,
                "face_tracking_rate": rates["face_tracking_rate"],
                "iris_tracking_rate": rates["iris_tracking_rate"],
            }
            export_path.mkdir(parents=True, exist_ok=True)
            with open(export_path / "camera_report.json", "w", encoding="utf-8") as f:
                json.dump(cam_rep, f, indent=2)
            with open(export_path / "latency_summary.json", "w", encoding="utf-8") as f:
                json.dump(lat_summary, f, indent=2)
            collector.export_confusion_matrix_csv(export_path / "direction_confusion_matrix.csv")
            print(f"[INFO] Real hardware artifacts exported to {export_path}/")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
