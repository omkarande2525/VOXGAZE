"""
VoxGaze Milestone M1 / MobileOne S0 Integration
Isolated Real-Time Live-Webcam Diagnostic & Structured Observation Tool

Runs the pretrained MobileOne S0 ONNX gaze estimator concurrently with the existing
pure MediaPipe geometry baseline to evaluate comparative signal behavior on real video.
Supports an optional structured 20-condition measurement mode (--measure).

RESEARCH & SAFETY INVARIANTS:
1. Diagnostic Observation Only: Zero connection to AAC command FSM, preview, or Piper TTS.
2. Independent Signals: Geometry ratios and MobileOne angles are kept strictly separate.
3. Privacy First: No raw webcam frames, face crops, or videos are saved to disk.
4. Clean Decoupling: CPUExecutionProvider only, graceful exit on 'q', ESC, or window closure.
5. Non-Destructive: Never silently overwrites measurement files; generates unique session files.
"""

import argparse
from collections import deque
import math
from pathlib import Path
import sys
import time
from typing import Any, Deque, Dict, Optional, Tuple

import cv2
import numpy as np

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from backend.app.core.contracts import Point2D
from backend.app.perception.face_crop import extract_face_crop, FaceCropResult
from backend.app.perception.geometry import (
    LEFT_INNER_CANTHUS,
    LEFT_IRIS_ALL,
    LEFT_LOWER_LID,
    LEFT_OUTER_CANTHUS,
    LEFT_UPPER_LID,
    RIGHT_INNER_CANTHUS,
    RIGHT_IRIS_ALL,
    RIGHT_LOWER_LID,
    RIGHT_OUTER_CANTHUS,
    RIGHT_UPPER_LID,
    compute_canthal_normal_distance_px,
    compute_ear,
    extract_eye_features,
    norm_vertical_gaze_canthal_normal_experimental,
    validate_eye_geometry,
    validate_bilateral_geometry,
)
from backend.app.perception.mobileone_estimator import (
    MobileOneGazeEstimator,
    MobileOneOutput,
)
from backend.app.perception.structured_observation import (
    OBSERVATION_CONDITIONS,
    ObservationSession,
    ObservationSessionV2,
    V2_FIXATION_TARGETS,
    V2_PHASE_RECORDING,
    V2_PHASE_SETTLING,
    V2_PHASE_WAITING,
)

DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "mobileone_s0_gaze.onnx"
DEFAULT_LANDMARKER_PATH = ROOT_DIR / "models" / "face_landmarker.task"
DEFAULT_OBSERVATIONS_DIR = ROOT_DIR / "artifacts" / "observations"


class DiagnosticTimer:
    """Rolling latency tracker for pipeline breakdown reporting."""

    def __init__(self, window_size: int = 30):
        self.t_cap: Deque[float] = deque(maxlen=window_size)
        self.t_mp: Deque[float] = deque(maxlen=window_size)
        self.t_crop: Deque[float] = deque(maxlen=window_size)
        self.t_model: Deque[float] = deque(maxlen=window_size)
        self.t_total: Deque[float] = deque(maxlen=window_size)

    def record(self, cap: float, mp_ms: float, crop_ms: float, model_ms: float, total_ms: float):
        self.t_cap.append(cap)
        self.t_mp.append(mp_ms)
        self.t_crop.append(crop_ms)
        self.t_model.append(model_ms)
        self.t_total.append(total_ms)

    def get_means(self) -> Dict[str, float]:
        def _m(d: Deque[float]) -> float:
            return float(np.mean(d)) if d else 0.0

        return {
            "cap": _m(self.t_cap),
            "mp": _m(self.t_mp),
            "crop": _m(self.t_crop),
            "model": _m(self.t_model),
            "total": _m(self.t_total),
        }


def draw_hud(
    frame: np.ndarray,
    face_detected: bool,
    crop_res: Optional[FaceCropResult],
    r_feat: Optional[Any],
    l_feat: Optional[Any],
    canthal_dist_px: float,
    model_output: Optional[MobileOneOutput],
    timer: DiagnosticTimer,
    obs_session: Optional[ObservationSession] = None,
) -> np.ndarray:
    """
    Renders high-contrast side-by-side diagnostic telemetry over the frame.
    Optionally overlays structured observation condition prompts and recording state.
    """
    canvas = frame.copy()
    h_px, w_px = canvas.shape[:2]

    # Semi-transparent dark backing panels for readability
    # Top header panel
    if obs_session is not None and obs_session.is_recording_active:
        cv2.rectangle(canvas, (0, 0), (w_px, 44), (20, 20, 150), -1)
        header_text = f"VOXGAZE MEASUREMENT [RECORDING ACTIVE] -- COND {obs_session.current_condition.index}/20: {obs_session.current_condition.name}"
        header_col = (255, 255, 255)
    elif obs_session is not None:
        cv2.rectangle(canvas, (0, 0), (w_px, 44), (25, 25, 25), -1)
        header_text = f"VOXGAZE MEASUREMENT [SETTLING / PAUSED] -- COND {obs_session.current_condition.index}/20: {obs_session.current_condition.name}"
        header_col = (0, 220, 255)
    else:
        cv2.rectangle(canvas, (0, 0), (w_px, 44), (20, 20, 20), -1)
        header_text = "VOXGAZE AUXILIARY DIAGNOSTIC -- [RESEARCH PROTOTYPE - NOT AN INTENT CLASSIFIER]"
        header_col = (0, 220, 255)

    cv2.line(canvas, (0, 44), (w_px, 44), (80, 80, 80), 1)
    cv2.putText(canvas, header_text, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.54, header_col, 2)

    # 1. Left Panel: MediaPipe Geometry Baseline
    panel_w = 410
    cv2.rectangle(canvas, (10, 52), (10 + panel_w, 230), (20, 20, 20), -1)
    cv2.rectangle(canvas, (10, 52), (10 + panel_w, 230), (180, 100, 0), 2)

    cv2.putText(canvas, "1. GEOMETRY BASELINE (MediaPipe)", (20, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 200, 0), 2)
    mp_status_str = "TRACKING ACTIVE" if face_detected else "NO FACE DETECTED"
    mp_status_col = (0, 255, 0) if face_detected else (0, 0, 255)
    cv2.putText(canvas, f"Status: {mp_status_str}", (20, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.48, mp_status_col, 1)

    if face_detected and r_feat is not None and l_feat is not None:
        cv2.putText(canvas, "Right Eye (Active M1):", (20, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)
        cv2.putText(canvas, f"  H={r_feat.h_ratio:.3f} | V={r_feat.v_ratio:.3f} | EAR={r_feat.ear:.3f}", (20, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 1)

        cv2.putText(canvas, "Left Eye (Bilateral):", (20, 166), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)
        cv2.putText(canvas, f"  H={l_feat.h_ratio:.3f} | V={l_feat.v_ratio:.3f} | EAR={l_feat.ear:.3f}", (20, 186), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 1)

        cv2.putText(canvas, f"Canthal Normal: {canthal_dist_px:+.2f} px", (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1)
    else:
        cv2.putText(canvas, "Awaiting landmark acquisition...", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 150, 150), 1)

    # 2. Right Panel: MobileOne S0 Gaze Estimator
    rx1 = w_px - panel_w - 10
    rx2 = w_px - 10
    cv2.rectangle(canvas, (rx1, 52), (rx2, 230), (20, 20, 20), -1)
    cv2.rectangle(canvas, (rx1, 52), (rx2, 230), (0, 165, 255), 2)

    cv2.putText(canvas, "2. MOBILEONE S0 ESTIMATOR (ONNX)", (rx1 + 10, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (0, 200, 255), 2)

    if model_output and model_output.is_valid:
        model_status_str = "INFERENCE OK"
        model_status_col = (0, 255, 0)
    elif not face_detected:
        model_status_str = "ABSTAIN (NO FACE)"
        model_status_col = (0, 0, 255)
    elif crop_res and not crop_res.is_valid:
        rejection = crop_res.rejection_reason or "Unknown"
        model_status_str = f"INVALID CROP ({rejection[:18]}...)"
        model_status_col = (0, 165, 255)
    elif model_output and not model_output.is_valid:
        err = model_output.error_message or "Unknown"
        model_status_str = f"ERROR: {err[:20]}"
        model_status_col = (0, 0, 255)
    else:
        model_status_str = "STANDBY"
        model_status_col = (150, 150, 150)

    cv2.putText(canvas, f"Status: {model_status_str}", (rx1 + 10, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.48, model_status_col, 1)

    if model_output and model_output.is_valid:
        cv2.putText(canvas, f"Yaw (H)  : {model_output.yaw_deg:+.2f} deg ({model_output.yaw_rad:+.4f} rad)", (rx1 + 10, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        cv2.putText(canvas, f"Pitch (V): {model_output.pitch_deg:+.2f} deg ({model_output.pitch_rad:+.4f} rad)", (rx1 + 10, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        cv2.putText(canvas, f"Preprocessing : {model_output.preprocessing_latency_ms:.1f} ms", (rx1 + 10, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1)
        cv2.putText(canvas, f"CPU Inference : {model_output.inference_latency_ms:.1f} ms", (rx1 + 10, 202), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 0), 1)
    else:
        cv2.putText(canvas, "No valid model inference.", (rx1 + 10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 150, 150), 1)

    # 3. Center Guidance Card (Only when structured measurement mode is active)
    if obs_session is not None:
        cx1 = 10 + panel_w + 10
        cx2 = rx1 - 10
        if cx2 > cx1 + 150:
            border_col = (0, 0, 255) if obs_session.is_recording_active else (0, 200, 255)
            cv2.rectangle(canvas, (cx1, 52), (cx2, 230), (25, 25, 25), -1)
            cv2.rectangle(canvas, (cx1, 52), (cx2, 230), border_col, 2)

            cond = obs_session.current_condition
            cv2.putText(canvas, f"CONDITION [{cond.index}/20]", (cx1 + 10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 220, 255), 1)
            cv2.putText(canvas, cond.name[:24], (cx1 + 10, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)

            # Split instruction to 2 lines if needed
            inst = cond.instruction
            words = inst.split()
            mid = len(words) // 2
            line1 = " ".join(words[:mid]) if len(inst) > 34 else inst
            line2 = " ".join(words[mid:]) if len(inst) > 34 else ""

            cv2.putText(canvas, line1, (cx1 + 10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1)
            if line2:
                cv2.putText(canvas, line2, (cx1 + 10, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1)

            # Status pill
            if obs_session.is_recording_active:
                cv2.rectangle(canvas, (cx1 + 10, 146), (cx2 - 10, 174), (0, 0, 180), -1)
                rec_count = len(obs_session.records)
                cv2.putText(canvas, f"* RECORDING ACTIVE ({rec_count} f)", (cx1 + 18, 166), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 2)
                cv2.putText(canvas, "[SPACE]/[N]: Finish Condition", (cx1 + 10, 194), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1)
                cv2.putText(canvas, "[S]: Skip | [A]: Abort | [Q]: Exit", (cx1 + 10, 212), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1)
            else:
                cv2.rectangle(canvas, (cx1 + 10, 146), (cx2 - 10, 174), (120, 80, 0), -1)
                cv2.putText(canvas, "[READY / SETTLING - PAUSED]", (cx1 + 18, 166), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
                cv2.putText(canvas, "[SPACE]: START Recording", (cx1 + 10, 194), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1)
                cv2.putText(canvas, "[S]: Skip | [A]: Abort | [Q]: Exit", (cx1 + 10, 212), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1)

    # 4. Bottom Timing & Telemetry Panel
    means = timer.get_means()
    fps = 1000.0 / means["total"] if means["total"] > 0 else 0.0
    cv2.rectangle(canvas, (10, h_px - 44), (w_px - 210, h_px - 10), (20, 20, 20), -1)
    cv2.rectangle(canvas, (10, h_px - 44), (w_px - 210, h_px - 10), (80, 80, 80), 1)

    timing_text = (
        f"FPS: {fps:.1f} | MP: {means['mp']:.1f}ms | Crop: {means['crop']:.1f}ms | "
        f"Model: {means['model']:.1f}ms | Total Frame: {means['total']:.1f}ms"
    )
    cv2.putText(canvas, timing_text, (18, h_px - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1)

    # 5. Draw Face Bounding Box & Gaze Vector on Main Canvas
    if crop_res and crop_res.is_valid and crop_res.bbox_clamped:
        bx1, by1, bx2, by2 = crop_res.bbox_clamped
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 200, 255), 2)
        cv2.putText(canvas, "MobileOne Face Crop (20% pad)", (bx1, max(18, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1)

        # Draw 3D gaze vector arrow from face center if prediction is available
        if model_output and model_output.is_valid:
            fc_x = (bx1 + bx2) // 2
            fc_y = (by1 + by2) // 2
            arrow_len = 70.0
            g_dx = int(arrow_len * math.sin(model_output.yaw_rad) * math.cos(model_output.pitch_rad))
            g_dy = int(-arrow_len * math.sin(model_output.pitch_rad))
            cv2.arrowedLine(canvas, (fc_x, fc_y), (fc_x + g_dx, fc_y + g_dy), (0, 0, 255), 3, tipLength=0.25)
            cv2.circle(canvas, (fc_x, fc_y), 4, (0, 255, 255), -1)

    # 6. Picture-in-Picture (PIP) Inset: The exact 448x448 crop supplied to the model
    pip_size = 180
    pip_x1 = w_px - pip_size - 10
    pip_y1 = h_px - pip_size - 10
    pip_x2 = pip_x1 + pip_size
    pip_y2 = pip_y1 + pip_size

    if crop_res and crop_res.is_valid and crop_res.crop_bgr is not None:
        try:
            pip_view = cv2.resize(crop_res.crop_bgr, (pip_size, pip_size))
            canvas[pip_y1:pip_y2, pip_x1:pip_x2] = pip_view
            cv2.rectangle(canvas, (pip_x1, pip_y1), (pip_x2, pip_y2), (0, 255, 255), 2)
            cv2.putText(canvas, "MODEL CROP (448x448)", (pip_x1 + 4, pip_y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
        except Exception:
            pass

    return canvas


def draw_fixation_target(
    canvas: np.ndarray,
    condition_name: str,
    phase: str,
) -> np.ndarray:
    """
    Draws a high-contrast visual fixation crosshair on the canvas for Run 2 protocol.
    Position is determined by the V2_FIXATION_TARGETS mapping.
    The crosshair pulses during RECORDING (alternating brightness).
    """
    target = V2_FIXATION_TARGETS.get(condition_name)
    if target is None:
        return canvas

    h_px, w_px = canvas.shape[:2]
    tx = int(target[0] * w_px)
    ty = int(target[1] * h_px)
    arm = 20  # Crosshair arm length in pixels

    if phase == V2_PHASE_RECORDING:
        # Bright green pulsing crosshair during recording
        color = (0, 255, 0)
        thickness = 3
    elif phase == V2_PHASE_SETTLING:
        # Yellow crosshair during settling
        color = (0, 220, 255)
        thickness = 2
    else:
        # Dim white crosshair when waiting
        color = (180, 180, 180)
        thickness = 1

    # Draw crosshair
    cv2.line(canvas, (tx - arm, ty), (tx + arm, ty), color, thickness)
    cv2.line(canvas, (tx, ty - arm), (tx, ty + arm), color, thickness)
    # Draw circle at center
    cv2.circle(canvas, (tx, ty), 6, color, thickness)

    return canvas


def draw_hud_v2(
    frame: np.ndarray,
    face_detected: bool,
    crop_res: Optional[FaceCropResult],
    r_feat: Optional[Any],
    l_feat: Optional[Any],
    canthal_dist_px: float,
    r_canthal_ratio: Optional[float],
    l_canthal_ratio: Optional[float],
    model_output: Optional[MobileOneOutput],
    timer: DiagnosticTimer,
    obs_v2: ObservationSessionV2,
) -> np.ndarray:
    """
    Renders the Run 2 protocol HUD with:
    - Auto-timed settling countdown and recording progress bars.
    - Bilateral canthal ratio display.
    - Visual fixation target crosshair.
    - Condition instructions and keyboard controls.
    """
    canvas = frame.copy()
    h_px, w_px = canvas.shape[:2]

    phase = obs_v2.current_phase
    cond = obs_v2.current_condition

    # Draw fixation target first (underneath HUD panels)
    draw_fixation_target(canvas, cond.name, phase)

    # Top header bar
    if phase == V2_PHASE_RECORDING:
        rec_remaining = obs_v2.get_recording_remaining_s()
        cv2.rectangle(canvas, (0, 0), (w_px, 48), (20, 20, 150), -1)
        header_text = f"VOXGAZE RUN 2 [RECORDING {rec_remaining:.1f}s] -- COND {cond.index}/20: {cond.name}"
        header_col = (255, 255, 255)
    elif phase == V2_PHASE_SETTLING:
        settle_remaining = obs_v2.get_settling_remaining_s()
        cv2.rectangle(canvas, (0, 0), (w_px, 48), (30, 100, 30), -1)
        header_text = f"VOXGAZE RUN 2 [SETTLING {settle_remaining:.1f}s - HOLD POSITION] -- COND {cond.index}/20: {cond.name}"
        header_col = (0, 255, 255)
    else:
        cv2.rectangle(canvas, (0, 0), (w_px, 48), (25, 25, 25), -1)
        header_text = f"VOXGAZE RUN 2 [READY] -- COND {cond.index}/20: {cond.name}"
        header_col = (0, 220, 255)

    cv2.line(canvas, (0, 48), (w_px, 48), (80, 80, 80), 1)
    cv2.putText(canvas, header_text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.56, header_col, 2)

    # 1. Left Panel: Geometry baseline with bilateral canthal ratios
    panel_w = 420
    cv2.rectangle(canvas, (10, 56), (10 + panel_w, 250), (20, 20, 20), -1)
    cv2.rectangle(canvas, (10, 56), (10 + panel_w, 250), (180, 100, 0), 2)
    cv2.putText(canvas, "1. GEOMETRY BASELINE (MediaPipe)", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 200, 0), 2)

    mp_status = "TRACKING ACTIVE" if face_detected else "NO FACE DETECTED"
    mp_col = (0, 255, 0) if face_detected else (0, 0, 255)
    cv2.putText(canvas, f"Status: {mp_status}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.44, mp_col, 1)

    if face_detected and r_feat is not None and l_feat is not None:
        cv2.putText(canvas, "Right Eye:", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        cv2.putText(canvas, f"  H={r_feat.h_ratio:.3f} V={r_feat.v_ratio:.3f} EAR={r_feat.ear:.3f}", (20, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        cv2.putText(canvas, "Left Eye:", (20, 156), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        cv2.putText(canvas, f"  H={l_feat.h_ratio:.3f} V={l_feat.v_ratio:.3f} EAR={l_feat.ear:.3f}", (20, 174), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        cv2.putText(canvas, f"Canthal Px: {canthal_dist_px:+.2f}", (20, 196), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        r_ratio_str = f"{r_canthal_ratio:+.4f}" if r_canthal_ratio is not None else "N/A"
        l_ratio_str = f"{l_canthal_ratio:+.4f}" if l_canthal_ratio is not None else "N/A"
        cv2.putText(canvas, f"R Canthal Ratio: {r_ratio_str}", (20, 216), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 200), 1)
        cv2.putText(canvas, f"L Canthal Ratio: {l_ratio_str}", (20, 236), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 200), 1)
    else:
        cv2.putText(canvas, "Awaiting landmarks...", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (150, 150, 150), 1)

    # 2. Right Panel: MobileOne
    rx1 = w_px - panel_w - 10
    rx2 = w_px - 10
    cv2.rectangle(canvas, (rx1, 56), (rx2, 250), (20, 20, 20), -1)
    cv2.rectangle(canvas, (rx1, 56), (rx2, 250), (0, 165, 255), 2)
    cv2.putText(canvas, "2. MOBILEONE S0 ESTIMATOR (ONNX)", (rx1 + 10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 200, 255), 2)

    if model_output and model_output.is_valid:
        cv2.putText(canvas, "Status: INFERENCE OK", (rx1 + 10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 0), 1)
        cv2.putText(canvas, f"Yaw (H)  : {model_output.yaw_deg:+.2f} deg", (rx1 + 10, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)
        cv2.putText(canvas, f"Pitch (V): {model_output.pitch_deg:+.2f} deg", (rx1 + 10, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)
        cv2.putText(canvas, f"Pre: {model_output.preprocessing_latency_ms:.1f}ms | Inf: {model_output.inference_latency_ms:.1f}ms", (rx1 + 10, 176), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)
    elif not face_detected:
        cv2.putText(canvas, "Status: ABSTAIN (NO FACE)", (rx1 + 10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 0, 255), 1)
    else:
        cv2.putText(canvas, "Status: STANDBY", (rx1 + 10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (150, 150, 150), 1)

    # 3. Center Guidance Card
    cx1 = 10 + panel_w + 10
    cx2 = rx1 - 10
    if cx2 > cx1 + 150:
        border_col = (0, 0, 255) if phase == V2_PHASE_RECORDING else ((0, 180, 0) if phase == V2_PHASE_SETTLING else (0, 200, 255))
        cv2.rectangle(canvas, (cx1, 56), (cx2, 250), (25, 25, 25), -1)
        cv2.rectangle(canvas, (cx1, 56), (cx2, 250), border_col, 2)

        cv2.putText(canvas, f"CONDITION [{cond.index}/20]", (cx1 + 10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 220, 255), 1)
        cv2.putText(canvas, cond.name[:24], (cx1 + 10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)

        # Instruction text (word-wrapped)
        inst = cond.instruction
        words = inst.split()
        mid = len(words) // 2
        line1 = " ".join(words[:mid]) if len(inst) > 34 else inst
        line2 = " ".join(words[mid:]) if len(inst) > 34 else ""
        cv2.putText(canvas, line1, (cx1 + 10, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (210, 210, 210), 1)
        if line2:
            cv2.putText(canvas, line2, (cx1 + 10, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (210, 210, 210), 1)

        # Phase status display
        if phase == V2_PHASE_RECORDING:
            rec_remaining = obs_v2.get_recording_remaining_s()
            rec_count = len(obs_v2.records)
            # Progress bar
            bar_w = cx2 - cx1 - 20
            progress = 1.0 - (rec_remaining / obs_v2.recording_duration_s)
            fill_w = int(bar_w * min(1.0, progress))
            cv2.rectangle(canvas, (cx1 + 10, 150), (cx1 + 10 + bar_w, 168), (40, 40, 40), -1)
            cv2.rectangle(canvas, (cx1 + 10, 150), (cx1 + 10 + fill_w, 168), (0, 0, 200), -1)
            cv2.rectangle(canvas, (cx1 + 10, 150), (cx1 + 10 + bar_w, 168), (100, 100, 100), 1)
            cv2.putText(canvas, f"REC {rec_remaining:.1f}s ({rec_count}f)", (cx1 + 14, 164), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
            cv2.putText(canvas, "Auto-stops after 5s", (cx1 + 10, 188), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 180, 180), 1)
        elif phase == V2_PHASE_SETTLING:
            settle_remaining = obs_v2.get_settling_remaining_s()
            bar_w = cx2 - cx1 - 20
            progress = 1.0 - (settle_remaining / obs_v2.settling_duration_s)
            fill_w = int(bar_w * min(1.0, progress))
            cv2.rectangle(canvas, (cx1 + 10, 150), (cx1 + 10 + bar_w, 168), (40, 40, 40), -1)
            cv2.rectangle(canvas, (cx1 + 10, 150), (cx1 + 10 + fill_w, 168), (0, 140, 0), -1)
            cv2.rectangle(canvas, (cx1 + 10, 150), (cx1 + 10 + bar_w, 168), (100, 100, 100), 1)
            cv2.putText(canvas, f"SETTLING {settle_remaining:.1f}s", (cx1 + 14, 164), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
            cv2.putText(canvas, "Hold position - NOT recording", (cx1 + 10, 188), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 200, 200), 1)
        else:
            cv2.rectangle(canvas, (cx1 + 10, 150), (cx2 - 10, 174), (120, 80, 0), -1)
            cv2.putText(canvas, "[READY - Press SPACE]", (cx1 + 18, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

        # Control hints
        cv2.putText(canvas, "[SPACE]: Start | [S]: Skip | [A]: Abort | [Q]: Exit", (cx1 + 10, 238), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (160, 160, 160), 1)

    # 4. Bottom Timing Panel
    means = timer.get_means()
    fps = 1000.0 / means["total"] if means["total"] > 0 else 0.0
    cv2.rectangle(canvas, (10, h_px - 44), (w_px - 210, h_px - 10), (20, 20, 20), -1)
    cv2.rectangle(canvas, (10, h_px - 44), (w_px - 210, h_px - 10), (80, 80, 80), 1)
    timing_text = (
        f"FPS: {fps:.1f} | MP: {means['mp']:.1f}ms | Crop: {means['crop']:.1f}ms | "
        f"Model: {means['model']:.1f}ms | Total: {means['total']:.1f}ms"
    )
    cv2.putText(canvas, timing_text, (18, h_px - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1)

    # 5. Face Bounding Box & Gaze Vector
    if crop_res and crop_res.is_valid and crop_res.bbox_clamped:
        bx1, by1, bx2, by2 = crop_res.bbox_clamped
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 200, 255), 2)
        if model_output and model_output.is_valid:
            fc_x = (bx1 + bx2) // 2
            fc_y = (by1 + by2) // 2
            arrow_len = 70.0
            g_dx = int(arrow_len * math.sin(model_output.yaw_rad) * math.cos(model_output.pitch_rad))
            g_dy = int(-arrow_len * math.sin(model_output.pitch_rad))
            cv2.arrowedLine(canvas, (fc_x, fc_y), (fc_x + g_dx, fc_y + g_dy), (0, 0, 255), 3, tipLength=0.25)

    # 6. PIP inset
    pip_size = 160
    pip_x1 = w_px - pip_size - 10
    pip_y1 = h_px - pip_size - 10
    pip_x2 = pip_x1 + pip_size
    pip_y2 = pip_y1 + pip_size
    if crop_res and crop_res.is_valid and crop_res.crop_bgr is not None:
        try:
            pip_view = cv2.resize(crop_res.crop_bgr, (pip_size, pip_size))
            canvas[pip_y1:pip_y2, pip_x1:pip_x2] = pip_view
            cv2.rectangle(canvas, (pip_x1, pip_y1), (pip_x2, pip_y2), (0, 255, 255), 2)
            cv2.putText(canvas, "CROP 448x448", (pip_x1 + 4, pip_y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 255, 255), 1)
        except Exception:
            pass

    return canvas


def run_live_diagnostic(
    camera_index: int = 0,
    width: int = 1280,
    height: int = 720,
    model_path: Path = DEFAULT_MODEL_PATH,
    landmarker_path: Path = DEFAULT_LANDMARKER_PATH,
    headless_frames: int = 0,
    measure: bool = False,
    measure_v2: bool = False,
    output_dir: Path = DEFAULT_OBSERVATIONS_DIR,
    output_file: Optional[Path] = None,
) -> bool:
    """
    Executes the live webcam diagnostic loop.

    Args:
        camera_index: Index of video capture device.
        width: Requested frame width.
        height: Requested frame height.
        model_path: Path to mobileone_s0_gaze.onnx.
        landmarker_path: Path to face_landmarker.task.
        headless_frames: If > 0, executes a synthetic non-GUI dry run of N frames for verification.
        measure: If True, enables Run 1 structured 20-condition observation measurement protocol.
        measure_v2: If True, enables Run 2 auto-timed protocol (2s settling + 5s recording).
        output_dir: Directory to save observation recordings.
        output_file: Optional explicit path for CSV log file.

    Returns:
        True if diagnostic finished cleanly, False otherwise.
    """
    model_path = Path(model_path).resolve()
    landmarker_path = Path(landmarker_path).resolve()

    # 1. Verify required model assets exist
    if not model_path.is_file():
        print(f"[FATAL] MobileOne model file not found: {model_path}")
        print("        Please run: python scripts/download_mobileone.py")
        return False

    if not landmarker_path.is_file():
        print(f"[FATAL] MediaPipe Face Landmarker model asset missing: {landmarker_path}")
        print("        Download required from official Google Cloud bucket:")
        print("        https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
        return False

    # 2. Initialize MobileOne ONNX estimator
    print(f"[INFO] Initializing MobileOne Gaze Estimator from: {model_path.name}")
    try:
        estimator = MobileOneGazeEstimator(model_path=model_path)
    except Exception as exc:
        print(f"[FATAL] Failed to initialize MobileOne ONNX Runtime session: {exc}")
        return False

    # 3. Initialize MediaPipe Face Landmarker in VIDEO mode
    print(f"[INFO] Initializing MediaPipe Face Landmarker from: {landmarker_path.name}")
    try:
        base_opts = python.BaseOptions(model_asset_path=str(landmarker_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_opts,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.6,
            min_face_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        detector = vision.FaceLandmarker.create_from_options(options)
    except Exception as exc:
        print(f"[FATAL] Failed to initialize MediaPipe Face Landmarker: {exc}")
        return False

    # 4. Initialize Observation Session if measurement mode requested
    obs_session: Optional[ObservationSession] = None
    obs_v2: Optional[ObservationSessionV2] = None

    if measure_v2:
        obs_v2 = ObservationSessionV2(
            output_dir=output_dir,
            output_file=output_file,
        )
        print("\n============================================================")
        print("   VOXGAZE RUN 2 -- AUTO-TIMED STRUCTURED MEASUREMENT       ")
        print("============================================================")
        print(f" Protocol     : v2 (2s settling + 5s recording per condition)")
        print(f" Session ID   : {obs_v2.session_id}")
        print(f" Output CSV   : {obs_v2.csv_path}")
        print(f" Summary      : {obs_v2.json_summary_path}")
        print(" Controls:")
        print("   [SPACE] : Start current condition (settling begins)")
        print("   [S]     : Skip current condition")
        print("   [A]     : Abort measurement session safely")
        print("   [Q]/ESC : Exit diagnostic")
        print(" Timing   : 2s settling (NOT recorded) + 5s recording (auto-stop)")
        print(" Privacy  : Numerical features only. No raw images or video saved.")
        print("============================================================\n")
    elif measure:
        obs_session = ObservationSession(
            output_dir=output_dir,
            output_file=output_file,
        )
        print("\n============================================================")
        print("      VOXGAZE STRUCTURED GAZE-OBSERVATION MEASUREMENT       ")
        print("============================================================")
        print(f" Session ID : {obs_session.session_id}")
        print(f" Output CSV : {obs_session.csv_path}")
        print(f" Summary    : {obs_session.json_summary_path}")
        print(" Controls:")
        print("   [SPACE] : Start / Finish data collection for current condition")
        print("   [N]     : Finish current condition and advance to next")
        print("   [S]     : Skip current condition")
        print("   [A]     : Abort measurement session safely")
        print("   [Q]/ESC : Exit diagnostic")
        print(" Privacy : Numerical features only. No raw images or video saved.")
        print("============================================================\n")

    # 5. Setup Video Capture (or headless synthetic generator)
    is_headless = headless_frames > 0
    cap = None

    if not is_headless:
        print(f"[INFO] Opening camera index {camera_index} ({width}x{height})...")
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"[FATAL] Unable to open camera index {camera_index}.")
            print("        Troubleshooting:")
            print("        1. Check that the webcam is connected and not in use by another application.")
            print("        2. Try specifying a different index: python scripts/mobileone_live_diagnostic.py --camera 1")
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        act_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        act_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] Camera active. Acquired stream: {act_w}x{act_h}")
        if not measure and not measure_v2:
            print("\n------------------------------------------------------------")
            print(" Controls: Press [Q] or [ESC] on the preview window to exit.")
            print(" Privacy : No raw frames or face crops are saved to disk.")
            print("------------------------------------------------------------\n")
    else:
        print(f"[INFO] Running in headless synthetic dry-run mode ({headless_frames} frames)...")
        if obs_session:
            obs_session.start_recording_current_condition()
        if obs_v2:
            obs_v2.start_condition()

    timer = DiagnosticTimer(window_size=30)
    frame_count = 0
    window_name = "VoxGaze -- MobileOne vs Geometry Diagnostic"

    try:
        while True:
            t_frame_start = time.perf_counter()

            # Read frame
            if not is_headless:
                t0_cap = time.perf_counter()
                ret, raw_frame = cap.read()
                t_cap = (time.perf_counter() - t0_cap) * 1000.0
                if not ret or raw_frame is None:
                    print("[WARN] Camera frame read dropped. Attempting to continue...")
                    time.sleep(0.01)
                    continue
            else:
                t_cap = 0.5
                raw_frame = np.full((height, width, 3), 128, dtype=np.uint8)
                cv2.circle(raw_frame, (width // 2, height // 2), 120, (200, 200, 200), -1)

            h_px, w_px = raw_frame.shape[:2]

            # MediaPipe Face Landmarker inference
            t0_mp = time.perf_counter()
            rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.perf_counter() * 1000)

            detection_result = detector.detect_for_video(mp_image, timestamp_ms)
            t_mp = (time.perf_counter() - t0_mp) * 1000.0

            face_detected = bool(detection_result.face_landmarks)
            r_feat = None
            l_feat = None
            r_quality = None
            l_quality = None
            bilateral_quality = None
            canthal_dist_px = 0.0
            r_canthal_ratio: Optional[float] = None
            l_canthal_ratio: Optional[float] = None
            crop_result: Optional[FaceCropResult] = None
            model_output: Optional[MobileOneOutput] = None
            t_crop = 0.0
            t_model = 0.0

            if face_detected:
                face = detection_result.face_landmarks[0]
                pts = np.array([[p.x * w_px, p.y * h_px] for p in face])

                # 1. Existing Geometry Baseline with Quality Audit
                try:
                    r_iris = Point2D(x=float(np.mean(pts[RIGHT_IRIS_ALL, 0])), y=float(np.mean(pts[RIGHT_IRIS_ALL, 1])))
                    r_inner = Point2D(x=float(pts[RIGHT_INNER_CANTHUS, 0]), y=float(pts[RIGHT_INNER_CANTHUS, 1]))
                    r_outer = Point2D(x=float(pts[RIGHT_OUTER_CANTHUS, 0]), y=float(pts[RIGHT_OUTER_CANTHUS, 1]))
                    r_upper = Point2D(x=float(pts[RIGHT_UPPER_LID, 0]), y=float(pts[RIGHT_UPPER_LID, 1]))
                    r_lower = Point2D(x=float(pts[RIGHT_LOWER_LID, 0]), y=float(pts[RIGHT_LOWER_LID, 1]))
                    r_feat = extract_eye_features(r_iris, r_inner, r_outer, r_upper, r_lower, is_right_eye=True)
                    r_quality = validate_eye_geometry(r_iris, r_inner, r_outer, r_upper, r_lower, is_right_eye=True)

                    l_iris = Point2D(x=float(np.mean(pts[LEFT_IRIS_ALL, 0])), y=float(np.mean(pts[LEFT_IRIS_ALL, 1])))
                    l_inner = Point2D(x=float(pts[LEFT_INNER_CANTHUS, 0]), y=float(pts[LEFT_INNER_CANTHUS, 1]))
                    l_outer = Point2D(x=float(pts[LEFT_OUTER_CANTHUS, 0]), y=float(pts[LEFT_OUTER_CANTHUS, 1]))
                    l_upper = Point2D(x=float(pts[LEFT_UPPER_LID, 0]), y=float(pts[LEFT_UPPER_LID, 1]))
                    l_lower = Point2D(x=float(pts[LEFT_LOWER_LID, 0]), y=float(pts[LEFT_LOWER_LID, 1]))
                    l_feat = extract_eye_features(l_iris, l_inner, l_outer, l_upper, l_lower, is_right_eye=False)
                    l_quality = validate_eye_geometry(l_iris, l_inner, l_outer, l_upper, l_lower, is_right_eye=False)

                    bilateral_quality = validate_bilateral_geometry(r_quality, l_quality, require_both_eyes=False)
                    canthal_dist_px = compute_canthal_normal_distance_px(r_iris, r_inner, r_outer, is_right_eye=True)
                except Exception as exc:
                    print(f"[WARN] Geometry feature calculation failed: {exc}")

                # 2. Bilateral normalized canthal ratios (Run 2 only, also displayed in V1)
                try:
                    if r_feat is not None:
                        r_canthal_ratio = norm_vertical_gaze_canthal_normal_experimental(
                            r_iris, r_inner, r_outer, is_right_eye=True
                        )
                    if l_feat is not None:
                        l_canthal_ratio = norm_vertical_gaze_canthal_normal_experimental(
                            l_iris, l_inner, l_outer, is_right_eye=False
                        )
                except Exception:
                    pass  # Non-critical: ratio computation failed

                # 3. Face Crop Extraction for MobileOne
                t0_crop = time.perf_counter()
                crop_result = extract_face_crop(raw_frame, face, padding_scale=1.20, min_size_px=64)
                t_crop = (time.perf_counter() - t0_crop) * 1000.0

                # 4. MobileOne Gaze Estimation (Separate from geometry)
                if crop_result.is_valid and crop_result.crop_bgr is not None:
                    model_output = estimator.predict(crop_result.crop_bgr)
                    t_model = model_output.inference_latency_ms + model_output.preprocessing_latency_ms

            t_total = (time.perf_counter() - t_frame_start) * 1000.0
            timer.record(t_cap, t_mp, t_crop, t_model, t_total)

            frame_count += 1

            # Determine geometry validity flag
            is_geom_valid = bool(bilateral_quality and bilateral_quality.is_valid)

            # Determine unavailable reason (shared by V1 and V2)
            unavailable_reason = None
            if not face_detected:
                unavailable_reason = "NO_FACE_DETECTED"
            elif crop_result and not crop_result.is_valid:
                unavailable_reason = crop_result.rejection_reason or "INVALID_FACE_CROP"
            elif model_output and not model_output.is_valid:
                unavailable_reason = model_output.error_message or "INFERENCE_FAILED"
            elif not is_geom_valid and bilateral_quality:
                unavailable_reason = "; ".join(bilateral_quality.bilateral_reasons) or "GEOMETRY_INVALID"

            # 5a. Record frame for Run 2 (V2) auto-timed protocol
            if obs_v2 is not None and not obs_v2.is_finished:
                obs_v2.tick()
                obs_v2.record_frame(
                    frame_idx=frame_count,
                    face_detected=face_detected,
                    face_crop_valid=bool(crop_result and crop_result.is_valid),
                    geometry_valid=is_geom_valid,
                    right_eye_h=r_feat.h_ratio if r_feat else None,
                    right_eye_v=r_feat.v_ratio if r_feat else None,
                    right_eye_ear=r_feat.ear if r_feat else None,
                    left_eye_h=l_feat.h_ratio if l_feat else None,
                    left_eye_v=l_feat.v_ratio if l_feat else None,
                    left_eye_ear=l_feat.ear if l_feat else None,
                    canthal_normal_dist_px=canthal_dist_px if r_feat else None,
                    right_eye_canthal_ratio=r_canthal_ratio,
                    left_eye_canthal_ratio=l_canthal_ratio,
                    mobileone_valid=bool(model_output and model_output.is_valid),
                    mobileone_yaw_deg=model_output.yaw_deg if (model_output and model_output.is_valid) else None,
                    mobileone_pitch_deg=model_output.pitch_deg if (model_output and model_output.is_valid) else None,
                    mediapipe_latency_ms=t_mp,
                    crop_latency_ms=t_crop,
                    mobileone_preprocess_ms=model_output.preprocessing_latency_ms if model_output else None,
                    mobileone_infer_ms=model_output.inference_latency_ms if model_output else None,
                    total_frame_latency_ms=t_total,
                    unavailable_reason=unavailable_reason,
                )
                # Check if V2 session auto-completed all conditions
                if obs_v2.is_finished:
                    print("[MEASURE-V2] All conditions completed!")
                    if not is_headless:
                        break

            # 5b. Record frame for Run 1 (V1) manual protocol
            if obs_session and obs_session.is_recording_active:
                obs_session.record_frame(
                    frame_idx=frame_count,
                    face_detected=face_detected,
                    face_crop_valid=bool(crop_result and crop_result.is_valid),
                    geometry_valid=is_geom_valid,
                    right_eye_h=r_feat.h_ratio if r_feat else None,
                    right_eye_v=r_feat.v_ratio if r_feat else None,
                    right_eye_ear=r_feat.ear if r_feat else None,
                    left_eye_h=l_feat.h_ratio if l_feat else None,
                    left_eye_v=l_feat.v_ratio if l_feat else None,
                    left_eye_ear=l_feat.ear if l_feat else None,
                    canthal_normal_dist_px=canthal_dist_px if r_feat else None,
                    mobileone_valid=bool(model_output and model_output.is_valid),
                    mobileone_yaw_deg=model_output.yaw_deg if (model_output and model_output.is_valid) else None,
                    mobileone_pitch_deg=model_output.pitch_deg if (model_output and model_output.is_valid) else None,
                    mediapipe_latency_ms=t_mp,
                    crop_latency_ms=t_crop,
                    mobileone_preprocess_ms=model_output.preprocessing_latency_ms if model_output else None,
                    mobileone_infer_ms=model_output.inference_latency_ms if model_output else None,
                    total_frame_latency_ms=t_total,
                    unavailable_reason=unavailable_reason,
                )

            if not is_headless:
                # Select appropriate HUD renderer
                if obs_v2 is not None:
                    hud_frame = draw_hud_v2(
                        raw_frame,
                        face_detected=face_detected,
                        crop_res=crop_result,
                        r_feat=r_feat,
                        l_feat=l_feat,
                        canthal_dist_px=canthal_dist_px,
                        r_canthal_ratio=r_canthal_ratio if face_detected else None,
                        l_canthal_ratio=l_canthal_ratio if face_detected else None,
                        model_output=model_output,
                        timer=timer,
                        obs_v2=obs_v2,
                    )
                else:
                    hud_frame = draw_hud(
                        raw_frame,
                        face_detected=face_detected,
                        crop_res=crop_result,
                        r_feat=r_feat,
                        l_feat=l_feat,
                        canthal_dist_px=canthal_dist_px,
                        model_output=model_output,
                        timer=timer,
                        obs_session=obs_session,
                    )

                cv2.imshow(window_name, hud_frame)
                key = cv2.waitKey(1) & 0xFF

                # Standard exit controls
                if key in (ord("q"), ord("Q"), 27):  # 'q' or ESC
                    print("[INFO] User requested diagnostic exit.")
                    if obs_session and not obs_session.is_finished:
                        print("[MEASURE] Session interrupted. Preserving collected data...")
                        obs_session.abort_session()
                    if obs_v2 and not obs_v2.is_finished:
                        print("[MEASURE-V2] Session interrupted. Preserving collected data...")
                        obs_v2.abort_session()
                    break

                # Run 2 (V2) controls
                if obs_v2 is not None and not obs_v2.is_finished:
                    if key == ord(" "):  # SPACE: Start condition (settling begins)
                        if obs_v2.current_phase == V2_PHASE_WAITING:
                            obs_v2.start_condition()
                            print(f"[MEASURE-V2] Started settling for condition [{obs_v2.current_condition.index}/20]: {obs_v2.current_condition.name}")
                    elif key in (ord("s"), ord("S")):  # S: Skip condition
                        more = obs_v2.skip_current_condition()
                        if more:
                            print(f"[MEASURE-V2] Condition skipped. Next: [{obs_v2.current_condition.index}/20]: {obs_v2.current_condition.name}")
                        else:
                            print("[MEASURE-V2] All conditions completed!")
                            break
                    elif key in (ord("a"), ord("A")):  # A: Abort session safely
                        print("[MEASURE-V2] User aborted session. Preserving recorded measurements...")
                        obs_v2.abort_session()
                        break

                # Run 1 (V1) controls
                elif obs_session is not None and not obs_session.is_finished:
                    if key == ord(" "):  # SPACE: Start or Finish condition
                        if not obs_session.is_recording_active:
                            obs_session.start_recording_current_condition()
                            print(f"[MEASURE] Started recording condition [{obs_session.current_condition.index}/20]: {obs_session.current_condition.name}")
                        else:
                            more = obs_session.finish_current_condition()
                            if more:
                                print(f"[MEASURE] Condition completed. Next: [{obs_session.current_condition.index}/20]: {obs_session.current_condition.name}")
                            else:
                                print("[MEASURE] All conditions completed!")
                                break
                    elif key in (ord("n"), ord("N")):  # N: Finish condition
                        more = obs_session.finish_current_condition()
                        if more:
                            print(f"[MEASURE] Condition completed. Next: [{obs_session.current_condition.index}/20]: {obs_session.current_condition.name}")
                        else:
                            print("[MEASURE] All conditions completed!")
                            break
                    elif key in (ord("s"), ord("S")):  # S: Skip condition
                        more = obs_session.skip_current_condition()
                        if more:
                            print(f"[MEASURE] Condition skipped. Next: [{obs_session.current_condition.index}/20]: {obs_session.current_condition.name}")
                        else:
                            print("[MEASURE] All conditions completed!")
                            break
                    elif key in (ord("a"), ord("A")):  # A: Abort session safely
                        print("[MEASURE] User aborted session. Preserving recorded measurements...")
                        obs_session.abort_session()
                        break

                # Check if window was closed via OS 'X' button
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("[INFO] Diagnostic window closed by user.")
                    if obs_session and not obs_session.is_finished:
                        obs_session.abort_session()
                    if obs_v2 and not obs_v2.is_finished:
                        obs_v2.abort_session()
                    break
            else:
                if frame_count >= headless_frames:
                    print(f"[INFO] Headless dry run completed ({frame_count} frames).")
                    if obs_session and not obs_session.is_finished:
                        obs_session.complete_session()
                    if obs_v2 and not obs_v2.is_finished:
                        obs_v2.complete_session()
                    break

        return True

    except KeyboardInterrupt:
        print("\n[INFO] Diagnostic interrupted via KeyboardInterrupt.")
        if obs_session and not obs_session.is_finished:
            obs_session.abort_session()
        if obs_v2 and not obs_v2.is_finished:
            obs_v2.abort_session()
        return True
    except Exception as exc:
        print(f"\n[FATAL] Unhandled diagnostic loop exception: {exc}")
        if obs_session and not obs_session.is_finished:
            obs_session.abort_session()
        if obs_v2 and not obs_v2.is_finished:
            obs_v2.abort_session()
        return False
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

        # Determine which session to report
        active_session_path = None
        active_summary_path = None
        if obs_v2 is not None:
            if not obs_v2.is_finished:
                obs_v2.abort_session()
            active_session_path = obs_v2.csv_path
            active_summary_path = obs_v2.json_summary_path
        elif obs_session is not None:
            if not obs_session.is_finished:
                obs_session.abort_session()
            active_session_path = obs_session.csv_path
            active_summary_path = obs_session.json_summary_path

        if active_summary_path is not None and active_summary_path.exists():
            import json
            try:
                with open(active_summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)

                proto_label = summary['metadata'].get('protocol_version', 'v1').upper()
                print(f"\n============================================================")
                print(f"       VOXGAZE {proto_label} MEASUREMENT SESSION SUMMARY       ")
                print(f"============================================================")
                print(f" Session ID       : {summary['metadata']['session_id']}")
                print(f" Status           : {summary['metadata']['session_status']}")
                print(f" Duration         : {summary['metadata']['elapsed_duration_s']} s")
                print(f" Frames Recorded  : {summary['metadata']['total_frames_recorded']}")
                print(f" CSV Output       : {active_session_path}")
                print(f" Summary JSON     : {active_summary_path}")
                if 'protocol_version' in summary['metadata']:
                    print(f" Protocol         : {summary['metadata']['protocol_version']}")
                    print(f" Settling Config  : {summary['metadata'].get('settling_duration_config_s', 'N/A')}s")
                    print(f" Recording Config : {summary['metadata'].get('recording_duration_config_s', 'N/A')}s")
                print("------------------------------------------------------------")
                co = summary["conditions_overview"]
                print(f" Conditions       : {co['completed_count']} completed, {co['skipped_count']} skipped, {co['interrupted_count']} interrupted")

                lat_m = summary["latency_metrics"]["mobileone_inference_ms"]
                lat_tot = summary["latency_metrics"]["total_frame_latency_ms"]
                print(f" Latency (Model)  : p50={lat_m['median_p50_ms']} ms | p95={lat_m['p95_ms']} ms (N={lat_m['sample_count']})")
                print(f" Latency (Frame)  : p50={lat_tot['median_p50_ms']} ms | p95={lat_tot['p95_ms']} ms (N={lat_tot['sample_count']})")

                cv_data = summary["center_observations_variability"]
                y_stat = cv_data["mobileone_yaw_deg"]["circular_stats"]
                p_stat = cv_data["mobileone_pitch_deg"]["circular_stats"]
                r_h_stat = cv_data["geometry_right_eye_h_ratio"]
                print(f" Center Dispersion: frames={cv_data['total_center_frames']}")
                print(f"   MobileOne Yaw  : circ_mean={y_stat['circular_mean_deg']} deg | circ_std={y_stat['circular_std_deg']} deg")
                print(f"   MobileOne Pitch: circ_mean={p_stat['circular_mean_deg']} deg | circ_std={p_stat['circular_std_deg']} deg")
                print(f"   Geometry R-Eye : median_h={r_h_stat['median']} | std_h={r_h_stat['std']}")
                print(" Privacy          : Zero raw frames or face crops saved to disk.")
                print(" Note             : Dispersion reflects signal stability, NOT accuracy.")
                print("============================================================\n")
            except Exception as exc:
                print(f"[WARN] Failed to print summary report: {exc}")

        print("[INFO] Clean diagnostic shutdown complete.")


def main():
    parser = argparse.ArgumentParser(
        description="VoxGaze MobileOne S0 vs Geometry Live Webcam Diagnostic & Measurement Harness."
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Requested frame width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Requested frame height (default: 720)")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help=f"MobileOne model path (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--landmarker-model", type=Path, default=DEFAULT_LANDMARKER_PATH, help=f"MediaPipe Face Landmarker path (default: {DEFAULT_LANDMARKER_PATH})")
    parser.add_argument("--headless", type=int, default=0, help="Run non-GUI dry run for N frames (useful for CI/automated testing)")
    parser.add_argument("--measure", action="store_true", default=False, help="Enable Run 1 structured 20-condition observation measurement mode.")
    parser.add_argument("--measure-v2", action="store_true", default=False, help="Enable Run 2 auto-timed protocol (2s settling + 5s recording per condition).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OBSERVATIONS_DIR, help=f"Output directory for measurements (default: {DEFAULT_OBSERVATIONS_DIR})")
    parser.add_argument("--output-file", type=Path, default=None, help="Optional explicit path for observation CSV log.")

    args = parser.parse_args()

    if args.measure and args.measure_v2:
        parser.error("--measure and --measure-v2 are mutually exclusive. Use one or the other.")

    success = run_live_diagnostic(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        model_path=args.model,
        landmarker_path=args.landmarker_model,
        headless_frames=args.headless,
        measure=args.measure,
        measure_v2=args.measure_v2,
        output_dir=args.output_dir,
        output_file=args.output_file,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
