"""
VoxGaze Milestone M1.5-B1: Physical Calibration Validation Harness

Runs automated 5-point calibration, independent evaluation trials, and head-contrast conditions.
Provides on-screen visual fixation targets, streams numerical features to CSV, and saves JSON audit.

SAFETY & PRIVACY INVARIANTS:
1. Numerical metrics only: Zero raw webcam images or face crops saved to disk.
2. Independent evaluation: Derives candidate thresholds from Phase 1, freezes them, and tests on Phase 2.
3. Unsupported upward gaze policy: Flags UP_GAZE_UNSUPPORTED if delta V < 0.05, never forcing a classifier.
4. MobileOne comparison: Runs MobileOne S0 as a head-movement diagnostic contrast only.
5. Zero AAC wiring: No live command execution or TTS integration.
"""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

from backend.app.calibration.evaluator import evaluate_calibration_validation_session
from backend.app.calibration.protocol import (
    CALIBRATION_VALIDATION_SEQUENCE,
    CalibrationValidationCondition,
    CalibrationValidationRecord,
    M15_CSV_HEADER,
    ProtocolPhase,
    TARGET_CENTER,
    TARGET_DOWN,
    TARGET_LEFT,
    TARGET_RIGHT,
    TARGET_UP,
)
from backend.app.perception.face_crop import extract_face_crop
from backend.app.core.contracts import Point2D
from backend.app.perception.geometry import (
    compute_canthal_normal_distance_px,
    extract_eye_features,
    norm_vertical_gaze_canthal_normal_experimental,
    validate_bilateral_geometry,
    validate_eye_geometry,
)
from backend.app.perception.mobileone_estimator import MobileOneGazeEstimator
from backend.app.perception.structured_observation import get_unique_output_path

# MediaPipe landmark indices
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


def emit_audio_cue(cue_type: str, is_headless: bool = False) -> None:
    """Emits an audible notification tone (Windows winsound or stdout bell)."""
    if is_headless:
        return
    try:
        import winsound
        if cue_type == "PREPARE_CLOSE":
            winsound.Beep(880, 150)
            time.sleep(0.04)
            winsound.Beep(1000, 150)
        elif cue_type == "CLOSE_NOW":
            winsound.Beep(1200, 400)
        elif cue_type == "EYES_OPEN":
            winsound.Beep(1200, 150)
            time.sleep(0.04)
            winsound.Beep(800, 250)
        elif cue_type == "STEP_ADVANCE":
            winsound.Beep(700, 100)
    except Exception:
        # Fallback to terminal bell
        try:
            sys.stdout.write("\a")
            sys.stdout.flush()
        except Exception:
            pass


class CalibrationValidationHarness:
    """Manages protocol state, automated timing, CSV logging, and HUD rendering."""

    def __init__(
        self,
        output_dir: Path,
        settling_duration_s: float = 2.0,
        recording_duration_s: float = 4.0,
        active_eye: str = "right",
        is_headless: bool = False,
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = f"calib_v15_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1000000:06x}"
        self.csv_path = get_unique_output_path(self.output_dir, prefix=f"calib_v15_{self.session_id[-6:]}", ext="csv")
        self.json_summary_path = self.csv_path.with_name(self.csv_path.stem + "_summary.json")

        self.settling_duration_s = settling_duration_s
        self.recording_duration_s = recording_duration_s
        self.active_eye = active_eye
        self.is_headless = is_headless

        self.conditions = CALIBRATION_VALIDATION_SEQUENCE
        self.current_cond_idx = 0  # 0-indexed into self.conditions
        self.phase_state = "SETTLING"  # "SETTLING", "RECORDING", "DONE"
        self.phase_start_mono = time.monotonic()
        self.session_start_mono = time.monotonic()

        self.records: List[CalibrationValidationRecord] = []
        self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        import csv
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(M15_CSV_HEADER)
        self._csv_file.flush()

        self.is_finished = False
        self.frame_count = 0

    @property
    def current_condition(self) -> CalibrationValidationCondition:
        return self.conditions[self.current_cond_idx]

    def tick(self) -> None:
        """Progresses phase timer and advances conditions when recording completes."""
        if self.is_finished:
            return

        now = time.monotonic()
        elapsed_in_phase = now - self.phase_start_mono

        if self.phase_state == "SETTLING":
            if elapsed_in_phase >= self.settling_duration_s:
                self.phase_state = "RECORDING"
                self.phase_start_mono = now
                if self.current_condition.target_direction == "CLOSED":
                    emit_audio_cue("CLOSE_NOW", is_headless=self.is_headless)
                    print("[AUDIO CUE] >> CLOSE BOTH EYES NOW <<")

        elif self.phase_state == "RECORDING":
            if elapsed_in_phase >= self.recording_duration_s:
                was_closed = (self.current_condition.target_direction == "CLOSED")
                # Advance to next condition
                if self.current_cond_idx + 1 < len(self.conditions):
                    self.current_cond_idx += 1
                    self.phase_state = "SETTLING"
                    self.phase_start_mono = now
                    if was_closed:
                        emit_audio_cue("EYES_OPEN", is_headless=self.is_headless)
                        print("[AUDIO CUE] >> EYES OPEN - ADVANCING TO NEXT TARGET <<")
                    elif self.current_condition.target_direction == "CLOSED":
                        emit_audio_cue("PREPARE_CLOSE", is_headless=self.is_headless)
                        print("[AUDIO CUE] >> PREPARE TO CLOSE EYES <<")
                    else:
                        emit_audio_cue("STEP_ADVANCE", is_headless=self.is_headless)
                else:
                    self.phase_state = "DONE"
                    self.is_finished = True
                    if was_closed:
                        emit_audio_cue("EYES_OPEN", is_headless=self.is_headless)

    def record_frame(
        self,
        face_detected: bool,
        face_crop_valid: bool,
        geometry_valid: bool,
        right_eye_valid: Optional[bool] = None,
        left_eye_valid: Optional[bool] = None,
        r_feat: Optional[Any] = None,
        l_feat: Optional[Any] = None,
        canthal_dist_px: Optional[float] = None,
        r_canthal_ratio: Optional[float] = None,
        l_canthal_ratio: Optional[float] = None,
        model_output: Optional[Any] = None,
        t_mp: float = 0.0,
        t_total: float = 0.0,
        unavailable_reason: Optional[str] = None,
    ) -> Optional[CalibrationValidationRecord]:
        """Records a frame if in RECORDING phase."""
        self.frame_count += 1
        if self.phase_state != "RECORDING" or self.is_finished:
            return None

        now_mono = time.monotonic()
        elapsed_s = now_mono - self.session_start_mono
        cond = self.current_condition

        rec = CalibrationValidationRecord(
            session_id=self.session_id,
            monotonic_time_s=now_mono,
            elapsed_session_s=elapsed_s,
            protocol_phase=cond.phase.value,
            condition_index=cond.index,
            condition_name=cond.name,
            target_direction=cond.target_direction,
            frame_index=self.frame_count,
            face_detected=face_detected,
            face_crop_valid=face_crop_valid,
            geometry_valid=geometry_valid,
            active_eye=self.active_eye,
            right_eye_valid=right_eye_valid,
            left_eye_valid=left_eye_valid,
            right_eye_h_ratio=r_feat.h_ratio if r_feat else None,
            right_eye_v_ratio=r_feat.v_ratio if r_feat else None,
            right_eye_ear=r_feat.ear if r_feat else None,
            right_eye_canthal_ratio=r_canthal_ratio,
            left_eye_h_ratio=l_feat.h_ratio if l_feat else None,
            left_eye_v_ratio=l_feat.v_ratio if l_feat else None,
            left_eye_ear=l_feat.ear if l_feat else None,
            left_eye_canthal_ratio=l_canthal_ratio,
            canthal_normal_dist_px=canthal_dist_px,
            mobileone_valid=bool(model_output and model_output.is_valid),
            mobileone_yaw_deg=model_output.yaw_deg if (model_output and model_output.is_valid) else None,
            mobileone_pitch_deg=model_output.pitch_deg if (model_output and model_output.is_valid) else None,
            mediapipe_latency_ms=t_mp,
            total_frame_latency_ms=t_total,
            unavailable_reason=unavailable_reason,
        )

        self.records.append(rec)
        if self._csv_writer and self._csv_file:
            self._csv_writer.writerow(rec.to_csv_row())
            self._csv_file.flush()

        return rec

    def complete_session(self) -> Dict[str, Any]:
        """Finalizes session, runs evaluation analysis, and saves summary JSON."""
        self.is_finished = True
        if self._csv_file:
            try:
                self._csv_file.flush()
                self._csv_file.close()
            except Exception:
                pass
            self._csv_file = None

        report = evaluate_calibration_validation_session(self.records, active_eye=self.active_eye)
        report["metadata"]["session_id"] = self.session_id
        report["metadata"]["csv_file"] = str(self.csv_path)
        report["metadata"]["json_summary_file"] = str(self.json_summary_path)
        report["metadata"]["settling_duration_s"] = self.settling_duration_s
        report["metadata"]["recording_duration_s"] = self.recording_duration_s
        report["metadata"]["total_conditions_in_protocol"] = len(self.conditions)
        report["metadata"]["completed_conditions"] = self.current_cond_idx + (1 if self.phase_state == "DONE" else 0)

        with open(self.json_summary_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report

    def draw_hud(self, frame: np.ndarray) -> np.ndarray:
        """Renders 5 on-screen visual target circles, countdown pill, and instructions."""
        h, w = frame.shape[:2]
        canvas = frame.copy()

        cond = self.current_condition
        phase = self.phase_state
        now = time.monotonic()
        time_left = max(0.0, (self.settling_duration_s if phase == "SETTLING" else self.recording_duration_s) - (now - self.phase_start_mono))

        # Target circles dictionary
        targets = {
            "CENTER": (int(w * TARGET_CENTER[0]), int(h * TARGET_CENTER[1])),
            "LEFT": (int(w * TARGET_LEFT[0]), int(h * TARGET_LEFT[1])),
            "RIGHT": (int(w * TARGET_RIGHT[0]), int(h * TARGET_RIGHT[1])),
            "UP": (int(w * TARGET_UP[0]), int(h * TARGET_UP[1])),
            "DOWN": (int(w * TARGET_DOWN[0]), int(h * TARGET_DOWN[1])),
        }

        # Draw inactive targets
        for t_name, (tx, ty) in targets.items():
            cv2.circle(canvas, (tx, ty), 22, (60, 60, 60), 2)
            cv2.circle(canvas, (tx, ty), 6, (100, 100, 100), -1)

        # Highlight active target if condition maps to one
        active_key = cond.target_direction if cond.target_direction in targets else ("CENTER" if cond.is_center else None)
        if active_key and active_key in targets:
            atx, aty = targets[active_key]
            glow_col = (0, 180, 255) if phase == "SETTLING" else (0, 255, 0)
            cv2.circle(canvas, (atx, aty), 32, glow_col, 3)
            cv2.circle(canvas, (atx, aty), 22, glow_col, -1)
            cv2.circle(canvas, (atx, aty), 8, (255, 255, 255), -1)

        # Top Information Banner
        cv2.rectangle(canvas, (0, 0), (w, 90), (15, 15, 15), -1)
        phase_col = (0, 220, 255) if phase == "SETTLING" else (0, 255, 0)
        cv2.putText(canvas, f"CONDITION {cond.index}/{len(self.conditions)}: {cond.name} [{cond.phase.value}]", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(canvas, cond.instruction, (16, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)

        # Status Pill (Settling vs Recording)
        pill_w = 260
        pill_col = (100, 60, 0) if phase == "SETTLING" else (0, 120, 0)
        pill_text = f"SETTLING: {time_left:.1f}s" if phase == "SETTLING" else f"RECORDING: {time_left:.1f}s ({len(self.records)} f)"
        cv2.rectangle(canvas, (w - pill_w - 16, 16), (w - 16, 68), pill_col, -1)
        cv2.rectangle(canvas, (w - pill_w - 16, 16), (w - 16, 68), phase_col, 2)
        cv2.putText(canvas, pill_text, (w - pill_w - 4, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

        # High-contrast center alert for closed-eye calibration
        if cond.target_direction == "CLOSED":
            box_w, box_h = min(840, w - 80), 160
            bx = (w - box_w) // 2
            by = (h - box_h) // 2
            cv2.rectangle(canvas, (bx, by), (bx + box_w, by + box_h), (0, 0, 180), -1)
            cv2.rectangle(canvas, (bx, by), (bx + box_w, by + box_h), (0, 220, 255), 3)
            if phase == "SETTLING":
                cv2.putText(canvas, "*** PREPARE TO CLOSE EYES ***", (bx + 40, by + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(canvas, f"Close eyes when tone sounds ({time_left:.1f}s)", (bx + 40, by + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 255), 2)
            else:
                cv2.putText(canvas, "*** CLOSE BOTH EYES AND HOLD CLOSED ***", (bx + 20, by + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
                cv2.putText(canvas, f"Hold closed until tone sounds ({time_left:.1f}s)", (bx + 40, by + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 255, 150), 2)

        # Bottom Instructions
        cv2.rectangle(canvas, (0, h - 36), (w, h), (15, 15, 15), -1)
        cv2.putText(canvas, "[ESC]/[Q]: Exit & Finalize | Visual target circle indicates where to direct gaze comfortably.", (16, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 160, 160), 1)

        return canvas


def run_session(
    camera_idx: int = 0,
    width: int = 1280,
    height: int = 720,
    settling_s: float = 2.0,
    recording_s: float = 4.0,
    active_eye: str = "right",
    is_headless: bool = False,
    max_conditions: Optional[int] = None,
) -> bool:
    """Runs the live calibration validation loop."""
    import mediapipe as mp

    output_dir = Path("artifacts/observations")
    harness = CalibrationValidationHarness(
        output_dir=output_dir,
        settling_duration_s=settling_s,
        recording_duration_s=recording_s,
        active_eye=active_eye,
        is_headless=is_headless,
    )

    if max_conditions is not None:
        harness.conditions = harness.conditions[:max_conditions]

    print(f"\n============================================================")
    print(f"      VOXGAZE M1.5-B1 PHYSICAL CALIBRATION VALIDATION       ")
    print(f"============================================================")
    print(f" Session ID          : {harness.session_id}")
    print(f" Conditions Planned  : {len(harness.conditions)}")
    print(f" Settling Timer      : {settling_s}s (no data logged)")
    print(f" Recording Timer     : {recording_s}s (streamed to CSV)")
    print(f" Active Eye Baseline : {active_eye.upper()}")
    print(f" CSV Stream Output   : {harness.csv_path}")
    print(f" Summary JSON Output : {harness.json_summary_path}")
    print(f" Privacy             : Zero raw frames or face crops stored.")
    print(f"============================================================\n")

    # Initialize MediaPipe FaceLandmarker
    landmarker_path = ROOT_DIR / "models" / "face_landmarker.task"
    if not landmarker_path.is_file():
        print(f"[FATAL] MediaPipe Face Landmarker model asset missing: {landmarker_path}")
        return False

    base_opts = python.BaseOptions(model_asset_path=str(landmarker_path))
    options = vision.FaceLandmarkerOptions(
        base_options=base_opts,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    model_path = ROOT_DIR / "models" / "mobileone_s0_gaze.onnx"
    estimator = MobileOneGazeEstimator(model_path=model_path) if model_path.is_file() else None

    cap = None
    if not is_headless:
        cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not cap.isOpened():
            print(f"[ERROR] Could not open camera {camera_idx}. Exiting.")
            return False

    window_name = "VoxGaze M1.5-B1 Physical Calibration Validation"
    if not is_headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width, height)

    try:
        while not harness.is_finished:
            t_start = time.perf_counter()
            if not is_headless and cap is not None:
                ret, raw_frame = cap.read()
                if not ret or raw_frame is None:
                    continue
                # Note: raw_frame is kept unmirrored so MediaPipe and MobileOne receive pristine camera images.
            else:
                # Headless synthetic frame
                raw_frame = np.zeros((height, width, 3), dtype=np.uint8)

            rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(time.perf_counter() * 1000)
            t_mp_start = time.perf_counter()
            detection_result = detector.detect_for_video(mp_image, ts_ms)
            t_mp = (time.perf_counter() - t_mp_start) * 1000.0

            face_detected = bool(detection_result.face_landmarks)
            crop_result = None
            r_feat, l_feat = None, None
            canthal_dist_px = None
            r_canthal_ratio, l_canthal_ratio = None, None
            bilateral_quality = None
            model_output = None
            unavailable_reason = None

            cond = harness.current_condition
            is_closed_condition = (cond.target_direction == "CLOSED")

            if face_detected:
                face = detection_result.face_landmarks[0]
                h_f, w_f = raw_frame.shape[:2]
                pts = np.array([[p.x * w_f, p.y * h_f] for p in face])

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

                r_feat = extract_eye_features(r_iris, r_inner, r_outer, r_up, r_low, is_right_eye=True)
                l_feat = extract_eye_features(l_iris, l_inner, l_outer, l_up, l_low, is_right_eye=False)

                canthal_dist_px = compute_canthal_normal_distance_px(r_iris, r_inner, r_outer, is_right_eye=True)
                r_canthal_ratio = norm_vertical_gaze_canthal_normal_experimental(r_iris, r_inner, r_outer, is_right_eye=True)
                l_canthal_ratio = norm_vertical_gaze_canthal_normal_experimental(l_iris, l_inner, l_outer, is_right_eye=False)

                r_q = validate_eye_geometry(
                    r_iris, r_inner, r_outer, r_up, r_low,
                    is_right_eye=True,
                    allow_closed_eyes=is_closed_condition,
                )
                l_q = validate_eye_geometry(
                    l_iris, l_inner, l_outer, l_up, l_low,
                    is_right_eye=False,
                    allow_closed_eyes=is_closed_condition,
                )
                bilateral_quality = validate_bilateral_geometry(r_q, l_q, require_both_eyes=False)

                crop_result = extract_face_crop(raw_frame, face, padding_scale=1.20)
                if estimator and crop_result.is_valid and crop_result.crop_bgr is not None:
                    model_output = estimator.predict(crop_result.crop_bgr)

            # Determine quality validity
            is_geom_valid = bool(bilateral_quality and bilateral_quality.is_valid)
            r_valid = bilateral_quality.right_eye.is_valid if bilateral_quality else None
            l_valid = bilateral_quality.left_eye.is_valid if bilateral_quality else None

            if not face_detected:
                unavailable_reason = "NO_FACE_DETECTED"
            elif not is_geom_valid and bilateral_quality:
                unavailable_reason = "; ".join(bilateral_quality.bilateral_reasons) or "GEOMETRY_INVALID"

            t_total = (time.perf_counter() - t_start) * 1000.0

            # Advance state & record
            harness.tick()
            harness.record_frame(
                face_detected=face_detected,
                face_crop_valid=bool(crop_result and crop_result.is_valid),
                geometry_valid=is_geom_valid,
                right_eye_valid=r_valid,
                left_eye_valid=l_valid,
                r_feat=r_feat,
                l_feat=l_feat,
                canthal_dist_px=canthal_dist_px,
                r_canthal_ratio=r_canthal_ratio,
                l_canthal_ratio=l_canthal_ratio,
                model_output=model_output,
                t_mp=t_mp,
                t_total=t_total,
                unavailable_reason=unavailable_reason,
            )

            if not is_headless:
                display_frame = cv2.flip(raw_frame, 1)
                hud = harness.draw_hud(display_frame)
                cv2.imshow(window_name, hud)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), ord("Q"), 27):
                    print("[INFO] Session interrupted by user keypress.")
                    break

        return True

    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        report = harness.complete_session()
        _print_summary(report)


def _print_summary(report: Dict[str, Any]) -> None:
    """Pretty prints the M1.5-B1 evaluation summary."""
    print("\n============================================================")
    print("        M1.5-B1 CALIBRATION VALIDATION SUMMARY REPORT       ")
    print("============================================================")
    meta = report["metadata"]
    print(f" Session ID          : {meta.get('session_id')}")
    print(f" Frames Recorded     : {meta.get('total_frames_recorded')}")
    print(f" Face Availability   : {meta.get('overall_face_availability_pct')}%")
    print(f" Geometry Valid Rate : {meta.get('overall_geometry_valid_pct')}%")
    print(f" CSV File            : {meta.get('csv_file')}")
    print(f" Summary JSON File   : {meta.get('json_summary_file')}")
    print("------------------------------------------------------------")

    p1 = report["phase1_calibration"]
    gates = p1.get("gate_results", {})
    sep = p1.get("directional_separations", {})
    print(" Phase 1 Directional Separation (vs Center Baseline):")
    print(f"   Left Separation   : delta H = {sep.get('delta_h_left')} (Supported: {gates.get('is_left_supported')})")
    print(f"   Right Separation  : delta H = {sep.get('delta_h_right')} (Supported: {gates.get('is_right_supported')})")
    print(f"   Up Separation (V) : delta V eyelid = {sep.get('delta_v_eyelid_up')} | canthal = {sep.get('delta_v_canthal_up')} (Supported: {gates.get('is_up_supported')})")
    print(f"   Down Separation   : delta V eyelid = {sep.get('delta_v_eyelid_down')} | canthal = {sep.get('delta_v_canthal_down')} (Supported: {gates.get('is_down_supported')})")
    print(f"   Blink Separation  : delta EAR = {sep.get('delta_ear_blink')} (Supported: {gates.get('is_blink_supported')})")

    if gates.get("unsupported_reasons"):
        print(" Unsupported Gates   :")
        for r in gates["unsupported_reasons"]:
            print(f"   * {r}")

    p2 = report.get("phase2_independent_evaluation")
    if p2:
        print("------------------------------------------------------------")
        print(" Phase 2 Independent Test Trial Evaluation:")
        status = p2.get("evaluation_status", "UNKNOWN")
        trial_acc = p2.get("trial_level_accuracy")
        frame_acc = p2.get("frame_level_accuracy")
        trial_str = f"{trial_acc * 100:.1f}% ({p2.get('trials_correct_count')}/{p2.get('trials_evaluated_count')} trials)" if trial_acc is not None else "UNAVAILABLE (All directions unsupported)"
        frame_str = f"{frame_acc * 100:.1f}% ({p2.get('correct_test_frames')}/{p2.get('total_test_frames')} frames)" if frame_acc is not None else "UNAVAILABLE"
        print(f"   Evaluation Status    : {status}")
        print(f"   Trial-Level Accuracy : {trial_str}")
        print(f"   Frame-Level Accuracy : {frame_str}")
        c_stab = p2.get("center_fixation_stability", {})
        if c_stab.get("trials_evaluated", 0) > 0:
            c_tr_acc = c_stab.get("trial_accuracy")
            c_tr_str = f"{c_tr_acc * 100:.1f}%" if c_tr_acc is not None else "N/A"
            print(f"   Center Fixation Stab : {c_tr_str} ({c_stab.get('trials_correct')}/{c_stab.get('trials_evaluated')} center trials)")
        d_sens = p2.get("directional_sensitivity", {})
        if d_sens:
            d_tr_acc = d_sens.get("trial_accuracy")
            d_tr_str = f"{d_tr_acc * 100:.1f}%" if d_tr_acc is not None else "UNAVAILABLE"
            print(f"   Directional Sensitiv : {d_tr_str} ({d_sens.get('trials_correct', 0)}/{d_sens.get('trials_evaluated', 0)} directional trials)")
        print("   Note: Intra-trial frames are temporally correlated; trial accuracy is primary.")

    drift = report.get("baseline_center_drift", {})
    if "max_absolute_drift_h" in drift:
        print("------------------------------------------------------------")
        print(" Center-Return Baseline Drift Stability:")
        print(f"   Max Drift (H)     : {drift.get('max_absolute_drift_h')} ratio")
        print(f"   Max Drift (V)     : {drift.get('max_absolute_drift_v')} ratio")
        print(f"   Center Trials     : {drift.get('center_returns_evaluated')} repetitions evaluated")

    print("============================================================\n")


def main():
    parser = argparse.ArgumentParser(description="VoxGaze M1.5-B1 Physical Calibration Validation Harness.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Frame width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Frame height (default: 720)")
    parser.add_argument("--settling", type=float, default=2.0, help="Settling duration in seconds (default: 2.0)")
    parser.add_argument("--recording", type=float, default=4.0, help="Recording duration in seconds (default: 4.0)")
    parser.add_argument("--eye", type=str, default="right", choices=["right", "left"], help="Active eye baseline (default: right)")
    parser.add_argument("--headless", action="store_true", help="Run in headless synthetic mode without GUI window")
    parser.add_argument("--max-conditions", type=int, default=None, help="Limit number of conditions for quick testing")
    args = parser.parse_args()

    run_session(
        camera_idx=args.camera,
        width=args.width,
        height=args.height,
        settling_s=args.settling,
        recording_s=args.recording,
        active_eye=args.eye,
        is_headless=args.headless,
        max_conditions=args.max_conditions,
    )


if __name__ == "__main__":
    main()
