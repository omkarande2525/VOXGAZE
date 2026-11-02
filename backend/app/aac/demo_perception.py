"""
VoxGaze AAC Demo: Live Perception Worker and Calibration Pipeline

Provides:
1. run_calibration: Fullscreen visual fixation calibration (BLOCK_1_TARGETS + BLOCK_2_TARGETS),
   MobileOne S0, hybrid features, alpha=1.0. Fits PersonalizedGazeMapper and returns (mapper, cap).
2. LiveGazeWorker: Background daemon thread reusing camera, running MediaPipe FaceLandmarker,
   face crop, MobileOne S0, and PersonalizedGazeMapper.predict(). Publishes latest sample,
   deque history, measured FPS, and latency. Clean stop() releases camera.
3. print_x_readout: Console validation tool printing x coordinate with dual-threshold hysteresis
   (LEFT enter/exit 0.35/0.40, RIGHT enter/exit 0.65/0.60) every ~0.25 s.

REUSE ATTRIBUTION:
Target coordinates, 18-trial block sequences, visual canvas drawing, eye geometry extraction,
and face crop inference copied and adapted from scripts/run_controlled_model_comparison.py:
- Lines 68-83 (TARGET_COORDS, TARGET_BBOXES)
- Lines 86-87 (BLOCK_1_TARGETS, BLOCK_2_TARGETS)
- Lines 888-995 (draw_centered_text, draw_target_canvas)
- Lines 1201-1273 (MediaPipe video detection, landmark indexing, eye features, face crop, MobileOne)
- Lines 522-550, 587-589 (CalibrationSample assembly and PersonalizedGazeMapper fitting)

SAFETY INVARIANTS:
1. Never writes raw webcam frames or face crops to disk.
2. Does not alter default is_selection_enabled in LiveGazeProvider or touch existing AAC routes.
"""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from backend.app.calibration.personalized_mapper import (
    CalibrationSample,
    FeatureSetType,
    MappingPrediction,
    PersonalizedGazeMapper,
)
from backend.app.core.contracts import Point2D
from backend.app.perception.face_crop import extract_face_crop
from backend.app.perception.geometry import (
    compute_canthal_normal_distance_px,
    extract_eye_features,
)
from backend.app.perception.model_registry import create_gaze_estimator

# Landmark indices from canonical MediaPipe mesh
# Source reference: scripts/run_controlled_model_comparison.py:56-65
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
# Source reference: scripts/run_controlled_model_comparison.py:68-74
TARGET_COORDS: Dict[str, Tuple[float, float]] = {
    "CENTER": (0.50, 0.50),
    "LEFT": (0.15, 0.50),
    "RIGHT": (0.85, 0.50),
    "UP": (0.50, 0.15),
    "DOWN": (0.50, 0.85),
}

# Standard AAC button bounding boxes
# Source reference: scripts/run_controlled_model_comparison.py:77-83
TARGET_BBOXES: Dict[str, Tuple[float, float, float, float]] = {
    "CENTER": (0.35, 0.35, 0.65, 0.65),  # Resting deadzone
    "LEFT": (0.00, 0.30, 0.35, 0.70),    # Left button
    "RIGHT": (0.65, 0.30, 1.00, 0.70),   # Right button
    "UP": (0.30, 0.00, 0.70, 0.35),      # Top button
    "DOWN": (0.30, 0.65, 0.70, 1.00),    # Bottom button
}

# Balanced sequences for the calibration blocks (interleaved center returns)
# Source reference: scripts/run_controlled_model_comparison.py:86-87
BLOCK_1_TARGETS: List[str] = ["CENTER", "LEFT", "CENTER", "RIGHT", "CENTER", "UP", "CENTER", "DOWN", "CENTER"]
BLOCK_2_TARGETS: List[str] = ["CENTER", "RIGHT", "CENTER", "DOWN", "CENTER", "LEFT", "CENTER", "UP", "CENTER"]
CALIBRATION_SEQUENCE: List[str] = BLOCK_1_TARGETS + BLOCK_2_TARGETS  # Exactly 18 trials


@dataclass(frozen=True)
class GazeSample:
    """Published gaze measurement packet."""
    timestamp: float          # time.perf_counter() timestamp
    x: float                  # Normalized mapped screen X [0.0, 1.0]
    y: float                  # Normalized mapped screen Y [0.0, 1.0]
    valid: bool               # True if face detected, crop valid, and mapper succeeded
    face_detected: bool       # Face presence flag
    latency_ms: float = 0.0   # End-to-end perception latency
    yaw_deg: Optional[float] = None
    pitch_deg: Optional[float] = None


@dataclass(frozen=True)
class ExtractedFrameFeatures:
    """Intermediate numerical features from a single frame."""
    face_detected: bool
    face_crop_valid: bool
    estimator_valid: bool
    yaw_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    right_eye_h: Optional[float] = None
    right_eye_v: Optional[float] = None
    right_eye_ear: Optional[float] = None
    left_eye_h: Optional[float] = None
    left_eye_v: Optional[float] = None
    left_eye_ear: Optional[float] = None
    canthal_normal_px: Optional[float] = None
    detector_ms: float = 0.0
    crop_ms: float = 0.0
    infer_ms: float = 0.0


def draw_centered_text(
    img: np.ndarray,
    text: str,
    cx: int,
    cy: int,
    font_scale: float,
    color: Tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """
    Draw text centered horizontally and vertically at (cx, cy).
    Source reference: scripts/run_controlled_model_comparison.py:888-902.
    """
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
    5-button layout matching TARGET_BBOXES.
    Source reference: scripts/run_controlled_model_comparison.py:904-995.
    """
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # 1. Draw 5-button layout using identical bounding boxes
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
                cv2.circle(canvas, (cx, cy), 20, border_col, 2, cv2.LINE_AA)
                cv2.circle(canvas, (cx, cy), 6, (0, 200, 255), -1, cv2.LINE_AA)
                draw_centered_text(canvas, f"{btn_name} [SETTLE]", cx, cy - 36, 0.7, border_col, 2)
            else:
                border_col = (0, 255, 0)  # Bright green for active fixation recording
                fill_col = (0, 60, 15)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), fill_col, -1)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), border_col, 4)
                cv2.circle(canvas, (cx, cy), 22, border_col, 3, cv2.LINE_AA)
                cv2.circle(canvas, (cx, cy), 6, (50, 255, 50), -1, cv2.LINE_AA)
                draw_centered_text(canvas, f"{btn_name} [FIXATE]", cx, cy - 36, 0.7, border_col, 2)
        else:
            border_col = (60, 60, 60)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), border_col, 1)
            draw_centered_text(canvas, btn_name, cx, cy, 0.65, (110, 110, 110), 1)

    # 2. Top status line
    if phase == "SETTLING":
        status_line = f"Block {block_num}/2, Trial {trial_num}/{total_trials}, SETTLE ({time_left_s:.1f}s)"
        status_col = (0, 180, 255)
    else:
        status_line = f"Block {block_num}/2, Trial {trial_num}/{total_trials}, fixate now ({time_left_s:.1f}s)"
        status_col = (0, 255, 0)

    draw_centered_text(canvas, status_line, canvas_w // 2, 40, 0.85, status_col, 2)

    # 3. Bottom instructions
    cv2.putText(
        canvas,
        "Press ESC to abort calibration",
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


def create_face_landmarker(model_path: Optional[Path] = None) -> Any:
    """
    Initializes MediaPipe FaceLandmarker for video mode.
    Source reference: scripts/run_controlled_model_comparison.py:1123-1141.
    """
    if model_path is None:
        root_dir = Path(__file__).resolve().parent.parent.parent.parent
        model_path = root_dir / "models" / "face_landmarker.task"

    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    base_opts = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def extract_gaze_features_from_frame(
    raw_frame: np.ndarray,
    detector: Any,
    estimator: Any,
    ts_ms: Optional[int] = None,
) -> ExtractedFrameFeatures:
    """
    Extracts ocular geometry, face crop, and MobileOne S0 gaze angles from a single frame.
    Never saves frames to disk.
    Source reference: scripts/run_controlled_model_comparison.py:1201-1273.
    """
    if ts_ms is None:
        ts_ms = int(time.perf_counter() * 1000)

    # 1. MediaPipe FaceLandmarker detection
    t_mp_s = time.perf_counter()
    rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    detection_result = detector.detect_for_video(mp_img, ts_ms)
    t_detector = (time.perf_counter() - t_mp_s) * 1000.0

    face_detected = bool(detection_result.face_landmarks)
    if not face_detected:
        return ExtractedFrameFeatures(
            face_detected=False,
            face_crop_valid=False,
            estimator_valid=False,
            detector_ms=round(t_detector, 2),
        )

    face = detection_result.face_landmarks[0]
    h_f, w_f = raw_frame.shape[:2]
    pts = np.array([[p.x * w_f, p.y * h_f] for p in face])

    # 2. Eye Landmark Geometry
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
    cn_dist = compute_canthal_normal_distance_px(r_iris, r_inner, r_outer, is_right_eye=True)

    # 3. Face Crop Extraction
    t_crop_s = time.perf_counter()
    crop_res = extract_face_crop(raw_frame, face, padding_scale=1.20)
    t_crop = (time.perf_counter() - t_crop_s) * 1000.0

    if not (crop_res.is_valid and crop_res.crop_bgr is not None):
        return ExtractedFrameFeatures(
            face_detected=True,
            face_crop_valid=False,
            estimator_valid=False,
            right_eye_h=round(r_f.h_ratio, 4) if r_f.h_ratio is not None else None,
            right_eye_v=round(r_f.v_ratio, 4) if r_f.v_ratio is not None else None,
            right_eye_ear=round(r_f.ear, 4) if r_f.ear is not None else None,
            left_eye_h=round(l_f.h_ratio, 4) if l_f.h_ratio is not None else None,
            left_eye_v=round(l_f.v_ratio, 4) if l_f.v_ratio is not None else None,
            left_eye_ear=round(l_f.ear, 4) if l_f.ear is not None else None,
            canthal_normal_px=round(cn_dist, 2) if cn_dist is not None else None,
            detector_ms=round(t_detector, 2),
            crop_ms=round(t_crop, 2),
        )

    # 4. MobileOne S0 Inference
    t_infer_s = time.perf_counter()
    m1_out = estimator.predict(crop_res.crop_bgr)
    t_infer = (time.perf_counter() - t_infer_s) * 1000.0

    return ExtractedFrameFeatures(
        face_detected=True,
        face_crop_valid=True,
        estimator_valid=bool(m1_out.is_valid),
        yaw_deg=round(m1_out.yaw_deg, 2) if (m1_out.is_valid and m1_out.yaw_deg is not None) else None,
        pitch_deg=round(m1_out.pitch_deg, 2) if (m1_out.is_valid and m1_out.pitch_deg is not None) else None,
        right_eye_h=round(r_f.h_ratio, 4) if r_f.h_ratio is not None else None,
        right_eye_v=round(r_f.v_ratio, 4) if r_f.v_ratio is not None else None,
        right_eye_ear=round(r_f.ear, 4) if r_f.ear is not None else None,
        left_eye_h=round(l_f.h_ratio, 4) if l_f.h_ratio is not None else None,
        left_eye_v=round(l_f.v_ratio, 4) if l_f.v_ratio is not None else None,
        left_eye_ear=round(l_f.ear, 4) if l_f.ear is not None else None,
        canthal_normal_px=round(cn_dist, 2) if cn_dist is not None else None,
        detector_ms=round(t_detector, 2),
        crop_ms=round(t_crop, 2),
        infer_ms=round(t_infer, 2),
    )


def run_calibration(
    camera_idx: int = 0,
    cap: Optional[Any] = None,
    detector: Optional[Any] = None,
    estimator: Optional[Any] = None,
    settling_duration_s: float = 1.5,
    recording_duration_s: float = 3.0,
    headless: bool = False,
    on_trial_start: Optional[Callable[[int, int, str], None]] = None,
) -> Tuple[Optional[PersonalizedGazeMapper], Optional[Any]]:
    """
    Runs a 2-block 5-point visual fixation calibration (BLOCK_1_TARGETS + BLOCK_2_TARGETS = 18 trials)
    using MobileOne S0 on hybrid features with Ridge alpha=1.0.
    Fits and returns (PersonalizedGazeMapper, open_camera_handle) for immediate reuse by LiveGazeWorker.
    Pressing ESC aborts cleanly without fitting or crashing, returning (None, cap).
    Never writes frames or images to disk.

    Source references:
    - scripts/run_controlled_model_comparison.py:1111-1330 (Sequential calibration trial loop)
    - scripts/run_controlled_model_comparison.py:522-550 (_to_samples)
    - scripts/run_controlled_model_comparison.py:587-589 (PersonalizedGazeMapper fit)
    """
    # 1. Initialize camera handle if not provided
    if cap is None:
        cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_idx)
        if not cap.isOpened():
            return None, None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 2. Initialize perception models
    if detector is None and not headless:
        detector = create_face_landmarker()
    if estimator is None and not headless:
        estimator = create_gaze_estimator("mobileone_s0")

    blocks = [
        (1, BLOCK_1_TARGETS),
        (2, BLOCK_2_TARGETS),
    ]

    training_samples: List[CalibrationSample] = []
    win_name = "VoxGaze Demo Calibration (Press ESC to Abort)"
    aborted = False

    if not headless:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        total_trials_per_block = len(BLOCK_1_TARGETS)
        for block_idx, targets in blocks:
            if aborted:
                break

            for cond_idx, target in enumerate(targets):
                if aborted:
                    break

                trial_name = f"B{block_idx}_{cond_idx+1}_{target}"
                if on_trial_start is not None:
                    on_trial_start(block_idx, cond_idx + 1, target)

                # Phase 1: SETTLING
                t_settle_start = time.monotonic()
                while (time.monotonic() - t_settle_start) < settling_duration_s:
                    ret, raw_frame = cap.read()
                    if not ret or raw_frame is None:
                        time.sleep(0.005)
                        continue

                    if not headless:
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
                    else:
                        if getattr(cap, "aborted", False):
                            aborted = True
                            break

                if aborted:
                    break

                # Phase 2: RECORDING
                t_rec_start = time.monotonic()
                while (time.monotonic() - t_rec_start) < recording_duration_s:
                    ret, raw_frame = cap.read()
                    if not ret or raw_frame is None:
                        time.sleep(0.005)
                        continue

                    feats = extract_gaze_features_from_frame(raw_frame, detector, estimator)
                    if (
                        feats.face_crop_valid
                        and feats.estimator_valid
                        and feats.yaw_deg is not None
                        and feats.pitch_deg is not None
                    ):
                        tx, ty = TARGET_COORDS.get(target, (0.50, 0.50))
                        training_samples.append(
                            CalibrationSample(
                                target_x=tx,
                                target_y=ty,
                                target_direction=target,
                                yaw_deg=feats.yaw_deg,
                                pitch_deg=feats.pitch_deg,
                                right_eye_h=feats.right_eye_h,
                                right_eye_v=feats.right_eye_v,
                                left_eye_h=feats.left_eye_h,
                                left_eye_v=feats.left_eye_v,
                                canthal_normal=feats.canthal_normal_px,
                                condition_name=trial_name,
                                trial_index=cond_idx + 1,
                            )
                        )

                    if not headless:
                        rem_s = max(0.0, recording_duration_s - (time.monotonic() - t_rec_start))
                        canvas = draw_target_canvas(
                            1280, 720, target, "RECORDING", rem_s,
                            block_idx, cond_idx + 1, total_trials_per_block,
                            camera_preview=raw_frame, face_detected=feats.face_detected,
                        )
                        cv2.imshow(win_name, canvas)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (27, ord("q"), ord("Q")):
                            aborted = True
                            break
                    else:
                        if getattr(cap, "aborted", False):
                            aborted = True
                            break

    finally:
        if not headless:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    # ESC abort: return without fitting and without crashing
    if aborted:
        return None, cap

    if len(training_samples) < 5:
        return None, cap

    # Fit PersonalizedGazeMapper on Phase 1 calibration data
    mapper = PersonalizedGazeMapper(feature_set=FeatureSetType.HYBRID, alpha=1.0)
    fit_ok = mapper.fit(training_samples)
    if not fit_ok:
        return None, cap

    return mapper, cap


class LiveGazeWorker:
    """
    Background daemon worker thread executing continuous live gaze estimation.
    Reuses the camera handle from run_calibration().
    Publishes latest sample, a sliding deque of recent samples, measured FPS, and per-frame latency.
    Cleanly shuts down on stop() and releases the camera.
    """

    def __init__(
        self,
        cap: Any,
        mapper: PersonalizedGazeMapper,
        estimator: Optional[Any] = None,
        detector: Optional[Any] = None,
        max_deque_len: int = 30,
    ):
        self.cap = cap
        self.mapper = mapper
        self.estimator = estimator or create_gaze_estimator("mobileone_s0")
        self.detector = detector or create_face_landmarker()
        self.samples: Deque[GazeSample] = deque(maxlen=max_deque_len)
        self.latest_sample: Optional[GazeSample] = None
        self.fps: float = 0.0
        self.per_frame_latency: float = 0.0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Starts the background perception worker thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="LiveGazeWorker")
        self._thread.start()

    def stop(self) -> None:
        """Stops the worker thread cleanly and releases the camera."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

    def _run_loop(self) -> None:
        frame_times: Deque[float] = deque(maxlen=30)

        while not self._stop_event.is_set():
            t0 = time.perf_counter()
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            feats = extract_gaze_features_from_frame(frame, self.detector, self.estimator)
            valid = False
            mapped_x = 0.50
            mapped_y = 0.50

            if feats.face_detected and feats.face_crop_valid and feats.estimator_valid:
                if (
                    feats.yaw_deg is not None
                    and feats.pitch_deg is not None
                    and self.mapper is not None
                    and getattr(self.mapper, "is_fitted", False)
                ):
                    sample = CalibrationSample(
                        target_x=0.50,
                        target_y=0.50,
                        target_direction="CENTER",
                        yaw_deg=feats.yaw_deg,
                        pitch_deg=feats.pitch_deg,
                        right_eye_h=feats.right_eye_h,
                        right_eye_v=feats.right_eye_v,
                        left_eye_h=feats.left_eye_h,
                        left_eye_v=feats.left_eye_v,
                        canthal_normal=feats.canthal_normal_px,
                    )
                    pred = self.mapper.predict(sample)
                    if pred.is_valid:
                        valid = True
                        mapped_x = pred.screen_x
                        mapped_y = pred.screen_y

            latency_ms = (time.perf_counter() - t0) * 1000.0
            gaze_sample = GazeSample(
                timestamp=t0,
                x=mapped_x,
                y=mapped_y,
                valid=valid,
                face_detected=feats.face_detected,
                latency_ms=round(latency_ms, 2),
                yaw_deg=feats.yaw_deg,
                pitch_deg=feats.pitch_deg,
            )

            t_now = time.perf_counter()
            frame_times.append(t_now)
            if len(frame_times) > 1:
                dt = frame_times[-1] - frame_times[0]
                fps = (len(frame_times) - 1) / dt if dt > 0 else 0.0
            else:
                fps = 0.0

            with self._lock:
                self.latest_sample = gaze_sample
                self.samples.append(gaze_sample)
                self.per_frame_latency = round(latency_ms, 2)
                self.fps = round(fps, 1)


def classify_x_hysteresis(x: float, current_state: str) -> str:
    """
    Applies dual-threshold hysteresis to classify horizontal screen coordinate x:
    - Enter LEFT: x < 0.35, Exit LEFT: x > 0.40
    - Enter RIGHT: x > 0.65, Exit RIGHT: x < 0.60
    """
    if current_state == "LEFT":
        if x > 0.65:
            return "RIGHT"
        elif x > 0.40:
            return "CENTER"
        return "LEFT"
    elif current_state == "RIGHT":
        if x < 0.35:
            return "LEFT"
        elif x < 0.60:
            return "CENTER"
        return "RIGHT"
    else:  # CENTER
        if x < 0.35:
            return "LEFT"
        elif x > 0.65:
            return "RIGHT"
        return "CENTER"


def print_x_readout(
    seconds: float = 10.0,
    worker: Optional[LiveGazeWorker] = None,
    interval_s: float = 0.25,
    output_fn: Callable[[str], None] = print,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    """
    Prints x readout about every 0.25 s with a LEFT / CENTER / RIGHT label
    using enter/exit thresholds 0.35/0.40 and 0.65/0.60 for visual verification.
    """
    start_t = time.monotonic()
    current_state = "CENTER"

    output_fn("\n" + "=" * 65)
    output_fn("       VOXGAZE LIVE X-READOUT (LOOK LEFT / RIGHT TO TEST)      ")
    output_fn(" Thresholds: LEFT enter < 0.35 (exit > 0.40) | RIGHT enter > 0.65 (exit < 0.60)")
    output_fn("=" * 65)

    while (time.monotonic() - start_t) < seconds:
        elapsed = time.monotonic() - start_t
        if worker is not None:
            sample = worker.latest_sample
            fps = worker.fps
            lat = worker.per_frame_latency
        else:
            sample = None
            fps = 0.0
            lat = 0.0

        if sample is not None and sample.valid:
            current_state = classify_x_hysteresis(sample.x, current_state)
            status_tag = f"[{current_state:^6}]"
            bar_pos = int(sample.x * 30)
            bar = [" "] * 31
            bar[min(30, max(0, bar_pos))] = "|"
            bar_str = "[" + "".join(bar) + "]"
            output_fn(
                f"{elapsed:5.1f}s | x={sample.x:.3f} y={sample.y:.3f} | {status_tag} {bar_str} | fps={fps:4.1f} | lat={lat:4.1f}ms"
            )
        elif sample is not None and sample.face_detected:
            output_fn(f"{elapsed:5.1f}s | x=  ---  y=  ---  | [ABSTAIN] (unmapped)                | fps={fps:4.1f} | lat={lat:4.1f}ms")
        else:
            output_fn(f"{elapsed:5.1f}s | x=  ---  y=  ---  | [NO_FACE] (look at camera)          | fps={fps:4.1f}")

        sleep_fn(interval_s)

    output_fn("=" * 65 + "\n")
