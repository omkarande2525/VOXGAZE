"""
Unit tests for VoxGaze Demo Perception:
Live Perception Worker, Visual Fixation Calibration, and Dual-Threshold X-Readout.

Tests:
1. Gaze feature extraction from frames (face vs no-face).
2. LiveGazeWorker lifecycle, sample publishing, FPS, and latency calculation.
3. No-face yields valid=False.
4. Clean stop() terminates thread and releases camera.
5. Calibration uses the exact 18-trial balanced sequence (BLOCK_1 + BLOCK_2).
6. ESC aborts calibration safely without fitting and without crashing.
7. Zero image or video files are written to disk.
8. Dual-threshold hysteresis classification and print_x_readout formatting.
"""

from pathlib import Path
import time
from typing import Any, List, Optional, Tuple
import numpy as np
import pytest

from backend.app.aac.demo_perception import (
    BLOCK_1_TARGETS,
    BLOCK_2_TARGETS,
    CALIBRATION_SEQUENCE,
    ExtractedFrameFeatures,
    GazeSample,
    LiveGazeWorker,
    classify_x_hysteresis,
    draw_target_canvas,
    extract_gaze_features_from_frame,
    print_x_readout,
    process_frame,
    run_calibration,
)
from backend.app.calibration.personalized_mapper import (
    CalibrationSample,
    FeatureSetType,
    MappingPrediction,
    PersonalizedGazeMapper,
)
from backend.app.perception.gaze_estimator_interface import BaseGazeEstimator, GazeModelOutput


# ---------------------------------------------------------------------------
# Test Mocks & Fixtures
# ---------------------------------------------------------------------------

class MockVideoCapture:
    """Mock OpenCV VideoCapture for unit tests."""
    def __init__(self, frame_shape=(720, 1280, 3)):
        self.frame = np.zeros(frame_shape, dtype=np.uint8)
        self.is_opened = True
        self.is_released = False
        self.aborted = False

    def isOpened(self) -> bool:
        return self.is_opened

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.is_opened or self.is_released:
            return False, None
        return True, self.frame.copy()

    def set(self, prop: int, val: float) -> bool:
        return True

    def release(self) -> None:
        self.is_released = True
        self.is_opened = False


class MockLandmark:
    def __init__(self, x: float = 0.5, y: float = 0.5, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


class MockDetectionResult:
    def __init__(self, landmarks: List[List[MockLandmark]]):
        self.face_landmarks = landmarks


class MockFaceDetector:
    """Mock MediaPipe FaceLandmarker returning plausible facial landmarks."""
    def __init__(self, has_face: bool = True):
        self.has_face = has_face

    def detect_for_video(self, mp_img: Any, ts_ms: int) -> MockDetectionResult:
        if not self.has_face:
            return MockDetectionResult([])

        # 478 landmarks with wide face extent for valid 1.20 face crop
        landmarks = [MockLandmark(0.50, 0.50) for _ in range(478)]

        # Face bounds for extract_face_crop: width ~ 512 px, height ~ 432 px
        landmarks[10] = MockLandmark(0.50, 0.20)   # Forehead top
        landmarks[152] = MockLandmark(0.50, 0.80)  # Chin bottom
        landmarks[234] = MockLandmark(0.30, 0.50)  # Left cheek
        landmarks[454] = MockLandmark(0.70, 0.50)  # Right cheek

        # Right eye: outer 33, inner 133, iris 468, upper 159, lower 145
        landmarks[33] = MockLandmark(0.38, 0.45)
        landmarks[133] = MockLandmark(0.46, 0.45)
        landmarks[468] = MockLandmark(0.42, 0.45)
        landmarks[159] = MockLandmark(0.42, 0.42)
        landmarks[145] = MockLandmark(0.42, 0.48)

        # Left eye: inner 362, outer 263, iris 473, upper 386, lower 374
        landmarks[362] = MockLandmark(0.54, 0.45)
        landmarks[263] = MockLandmark(0.62, 0.45)
        landmarks[473] = MockLandmark(0.58, 0.45)
        landmarks[386] = MockLandmark(0.58, 0.42)
        landmarks[374] = MockLandmark(0.58, 0.48)

        return MockDetectionResult([landmarks])


class MockGazeEstimator(BaseGazeEstimator):
    """Mock MobileOne S0 Gaze Estimator."""
    def __init__(self, is_valid: bool = True, yaw_deg: float = -6.0, pitch_deg: float = -12.0):
        self.is_valid = is_valid
        self.yaw_deg = yaw_deg
        self.pitch_deg = pitch_deg

    @property
    def model_name(self) -> str:
        return "mock_mobileone"

    def preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        return np.zeros((1, 3, 224, 224), dtype=np.float32)

    def decode_logits(self, yaw_logits: np.ndarray, pitch_logits: np.ndarray) -> Tuple[float, float, float, float]:
        return (-0.2, -0.1, self.pitch_deg, self.yaw_deg)

    def predict(self, face_bgr: np.ndarray) -> GazeModelOutput:
        return GazeModelOutput(
            pitch_rad=-0.2,
            yaw_rad=-0.1,
            pitch_deg=self.pitch_deg,
            yaw_deg=self.yaw_deg,
            inference_latency_ms=4.2,
            preprocessing_latency_ms=0.8,
            model_name="mock_mobileone",
            is_valid=self.is_valid,
        )


class MockGazeMapper:
    """Mock PersonalizedGazeMapper returning fixed valid predictions."""
    def __init__(self, screen_x: float = 0.52, screen_y: float = 0.48, is_valid: bool = True):
        self.screen_x = screen_x
        self.screen_y = screen_y
        self.is_valid = is_valid
        self.is_fitted = True

    def predict(self, sample: CalibrationSample) -> MappingPrediction:
        return MappingPrediction(
            screen_x=self.screen_x,
            screen_y=self.screen_y,
            is_valid=self.is_valid,
            mapping_latency_us=4.5,
            predicted_direction="CENTER",
        )


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

def test_extract_gaze_features_from_frame_valid():
    """Verify feature extraction when face is detected, landmarks are valid, and model succeeds."""
    detector = MockFaceDetector(has_face=True)
    estimator = MockGazeEstimator(is_valid=True, yaw_deg=-7.5, pitch_deg=-11.0)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    feats = extract_gaze_features_from_frame(frame, detector, estimator)
    assert isinstance(feats, ExtractedFrameFeatures)
    assert feats.face_detected is True
    assert feats.face_crop_valid is True
    assert feats.estimator_valid is True
    assert feats.yaw_deg == -7.5
    assert feats.pitch_deg == -11.0
    assert feats.right_eye_h is not None
    assert feats.right_eye_v is not None
    assert feats.right_eye_ear is not None


def test_extract_gaze_features_from_frame_no_face():
    """Verify that when detector finds no face, features honestly report face_detected=False."""
    detector = MockFaceDetector(has_face=False)
    estimator = MockGazeEstimator()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    feats = extract_gaze_features_from_frame(frame, detector, estimator)
    assert feats.face_detected is False
    assert feats.face_crop_valid is False
    assert feats.estimator_valid is False
    assert feats.yaw_deg is None


def test_live_gaze_worker_lifecycle_and_samples_published():
    """Verify LiveGazeWorker starts daemon thread, publishes latest sample, deque history, and FPS."""
    cap = MockVideoCapture()
    detector = MockFaceDetector(has_face=True)
    estimator = MockGazeEstimator(is_valid=True)
    mapper = MockGazeMapper(screen_x=0.55, screen_y=0.45, is_valid=True)

    worker = LiveGazeWorker(
        cap=cap,
        mapper=mapper,
        estimator=estimator,
        detector=detector,
        max_deque_len=15,
    )

    try:
        worker.start()
        # Allow thread to capture and process several frames
        time.sleep(0.08)

        sample = worker.latest_sample
        assert sample is not None
        assert isinstance(sample, GazeSample)
        assert sample.valid is True
        assert sample.face_detected is True
        assert sample.x == pytest.approx(0.55, abs=0.01)
        assert sample.y == pytest.approx(0.45, abs=0.01)
        assert sample.latency_ms > 0.0

        # Deque history
        assert len(worker.samples) > 0
        assert worker.samples[-1] == sample
        assert worker.fps >= 0.0
        assert worker.measured_fps >= 0.0
        assert worker.per_frame_latency > 0.0
        assert worker.per_frame_latency_ms > 0.0

        # Tuple unpacking test
        ts, mx, my, val, face = sample
        assert val is True
        assert face is True
        assert mx == pytest.approx(0.55, abs=0.01)
        assert my == pytest.approx(0.45, abs=0.01)

    finally:
        worker.stop()


def test_live_gaze_worker_no_face_publishes_invalid():
    """Verify LiveGazeWorker honestly publishes valid=False when no face is detected."""
    cap = MockVideoCapture()
    detector = MockFaceDetector(has_face=False)
    estimator = MockGazeEstimator()
    mapper = MockGazeMapper()

    worker = LiveGazeWorker(cap=cap, mapper=mapper, estimator=estimator, detector=detector)

    try:
        worker.start()
        time.sleep(0.05)

        sample = worker.latest_sample
        assert sample is not None
        assert sample.face_detected is False
        assert sample.valid is False

    finally:
        worker.stop()


def test_live_gaze_worker_clean_stop_releases_camera():
    """Verify that calling stop() terminates the worker thread cleanly and releases the camera."""
    cap = MockVideoCapture()
    mapper = MockGazeMapper()
    worker = LiveGazeWorker(
        cap=cap,
        mapper=mapper,
        estimator=MockGazeEstimator(),
        detector=MockFaceDetector(),
    )

    worker.start()
    assert worker._thread is not None
    assert worker._thread.is_alive()

    worker.stop()
    assert worker._thread is None
    assert cap.is_released is True
    assert cap.is_opened is False


def test_calibration_sequence_18_trials():
    """
    Verify calibration uses the balanced 18-trial sequence (BLOCK_1 + BLOCK_2).
    Checks that all 18 targets are visited in exact order and a fitted mapper is returned.
    """
    assert len(BLOCK_1_TARGETS) == 9
    assert len(BLOCK_2_TARGETS) == 9
    assert len(CALIBRATION_SEQUENCE) == 18
    assert CALIBRATION_SEQUENCE == BLOCK_1_TARGETS + BLOCK_2_TARGETS

    cap = MockVideoCapture()
    detector = MockFaceDetector(has_face=True)
    estimator = MockGazeEstimator(is_valid=True)

    visited_trials: List[Tuple[int, int, str]] = []

    def _on_trial(block: int, cond: int, target: str):
        visited_trials.append((block, cond, target))

    # Run calibration headless with fast durations (0.002s settling, 0.005s recording)
    mapper, returned_cap = run_calibration(
        cap=cap,
        detector=detector,
        estimator=estimator,
        settling_duration_s=0.002,
        recording_duration_s=0.005,
        headless=True,
        on_trial_start=_on_trial,
    )

    # 1. Exact 18 trials visited
    assert len(visited_trials) == 18
    expected_sequence = [
        (1, idx + 1, target) for idx, target in enumerate(BLOCK_1_TARGETS)
    ] + [
        (2, idx + 1, target) for idx, target in enumerate(BLOCK_2_TARGETS)
    ]
    assert visited_trials == expected_sequence

    # 2. Fitted PersonalizedGazeMapper returned
    assert mapper is not None
    assert isinstance(mapper, PersonalizedGazeMapper)
    assert mapper.is_fitted is True

    # 3. Open camera handle returned for reuse
    assert returned_cap is cap
    assert returned_cap.isOpened() is True


def test_calibration_esc_aborts_safely():
    """Verify that an abort signal (ESC) exits cleanly without fitting mapper and without crashing."""
    cap = MockVideoCapture()
    cap.aborted = True  # Signal abort immediately

    visited_trials: List[Tuple[int, int, str]] = []

    mapper, returned_cap = run_calibration(
        cap=cap,
        detector=MockFaceDetector(),
        estimator=MockGazeEstimator(),
        settling_duration_s=0.001,
        recording_duration_s=0.001,
        headless=True,
        on_trial_start=lambda b, c, t: visited_trials.append((b, c, t)),
    )

    # Must return None for mapper, returning camera handle safely
    assert mapper is None
    assert returned_cap is cap


def test_calibration_and_worker_never_save_images_or_frames_to_disk(tmp_path):
    """
    SAFETY INVARIANT: Verify that running calibration and worker execution never
    writes any webcam frames or image/video files to disk.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    image_extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.mp4", "*.avi")

    def _count_image_files() -> int:
        count = 0
        for ext in image_extensions:
            count += len(list(repo_root.glob(f"**/{ext}")))
        return count

    count_before = _count_image_files()

    # Run calibration
    cap = MockVideoCapture()
    mapper, _ = run_calibration(
        cap=cap,
        detector=MockFaceDetector(),
        estimator=MockGazeEstimator(),
        settling_duration_s=0.001,
        recording_duration_s=0.003,
        headless=True,
    )

    # Run worker
    worker_cap = MockVideoCapture()
    worker = LiveGazeWorker(
        cap=worker_cap,
        mapper=mapper,
        estimator=MockGazeEstimator(),
        detector=MockFaceDetector(),
    )
    worker.start()
    time.sleep(0.04)
    worker.stop()

    count_after = _count_image_files()
    assert count_after == count_before, "Violated Invariant: Image or video files were written to disk."


def test_classify_x_hysteresis():
    """
    Verify dual-threshold hysteresis classification:
    - LEFT: enter < 0.35, exit > 0.40
    - RIGHT: enter > 0.65, exit < 0.60
    """
    # 1. From CENTER
    assert classify_x_hysteresis(0.50, "CENTER") == "CENTER"
    assert classify_x_hysteresis(0.34, "CENTER") == "LEFT"     # < 0.35 enters LEFT
    assert classify_x_hysteresis(0.66, "CENTER") == "RIGHT"    # > 0.65 enters RIGHT
    assert classify_x_hysteresis(0.38, "CENTER") == "CENTER"   # in deadzone

    # 2. From LEFT (exit threshold is 0.40)
    assert classify_x_hysteresis(0.30, "LEFT") == "LEFT"       # stays LEFT
    assert classify_x_hysteresis(0.38, "LEFT") == "LEFT"       # <= 0.40 stays LEFT
    assert classify_x_hysteresis(0.41, "LEFT") == "CENTER"     # > 0.40 exits to CENTER
    assert classify_x_hysteresis(0.70, "LEFT") == "RIGHT"      # jump to RIGHT

    # 3. From RIGHT (exit threshold is 0.60)
    assert classify_x_hysteresis(0.75, "RIGHT") == "RIGHT"     # stays RIGHT
    assert classify_x_hysteresis(0.62, "RIGHT") == "RIGHT"     # >= 0.60 stays RIGHT
    assert classify_x_hysteresis(0.58, "RIGHT") == "CENTER"    # < 0.60 exits to CENTER
    assert classify_x_hysteresis(0.20, "RIGHT") == "LEFT"      # jump to LEFT


def test_print_x_readout_formatting():
    """Verify print_x_readout executes and outputs properly formatted rows."""
    cap = MockVideoCapture()
    mapper = MockGazeMapper(screen_x=0.25, screen_y=0.50)  # Should trigger LEFT
    worker = LiveGazeWorker(
        cap=cap,
        mapper=mapper,
        estimator=MockGazeEstimator(),
        detector=MockFaceDetector(),
    )
    worker.start()
    time.sleep(0.04)

    printed_lines: List[str] = []

    try:
        # Run print_x_readout with mock output_fn and mock sleep
        print_x_readout(
            seconds=0.02,
            worker=worker,
            interval_s=0.01,
            output_fn=printed_lines.append,
            sleep_fn=lambda s: None,
        )
    finally:
        worker.stop()

    assert len(printed_lines) >= 3
    # Header check
    assert any("LIVE X-READOUT" in line for line in printed_lines)
    # Output rows check
    assert any("LEFT" in line or "CENTER" in line for line in printed_lines)


def test_draw_target_canvas_renders_without_crash():
    """Verify canvas drawing helper for visual calibration runs cleanly."""
    canvas_settling = draw_target_canvas(
        canvas_w=1280,
        canvas_h=720,
        target_name="LEFT",
        phase="SETTLING",
        time_left_s=1.2,
        block_num=1,
        trial_num=2,
        total_trials=9,
        camera_preview=np.zeros((720, 1280, 3), dtype=np.uint8),
        face_detected=True,
    )
    assert canvas_settling.shape == (720, 1280, 3)

    canvas_recording = draw_target_canvas(
        canvas_w=1280,
        canvas_h=720,
        target_name="RIGHT",
        phase="RECORDING",
        time_left_s=2.8,
        block_num=2,
        trial_num=5,
        total_trials=9,
        face_detected=False,
    )
    assert canvas_recording.shape == (720, 1280, 3)


def test_process_frame_shared_per_frame_function_valid():
    """Verify the shared process_frame function extracts features, builds calib_sample, and predicts GazeSample."""
    raw_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    detector = MockFaceDetector(has_face=True)
    estimator = MockGazeEstimator(yaw_deg=-5.0, pitch_deg=2.0, is_valid=True)

    mapper = PersonalizedGazeMapper(feature_set=FeatureSetType.HYBRID, alpha=1.0)
    # Fit mapper with dummy samples
    dummy_samples = [
        CalibrationSample(0.50, 0.50, "CENTER", -5.0 + i * 0.1, 2.0, 0.5, 0.5, 0.5, 0.5, 30.0)
        for i in range(10)
    ]
    assert mapper.fit(dummy_samples) is True

    feats, gaze_sample, calib_sample = process_frame(
        raw_frame=raw_frame,
        detector=detector,
        estimator=estimator,
        mapper=mapper,
        target_direction="LEFT",
        target_x=0.15,
        target_y=0.50,
        condition_name="B1_2_LEFT",
        trial_index=2,
    )

    # Features check
    assert feats.face_detected is True
    assert feats.face_crop_valid is True
    assert feats.estimator_valid is True
    assert feats.yaw_deg == -5.0
    assert feats.pitch_deg == 2.0

    # Calibration sample check
    assert calib_sample is not None
    assert calib_sample.target_direction == "LEFT"
    assert calib_sample.target_x == 0.15
    assert calib_sample.target_y == 0.50
    assert calib_sample.condition_name == "B1_2_LEFT"
    assert calib_sample.trial_index == 2

    # Gaze sample check
    assert gaze_sample.valid is True
    assert gaze_sample.face_detected is True
    assert 0.0 <= gaze_sample.x <= 1.0
    assert 0.0 <= gaze_sample.y <= 1.0
    assert gaze_sample.latency_ms > 0.0


def test_process_frame_shared_per_frame_function_no_face():
    """Verify process_frame handles no face honestly without leaking coordinates or calib samples."""
    raw_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    detector = MockFaceDetector(has_face=False)
    estimator = MockGazeEstimator()

    feats, gaze_sample, calib_sample = process_frame(
        raw_frame=raw_frame,
        detector=detector,
        estimator=estimator,
        mapper=None,
    )

    assert feats.face_detected is False
    assert feats.face_crop_valid is False
    assert calib_sample is None
    assert gaze_sample.valid is False
    assert gaze_sample.face_detected is False
    assert gaze_sample.x == 0.50
    assert gaze_sample.y == 0.50


def test_process_frame_strictly_increasing_timestamps():
    """Verify process_frame guarantees strictly increasing timestamps on the detector instance."""
    raw_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    detector = MockFaceDetector(has_face=True)
    estimator = MockGazeEstimator()

    timestamps = []
    # Intercept detect_for_video calls to log timestamps
    orig_detect = detector.detect_for_video
    def _detect(img, ts):
        timestamps.append(ts)
        return orig_detect(img, ts)
    detector.detect_for_video = _detect

    for _ in range(10):
        process_frame(raw_frame, detector, estimator)

    assert len(timestamps) == 10
    # Strictly increasing: ts[i] > ts[i-1]
    for i in range(1, len(timestamps)):
        assert timestamps[i] > timestamps[i - 1], f"Timestamps not strictly increasing: {timestamps}"


def test_calibration_rejects_insufficient_valid_frame_fraction_per_trial():
    """Verify calibration fails when any trial fails the minimum valid-frame fraction requirement."""
    cap = MockVideoCapture()

    class AlternatingFaceDetector:
        """Returns face on first trial, no face on subsequent trials."""
        def __init__(self):
            self.call_count = 0

        def detect_for_video(self, mp_img, ts_ms):
            self.call_count += 1
            # Only first trial has face, all others fail
            if self.call_count <= 2:
                landmarks = [MockLandmark(0.50, 0.50) for _ in range(478)]
                landmarks[10] = MockLandmark(0.50, 0.20)
                landmarks[152] = MockLandmark(0.50, 0.80)
                landmarks[234] = MockLandmark(0.30, 0.50)
                landmarks[454] = MockLandmark(0.70, 0.50)
                return MockDetectionResult([landmarks])
            return MockDetectionResult([])

    detector = AlternatingFaceDetector()
    estimator = MockGazeEstimator(is_valid=True)

    mapper, returned_cap = run_calibration(
        cap=cap,
        detector=detector,
        estimator=estimator,
        settling_duration_s=0.002,
        recording_duration_s=0.005,
        headless=True,
        min_valid_fraction_per_trial=0.5,
    )

    # Must reject calibration because trials after the first had 0% valid frames
    assert mapper is None
    assert returned_cap is cap


def test_calibration_prints_per_trial_counts_total_fit_and_residual(capsys):
    """Verify calibration prints valid-frame count per trial, total fit samples, and fit residual."""
    cap = MockVideoCapture()
    detector = MockFaceDetector(has_face=True)
    estimator = MockGazeEstimator(is_valid=True)

    mapper, _ = run_calibration(
        cap=cap,
        detector=detector,
        estimator=estimator,
        settling_duration_s=0.002,
        recording_duration_s=0.005,
        headless=True,
        min_valid_fraction_per_trial=0.5,
    )

    assert mapper is not None
    assert mapper.is_fitted is True

    captured = capsys.readouterr().out
    # Check valid-frame count per trial printed
    assert "[Calibration] Trial B1_1_CENTER:" in captured
    assert "valid frames" in captured
    # Check total used for fit printed
    assert "[Calibration] Total valid frames used for fit:" in captured
    # Check fit residual printed
    assert "[Calibration] Fit residual (RMSE):" in captured


def test_live_gaze_worker_reuses_detector_and_estimator_from_cap():
    """Verify LiveGazeWorker automatically reuses detector and estimator attached to cap from calibration."""
    cap = MockVideoCapture()
    detector = MockFaceDetector(has_face=True)
    estimator = MockGazeEstimator(is_valid=True)

    mapper, returned_cap = run_calibration(
        cap=cap,
        detector=detector,
        estimator=estimator,
        settling_duration_s=0.002,
        recording_duration_s=0.005,
        headless=True,
    )
    assert mapper is not None

    worker = LiveGazeWorker(cap=returned_cap, mapper=mapper)
    # Must reuse the same detector and estimator without re-creating
    assert worker.detector is detector
    assert worker.estimator is estimator
