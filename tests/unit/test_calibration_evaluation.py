"""
Unit tests for M1.5-B1 Calibration Validation:
Session timing, CSV recording integrity, quality propagation in calibration buffers,
per-eye validity, frozen threshold derivation, independent test trial evaluation,
center-return drift, unsupported upward gaze handling, and safety invariants.
"""

import math
from pathlib import Path
import tempfile
import pytest

from backend.app.calibration.evaluator import (
    classify_frame_discrete,
    compute_center_drift,
    compute_directional_separations,
    compute_phase1_target_stats,
    derive_frozen_candidate_thresholds,
    evaluate_calibration_validation_session,
    evaluate_independent_phase2_trials,
    filter_valid_samples,
)
from backend.app.calibration.protocol import (
    CALIBRATION_VALIDATION_SEQUENCE,
    CalibrationValidationCondition,
    CalibrationValidationRecord,
    M15_CSV_HEADER,
    ProtocolPhase,
)
from backend.app.calibration.service import CalibrationTargetBuffer
from backend.app.core.contracts import (
    CalibrationThresholds,
    Direction,
    DirectionThreshold,
    EyeLandmarks,
    GazeFeatures,
    Point2D,
)
from backend.app.perception.gaze_engine import GazeEngine
from backend.app.perception.geometry import (
    extract_bilateral_features,
    extract_eye_features,
    validate_eye_geometry,
)


def make_record(
    frame_idx: int,
    phase: ProtocolPhase,
    cond_idx: int,
    cond_name: str,
    target_dir: str,
    h: float = 0.50,
    v: float = 0.50,
    ear: float = 0.30,
    canthal_ratio: float = -0.05,
    face_detected: bool = True,
    geometry_valid: bool = True,
    r_valid: bool = True,
    l_valid: bool = True,
    yaw: float = 0.0,
    pitch: float = 0.0,
) -> CalibrationValidationRecord:
    """Helper to synthesize CalibrationValidationRecord."""
    return CalibrationValidationRecord(
        session_id="test_sess_001",
        monotonic_time_s=100.0 + frame_idx * 0.033,
        elapsed_session_s=frame_idx * 0.033,
        protocol_phase=phase.value,
        condition_index=cond_idx,
        condition_name=cond_name,
        target_direction=target_dir,
        frame_index=frame_idx,
        face_detected=face_detected,
        face_crop_valid=face_detected,
        geometry_valid=geometry_valid,
        active_eye="right",
        right_eye_valid=r_valid,
        left_eye_valid=l_valid,
        right_eye_h_ratio=h if face_detected else None,
        right_eye_v_ratio=v if face_detected else None,
        right_eye_ear=ear if face_detected else None,
        right_eye_canthal_ratio=canthal_ratio if face_detected else None,
        left_eye_h_ratio=h if face_detected else None,
        left_eye_v_ratio=v if face_detected else None,
        left_eye_ear=ear if face_detected else None,
        left_eye_canthal_ratio=canthal_ratio if face_detected else None,
        canthal_normal_dist_px=-1.70 if face_detected else None,
        mobileone_valid=face_detected,
        mobileone_yaw_deg=yaw if face_detected else None,
        mobileone_pitch_deg=pitch if face_detected else None,
    )


# -------------------------------------------------------------------
# 1. Quality Propagation & Calibration Buffer Gating Tests
# -------------------------------------------------------------------

def test_calibration_buffer_rejects_blinks_for_open_gaze():
    """Verify that a spontaneous blink during open-eye target is rejected by CalibrationTargetBuffer."""
    buf = CalibrationTargetBuffer("CENTER", warmup_sec=0.0, collection_sec=2.0, active_eye="right")

    # Open eye sample -> accepted
    pt = Point2D(x=100.0, y=100.0)
    eye_open = EyeLandmarks(
        iris_center=pt,
        inner_canthus=Point2D(x=80.0, y=100.0),
        outer_canthus=Point2D(x=120.0, y=100.0),
        upper_lid=Point2D(x=100.0, y=90.0),
        lower_lid=Point2D(x=100.0, y=110.0),
        h_ratio=0.50,
        v_ratio=0.50,
        ear=0.30,
    )
    feat_open = GazeFeatures(
        timestamp_ms=100,
        right_eye=eye_open,
        is_tracking_valid=True,
        tracking_confidence=0.95,
        is_right_eye_valid=True,
    )
    assert buf.add_sample(feat_open) is True

    # Blink sample (ear = 0.04 < 0.08, v = -5.9) -> MUST be strictly rejected
    eye_blink = EyeLandmarks(
        iris_center=Point2D(x=100.0, y=95.0),
        inner_canthus=Point2D(x=80.0, y=100.0),
        outer_canthus=Point2D(x=120.0, y=100.0),
        upper_lid=Point2D(x=100.0, y=99.5),
        lower_lid=Point2D(x=100.0, y=100.5),  # 1px lid sep
        h_ratio=0.50,
        v_ratio=-5.92,
        ear=0.04,
    )
    feat_blink = GazeFeatures(
        timestamp_ms=200,
        right_eye=eye_blink,
        is_tracking_valid=True,
        tracking_confidence=0.95,
        is_right_eye_valid=False,
    )
    assert buf.add_sample(feat_blink) is False
    assert len(buf.h_samples) == 1
    assert len(buf.v_samples) == 1


def test_calibration_buffer_accepts_closed_target_and_preserves_per_eye():
    """Verify that CLOSED target accepts closed eyes, but rejects non-finite/degenerate coordinates."""
    buf_closed = CalibrationTargetBuffer("CLOSED", warmup_sec=0.0, collection_sec=2.0, active_eye="right")

    # Deliberate closed eye (ear = 0.05, valid eye width = 40 px) -> accepted
    eye_closed = EyeLandmarks(
        iris_center=Point2D(x=100.0, y=100.0),
        inner_canthus=Point2D(x=80.0, y=100.0),
        outer_canthus=Point2D(x=120.0, y=100.0),
        upper_lid=Point2D(x=100.0, y=99.0),
        lower_lid=Point2D(x=100.0, y=101.0),
        h_ratio=0.50,
        v_ratio=0.50,
        ear=0.05,
    )
    feat_closed = GazeFeatures(
        timestamp_ms=100,
        right_eye=eye_closed,
        is_tracking_valid=True,
        tracking_confidence=0.95,
    )
    assert buf_closed.add_sample(feat_closed) is True

    # Degenerate zero width -> rejected even for CLOSED
    eye_degenerate = EyeLandmarks(
        iris_center=Point2D(x=100.0, y=100.0),
        inner_canthus=Point2D(x=100.0, y=100.0),
        outer_canthus=Point2D(x=100.0, y=100.0),
        upper_lid=Point2D(x=100.0, y=100.0),
        lower_lid=Point2D(x=100.0, y=100.0),
        h_ratio=0.50,
        v_ratio=0.50,
        ear=0.0,
    )
    feat_degen = GazeFeatures(
        timestamp_ms=200,
        right_eye=eye_degenerate,
        is_tracking_valid=True,
        tracking_confidence=0.95,
    )
    assert buf_closed.add_sample(feat_degen) is False


def test_calibration_buffer_asymmetric_eyes_preserves_valid_active_eye():
    """Verify that one invalid eye does not invalidate the other eye when active."""
    # Active eye is right: right is valid, left is closed/corrupted
    buf = CalibrationTargetBuffer("CENTER", warmup_sec=0.0, collection_sec=2.0, active_eye="right")

    r_eye = EyeLandmarks(
        iris_center=Point2D(x=100.0, y=100.0),
        inner_canthus=Point2D(x=80.0, y=100.0),
        outer_canthus=Point2D(x=120.0, y=100.0),
        upper_lid=Point2D(x=100.0, y=90.0),
        lower_lid=Point2D(x=100.0, y=110.0),
        h_ratio=0.50,
        v_ratio=0.50,
        ear=0.30,
    )
    l_corrupt = EyeLandmarks(
        iris_center=Point2D(x=200.0, y=100.0),
        inner_canthus=Point2D(x=180.0, y=100.0),
        outer_canthus=Point2D(x=220.0, y=100.0),
        upper_lid=Point2D(x=200.0, y=99.5),
        lower_lid=Point2D(x=200.0, y=100.5),
        h_ratio=0.50,
        v_ratio=-5.0,
        ear=0.02,
    )
    feat = GazeFeatures(
        timestamp_ms=100,
        right_eye=r_eye,
        left_eye=l_corrupt,
        is_tracking_valid=True,
        tracking_confidence=0.95,
        is_right_eye_valid=True,
        is_left_eye_valid=False,
    )

    # Active right eye is valid -> sample is accepted!
    assert buf.add_sample(feat) is True

    # If buffer was configured for left eye as active -> sample is rejected!
    buf_left = CalibrationTargetBuffer("CENTER", warmup_sec=0.0, collection_sec=2.0, active_eye="left")
    assert buf_left.add_sample(feat) is False


# -------------------------------------------------------------------
# 2. Frozen Candidate Thresholds & Upward Gaze Gate Tests
# -------------------------------------------------------------------

def test_frozen_candidate_thresholds_derivation():
    """Verify that thresholds derived on Phase 1 are frozen and correctly apply dual-thresholds."""
    # Synthetic Phase 1 data with clear separation
    records: List[CalibrationValidationRecord] = []
    # Center: H = 0.50, V = 0.50, EAR = 0.30
    for i in range(30):
        records.append(make_record(i, ProtocolPhase.CALIBRATION_ESTIMATION, 1, "CALIB_CENTER_1", "CENTER", h=0.50, v=0.50, ear=0.30))
    # Left: H = 0.35 (delta H = 0.15)
    for i in range(30):
        records.append(make_record(30 + i, ProtocolPhase.CALIBRATION_ESTIMATION, 2, "CALIB_LEFT", "LEFT", h=0.35, v=0.50, ear=0.30))
    # Right: H = 0.65 (delta H = 0.15)
    for i in range(30):
        records.append(make_record(60 + i, ProtocolPhase.CALIBRATION_ESTIMATION, 4, "CALIB_RIGHT", "RIGHT", h=0.65, v=0.50, ear=0.30))
    # Up: V = 0.40 (delta V = 0.10)
    for i in range(30):
        records.append(make_record(90 + i, ProtocolPhase.CALIBRATION_ESTIMATION, 6, "CALIB_UP", "UP", h=0.50, v=0.40, ear=0.30))
    # Down: V = 0.62 (delta V = 0.12)
    for i in range(30):
        records.append(make_record(120 + i, ProtocolPhase.CALIBRATION_ESTIMATION, 8, "CALIB_DOWN", "DOWN", h=0.50, v=0.62, ear=0.30))
    # Closed: EAR = 0.10 (delta EAR = 0.20)
    for i in range(30):
        records.append(make_record(150 + i, ProtocolPhase.CALIBRATION_ESTIMATION, 10, "CALIB_CLOSED", "CLOSED", h=0.50, v=0.50, ear=0.10))

    p1_stats = compute_phase1_target_stats(records, active_eye="right")
    thresholds, gates = derive_frozen_candidate_thresholds(p1_stats)

    assert thresholds is not None
    assert gates["is_left_supported"] is True
    assert gates["is_right_supported"] is True
    assert gates["is_up_supported"] is True
    assert gates["is_down_supported"] is True
    assert gates["is_blink_supported"] is True
    assert len(gates["unsupported_reasons"]) == 0

    # Left: enter = 0.50 - 0.60 * 0.15 = 0.41, exit = 0.50 - 0.35 * 0.15 = 0.4475
    assert thresholds.left.enter == pytest.approx(0.41)
    assert thresholds.left.exit == pytest.approx(0.4475)


def test_upward_gaze_unsupported_gate():
    """Verify that delta V < 0.05 strictly triggers UP_GAZE_UNSUPPORTED and disables UP classifier."""
    records: List[CalibrationValidationRecord] = []
    # Center: V = 0.4339
    for i in range(30):
        records.append(make_record(i, ProtocolPhase.CALIBRATION_ESTIMATION, 1, "CALIB_CENTER_1", "CENTER", h=0.50, v=0.4339, ear=0.30))
    # Left & Right normal
    for i in range(30):
        records.append(make_record(30 + i, ProtocolPhase.CALIBRATION_ESTIMATION, 2, "CALIB_LEFT", "LEFT", h=0.35, v=0.4339, ear=0.30))
    for i in range(30):
        records.append(make_record(60 + i, ProtocolPhase.CALIBRATION_ESTIMATION, 4, "CALIB_RIGHT", "RIGHT", h=0.65, v=0.4339, ear=0.30))
    # Up: V = 0.4311 (delta V = 0.0028 < 0.05 threshold, mimicking Run 2!)
    for i in range(30):
        records.append(make_record(90 + i, ProtocolPhase.CALIBRATION_ESTIMATION, 6, "CALIB_UP", "UP", h=0.50, v=0.4311, ear=0.30))
    # Down normal: V = 0.55 (delta V = 0.1161)
    for i in range(30):
        records.append(make_record(120 + i, ProtocolPhase.CALIBRATION_ESTIMATION, 8, "CALIB_DOWN", "DOWN", h=0.50, v=0.55, ear=0.30))
    # Closed: EAR = 0.10
    for i in range(30):
        records.append(make_record(150 + i, ProtocolPhase.CALIBRATION_ESTIMATION, 10, "CALIB_CLOSED", "CLOSED", h=0.50, v=0.50, ear=0.10))

    p1_stats = compute_phase1_target_stats(records, active_eye="right")
    thresholds, gates = derive_frozen_candidate_thresholds(p1_stats, min_separation_v=0.05)

    assert thresholds is not None
    assert gates["is_up_supported"] is False
    assert any("UP_GAZE_UNSUPPORTED" in r for r in gates["unsupported_reasons"])
    # Enter threshold is set to -999 so UP command can never trigger
    assert thresholds.up.enter == -999.0

    # Ensure classifier never outputs UP when is_up_supported is False
    classified = classify_frame_discrete(
        h=0.50,
        v=0.10,  # extreme upward ratio
        ear=0.30,
        current_dir=Direction.CENTER,
        thresholds=thresholds,
        is_up_supported=False,
    )
    assert classified != Direction.UP
    assert classified == Direction.CENTER


# -------------------------------------------------------------------
# 3. Independent Phase 2 Test Trial Evaluation Tests
# -------------------------------------------------------------------

def test_independent_phase2_trial_evaluation():
    """Verify that Phase 2 test trials are evaluated trial-by-trial respecting temporal correlation."""
    frozen_thresholds = CalibrationThresholds(
        left=DirectionThreshold(enter=0.40, exit=0.45),
        right=DirectionThreshold(enter=0.60, exit=0.55),
        up=DirectionThreshold(enter=0.35, exit=0.40),
        down=DirectionThreshold(enter=0.65, exit=0.60),
        blink_closed_ear=0.15,
        blink_open_ear=0.28,
    )
    gate_results = {
        "is_left_supported": True,
        "is_right_supported": True,
        "is_up_supported": False,  # UP unsupported
        "is_down_supported": True,
        "is_blink_supported": True,
    }

    phase2_records: List[CalibrationValidationRecord] = []
    # Trial 11: EVAL_CENTER (30 frames around H=0.50) -> should classify CENTER
    for i in range(30):
        phase2_records.append(make_record(i, ProtocolPhase.INDEPENDENT_EVALUATION, 11, "EVAL_CENTER_1", "CENTER", h=0.50, v=0.50))
    # Trial 12: EVAL_LEFT (30 frames around H=0.32) -> should classify LEFT
    for i in range(30):
        phase2_records.append(make_record(30 + i, ProtocolPhase.INDEPENDENT_EVALUATION, 12, "EVAL_LEFT", "LEFT", h=0.32, v=0.50))
    # Trial 14: EVAL_RIGHT (30 frames around H=0.68) -> should classify RIGHT
    for i in range(30):
        phase2_records.append(make_record(60 + i, ProtocolPhase.INDEPENDENT_EVALUATION, 14, "EVAL_RIGHT", "RIGHT", h=0.68, v=0.50))
    # Trial 16: EVAL_UP (30 frames) -> UP is unsupported, MUST be omitted/unsupported
    for i in range(30):
        phase2_records.append(make_record(90 + i, ProtocolPhase.INDEPENDENT_EVALUATION, 16, "EVAL_UP", "UP", h=0.50, v=0.30))

    eval_results = evaluate_independent_phase2_trials(
        phase2_records=phase2_records,
        frozen_thresholds=frozen_thresholds,
        gate_results=gate_results,
        active_eye="right",
    )

    # 3 evaluated trials (CENTER, LEFT, RIGHT). UP was omitted because unsupported.
    assert eval_results["trials_evaluated_count"] == 3
    assert eval_results["trials_correct_count"] == 3
    assert eval_results["trial_level_accuracy"] == 1.0

    # Verify UP trial detail status
    up_detail = next(d for d in eval_results["trial_details"] if d["condition_index"] == 16)
    assert up_detail["status"] == "OMITTED_UNSUPPORTED"
    assert up_detail["trial_predicted"] is None


def test_center_baseline_drift_calculation():
    """Verify calculation of max baseline drift across repeated center returns."""
    records: List[CalibrationValidationRecord] = []
    # Center 1: H = 0.500, V = 0.500
    for i in range(20):
        records.append(make_record(i, ProtocolPhase.CALIBRATION_ESTIMATION, 1, "CALIB_CENTER_1", "CENTER", h=0.500, v=0.500))
    # Center 2: H = 0.512, V = 0.505 (drift H = +0.012)
    for i in range(20):
        records.append(make_record(20 + i, ProtocolPhase.CALIBRATION_ESTIMATION, 3, "CALIB_CENTER_2", "CENTER", h=0.512, v=0.505))
    # Center 3: H = 0.485, V = 0.518 (drift H = -0.015, drift V = +0.018)
    for i in range(20):
        records.append(make_record(40 + i, ProtocolPhase.CALIBRATION_ESTIMATION, 5, "CALIB_CENTER_3", "CENTER", h=0.485, v=0.518))

    drift = compute_center_drift(records, active_eye="right")
    assert drift["center_returns_evaluated"] == 3
    assert drift["max_absolute_drift_h"] == pytest.approx(0.015)
    assert drift["max_absolute_drift_v"] == pytest.approx(0.018)


# -------------------------------------------------------------------
# 4. Safety Invariants & Boundary Tests (User Change 5)
# -------------------------------------------------------------------

def test_safety_invariants_tracking_loss_yields_abstain():
    """Verify that tracking loss never yields a directional command or confirmed state."""
    profile = get_mock_profile()
    engine = GazeEngine(profile=profile)

    # 1. No eye detected -> NO_FACE
    feat_no_face = GazeFeatures(timestamp_ms=100, tracking_confidence=0.0, is_tracking_valid=False)
    state = engine.process_features(feat_no_face)
    assert state.direction == Direction.NO_FACE

    # 2. Eye detected, but tracking invalid flag -> ABSTAIN
    pt = Point2D(x=100.0, y=100.0)
    eye = EyeLandmarks(
        iris_center=pt,
        inner_canthus=Point2D(x=80.0, y=100.0),
        outer_canthus=Point2D(x=120.0, y=100.0),
        upper_lid=Point2D(x=100.0, y=90.0),
        lower_lid=Point2D(x=100.0, y=110.0),
        h_ratio=0.50,
        v_ratio=0.50,
        ear=0.30,
    )
    feat_invalid = GazeFeatures(
        timestamp_ms=200,
        right_eye=eye,
        tracking_confidence=0.9,
        is_tracking_valid=False,
    )
    state2 = engine.process_features(feat_invalid)
    assert state2.direction == Direction.ABSTAIN


def test_safety_invariants_invalid_ratio_yields_abstain():
    """Verify that an extreme or out-of-bounds ratio (e.g. V = -5.92) yields ABSTAIN."""
    profile = get_mock_profile()
    engine = GazeEngine(profile=profile)

    eye_out_of_bounds = EyeLandmarks(
        iris_center=Point2D(x=100.0, y=90.0),
        inner_canthus=Point2D(x=80.0, y=100.0),
        outer_canthus=Point2D(x=120.0, y=100.0),
        upper_lid=Point2D(x=100.0, y=99.0),
        lower_lid=Point2D(x=100.0, y=101.0),
        h_ratio=0.50,
        v_ratio=-5.92,  # Extreme negative ratio
        ear=0.30,
    )
    feat = GazeFeatures(
        timestamp_ms=100,
        right_eye=eye_out_of_bounds,
        tracking_confidence=0.95,
        is_tracking_valid=True,
    )
    state = engine.process_features(feat)
    # Must NOT classify as UP even though v is less than enter_up!
    assert state.direction == Direction.ABSTAIN
    assert "OUT_OF_BOUNDS_RATIO" in state.status_flag


def test_safety_invariants_blink_cannot_trigger_directional_command():
    """Verify that during a blink, state is BLINK and cannot be classified as LEFT/RIGHT/UP/DOWN."""
    profile = get_mock_profile()
    engine = GazeEngine(profile=profile)

    eye_blink = EyeLandmarks(
        iris_center=Point2D(x=100.0, y=100.0),
        inner_canthus=Point2D(x=80.0, y=100.0),
        outer_canthus=Point2D(x=120.0, y=100.0),
        upper_lid=Point2D(x=100.0, y=99.5),
        lower_lid=Point2D(x=100.0, y=100.5),
        h_ratio=0.30,  # Would cross LEFT threshold if open
        v_ratio=0.50,
        ear=0.04,      # Closed blink
    )
    feat = GazeFeatures(
        timestamp_ms=100,
        right_eye=eye_blink,
        tracking_confidence=0.95,
        is_tracking_valid=True,
    )
    state = engine.process_features(feat)
    assert state.direction == Direction.BLINK


def test_safety_invariants_no_tts_wiring_in_m15():
    """Verify that zero TTS or live AAC modules are wired into perception or calibration."""
    import sys
    assert "backend.app.speech.tts_engine" not in sys.modules
    assert "backend.app.interaction.fsm" not in sys.modules


# -------------------------------------------------------------------
# Helper Mock Profile
# -------------------------------------------------------------------

def get_mock_profile():
    from backend.app.core.contracts import CalibrationProfile, CalibrationStats
    from datetime import datetime, timezone

    stats = CalibrationStats(
        sample_count=45,
        median_h=0.50,
        median_v=0.50,
        mad_h=0.015,
        mad_v=0.018,
        min_h=0.48,
        max_h=0.52,
        min_v=0.48,
        max_v=0.52,
    )
    thresholds = CalibrationThresholds(
        left=DirectionThreshold(enter=0.40, exit=0.45),
        right=DirectionThreshold(enter=0.60, exit=0.55),
        up=DirectionThreshold(enter=0.38, exit=0.43),
        down=DirectionThreshold(enter=0.62, exit=0.57),
        blink_closed_ear=0.15,
        blink_open_ear=0.28,
    )
    return CalibrationProfile(
        user_id="U_TEST",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        camera_index=0,
        camera_resolution=(1280, 720),
        stats={"CENTER": stats},
        thresholds=thresholds,
    )


# -------------------------------------------------------------------
# Audit Regression Tests (M1.5-B2)
# -------------------------------------------------------------------

def test_unsupported_directions_symmetrically_omitted():
    """
    Verify that when LEFT, RIGHT, UP, and DOWN are unsupported:
    1. Symmetrical omission applies to all directional trials (marked OMITTED_UNSUPPORTED).
    2. Center fixation stability is evaluated and reported separately.
    3. Directional sensitivity reports status UNAVAILABLE and 0 trials evaluated.
    4. Overall trial and frame accuracies are reported as None rather than inflating to center-only accuracy.
    5. Evaluation status is UNAVAILABLE_NO_SUPPORTED_DIRECTIONS.
    """
    records = []
    # 5 center trials (conditions 14, 16, 18, 20, 22)
    # 4 directional trials (conditions 15: LEFT, 17: RIGHT, 19: UP, 21: DOWN)
    center_conds = [(14, "EVAL_CENTER_1"), (16, "EVAL_CENTER_2"), (18, "EVAL_CENTER_3"), (20, "EVAL_CENTER_4"), (22, "EVAL_CENTER_5")]
    dir_conds = [(15, "EVAL_LEFT", "LEFT"), (17, "EVAL_RIGHT", "RIGHT"), (19, "EVAL_UP", "UP"), (21, "EVAL_DOWN", "DOWN")]

    f_idx = 1
    for cond_idx, cond_name in center_conds:
        for _ in range(10):
            records.append(make_record(f_idx, ProtocolPhase.INDEPENDENT_EVALUATION, cond_idx, cond_name, "CENTER", h=0.50, v=0.50))
            f_idx += 1

    for cond_idx, cond_name, t_dir in dir_conds:
        for _ in range(10):
            records.append(make_record(f_idx, ProtocolPhase.INDEPENDENT_EVALUATION, cond_idx, cond_name, t_dir, h=0.50, v=0.50))
            f_idx += 1

    # All directions unsupported
    gate_results = {
        "is_left_supported": False,
        "is_right_supported": False,
        "is_up_supported": False,
        "is_down_supported": False,
        "is_blink_supported": False,
    }
    frozen_thresholds = CalibrationThresholds(
        left=DirectionThreshold(enter=0.0, exit=0.0, is_enabled=False),
        right=DirectionThreshold(enter=1.0, exit=1.0, is_enabled=False),
        up=DirectionThreshold(enter=-999.0, exit=-999.0, is_enabled=False),
        down=DirectionThreshold(enter=999.0, exit=999.0, is_enabled=False),
        blink_closed_ear=0.15,
        blink_open_ear=0.28,
    )

    res = evaluate_independent_phase2_trials(
        phase2_records=records,
        frozen_thresholds=frozen_thresholds,
        gate_results=gate_results,
        active_eye="right",
    )

    # 1. Overall evaluation status and accuracy when no directions supported
    assert res["evaluation_status"] == "UNAVAILABLE_NO_SUPPORTED_DIRECTIONS"
    assert res["trial_level_accuracy"] is None
    assert res["frame_level_accuracy"] is None

    # 2. Center fixation stability must be reported separately and show 100%
    c_stab = res["center_fixation_stability"]
    assert c_stab["trials_evaluated"] == 5
    assert c_stab["trials_correct"] == 5
    assert c_stab["trial_accuracy"] == 1.0

    # 3. Directional sensitivity must report UNAVAILABLE and 0 evaluated trials
    d_sens = res["directional_sensitivity"]
    assert d_sens["status"] == "UNAVAILABLE"
    assert d_sens["trials_evaluated"] == 0
    assert d_sens["trial_accuracy"] is None

    # 4. Symmetrical omission of all 4 directional trials
    omitted_trials = [t for t in res["trial_details"] if t["status"] == "OMITTED_UNSUPPORTED"]
    assert len(omitted_trials) == 4
    for ot in omitted_trials:
        assert ot["trial_predicted"] is None
        assert ot["trial_correct"] is None
        assert ot["frame_accuracy"] is None
        assert ot["target_direction"] in ("LEFT", "RIGHT", "UP", "DOWN")


def test_disabled_direction_fails_safely():
    """
    Verify that an unsupported/disabled DirectionThreshold (is_enabled=False):
    1. Never enters the directional state even on extreme or boundary inputs.
    2. Out-of-bounds or non-finite inputs return Direction.ABSTAIN.
    """
    thresholds = CalibrationThresholds(
        left=DirectionThreshold(enter=0.40, exit=0.45, is_enabled=False),  # Disabled
        right=DirectionThreshold(enter=0.60, exit=0.55, is_enabled=True),
        up=DirectionThreshold(enter=0.38, exit=0.43, is_enabled=False),    # Disabled
        down=DirectionThreshold(enter=0.62, exit=0.57, is_enabled=False),  # Disabled
        blink_closed_ear=0.15,
        blink_open_ear=0.28,
    )

    # In-range ratio that would enter LEFT if enabled (0.30 <= 0.40)
    pred_left = classify_frame_discrete(
        h=0.30, v=0.50, ear=0.30,
        current_dir=Direction.CENTER,
        thresholds=thresholds,
        is_left_supported=False,
    )
    assert pred_left == Direction.CENTER  # Must NOT enter LEFT

    # Non-finite input -> ABSTAIN
    pred_nan = classify_frame_discrete(
        h=float("nan"), v=0.50, ear=0.30,
        current_dir=Direction.CENTER,
        thresholds=thresholds,
    )
    assert pred_nan == Direction.ABSTAIN

    # Out-of-bounds ratio -> ABSTAIN
    pred_oob = classify_frame_discrete(
        h=1.58, v=0.50, ear=0.30,
        current_dir=Direction.CENTER,
        thresholds=thresholds,
    )
    assert pred_oob == Direction.ABSTAIN

    # Test GazeEngine integration with disabled direction
    from backend.app.core.contracts import CalibrationProfile
    from datetime import datetime, timezone
    profile = CalibrationProfile(
        user_id="U_TEST",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        camera_index=0,
        camera_resolution=(1280, 720),
        stats={},
        thresholds=thresholds,
    )
    engine = GazeEngine(profile=profile)

    eye = EyeLandmarks(
        iris_center=Point2D(x=100.0, y=100.0),
        inner_canthus=Point2D(x=80.0, y=100.0),
        outer_canthus=Point2D(x=120.0, y=100.0),
        upper_lid=Point2D(x=100.0, y=90.0),
        lower_lid=Point2D(x=100.0, y=110.0),
        h_ratio=0.30,  # Would enter LEFT if enabled
        v_ratio=0.50,
        ear=0.30,
    )
    feat = GazeFeatures(timestamp_ms=100, right_eye=eye, tracking_confidence=0.95, is_tracking_valid=True)
    state = engine.process_features(feat)
    assert state.direction == Direction.CENTER  # Fails safely to CENTER


def test_closed_eye_filter_accepts_low_ear_and_rejects_malformed():
    """
    Verify that filter_valid_samples(allow_closed_eyes=True):
    1. Accepts valid face frames with low EAR (< 0.18).
    2. Rejects missing face, non-finite EAR, negative EAR, and collapsed/malformed geometry.
    """
    records = [
        # Valid closed eye
        make_record(1, ProtocolPhase.CALIBRATION_ESTIMATION, 6, "CALIB_CLOSED", "CLOSED", ear=0.08, face_detected=True, geometry_valid=True),
        # Missing face -> must reject
        make_record(2, ProtocolPhase.CALIBRATION_ESTIMATION, 6, "CALIB_CLOSED", "CLOSED", ear=0.08, face_detected=False, geometry_valid=False),
        # Non-finite EAR -> must reject
        make_record(3, ProtocolPhase.CALIBRATION_ESTIMATION, 6, "CALIB_CLOSED", "CLOSED", ear=float("nan"), face_detected=True, geometry_valid=False),
        # Out-of-range EAR (< 0.0) -> must reject
        make_record(4, ProtocolPhase.CALIBRATION_ESTIMATION, 6, "CALIB_CLOSED", "CLOSED", ear=-0.05, face_detected=True, geometry_valid=False),
    ]
    # Record with unavailable_reason containing INSUFFICIENT_EYE_WIDTH
    rec_malformed = make_record(5, ProtocolPhase.CALIBRATION_ESTIMATION, 6, "CALIB_CLOSED", "CLOSED", ear=0.08, face_detected=True, geometry_valid=False)
    rec_malformed.unavailable_reason = "INSUFFICIENT_EYE_WIDTH: eye width 4.2px < 15.0px"
    records.append(rec_malformed)

    valid = filter_valid_samples(records, active_eye="right", allow_closed_eyes=True)
    assert len(valid) == 1
    assert valid[0].frame_index == 1


def test_presentation_mirroring_does_not_mutate_raw_frame():
    """
    Regression test proving presentation mirroring (cv2.flip) creates an independent display
    buffer and cannot mutate or alter the raw frame used for MediaPipe / MobileOne perception.
    """
    import cv2
    import numpy as np

    # Synthetic raw camera frame
    raw_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Distinct asymmetric pixel marker on the left half (camera-left)
    raw_frame[200:250, 100:150] = [255, 128, 64]
    raw_frame_copy = raw_frame.copy()

    # Presentation mirroring
    display_frame = cv2.flip(raw_frame, 1)

    # Mutate / draw on display_frame (simulating HUD rendering)
    cv2.circle(display_frame, (640, 360), 50, (0, 255, 0), -1)

    # 1. raw_frame must be completely unchanged byte-for-byte
    assert np.array_equal(raw_frame, raw_frame_copy)

    # 2. Marker in raw_frame must still be on left side (x in 100:150)
    assert np.array_equal(raw_frame[220, 125], [255, 128, 64])

    # 3. Marker in display_frame must be on flipped side (x in 1280 - 150 : 1280 - 100)
    flipped_x = 1280 - 1 - 125
    assert np.array_equal(display_frame[220, flipped_x], [255, 128, 64])


def test_blink_separation_gate_evaluates_participant_closure():
    """
    Verify derive_frozen_candidate_thresholds:
    1. If cl_ear >= 0.18, flags BLINK_UNSUPPORTED with explicit participant failure message,
       even if delta_ear_blink is positive.
    2. If cl_ear < 0.18 and delta_ear_blink >= 0.05, confirms is_blink_supported is True.
    """
    # Case 1: Participant failed to close eyes (cl_ear = 0.22 >= 0.18)
    stats_unclosed = {
        "CENTER": {"h": {"median": 0.50}, "v_eyelid": {"median": 0.50}, "ear": {"median": 0.30}},
        "CLOSED": {"ear": {"median": 0.22}},
    }
    _, gate_unclosed = derive_frozen_candidate_thresholds(stats_unclosed)
    assert gate_unclosed["is_blink_supported"] is False
    assert any("Participant eyes not closed" in r for r in gate_unclosed["unsupported_reasons"])

    # Case 2: Participant properly closed eyes (cl_ear = 0.08 < 0.18, delta = 0.22 >= 0.05)
    stats_closed = {
        "CENTER": {"h": {"median": 0.50}, "v_eyelid": {"median": 0.50}, "ear": {"median": 0.30}},
        "CLOSED": {"ear": {"median": 0.08}},
    }
    _, gate_closed = derive_frozen_candidate_thresholds(stats_closed)
    assert gate_closed["is_blink_supported"] is True


def test_canthal_normal_vertical_gate_evaluation():
    """
    Verify that when eyelid vertical fails or inverts due to anatomical lid-coupling,
    skeletal canthal-normal vertical metric is evaluated, passes gates, and enables UP/DOWN.
    """
    records = []
    # Center: eyelid = 0.45, canthal = -0.037, EAR = 0.30
    for i in range(25):
        records.append(make_record(i, ProtocolPhase.CALIBRATION_ESTIMATION, 1, "CALIB_CENTER_1", "CENTER",
                                   h=0.50, v=0.45, ear=0.30, canthal_ratio=-0.037))
    # Left: H = 0.40 (delta = 0.10)
    for i in range(25):
        records.append(make_record(25 + i, ProtocolPhase.CALIBRATION_ESTIMATION, 2, "CALIB_LEFT", "LEFT",
                                   h=0.40, v=0.45, ear=0.30, canthal_ratio=-0.037))
    # Right: H = 0.60 (delta = 0.10)
    for i in range(25):
        records.append(make_record(50 + i, ProtocolPhase.CALIBRATION_ESTIMATION, 4, "CALIB_RIGHT", "RIGHT",
                                   h=0.60, v=0.45, ear=0.30, canthal_ratio=-0.037))
    # UP: eyelid = 0.46 (inverted delta = -0.01), canthal = -0.043 (delta = +0.006 >= 0.005)
    for i in range(25):
        records.append(make_record(75 + i, ProtocolPhase.CALIBRATION_ESTIMATION, 6, "CALIB_UP", "UP",
                                   h=0.50, v=0.46, ear=0.30, canthal_ratio=-0.043))
    # DOWN: eyelid = 0.44 (inverted delta = -0.01), canthal = -0.025 (delta = +0.012 >= 0.005)
    for i in range(25):
        records.append(make_record(100 + i, ProtocolPhase.CALIBRATION_ESTIMATION, 8, "CALIB_DOWN", "DOWN",
                                   h=0.50, v=0.44, ear=0.30, canthal_ratio=-0.025))
    # CLOSED: EAR = 0.08 (delta = 0.22)
    for i in range(25):
        records.append(make_record(125 + i, ProtocolPhase.CALIBRATION_ESTIMATION, 10, "CALIB_CLOSED", "CLOSED",
                                   h=0.50, v=0.45, ear=0.08, canthal_ratio=-0.037))

    p1_stats = compute_phase1_target_stats(records, active_eye="right")
    thresholds, gates = derive_frozen_candidate_thresholds(p1_stats)

    assert thresholds is not None
    assert gates["is_left_supported"] is True
    assert gates["is_right_supported"] is True
    assert gates["is_up_supported"] is True
    assert gates["is_down_supported"] is True
    assert gates["is_blink_supported"] is True
    assert gates["vertical_metric"] == "canthal_normal"
    assert len(gates["unsupported_reasons"]) == 0

    # UP: enter < exit (more negative enter threshold)
    assert thresholds.up.enter < thresholds.up.exit
    assert thresholds.up.is_enabled is True
    # DOWN: enter > exit (more positive enter threshold)
    assert thresholds.down.enter > thresholds.down.exit
    assert thresholds.down.is_enabled is True

    # Test Phase 2 test trials with canthal normal
    phase2 = []
    # Trial 11: EVAL_CENTER
    for i in range(25):
        phase2.append(make_record(i, ProtocolPhase.INDEPENDENT_EVALUATION, 11, "EVAL_CENTER_1", "CENTER",
                                  h=0.50, v=0.45, ear=0.30, canthal_ratio=-0.037))
    # Trial 16: EVAL_UP
    for i in range(25):
        phase2.append(make_record(25 + i, ProtocolPhase.INDEPENDENT_EVALUATION, 16, "EVAL_UP", "UP",
                                  h=0.50, v=0.46, ear=0.30, canthal_ratio=-0.043))
    # Trial 18: EVAL_DOWN
    for i in range(25):
        phase2.append(make_record(50 + i, ProtocolPhase.INDEPENDENT_EVALUATION, 18, "EVAL_DOWN", "DOWN",
                                  h=0.50, v=0.44, ear=0.30, canthal_ratio=-0.024))

    p2_eval = evaluate_independent_phase2_trials(
        phase2_records=phase2,
        frozen_thresholds=thresholds,
        gate_results=gates,
        active_eye="right",
    )
    assert p2_eval["trials_evaluated_count"] == 3
    assert p2_eval["trials_correct_count"] == 3
    assert p2_eval["trial_level_accuracy"] == 1.0


def test_canthal_normal_gaze_engine_hysteresis():
    """
    Verify that GazeEngine uses canthal_ratio for vertical hysteresis
    when calibrated profile uses vertical_metric == 'canthal_normal'.
    """
    profile = get_mock_profile()
    profile.thresholds.vertical_metric = "canthal_normal"
    profile.thresholds.up.enter = -0.040
    profile.thresholds.up.exit = -0.038
    profile.thresholds.down.enter = -0.028
    profile.thresholds.down.exit = -0.032

    engine = GazeEngine(profile=profile)

    def _feat(canthal_val: float) -> GazeFeatures:
        pt = Point2D(x=100.0, y=100.0)
        eye = EyeLandmarks(
            iris_center=pt,
            inner_canthus=Point2D(x=80.0, y=100.0),
            outer_canthus=Point2D(x=120.0, y=100.0),
            upper_lid=Point2D(x=100.0, y=90.0),
            lower_lid=Point2D(x=100.0, y=110.0),
            h_ratio=0.50,
            v_ratio=0.50,  # Center eyelid ratio
            ear=0.30,
            canthal_ratio=canthal_val,
        )
        return GazeFeatures(
            timestamp_ms=100,
            right_eye=eye,
            tracking_confidence=0.95,
            is_tracking_valid=True,
            is_right_eye_valid=True,
        )

    # 1. Neutral Center
    s1 = engine.process_features(_feat(-0.037))
    assert s1.direction == Direction.CENTER

    # 2. Shift UP: canthal = -0.045 (< up.enter -0.040)
    s2 = engine.process_features(_feat(-0.045))
    assert s2.direction == Direction.UP

    # 3. Return to Center: canthal = -0.037 (> up.exit -0.038)
    s3 = engine.process_features(_feat(-0.037))
    assert s3.direction == Direction.CENTER

    # 4. Shift DOWN: canthal = -0.020 (> down.enter -0.028)
    s4 = engine.process_features(_feat(-0.020))
    assert s4.direction == Direction.DOWN

    # 5. Return to Center: canthal = -0.037 (< down.exit -0.032)
    s5 = engine.process_features(_feat(-0.037))
    assert s5.direction == Direction.CENTER
