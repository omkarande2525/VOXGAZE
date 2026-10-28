"""
VoxGaze Perception Unit Tests: Structured Gaze-Observation Measurement Protocol

Validates:
1. Condition sequence integrity (20 required conditions).
2. Schema serialization and missing-value representation (empty string, not zero).
3. Non-destructive unique output path generation.
4. Observation session interactive state machine.
5. Aggregated summary calculations and statistics (availability, latency, circular stats).
6. Graceful handling of empty or all-missing data.
7. Aborted session handling and partial data preservation.
"""

import csv
import json
import math
from pathlib import Path
import tempfile
import pytest

from backend.app.perception.structured_observation import (
    CSV_HEADER,
    OBSERVATION_CONDITIONS,
    FrameObservationRecord,
    ObservationCondition,
    ObservationSession,
    calculate_circular_stats,
    compute_distribution_stats,
    compute_observation_summary,
    get_unique_output_path,
)


def test_observation_conditions_sequence_integrity():
    """Verify that the 20 guided observation conditions strictly match the required specification."""
    assert len(OBSERVATION_CONDITIONS) == 20

    expected_names = [
        "CENTER_BASELINE",
        "EYES_LEFT",
        "EYES_CENTER_RETURN",
        "EYES_RIGHT",
        "EYES_CENTER_RETURN",
        "EYES_UP",
        "EYES_CENTER_RETURN",
        "EYES_DOWN",
        "EYES_CENTER_RETURN",
        "HEAD_LEFT_EYES_FORWARD",
        "HEAD_CENTER_RETURN",
        "HEAD_RIGHT_EYES_FORWARD",
        "HEAD_CENTER_RETURN",
        "HEAD_UP_EYES_FORWARD",
        "HEAD_CENTER_RETURN",
        "HEAD_DOWN_EYES_FORWARD",
        "HEAD_CENTER_RETURN",
        "COMBINED_MOVEMENT",
        "TRACKING_RECOVERY",
        "FINAL_CENTER_BASELINE",
    ]

    for idx, (cond, expected_name) in enumerate(zip(OBSERVATION_CONDITIONS, expected_names), start=1):
        assert cond.index == idx
        assert cond.name == expected_name
        assert len(cond.instruction.strip()) > 5
        # Verify center conditions are properly flagged
        if "CENTER" in expected_name:
            assert cond.is_center is True
        else:
            assert cond.is_center is False


def test_measurement_schema_construction_and_csv_formatting():
    """Verify CSV formatting rules: empty string for None/NaN, correct column count, boolean strings."""
    # 1. Valid complete record
    rec_valid = FrameObservationRecord(
        session_id="test_sess_01",
        monotonic_time_s=123.456789,
        elapsed_session_s=1.234,
        condition_index=1,
        condition_name="CENTER_BASELINE",
        frame_index=10,
        face_detected=True,
        face_crop_valid=True,
        geometry_valid=True,
        right_eye_h_ratio=0.5123,
        right_eye_v_ratio=0.4891,
        right_eye_ear=0.2850,
        left_eye_h_ratio=0.4950,
        left_eye_v_ratio=0.4912,
        left_eye_ear=0.2810,
        canthal_normal_dist_px=-0.42,
        mobileone_valid=True,
        mobileone_yaw_deg=1.25,
        mobileone_pitch_deg=-0.85,
        mediapipe_latency_ms=12.4,
        crop_latency_ms=1.1,
        mobileone_preprocess_ms=0.5,
        mobileone_infer_ms=8.2,
        total_frame_latency_ms=22.2,
        unavailable_reason=None,
    )

    row_valid = rec_valid.to_csv_row()
    assert len(row_valid) == len(CSV_HEADER)
    assert row_valid[0] == "test_sess_01"
    assert row_valid[3] == "1"
    assert row_valid[4] == "CENTER_BASELINE"
    assert row_valid[6] == "true"
    assert row_valid[9] == "0.5123"
    assert row_valid[17] == "1.25"
    assert row_valid[24] == ""  # unavailable_reason is empty string when None

    # 2. Incomplete / missing values record (MUST NOT substitute zero)
    rec_missing = FrameObservationRecord(
        session_id="test_sess_02",
        monotonic_time_s=124.0,
        elapsed_session_s=2.0,
        condition_index=2,
        condition_name="EYES_LEFT",
        frame_index=11,
        face_detected=False,
        face_crop_valid=False,
        geometry_valid=False,
        right_eye_h_ratio=None,
        right_eye_v_ratio=None,
        right_eye_ear=None,
        left_eye_h_ratio=None,
        left_eye_v_ratio=None,
        left_eye_ear=None,
        canthal_normal_dist_px=None,
        mobileone_valid=False,
        mobileone_yaw_deg=None,
        mobileone_pitch_deg=None,
        mediapipe_latency_ms=10.0,
        crop_latency_ms=0.0,
        mobileone_preprocess_ms=None,
        mobileone_infer_ms=None,
        total_frame_latency_ms=10.5,
        unavailable_reason="NO_FACE_DETECTED",
    )

    row_missing = rec_missing.to_csv_row()
    assert len(row_missing) == len(CSV_HEADER)
    assert row_missing[6] == "false"   # face_detected
    assert row_missing[7] == "false"   # face_crop_valid
    assert row_missing[8] == "false"   # geometry_valid
    assert row_missing[9] == ""        # right_eye_h_ratio MUST be empty, not "0.0"
    assert row_missing[10] == ""       # right_eye_v_ratio MUST be empty, not "0.0"
    assert row_missing[16] == "false"  # mobileone_valid
    assert row_missing[17] == ""       # mobileone_yaw_deg MUST be empty, not "0.0"
    assert row_missing[18] == ""       # mobileone_pitch_deg MUST be empty, not "0.0"
    assert row_missing[21] == ""       # mobileone_preprocess_ms
    assert row_missing[22] == ""       # mobileone_infer_ms
    assert row_missing[24] == "NO_FACE_DETECTED"


def test_unique_output_path_non_destructive():
    """Verify that output file generation never silently overwrites an existing file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        p1 = get_unique_output_path(tmp_path, prefix="test_obs", ext="csv")
        assert not p1.exists()

        # Create the file
        p1.touch()
        assert p1.exists()

        # Generating another path in the same directory must yield a distinct, non-existent path
        p2 = get_unique_output_path(tmp_path, prefix="test_obs", ext="csv")
        assert p2 != p1
        assert not p2.exists()


def test_observation_session_lifecycle_and_streaming():
    """Verify session progression: condition recording, skipping, CSV streaming, and completion."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        session = ObservationSession(output_dir=tmp_path, session_id="test_session_life")

        assert session.status == "INITIALIZED"
        assert session.current_condition_idx == 1
        assert session.current_condition.name == "CENTER_BASELINE"
        assert session.csv_path.exists()

        # Check CSV header written
        with open(session.csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == CSV_HEADER

        # 1. Start recording Condition 1
        session.start_recording_current_condition()
        assert session.status == "ACTIVE"
        assert session.is_recording_active is True

        # Record 2 frames
        rec1 = session.record_frame(
            frame_idx=1,
            face_detected=True,
            face_crop_valid=True,
            geometry_valid=True,
            right_eye_h=0.5,
            right_eye_v=0.5,
            mobileone_valid=True,
            mobileone_yaw_deg=0.1,
            mobileone_pitch_deg=-0.1,
            mobileone_infer_ms=8.0,
            total_frame_latency_ms=20.0,
        )
        assert rec1 is not None

        rec2 = session.record_frame(
            frame_idx=2,
            face_detected=True,
            face_crop_valid=True,
            geometry_valid=True,
            right_eye_h=0.51,
            right_eye_v=0.49,
            mobileone_valid=True,
            mobileone_yaw_deg=0.2,
            mobileone_pitch_deg=-0.05,
            mobileone_infer_ms=8.5,
            total_frame_latency_ms=21.0,
        )
        assert rec2 is not None

        # Advance to Condition 2
        more = session.finish_current_condition()
        assert more is True
        assert session.current_condition_idx == 2
        assert session.current_condition.name == "EYES_LEFT"
        assert session.is_recording_active is False
        assert session.condition_states[1] == "COMPLETED"

        # Skip Condition 2
        more = session.skip_current_condition()
        assert more is True
        assert session.current_condition_idx == 3
        assert session.current_condition.name == "EYES_CENTER_RETURN"
        assert session.condition_states[2] == "SKIPPED"

        # Complete session prematurely
        summary = session.complete_session()
        assert session.status == "COMPLETED"
        assert session.json_summary_path.exists()
        assert summary["metadata"]["session_status"] == "COMPLETED"
        assert summary["conditions_overview"]["completed"] == ["CENTER_BASELINE"]
        assert summary["conditions_overview"]["skipped"] == ["EYES_LEFT"]

        # Verify CSV has exactly 2 data rows
        with open(session.csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            assert len(reader) == 3  # Header + 2 data rows


def test_aborted_session_preserves_data():
    """Verify that aborting a session marks it as ABORTED and preserves previously recorded frames."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        session = ObservationSession(output_dir=tmp_path, session_id="test_session_abort")

        session.start_recording_current_condition()
        session.record_frame(
            frame_idx=1,
            face_detected=True,
            face_crop_valid=True,
            geometry_valid=True,
            mobileone_valid=True,
            mobileone_yaw_deg=0.5,
            mobileone_pitch_deg=0.2,
            mobileone_infer_ms=7.8,
            total_frame_latency_ms=19.5,
        )

        summary = session.abort_session()
        assert session.status == "ABORTED"
        assert summary["metadata"]["session_status"] == "ABORTED"
        assert summary["conditions_overview"]["interrupted_count"] == 1
        assert "CENTER_BASELINE" in summary["conditions_overview"]["interrupted"]

        # Check file still contains data
        with open(session.csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
            assert len(rows) == 2  # header + 1 record


def test_summary_calculations_on_synthetic_records():
    """Verify availability rates, latency percentiles, and circular statistics on synthetic records."""
    records = []
    # Condition 1: CENTER_BASELINE (10 frames, all valid)
    for i in range(10):
        records.append(
            FrameObservationRecord(
                session_id="synth_sess",
                monotonic_time_s=100.0 + i * 0.033,
                elapsed_session_s=i * 0.033,
                condition_index=1,
                condition_name="CENTER_BASELINE",
                frame_index=i,
                face_detected=True,
                face_crop_valid=True,
                geometry_valid=True,
                right_eye_h_ratio=0.50 + 0.01 * (i % 3),
                right_eye_v_ratio=0.50 - 0.01 * (i % 2),
                mobileone_valid=True,
                mobileone_yaw_deg=1.0 * (i % 3 - 1),  # -1.0, 0.0, 1.0
                mobileone_pitch_deg=0.5 * (i % 2),
                mediapipe_latency_ms=10.0,
                crop_latency_ms=1.0,
                mobileone_infer_ms=8.0 + (i % 5),
                total_frame_latency_ms=20.0 + i,
            )
        )

    # Condition 2: EYES_LEFT (5 frames: 2 face lost, 1 invalid crop, 2 valid)
    # Frame 0: face lost
    records.append(FrameObservationRecord(
        session_id="synth_sess", monotonic_time_s=101.0, elapsed_session_s=1.0,
        condition_index=2, condition_name="EYES_LEFT", frame_index=10,
        face_detected=False, face_crop_valid=False, geometry_valid=False,
        unavailable_reason="NO_FACE_DETECTED"
    ))
    # Frame 1: face lost
    records.append(FrameObservationRecord(
        session_id="synth_sess", monotonic_time_s=101.033, elapsed_session_s=1.033,
        condition_index=2, condition_name="EYES_LEFT", frame_index=11,
        face_detected=False, face_crop_valid=False, geometry_valid=False,
        unavailable_reason="NO_FACE_DETECTED"
    ))
    # Frame 2: invalid crop
    records.append(FrameObservationRecord(
        session_id="synth_sess", monotonic_time_s=101.066, elapsed_session_s=1.066,
        condition_index=2, condition_name="EYES_LEFT", frame_index=12,
        face_detected=True, face_crop_valid=False, geometry_valid=True,
        unavailable_reason="CROP_BELOW_MIN_GATE"
    ))
    # Frames 3, 4: valid
    for i in (13, 14):
        records.append(FrameObservationRecord(
            session_id="synth_sess", monotonic_time_s=101.1 + i * 0.033, elapsed_session_s=1.1,
            condition_index=2, condition_name="EYES_LEFT", frame_index=i,
            face_detected=True, face_crop_valid=True, geometry_valid=True,
            mobileone_valid=True, mobileone_yaw_deg=-15.0, mobileone_pitch_deg=0.0,
            mobileone_infer_ms=9.0, total_frame_latency_ms=22.0
        ))

    cond_states = {1: "COMPLETED", 2: "COMPLETED"}

    summary = compute_observation_summary(
        records=records,
        session_id="synth_sess",
        session_status="COMPLETED",
        condition_states=cond_states,
        start_time_iso="2026-10-14T10:00:00",
        end_time_iso="2026-10-14T10:00:10",
        elapsed_duration_s=10.0,
    )

    # Total frames = 15
    assert summary["overall_counts"]["total_frames"] == 15
    assert summary["overall_counts"]["tracking_loss_frames"] == 2
    assert summary["overall_counts"]["invalid_crop_frames"] == 1
    assert summary["overall_counts"]["mobileone_inference_error_frames"] == 0

    # Condition 1: 10 frames, 100% rates
    c1 = summary["condition_summaries"][0]
    assert c1["condition_name"] == "CENTER_BASELINE"
    assert c1["frames_collected"] == 10
    assert c1["face_detection_availability_rate"] == 1.0
    assert c1["valid_geometry_estimate_rate"] == 1.0
    assert c1["valid_mobileone_estimate_rate"] == 1.0

    # Condition 2: 5 frames, 3 face detected (rate = 0.6), 3 geom valid (0.6), 2 mobileone valid (0.4)
    c2 = summary["condition_summaries"][1]
    assert c2["condition_name"] == "EYES_LEFT"
    assert c2["frames_collected"] == 5
    assert pytest.approx(c2["face_detection_availability_rate"], rel=1e-3) == 0.6
    assert pytest.approx(c2["valid_geometry_estimate_rate"], rel=1e-3) == 0.6
    assert pytest.approx(c2["valid_mobileone_estimate_rate"], rel=1e-3) == 0.4

    # Latencies
    lat = summary["latency_metrics"]["mobileone_inference_ms"]
    assert lat["sample_count"] == 12  # 10 from c1 + 2 from c2
    assert lat["median_p50_ms"] is not None
    assert lat["p95_ms"] is not None
    assert lat["p95_ms"] >= lat["median_p50_ms"]

    # Center variability
    c_var = summary["center_observations_variability"]
    assert c_var["total_center_frames"] == 10
    assert "CENTER_BASELINE" in c_var["conditions_included"]
    yaw_stats = c_var["mobileone_yaw_deg"]["circular_stats"]
    assert yaw_stats["sample_count"] == 10
    assert pytest.approx(yaw_stats["circular_mean_deg"], abs=0.5) == 0.0


def test_circular_and_distribution_stats_edge_cases():
    """Verify stats computation with empty lists, identical angles, and wrap-boundary values."""
    # Empty
    empty_circ = calculate_circular_stats([])
    assert empty_circ["sample_count"] == 0
    assert empty_circ["circular_mean_deg"] is None

    empty_dist = compute_distribution_stats([])
    assert empty_dist["sample_count"] == 0
    assert empty_dist["median"] is None

    # Identical angles (e.g. all 0.0)
    zero_circ = calculate_circular_stats([0.0, 0.0, 0.0])
    assert zero_circ["sample_count"] == 3
    assert zero_circ["circular_mean_deg"] == 0.0
    assert zero_circ["circular_variance"] == 0.0
    assert zero_circ["circular_std_deg"] == 0.0

    # Opposing angles near boundary (-179° and +179°)
    # Linear mean would be 0° (which is completely wrong!). Circular mean is ±180°.
    opp_circ = calculate_circular_stats([-179.0, 179.0])
    assert opp_circ["sample_count"] == 2
    assert abs(abs(opp_circ["circular_mean_deg"]) - 180.0) < 0.1


def test_offline_analysis_tool_logic():
    """Verify that the offline analysis tool parses rows, detects transitions, and calculates stats."""
    from scripts.analyze_observation_session import analyze_session

    synthetic_rows = [
        # Condition 1: CENTER_BASELINE (2 frames)
        {
            "condition_index": "1",
            "condition_name": "CENTER_BASELINE",
            "elapsed_session_s": "1.0",
            "frame_index": "1",
            "face_detected": "true",
            "face_crop_valid": "true",
            "geometry_valid": "true",
            "mobileone_valid": "true",
            "mobileone_yaw_deg": "-4.0",
            "mobileone_pitch_deg": "1.0",
            "right_eye_h_ratio": "0.52",
            "right_eye_v_ratio": "0.43",
            "right_eye_ear": "0.28",
            "left_eye_h_ratio": "0.48",
            "left_eye_v_ratio": "0.43",
            "left_eye_ear": "0.27",
            "canthal_normal_dist_px": "-2.5",
            "mediapipe_latency_ms": "10.0",
            "crop_latency_ms": "0.3",
            "mobileone_preprocess_ms": "4.5",
            "mobileone_infer_ms": "7.0",
            "total_frame_latency_ms": "25.0",
            "unavailable_reason": "",
        },
        {
            "condition_index": "1",
            "condition_name": "CENTER_BASELINE",
            "elapsed_session_s": "1.033",
            "frame_index": "2",
            "face_detected": "true",
            "face_crop_valid": "true",
            "geometry_valid": "true",
            "mobileone_valid": "true",
            "mobileone_yaw_deg": "-5.0",
            "mobileone_pitch_deg": "1.5",
            "right_eye_h_ratio": "0.53",
            "right_eye_v_ratio": "0.44",
            "right_eye_ear": "0.29",
            "left_eye_h_ratio": "0.49",
            "left_eye_v_ratio": "0.44",
            "left_eye_ear": "0.28",
            "canthal_normal_dist_px": "-2.6",
            "mediapipe_latency_ms": "10.2",
            "crop_latency_ms": "0.2",
            "mobileone_preprocess_ms": "4.6",
            "mobileone_infer_ms": "7.2",
            "total_frame_latency_ms": "26.0",
            "unavailable_reason": "",
        },
        # Condition 19: TRACKING_RECOVERY (1 detected, 2 lost, 1 detected)
        {
            "condition_index": "19",
            "condition_name": "TRACKING_RECOVERY",
            "elapsed_session_s": "10.0",
            "frame_index": "100",
            "face_detected": "true",
            "face_crop_valid": "true",
            "geometry_valid": "true",
            "mobileone_valid": "true",
            "mobileone_yaw_deg": "-4.5",
            "mobileone_pitch_deg": "0.0",
            "right_eye_h_ratio": "0.50",
            "right_eye_v_ratio": "0.45",
            "right_eye_ear": "0.28",
            "left_eye_h_ratio": "0.50",
            "left_eye_v_ratio": "0.45",
            "left_eye_ear": "0.28",
            "canthal_normal_dist_px": "-2.0",
            "mediapipe_latency_ms": "10.0",
            "crop_latency_ms": "0.2",
            "mobileone_preprocess_ms": "4.5",
            "mobileone_infer_ms": "7.0",
            "total_frame_latency_ms": "25.0",
            "unavailable_reason": "",
        },
        {
            "condition_index": "19",
            "condition_name": "TRACKING_RECOVERY",
            "elapsed_session_s": "10.033",
            "frame_index": "101",
            "face_detected": "false",
            "face_crop_valid": "false",
            "geometry_valid": "false",
            "mobileone_valid": "false",
            "mobileone_yaw_deg": "",
            "mobileone_pitch_deg": "",
            "right_eye_h_ratio": "",
            "right_eye_v_ratio": "",
            "right_eye_ear": "",
            "left_eye_h_ratio": "",
            "left_eye_v_ratio": "",
            "left_eye_ear": "",
            "canthal_normal_dist_px": "",
            "mediapipe_latency_ms": "8.0",
            "crop_latency_ms": "0.0",
            "mobileone_preprocess_ms": "",
            "mobileone_infer_ms": "",
            "total_frame_latency_ms": "9.0",
            "unavailable_reason": "NO_FACE_DETECTED",
        },
        {
            "condition_index": "19",
            "condition_name": "TRACKING_RECOVERY",
            "elapsed_session_s": "10.066",
            "frame_index": "102",
            "face_detected": "false",
            "face_crop_valid": "false",
            "geometry_valid": "false",
            "mobileone_valid": "false",
            "mobileone_yaw_deg": "",
            "mobileone_pitch_deg": "",
            "right_eye_h_ratio": "",
            "right_eye_v_ratio": "",
            "right_eye_ear": "",
            "left_eye_h_ratio": "",
            "left_eye_v_ratio": "",
            "left_eye_ear": "",
            "canthal_normal_dist_px": "",
            "mediapipe_latency_ms": "8.0",
            "crop_latency_ms": "0.0",
            "mobileone_preprocess_ms": "",
            "mobileone_infer_ms": "",
            "total_frame_latency_ms": "9.0",
            "unavailable_reason": "NO_FACE_DETECTED",
        },
        {
            "condition_index": "19",
            "condition_name": "TRACKING_RECOVERY",
            "elapsed_session_s": "10.100",
            "frame_index": "103",
            "face_detected": "true",
            "face_crop_valid": "true",
            "geometry_valid": "true",
            "mobileone_valid": "true",
            "mobileone_yaw_deg": "-4.0",
            "mobileone_pitch_deg": "0.5",
            "right_eye_h_ratio": "0.51",
            "right_eye_v_ratio": "0.44",
            "right_eye_ear": "0.28",
            "left_eye_h_ratio": "0.49",
            "left_eye_v_ratio": "0.44",
            "left_eye_ear": "0.28",
            "canthal_normal_dist_px": "-2.1",
            "mediapipe_latency_ms": "12.0",
            "crop_latency_ms": "0.3",
            "mobileone_preprocess_ms": "4.8",
            "mobileone_infer_ms": "7.5",
            "total_frame_latency_ms": "35.0",
            "unavailable_reason": "",
        },
    ]

    res = analyze_session(synthetic_rows)
    assert res["total_frames"] == 6
    assert 1 in res["conditions"]
    assert 19 in res["conditions"]

    c1 = res["conditions"][1]
    assert c1["frames"] == 2
    assert c1["face_availability"] == 1.0
    assert c1["yaw"]["median"] == -4.5
    assert c1["pitch"]["median"] == 1.25

    rec = res["recovery_chronology"]
    assert rec["total_frames"] == 4
    assert rec["before_loss_frames"] == 1
    assert rec["during_loss_frames"] == 2
    assert rec["after_reacquisition_frames"] == 1
    assert rec["first_reacq_frame"]["frame_index"] == 103


# ======================================================================
# Run 2 (V2) Protocol Tests
# ======================================================================

from backend.app.perception.structured_observation import (
    CSV_HEADER_V2,
    FrameObservationRecordV2,
    ObservationSessionV2,
    V2ConditionTiming,
    V2_FIXATION_TARGETS,
    V2_PHASE_DONE,
    V2_PHASE_RECORDING,
    V2_PHASE_SETTLING,
    V2_PHASE_WAITING,
    V2_RECORDING_DURATION_S,
    V2_SETTLING_DURATION_S,
)


def test_v2_csv_header_has_bilateral_canthal_columns():
    """Verify that the V2 CSV header includes right and left eye canthal ratio columns."""
    assert "right_eye_canthal_ratio" in CSV_HEADER_V2
    assert "left_eye_canthal_ratio" in CSV_HEADER_V2
    # Verify the V2 header has exactly 2 more columns than V1
    assert len(CSV_HEADER_V2) == len(CSV_HEADER) + 2
    # V2 should still have all V1 columns except the two new ones are inserted
    for col in CSV_HEADER:
        assert col in CSV_HEADER_V2


def test_v2_frame_record_serialization_with_canthal_ratios():
    """Verify V2 record serialization preserves bilateral canthal ratios and uses empty for None."""
    rec = FrameObservationRecordV2(
        session_id="v2_test",
        monotonic_time_s=100.0,
        elapsed_session_s=1.0,
        condition_index=1,
        condition_name="CENTER_BASELINE",
        frame_index=1,
        face_detected=True,
        face_crop_valid=True,
        geometry_valid=True,
        right_eye_canthal_ratio=-0.0712,
        left_eye_canthal_ratio=-0.0685,
        mobileone_valid=True,
        mobileone_yaw_deg=0.5,
        mobileone_pitch_deg=-0.3,
        mobileone_infer_ms=7.0,
        total_frame_latency_ms=20.0,
    )
    row = rec.to_csv_row()
    assert len(row) == len(CSV_HEADER_V2)
    # Find the canthal ratio columns by position
    r_idx = CSV_HEADER_V2.index("right_eye_canthal_ratio")
    l_idx = CSV_HEADER_V2.index("left_eye_canthal_ratio")
    assert row[r_idx] == "-0.0712"
    assert row[l_idx] == "-0.0685"

    # Missing canthal ratios should be empty string
    rec_missing = FrameObservationRecordV2(
        session_id="v2_test",
        monotonic_time_s=101.0,
        elapsed_session_s=2.0,
        condition_index=1,
        condition_name="CENTER_BASELINE",
        frame_index=2,
        face_detected=False,
        face_crop_valid=False,
        geometry_valid=False,
    )
    row_missing = rec_missing.to_csv_row()
    assert row_missing[r_idx] == ""
    assert row_missing[l_idx] == ""


def test_v2_session_auto_timed_settling_and_recording():
    """
    Verify the V2 session auto-transitions:
    WAITING → SETTLING (after start_condition) → RECORDING (after 2s) → DONE (after 5s)
    and that settling frames are NOT recorded.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        session = ObservationSessionV2(
            output_dir=tmp_dir,
            session_id="v2_auto_test",
            settling_duration_s=0.1,  # Use short durations for deterministic testing
            recording_duration_s=0.2,
        )

        assert session.status == "INITIALIZED"
        assert session.current_phase == V2_PHASE_WAITING
        assert session.current_condition.name == "CENTER_BASELINE"

        t0 = 1000.0  # Synthetic monotonic time

        # Start condition → enters SETTLING
        session.start_condition(now_mono=t0)
        assert session.current_phase == V2_PHASE_SETTLING
        assert session.condition_states[1] == "SETTLING"

        # Record a frame during SETTLING — should return None
        result = session.record_frame(
            frame_idx=1,
            face_detected=True,
            face_crop_valid=True,
            geometry_valid=True,
            mobileone_valid=True,
            mobileone_yaw_deg=0.5,
            mobileone_pitch_deg=-0.1,
            mobileone_infer_ms=7.0,
            total_frame_latency_ms=20.0,
        )
        assert result is None, "Frames during SETTLING must NOT be recorded"
        assert len(session.records) == 0

        # Tick past settling duration → auto-transition to RECORDING
        t1 = t0 + 0.15  # Past the 0.1s settling duration
        phase = session.tick(now_mono=t1)
        assert phase == V2_PHASE_RECORDING
        assert session.current_phase == V2_PHASE_RECORDING
        assert session.condition_states[1] == "RECORDING"

        # Record frames during RECORDING — should succeed
        r1 = session.record_frame(
            frame_idx=2,
            face_detected=True,
            face_crop_valid=True,
            geometry_valid=True,
            right_eye_canthal_ratio=-0.07,
            left_eye_canthal_ratio=-0.065,
            mobileone_valid=True,
            mobileone_yaw_deg=0.3,
            mobileone_pitch_deg=-0.2,
            mobileone_infer_ms=7.5,
            total_frame_latency_ms=21.0,
        )
        assert r1 is not None
        assert len(session.records) == 1
        assert r1.right_eye_canthal_ratio == -0.07
        assert r1.left_eye_canthal_ratio == -0.065

        r2 = session.record_frame(
            frame_idx=3,
            face_detected=True,
            face_crop_valid=True,
            geometry_valid=True,
            mobileone_valid=True,
            mobileone_yaw_deg=0.4,
            mobileone_pitch_deg=-0.15,
            mobileone_infer_ms=7.2,
            total_frame_latency_ms=20.5,
        )
        assert r2 is not None
        assert len(session.records) == 2

        # Tick past recording duration → auto-finish condition 1, advance to condition 2
        t2 = t1 + 0.25  # Past the 0.2s recording duration
        phase = session.tick(now_mono=t2)
        assert session.condition_states[1] == "COMPLETED"
        assert session.current_condition_idx == 2
        assert session.current_condition.name == "EYES_LEFT"
        assert session.current_phase == V2_PHASE_WAITING

        # Verify settling frame count is tracked
        assert session._condition_timing[1].frames_during_settling == 1
        assert session._condition_timing[1].frames_during_recording == 2

        session.complete_session()


def test_v2_session_skip_and_abort():
    """Verify that skipping and aborting preserve state correctly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        session = ObservationSessionV2(
            output_dir=tmp_dir,
            session_id="v2_skip_test",
            settling_duration_s=0.05,
            recording_duration_s=0.1,
        )

        # Skip condition 1 without starting
        more = session.skip_current_condition()
        assert more is True
        assert session.condition_states[1] == "SKIPPED"
        assert session.current_condition_idx == 2

        # Start condition 2, then abort during settling
        session.start_condition(now_mono=2000.0)
        assert session.current_phase == V2_PHASE_SETTLING

        summary = session.abort_session()
        assert session.status == "ABORTED"
        assert summary["metadata"]["session_status"] == "ABORTED"
        assert summary["conditions_overview"]["skipped_count"] == 1
        assert summary["conditions_overview"]["interrupted_count"] == 1


def test_v2_session_csv_output_has_correct_header():
    """Verify the V2 session writes the V2 CSV header with bilateral canthal columns."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        session = ObservationSessionV2(
            output_dir=tmp_dir,
            session_id="v2_header_test",
        )
        session.complete_session()

        with open(session.csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == CSV_HEADER_V2


def test_v2_session_summary_contains_timing_metadata():
    """Verify that the V2 summary JSON includes per-condition timing and protocol metadata."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        session = ObservationSessionV2(
            output_dir=tmp_dir,
            session_id="v2_timing_test",
            settling_duration_s=0.05,
            recording_duration_s=0.1,
        )

        t0 = 3000.0
        session.start_condition(now_mono=t0)

        # Tick past settling
        t1 = t0 + 0.06
        session.tick(now_mono=t1)
        assert session.is_recording_active

        # Record one frame
        session.record_frame(
            frame_idx=1,
            face_detected=True,
            face_crop_valid=True,
            geometry_valid=True,
            mobileone_valid=True,
            mobileone_yaw_deg=0.0,
            mobileone_pitch_deg=0.0,
            mobileone_infer_ms=7.0,
            total_frame_latency_ms=20.0,
        )

        # Tick past recording
        t2 = t1 + 0.15
        session.tick(now_mono=t2)

        # Complete
        summary = session.complete_session()

        # Protocol metadata
        assert summary["metadata"]["protocol_version"] == "v2"
        assert summary["metadata"]["settling_duration_config_s"] == 0.05
        assert summary["metadata"]["recording_duration_config_s"] == 0.1

        # Condition timing
        assert "condition_timing_v2" in summary
        c1_timing = summary["condition_timing_v2"][1]
        assert c1_timing["settling_duration_actual_s"] is not None
        assert c1_timing["recording_duration_actual_s"] is not None
        assert c1_timing["frames_during_recording"] == 1


def test_v2_fixation_targets_cover_all_condition_names():
    """Verify that every unique condition name has a fixation target entry."""
    unique_names = set(c.name for c in OBSERVATION_CONDITIONS)
    for name in unique_names:
        assert name in V2_FIXATION_TARGETS, f"Missing fixation target for condition: {name}"


def test_v2_settling_and_recording_remaining_timers():
    """Verify the settling and recording remaining time calculations."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        session = ObservationSessionV2(
            output_dir=tmp_dir,
            session_id="v2_timer_test",
            settling_duration_s=2.0,
            recording_duration_s=5.0,
        )

        # Before starting: no remaining time
        assert session.get_settling_remaining_s() == 0.0
        assert session.get_recording_remaining_s() == 0.0

        t0 = 5000.0
        session.start_condition(now_mono=t0)

        # During settling: remaining should decrease
        assert pytest.approx(session.get_settling_remaining_s(now_mono=t0 + 0.5), abs=0.01) == 1.5
        assert pytest.approx(session.get_settling_remaining_s(now_mono=t0 + 1.8), abs=0.01) == 0.2
        assert session.get_recording_remaining_s(now_mono=t0 + 1.0) == 0.0  # Not recording yet

        # Transition to recording
        session.tick(now_mono=t0 + 2.1)
        assert session.is_recording_active

        # During recording: remaining should decrease
        assert pytest.approx(session.get_recording_remaining_s(now_mono=t0 + 3.1), abs=0.1) == 4.0
        assert session.get_settling_remaining_s(now_mono=t0 + 3.0) == 0.0  # No longer settling

        session.abort_session()


def test_v2_constants():
    """Verify the V2 protocol timing constants match the specification."""
    assert V2_SETTLING_DURATION_S == 2.0
    assert V2_RECORDING_DURATION_S == 5.0


def test_headless_run2_diagnostic_execution():
    """Verify headless execution of Run 2 diagnostic generates valid CSV and JSON summary."""
    from scripts.mobileone_live_diagnostic import run_live_diagnostic

    with tempfile.TemporaryDirectory() as tmp_dir:
        success = run_live_diagnostic(
            headless_frames=5,
            measure_v2=True,
            output_dir=Path(tmp_dir),
        )
        assert success is True

        csv_files = list(Path(tmp_dir).glob("*.csv"))
        json_files = list(Path(tmp_dir).glob("*_summary.json"))
        assert len(csv_files) == 1
        assert len(json_files) == 1

        with open(json_files[0], "r", encoding="utf-8") as f:
            summary = json.load(f)
        assert summary["metadata"]["protocol_version"] == "v2"
        assert summary["metadata"]["session_status"] == "COMPLETED"



