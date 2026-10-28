"""
Unit tests for Controlled Sequential Model Comparison Protocol.
Verifies balanced sequence generation, sequential record structure, dwell hit-testing,
block train/test splitting, and ResNet-34 adoption decision rule logic.
"""

from pathlib import Path
import pytest

from backend.app.calibration.personalized_mapper import FeatureSetType
from scripts.run_controlled_model_comparison import (
    BLOCK_1_TARGETS,
    BLOCK_2_TARGETS,
    BLOCK_3_TARGETS,
    BLOCK_4_TARGETS,
    BLOCK_5_TARGETS,
    FrameComparisonRecord,
    TARGET_BBOXES,
    TARGET_COORDS,
    analyze_block_evaluation,
    evaluate_model_adoption,
    evaluate_trial_dwell,
    execute_comparison_session,
    is_inside_box,
)


def test_balanced_target_sequences():
    # All 5 blocks must have equal length (9 items: 4 peripheral + 5 center)
    all_blocks = [BLOCK_1_TARGETS, BLOCK_2_TARGETS, BLOCK_3_TARGETS, BLOCK_4_TARGETS, BLOCK_5_TARGETS]
    assert len(all_blocks) == 5
    for targets in all_blocks:
        assert len(targets) == 9
        assert targets.count("CENTER") == 5
        assert targets.count("LEFT") == 1
        assert targets.count("RIGHT") == 1
        assert targets.count("UP") == 1
        assert targets.count("DOWN") == 1

    # Sequences must not be identical across blocks (prevent sequence-order bias)
    for i in range(len(all_blocks)):
        for j in range(i + 1, len(all_blocks)):
            assert all_blocks[i] != all_blocks[j]

    # Held-out blocks (3, 4, 5) must yield exactly 12 peripheral and 15 center targets
    held_out_blocks = [BLOCK_3_TARGETS, BLOCK_4_TARGETS, BLOCK_5_TARGETS]
    total_held_out_center = sum(b.count("CENTER") for b in held_out_blocks)
    total_held_out_peripheral = sum(len([t for t in b if t != "CENTER"]) for b in held_out_blocks)
    assert total_held_out_center == 15
    assert total_held_out_peripheral == 12


def test_bounding_box_hit_testing():
    left_box = TARGET_BBOXES["LEFT"]
    assert is_inside_box(0.15, 0.50, left_box) is True
    assert is_inside_box(0.85, 0.50, left_box) is False

    center_box = TARGET_BBOXES["CENTER"]
    assert is_inside_box(0.50, 0.50, center_box) is True
    assert is_inside_box(0.15, 0.50, center_box) is False


def test_dwell_evaluation():
    left_box = TARGET_BBOXES["LEFT"]

    # 15 consecutive points inside box -> pass (dwell >= 12)
    stable_left = [(0.15, 0.50) for _ in range(15)]
    assert evaluate_trial_dwell(stable_left, left_box, min_dwell_frames=12) is True

    # 5 inside, 1 outside, 5 inside -> fail (interrupted)
    broken_dwell = [(0.15, 0.50)] * 5 + [(0.50, 0.50)] + [(0.15, 0.50)] * 5
    assert evaluate_trial_dwell(broken_dwell, left_box, min_dwell_frames=12) is False


def test_execute_comparison_session_mock():
    report = execute_comparison_session(mock_mode=True)
    assert report["mock_mode"] is True
    assert report["is_physical_validation"] is False
    assert report["validation_notice"] == "MOCK, not physical validation"
    assert "models" in report
    assert "mobileone_s0_hybrid" in report["models"]
    assert "resnet34_hybrid" in report["models"]

    # Decision rule verification
    rule = report["decision_rule"]
    assert "adopted" in rule
    assert "rule_statement" in rule
    assert rule["adopted"] is False
    assert rule["is_physical_validation"] is False
    assert rule["validation_label"] == "MOCK, not physical validation"


def test_adoption_gate_rejects_100pct_peripheral_with_excessive_center_failures():
    """Regression test: 100% peripheral accuracy must still be rejected if center false activations exceed 5%."""
    eval_data = {
        "peripheral_trials_passed": 12,
        "peripheral_trials_total": 12,
        "peripheral_trial_accuracy": 1.0,
        "center_trials_false_act": 3,
        "center_trials_total": 15,
        "center_return_false_activation_rate": 0.20,  # 20.0% > 5.0% threshold
    }
    lat_info = {"p50": 25.0, "p95": 35.0, "status": "VALID"}

    res = evaluate_model_adoption(
        eval_data,
        lat_info,
        sessions_passed=2,
        is_mock=False,
        model_name="MobileOne S0",
    )
    assert res["adopted"] is False
    assert res["status"] == "REJECTED_CENTER_FALSE_ACTIVATION_HIGH"
    assert res["criteria_gates"]["peripheral_accuracy"] is True
    assert res["criteria_gates"]["center_false_activation"] is False
    assert any("Center false activation rate 20.0%" in r for r in res["reasons"])


def test_adoption_gate_handles_missing_latency():
    """Regression test: Missing or invalid latency must yield explicit INVALID status, never KeyError or adoption."""
    eval_data = {
        "peripheral_trials_passed": 12,
        "peripheral_trials_total": 12,
        "peripheral_trial_accuracy": 1.0,
        "center_trials_false_act": 0,
        "center_trials_total": 15,
        "center_return_false_activation_rate": 0.0,
    }

    # None latency info
    res_none = evaluate_model_adoption(eval_data, None, sessions_passed=2, is_mock=False)
    assert res_none["adopted"] is False
    assert res_none["status"] == "INVALID_MISSING_LATENCY"
    assert res_none["criteria_gates"]["valid_latency"] is False

    # Explicit missing latency dict
    lat_missing = {"p50": None, "p95": None, "status": "INVALID_MISSING_LATENCY"}
    res_missing = evaluate_model_adoption(eval_data, lat_missing, sessions_passed=2, is_mock=False)
    assert res_missing["adopted"] is False
    assert res_missing["status"] == "INVALID_MISSING_LATENCY"

    # NaN latency
    lat_nan = {"p50": float("nan"), "status": "VALID"}
    res_nan = evaluate_model_adoption(eval_data, lat_nan, sessions_passed=2, is_mock=False)
    assert res_nan["adopted"] is False
    assert res_nan["status"] == "INVALID_MISSING_LATENCY"


def test_adoption_gate_handles_incomplete_trials_and_nan():
    """Regression test: Incomplete trials, empty data, or NaN metrics must return explicit INVALID status."""
    lat_info = {"p50": 25.0, "status": "VALID"}

    # Case A: None eval dict
    res_none = evaluate_model_adoption(None, lat_info, sessions_passed=2, is_mock=False)
    assert res_none["adopted"] is False
    assert res_none["status"] == "INVALID_OR_INCOMPLETE_DATA"

    # Case B: Status INSUFFICIENT_DATA
    res_insuf = evaluate_model_adoption({"status": "INSUFFICIENT_DATA"}, lat_info, sessions_passed=2, is_mock=False)
    assert res_insuf["adopted"] is False
    assert res_insuf["status"] == "INSUFFICIENT_DATA"

    # Case C: NaN in metric accuracy
    eval_nan = {
        "peripheral_trials_passed": 12,
        "peripheral_trials_total": 12,
        "peripheral_trial_accuracy": float("nan"),
        "center_trials_false_act": 0,
        "center_trials_total": 15,
        "center_return_false_activation_rate": 0.0,
    }
    res_nan = evaluate_model_adoption(eval_nan, lat_info, sessions_passed=2, is_mock=False)
    assert res_nan["adopted"] is False
    assert res_nan["status"] == "INVALID_NAN_OR_MISSING_METRICS"

    # Case D: analyze_block_evaluation on empty records
    res_block = analyze_block_evaluation([], model_key="m1")
    assert res_block["status"] == "INSUFFICIENT_DATA"
    assert res_block["end_to_end_latency_ms"]["status"] == "INVALID_MISSING_LATENCY"


def test_mock_mode_never_presented_as_physical_validation():
    """Regression test: Mock mode must carry visible 'MOCK, not physical validation' and never confer adoption."""
    perfect_eval = {
        "peripheral_trials_passed": 12,
        "peripheral_trials_total": 12,
        "peripheral_trial_accuracy": 1.0,
        "center_trials_false_act": 0,
        "center_trials_total": 15,
        "center_return_false_activation_rate": 0.0,
    }
    lat_info = {"p50": 25.0, "status": "VALID"}

    # Mock mode rejected despite perfect scores
    res = evaluate_model_adoption(perfect_eval, lat_info, sessions_passed=2, is_mock=True, model_name="ResNet-34")
    assert res["adopted"] is False
    assert res["status"] == "MOCK_REJECTED"
    assert res["validation_label"] == "MOCK, not physical validation"
    assert res["criteria_gates"]["non_mock"] is False
    assert any("MOCK" in r for r in res["reasons"])


def test_adoption_gate_requires_all_criteria():
    """Regression test: Adoption strictly requires peripheral >= 85% (>=12 trials), center FA <= 5% (>=15 trials), and 2 sessions."""
    lat_info = {"p50": 25.0, "status": "VALID"}

    # 1. All passing: 12/12 (100%), 0/15 (0%), 2 sessions, valid latency, non-mock
    pass_eval = {
        "peripheral_trials_passed": 12,
        "peripheral_trials_total": 12,
        "peripheral_trial_accuracy": 1.0,
        "center_trials_false_act": 0,
        "center_trials_total": 15,
        "center_return_false_activation_rate": 0.0,
    }
    res_pass = evaluate_model_adoption(pass_eval, lat_info, sessions_passed=2, is_mock=False)
    assert res_pass["adopted"] is True
    assert res_pass["status"] == "ADOPTED_VALIDATED"
    assert res_pass["is_preliminary"] is False

    # 2. Fails session criterion (only 1 session)
    res_1sess = evaluate_model_adoption(pass_eval, lat_info, sessions_passed=1, is_mock=False)
    assert res_1sess["adopted"] is False
    assert res_1sess["status"] == "PRELIMINARY_INSUFFICIENT_SESSIONS"
    assert res_1sess["criteria_gates"]["cross_session_generalization"] is False

    # 3. Fails peripheral sample size (11 trials < 12)
    eval_11peri = {
        "peripheral_trials_passed": 11,
        "peripheral_trials_total": 11,
        "peripheral_trial_accuracy": 1.0,
        "center_trials_false_act": 0,
        "center_trials_total": 15,
        "center_return_false_activation_rate": 0.0,
    }
    res_11peri = evaluate_model_adoption(eval_11peri, lat_info, sessions_passed=2, is_mock=False)
    assert res_11peri["adopted"] is False
    assert res_11peri["status"] == "PRELIMINARY_SAMPLE_SIZE"
    assert res_11peri["criteria_gates"]["peripheral_sample_size"] is False

    # 4. Fails center sample size (14 trials < 15)
    eval_14center = {
        "peripheral_trials_passed": 12,
        "peripheral_trials_total": 12,
        "peripheral_trial_accuracy": 1.0,
        "center_trials_false_act": 0,
        "center_trials_total": 14,
        "center_return_false_activation_rate": 0.0,
    }
    res_14center = evaluate_model_adoption(eval_14center, lat_info, sessions_passed=2, is_mock=False)
    assert res_14center["adopted"] is False
    assert res_14center["status"] == "PRELIMINARY_SAMPLE_SIZE"
    assert res_14center["criteria_gates"]["center_sample_size"] is False

    # 5. Fails peripheral accuracy threshold (10/12 = 83.3% < 85%)
    eval_low_acc = {
        "peripheral_trials_passed": 10,
        "peripheral_trials_total": 12,
        "peripheral_trial_accuracy": 10 / 12,
        "center_trials_false_act": 0,
        "center_trials_total": 15,
        "center_return_false_activation_rate": 0.0,
    }
    res_low_acc = evaluate_model_adoption(eval_low_acc, lat_info, sessions_passed=2, is_mock=False)
    assert res_low_acc["adopted"] is False
    assert res_low_acc["status"] == "REJECTED_PERIPHERAL_ACCURACY_LOW"
    assert res_low_acc["criteria_gates"]["peripheral_accuracy"] is False

    # 6. Fails center false activation threshold (1/15 = 6.67% > 5%)
    eval_high_fa = {
        "peripheral_trials_passed": 12,
        "peripheral_trials_total": 12,
        "peripheral_trial_accuracy": 1.0,
        "center_trials_false_act": 1,
        "center_trials_total": 15,
        "center_return_false_activation_rate": 1 / 15,
    }
    res_high_fa = evaluate_model_adoption(eval_high_fa, lat_info, sessions_passed=2, is_mock=False)
    assert res_high_fa["adopted"] is False
    assert res_high_fa["status"] == "REJECTED_CENTER_FALSE_ACTIVATION_HIGH"
    assert res_high_fa["criteria_gates"]["center_false_activation"] is False


def test_real_timestamp_dwell():
    """Verifies that 400 ms dwell requires actual elapsed time >= 0.400s, not frame counts."""
    left_box = TARGET_BBOXES["LEFT"]

    # Case A: 5 points spanning 0.50s (t=0.0 to t=0.50) -> PASS (>= 0.400s)
    pts_pass = [
        (0.15, 0.50, 10.00),
        (0.15, 0.50, 10.15),
        (0.15, 0.50, 10.30),
        (0.15, 0.50, 10.45),
        (0.15, 0.50, 10.50),
    ]
    assert evaluate_trial_dwell(pts_pass, left_box, min_dwell_s=0.400) is True

    # Case B: 15 points spanning only 0.20s (e.g. 60 FPS burst: t=10.0 to t=10.20) -> FAIL (< 0.400s)
    pts_short = [(0.15, 0.50, 10.0 + i * (0.20 / 14)) for i in range(15)]
    assert evaluate_trial_dwell(pts_short, left_box, min_dwell_s=0.400) is False

    # Case C: Interrupted dwell (stays inside for 0.3s, slips out, stays for 0.3s) -> FAIL
    pts_interrupted = [
        (0.15, 0.50, 10.00),
        (0.15, 0.50, 10.30),
        (0.50, 0.50, 10.35),  # Slips to center
        (0.15, 0.50, 10.40),
        (0.15, 0.50, 10.70),
    ]
    assert evaluate_trial_dwell(pts_interrupted, left_box, min_dwell_s=0.400) is False


def test_transient_center_escapes_and_acquisition_latency_in_block_eval():
    """Verifies that 5-block evaluation records transient escapes and target acquisition latency."""
    # Build synthetic records across blocks 1-5
    records = []
    frame_idx = 0
    base_t = 100.0

    # Blocks 1 & 2 for training, 3 for test
    for b_idx in [1, 2, 3]:
        for c_idx, target in enumerate(["CENTER", "LEFT", "CENTER", "DOWN"]):
            t_name = f"B{b_idx}_{c_idx+1}_{target}"
            for f_i in range(15):
                frame_idx += 1
                t_frame = base_t + frame_idx * 0.0333  # ~30 FPS

                # In block 3 trial 1 (CENTER), inject 2 transient flutter frames to LEFT
                if b_idx == 3 and t_name == "B3_1_CENTER" and f_i in (5, 6):
                    rh, rv = 0.41, 0.52  # Left geometry
                    m1_yaw, m1_pitch = -18.0, -10.0
                elif target == "CENTER":
                    rh, rv = 0.50, 0.50
                    m1_yaw, m1_pitch = -4.0, -10.0
                elif target == "LEFT":
                    rh, rv = 0.41, 0.52
                    m1_yaw, m1_pitch = -18.0, -10.0
                elif target == "DOWN":
                    rh, rv = 0.50, 0.42
                    m1_yaw, m1_pitch = -4.0, -22.0

                rec = FrameComparisonRecord(
                    session_id="test_sess",
                    block_index=b_idx,
                    condition_index=c_idx + 1,
                    trial_name=t_name,
                    target_direction=target,
                    frame_index=frame_idx,
                    monotonic_time_s=t_frame,
                    face_detected=True,
                    face_crop_valid=True,
                    right_eye_h=rh,
                    right_eye_v=rv,
                    right_eye_ear=0.32,
                    left_eye_h=rh,
                    left_eye_v=rv,
                    left_eye_ear=0.32,
                    canthal_normal_px=-1.5,
                    m1_valid=True,
                    m1_yaw_deg=m1_yaw,
                    m1_pitch_deg=m1_pitch,
                    m1_end_to_end_ms=25.0,
                    r34_valid=True,
                    r34_yaw_deg=m1_yaw * 1.05,
                    r34_pitch_deg=m1_pitch * 1.05,
                    r34_end_to_end_ms=75.0,
                )
                records.append(rec)

    # Evaluate MobileOne
    m1_eval = analyze_block_evaluation(records, model_key="m1", feature_set=FeatureSetType.HYBRID)
    assert m1_eval["status"] == "VALID"

    # Check transient center escape analysis
    assert "transient_center_escapes" in m1_eval
    trans = m1_eval["transient_center_escapes"]
    assert trans["total_center_trials"] == 2
    assert trans["trials_with_transient_escapes"] >= 1
    assert trans["transient_escape_frames_count"] >= 2

    # Check target acquisition latency
    assert "target_acquisition_latency" in m1_eval
    acq = m1_eval["target_acquisition_latency"]
    assert acq["successful_acquisitions_count"] >= 1
    assert acq["p50_ms"] is not None
    assert acq["p50_ms"] >= 0.0

    # Check that preceding target is logged in both trial lists
    assert len(trans["per_trial_breakdown"]) >= 1
    assert "preceding_target" in trans["per_trial_breakdown"][0]
    assert len(acq["per_trial_acquisitions"]) >= 1
    assert "preceding_target" in acq["per_trial_acquisitions"][0]
