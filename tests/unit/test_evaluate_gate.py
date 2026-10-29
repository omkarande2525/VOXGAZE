"""
Unit tests for Cross-Session Acceptance Gate Evaluator (scripts/evaluate_gate.py).
Tests multi-session physical validation rules, same-day vs separate-day logic,
sample size enforcement, peripheral/center threshold gates, mock rejection, and read-only invariants.
"""

import copy
import json
from pathlib import Path
import pytest

from scripts.evaluate_gate import (
    evaluate_cross_session_gate,
    parse_session_date,
    extract_session_layout_fingerprint,
    MIN_PERIPHERAL_TRIALS,
    MIN_CENTER_TRIALS,
    PERIPHERAL_ACCURACY_THRESHOLD,
    CENTER_FALSE_ACTIVATION_MAX,
)

# Reference valid Session A template
SESSION_A_TEMPLATE = {
    "session_id": "cmp_v16_20261029_120000_aaaaaa",
    "timestamp_utc": "2026-10-29T12:00:00Z",
    "mock_mode": False,
    "validation_notice": "PHYSICAL ACQUISITION",
    "is_physical_validation": True,
    "total_frames_recorded": 3800,
    "status": "VALID",
    "protocol": {
        "blocks_count": 5,
        "targets_per_block": 9,
        "held_out_test_blocks": [3, 4, 5],
        "balanced_orders": {
            "block_1": ["CENTER", "LEFT", "CENTER", "RIGHT", "CENTER", "UP", "CENTER", "DOWN", "CENTER"],
            "block_2": ["CENTER", "RIGHT", "CENTER", "DOWN", "CENTER", "LEFT", "CENTER", "UP", "CENTER"],
            "block_3": ["CENTER", "DOWN", "CENTER", "LEFT", "CENTER", "UP", "CENTER", "RIGHT", "CENTER"],
            "block_4": ["CENTER", "UP", "CENTER", "RIGHT", "CENTER", "DOWN", "CENTER", "LEFT", "CENTER"],
            "block_5": ["CENTER", "LEFT", "CENTER", "UP", "CENTER", "DOWN", "CENTER", "RIGHT", "CENTER"],
        },
    },
    "models": {
        "mobileone_s0_hybrid": {
            "status": "VALID",
            "model": "MobileOne S0",
            "feature_set": "hybrid",
            "evaluation_5dir": {
                "status": "VALID",
                "peripheral_trials_passed": 12,
                "peripheral_trials_total": 12,
                "peripheral_trial_accuracy": 1.0,
                "center_trials_false_act": 0,
                "center_trials_stayed": 15,
                "center_trials_total": 15,
                "center_return_false_activation_rate": 0.0,
                "overall_trials_correct": 27,
                "overall_trials_total": 27,
                "overall_trial_correctness_rate": 1.0,
            },
            "end_to_end_latency_ms": {"p50": 24.5, "p95": 38.0, "status": "VALID"},
            "transient_center_escapes": {"trials_with_transient_escapes": 2, "total_center_trials": 15},
        }
    },
}


def create_test_report_file(tmp_path: Path, filename: str, data: dict) -> Path:
    p = tmp_path / filename
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return p


def test_evaluate_gate_rejects_single_report(tmp_path):
    p1 = create_test_report_file(tmp_path, "session_1.json", SESSION_A_TEMPLATE)
    res = evaluate_cross_session_gate([p1])
    assert res["status"] == "INVALID"
    assert any("Insufficient sessions provided: 1 report(s) given" in r for r in res["reasons"])


def test_evaluate_gate_rejects_mock_report(tmp_path):
    rep_mock = copy.deepcopy(SESSION_A_TEMPLATE)
    rep_mock["mock_mode"] = True
    rep_mock["is_physical_validation"] = False
    rep_mock["session_id"] = "cmp_mock_1"

    p1 = create_test_report_file(tmp_path, "session_1.json", SESSION_A_TEMPLATE)
    p2 = create_test_report_file(tmp_path, "session_2_mock.json", rep_mock)

    res = evaluate_cross_session_gate([p1, p2], declared_camera_repositioning=True)
    assert res["status"] == "INVALID"
    assert any("MOCK run" in r for r in res["reasons"])


def test_evaluate_gate_rejects_same_day_without_repositioning(tmp_path):
    rep1 = copy.deepcopy(SESSION_A_TEMPLATE)
    rep1["session_id"] = "cmp_v16_20261029_100000_111111"
    rep1["timestamp_utc"] = "2026-10-29T10:00:00Z"

    rep2 = copy.deepcopy(SESSION_A_TEMPLATE)
    rep2["session_id"] = "cmp_v16_20261029_140000_222222"
    rep2["timestamp_utc"] = "2026-10-29T14:00:00Z"

    p1 = create_test_report_file(tmp_path, "rep1.json", rep1)
    p2 = create_test_report_file(tmp_path, "rep2.json", rep2)

    # Without declared repositioning -> INVALID
    res = evaluate_cross_session_gate([p1, p2], declared_camera_repositioning=False)
    assert res["status"] == "INVALID"
    assert any("same day (2026-10-29) without a declared camera/laptop repositioning" in r for r in res["reasons"])


def test_evaluate_gate_accepts_same_day_with_declared_repositioning(tmp_path):
    rep1 = copy.deepcopy(SESSION_A_TEMPLATE)
    rep1["session_id"] = "cmp_v16_20261029_100000_111111"
    rep1["timestamp_utc"] = "2026-10-29T10:00:00Z"

    rep2 = copy.deepcopy(SESSION_A_TEMPLATE)
    rep2["session_id"] = "cmp_v16_20261029_140000_222222"
    rep2["timestamp_utc"] = "2026-10-29T14:00:00Z"

    p1 = create_test_report_file(tmp_path, "rep1.json", rep1)
    p2 = create_test_report_file(tmp_path, "rep2.json", rep2)

    # With declared repositioning -> PASS
    res = evaluate_cross_session_gate([p1, p2], declared_camera_repositioning=True)
    assert res["status"] == "PASS"
    assert len(res["session_summaries"]) == 2
    assert res["session_summaries"][0]["passed"] is True
    assert res["session_summaries"][1]["passed"] is True


def test_evaluate_gate_passes_separate_calendar_days(tmp_path):
    # Session A: Thursday 2026-10-29
    rep_a = copy.deepcopy(SESSION_A_TEMPLATE)
    rep_a["session_id"] = "cmp_v16_20261029_120000_aaaaaa"
    rep_a["timestamp_utc"] = "2026-10-29T12:00:00Z"

    # Session B: Friday 2026-10-30
    rep_b = copy.deepcopy(SESSION_A_TEMPLATE)
    rep_b["session_id"] = "cmp_v16_20261030_090000_bbbbbb"
    rep_b["timestamp_utc"] = "2026-10-30T09:00:00Z"
    # Say Session B has 11/12 peripheral (91.7%) and 0/15 center
    rep_b["models"]["mobileone_s0_hybrid"]["evaluation_5dir"]["peripheral_trials_passed"] = 11
    rep_b["models"]["mobileone_s0_hybrid"]["evaluation_5dir"]["peripheral_trial_accuracy"] = 11 / 12

    p_a = create_test_report_file(tmp_path, "rep_a.json", rep_a)
    p_b = create_test_report_file(tmp_path, "rep_b.json", rep_b)

    res = evaluate_cross_session_gate([p_a, p_b], declared_camera_repositioning=False)
    assert res["status"] == "PASS"
    assert res["unique_dates_count"] == 2
    assert res["session_summaries"][0]["peripheral"]["display"] == "12/12 (100.0%)"
    assert res["session_summaries"][1]["peripheral"]["display"] == "11/12 (91.7%)"


def test_evaluate_gate_fails_when_one_session_misses_peripheral_accuracy(tmp_path):
    rep_a = copy.deepcopy(SESSION_A_TEMPLATE)
    rep_a["timestamp_utc"] = "2026-10-29T12:00:00Z"

    # Session B misses peripheral threshold: 9/12 = 75.0% < 85.0%
    rep_b = copy.deepcopy(SESSION_A_TEMPLATE)
    rep_b["timestamp_utc"] = "2026-10-30T12:00:00Z"
    rep_b["models"]["mobileone_s0_hybrid"]["evaluation_5dir"]["peripheral_trials_passed"] = 9
    rep_b["models"]["mobileone_s0_hybrid"]["evaluation_5dir"]["peripheral_trial_accuracy"] = 0.75

    p_a = create_test_report_file(tmp_path, "rep_a.json", rep_a)
    p_b = create_test_report_file(tmp_path, "rep_b.json", rep_b)

    res = evaluate_cross_session_gate([p_a, p_b])
    assert res["status"] == "FAIL"
    assert any("failed peripheral accuracy gate: 9/12 (75.0%) < required 85.0%" in r for r in res["reasons"])


def test_evaluate_gate_fails_when_one_session_exceeds_center_false_activations(tmp_path):
    rep_a = copy.deepcopy(SESSION_A_TEMPLATE)
    rep_a["timestamp_utc"] = "2026-10-29T12:00:00Z"

    # Session B has 2 center false activations out of 15: 13.3% > 5.0%
    rep_b = copy.deepcopy(SESSION_A_TEMPLATE)
    rep_b["timestamp_utc"] = "2026-10-30T12:00:00Z"
    rep_b["models"]["mobileone_s0_hybrid"]["evaluation_5dir"]["center_trials_false_act"] = 2
    rep_b["models"]["mobileone_s0_hybrid"]["evaluation_5dir"]["center_trials_stayed"] = 13
    rep_b["models"]["mobileone_s0_hybrid"]["evaluation_5dir"]["center_return_false_activation_rate"] = 2 / 15

    p_a = create_test_report_file(tmp_path, "rep_a.json", rep_a)
    p_b = create_test_report_file(tmp_path, "rep_b.json", rep_b)

    res = evaluate_cross_session_gate([p_a, p_b])
    assert res["status"] == "FAIL"
    assert any("failed center false-activation gate: 2/15 (13.3%) > allowed 5.0%" in r for r in res["reasons"])


def test_evaluate_gate_invalid_on_insufficient_sample_size(tmp_path):
    rep_a = copy.deepcopy(SESSION_A_TEMPLATE)
    rep_a["timestamp_utc"] = "2026-10-29T12:00:00Z"

    # Only 8 peripheral trials
    rep_b = copy.deepcopy(SESSION_A_TEMPLATE)
    rep_b["timestamp_utc"] = "2026-10-30T12:00:00Z"
    rep_b["models"]["mobileone_s0_hybrid"]["evaluation_5dir"]["peripheral_trials_passed"] = 8
    rep_b["models"]["mobileone_s0_hybrid"]["evaluation_5dir"]["peripheral_trials_total"] = 8
    rep_b["models"]["mobileone_s0_hybrid"]["evaluation_5dir"]["peripheral_trial_accuracy"] = 1.0

    p_a = create_test_report_file(tmp_path, "rep_a.json", rep_a)
    p_b = create_test_report_file(tmp_path, "rep_b.json", rep_b)

    res = evaluate_cross_session_gate([p_a, p_b])
    assert res["status"] == "INVALID"
    assert any("peripheral sample size insufficient: 8/12 required" in r for r in res["reasons"])


def test_evaluate_gate_is_strictly_read_only(tmp_path):
    """Verifies that evaluate_cross_session_gate never mutates input report files on disk."""
    rep_a = copy.deepcopy(SESSION_A_TEMPLATE)
    rep_b = copy.deepcopy(SESSION_A_TEMPLATE)
    rep_b["timestamp_utc"] = "2026-10-30T12:00:00Z"

    p_a = create_test_report_file(tmp_path, "rep_a.json", rep_a)
    p_b = create_test_report_file(tmp_path, "rep_b.json", rep_b)

    content_a_before = p_a.read_text(encoding="utf-8")
    content_b_before = p_b.read_text(encoding="utf-8")

    _ = evaluate_cross_session_gate([p_a, p_b])

    assert p_a.read_text(encoding="utf-8") == content_a_before
    assert p_b.read_text(encoding="utf-8") == content_b_before
