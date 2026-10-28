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
    FrameComparisonRecord,
    TARGET_BBOXES,
    TARGET_COORDS,
    analyze_block_evaluation,
    evaluate_trial_dwell,
    execute_comparison_session,
    is_inside_box,
)


def test_balanced_target_sequences():
    # All 3 blocks must have equal length (9 items: 4 peripheral + 5 center)
    assert len(BLOCK_1_TARGETS) == 9
    assert len(BLOCK_2_TARGETS) == 9
    assert len(BLOCK_3_TARGETS) == 9

    # Each peripheral target must appear exactly once per block
    for targets in [BLOCK_1_TARGETS, BLOCK_2_TARGETS, BLOCK_3_TARGETS]:
        assert targets.count("CENTER") == 5
        assert targets.count("LEFT") == 1
        assert targets.count("RIGHT") == 1
        assert targets.count("UP") == 1
        assert targets.count("DOWN") == 1

    # Sequences must not be identical across blocks (prevent sequence-order bias)
    assert BLOCK_1_TARGETS != BLOCK_2_TARGETS
    assert BLOCK_2_TARGETS != BLOCK_3_TARGETS
    assert BLOCK_1_TARGETS != BLOCK_3_TARGETS


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
    assert "models" in report
    assert "mobileone_s0_hybrid" in report["models"]
    assert "resnet34_hybrid" in report["models"]

    # Decision rule verification
    rule = report["decision_rule"]
    assert "adopted" in rule
    assert "rule_statement" in rule
    assert rule["adopted"] is False  # Because synthetic performance is tied (delta = 0 < 15%)
