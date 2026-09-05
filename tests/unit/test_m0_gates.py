"""
Milestone M0 Automated Acceptance Gate Tests
Verifies governance files, schema validity, data contracts, and anti-scope-creep invariants.
"""

import json
from pathlib import Path
import pytest
from backend.app.core.contracts import (
    Point2D,
    EyeLandmarks,
    GazeFeatures,
    DirectionThreshold,
    CalibrationThresholds,
    CalibrationStats,
    CalibrationProfile,
    Direction,
    GazeState,
    IntentCandidate,
    InteractionStateEnum,
    PreviewState,
    InteractionEvent,
)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def test_governance_and_doc_files_exist():
    """Verify all mandatory M0 governance and architectural documentation files exist."""
    required_files = [
        "AGENTS.md",
        "README.md",
        ".gitignore",
        ".env.example",
        "requirements.txt",
        "pyproject.toml",
        "docs/acceptance-gates.md",
        "docs/data-contracts.md",
        "docs/non-goals.md",
        "docs/DECISIONS.md",
        "docs/architecture.md",
        "docs/research-plan.md",
        "docs/ui-principles.md",
        "config/default_intents.json",
        ".agents/rules/voxgaze-core.md",
        ".agents/rules/safety.md",
        ".agents/rules/testing.md",
        ".agents/skills/voxgaze-engineering/SKILL.md",
        ".agents/skills/gaze-validation/SKILL.md",
        ".agents/skills/accessibility-ui/SKILL.md",
        ".codex/skills/voxgaze-engineering/SKILL.md",
        ".codex/skills/gaze-validation/SKILL.md",
        ".codex/skills/accessibility-ui/SKILL.md",
    ]
    for rel_path in required_files:
        p = ROOT_DIR / rel_path
        assert p.is_file(), f"Missing required M0 file: {rel_path}"


def test_default_intents_json_validity():
    """Verify default_intents.json loads, has version, fast_path, and domain hierarchy."""
    intents_file = ROOT_DIR / "config" / "default_intents.json"
    with open(intents_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("version") == 1
    assert "fast_path" in data
    assert "domains" in data

    # Verify all 4 fast-path directions exist
    for direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
        assert direction in data["fast_path"], f"Missing fast-path direction {direction}"
        item = data["fast_path"][direction]
        assert "id" in item
        assert "label" in item
        assert "phrase" in item

    # Verify standard domains exist
    for domain in ["MEDICAL", "BASIC", "SOCIAL"]:
        assert domain in data["domains"], f"Missing standard domain {domain}"
        assert "items" in data["domains"][domain]


def test_pydantic_data_contracts():
    """Verify core Pydantic contracts instantiate and serialize properly."""
    # 1. Landmarks & Bilateral features
    pt = Point2D(x=100.0, y=150.0)
    eye = EyeLandmarks(
        iris_center=pt,
        inner_canthus=Point2D(x=80.0, y=150.0),
        outer_canthus=Point2D(x=120.0, y=150.0),
        upper_lid=Point2D(x=100.0, y=140.0),
        lower_lid=Point2D(x=100.0, y=160.0),
        h_ratio=0.5,
        v_ratio=0.5,
        ear=0.3,
    )
    features = GazeFeatures(
        timestamp_ms=1000,
        left_eye=eye,
        tracking_confidence=0.95,
        is_tracking_valid=True,
    )
    assert features.left_eye.h_ratio == 0.5
    assert features.right_eye is None

    # 2. Calibration profile
    stats = CalibrationStats(
        sample_count=45,
        median_h=0.51,
        median_v=0.49,
        mad_h=0.015,
        mad_v=0.018,
        min_h=0.48,
        max_h=0.54,
        min_v=0.45,
        max_v=0.53,
    )
    thresholds = CalibrationThresholds(
        left=DirectionThreshold(enter=0.40, exit=0.45),
        right=DirectionThreshold(enter=0.62, exit=0.57),
        up=DirectionThreshold(enter=0.34, exit=0.39),
        down=DirectionThreshold(enter=0.65, exit=0.60),
        blink_closed_ear=0.15,
        blink_open_ear=0.30,
    )
    profile = CalibrationProfile(
        user_id="U01",
        created_at_utc="2026-09-05T00:00:00Z",
        camera_index=0,
        camera_resolution=(1280, 720),
        stats={"CENTER": stats},
        thresholds=thresholds,
    )
    assert profile.user_id == "U01"
    assert profile.thresholds.left.enter == 0.40

    # 3. Gaze State
    state = GazeState(
        timestamp_ms=1050,
        direction=Direction.RIGHT,
        raw_h=0.68,
        raw_v=0.50,
        ear=0.30,
        confidence=0.92,
        status_flag="● TRACKING",
    )
    assert state.direction == Direction.RIGHT


def test_no_premature_future_modules():
    """Verify downstream modules (interaction, intent, speech, etc.) have no premature code."""
    for mod in ["interaction", "intent", "speech", "safety", "telemetry"]:
        pkg_dir = ROOT_DIR / "backend" / "app" / mod
        py_files = list(pkg_dir.glob("*.py"))
        assert len(py_files) == 0, f"Premature Python implementation detected in {mod}: {py_files}"
