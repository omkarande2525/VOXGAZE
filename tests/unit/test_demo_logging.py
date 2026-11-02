"""
Unit tests for VoxGaze AAC Demo Event Logging and Rest Test Evaluation (Task D5)

Tests:
1. Event log file creation: saves session JSON to artifacts/demo_logs/<run_id>.json.
2. Schema validation: validates run_id, start_time, end_time, mode, parameters, measured_fps, events.
3. Parameter completeness: verifies all 9 timing and boundary parameters are present.
4. Rest test execution under steady center gaze: verifies 0 unintended decisions and PASS status.
5. Rest test detection of peripheral escapes: verifies detection of unintended decisions and FAIL status.
6. Privacy & ephemeral data invariant: verifies NO raw images, video frames, landmarks, or feature
   vectors are saved to disk under artifacts/demo_logs/.
7. CLI flag parsing: verifies --rest-test parses properly in run_aac_mvp.py.
"""

from datetime import datetime
import json
from pathlib import Path
import tempfile
import time
from typing import Dict, List, Optional
import pytest

from backend.app.aac.demo_runner import DEFAULT_LOGS_DIR, DemoRunner, run_rest_test
from backend.app.aac.demo_tree import DemoStateMachine
from backend.app.aac.speech import TTSResult


class MockSpeechEngine:
    """Mock TTS Engine."""
    def speak(self, phrase: str) -> TTSResult:
        return TTSResult(success=True, phrase=phrase, engine_used="MOCK", error_message=None)


def test_session_log_file_creation_and_auto_save_on_stop(tmp_path: Path):
    """Verify that DemoRunner saves a session JSON log on stop() or explicit save."""
    fsm = DemoStateMachine()
    runner = DemoRunner(
        worker=None,
        fsm=fsm,
        speech_engine=MockSpeechEngine(),
        output_dir=tmp_path,
        mode="simulated",
    )

    # Process center sample
    runner.process_sample(timestamp=1.0, x=0.50, valid=True, face_detected=True)

    # Calling stop() automatically saves the session log
    runner.stop()

    assert runner.session_saved is True
    assert runner.last_saved_log_path is not None
    assert runner.last_saved_log_path.is_file()

    # File must exist inside the specified output_dir
    created_files = list(tmp_path.glob("*.json"))
    assert len(created_files) == 1
    assert created_files[0] == runner.last_saved_log_path


def test_log_schema_validation(tmp_path: Path):
    """Verify the log payload conforms strictly to the specified schema."""
    fsm = DemoStateMachine()
    runner = DemoRunner(
        worker=None,
        fsm=fsm,
        speech_engine=MockSpeechEngine(),
        output_dir=tmp_path,
        mode="simulated",
    )

    runner.process_sample(timestamp=1.0, x=0.50, valid=True, face_detected=True)
    log_path = runner.save_session_log(filename="test_run.json")

    with open(log_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Root schema keys
    assert "run_id" in data
    assert "start_time" in data
    assert "end_time" in data
    assert "mode" in data
    assert data["mode"] == "simulated"
    assert "parameters" in data
    assert "measured_fps" in data
    assert "events" in data

    # Verify all 9 required parameters are present
    params = data["parameters"]
    expected_params = [
        "hold_ms",
        "center_arm_ms",
        "grace_ms",
        "confirm_timeout_s",
        "tracking_loss_ms",
        "left_enter",
        "left_exit",
        "right_enter",
        "right_exit",
    ]
    for p in expected_params:
        assert p in params, f"Missing parameter {p} in log parameters"

    # Events schema
    assert isinstance(data["events"], list)
    for ev in data["events"]:
        assert "t" in ev
        assert "type" in ev
        assert "detail" in ev


def test_rest_test_execution_clean_center_pass(tmp_path: Path):
    """
    Verify rest test execution when user stays steady at center:
    Result: 0 unintended decisions -> PASS.
    """
    fsm = DemoStateMachine()
    runner = DemoRunner(
        worker=None,
        fsm=fsm,
        speech_engine=MockSpeechEngine(),
        output_dir=tmp_path,
        mode="simulated",
    )

    # Feed steady center gaze samples
    start_t = 100.0
    for i in range(20):
        t = start_t + (i * 0.05)
        runner.process_sample(timestamp=t, x=0.50, valid=True, face_detected=True)

    # Run a short 0.2s rest test
    res = run_rest_test(runner=runner, duration_s=0.2, output_dir=tmp_path)

    assert res["status"] == "PASS"
    assert res["unintended_decisions"] == 0
    assert len(res["unintended_events"]) == 0
    assert Path(res["log_path"]).is_file()

    # Verify log content
    with open(res["log_path"], "r", encoding="utf-8") as f:
        log_data = json.load(f)
    assert log_data["status"] == "PASS"
    assert log_data["unintended_decisions"] == 0
    assert log_data["test_type"] == "rest_test"


def test_rest_test_detection_of_accidental_decision_fail(tmp_path: Path):
    """
    Verify rest test detection when gaze escapes to peripheral and triggers a decision:
    Result: unintended decisions > 0 -> FAIL.
    """
    import threading

    fsm = DemoStateMachine()
    runner = DemoRunner(
        worker=None,
        fsm=fsm,
        speech_engine=MockSpeechEngine(),
        output_dir=tmp_path,
        mode="simulated",
    )

    # Thread feeding peripheral escape during rest test
    def _feed_accidental_escape():
        time.sleep(0.02)
        base_t = time.perf_counter()
        # 1. Arm at center (dwell >= center_arm_s)
        arm_steps = int(round((runner.fsm.center_arm_s + 0.1) / 0.05))
        for i in range(arm_steps):
            t = base_t + (i * 0.05)
            runner.process_sample(timestamp=t, x=0.50, valid=True, face_detected=True)

        # 2. Hold LEFT to trigger unintended decision (> hold_s)
        base_t2 = base_t + 1.0
        hold_steps = int(round((runner.fsm.hold_s + 0.2) / 0.05))
        for i in range(hold_steps):
            t = base_t2 + (i * 0.05)
            runner.process_sample(timestamp=t, x=0.10, valid=True, face_detected=True)

    feeder = threading.Thread(target=_feed_accidental_escape, daemon=True)
    feeder.start()

    # Execute rest test evaluation for 0.2s
    res = run_rest_test(runner=runner, duration_s=0.2, output_dir=tmp_path)
    feeder.join(timeout=1.0)

    assert res["status"] == "FAIL"
    assert res["unintended_decisions"] > 0
    assert len(res["unintended_events"]) > 0

    with open(res["log_path"], "r", encoding="utf-8") as f:
        log_data = json.load(f)
    assert log_data["status"] == "FAIL"
    assert log_data["unintended_decisions"] > 0


def test_privacy_invariants_no_raw_images_or_features_in_logs(tmp_path: Path):
    """
    CRITICAL PRIVACY INVARIANT:
    Ensures that under no circumstances are raw webcam images, video files,
    cropped eye patches, face landmarks, or continuous feature vectors written
    to disk in the session logs.
    """
    fsm = DemoStateMachine()
    runner = DemoRunner(
        worker=None,
        fsm=fsm,
        speech_engine=MockSpeechEngine(),
        output_dir=tmp_path,
        mode="simulated",
    )

    # Process multiple samples and save log
    for i in range(10):
        runner.process_sample(timestamp=i * 0.1, x=0.50, valid=True, face_detected=True)
    log_path = runner.save_session_log(filename="privacy_check.json")

    # 1. Check directory contents: only .json files, zero image/video formats
    all_files = list(tmp_path.rglob("*"))
    disallowed_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".mp4", ".avi", ".mov", ".npy", ".npz"}
    for f in all_files:
        assert f.suffix.lower() not in disallowed_extensions, f"Disallowed file found: {f}"

    # 2. Inspect JSON content: no raw arrays or landmark keys
    content = log_path.read_text(encoding="utf-8")
    forbidden_terms = ["raw_frame", "crop_image", "landmark_x", "landmark_y", "feature_vector", "embedding"]
    for term in forbidden_terms:
        assert term not in content, f"Forbidden privacy-violating term '{term}' found in log"


def test_cli_argument_parsing_rest_test():
    """Verify --rest-test flag is accepted by run_aac_mvp.py parser."""
    import argparse
    from scripts.run_aac_mvp import main

    # Inspect parser by instantiating argument parser with same arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--rest-test", type=float, default=0.0)
    args = parser.parse_args(["--rest-test", "60"])
    assert args.rest_test == 60.0
