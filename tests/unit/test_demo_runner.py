"""
Unit tests for VoxGaze AAC Demo Runner & Bridge (Task D2b)

Tests:
1. Flag-off: no demo routes registered, is_selection_enabled remains False,
   PermissionError behavior unchanged on live selection attempt.
2. Direction converter hysteresis: no flicker around 0.35/0.40 and 0.60/0.65 boundaries.
3. Vertical axis (y) is completely ignored: x=0.10 with y=0.90 is still LEFT.
4. Simulated samples drive full "I NEED WATER" + SPEAK sequence to a fake speech backend.
5. Speech failure sets error and preserves phrase in phrase buffer.
6. Gaze input is strictly ignored while speaking.
7. Allow-keys flag enforcement: key simulation rejected without --allow-keys.
8. No image, video, or frame files written to disk under artifacts/.
"""

import os
from pathlib import Path
import time
from typing import List, Optional

import pytest

from backend.app.aac.board import AACBoard
from backend.app.aac.demo_perception import GazeSample
from backend.app.aac.demo_runner import DemoRunner, DirectionConverter
from backend.app.aac.demo_tree import DemoStateMachine
from backend.app.aac.server import AACService, create_app
from backend.app.aac.speech import TTSResult


class FakeSpeechEngine:
    """Mock TTS Engine recording synthesis requests."""

    def __init__(self, should_fail: bool = False, error_message: str = "Audio device error"):
        self.should_fail = should_fail
        self.error_message = error_message
        self.spoken_phrases: List[str] = []
        self.call_count: int = 0

    def speak(self, phrase: str) -> TTSResult:
        self.call_count += 1
        if self.should_fail:
            return TTSResult(
                success=False,
                phrase=phrase,
                engine_used="MOCK_FAIL",
                error_message=self.error_message,
            )
        self.spoken_phrases.append(phrase)
        return TTSResult(
            success=True,
            phrase=phrase,
            engine_used="MOCK_OFFLINE",
            error_message=None,
        )


def feed_continuous_samples(
    runner: DemoRunner,
    start_t: float,
    duration_s: float,
    x: Optional[float],
    y: Optional[float] = 0.50,
    valid: bool = True,
    face_detected: bool = True,
    step_s: float = 0.05,
) -> float:
    """Helper to feed continuous samples to runner at fixed time steps."""
    steps = int(round(duration_s / step_s))
    cur_t = start_t
    for i in range(steps):
        cur_t = start_t + (i + 1) * step_s
        runner.process_sample(
            timestamp=cur_t,
            x=x,
            y=y,
            valid=valid,
            face_detected=face_detected,
        )
    return cur_t


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

def test_flag_off_registers_no_demo_routes_and_selection_disabled():
    """Verify that with experimental_demo=False, no demo routes exist and live selection remains blocked."""
    app = create_app(experimental_demo=False)
    route_paths = [r.path for r in app.routes]

    # Invariant: Demo routes must NOT be present
    assert "/demo" not in route_paths
    assert "/api/demo/state" not in route_paths
    assert "/api/demo/reset" not in route_paths
    assert "/api/demo/pause" not in route_paths
    assert "/api/demo/key" not in route_paths

    # Invariant: LIVE mode selection remains strictly guarded by PermissionError
    srv = AACService()
    srv.set_mode("LIVE")
    assert srv.live_provider.is_selection_enabled is False
    with pytest.raises(PermissionError):
        srv.append_tile("tile_1_yes")


def test_direction_converter_hysteresis():
    """
    Verify dual-threshold hysteresis prevents jitter oscillation:
    - LEFT enter: x < 0.35, exit: x > 0.40
    - RIGHT enter: x > 0.65, exit: x < 0.60
    """
    conv = DirectionConverter(left_enter=0.35, left_exit=0.40, right_enter=0.65, right_exit=0.60)
    assert conv.state == "CENTER"

    # Moving toward left border but not crossing enter threshold
    assert conv.process(0.45) == "CENTER"
    assert conv.process(0.38) == "CENTER"
    assert conv.process(0.35) == "CENTER"

    # Crosses left enter threshold (< 0.35)
    assert conv.process(0.34) == "LEFT"

    # Jitter between 0.34 and 0.39 must NOT exit LEFT (exit is > 0.40)
    assert conv.process(0.38) == "LEFT"
    assert conv.process(0.39) == "LEFT"
    assert conv.process(0.35) == "LEFT"

    # Crosses exit threshold (> 0.40) -> returns to CENTER
    assert conv.process(0.41) == "CENTER"

    # Moving toward right border
    assert conv.process(0.62) == "CENTER"
    assert conv.process(0.65) == "CENTER"

    # Crosses right enter threshold (> 0.65)
    assert conv.process(0.66) == "RIGHT"

    # Jitter between 0.61 and 0.66 must NOT exit RIGHT (exit is < 0.60)
    assert conv.process(0.63) == "RIGHT"
    assert conv.process(0.61) == "RIGHT"

    # Crosses exit threshold (< 0.60) -> returns to CENTER
    assert conv.process(0.59) == "CENTER"


def test_vertical_axis_y_is_ignored():
    """Verify that vertical y coordinates have zero impact on directional decisions."""
    runner = DemoRunner(worker=None, fsm=DemoStateMachine(), speech_engine=FakeSpeechEngine())

    # Extreme vertical coordinates with x=0.10 (LEFT)
    d1 = runner.process_sample(timestamp=1.0, x=0.10, y=0.01)
    assert d1 == "LEFT"

    d2 = runner.process_sample(timestamp=1.1, x=0.10, y=0.99)
    assert d2 == "LEFT"

    # Return center
    runner.process_sample(timestamp=1.2, x=0.50, y=0.10)

    # Extreme vertical coordinates with x=0.90 (RIGHT)
    d3 = runner.process_sample(timestamp=1.3, x=0.90, y=0.05)
    assert d3 == "RIGHT"

    d4 = runner.process_sample(timestamp=1.4, x=0.90, y=0.95)
    assert d4 == "RIGHT"


def test_simulated_samples_drive_full_phrase_and_speak():
    """
    End-to-end integration test:
    Simulated samples drive:
    1. Select "I NEED" -> Confirm YES -> added to phrase.
    2. Select "WATER" -> Confirm YES -> added to phrase.
    3. Select "SPEAK" -> Confirm YES -> triggers speech with "I NEED WATER".
    """
    fake_speech = FakeSpeechEngine()
    fsm = DemoStateMachine()
    runner = DemoRunner(worker=None, fsm=fsm, speech_engine=fake_speech)

    t = 100.0

    # Arm initially at center (500 ms)
    t = feed_continuous_samples(runner, t, 0.6, x=0.50)
    assert runner.fsm.armed is True

    # --- WORD 1: "I NEED" ---
    # ROOT: Hold LEFT for 1.0s -> TALK
    t = feed_continuous_samples(runner, t, 1.1, x=0.20)
    assert runner.fsm.current_node.label == "TALK"
    assert runner.fsm.armed is False

    # Center re-arm (500 ms)
    t = feed_continuous_samples(runner, t, 0.6, x=0.50)
    assert runner.fsm.armed is True

    # TALK: Hold RIGHT for 1.0s -> NEEDS
    t = feed_continuous_samples(runner, t, 1.1, x=0.80)
    assert runner.fsm.current_node.label == "NEEDS"

    # Center re-arm
    t = feed_continuous_samples(runner, t, 0.6, x=0.50)
    assert runner.fsm.armed is True

    # NEEDS: Hold LEFT for 1.0s -> leaf "I NEED" -> enters CONFIRM
    t = feed_continuous_samples(runner, t, 1.1, x=0.20)
    assert runner.fsm.state == "CONFIRM"
    assert runner.fsm.confirm_leaf.word == "I NEED"

    # Center re-arm
    t = feed_continuous_samples(runner, t, 0.6, x=0.50)
    assert runner.fsm.armed is True

    # CONFIRM: Hold LEFT (YES) for 1.0s -> commits word
    t = feed_continuous_samples(runner, t, 1.1, x=0.20)
    assert runner.fsm.phrase == ["I NEED"]
    assert runner.fsm.state == "NODE"
    assert runner.fsm.current_node == runner.fsm.root_node

    # --- WORD 2: "WATER" ---
    # Center re-arm
    t = feed_continuous_samples(runner, t, 0.6, x=0.50)
    assert runner.fsm.armed is True

    # ROOT: Hold LEFT -> TALK
    t = feed_continuous_samples(runner, t, 1.1, x=0.20)
    assert runner.fsm.current_node.label == "TALK"

    # Center re-arm
    t = feed_continuous_samples(runner, t, 0.6, x=0.50)

    # TALK: Hold RIGHT -> NEEDS
    t = feed_continuous_samples(runner, t, 1.1, x=0.80)
    assert runner.fsm.current_node.label == "NEEDS"

    # Center re-arm
    t = feed_continuous_samples(runner, t, 0.6, x=0.50)

    # NEEDS: Hold RIGHT -> leaf "WATER" -> enters CONFIRM
    t = feed_continuous_samples(runner, t, 1.1, x=0.80)
    assert runner.fsm.state == "CONFIRM"
    assert runner.fsm.confirm_leaf.word == "WATER"

    # Center re-arm
    t = feed_continuous_samples(runner, t, 0.6, x=0.50)

    # CONFIRM: Hold LEFT (YES) -> commits "WATER"
    t = feed_continuous_samples(runner, t, 1.1, x=0.20)
    assert runner.fsm.phrase == ["I NEED", "WATER"]

    # --- ACTION: "SPEAK" ---
    # Center re-arm
    t = feed_continuous_samples(runner, t, 0.6, x=0.50)

    # ROOT: Hold RIGHT -> ALERT & ACTIONS
    t = feed_continuous_samples(runner, t, 1.1, x=0.80)
    assert runner.fsm.current_node.label == "ALERT & ACTIONS"

    # Center re-arm
    t = feed_continuous_samples(runner, t, 0.6, x=0.50)

    # ALERT & ACTIONS: Hold RIGHT -> ACTIONS
    t = feed_continuous_samples(runner, t, 1.1, x=0.80)
    assert runner.fsm.current_node.label == "ACTIONS"

    # Center re-arm
    t = feed_continuous_samples(runner, t, 0.6, x=0.50)

    # ACTIONS: Hold LEFT -> leaf "SPEAK" -> enters CONFIRM
    t = feed_continuous_samples(runner, t, 1.1, x=0.20)
    assert runner.fsm.state == "CONFIRM"
    assert runner.fsm.confirm_leaf.action == "SPEAK"

    # Center re-arm
    t = feed_continuous_samples(runner, t, 0.6, x=0.50)

    # CONFIRM: Hold LEFT (YES) -> commits SPEAK
    t = feed_continuous_samples(runner, t, 1.1, x=0.20)

    # Wait for daemon speech thread to finish
    if runner._speech_thread is not None:
        runner._speech_thread.join(timeout=1.0)

    # Verify speech output
    assert fake_speech.call_count == 1
    assert fake_speech.spoken_phrases == ["I NEED WATER"]
    assert runner.fsm.phrase == ["I NEED", "WATER"]


def test_speech_failure_records_error_and_preserves_phrase():
    """Verify that speech failure updates speech_error and keeps phrase intact."""
    fake_speech = FakeSpeechEngine(should_fail=True, error_message="Synthetic audio card failure")
    fsm = DemoStateMachine()
    fsm.phrase = ["I NEED", "HELP"]
    runner = DemoRunner(worker=None, fsm=fsm, speech_engine=fake_speech)

    runner._trigger_speech("I NEED HELP")
    if runner._speech_thread is not None:
        runner._speech_thread.join(timeout=1.0)

    assert runner.speaking is False
    assert runner.last_speech_error == "Synthetic audio card failure"
    assert runner.fsm.phrase == ["I NEED", "HELP"]


def test_gaze_input_ignored_while_speaking():
    """Verify that incoming gaze samples are ignored when speaking flag is active."""
    runner = DemoRunner(worker=None, fsm=DemoStateMachine(), speech_engine=FakeSpeechEngine())
    runner.speaking = True

    # Try feeding LEFT gaze while speaking
    res = runner.process_sample(timestamp=10.0, x=0.10)
    assert res == "SPEAKING"
    # State machine must remain at root
    assert runner.fsm.current_node == runner.fsm.root_node
    assert runner.fsm.hold_dir is None


def test_allow_keys_flag_enforcement():
    """Verify key simulation is rejected when allow_keys=False and allowed when allow_keys=True."""
    # When allow_keys is False:
    runner_no_keys = DemoRunner(worker=None, fsm=DemoStateMachine(), allow_keys=False)
    with pytest.raises(PermissionError):
        runner_no_keys.inject_key("left")

    app_no_keys = create_app(experimental_demo=True, allow_keys=False)
    route_paths = [r.path for r in app_no_keys.routes]
    assert "/api/demo/key" not in route_paths

    # When allow_keys is True:
    runner_with_keys = DemoRunner(worker=None, fsm=DemoStateMachine(), allow_keys=True)
    runner_with_keys.inject_key("center")
    assert runner_with_keys.current_direction == "CENTER"

    app_with_keys = create_app(experimental_demo=True, allow_keys=True)
    route_paths_keys = [r.path for r in app_with_keys.routes]
    assert "/api/demo/key" in route_paths_keys


def test_tracking_loss_sample_resets_direction_to_none():
    """Verify face_detected=False or valid=False maps to NONE and resets converter."""
    runner = DemoRunner(worker=None, fsm=DemoStateMachine(), speech_engine=FakeSpeechEngine())

    # Valid LEFT
    assert runner.process_sample(timestamp=1.0, x=0.10, valid=True, face_detected=True) == "LEFT"

    # Face lost
    assert runner.process_sample(timestamp=1.1, x=0.10, valid=True, face_detected=False) == "NONE"
    assert runner.current_direction == "NONE"
    assert runner.converter.state == "CENTER"

    # Tracking invalid
    assert runner.process_sample(timestamp=1.2, x=0.10, valid=False, face_detected=True) == "NONE"


def test_no_image_or_frame_files_written_to_disk():
    """Verify that running demo runner produces zero image/video artifact files."""
    obs_dir = Path("artifacts/observations")
    images_before = list(obs_dir.glob("*.jpg")) + list(obs_dir.glob("*.png")) + list(obs_dir.glob("*.mp4"))
    assert len(images_before) == 0

    fake_speech = FakeSpeechEngine()
    runner = DemoRunner(worker=None, fsm=DemoStateMachine(), speech_engine=fake_speech)
    t = 1.0
    for _ in range(20):
        t += 0.05
        runner.process_sample(timestamp=t, x=0.20)

    images_after = list(obs_dir.glob("*.jpg")) + list(obs_dir.glob("*.png")) + list(obs_dir.glob("*.mp4"))
    assert len(images_after) == 0
