"""
Unit tests for VoxGaze AAC Demo: Pure Binary Gaze-Tree State Machine (Task D1 & D5)

Verifies:
1. Tree configuration loads correctly and exposes all 16 vocabulary choices.
2. A hold shorter than HOLD_MS does nothing.
3. A hold of HOLD_MS commits exactly one level.
4. A continuous long hold CANNOT commit multiple levels without center re-arming.
5. Without center_arm_ms at centre, a new hold is ignored.
6. Reaching leaf displays confirm screen; word is NEVER added without confirm YES.
7. Confirm NO or timeout returns to ROOT.
8. UP, DOWN, or NONE never advance tree.
9. NONE > 300 ms pauses state machine cleanly.
10. Brief dropouts <= 150 ms grace do not reset hold timer.
11. UNDO removes last word; SPEAK generates speak_request event with phrase.
12. Timestamps strictly drive state transitions, not call counts.
"""

from pathlib import Path
import random
import pytest

from backend.app.aac.demo_tree import DemoScreen, DemoStateMachine, DemoTreeStateMachine, TreeEvent


# ---------------------------------------------------------------------------
# Test Fixtures & Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def fsm():
    """Returns a fresh DemoStateMachine loaded from demo_config.json."""
    return DemoStateMachine()


def feed_stream(sm: DemoStateMachine, start_t: float, duration_s: float, direction: str, step_s: float = 0.05) -> float:
    """Feeds a continuous stream of directional samples for duration_s at step_s intervals."""
    steps = int(round(duration_s / step_s))
    for i in range(steps + 1):
        t = start_t + (i * step_s)
        sm.on_sample(round(t, 5), direction)
    return round(start_t + duration_s, 5)


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

def test_tree_loads_from_config_and_exposes_all_16_leaves(fsm):
    """Verify that tree loads from demo_config.json and contains all 16 vocabulary leaves."""
    leaves = fsm.get_leaves()
    assert len(leaves) == 16

    leaf_texts = [leaf.get("word") or leaf.get("action") for leaf in leaves]
    expected_leaves = {
        "YES", "NO", "I NEED", "HELP", "WATER", "FOOD", "DOCTOR", "REST",
        "PAIN", "CHEST PAIN", "FEEL UNCOMFORTABLE", "SICK",
        "SPEAK", "UNDO", "OK", "PLEASE"
    }
    assert set(leaf_texts) == expected_leaves

    # Check root screen metadata
    screen = fsm.current_screen
    assert screen.kind == "node"
    assert screen.title == "ROOT"
    assert screen.left_label == "TALK & NEEDS"
    assert screen.right_label == "MEDICAL & ACTIONS"


def test_hold_shorter_than_hold_ms_does_nothing(fsm):
    """Verify that holding a direction for less than HOLD_MS causes no transition."""
    t = 0.0
    arm_dur = fsm.center_arm_s + 0.05
    short_hold = fsm.hold_s * 0.7

    # 1. Arm at CENTER
    t = feed_stream(fsm, t, arm_dur, "CENTER")
    assert fsm.armed is True
    assert fsm.current_screen.title == "ROOT"

    # 2. Hold LEFT shorter than hold_s
    t = feed_stream(fsm, t, short_hold, "LEFT")
    assert fsm.current_screen.title == "ROOT"  # Stays at ROOT
    assert fsm.hold_progress > 0.5

    # 3. Return to CENTER
    t = feed_stream(fsm, t, 0.20, "CENTER")
    assert fsm.current_screen.title == "ROOT"
    assert fsm.hold_progress == 0.0  # Hold progress cleared


def test_hold_of_hold_ms_commits_exactly_one_level(fsm):
    """Verify that a full HOLD_MS hold commits exactly one hierarchical level."""
    t = 0.0
    arm_dur = fsm.center_arm_s + 0.05
    hold_dur = fsm.hold_s + 0.05

    # Arm at CENTER
    t = feed_stream(fsm, t, arm_dur, "CENTER")
    assert fsm.armed is True

    # Hold LEFT for full hold_s
    t = feed_stream(fsm, t, hold_dur, "LEFT")
    # Transitions to TALK & NEEDS
    assert fsm.current_screen.title == "TALK & NEEDS"
    assert fsm.current_screen.left_label == "TALK"
    assert fsm.current_screen.right_label == "NEEDS"
    # Disarms until center rest
    assert fsm.armed is False


def test_long_left_hold_cannot_commit_two_levels(fsm):
    """CRITICAL SAFETY INVARIANT: A continuous 3000 ms hold cannot jump two levels."""
    t = 0.0
    arm_dur = fsm.center_arm_s + 0.05
    t = feed_stream(fsm, t, arm_dur, "CENTER")
    assert fsm.armed is True

    # Hold LEFT continuously for 3.5 seconds
    t = feed_stream(fsm, t, 3.50, "LEFT")
    # First level committed to TALK & NEEDS, but second level (TALK) must NOT commit without center re-arming
    assert fsm.current_screen.title == "TALK & NEEDS"
    assert fsm.armed is False


def test_without_500ms_at_centre_new_hold_is_ignored(fsm):
    """Verify that returning to CENTER for less than center_arm_ms does NOT arm the next choice."""
    t = 0.0
    arm_dur = fsm.center_arm_s + 0.05
    hold_dur = fsm.hold_s + 0.05
    short_center = fsm.center_arm_s * 0.4

    # Arm and choose TALK & NEEDS
    t = feed_stream(fsm, t, arm_dur, "CENTER")
    t = feed_stream(fsm, t, hold_dur, "LEFT")
    assert fsm.current_screen.title == "TALK & NEEDS"
    assert fsm.armed is False

    # Return to CENTER for less than center_arm_s
    t = feed_stream(fsm, t, short_center, "CENTER")
    assert fsm.armed is False

    # Try to hold LEFT again
    t = feed_stream(fsm, t, hold_dur, "LEFT")
    # Must still be at TALK & NEEDS (not TALK) because state machine was never re-armed
    assert fsm.current_screen.title == "TALK & NEEDS"


def test_confirm_yes_adds_word_no_and_timeout_add_nothing(fsm):
    """
    Verify confirmation loop:
    1. Reaching leaf displays confirm screen without adding word.
    2. Confirm YES adds word.
    3. Confirm NO cancels back to ROOT without adding word.
    4. Confirm Timeout cancels back to ROOT without adding word.
    """
    t = 0.0
    arm_dur = fsm.center_arm_s + 0.05
    hold_dur = fsm.hold_s + 0.05

    def navigate_to_water(start_t: float) -> float:
        cur = start_t
        # ROOT -> TALK & NEEDS (LEFT)
        cur = feed_stream(fsm, cur, arm_dur, "CENTER")
        cur = feed_stream(fsm, cur, hold_dur, "LEFT")
        # TALK & NEEDS -> NEEDS (RIGHT)
        cur = feed_stream(fsm, cur, arm_dur, "CENTER")
        cur = feed_stream(fsm, cur, hold_dur, "RIGHT")
        # NEEDS -> CONSUME (LEFT)
        cur = feed_stream(fsm, cur, arm_dur, "CENTER")
        cur = feed_stream(fsm, cur, hold_dur, "LEFT")
        # CONSUME -> WATER (LEFT)
        cur = feed_stream(fsm, cur, arm_dur, "CENTER")
        cur = feed_stream(fsm, cur, hold_dur, "LEFT")
        return cur

    # 1. Navigate to WATER
    t = navigate_to_water(t)
    assert fsm.current_screen.kind == "confirm"
    assert fsm.current_screen.candidate == "WATER"
    assert fsm.phrase == []  # Word is NOT added yet

    # Confirm YES (re-arm at center, then hold LEFT)
    t = feed_stream(fsm, t, arm_dur, "CENTER")
    t = feed_stream(fsm, t, hold_dur, "LEFT")
    assert fsm.phrase == ["WATER"]
    assert fsm.current_screen.title == "ROOT"

    # 2. Navigate to WATER again and test Confirm NO
    t = navigate_to_water(t)
    assert fsm.current_screen.kind == "confirm"
    t = feed_stream(fsm, t, arm_dur, "CENTER")
    t = feed_stream(fsm, t, hold_dur, "RIGHT")  # NO cancels
    assert fsm.phrase == ["WATER"]  # Unchanged
    assert fsm.current_screen.title == "ROOT"

    # 3. Navigate to WATER again and test Timeout
    t = navigate_to_water(t)
    assert fsm.current_screen.kind == "confirm"
    # Idle at CENTER for 10.5 seconds (timeout = 10s)
    t = feed_stream(fsm, t, 10.50, "CENTER")
    assert fsm.current_screen.title == "ROOT"  # Timed out to ROOT
    assert fsm.phrase == ["WATER"]  # Unchanged


def test_property_word_is_never_added_without_confirm_screen(fsm):
    """
    Property test: Random sequence of inputs without explicit confirm screen
    can never append a word to the phrase buffer.
    """
    t = 0.0
    random.seed(42)
    directions = ["LEFT", "RIGHT", "CENTER", "NONE"]

    for _ in range(200):
        d = random.choice(directions)
        dur = random.uniform(0.05, 0.40)  # Short durations, never reaching full confirm sequence
        t = feed_stream(fsm, t, dur, d)
        # If we accidentally entered confirm screen, we do not confirm it
        if fsm.current_screen.kind == "confirm":
            fsm.reset(t)

    assert fsm.phrase == []


def test_up_down_and_none_never_trigger_transitions(fsm):
    """Verify that UP, DOWN, or NONE signals never advance the tree or arm choices."""
    t = 0.0
    t = feed_stream(fsm, t, 2.0, "UP")
    assert fsm.current_screen.title == "ROOT"
    assert fsm.armed is False

    t = feed_stream(fsm, t, 2.0, "DOWN")
    assert fsm.current_screen.title == "ROOT"
    assert fsm.armed is False

    t = feed_stream(fsm, t, 2.0, "NONE")
    assert fsm.current_screen.title == "ROOT"
    assert fsm.armed is False


def test_none_longer_than_tracking_loss_ms_pauses_and_clears_hold(fsm):
    """Verify that tracking loss (NONE >= 300 ms) pauses tracking, disarms, and resets active hold."""
    t = 0.0
    arm_dur = fsm.center_arm_s + 0.05
    # Arm and start holding LEFT for 500 ms
    t = feed_stream(fsm, t, arm_dur, "CENTER")
    t = feed_stream(fsm, t, 0.50, "LEFT")
    assert fsm.hold_progress > 0.3

    # Tracking drops for 400 ms (> 300 ms TRACKING_LOSS_MS)
    t = feed_stream(fsm, t, 0.40, "NONE")
    assert fsm.paused is True
    assert fsm.armed is False
    assert fsm.hold_progress == 0.0

    # Tracking resumes at CENTER
    t = feed_stream(fsm, t, 0.10, "CENTER")
    assert fsm.paused is False


def test_grace_window_preserves_hold_during_brief_dropout(fsm):
    """
    Verify that a brief dropout <= GRACE_MS (150 ms) does NOT reset an active hold.
    """
    t = 0.0
    arm_dur = fsm.center_arm_s + 0.05
    half_hold = fsm.hold_s * 0.5
    rem_hold = fsm.hold_s * 0.55

    t = feed_stream(fsm, t, arm_dur, "CENTER")

    # 1. Hold LEFT for half duration
    t = feed_stream(fsm, t, half_hold, "LEFT")
    assert fsm.hold_progress >= 0.40

    # 2. Brief 100 ms dropout to CENTER (<= 150 ms GRACE_MS)
    t = feed_stream(fsm, t, 0.10, "CENTER")
    assert fsm.current_screen.title == "ROOT"  # Stays at ROOT

    # 3. Resume LEFT to complete hold
    t = feed_stream(fsm, t, rem_hold, "LEFT")
    # Total hold elapsed >= hold_s -> commits to TALK & NEEDS!
    assert fsm.current_screen.title == "TALK & NEEDS"


def test_undo_and_speak_behave_properly(fsm):
    """
    Verify ACTIONS branch:
    1. SPEAK emits a speak_request event with assembled phrase.
    2. UNDO removes the last word.
    3. UNDO on empty phrase is completely safe.
    """
    t = 0.0
    arm_dur = fsm.center_arm_s + 0.05
    hold_dur = fsm.hold_s + 0.05

    # Manually populate phrase buffer
    fsm.phrase = ["I NEED", "WATER"]

    def navigate_to_control(start_t: float) -> float:
        cur = start_t
        # ROOT -> MEDICAL & ACTIONS (RIGHT)
        cur = feed_stream(fsm, cur, arm_dur, "CENTER")
        cur = feed_stream(fsm, cur, hold_dur, "RIGHT")
        # MEDICAL & ACTIONS -> ACTIONS (RIGHT)
        cur = feed_stream(fsm, cur, arm_dur, "CENTER")
        cur = feed_stream(fsm, cur, hold_dur, "RIGHT")
        # ACTIONS -> CONTROL (LEFT)
        cur = feed_stream(fsm, cur, arm_dur, "CENTER")
        cur = feed_stream(fsm, cur, hold_dur, "LEFT")
        return cur

    # 1. Trigger UNDO (RIGHT in CONTROL)
    t = navigate_to_control(t)
    assert fsm.current_screen.title == "CONTROL"
    assert fsm.current_screen.right_label == "UNDO"

    t = feed_stream(fsm, t, arm_dur, "CENTER")
    t = feed_stream(fsm, t, hold_dur, "RIGHT")  # Select UNDO
    assert fsm.current_screen.kind == "confirm"
    assert fsm.current_screen.candidate == "UNDO"

    # Confirm UNDO
    t = feed_stream(fsm, t, arm_dur, "CENTER")
    t = feed_stream(fsm, t, hold_dur, "LEFT")  # YES
    assert fsm.phrase == ["I NEED"]  # "WATER" removed

    # 2. Trigger SPEAK (LEFT in CONTROL)
    t = navigate_to_control(t)
    assert fsm.current_screen.title == "CONTROL"
    assert fsm.current_screen.left_label == "SPEAK"

    t = feed_stream(fsm, t, arm_dur, "CENTER")
    t = feed_stream(fsm, t, hold_dur, "LEFT")  # Select SPEAK
    assert fsm.current_screen.kind == "confirm"
    assert fsm.current_screen.candidate == "SPEAK"

    # Confirm SPEAK
    t = feed_stream(fsm, t, arm_dur, "CENTER")
    t = feed_stream(fsm, t, hold_dur, "LEFT")  # YES

    speak_events = [e for e in fsm.events if e.type == "speak_request"]
    assert len(speak_events) >= 1
    assert speak_events[-1].detail == "I NEED"
    assert fsm.phrase == ["I NEED"]  # Phrase preserved

    # 3. UNDO on empty phrase safety check
    fsm.phrase.clear()
    t = navigate_to_control(t)
    t = feed_stream(fsm, t, arm_dur, "CENTER")
    t = feed_stream(fsm, t, hold_dur, "RIGHT")  # Select UNDO
    t = feed_stream(fsm, t, arm_dur, "CENTER")
    t = feed_stream(fsm, t, hold_dur, "LEFT")   # Confirm YES
    assert fsm.phrase == []  # Safe, no exception


def test_timestamps_not_call_counts_drive_timing(fsm):
    """Verify that irregular sample intervals evaluate strictly based on timestamps, not call counts."""
    # Samples over 100 ms -> should NOT arm (center_arm_s needed)
    for i in range(100):
        fsm.on_sample(i * 0.001, "CENTER")
    assert fsm.armed is False

    # 2 samples spanning center_arm_s + 0.05s -> arms!
    fsm.on_sample(1.0, "CENTER")
    fsm.on_sample(1.0 + fsm.center_arm_s + 0.05, "CENTER")
    assert fsm.armed is True


def test_reset_returns_to_root_from_any_state(fsm):
    """Verify that calling reset() immediately aborts back to ROOT and clears hold."""
    t = 0.0
    arm_dur = fsm.center_arm_s + 0.05
    hold_dur = fsm.hold_s + 0.05

    t = feed_stream(fsm, t, arm_dur, "CENTER")
    t = feed_stream(fsm, t, hold_dur, "LEFT")  # TALK & NEEDS
    assert fsm.current_screen.title == "TALK & NEEDS"

    fsm.reset(t)
    assert fsm.current_screen.title == "ROOT"
    assert fsm.state == "NODE"
    assert fsm.hold_progress == 0.0
    assert fsm.armed is False


def test_demo_tree_state_machine_interface_and_properties():
    """Verify Task D1 DemoTreeStateMachine class and property outputs."""
    tree_sm = DemoTreeStateMachine()
    assert isinstance(tree_sm, DemoStateMachine)

    # current_screen_info() output check
    info = tree_sm.current_screen_info()
    assert info["title"] == "ROOT"
    assert info["left_label"] == "TALK & NEEDS"
    assert info["right_label"] == "MEDICAL & ACTIONS"
    assert info["kind"] == "node"

    # properties check
    assert tree_sm.hold_progress == 0.0
    assert tree_sm.armed is False
    assert tree_sm.phrase_list == []
    assert isinstance(tree_sm.events, list)

    # Arm and test events dict access
    feed_stream(tree_sm, 0.0, tree_sm.center_arm_s + 0.05, "CENTER")
    assert tree_sm.armed is True
    assert len(tree_sm.events) > 0
    # Test dictionary-style access on event
    first_event = tree_sm.events[0]
    assert "t" in first_event.to_dict()
    assert first_event["type"] == first_event.type
