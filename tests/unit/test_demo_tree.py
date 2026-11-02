"""
Unit tests for VoxGaze Demo State Machine (Task D1)

Test Requirements (from Section 18):
1. Tree loads from config and exposes all 8 leaves.
2. A hold shorter than HOLD_MS does nothing.
3. A hold of HOLD_MS commits exactly one level.
4. One long LEFT hold cannot commit two levels (needs re-arm at centre).
5. Without 500 ms at centre, a new hold is ignored.
6. A confirm YES adds the word; NO and timeout add nothing.
7. A word is never added if no confirm screen was shown (property test over random sequences).
8. UP/DOWN do not exist as inputs; NONE never triggers a decision.
9. NONE longer than TRACKING_LOSS_MS pauses and clears the hold.
10. A dropout shorter than GRACE_MS does not reset a hold.
11. SPEAK emits a speak_request only after confirm YES, with the current phrase.
12. UNDO removes the last word and is safe on an empty phrase.
13. Timestamps, not call counts, drive timing (feed irregular intervals).
14. Esc/reset returns to root from any state.
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

def test_tree_loads_from_config_and_exposes_all_8_leaves(fsm):
    """Verify that tree loads from demo_config.json and contains all 8 vocabulary leaves."""
    leaves = fsm.get_leaves()
    assert len(leaves) == 8

    leaf_texts = [leaf.get("word") or leaf.get("action") for leaf in leaves]
    expected_leaves = {"YES", "NO", "I NEED", "WATER", "HELP", "PAIN", "SPEAK", "UNDO"}
    assert set(leaf_texts) == expected_leaves

    # Check root screen metadata
    screen = fsm.current_screen
    assert screen.kind == "node"
    assert screen.title == "ROOT"
    assert screen.left_label == "TALK"
    assert screen.right_label == "ALERT & ACTIONS"


def test_hold_shorter_than_hold_ms_does_nothing(fsm):
    """Verify that holding a direction for less than HOLD_MS (e.g. 800 ms < 1000 ms) causes no transition."""
    t = 0.0
    # 1. Arm at CENTER for 500 ms
    t = feed_stream(fsm, t, 0.50, "CENTER")
    assert fsm.armed is True
    assert fsm.current_screen.title == "ROOT"

    # 2. Hold LEFT for 800 ms (< 1000 ms)
    t = feed_stream(fsm, t, 0.80, "LEFT")
    assert fsm.current_screen.title == "ROOT"  # Stays at ROOT
    assert fsm.hold_progress > 0.7

    # 3. Return to CENTER
    t = feed_stream(fsm, t, 0.20, "CENTER")
    assert fsm.current_screen.title == "ROOT"
    assert fsm.hold_progress == 0.0  # Hold progress cleared


def test_hold_of_hold_ms_commits_exactly_one_level(fsm):
    """Verify that a full 1000 ms hold commits exactly one hierarchical level."""
    t = 0.0
    # Arm at CENTER
    t = feed_stream(fsm, t, 0.50, "CENTER")
    assert fsm.armed is True

    # Hold LEFT for 1000 ms
    t = feed_stream(fsm, t, 1.05, "LEFT")
    # Transitions to TALK
    assert fsm.current_screen.title == "TALK"
    assert fsm.current_screen.left_label == "REPLY"
    assert fsm.current_screen.right_label == "NEEDS"
    # Disarms until center rest
    assert fsm.armed is False


def test_long_left_hold_cannot_commit_two_levels(fsm):
    """CRITICAL SAFETY INVARIANT: A continuous 3000 ms hold cannot jump two levels."""
    t = 0.0
    t = feed_stream(fsm, t, 0.50, "CENTER")
    assert fsm.armed is True

    # Hold LEFT continuously for 3.0 seconds
    t = feed_stream(fsm, t, 3.00, "LEFT")
    # First level committed to TALK, but second level (REPLY) must NOT commit without center re-arming
    assert fsm.current_screen.title == "TALK"
    assert fsm.armed is False


def test_without_500ms_at_centre_new_hold_is_ignored(fsm):
    """Verify that returning to CENTER for only 200 ms (< 500 ms) does NOT arm the next choice."""
    t = 0.0
    # Arm and choose TALK
    t = feed_stream(fsm, t, 0.50, "CENTER")
    t = feed_stream(fsm, t, 1.05, "LEFT")
    assert fsm.current_screen.title == "TALK"
    assert fsm.armed is False

    # Return to CENTER for only 200 ms (< 500 ms)
    t = feed_stream(fsm, t, 0.20, "CENTER")
    assert fsm.armed is False

    # Try to hold LEFT again
    t = feed_stream(fsm, t, 1.20, "LEFT")
    # Must still be at TALK (not REPLY) because state machine was never re-armed
    assert fsm.current_screen.title == "TALK"


def test_confirm_yes_adds_word_no_and_timeout_add_nothing(fsm):
    """
    Verify confirmation loop:
    1. Reaching leaf displays confirm screen without adding word.
    2. Confirm YES adds word.
    3. Confirm NO cancels back to ROOT without adding word.
    4. Confirm Timeout cancels back to ROOT without adding word.
    """
    t = 0.0

    def navigate_to_water(start_t: float) -> float:
        cur = start_t
        # ROOT -> TALK (LEFT)
        cur = feed_stream(fsm, cur, 0.50, "CENTER")
        cur = feed_stream(fsm, cur, 1.05, "LEFT")
        # TALK -> NEEDS (RIGHT)
        cur = feed_stream(fsm, cur, 0.50, "CENTER")
        cur = feed_stream(fsm, cur, 1.05, "RIGHT")
        # NEEDS -> WATER (RIGHT)
        cur = feed_stream(fsm, cur, 0.50, "CENTER")
        cur = feed_stream(fsm, cur, 1.05, "RIGHT")
        return cur

    # 1. Navigate to WATER
    t = navigate_to_water(t)
    assert fsm.current_screen.kind == "confirm"
    assert fsm.current_screen.candidate == "WATER"
    assert fsm.phrase == []  # Word is NOT added yet

    # Confirm YES (re-arm at center, then hold LEFT)
    t = feed_stream(fsm, t, 0.50, "CENTER")
    t = feed_stream(fsm, t, 1.05, "LEFT")
    assert fsm.phrase == ["WATER"]
    assert fsm.current_screen.title == "ROOT"

    # 2. Navigate to WATER again and test Confirm NO
    t = navigate_to_water(t)
    assert fsm.current_screen.kind == "confirm"
    t = feed_stream(fsm, t, 0.50, "CENTER")
    t = feed_stream(fsm, t, 1.05, "RIGHT")  # NO cancels
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
    # Arm and start holding LEFT for 400 ms
    t = feed_stream(fsm, t, 0.50, "CENTER")
    t = feed_stream(fsm, t, 0.40, "LEFT")
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
    Hold 500 ms LEFT + 100 ms CENTER (grace) + 500 ms LEFT -> total 1000 ms -> commits!
    """
    t = 0.0
    t = feed_stream(fsm, t, 0.50, "CENTER")

    # 1. Hold LEFT for 500 ms
    t = feed_stream(fsm, t, 0.50, "LEFT")
    assert fsm.hold_progress >= 0.45

    # 2. Brief 100 ms dropout to CENTER (<= 150 ms GRACE_MS)
    t = feed_stream(fsm, t, 0.10, "CENTER")
    assert fsm.current_screen.title == "ROOT"  # Stays at ROOT

    # 3. Resume LEFT for 500 ms
    t = feed_stream(fsm, t, 0.55, "LEFT")
    # Total hold elapsed >= 1.0s -> commits to TALK!
    assert fsm.current_screen.title == "TALK"


def test_undo_and_speak_behave_properly(fsm):
    """
    Verify ACTIONS branch:
    1. SPEAK emits a speak_request event with assembled phrase.
    2. UNDO removes the last word.
    3. UNDO on empty phrase is completely safe.
    """
    t = 0.0

    # Manually populate phrase buffer
    fsm.phrase = ["I NEED", "WATER"]

    def navigate_to_actions(start_t: float) -> float:
        cur = start_t
        # ROOT -> ALERT & ACTIONS (RIGHT)
        cur = feed_stream(fsm, cur, 0.50, "CENTER")
        cur = feed_stream(fsm, cur, 1.05, "RIGHT")
        # ALERT & ACTIONS -> ACTIONS (RIGHT)
        cur = feed_stream(fsm, cur, 0.50, "CENTER")
        cur = feed_stream(fsm, cur, 1.05, "RIGHT")
        return cur

    # 1. Trigger UNDO (RIGHT in ACTIONS)
    t = navigate_to_actions(t)
    assert fsm.current_screen.title == "ACTIONS"
    assert fsm.current_screen.right_label == "UNDO"

    t = feed_stream(fsm, t, 0.50, "CENTER")
    t = feed_stream(fsm, t, 1.05, "RIGHT")  # Select UNDO
    assert fsm.current_screen.kind == "confirm"
    assert fsm.current_screen.candidate == "UNDO"

    # Confirm UNDO
    t = feed_stream(fsm, t, 0.50, "CENTER")
    t = feed_stream(fsm, t, 1.05, "LEFT")  # YES
    assert fsm.phrase == ["I NEED"]  # "WATER" removed

    # 2. Trigger SPEAK (LEFT in ACTIONS)
    t = navigate_to_actions(t)
    assert fsm.current_screen.title == "ACTIONS"
    assert fsm.current_screen.left_label == "SPEAK"

    t = feed_stream(fsm, t, 0.50, "CENTER")
    t = feed_stream(fsm, t, 1.05, "LEFT")  # Select SPEAK
    assert fsm.current_screen.kind == "confirm"
    assert fsm.current_screen.candidate == "SPEAK"

    # Confirm SPEAK
    t = feed_stream(fsm, t, 0.50, "CENTER")
    t = feed_stream(fsm, t, 1.05, "LEFT")  # YES

    speak_events = [e for e in fsm.events if e.type == "speak_request"]
    assert len(speak_events) >= 1
    assert speak_events[-1].detail == "I NEED"
    assert fsm.phrase == ["I NEED"]  # Phrase preserved

    # 3. UNDO on empty phrase safety check
    fsm.phrase.clear()
    t = navigate_to_actions(t)
    t = feed_stream(fsm, t, 0.50, "CENTER")
    t = feed_stream(fsm, t, 1.05, "RIGHT")  # Select UNDO
    t = feed_stream(fsm, t, 0.50, "CENTER")
    t = feed_stream(fsm, t, 1.05, "LEFT")   # Confirm YES
    assert fsm.phrase == []  # Safe, no exception


def test_timestamps_not_call_counts_drive_timing(fsm):
    """Verify that irregular sample intervals evaluate strictly based on timestamps, not call counts."""
    # 100 samples in 100 ms -> should NOT arm (500 ms needed)
    for i in range(100):
        fsm.on_sample(i * 0.001, "CENTER")
    assert fsm.armed is False

    # Just 2 samples spanning 510 ms -> arms!
    fsm.on_sample(0.50, "CENTER")
    fsm.on_sample(1.02, "CENTER")
    assert fsm.armed is True


def test_reset_returns_to_root_from_any_state(fsm):
    """Verify that calling reset() immediately aborts back to ROOT and clears hold."""
    t = 0.0
    t = feed_stream(fsm, t, 0.50, "CENTER")
    t = feed_stream(fsm, t, 1.05, "LEFT")  # TALK
    assert fsm.current_screen.title == "TALK"

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
    assert info["left_label"] == "TALK"
    assert info["right_label"] == "ALERT & ACTIONS"
    assert info["kind"] == "node"

    # properties check
    assert tree_sm.hold_progress == 0.0
    assert tree_sm.armed is False
    assert tree_sm.phrase_list == []
    assert isinstance(tree_sm.events, list)

    # Arm and test events dict access
    feed_stream(tree_sm, 0.0, 0.50, "CENTER")
    assert tree_sm.armed is True
    assert len(tree_sm.events) > 0
    # Test dictionary-style access on event
    first_event = tree_sm.events[0]
    assert "t" in first_event.to_dict()
    assert first_event["type"] == first_event.type
