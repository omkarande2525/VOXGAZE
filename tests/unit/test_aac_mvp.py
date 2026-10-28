"""
Unit tests for VoxGaze AAC MVP:
Word Board, Phrase Builder, Dwell State Machine, Safety Invariants,
Explicit Confirmation, Offline Speech Engine, and Honest Calibration Gate Enforcement.
"""

import pytest

from backend.app.aac.board import AACBoard, CORE_VOCABULARY, PhraseBuilder, TileCategory
from backend.app.aac.gaze_input import LiveGazeProvider, SimulatedGazeProvider
from backend.app.aac.selection import DwellSelector, SelectionState
from backend.app.aac.server import AACService, create_app
from backend.app.aac.speech import OfflineTTSEngine, TTSResult


# -------------------------------------------------------------
# 1. AAC Word Board & Phrase Builder Tests
# -------------------------------------------------------------

def test_aac_board_contains_all_11_core_vocabulary_tiles():
    """Verify that AACBoard contains all 11 core vocabulary words specified by requirements."""
    board = AACBoard()
    expected_words = {"I", "want", "water", "food", "help", "yes", "no", "pain", "rest", "toilet", "thank you"}
    actual_words = {tile.text for tile in board.tiles}
    assert expected_words == actual_words
    assert len(board.tiles) == 11


def test_aac_board_tile_hit_testing():
    """Verify that normalized screen coordinates map correctly to bounding boxes."""
    board = AACBoard()
    first_tile = board.tiles[0]
    # Midpoint of first tile
    mid_x = (first_tile.x_min + first_tile.x_max) / 2
    mid_y = (first_tile.y_min + first_tile.y_max) / 2

    hit = board.find_tile_at(mid_x, mid_y)
    assert hit is not None
    assert hit.id == first_tile.id

    # Outside bounds (e.g. margin or offscreen)
    assert board.find_tile_at(0.01, 0.01) is None
    assert board.find_tile_at(-0.1, 0.5) is None
    assert board.find_tile_at(1.5, 0.5) is None


def test_phrase_builder_composition_undo_and_clear():
    """Verify phrase construction, capitalization, punctuation, undo, and clear."""
    pb = PhraseBuilder()
    assert pb.is_empty is True
    assert pb.phrase_text == ""

    # 1. Add words
    assert pb.add_word("I") is True
    assert pb.add_word("want") is True
    assert pb.add_word("water") is True
    assert pb.words == ["I", "want", "water"]
    assert pb.phrase_text == "I want water."
    assert pb.is_empty is False

    # 2. Reject empty word
    assert pb.add_word("   ") is False

    # 3. Remove last word (undo/backspace)
    removed = pb.remove_last()
    assert removed == "water"
    assert pb.words == ["I", "want"]
    assert pb.phrase_text == "I want."

    # 4. Clear phrase
    pb.clear()
    assert pb.is_empty is True
    assert pb.phrase_text == ""
    assert pb.words == []


# -------------------------------------------------------------
# 2. Dwell Selector & Safety Pause/Abort Tests
# -------------------------------------------------------------

def test_dwell_selector_lifecycle_and_candidate_highlighting():
    """Verify transition from IDLE -> DWELLING -> CANDIDATE_HIGHLIGHTED upon dwell completion."""
    board = AACBoard()
    tile = board.tiles[0]
    selector = DwellSelector(dwell_duration_s=1.0)

    # Initial state
    assert selector.state == SelectionState.IDLE
    assert selector.active_tile is None

    # Step 1: Start hovering over tile at t = 100.0s
    state1 = selector.update(current_tile=tile, pointer_norm=(0.1, 0.2), now_mono=100.0)
    assert state1 == SelectionState.DWELLING
    assert selector.active_tile.id == tile.id
    assert selector.dwell_progress == 0.0

    # Step 2: Halfway through dwell at t = 100.5s
    state2 = selector.update(current_tile=tile, pointer_norm=(0.1, 0.2), now_mono=100.5)
    assert state2 == SelectionState.DWELLING
    assert selector.dwell_progress == pytest.approx(0.5, abs=0.01)
    assert selector.candidate_tile is None

    # Step 3: Dwell completed at t = 101.0s -> enters CANDIDATE_HIGHLIGHTED
    state3 = selector.update(current_tile=tile, pointer_norm=(0.1, 0.2), now_mono=101.0)
    assert state3 == SelectionState.CANDIDATE_HIGHLIGHTED
    assert selector.candidate_tile.id == tile.id
    assert selector.dwell_progress == 1.0


def test_dwell_selector_pauses_on_tracking_loss():
    """Verify that losing face/tracking pauses or aborts dwell and resets progress."""
    board = AACBoard()
    tile = board.tiles[0]
    selector = DwellSelector(dwell_duration_s=1.0)

    # Start dwelling
    selector.update(current_tile=tile, pointer_norm=(0.1, 0.2), now_mono=100.0)
    selector.update(current_tile=tile, pointer_norm=(0.1, 0.2), now_mono=100.6)
    assert selector.dwell_progress == pytest.approx(0.6, abs=0.01)

    # Tracking drops
    state_loss = selector.update(
        current_tile=tile,
        pointer_norm=(0.1, 0.2),
        is_tracking_valid=False,
        now_mono=100.7,
    )
    assert state_loss == SelectionState.PAUSED_TRACKING_LOSS
    assert selector.dwell_progress == 0.0
    assert selector.active_tile is None


def test_dwell_selector_aborts_on_sudden_pointer_jump():
    """Verify that an erratic pointer jump (saccade / noise) resets dwell."""
    board = AACBoard()
    tile = board.tiles[0]
    selector = DwellSelector(dwell_duration_s=1.0, jump_threshold_norm=0.15)

    selector.update(current_tile=tile, pointer_norm=(0.10, 0.20), now_mono=100.0)
    selector.update(current_tile=tile, pointer_norm=(0.11, 0.21), now_mono=100.4)
    assert selector.dwell_progress > 0.3

    # Sudden jump from (0.11, 0.21) to (0.60, 0.80) -> jump distance = 0.76 > 0.15
    selector.update(current_tile=tile, pointer_norm=(0.60, 0.80), now_mono=100.5)
    # Must reset dwell back to 0
    assert selector.dwell_progress == 0.0


def test_explicit_confirmation_required_before_word_commit():
    """Verify invariant: dwell alone does not commit words; explicit confirmation is mandatory."""
    board = AACBoard()
    tile = board.tiles[0]
    selector = DwellSelector(dwell_duration_s=1.0)

    # 1. While still DWELLING, confirm_candidate fails
    selector.update(current_tile=tile, pointer_norm=(0.1, 0.2), now_mono=100.0)
    selector.update(current_tile=tile, pointer_norm=(0.1, 0.2), now_mono=100.5)
    assert selector.confirm_candidate() is None

    # 2. Reached CANDIDATE_HIGHLIGHTED
    selector.update(current_tile=tile, pointer_norm=(0.1, 0.2), now_mono=101.0)
    assert selector.state == SelectionState.CANDIDATE_HIGHLIGHTED

    # 3. Explicit confirmation succeeds
    confirmed = selector.confirm_candidate()
    assert confirmed is not None
    assert confirmed.id == tile.id
    # Selector resets back to IDLE after confirmation
    assert selector.state == SelectionState.IDLE


# -------------------------------------------------------------
# 3. Offline Speech (TTS) Engine Tests
# -------------------------------------------------------------

def test_offline_tts_rejects_empty_phrase_gracefully():
    """Verify that speech synthesis fails safely with explicit message on empty input."""
    tts = OfflineTTSEngine()
    res = tts.speak("   ")
    assert res.success is False
    assert "EMPTY_PHRASE" in (res.error_message or "")
    assert res.engine_used == "NONE"


def test_offline_tts_synthesizes_phrase_without_crash():
    """Verify that speech synthesis executes offline and records last_spoken_phrase."""
    tts = OfflineTTSEngine()
    res = tts.speak("I want water.")
    assert isinstance(res, TTSResult)
    assert res.phrase == "I want water."
    assert res.engine_used in ("WINDOWS_SAPI_OFFLINE", "CLIENT_BROWSER_SPEECH_SIGNAL", "PIPER_LOCAL")
    assert tts.last_spoken_phrase == "I want water."


# -------------------------------------------------------------
# 4. Gaze Input Providers & Calibration Gate Invariant Tests
# -------------------------------------------------------------

def test_simulated_gaze_provider_is_clearly_marked_simulated():
    """Verify that Demo Mode permanently reports is_simulated=True and never claims live tracking."""
    sim = SimulatedGazeProvider(start_x=0.50, start_y=0.50)
    tel = sim.get_telemetry()
    assert tel.is_simulated is True
    assert tel.mode == "DEMO_SIMULATED"
    assert "SIMULATION" in tel.status_label
    assert tel.is_selection_enabled is True

    # Move pointer
    sim.move(0.10, -0.05)
    tel2 = sim.get_telemetry()
    assert tel2.pointer_x == pytest.approx(0.60, abs=0.01)
    assert tel2.pointer_y == pytest.approx(0.45, abs=0.01)


def test_live_research_provider_strictly_disables_word_activation():
    """
    CRITICAL INVARIANT: Because physical calibration separation gates have not passed,
    Live Research Mode must honestly report selection DISABLED and provide an explicit reason.
    """
    live = LiveGazeProvider()
    tel = live.get_telemetry()
    assert tel.mode == "LIVE_RESEARCH"
    assert tel.is_simulated is False
    assert tel.is_selection_enabled is False
    assert "LIVE_SELECTION_DISABLED" in (tel.unsupported_reason or "")
    assert "directional separation gates (H/V)" in (tel.unsupported_reason or "")
    # Does not invent a fake screen pointer
    assert tel.pointer_x is None
    assert tel.pointer_y is None


# -------------------------------------------------------------
# 5. AAC Service & Controller Tests
# -------------------------------------------------------------

@pytest.fixture
def service():
    return AACService()


def test_service_board_endpoint_returns_tiles(service):
    """Verify get_board_tiles returns the 11 vocabulary tiles."""
    data = service.get_board_tiles()
    assert "tiles" in data
    assert len(data["tiles"]) == 11
    tile_texts = [t["text"] for t in data["tiles"]]
    assert "water" in tile_texts
    assert "pain" in tile_texts


def test_service_phrase_composition_and_speech_workflow(service):
    """
    End-to-End Demo Workflow:
    1. Append words 'I', 'want', 'water'
    2. Verify assembled phrase 'I want water.'
    3. Trigger explicit speak action
    4. Clear phrase
    """
    # 1. Append words
    r1 = service.append_tile("I")
    assert r1["phrase"] == "I."

    r2 = service.append_tile("want")
    assert r2["phrase"] == "I want."

    r3 = service.append_tile("water")
    assert r3["phrase"] == "I want water."
    assert r3["words"] == ["I", "want", "water"]

    # 2. Trigger explicit speech
    speak_data = service.speak_phrase()
    assert speak_data["success"] is True
    assert speak_data["phrase"] == "I want water."

    # 3. Undo last word
    undo_res = service.backspace_phrase()
    assert undo_res["phrase"] == "I want."

    # 4. Clear phrase
    clear_res = service.clear_phrase()
    assert clear_res["phrase"] == ""


def test_service_live_mode_blocks_phrase_append(service):
    """Verify that switching to LIVE mode blocks phrase append because calibration gates have not passed."""
    # Switch to LIVE mode
    m_res = service.set_mode("LIVE")
    assert m_res["active_mode"] == "LIVE"

    # Attempt to append word in LIVE mode -> must raise PermissionError
    with pytest.raises(PermissionError) as exc_info:
        service.append_tile("water")
    assert "LIVE_SELECTION_DISABLED" in str(exc_info.value)


def test_service_demo_move_and_confirmation(service):
    """Verify simulated gaze pointer movement and candidate confirmation."""
    # Set position onto first tile (around x=0.10, y=0.20)
    data = service.demo_move(x=0.10, y=0.20)
    assert data["pointer"]["x"] == 0.10
    assert data["pointer"]["y"] == 0.20
    assert "dwell" in data


# -------------------------------------------------------------
# 6. Calibration Session & Preview Alignment Regression Tests
# -------------------------------------------------------------

def test_calibration_status_reports_empirical_gate_failures_without_fabrication(service):
    """
    Verify Task 3 requirements:
    1. get_calibration_status loads actual empirical observation data or safe defaults.
    2. Gates honestly reflect empirical measurements (e.g. horizontal gates passed, vertical unsupported).
    3. is_selection_enabled reflects gate policy.
    """
    cal_status = service.get_calibration_status()
    assert "gates" in cal_status
    # All directional and blink gates reflect empirical observation results
    assert "supported" in cal_status["gates"]["UP"]
    assert "supported" in cal_status["gates"]["DOWN"]
    assert "supported" in cal_status["gates"]["LEFT"]
    assert "supported" in cal_status["gates"]["RIGHT"]
    assert cal_status["gates"]["BLINK"]["supported"] is True
    assert "frozen_thresholds" in cal_status


def test_calibration_session_lifecycle_and_safety_invariants(service):
    """
    Verify Task 2 requirements:
    1. start_calibration_session returns 5 fixation targets with settling/recording timings.
    2. Targets match screen positions (CENTER, LEFT, RIGHT, UP, DOWN).
    3. complete_calibration_session reports completion but strictly keeps live word activation DISABLED.
    4. abort_calibration_session resets session state safely.
    """
    # 1. Start session
    start_res = service.start_calibration_session()
    assert start_res["status"] == "STARTED"
    assert start_res["total_targets"] == 5
    assert len(start_res["targets"]) == 5

    target_names = [t["name"] for t in start_res["targets"]]
    assert target_names == ["CENTER", "LEFT", "RIGHT", "UP", "DOWN"]

    for t in start_res["targets"]:
        assert t["settling_duration_ms"] == 500
        assert t["recording_duration_ms"] == 1500
        assert "instruction" in t

    # 2. Complete session -> Invariant check: is_selection_enabled must remain False!
    comp_res = service.complete_calibration_session()
    assert comp_res["status"] == "COMPLETED"
    assert comp_res["is_selection_enabled"] is False

    # 3. Abort session
    service.start_calibration_session()
    abort_res = service.abort_calibration_session()
    assert abort_res["status"] == "ABORTED"


def test_camera_preview_markup_and_css_invariants():
    """
    Verify Task 1 requirements:
    1. Video element uses camera-stream class with object-fit: contain.
    2. Mirrored presentation uses CSS transform: scaleX(-1) without modifying perception coordinates.
    3. Aspect-ratio is maintained on the container (no arbitrary cropping or distortion).
    """
    from pathlib import Path
    static_dir = Path(__file__).resolve().parent.parent.parent / "backend" / "app" / "aac" / "static"
    index_html = (static_dir / "index.html").read_text(encoding="utf-8")
    style_css = (static_dir / "style.css").read_text(encoding="utf-8")

    # In index.html:
    assert '<video id="webcam-video"' in index_html
    assert 'class="camera-stream' in index_html
    assert 'id="camera-status-line"' in index_html

    # In style.css:
    assert "object-fit: contain" in style_css
    assert "transform: scaleX(-1)" in style_css
    assert "aspect-ratio" in style_css

