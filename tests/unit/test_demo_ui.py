"""
Unit tests for VoxGaze AAC Demo Web UI & Accessibility Frontend (Tasks D3 & D4)

Verifies:
1. Route registration: /demo is ONLY registered when experimental_demo=True; absent when False.
2. Endpoint execution: GET /demo returns FileResponse pointing to static/demo.html.
3. HTML content & accessibility invariants:
   - Top warning banner removed
   - Phrase container ('PHRASE:') and speaking indicator ('SPEAKING...')
   - Dual selection panels (LEFT and RIGHT)
   - Hold-progress indicator bar
   - Center rest/arming cue ('ARMED / LOOK TO CENTRE', 'RE-ARMING AT CENTRE...')
   - Bottom status bar with direction, tracking, FPS, and latency
4. Privacy & sensor invariants:
   - Zero getUserMedia calls or <video> elements in demo.html or demo.js
   - Web client is strictly a passive renderer; camera is held solely by backend worker
5. Frontend script invariants:
   - 10 Hz polling loop for /api/demo/state
   - Keyboard shortcuts for Esc (reset), P (pause), and Arrow keys (key nav)
"""

from pathlib import Path
import pytest
from starlette.responses import FileResponse

from backend.app.aac.server import create_app, DemoKeyRequest


STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "backend" / "app" / "aac" / "static"
DEMO_HTML = STATIC_DIR / "demo.html"
DEMO_JS = STATIC_DIR / "demo.js"
DEMO_CSS = STATIC_DIR / "demo.css"


def test_demo_route_flag_off_omits_demo_endpoint():
    """Verify that when experimental_demo=False, /demo is NOT registered."""
    app = create_app(experimental_demo=False)
    paths = [r.path for r in app.routes]
    assert "/demo" not in paths
    assert "/api/demo/state" not in paths


def test_demo_route_flag_on_registers_and_serves_demo_html():
    """Verify that when experimental_demo=True, /demo serves static/demo.html."""
    app = create_app(experimental_demo=True)
    paths = [r.path for r in app.routes]
    assert "/demo" in paths
    assert "/api/demo/state" in paths

    endpoints = {r.path: r.endpoint for r in app.routes if hasattr(r, "endpoint")}
    demo_endpoint = endpoints["/demo"]
    response = demo_endpoint()

    assert isinstance(response, FileResponse)
    assert Path(response.path).resolve() == DEMO_HTML.resolve()
    assert response.status_code == 200


def test_demo_html_file_exists():
    """Verify static demo assets exist on disk."""
    assert DEMO_HTML.is_file(), f"Missing {DEMO_HTML}"
    assert DEMO_JS.is_file(), f"Missing {DEMO_JS}"
    assert DEMO_CSS.is_file(), f"Missing {DEMO_CSS}"


def test_demo_html_warning_banner_removed():
    """Verify that the permanent unvalidated warning header is removed."""
    content = DEMO_HTML.read_text(encoding="utf-8")
    assert "EXPERIMENTAL - NOT VALIDATED - GATE NOT PASSED" not in content


def test_demo_html_contains_phrase_and_speech_indicator():
    """Verify phrase container and active speech indicator elements are present."""
    content = DEMO_HTML.read_text(encoding="utf-8")
    assert "PHRASE:" in content
    assert "phrase-display" in content
    assert "speaking-indicator" in content
    assert "SPEAKING..." in content


def test_demo_html_contains_dual_choice_panels():
    """Verify large dual panels for LEFT and RIGHT selections exist."""
    content = DEMO_HTML.read_text(encoding="utf-8")
    assert "panel-left" in content
    assert "panel-right" in content
    assert "LEFT" in content
    assert "RIGHT" in content


def test_demo_html_contains_progress_and_arm_cues():
    """Verify hold-progress indicator and center arming status elements exist."""
    content = DEMO_HTML.read_text(encoding="utf-8")
    assert "progress-bar-fill" in content
    assert "center-rest-zone" in content
    assert "arm-cue-badge" in content
    assert "center-pill" in content


def test_demo_html_contains_bottom_status_bar():
    """Verify status bar displaying direction, tracking, speed, and latency."""
    content = DEMO_HTML.read_text(encoding="utf-8")
    assert "bottom-status-bar" in content
    assert "val-direction" in content
    assert "val-tracking" in content
    assert "val-fps" in content
    assert "val-latency" in content


def test_demo_privacy_invariants_no_camera_or_video_in_frontend():
    """
    CRITICAL PRIVACY INVARIANT:
    The browser UI must NEVER attempt to access the webcam via getUserMedia
    or render a <video> element. The backend perception worker holds exclusive
    hardware access.
    """
    html_content = DEMO_HTML.read_text(encoding="utf-8").lower()
    js_content = DEMO_JS.read_text(encoding="utf-8").lower()

    # Neither HTML nor JS may contain getUserMedia
    assert "getusermedia" not in html_content
    assert "getusermedia" not in js_content

    # No video preview tag
    assert "<video" not in html_content


def test_demo_js_polling_and_key_bindings():
    """Verify frontend polling rate and keyboard event listeners."""
    js_content = DEMO_JS.read_text(encoding="utf-8")

    # 10 Hz polling verification (/api/demo/state every 100 ms)
    assert "/api/demo/state" in js_content
    assert "100" in js_content

    # Escape key resets state
    assert "Escape" in js_content
    assert "/api/demo/reset" in js_content

    # 'P' key pauses input
    assert "/api/demo/pause" in js_content

    # Arrow keys inject simulated navigation
    assert "/api/demo/key" in js_content
    assert "ArrowLeft" in js_content
    assert "ArrowRight" in js_content

    # Arming cues
    assert "ARMED / LOOK TO CENTRE" in js_content
    assert "RE-ARMING AT CENTRE..." in js_content


def test_demo_api_reset_and_pause_via_app():
    """Verify /api/demo/reset and /api/demo/pause endpoints operate correctly."""
    app = create_app(experimental_demo=True, allow_keys=True)
    endpoints = {r.path: r.endpoint for r in app.routes if hasattr(r, "endpoint")}

    state_endpoint = endpoints["/api/demo/state"]
    reset_endpoint = endpoints["/api/demo/reset"]
    pause_endpoint = endpoints["/api/demo/pause"]
    key_endpoint = endpoints["/api/demo/key"]

    initial_state = state_endpoint()
    assert initial_state["banner_text"] == ""
    assert "screen" in initial_state

    # Pause toggles tracking status
    pause_res = pause_endpoint()
    assert pause_res["status"] == "SUCCESS"
    assert pause_res["paused"] is True
    paused_state = state_endpoint()
    assert paused_state["tracking"] == "paused"

    # Unpause
    unpause_res = pause_endpoint()
    assert unpause_res["paused"] is False

    # Reset
    reset_res = reset_endpoint()
    assert reset_res["status"] == "SUCCESS"

    # Key navigation with allow_keys=True
    key_res = key_endpoint(DemoKeyRequest(key="left"))
    assert key_res["status"] == "SUCCESS"

