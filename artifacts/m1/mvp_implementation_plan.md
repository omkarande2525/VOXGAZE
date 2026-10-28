# VoxGaze MVP — Implementation Plan
**Milestone**: Working Gaze-to-Word AAC Demonstration  
**Target Directory**: `backend/app/aac/` & `scripts/run_aac_mvp.py`  
**Branch**: `feature/mobileone-s0-integration`

---

## 1. Objective & Scope Boundaries
- **Product Goal**: Accessible AAC word board demonstrating:
  `Look at a word → see it highlighted → select it → build a phrase → explicitly confirm → hear offline speech.`
- **Honest Prototype Boundary**: Acknowledge that physical calibration directional gates (H and V separation) are currently unsupported. Live gaze-to-word activation remains safely disabled until empirical calibration passes; Demo / Simulated Gaze Mode provides a reliable, accessible interactive demonstration.
- **Non-Goals**: No cloud dependencies, no LLMs, no raw video persistence, no medical/emergency claims, no synthetic gate-lowering.

---

## 2. Architecture & Module Design

```text
[ Browser Web Interface (index.html / CSS / JS) ]
      │                               ▲
      │ REST / WebSocket              │ State / Telemetry
      ▼                               │
[ FastAPI AAC Server (server.py) ]────┘
      ├── [ AAC Board & Phrase Builder (board.py) ]
      ├── [ Dwell & Confirmation State Machine (selection.py) ]
      ├── [ Offline TTS Engine (speech.py) ] (Piper / Windows SAPI / Browser Signal)
      └── [ Gaze Input Providers (gaze_input.py) ]
            ├── SimulatedGazeProvider (Demo Mode: Arrow keys, WASD, Space)
            └── LiveGazeProvider (Live Research: MediaPipe, MobileOne, GazeEngine, Calibration Gates)
```

### Module Responsibilities:
1. **`backend/app/aac/board.py`**:
   - Manages standard vocabulary tiles: `I`, `want`, `water`, `food`, `help`, `yes`, `no`, `pain`, `rest`, `toilet`, `thank you`.
   - Phrase buffer management: append tile, remove last word (backspace), clear phrase, format for speech.
   - Tile coordinate layout and hit-testing.
2. **`backend/app/aac/selection.py`**:
   - Dwell detection: configurable dwell duration (e.g. 1.2s), tracking state (`IDLE`, `DWELLING`, `CANDIDATE_HIGHLIGHTED`, `CONFIRMED`).
   - Safety pauses/aborts on tracking loss, low confidence, or sudden pointer jumps.
   - Explicit confirmation requirement before adding word to phrase.
3. **`backend/app/aac/speech.py`**:
   - Modular offline speech synthesizer.
   - Order of preference: Local Piper TTS -> Windows SAPI (`System.Speech` / `SAPI.SpVoice`) -> Browser Web Speech signal.
   - Strictly offline, no external network calls.
   - Explicit error handling for empty phrase, missing voices, and process failures.
4. **`backend/app/aac/gaze_input.py`**:
   - Abstract `GazeInputProvider` interface.
   - `SimulatedGazeProvider`: Keyboard navigation, smooth pointer interpolation, dwell simulation.
   - `LiveGazeProvider`: Consumes webcam stream via existing MediaPipe FaceLandmarker, MobileOne, and GazeEngine; enforces `is_enabled` and directional separation gates; marks selection `DISABLED` when gates fail.
5. **`backend/app/aac/server.py`**:
   - FastAPI server with endpoints:
     - `GET /`: Serves accessible AAC single-page app.
     - `GET /api/board`: Returns word tiles and layout.
     - `POST /api/phrase/append`, `POST /api/phrase/backspace`, `POST /api/phrase/clear`.
     - `POST /api/speak`: Explicit phrase speech trigger.
     - `GET /api/status`: Camera, tracking, calibration gates, active mode.
     - `GET /api/calibration/summary`: Returns latest calibration gate findings.
     - `WS /ws/telemetry`: Real-time gaze pointer, dwell progress, and tracking flags.
6. **`backend/app/aac/static/`**:
   - `index.html`: High-contrast, WCAG-accessible UI, large word tiles, phrase bar, preview container, status panel, camera preview canvas, demo walkthrough overlay.
   - `style.css`: Clean, high-contrast, scalable CSS, accessible focus rings, dwell progress animation, colorblind-safe icons.
   - `app.js`: Client-side coordinator, keyboard handling (WASD/Arrows/Space), dwell rendering, sound cues.
7. **`scripts/run_aac_mvp.py`**:
   - CLI launcher for the VoxGaze AAC MVP.

---

## 3. Implementation Steps

1. Create `backend/app/aac/board.py` (AAC vocabulary, phrase composition, unit-tested).
2. Create `backend/app/aac/selection.py` (dwell state machine, candidate preview, pause/abort gates).
3. Create `backend/app/aac/speech.py` (offline TTS: Piper + Windows SAPI + speech status).
4. Create `backend/app/aac/gaze_input.py` (simulated & live webcam input providers, gate enforcement).
5. Create `backend/app/aac/server.py` (FastAPI app, WebSocket, REST routes).
6. Create `backend/app/aac/static/` (`index.html`, `style.css`, `app.js` with responsive accessible AAC board, demo tour, live diagnostic view).
7. Create `scripts/run_aac_mvp.py` (CLI runner).
8. Create `tests/unit/test_aac_mvp.py` (comprehensive automated test suite).
9. Run full test suite (`pytest -v`), build/lint checks, verify Demo Mode and Live Research Mode.
10. Final report and presentation script.
