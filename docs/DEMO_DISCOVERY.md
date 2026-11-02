# VoxGaze AAC MVP & Gaze-Tree Discovery (Task D0)

## Overview
This discovery document investigates the current state of live gaze acquisition, data contracts, calibration workflows, state machines, offline speech synthesis, and safety guardrails across the VoxGaze codebase. All findings cite precise file and line references based on the repository state as of branch `demo/gaze-tree`.

---

## 1. How `run_aac_mvp.py` Gets Live Gaze Today

- **Entry Point**: [`scripts/run_aac_mvp.py`](file:///d:/VOXGAZE/scripts/run_aac_mvp.py#L24-L60)
  - `main()` parses CLI options `--host`, `--port`, `--no-browser`, and `--camera` (lines 26–30).
  - Line 32 invokes `app = create_app()`, which initializes an `AACService` instance ([`backend/app/aac/server.py#L454-L471`](file:///d:/VOXGAZE/backend/app/aac/server.py#L454-L471)).
- **Provider & Model Wiring**:
  - In [`backend/app/aac/server.py#L58-L65`](file:///d:/VOXGAZE/backend/app/aac/server.py#L58-L65), `AACService.__init__` defaults to `self.live_provider = live_provider or LiveGazeProvider()`.
  - In [`backend/app/aac/gaze_input.py#L123-L149`](file:///d:/VOXGAZE/backend/app/aac/gaze_input.py#L123-L149), `LiveGazeProvider.__init__` takes `camera_idx: int = 0`, `mapper: Optional[PersonalizedGazeMapper] = None`, and `estimator: Optional[BaseGazeEstimator] = None`.
  - Neither `run_aac_mvp.py` nor `create_app()` passes a model estimator (`MobileOne` or `ResNet-34`) or a fitted mapper to `LiveGazeProvider`.
- **Backend Perception Loop Status**:
  - `run_aac_mvp.py` launches Uvicorn (`uvicorn.run(app, ...)` at line 59) but does **not** spawn a background capture thread or call OpenCV `VideoCapture`.
  - The method `LiveGazeProvider.update_frame_measurements(...)` ([`backend/app/aac/gaze_input.py#L150-L179`](file:///d:/VOXGAZE/backend/app/aac/gaze_input.py#L150-L179)) is never called at runtime.
  - In the browser frontend ([`backend/app/aac/static/app.js#L611-L635`](file:///d:/VOXGAZE/backend/app/aac/static/app.js#L611-L635)), `navigator.mediaDevices.getUserMedia(...)` requests webcam access locally to populate a client-side `<video>` element for presenter framing, but video frames are never streamed to the server.
- **Summary**: Today, `run_aac_mvp.py` gets **no** live gaze frames on the backend. `LiveGazeProvider` remains an unpolled placeholder with `is_tracking_valid = False`, `status_label = "INITIALIZING_CAMERA"`, and `is_selection_enabled = False`.

---

## 2. Gaze Sample Contents and Mapped Screen Coordinates

### Gaze Data Structures
1. **Telemetry Stream Packet (`GazeTelemetry`)**:
   - Defined in [`backend/app/aac/gaze_input.py#L28-L49`](file:///d:/VOXGAZE/backend/app/aac/gaze_input.py#L28-L49).
   - Contains:
     - `mode`: `"DEMO_SIMULATED"` or `"LIVE_RESEARCH"`
     - `timestamp_ms`: Current Unix epoch millisecond timestamp (`int(time.time() * 1000)`)
     - `pointer_x`: Normalized $[0.0, 1.0]$ horizontal screen coordinate (`Optional[float]`)
     - `pointer_y`: Normalized $[0.0, 1.0]$ vertical screen coordinate (`Optional[float]`)
     - `is_tracking_valid`: Boolean tracking health flag
     - `face_detected`: Boolean face detection flag
     - `confidence`: Confidence score in $[0.0, 1.0]$
     - `status_label`: Human-readable status (e.g., `"● TRACKING ACTIVE"`, `"! ABSTAIN"`, `"✕ NO_FACE"`)
     - `is_selection_enabled`: Strict gating flag (`bool`)
     - `unsupported_reason`: Explanatory message when selection is disabled (`Optional[str]`)
     - Raw/intermediate metrics: `h_ratio`, `v_ratio`, `ear`, `canthal_dist_px`, `head_yaw_deg`, `head_pitch_deg`, `is_simulated`
2. **Mapper Prediction (`MappingPrediction`)**:
   - Defined in [`backend/app/calibration/personalized_mapper.py#L53-L63`](file:///d:/VOXGAZE/backend/app/calibration/personalized_mapper.py#L53-L63).
   - Contains: `screen_x` (float in $[0, 1]$), `screen_y` (float in $[0, 1]$), `is_valid` (`bool`), `mapping_latency_us` (`float`), `predicted_direction` (`str`, closest canonical target: `"CENTER"`, `"LEFT"`, `"RIGHT"`, `"UP"`, `"DOWN"`), and `error_message` (`Optional[str]`).
3. **Discrete Gaze Engine State (`GazeState`)**:
   - Defined in [`backend/app/core/contracts.py`](file:///d:/VOXGAZE/backend/app/core/contracts.py) and output by `GazeEngine.process_features()` ([`backend/app/perception/gaze_engine.py#L51-L171`](file:///d:/VOXGAZE/backend/app/perception/gaze_engine.py#L51-L171)).
   - Contains: `timestamp_ms`, `direction` (`Direction` enum: `CENTER`, `LEFT`, `RIGHT`, `UP`, `DOWN`, `BLINK`, `ABSTAIN`, `NO_FACE`, `UNCALIBRATED`), `raw_h`, `raw_v`, `ear`, `confidence`, and `status_flag`. Note that `GazeEngine` outputs discrete direction states, not screen coordinates.

### Can We Get Mapped $(x, y)$ Per Frame?
- **Yes**: In `LiveGazeProvider.get_telemetry()` ([`backend/app/aac/gaze_input.py#L180-L202`](file:///d:/VOXGAZE/backend/app/aac/gaze_input.py#L180-L202)), if `self.mapper` is fitted (`self.mapper.is_fitted == True`) and tracking is valid, `self.mapper.predict(sample)` generates continuous normalized screen coordinates `(pointer_x, pointer_y)`.
- If an active perception loop feeds `update_frame_measurements(...)` on every camera frame, `self.mapper.predict()` evaluates in under $10\ \mu\text{s}$ per frame ([`backend/app/calibration/personalized_mapper.py#L203-L246`](file:///d:/VOXGAZE/backend/app/calibration/personalized_mapper.py#L203-L246)), yielding instantaneous per-frame mapped $(x, y)$.

---

## 3. Calibration in AAC MVP vs Controlled Model Comparison

### How Calibration is Done for AAC Demo Today
- **UI Sequence**:
  - `AACService.start_calibration_session()` ([`backend/app/aac/server.py#L297-L364`](file:///d:/VOXGAZE/backend/app/aac/server.py#L297-L364)) defines a 5-point sequence (`CENTER`, `LEFT`, `RIGHT`, `UP`, `DOWN`) with $500\text{ ms}$ settling and $1500\text{ ms}$ recording per target.
  - The client frontend ([`backend/app/aac/static/app.js#L814-L936`](file:///d:/VOXGAZE/backend/app/aac/static/app.js#L814-L936)) renders an animated fixation circle moving across the 5 positions.
- **Completion & Honest Reporting**:
  - `AACService.complete_calibration_session()` ([`backend/app/aac/server.py#L366-L396`](file:///d:/VOXGAZE/backend/app/aac/server.py#L366-L396)) calls `load_latest_calibration_gate_results()` (lines 404–450).
  - This helper loads existing JSON reports (`calib_v15_*_summary.json`) from `artifacts/observations/`.
  - Completing the UI sequence does **not** compute weights, fit a mapper, or enable live selection (lines 369–371).

### Fitting and Injecting `PersonalizedGazeMapper` into `LiveGazeProvider`
1. Instantiate: `mapper = PersonalizedGazeMapper(feature_set=FeatureSetType.HYBRID, alpha=1.0)` ([`backend/app/calibration/personalized_mapper.py#L87-L99`](file:///d:/VOXGAZE/backend/app/calibration/personalized_mapper.py#L87-L99)).
2. Assemble observations into `List[CalibrationSample]` with target coordinates `(target_x, target_y)` and features (`yaw_deg`, `pitch_deg`, `right_eye_h`, `right_eye_v`, `left_eye_h`, `left_eye_v`, `canthal_normal`).
3. Fit: `success = mapper.fit(training_samples)` ([`backend/app/calibration/personalized_mapper.py#L144-L197`](file:///d:/VOXGAZE/backend/app/calibration/personalized_mapper.py#L144-L197)). Requires $\ge 5$ valid samples; computes Ridge normal equations $(X^T X + \alpha I) W = X^T Y$.
4. Inject: Assign to provider via `live_provider.mapper = mapper` or pass at instantiation `LiveGazeProvider(mapper=mapper)` ([`backend/app/aac/gaze_input.py#L126-L130`](file:///d:/VOXGAZE/backend/app/aac/gaze_input.py#L126-L130)).

### Reusing Calibration Blocks from the Comparison Script
The comparison script ([`scripts/run_controlled_model_comparison.py`](file:///d:/VOXGAZE/scripts/run_controlled_model_comparison.py)) contains the battle-tested physical calibration capture and fitting pipeline:
- **Constants**:
  - `TARGET_COORDS` ([`line 68-74`](file:///d:/VOXGAZE/scripts/run_controlled_model_comparison.py#L68-L74)): Standard normalized coordinates for 5 targets.
  - `BLOCK_1_TARGETS`, `BLOCK_2_TARGETS` ([`lines 86-87`](file:///d:/VOXGAZE/scripts/run_controlled_model_comparison.py#L86-L87)): Balanced fixation sequences with interleaved center returns.
- **Functions to Reuse**:
  - `draw_target_canvas(...)` ([`lines 904-995`](file:///d:/VOXGAZE/scripts/run_controlled_model_comparison.py#L904-L995)): Renders fixation targets, countdowns, settling state, and camera preview inset.
  - `extract_eye_features(...)` & `compute_canthal_normal_distance_px(...)` ([`lines 1241-1245`](file:///d:/VOXGAZE/scripts/run_controlled_model_comparison.py#L1241-L1245)): Canonical facial landmark geometry extraction.
  - `extract_face_crop(...)` ([`lines 1248-1250`](file:///d:/VOXGAZE/scripts/run_controlled_model_comparison.py#L1248-L1250)): Bounding box extraction with 1.20 padding.
  - `m1_estimator.predict(...)` / `r34_estimator.predict(...)` ([`lines 1257-1272`](file:///d:/VOXGAZE/scripts/run_controlled_model_comparison.py#L1257-L1272)): Gaze model inference.
  - `_to_samples(...)` ([`lines 522-550`](file:///d:/VOXGAZE/scripts/run_controlled_model_comparison.py#L522-L550)): Converts raw frame comparison records into `CalibrationSample` objects.
  - `mapper.fit(...)` ([`lines 587-589`](file:///d:/VOXGAZE/scripts/run_controlled_model_comparison.py#L587-L589)): Fits the regression weights.

---

## 4. `SimulatedGazeProvider`, `DwellSelector`, and State Machine Conflicts

### How They Are Used
- **`SimulatedGazeProvider`** ([`backend/app/aac/gaze_input.py#L65-L112`](file:///d:/VOXGAZE/backend/app/aac/gaze_input.py#L65-L112)):
  - Manages coordinates $(x, y) \in [0.05, 0.95]$ manipulated via `move(dx, dy)` or `jump_to(x, y)`.
  - Streamed to UI via `get_telemetry()`, labeled as simulated (`is_simulated = True`).
- **`DwellSelector`** ([`backend/app/aac/selection.py#L23-L138`](file:///d:/VOXGAZE/backend/app/aac/selection.py#L23-L138)):
  - Stateful dwell controller for a 2D planar word grid.
  - Takes `current_tile: Optional[WordTile]` from `AACBoard.find_tile_at(x, y)`.
  - Advances state: `IDLE` $\rightarrow$ `DWELLING` (accumulates progress up to $1.2\text{ s}$) $\rightarrow$ `CANDIDATE_HIGHLIGHTED`.
  - Requires explicit confirmation via `confirm_candidate()` to transition to `CONFIRMED` and return the committed tile.
  - Enforces safety: resets on tracking loss (`PAUSED_TRACKING_LOSS`) or sudden pointer jumps ($> 0.20$ normalized distance).

### Potential Conflicts with a New Independent Gaze-Tree State Machine
1. **Grid vs. Hierarchical Topology**:
   - `DwellSelector` assumes flat 2D hit-testing over `AACBoard` tiles ([`backend/app/aac/board.py#L157-L165`](file:///d:/VOXGAZE/backend/app/aac/board.py#L157-L165)).
   - A gaze-tree navigates hierarchical branch levels (e.g., Root $\rightarrow$ Category $\rightarrow$ Phrase/Leaf) using discrete directional choices (`LEFT`, `RIGHT`, `CENTER`, `BLINK`) rather than free-floating 2D coordinates on an 11-tile grid.
2. **State Variable Clashes**:
   - `DwellSelector.state` uses `SelectionState` (`IDLE`, `DWELLING`, `CANDIDATE_HIGHLIGHTED`, `CONFIRMED`, `PAUSED_TRACKING_LOSS`).
   - A tree state machine requires its own states (e.g., `TREE_IDLE`, `NODE_HOVER`, `SUBTREE_ACTIVE`, `CONFIRMING_BRANCH`, `BACKTRACK`). Reusing or overloading `DwellSelector` would pollute its invariants and break existing unit tests.
3. **Telemetry & API Coupling**:
   - `AACService.get_status()` ([`backend/app/aac/server.py#L110-L115`](file:///d:/VOXGAZE/backend/app/aac/server.py#L110-L115)) expects `dwell.active_tile` and `dwell.candidate_tile` to reference `WordTile.id`.
   - Modifying `DwellSelector` directly would conflict with `test_dwell_selector_*` in `tests/unit/test_aac_mvp.py`.
4. **Resolution**: The gaze-tree state machine must be implemented as a separate, self-contained class (e.g., `GazeTreeStateMachine` or `GazeTreeNavigator`) rather than modifying `DwellSelector`.

---

## 5. Offline Speech Synthesis (`/api/aac/speak`) & Thread Blocking

### Implementation Hierarchy
- Defined in [`backend/app/aac/speech.py#L28-L148`](file:///d:/VOXGAZE/backend/app/aac/speech.py#L28-L148).
- Fallback sequence in `_execute_speak(phrase)`:
  1. **Piper Local Process** (lines 82–97): If `piper_binary` and `piper_model` exist on disk, runs `subprocess.run([piper_binary, "--model", piper_model, "--output_file", "-"], input=phrase.encode(), timeout=5.0)`.
  2. **Windows SAPI Desktop Voice** (lines 100–139): On Windows, executes a PowerShell script:
     ```powershell
     powershell.exe -NoProfile -NonInteractive -Command "Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Speak('...')"
     ```
     Has a `timeout=6.0` seconds.
  3. **Browser Web Speech Signal** (lines 141–148): Returns `engine_used = "CLIENT_BROWSER_SPEECH_SIGNAL"`.

### What Blocks the Server Thread?
1. **Synchronous PowerShell Invocation**:
   - In [`backend/app/aac/server.py#L510-L513`](file:///d:/VOXGAZE/backend/app/aac/server.py#L510-L513), the endpoint is defined as synchronous `def speak_phrase(...)`.
   - `srv.speak_phrase(...)` ([`server.py#L182`](file:///d:/VOXGAZE/backend/app/aac/server.py#L182)) calls `tts_engine.speak(phrase)` with default `non_blocking=False`.
   - In `speech.py#L109-L114`, `subprocess.run(["powershell.exe", ...])` synchronously spawns PowerShell, loads .NET `System.Speech`, and plays audio to completion. This blocks the calling thread for $1.5\text{ s}$ to $5.0\text{ s}$.
2. **Global Mutex Contention**:
   - `_execute_speak()` acquires `with self._lock:` ([`speech.py#L81`](file:///d:/VOXGAZE/backend/app/aac/speech.py#L81)), serializing all speech synthesis calls.
3. **Non-Blocking Capability**:
   - `OfflineTTSEngine.speak()` supports `non_blocking=True` ([`speech.py#L68-L76`](file:///d:/VOXGAZE/backend/app/aac/speech.py#L68-L76)), which spawns a daemon thread `threading.Thread(target=self._execute_speak, ...)` and returns immediately with `engine_used = "ASYNC_LAUNCHED"`. However, `server.py` does not currently enable `non_blocking=True`.

---

## 6. Guards on `is_selection_enabled`, `PermissionError`, and Covering Tests

### Guard Implementation
1. **Provider Invariant**:
   - In [`backend/app/aac/gaze_input.py#L144-L148`](file:///d:/VOXGAZE/backend/app/aac/gaze_input.py#L144-L148), `LiveGazeProvider` explicitly sets:
     ```python
     self.is_selection_enabled = False
     self.unsupported_reason = (
         "LIVE_SELECTION_DISABLED: Calibration directional separation gates (H/V) "
         "have not passed in physical trials. Live word activation remains disabled."
     )
     ```
2. **Service Gating Invariant**:
   - In [`backend/app/aac/server.py#L133-L136`](file:///d:/VOXGAZE/backend/app/aac/server.py#L133-L136), `AACService.append_tile()` strictly checks:
     ```python
     if self.active_mode == "LIVE" and not self.live_provider.is_selection_enabled:
         raise PermissionError(self.live_provider.unsupported_reason or "Live word selection is disabled.")
     ```
   - In [`backend/app/aac/server.py#L496-L497`](file:///d:/VOXGAZE/backend/app/aac/server.py#L496-L497), the REST handler intercepts `PermissionError` and maps it to HTTP 403 Forbidden.
3. **Calibration Persistence Guard**:
   - In [`backend/app/aac/server.py#L350-L396`](file:///d:/VOXGAZE/backend/app/aac/server.py#L350-L396), `complete_calibration_session()` returns `is_selection_enabled = False` because empirical observation gates did not validate 5-direction separation.

### Test Coverage
The following unit tests in [`tests/unit/test_aac_mvp.py`](file:///d:/VOXGAZE/tests/unit/test_aac_mvp.py) strictly enforce these guards:
- [`lines 201-207`](file:///d:/VOXGAZE/tests/unit/test_aac_mvp.py#L201-L207) (`test_simulated_gaze_provider_is_clearly_marked_simulated`): Asserts `tel.is_selection_enabled is True` in DEMO mode.
- [`lines 215-230`](file:///d:/VOXGAZE/tests/unit/test_aac_mvp.py#L215-L230) (`test_live_research_provider_strictly_disables_word_activation`): Asserts `tel.is_selection_enabled is False` and verifies the exact `unsupported_reason` string.
- [`lines 284-295`](file:///d:/VOXGAZE/tests/unit/test_aac_mvp.py#L284-L295) (`test_service_live_mode_blocks_phrase_append`): Switches to `LIVE` mode and asserts that `service.append_tile("water")` raises `PermissionError` containing `"LIVE_SELECTION_DISABLED"`.
- [`lines 309-326`](file:///d:/VOXGAZE/tests/unit/test_aac_mvp.py#L309-L326) (`test_calibration_status_reports_empirical_gate_failures_without_fabrication`): Ensures calibration status loads honest gate measurements and never fabricates pass status.
- [`lines 327-358`](file:///d:/VOXGAZE/tests/unit/test_aac_mvp.py#L327-L358) (`test_calibration_session_lifecycle_and_safety_invariants`): Completes calibration and asserts `comp_res["is_selection_enabled"] is False`.

---

## 7. Risks and Minimal File Scope for `--experimental-demo` Mode

### Architectural and Safety Risks
1. **Regressing Existing Acceptance Gates / Breaking Safety Tests**:
   - If `LiveGazeProvider.is_selection_enabled` default is changed to `True`, `test_live_research_provider_strictly_disables_word_activation` and `test_service_live_mode_blocks_phrase_append` will immediately fail.
   - *Mitigation*: The default value of `is_selection_enabled` must remain `False`. Any experimental live selection must be encapsulated within a distinct, opt-in mode (e.g., `EXPERIMENTAL_GAZE_TREE`) activated exclusively when the CLI flag is present.
2. **CPU Starvation in the Video Capture Loop**:
   - Running MediaPipe FaceLandmarker + Model Crop + MobileOne inference sequentially on every frame requires $20\text{--}35\text{ ms}$ on CPU.
   - If run on the Uvicorn asyncio main thread, it will degrade WebSocket telemetry and HTTP response latencies.
   - *Mitigation*: Run camera capture and inference in a dedicated background worker thread communicating via thread-safe state snapshots.
3. **Audio Playback Latency Stalling Video**:
   - Synchronous `powershell.exe` execution in `speak()` blocks for up to $6\text{ s}$.
   - *Mitigation*: Always invoke `tts_engine.speak(..., non_blocking=True)` so audio synthesis runs asynchronously without stalling tracking or state evaluation.
4. **Coordinate Drift and False Activations**:
   - Using uncalibrated or raw regression coordinates without center deadzones can cause erratic tree node hopping.
   - *Mitigation*: Use hierarchical tree navigation with a resting center deadzone and explicit confirmation blinks ($300\text{--}800\text{ ms}$).

### Smallest Set of Files to Touch for `--experimental-demo` Mode
To build an experimental gaze-tree demo cleanly without modifying default paths or weakening existing tests:
1. **`backend/app/aac/gaze_tree.py`** *(NEW file)*:
   - Contains the `GazeTreeNode`, `GazeTreeHierarchy`, and `GazeTreeNavigator` state machines.
   - Implements hierarchical node traversal, dwell timing, candidate highlighting, and explicit blink confirmation independently of `DwellSelector`.
2. **`scripts/run_aac_mvp.py`**:
   - Add `--experimental-demo` CLI argument.
   - If enabled, initialize camera capture pipeline (reusing MediaPipe and MobileOne estimator from `run_controlled_model_comparison.py`) and bind the gaze-tree navigator.
3. **`backend/app/aac/server.py`**:
   - Add routes `/api/gaze-tree/status`, `/api/gaze-tree/confirm`, `/api/gaze-tree/backtrack` (or mount a sub-router) to expose tree state to the UI without touching `/api/board` or `/api/mode`.
4. **`backend/app/aac/static/` (`index.html`, `app.js`, `style.css`)**:
   - Add visual tree overlay / navigation branches when experimental mode is detected.
5. **`tests/unit/test_gaze_tree.py`** *(NEW test file)*:
   - Comprehensive unit tests covering tree hierarchy, navigation transitions, center deadzones, explicit confirmation, and safety pauses.
