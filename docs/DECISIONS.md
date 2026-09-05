# Architecture Decision Records (ADRs)

## ADR-001: Use MediaPipe Tasks Face Landmarker in Video Mode
- **Context**: Gaze tracking requires facial geometry and iris localization on consumer webcams without specialized IR hardware.
- **Decision**: Use the pretrained MediaPipe Face Landmarker Tasks API (`RunningMode.VIDEO`) with monotonic timestamps (`time.perf_counter()`).
- **Rationale**: Solves expensive landmark localization in real-time (~10-15 ms) without training custom perception models. `VIDEO` mode provides temporal coherence.
- **Status**: Approved.

---

## ADR-002: Local Eye-Axis Coordinate Normalization
- **Context**: Screen-space coordinates are sensitive to head movement, distance, and tilt.
- **Decision**: Normalize iris center position relative to inner/outer eye canthi ($H$) and upper/lower eyelids ($V$), computing local ratios in $[0, 1]$.
- **Rationale**: Isolates rotational eye movement from gross translation and makes calibration geometry robust.
- **Status**: Approved.

---

## ADR-003: Discrete Directional States with Dual-Threshold Hysteresis
- **Context**: Raw iris ratios suffer from micro-saccades and boundary jitter. Single-threshold systems oscillate at boundaries.
- **Decision**: Map continuous $(H, V)$ features into discrete states (`CENTER`, `LEFT`, `RIGHT`, `UP`, `DOWN`) using separate entry and exit thresholds per direction.
- **Rationale**: Eliminates state flickering and gives users a solid, predictable physical control vocabulary.
- **Status**: Approved.

---

## ADR-004: Zero LLM / Zero Cloud in Critical Loop
- **Context**: Assistive communication requires deterministic guarantees, zero hallucination risk, low latency, and offline availability.
- **Decision**: Intent mapping resolves to deterministic phrase templates stored in a local JSON registry.
- **Rationale**: Protects patient communication safety, enables auditing, and guarantees fast execution ($<10\text{ ms}$).
- **Status**: Approved.

---

## ADR-005: Speech Requires Explicit Preview and Confirmation
- **Context**: The "Midas Touch" problem causes unintended selections if gaze directly triggers speech.
- **Decision**: All candidate selections enter a persistent visual preview. Speech is emitted only after an intentional confirmation blink (or deliberate gesture).
- **Rationale**: Invariant `gaze -> preview -> confirm -> speech`. Prevents accidental utterances and allows immediate cancellation.
- **Status**: Approved.

---

## ADR-006: Abstain on Ambiguous Tracking
- **Context**: When face tracking is lost or visibility drops, guessing produces false activations.
- **Decision**: Transition state to `ABSTAIN` / `TRACKING_PAUSED` whenever detection confidence drops below threshold or landmarks are missing.
- **Rationale**: "I do not know" is safer than an incorrect utterance.
- **Status**: Approved.

---

## ADR-007: Modular Architecture (FastAPI Backend + React Frontend)
- **Context**: Assistive UI needs high accessibility, dynamic rendering, and responsive visual feedback, decoupled from computer vision execution.
- **Decision**: Decouple CV/Perception and FSM in Python (FastAPI/WebSocket) from the UI layer in React.
- **Rationale**: Clean separation of concerns; enables independent testing of perception algorithms and headless regression benchmarking.
- **Status**: Approved.
