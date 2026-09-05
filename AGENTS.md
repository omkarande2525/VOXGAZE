# VoxGaze Repository Instructions

## Read First
Before inspecting or modifying any code in this repository, consult the core specification documents:
1. [docs/acceptance-gates.md](file:///d:/VOXGAZE/docs/acceptance-gates.md) - Pass/fail criteria for each milestone.
2. [docs/data-contracts.md](file:///d:/VOXGAZE/docs/data-contracts.md) - Schemas for gaze features, calibration, states, and telemetry.
3. [docs/non-goals.md](file:///d:/VOXGAZE/docs/non-goals.md) - Explicit architectural and scientific boundaries.
4. [docs/DECISIONS.md](file:///d:/VOXGAZE/docs/DECISIONS.md) - Architectural Decision Records (ADRs).
5. [docs/architecture.md](file:///d:/VOXGAZE/docs/architecture.md) - End-to-end pipeline and module ownership.
6. [docs/research-plan.md](file:///d:/VOXGAZE/docs/research-plan.md) - Evaluation metrics, event-level accuracy, and baseline trials.
7. [docs/ui-principles.md](file:///d:/VOXGAZE/docs/ui-principles.md) - Assistive interface requirements.

---

## Current Milestone
**Milestone M0 (Governance & Scaffolding) & M1 (Sensor Validation)** ONLY.
Do not implement downstream modules (UI, FSM, TTS, API endpoints, SLMs) until M1 acceptance gates pass.

---

## Frozen v1 Critical Path
```text
Webcam
  ↓
MediaPipe Face Landmarker (Iris)
  ↓
Quality Gate (Abstain on loss)
  ↓
User Calibration (5-point + blink baseline)
  ↓
Normalized Gaze Features (Bilateral geometry)
  ↓
Smoothing + Dual-Threshold Hysteresis
  ↓
Directional State (CENTER / LEFT / RIGHT / UP / DOWN)
  ↓
Semantic FSM (Fast path & hierarchical traversal)
  ↓
Visual Preview (Persistent state & candidate display)
  ↓
Explicit Confirmation Blink (Deliberate closure)
  ↓
Deterministic Phrase Registry (Local JSON)
  ↓
Offline Piper TTS
  ↓
Audio Speech Output
```

---

## Absolute Invariants
1. **No Speech Without Confirmation**: The system must never convert raw gaze or unconfirmed candidate intents into speech.
2. **Intent is Protocol-Defined**: Do not infer hidden thoughts from eyes. Eye movement is a physical control signal; the interaction state determines meaning.
3. **Abstain on Ambiguity**: Poor lighting, face loss, or landmark jitter must trigger `ABSTAIN` / `TRACKING_PAUSED` rather than guessing.
4. **Offline First**: Zero cloud dependencies, zero external network calls in the critical loop.
5. **No Premature ML**: Rule-based geometry and hysteresis are the baseline. ML is added only when empirical measurements justify it.
6. **Separation of Concerns**: Pure geometry math has zero OpenCV/UI dependencies. Calibration is decoupled from the GazeEngine.
7. **Scientific Integrity**: Never fabricate benchmark measurements. Distinguish frame-level stability from event-level command accuracy. Report p50 and p95 latencies.
8. **Dependency Discipline**: Do not add dependencies unless strictly justified and verified against project requirements.

---

## Development Method
```text
Inspect → Plan → Implement → Test → Verify Acceptance Gates → Report
```
Never declare a milestone complete without passing its corresponding automated and empirical acceptance gates in [docs/acceptance-gates.md](file:///d:/VOXGAZE/docs/acceptance-gates.md).
