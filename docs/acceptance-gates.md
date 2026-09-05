# VoxGaze Acceptance Gates

This document defines the strict, testable acceptance criteria for every development milestone. No milestone may be marked as PASSED without meeting 100% of its required gates.

---

## Milestone M0: Governance, Scaffolding & Data Contracts
- [ ] **Repository Structure**: Root directories `docs/`, `backend/app/`, `config/`, `data/`, `scripts/`, `tests/`, `.codex/skills/` exist.
- [ ] **Governance & Rules**: `AGENTS.md`, `README.md`, `.gitignore`, `.env.example`, `requirements.txt` are created.
- [ ] **Architecture Documentation**: `docs/DECISIONS.md`, `docs/data-contracts.md`, `docs/architecture.md`, `docs/research-plan.md`, `docs/ui-principles.md`, `docs/non-goals.md`, and `docs/acceptance-gates.md` are present and consistent.
- [ ] **Data Contracts**: Pydantic models for bilateral `GazeFeatures`, `CalibrationProfile`, `GazeState`, `IntentCandidate`, `PreviewState`, and `InteractionEvent` validate without syntax errors.
- [ ] **Single Source of Truth**: `config/default_intents.json` contains validated fast-path and domain-hierarchy mappings.
- [ ] **Codex Skills**: `.codex/skills/` contains `voxgaze-engineering`, `gaze-validation`, and `accessibility-ui`.
- [ ] **No Premature Modules**: Downstream modules (UI, FSM, Speech, API routes, SLMs) are not stubbed or implemented.
- [ ] **Test Runner Validation**: `pytest` executes and validates M0 contracts.

---

## Milestone M1: Webcam & Iris Sensor Validation Gate
The goal of M1 is to **prove or falsify the assumption that a commodity webcam + MediaPipe Face Landmarker can provide sufficiently stable, user-calibrated four-direction gaze and intentional blink input**.

- [ ] **Camera Profiling**:
  - Script reports requested vs. actual resolution and FPS.
  - Video backend and frame acquisition latency measured.
- [ ] **Landmark Pipeline**:
  - MediaPipe Face Landmarker runs in `RunningMode.VIDEO` using monotonic time (`time.perf_counter()`).
  - Face detection confidence $\ge 0.6$, Iris landmarks (indices 468–477) accurately resolved.
- [ ] **Pure Geometry**:
  - $H, V$ local eye ratio normalization and $EAR$ (Eye Aspect Ratio) calculation executed via pure functions with zero OpenCV/UI dependencies.
  - Bilateral eye signature contract supported.
- [ ] **Statistically Robust Calibration**:
  - 5-point calibration protocol (`CENTER`, `LEFT`, `RIGHT`, `UP`, `DOWN`) plus `CLOSED` blink baseline.
  - 0.5s warm-up + 1.5s collection window with sample filtering.
  - Computes Median, MAD (Median Absolute Deviation), and range spread.
  - Quality gate rejects calibration if sample count $< 30$ or MAD exceeds stability threshold.
  - Profile serialized with versioning, camera config, and timestamp.
- [ ] **Gaze Classification & Hysteresis**:
  - Dual enter/exit thresholds per direction to eliminate boundary jitter.
  - Quality Gate produces `ABSTAIN` / `NO_FACE` / `UNCALIBRATED` when landmarks degrade.
- [ ] **Synthetic Test Suite**:
  - Synthetic gaze generator runs unit tests across: clean trajectories, Gaussian jitter, slow head drift, boundary oscillation, dropouts, blink vs. intentional closure.
  - Zero false activations on noisy synthetic baseline.
- [ ] **Empirical Benchmark & Machine-Readable Reporting**:
  - 20 controlled trials per direction (100 trials total).
  - Event-level direction accuracy $\ge 90\%$ under normal room lighting.
  - Frame-level stability reported.
  - Latency profiled: p50 and p95 $\le 30\text{ ms}$ processing time.
  - 10-minute continuous run without unhandled exceptions or memory growth.
  - Outputs generated under `artifacts/m1/`:
    - `camera_report.json`
    - `calibration_report.json`
    - `direction_confusion_matrix.csv`
    - `latency_summary.json`

---

## Milestone M1.5: Minimal 4-Command Intent & Speech Loop
- [ ] Gaze candidate triggers Preview state for 4 direct commands.
- [ ] Deliberate confirmation blink triggers offline Piper TTS synthesis.
- [ ] Look-away gesture or cancel timeout cleanly aborts to IDLE without speech.
- [ ] No speech occurs without explicit confirmation gate.

---

## Milestone M2: Variable-Depth Hierarchical Menu
- [ ] Fast path (1 action + confirm) and Deep path (2–3 actions + confirm) functional.
- [ ] State machine enforces strict transition invariants.

---

## Milestone M3: Personalized Intent Configuration
- [ ] Intent spaces configured dynamically via JSON without code edits.

---

## Milestone M4: Research Baseline Comparison
- [ ] Controlled evaluation comparing VoxGaze vs. Character Dwell Keyboard baseline (WPM, selection time, error rate, abort rate, subjective load).
