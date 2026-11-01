# VoxGaze Demonstration Script & Presentation Guide

**A Structured Walkthrough of the VoxGaze Assistive Communication Prototype**

---

## 1. Demonstration Overview & Scientific Scope

This script provides a 7–10 minute live presentation and technical walkthrough of VoxGaze.

### Mandatory Speaker Disclaimers
1. **Research Prototype**: VoxGaze is an assistive communication research prototype, **not a clinically validated medical device**.
2. **Acceptance Gate Status**: In multi-session physical benchmarking across independent days, the system achieved $100\%$ horizontal accuracy ($12/12$) and zero center false activations ($0/30$), but failed the vertical acceptance threshold in Session B ($75\% < 85\%$).
3. **Safety Status**: In accordance with project safety invariants, **live gaze-to-speech selection remains strictly disabled** (`is_selection_enabled = False`). The interaction loop is demonstrated using Demo Mode with offline Piper TTS synthesis.
4. **Blink Confirmation**: Blink separation is calibrated and evaluated in data contracts ($\Delta\text{EAR} \ge 0.05$), but real-time blink detection is not yet wired to the AAC dwell selector.

---

## 2. Screenshot Checklist (6 Mandatory Figures)

Capture and archive the following 6 screenshots in `artifacts/screenshots/`:

- [ ] **Figure 1 (`fig1_comparison_display.png`)**: Full-screen 5-button evaluation display showing black background, highlighted active target (`LEFT`), settle indicator, and `Block 3/5, Trial 4/9`.
- [ ] **Figure 2 (`fig2_aac_board_ui.png`)**: Assistive AAC web interface showing the 11-tile core vocabulary layout (`YES`, `NO`, `HELP`, `WATER`, etc.), phrase builder ribbon, and mode toggle.
- [ ] **Figure 3 (`fig3_candidate_highlight.png`)**: Active dwell progress on an AAC tile (`HELP`) showing visual candidate highlight prior to confirmation.
- [ ] **Figure 4 (`fig4_gate_evaluator_terminal.png`)**: CLI execution of `evaluate_gate.py` displaying the multi-session comparison report and honest `FAIL` decision log.
- [ ] **Figure 5 (`fig5_quantization_comparison.png`)**: Terminal benchmark output of `scripts/quantize_resnet34.py` showing the rejection of dynamic vs static INT8 models.
- [ ] **Figure 6 (`fig6_terminal_test_suite.png`)**: Full pytest suite passing (174/174 tests) in `.venv`.

---

## 3. Step-by-Step Demonstration Walkthrough

### Part 1: Problem Statement & System Architecture (2 Minutes)
- **Visual**: Show system architecture diagram from `README.md`.
- **Narration**:
  > *"Traditional eye-tracking AAC requires users with ALS or motor impairments to type letter-by-letter on a virtual keyboard. Staring at small keys for 400 to 1000 ms per letter causes intense visual fatigue and requires expensive infrared hardware.*
  >
  > *VoxGaze re-engineers this pipeline for standard 30 FPS webcams. Instead of pixel-precise typing, we map deliberate coarse directional gaze gestures—LEFT, RIGHT, UP, DOWN—into structured semantic intents.*
  >
  > *Our architecture decouples raw gaze perception from personalized spatial calibration, enforces dual-threshold hysteresis, displays visual candidate previews, and requires explicit confirmation before any speech is synthesized."*

### Part 2: The Controlled 5-Block Benchmark Protocol (2 Minutes)
- **Visual**: Show `Figure 1` (5-button OpenCV evaluation canvas) and run mock comparison:
  ```powershell
  python scripts/run_controlled_model_comparison.py --mock
  ```
- **Narration**:
  > *"To evaluate perception backends under identical physical conditions, we developed a 5-block protocol: 2 calibration blocks to train user-specific ridge mappers, followed by 3 held-out evaluation blocks with 12 peripheral and 15 center trials in balanced pseudo-random order.*
  >
  > *Every trial enforces a 1.5-second visual settle period followed by 3.0 seconds of measurement with wall-clock timestamped dwell tracking. Notice that no raw camera frames are ever written to disk—protecting participant privacy."*

### Part 3: Empirical Multi-Session Gate Results (2 Minutes)
- **Visual**: Show `Figure 4` (Terminal gate evaluation) and `docs/RESULTS.md`.
  ```powershell
  python scripts/evaluate_gate.py artifacts/observations/cmp_v16_20261029_125352_0295b8_comparison_report.json artifacts/observations/cmp_v16_20261101_164725_06ac34_comparison_report.json
  ```
- **Narration**:
  > *"Rather than claiming success from a single session, VoxGaze enforces an automated multi-session acceptance gate across independent days.*
  >
  > *In Session A, MobileOne S0 achieved 100% peripheral accuracy and 0% center false activations. But in Session B, peripheral accuracy dropped to 75%—narrowly missing the 85% requirement due to vertical margin sensitivity on UP and DOWN targets.*
  >
  > *Because our gate strictly requires both sessions to pass, the evaluator output is an honest FAIL. Crucially, horizontal accuracy was 100% across all sessions (12/12 trials), and center false activation was 0.0% by median (0/30 trials). Because the gate failed, live selection remains disabled."*

### Part 4: ResNet-34 INT8 Quantization Negative Result (1.5 Minutes)
- **Visual**: Show `Figure 5` (Quantization report).
- **Narration**:
  > *"We also investigated whether a larger ResNet-34 model could be accelerated to real-time speeds using INT8 quantization. Dynamic INT8 yielded zero speedup on CPU due to runtime overhead. Static INT8 achieved 22 ms latency but suffered catastrophic 15-degree angular error—larger than our target separation margins.*
  >
  > *We formally rejected INT8 quantization and documented it as a negative result. MobileOne S0 remains our primary edge candidate, delivering 22 to 24 ms end-to-end latency at FP32."*

### Part 5: Assistive AAC Board & Audio Speech Demo (2 Minutes)
- **Visual**: Open `http://127.0.0.1:8000` (`Figure 2` and `Figure 3`).
  ```powershell
  python scripts/run_aac_mvp.py
  ```
- **Live Action**:
  1. Show the 11-tile assistive communication board.
  2. Demonstrate how hovering over a tile (`WATER`) accumulates dwell progress.
  3. Highlight how the tile enters `CANDIDATE_HIGHLIGHTED` state.
  4. Confirm the tile into the phrase ribbon ("I WANT WATER").
  5. Click **Speak Phrase** to trigger local, offline Piper TTS audio speech output.
- **Narration**:
  > *"Here is the working AAC interface. Tiles subtend large visual angles to minimize foveal strain. In this demo mode, hovering initiates visual dwell feedback. When dwell completes, the tile enters visual preview. Only upon explicit confirmation is the word committed to the phrase buffer.*
  >
  > *Speech output is synthesized entirely locally using offline Piper TTS with zero external network requests."*

---

## 4. Rehearsal Checklist & Q&A Preparation

### Rehearsal Protocol
- [ ] **Rehearsal 1**: Dry-run terminal commands and verify timing ($\le 8\text{ min}$).
- [ ] **Rehearsal 2**: Full walkthrough with browser AAC demo and audio verification.

### Antigravity Q&A Defense

**Q1: Why did Session B fail when Session A scored 100%?**
> *A: Horizontal gaze was 100% in both sessions. The failures in Session B occurred exclusively on vertical targets (`UP` and `DOWN`) where the calibrated median coordinate fell slightly inside the deadzone boundary ($y=0.374$ vs $0.35$ for UP; $y=0.612$ vs $0.65$ for DOWN). This demonstrates why multi-session gates are essential to detect subtle postural and vertical baseline sensitivity.*

**Q2: Why is live eye selection disabled in the demo?**
> *A: Project Safety Invariant #1 strictly prohibits speech without verified empirical gates. Because Session B scored 75% (< 85%), live word activation remains locked (`is_selection_enabled = False`). We demonstrate the interaction and speech loop safely via Demo Mode.*

**Q3: Why not use a cloud LLM for predictive gaze text completion?**
> *A: VoxGaze is strictly offline-first. Cloud dependencies introduce variable network latency, privacy risks with sensitive personal speech, and hallucination risks where unconfirmed words might be spoken aloud.*
