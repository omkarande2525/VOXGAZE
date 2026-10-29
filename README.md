# VoxGaze AI

**Personalized, offline-first assistive gaze communication for commodity webcams.**

---

## 1. Overview & System Mission

VoxGaze converts deliberate, coarse physical eye movements (`LEFT`, `RIGHT`, `UP`, `DOWN`) into structured assistive communication intents rather than requiring exhaustive, high-fatigue letter-by-letter typing. The system is designed from the ground up for low-cost hardware (standard 720p/1080p webcams at 30 FPS) with zero cloud dependencies, zero external network calls, and strict safety invariants against accidental speech generation.

### Core Architectural Invariants
1. **No Speech Without Confirmation**: Gaze movements never directly trigger speech output. Every candidate intent requires visual preview and an explicit confirmation gesture (deliberate blink, $300\text{ ms} - 800\text{ ms}$).
2. **Protocol-Defined Intent**: Physical eye movement acts strictly as an input control vector; the interaction state determines semantic meaning.
3. **Abstain on Ambiguity**: Poor illumination, head turning beyond limits, or landmark loss immediately yields `ABSTAIN` / `TRACKING_PAUSED` rather than guessing.
4. **Offline First & Private**: All inference (MediaPipe, lightweight neural estimators, Piper TTS) executes locally on CPU. No raw webcam frames are ever persisted to disk.
5. **Decoupled Architecture**: Pure geometry mathematics and feature extraction are decoupled from UI rendering, and gaze estimation is decoupled from personalized coordinate mapping.

---

## 2. End-to-End Interaction Pipeline

```text
Webcam (30 FPS, RGB)
  │
  ▼
MediaPipe Face Landmarker (Iris Mesh & Facial Contours)
  │
  ▼
Quality Gate (Abstains on face loss, excessive roll/yaw, landmark jitter)
  │
  ▼
Gaze Feature Extraction:
  ├─ Geometric Baseline (Bilateral canthal & eyelid normal ratios)
  └─ Neural Estimators (MobileOne-S0 / ResNet-34 crop inference)
  │
  ▼
Personalized Spatial Mapper (Ridge regression fit on user calibration)
  │
  ▼
Smoothing & Dual-Threshold Hysteresis
  │
  ▼
Directional State (CENTER / LEFT / RIGHT / UP / DOWN / ABSTAIN)
  │
  ▼
Assistive State Machine (5-button board navigation)
  │
  ▼
Visual Candidate Preview (Target highlight & dwell feedback)
  │
  ▼
Explicit Confirmation Blink (EAR drop 300–800 ms)
  │
  ▼
Deterministic Phrase Registry & Offline Piper TTS Audio Output
```

---

## 3. Empirical Model Evaluation Protocol

VoxGaze employs a rigorous 5-block controlled comparison protocol to benchmark candidate gaze estimation backends under identical physical conditions:

- **Block Structure**: 5 blocks $\times$ 9 trials = 45 trials total.
  - **Blocks 1–2 (Calibration)**: 4 peripheral targets (`LEFT`, `RIGHT`, `UP`, `DOWN`) + 5 `CENTER` fixations per block (18 trials total). Used to fit the personalized ridge mapper separately per candidate model.
  - **Blocks 3–5 (Held-Out Evaluation)**: 4 peripheral targets + 5 `CENTER` fixations per block in balanced pseudo-random order (12 peripheral and 15 center held-out trials).
- **Trial Timing**:
  - 1.5 s settling window (visual target presented, gaze settles).
  - 3.0 s measurement window (real-timestamped continuous dwell and frame-by-frame coordinate tracking).
- **Safety & Integrity Invariants**:
  - Full-screen black OpenCV display showing balanced 5-button target layout with current target highlighted and settling countdown.
  - ESC or Q abort writes a partial report marked `INCOMPLETE` without corrupting metrics.
  - Numerical telemetry only: raw camera frames are **never** written to disk.
  - Each session writes a uniquely hashed timestamped JSON report in `artifacts/observations/`.

### Multi-Session Acceptance Gate
To prevent premature adoption from single-session flukes, a candidate model must satisfy the following criteria across **at least two independent sessions** (conducted on separate calendar days or with a verified camera repositioning):
1. **Peripheral Command Accuracy**: $\ge 85\%$ ($11/12$ or $12/12$ held-out trials).
2. **Center False Activation Rate**: $\le 5\%$ ($0/15$ held-out trials falsely leaving center deadzone $[0.35, 0.65]$ by trial median).
3. **Transient Center Escapes**: Any frame where pointer leaves center deadzone is recorded and reported separately from trial median.
4. **Adoption Differential**: The challenger model must demonstrate $\ge +15.0$ percentage points higher accuracy or $\ge 30\%$ lower latency than baseline to warrant architectural adoption.

---

## 4. Benchmark Measurements & Findings

### Session A Physical Comparison (2026-10-29)
*Hardware: Intel Core i7 Windows CPU, 30 FPS USB Webcam. Report: `cmp_v16_20261029_125352_0295b8`.*

| Metric | MobileOne S0 Hybrid | ResNet-34 Hybrid | Gate Requirement |
| :--- | :---: | :---: | :---: |
| **Held-Out Peripheral Accuracy** | **12 / 12 (100.0%)** | 11 / 12 (91.7%) | $\ge 85\%$ (PASS) |
| **Center False Activation (Trial Median)** | **0 / 15 (0.0%)** | **0 / 15 (0.0%)** | $\le 5\%$ (PASS) |
| **Center Transient Escapes (Any Frame)** | 7 / 15 trials (1–3 frames/trial) | 9 / 15 trials (1–7 frames/trial) | Reported honestly |
| **Dwell Containment ($\ge 400\text{ ms}$)** | 12 / 12 (100.0%) | 12 / 12 (100.0%) | Secondary validation |
| **Pipeline Latency (p50)** | **24.66 ms** | 130.19 ms | MobileOne ~5.3x faster |
| **Pipeline Latency (p95)** | **28.53 ms** | 147.22 ms | Target $< 30\text{ ms}$ |

> [!NOTE]
> **Transient Center Escapes**: Both models passed the median-based center safety gate ($0/15$), but frame-by-frame analysis revealed transient escapes on 7/15 (MobileOne) and 9/15 (ResNet-34) center trials. These excursions were brief (1–3 frames out of ~90) and predominantly biased upwards (`UP`), caused by baseline vertical calibration proximity rather than post-saccadic lag. Dwell hysteresis and explicit blink confirmation prevent these excursions from causing accidental selections, but selection remains intentionally disabled during evaluation.

> [!IMPORTANT]
> **Omission of Target Acquisition Latency**: Initial prototype runs reported $0.0\text{ ms}$ acquisition latency because metric logging began after the 1.5 s settling window (when the eye was already fixated on target). This metric was methodologically invalid and has been formally removed from acceptance evaluation.

### Session B Physical Replication (Pending Friday)
*Validation requirement: Independent session conducted on a separate calendar day.*

| Metric | MobileOne S0 Hybrid | ResNet-34 Hybrid | Target Gate |
| :--- | :---: | :---: | :---: |
| **Peripheral Accuracy (Held-Out)** | *[Pending Session B]* | *[Pending Session B]* | $\ge 85\%$ ($11/12$) |
| **Center False Activation Rate** | *[Pending Session B]* | *[Pending Session B]* | $\le 5\%$ ($0/15$) |
| **Center Transient Escapes** | *[Pending Session B]* | *[Pending Session B]* | Characterized |
| **Pipeline Latency (p50)** | *[Pending Session B]* | *[Pending Session B]* | $< 30\text{ ms}$ |
| **Multi-Session Gate Decision** | *[Pending Session B]* | *[Pending Session B]* | `PASS` / `FAIL` |

---

## 5. ResNet-34 Post-Training Quantization Study (Negative Result)

In Task T4, post-training INT8 quantization was systematically benchmarked for ResNet-34 on CPU to assess whether inference latency could be reduced from ~61 ms to the 25–30 ms real-time budget:

| Precision Variant | Mean Angular Error | Max Yaw Error | p50 Latency | p95 Latency | Decision |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FP32 Baseline** | Baseline ($0.0^\circ$) | Baseline | 61.47 ms | 64.91 ms | Accurate, High Latency |
| **Dynamic INT8** | $0.049^\circ$ | $0.23^\circ$ | 63.95 ms | 68.20 ms | **Rejected** (No speedup) |
| **Static INT8** | $15.01^\circ$ | $40.85^\circ$ | 21.90 ms | 23.40 ms | **Rejected** (Catastrophic drift) |

**Conclusion**: Dynamic INT8 quantization yielded zero CPU speedup due to run-time activation quant overhead. Static INT8 achieved the desired latency (21.9 ms) but suffered severe accuracy degradation ($15^\circ$ average yaw error, exceeding the $12^\circ - 25^\circ$ target separation margin). Post-training INT8 quantization was therefore formally **rejected**. ResNet-34 remains at FP32 (~130 ms end-to-end), making MobileOne-S0 the primary low-latency candidate.

---

## 6. Project Structure

```text
VOXGAZE/
├── backend/
│   └── app/
│       ├── aac/             # Assistive board, dwell selector, phrase builder, server
│       ├── calibration/     # 5-point calibration protocol, ridge mapper, gate evaluation
│       ├── core/            # System data contracts, coordinate boundaries
│       └── perception/      # Face crop extraction, MediaPipe, MobileOne, ResNet-34
├── scripts/
│   ├── run_controlled_model_comparison.py  # 5-block physical comparison protocol runner
│   ├── evaluate_gate.py                    # Multi-session acceptance gate evaluator
│   ├── quantize_resnet34.py                # ONNX INT8 quantization & latency benchmark
│   └── run_aac_mvp.py                      # Local AAC demonstration server
├── tests/
│   └── unit/                # 174 automated unit tests
├── artifacts/
│   └── observations/        # Immutable JSON evaluation reports & forensics
└── docs/                    # Specifications, acceptance gates, ADRs, UI principles
```

---

## 7. Developer Quickstart

### Prerequisites
- Python 3.10+ (tested on Python 3.10.11 and 3.13)
- Windows / Linux / macOS
- Virtual environment with dependencies installed:
  ```powershell
  python -m venv .venv
  .venv\Scripts\activate
  pip install -r requirements.txt
  ```

### Running Automated Tests
The repository enforces 100% pass on all unit tests:
```powershell
.venv\Scripts\pytest tests/
```
*(174 tests covering pure geometry, ridge mapper, FSM lifecycle, offline TTS, synthetic jitter, gate evaluator, and quantization models).*

### Running the Controlled Model Comparison Protocol
1. **Mock Dry Run** (runs in seconds without camera, headless):
   ```powershell
   python scripts/run_controlled_model_comparison.py --mock
   ```
2. **Physical Live Session** (opens 5-button calibration & evaluation display):
   ```powershell
   python scripts/run_controlled_model_comparison.py
   ```
   - Follow on-screen visual prompts (1.5 s settle, 3.0 s dwell per trial).
   - ESC or Q cleanly aborts and saves partial telemetry marked `INCOMPLETE`.

### Evaluating Multi-Session Acceptance Gates
Pass two or more real session reports to verify cross-session gates:
```powershell
# Single session check (outputs INVALID - requires >=2 independent sessions):
python scripts/evaluate_gate.py artifacts/observations/cmp_v16_20261029_125352_0295b8_comparison_report.json

# Multi-session cross-day evaluation:
python scripts/evaluate_gate.py <session_a_report.json> <session_b_report.json>
```

### Launching the Assistive AAC Interface
```powershell
python scripts/run_aac_mvp.py
```
Open `http://127.0.0.1:8000` in a web browser to access the 11-tile assistive communication board.
