# VoxGaze Empirical Results & Acceptance Gate Reports

This document compiles the empirical evaluation history of VoxGaze across all physical benchmark sessions. All figures are derived directly from immutable JSON reports in `artifacts/observations/`.

---

## 1. Executive Summary: Multi-Session Acceptance Gate

To establish Milestone M1 (Sensor Validation), VoxGaze enforces a strict, non-negotiable multi-session acceptance gate evaluated by `scripts/evaluate_gate.py`:
- **Sample Size**: Minimum 2 independent sessions (separate calendar days or verified camera repositioning).
- **Peripheral Command Accuracy**: $\ge 85.0\%$ held-out trial accuracy in **EACH** session (at least $11/12$ trials).
- **Center Deadzone Safety**: $\le 5.0\%$ center false activation rate by trial median in **EACH** session ($0/15$ trials falsely leaving $[0.35, 0.65]$).
- **Pipeline Latency**: Target $< 30\text{ ms}$ on commodity CPU.

### Multi-Session Decision (Evaluated 2026-11-01)
*Decision log persisted at `artifacts/observations/gate_decision_20261101.txt`.*

| Model Backend | Session A (2026-10-29) | Session B (2026-11-01) | Multi-Session Gate Status |
| :--- | :---: | :---: | :---: |
| **MobileOne S0 Hybrid** | **PASS** (12/12, 0/15) | **FAIL** (9/12, 0/15) | **>>> FAIL <<<** |
| **ResNet-34 Hybrid** | **PASS** (11/12, 0/15) | **FAIL** (8/12, 0/15) | **>>> FAIL <<<** |

**Current Architectural Consequence**:
Because Session B fell short of the $85.0\%$ threshold ($75.0\%$ for MobileOne, $66.7\%$ for ResNet-34), Milestone M1 does **not** pass. Neither model is adopted for live speech generation, and `LiveResearchProvider.is_selection_enabled` remains strictly hardcoded to `False`.

---

## 2. Session History & Benchmark Records

### A. Historical Single-Session Benchmark (2026-10-25)
*Report: `artifacts/observations/cmp_v16_20261025_164239_0d9a6c_comparison_report.json`*
*Protocol: Early 3-Block Protocol (1 held-out block: 4 peripheral trials, 5 center trials).*

| Metric | MobileOne S0 Hybrid | ResNet-34 Hybrid | Gate Criteria | Audit Outcome |
| :--- | :---: | :---: | :---: | :---: |
| **Peripheral Accuracy** | 3 / 4 (75.0%) | 4 / 4 (100.0%) | $\ge 85\%$ on $\ge 12$ trials | **FAIL** (Insufficient sample size $n=4$) |
| **Center False Activation** | 2 / 5 (40.0%) | 2 / 5 (40.0%) | $\le 5\%$ on $\ge 15$ trials | **FAIL** ($40.0\% > 5.0\%$) |
| **Pipeline Latency (p50)** | 24.3 ms | 134.2 ms | $< 30\text{ ms}$ | MobileOne passed; ResNet failed |

**Forensic Audit (Task T1)**:
Initial analysis prematurely suggested adopting ResNet-34 based solely on $4/4$ vs $3/4$ peripheral accuracy. Task T1 audited this session and established that:
1. $n=4$ peripheral trials has zero statistical power.
2. Both models had catastrophic $40\%$ false activation rates from the center deadzone.
3. This session led directly to the design of the 5-block protocol (Task T3) to require $\ge 12$ peripheral and $\ge 15$ center trials before any gate evaluation.

---

### B. Session A Physical Comparison (2026-10-29)
*Report: `artifacts/observations/cmp_v16_20261029_125352_0295b8_comparison_report.json`*
*Protocol: Standard 5-Block Protocol (2 calibration blocks, 3 held-out blocks = 12 peripheral, 15 center).*

| Metric | MobileOne S0 Hybrid | ResNet-34 Hybrid | Gate Threshold | Result |
| :--- | :---: | :---: | :---: | :---: |
| **Peripheral Accuracy (Held-Out)** | **12 / 12 (100.0%)** | 11 / 12 (91.7%) | $\ge 85.0\%$ ($11/12$) | **PASS** (Both models) |
| **Center False Activation (Median)** | **0 / 15 (0.0%)** | **0 / 15 (0.0%)** | $\le 5.0\%$ ($0/15$) | **PASS** (Both models) |
| **Center Transient Escapes (Any Frame)** | 7 / 15 trials | 9 / 15 trials | Logged separately | Characterized |
| **Continuous Dwell ($\ge 400\text{ ms}$)** | 12 / 12 (100.0%) | 12 / 12 (100.0%) | Secondary validation | 100% containment |
| **Pipeline Latency (p50)** | **24.66 ms** | 130.19 ms | $< 30.0\text{ ms}$ | MobileOne passed; ResNet failed |
| **Pipeline Latency (p95)** | **28.53 ms** | 147.22 ms | $< 30.0\text{ ms}$ | MobileOne passed; ResNet failed |

**Key Findings**:
- First session where both models passed the median accuracy and center stability gates.
- MobileOne S0 demonstrated higher accuracy ($100\%$ vs $91.7\%$) at $5.3\times$ lower latency.
- Transient center escapes occurred in 7/15 (MobileOne) and 9/15 (ResNet) trials, primarily biased upwards (`UP`).

---

### C. Session B Physical Replication (2026-11-01)
*Report: `artifacts/observations/cmp_v16_20261101_164725_06ac34_comparison_report.json`*
*Protocol: Standard 5-Block Protocol (Identical conditions, independent calendar day).*

| Metric | MobileOne S0 Hybrid | ResNet-34 Hybrid | Gate Threshold | Result |
| :--- | :---: | :---: | :---: | :---: |
| **Peripheral Accuracy (Held-Out)** | 9 / 12 (75.0%) | 8 / 12 (66.7%) | $\ge 85.0\%$ ($11/12$) | **FAIL** (Both models) |
| **Center False Activation (Median)** | **0 / 15 (0.0%)** | **0 / 15 (0.0%)** | $\le 5.0\%$ ($0/15$) | **PASS** (Both models) |
| **Center Transient Escapes (Any Frame)** | 9 / 15 trials | 9 / 15 trials | Logged separately | Characterized |
| **Continuous Dwell ($\ge 400\text{ ms}$)** | 10 / 12 (83.3%) | 10 / 12 (83.3%) | Secondary validation | 83.3% containment |
| **Pipeline Latency (p50)** | **22.90 ms** | 111.50 ms | $< 30.0\text{ ms}$ | MobileOne passed; ResNet failed |
| **Pipeline Latency (p95)** | **38.20 ms** | 120.46 ms | $< 30.0\text{ ms}$ | MobileOne borderline; ResNet failed |

**Sampling Rate & Transient Escapes**:
- Across both physical sessions, both models execute sequentially on every frame within the comparison runner. Because ResNet-34 took ~110–130 ms per frame, the actual loop frequency was ~6.3 Hz in Session A (18–20 recorded frames per 3.0 s trial) and ~7.7 Hz in Session B (22–24 recorded frames per trial).
- A transient escape of 1–3 frames therefore corresponds to ~130–400 ms of real time outside the deadzone.
- 8 of the 9 center trials with transient escapes followed a peripheral fixation (highest following `LEFT`), indicating that the root cause of these brief excursions remains undetermined.

---

### D. Session B Aborted Runs & Mock Test Artifacts (2026-11-01)
During the test and calibration runs on 2026-11-01, additional report artifacts were generated and are archived in `artifacts/observations/`:
- **Aborted Physical Runs (`08b290`, `05ead8`)**:
  - `cmp_v16_20261101_163909_08b290_comparison_report.json`
  - `cmp_v16_20261101_164451_05ead8_comparison_report.json`
  - *Status*: Two physical execution attempts aborted via ESC/Q at 0 frames before any fixation data was recorded. Both are saved and marked `INCOMPLETE` with `total_frames_recorded: 0`.
- **Mock Test Artifacts (`084f58`, `088c48`, `01f914`, `04f1a0`)**:
  - `cmp_v16_20261101_163909_084f58_comparison_report.json`
  - `cmp_v16_20261101_163909_088c48_comparison_report.json`
  - `cmp_v16_20261101_164450_01f914_comparison_report.json`
  - `cmp_v16_20261101_164450_04f1a0_comparison_report.json`
  - *Status*: Automated synthetic mock reports generated during pytest suite verification (`test_controlled_comparison.py`). All are clearly marked `mock_mode: true` and labeled *"MOCK, not physical validation"*. They are software regression checks only and can never confer physical validation or adoption.

---

## 3. Detailed Trial-by-Trial Breakdown: Session B

### Peripheral Targets (12 Held-Out Trials)

| Trial ID | Target | MobileOne Median $(x, y)$ | MobileOne Hit | ResNet Median $(x, y)$ | ResNet Hit | Target Criteria |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **B3_2_DOWN** | `DOWN` | $(0.454, 0.612)$ | **False** (Miss) | $(0.460, 0.630)$ | **False** (Miss) | $y \ge 0.65$ |
| **B3_4_LEFT** | `LEFT` | $(0.118, 0.428)$ | **True** (Hit) | $(0.132, 0.441)$ | **True** (Hit) | $x \le 0.35$ |
| **B3_6_UP** | `UP` | $(0.391, 0.263)$ | **True** (Hit) | $(0.399, 0.235)$ | **True** (Hit) | $y \le 0.35$ |
| **B3_8_RIGHT** | `RIGHT` | $(0.740, 0.419)$ | **True** (Hit) | $(0.740, 0.435)$ | **True** (Hit) | $x \ge 0.65$ |
| **B4_2_UP** | `UP` | $(0.395, 0.374)$ | **False** (Miss) | $(0.401, 0.380)$ | **False** (Miss) | $y \le 0.35$ |
| **B4_4_RIGHT** | `RIGHT` | $(0.775, 0.356)$ | **True** (Hit) | $(0.780, 0.399)$ | **True** (Hit) | $x \ge 0.65$ |
| **B4_6_DOWN** | `DOWN` | $(0.328, 0.980)$ | **True** (Hit) | $(0.350, 0.945)$ | **True** (Hit) | $y \ge 0.65$ |
| **B4_8_LEFT** | `LEFT` | $(0.124, 0.511)$ | **True** (Hit) | $(0.153, 0.522)$ | **True** (Hit) | $x \le 0.35$ |
| **B5_2_LEFT** | `LEFT` | $(0.125, 0.568)$ | **True** (Hit) | $(0.147, 0.562)$ | **True** (Hit) | $x \le 0.35$ |
| **B5_4_UP** | `UP` | $(0.404, 0.316)$ | **True** (Hit) | $(0.419, 0.356)$ | **False** (Miss) | $y \le 0.35$ |
| **B5_6_DOWN** | `DOWN` | $(0.334, 0.606)$ | **False** (Miss) | $(0.343, 0.635)$ | **False** (Miss) | $y \ge 0.65$ |
| **B5_8_RIGHT** | `RIGHT` | $(0.715, 0.362)$ | **True** (Hit) | $(0.737, 0.391)$ | **True** (Hit) | $x \ge 0.65$ |

### Directional Accuracy Analysis
1. **Horizontal Axis (`LEFT` & `RIGHT`)**:
   - `LEFT`: $3/3$ (100.0%)
   - `RIGHT`: $3/3$ (100.0%)
   - **Horizontal Total**: **6 / 6 (100.0%)** for both models. Across Sessions A and B combined, horizontal peripheral accuracy is **12 / 12 (100.0%)**.
2. **Vertical Axis (`UP` & `DOWN`)**:
   - `UP`: $2/3$ (66.7%) for MobileOne, $1/3$ (33.3%) for ResNet.
   - `DOWN`: $1/3$ (33.3%) for MobileOne, $1/3$ (33.3%) for ResNet.
   - **Vertical Total**: **3 / 6 (50.0%)** for MobileOne, **2 / 6 (33.3%)** for ResNet.
3. **Threshold Margin Proximity**:
   - The three missed trials for MobileOne were near-misses on the fixed deadzone boundary $[0.35, 0.65]$:
     - `B3_2_DOWN`: $y = 0.612$ (required $y \ge 0.65$, delta $-0.038$)
     - `B4_2_UP`: $y = 0.374$ (required $y \le 0.35$, delta $+0.024$)
     - `B5_6_DOWN`: $y = 0.606$ (required $y \ge 0.65$, delta $-0.044$)

---

## 4. ResNet-34 INT8 Quantization Benchmark (Negative Result)
*Report: `artifacts/observations/resnet34_int8_quantization_report_20261028_232549.json`*

| Variant | Mean Angle Error | Max Yaw Error | p50 Latency | Gate Status |
| :--- | :---: | :---: | :---: | :---: |
| **FP32 Baseline** | Baseline ($0.0^\circ$) | Baseline | 61.47 ms | Accurate, exceeds 30 ms budget |
| **Dynamic INT8** | $0.049^\circ$ | $0.23^\circ$ | 63.95 ms | **REJECTED** (No speedup, runtime overhead) |
| **Static INT8** | $15.01^\circ$ | $40.85^\circ$ | 21.90 ms | **REJECTED** (Catastrophic drift exceeding target boundaries) |

---

## 5. Summary of Findings & Next Steps

1. **Gate Verdict**: Multi-session acceptance gate evaluated across Session A and Session B returned **FAIL** due to vertical accuracy in Session B falling below $85\%$.
2. **Safety Stability**: Center false activation remained $0.0\%$ by median across both sessions ($0/30$ center trials falsely triggered an action).
3. **Relative Model Comparison**: MobileOne S0 remains strictly superior to ResNet-34 ($75.0\%$ vs $66.7\%$ in Session B; $100\%$ vs $91.7\%$ in Session A; $22.9\text{ ms}$ vs $111.5\text{ ms}$ latency).
4. **Current Status**:
   - Live speech selection remains disabled (`is_selection_enabled = False`).
   - Milestone M1 remains open pending vertical calibration margin refinement.
