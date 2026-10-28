# VoxGaze: Evaluator Integrity & Adoption Decision Audit Report

**Date**: 2026-10-28  
**Audit Scope**: Milestone M1 / M1.5 Gaze Estimation & Model Adoption Protocol  
**Audit Purpose**: Correct evaluator metrics, enforce non-negotiable adoption criteria, eliminate misleading peripheral-only accuracy naming, prevent premature adoption decisions, and re-evaluate existing observation reports.  
**Auditor**: Antigravity Pair Programming (Milestone M1 Sensor Validation)  

---

## 1. Executive Summary & Root Cause Analysis

An audit of `scripts/run_controlled_model_comparison.py` and `backend/app/calibration/personalized_mapper.py` revealed four critical scientific and safety flaws in how evaluation metrics and adoption decisions were previously computed:

1. **Misleading Accuracy Naming**: `trial_selection_accuracy` was computed solely over peripheral trials ($N=4$), omitting center fixation trials entirely. This created the false impression that the system achieved 75% or 100% total command accuracy, when in fact center resting behavior was not included in that metric.
2. **Absence of Overall Trial Correctness**: There was no combined trial-level correctness metric penalizing center drift. A system could trigger unintended activations on every center rest and still report "100% selection accuracy."
3. **Flawed Adoption Decision Rule**: The previous comparison harness adopted ResNet-34 based solely on a single heuristic condition:
   $$\Delta \text{Accuracy} = \text{ResNet34}_{\text{peripheral}} - \text{MobileOne}_{\text{peripheral}} \ge 15.0\text{ percentage points}$$
   This rule allowed ResNet-34 to be marked as `adopted: True` on a single session with only **4 peripheral trials**, despite suffering a **40% center false-activation rate** (2 out of 5 center trials drifted into active buttons) and a prohibitive CPU latency of **142.4 ms (~7 FPS)**.
4. **Vulnerability to Missing Fields & Mock Conflation**: Missing latency dictionaries or NaN outputs could trigger uncaught exceptions (`KeyError: 'end_to_end_latency_ms'`), and synthetic mock test runs lacked visible non-validation labeling.

---

## 2. Updated Non-Negotiable Adoption Criteria

Under the updated evaluator, neither model can be adopted for live word selection or declared superior without satisfying all of the following non-negotiable criteria:

| Criterion Gate | Non-Negotiable Standard | Required Minimum Sample | Justification |
| :--- | :--- | :--- | :--- |
| **1. Peripheral Target Accuracy** | **$\ge 85.0\%$** | $\ge 12$ held-out trials across 4 directions | Ensures intentional selection reliability against AAC button targets. |
| **2. Center False Activation Rate** | **$\le 5.0\%$** | $\ge 15$ held-out center return trials | Prevents resting eyes from triggering accidental words (Safety Rule #1). |
| **3. Overall Trial Correctness** | **$\ge 85.0\%$** | $\ge 27$ total held-out trials (peripheral + center) | Center trials are correct **only** if gaze remains inside center deadzone. |
| **4. Multi-Session Generalization** | **Passing across $\ge 2$ sessions** | 2 independent seated sessions on separate days | Proves calibration portability beyond a single seated position or camera angle. |
| **5. CPU Latency Gate** | **$p50 \le 40\text{ ms}$ ($\ge 25\text{ FPS}$)** | Held-out video frames | Ensures smooth visual dwell cursor tracking without UI stutter. |
| **6. Non-Mock Verification** | **Physical acquisition only** | Real webcam photons | Mock/synthetic data is prominently labeled `'MOCK, not physical validation'`. |

> [!IMPORTANT]
> **Preliminary Sample Rule**: Any evaluation run with fewer than 12 peripheral trials or 15 center trials is automatically labeled **`PRELIMINARY`** and cannot confer adoption, regardless of measured percentages.

---

## 3. Metric Definitions & Disambiguation

All metrics now provide explicit counts, denominators, and separate classifications:

- **`peripheral_trial_accuracy`**: $\frac{\text{Peripheral Trials Hit}}{\text{Peripheral Trials Total}}$ (Tested against peripheral AAC bounding boxes).
- **`center_return_false_activation_rate`**: $\frac{\text{Center Trials Escaped Deadzone}}{\text{Center Trials Total}}$ (Evaluates resting deadzone $[0.35, 0.35, 0.65, 0.65]$).
- **`overall_trial_correctness_rate`**: $\frac{\text{Peripheral Hits} + \text{Center Stays}}{\text{Peripheral Trials Total} + \text{Center Trials Total}}$ (True end-to-end command correctness).
- **`peripheral_dwell_accuracy`**: $\frac{\text{Peripheral Trials with Continuous Dwell } \ge 400\text{ ms}}{\text{Peripheral Trials Total}}$.

---

## 4. Retrospective Re-Evaluation of Existing JSON Reports

Without altering or overwriting any historical observation file, the updated evaluator re-assessed the three primary physical calibration reports and the physical comparison report:

### Report 1: `calib_v15_0a6b30_20261025_130951_1ef58c_summary.json`
- **Context**: Natural laptop posture, 3,228 total frames, 27 conditions.
- **Handcrafted Baseline (Phase 2)**:
  - Peripheral Trials: 2 evaluated (`UP`, `DOWN`), 2 passed $\to$ **100.0% (2/2)** (`LEFT` and `RIGHT` omitted due to threshold saturation).
  - Center Return Trials: 5 trials, 1 false activation (`EVAL_CENTER_4` triggered `UP`) $\to$ **20.0% (1/5) false activation rate** (Fails $\le 5\%$ gate).
  - Overall Correctness: **7/9 (77.8%)**.
- **Personalized Hybrid Mapper (Held-Out)**:
  - Peripheral Accuracy: **75.0% (3/4)** (`LEFT`, `RIGHT`, `DOWN` passed; `UP` missed due to eyelid occlusion).
  - Center False Activation Rate: **0.0% (0/5)** (5/5 stayed inside $[0.35, 0.35, 0.65, 0.65]$).
  - Overall Trial Correctness: **8/9 (88.9%)**.
- **Adoption Status**: **REJECTED / PRELIMINARY** (Sample size $4/12$ peripheral, $5/15$ center; single session).

---

### Report 2: `calib_v15_0a3228_20261022_105540_7b0e80_summary.json`
- **Context**: Stable posture, controlled lighting, 3,221 total frames.
- **Handcrafted Baseline (Phase 2)**:
  - Peripheral Trials: 4 evaluated, 4 passed $\to$ **100.0% (4/4)**.
  - Center Return Trials: 5 trials, 4 false activations (`EVAL_CENTER_2..5` drifted into `DOWN`) $\to$ **80.0% (4/5) false activation rate** (Severe safety violation).
  - Overall Correctness: **5/9 (55.6%)**.
- **Personalized Hybrid Mapper (Held-Out)**:
  - Peripheral Accuracy: **75.0% (3/4)**.
  - Center False Activation Rate: **0.0% (0/5)** (Sustained trial-median remained centered).
  - Transient Frame-Level Center False Rate: **24.8% (231/931 frames)** (Flutter on resting gaze).
  - Overall Trial Correctness: **8/9 (88.9%)**.
- **Adoption Status**: **REJECTED / PRELIMINARY** (Sample size $4/12$ peripheral, $5/15$ center; baseline fails center safety; single session).

---

### Report 3: `calib_v15_0164a4_20261022_091508_cd5344_summary.json`
- **Context**: Natural posture with progressive head drift, 3,228 total frames.
- **Handcrafted Baseline (Phase 2)**:
  - Peripheral Trials: 4 trials, 0 passed $\to$ **0.0% (0/4)** (System correctly entered `ABSTAIN`).
  - Center Return Trials: 5 trials, 0 false activations $\to$ **0.0% (0/5)**.
  - Overall Correctness: **5/9 (55.6%)**.
- **Personalized Hybrid Mapper (Held-Out)**:
  - Peripheral Accuracy: **0.0% (0/4)** (All mappers safely collapsed to center on drift).
  - Center False Activation Rate: **0.0% (0/5)**.
  - Overall Trial Correctness: **5/9 (55.6%)**.
- **Adoption Status**: **REJECTED** (0% peripheral selection accuracy; correctly flagged as uncalibrated).

---

### Comparative Evaluation: `cmp_v16_20261025_164239_0d9a6c_comparison_report.json`

| Metric / Requirement | Old Naive Decision | MobileOne S0 (Hybrid) | ResNet-34 (Hybrid) | Corrected Gate Result |
| :--- | :--- | :--- | :--- | :--- |
| **Peripheral Trial Accuracy** | 75% vs 100% | **75.0% (3/4)** | **100.0% (4/4)** | **PRELIMINARY** ($4/12$ trials) |
| **Center False Activation Rate** | *Not checked* | **40.0% (2/5)** | **40.0% (2/5)** | **FAIL** ($40\% > 5\%$ gate) |
| **Overall Trial Correctness** | *Not computed* | **66.7% (6/9)** | **77.8% (7/9)** | **FAIL** ($< 85\%$ threshold) |
| **End-to-End Latency (p50)** | *Informational only* | **29.78 ms (~33.6 FPS)** | **142.41 ms (~7.0 FPS)** | **FAIL on ResNet-34** (CPU bottleneck) |
| **Multi-Session Generalization** | *Not checked* | 1 of 2 sessions | 1 of 2 sessions | **FAIL** (1 session only) |
| **Final Adoption Decision** | `adopted: true` (ResNet-34) | **REJECTED** | **REJECTED** | **REJECTED (Both models fail safety gates)** |

---

## 5. Summary of Evaluator Improvements & Regression Verification

1. **Safety Enforced**: ResNet-34's premature adoption in Session `0d9a6c` has been explicitly overturned. Its 40% center false-activation rate and 142 ms latency violate assistive safety invariants.
2. **KeyError Eliminated**: Missing latency dictionaries, None values, or empty trial arrays return explicit `INVALID_MISSING_LATENCY` or `INSUFFICIENT_DATA` statuses without throwing exceptions.
3. **Mock Protection**: Mock runs cannot confer adoption and must carry visible `'MOCK, not physical validation'` labeling.
4. **Automated Regression Suite**: 5 new regression tests verify:
   - Rejection of 100% peripheral accuracy when center false activations exceed 5%.
   - Graceful handling of missing or NaN latency fields.
   - Rejection of incomplete trials, empty records, and NaN metric inputs.
   - Enforcement of non-mock physical acquisition validation labels.
   - Strict requirement for all criteria (sample size, accuracy, false activations, sessions, latency) before adoption.
5. **Full Test Suite Status**: **156 passed in 10.67s**, zero failures.
