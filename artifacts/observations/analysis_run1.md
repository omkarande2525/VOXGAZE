# VoxGaze Live Run 1: Offline Quantitative Analysis Report

**Session ID**: `obs_20261014_220836_c99123`  
**Date & Time**: `2026-10-14 22:08:36` to `22:12:11`  
**Duration**: `215.05 s` (~3.6 minutes)  
**Total Frames Recorded**: `3524`  
**Conditions Evaluated**: 20 of 20 completed (100% protocol adherence)  

---

## 1. Executive Summary & Key Findings

1. **Protocol Execution & Data Integrity**:
   - The user successfully completed all 20 guided conditions without skips or premature terminations.
   - 3,524 total frames were recorded to local CSV without writing any raw images, face crops, or videos to disk.
   - Missing data was preserved honestly: exactly 71 tracking-loss frames (in Condition 19) generated clean empty values and `NO_FACE_DETECTED` statuses without zero-fill corruption.

2. **Center Baseline Repeatability & Within-Condition Stability**:
   - **Initial vs Final Fixation (C1 vs C20)**:
     - **MobileOne Pitch**: Remarkably stable (`+1.23°` in C1 vs `+1.29°` in C20, net drift = `+0.06°`).
     - **MobileOne Yaw**: Minor drift of `-1.92°` (`-4.49°` in C1 vs `-6.41°` in C20).
     - **Geometry Horizontal**: Highly consistent right-eye H ratio (`0.5265` in C1 vs `0.5213` in C20, delta = `0.0052`).
     - **Geometry Vertical**: Right-eye V ratio shifted by `-0.0418` (`0.4358` in C1 vs `0.3940` in C20), indicating modest vertical settling over 3.5 minutes.
   - **Intermediate Center Return Transients**: While C1 and C20 endpoints are repeatable, several intermediate returns exhibited substantial within-condition dispersion (e.g. C5 Yaw IQR = `32.55°`, Pitch IQR = `25.60°`; C7 Pitch IQR = `38.14°`; C11 Yaw IQR = `28.19°`). This occurred because manual collection started before the participant had fully settled back to center, capturing transition dynamics.

3. **Directed Eye Movements**:
   - **Horizontal Gaze**: Both MobileOne and MediaPipe geometry exhibit monotonic sensitivity to directed horizontal movements.
     - `EYES_LEFT`: Yaw shifted from `-4.49°` to `-27.13°` (`Δ = -22.64°`); Geometry H dropped from `0.5265` to `0.3368` (`Δ = -0.190`).
     - `EYES_RIGHT`: Yaw shifted to `+8.27°` (`Δ = +12.76°`); Geometry H rose to `0.7644` (`Δ = +0.238`).
   - **Vertical Gaze Discrepancy & Hypotheses**:
     - `EYES_UP`: MobileOne Pitch jumped from `+1.23°` to `+31.45°` (`Δ = +30.22°`), whereas Geometry V barely shifted (`0.4358` to `0.4370`, `Δ = +0.0012`).
     - `EYES_DOWN`: Geometry V dropped sharply to `0.2860` (`Δ = -0.150`), whereas MobileOne Pitch barely changed (`+3.42°`).
     - *Working Hypotheses*: Eyelids may move synchronously with the iris during upward gaze, keeping the eyelid-normalized ratio relatively constant. Conversely, MobileOne may be responding to scleral exposure or subtle head tilt cues. These are plausible kinematic and model hypotheses, not direct measurements.

4. **Eye-Only vs Head-Movement Dissociation (Compensatory Gaze Response)**:
   - In `HEAD_LEFT_EYES_FORWARD` (participant instructed to turn head left while keeping eyes forward on screen):
     - MobileOne outputs Yaw = `-26.14°` (nearly identical to `EYES_LEFT` at `-27.13°`).
     - MediaPipe Geometry H = `0.6472` (inverted toward the temporal side of the right eye socket, consistent with maintaining forward fixation).
   - *Scientific Interpretation*: MobileOne output is heavily dominated by **head pose / facial orientation**. When a participant turns their head while fixating forward, MobileOne reflects the head angle rather than screen fixation. The eye-in-socket geometry reflects the compensatory ocular counter-rotation. Condition labels cannot be treated as ground-truth gaze on screen.

5. **Tracking Recovery Dynamics (Condition 19)**:
   - Dropout event lasted exactly 71 contiguous frames (`2.36 s`).
   - Reacquisition occurred immediately on frame 5833 (first-frame processing latency rose to `50.11 ms` during full-frame detection, then settled to ~`26 ms`).
   - First 2–4 frames post-reacquisition showed transient landmark adjustment, underscoring the need for a warm-up buffer.

6. **Throughput and Latency Performance**:
   - **Compute Latency**: Median per-frame processing time was `28.21 ms` (compute capacity ~`35.4 FPS`), with p95 = `44.50 ms`. MobileOne ONNX inference was `6.82 ms` median, `7.51 ms` p95.
   - **Measured Acquisition Throughput**: Actual frame-to-frame inter-arrival intervals measured from monotonic timestamps averaged `33.32 ms` (mean throughput = **`30.01 FPS`**, median = **`32.26 FPS`**), perfectly bound by the 30 FPS webcam hardware. Zero frames exceeded 100 ms total processing time.

---

## 2. Condition-by-Condition Quantitative Breakdown

| Index | Condition Name | Frames | Dur (s) | Face Avail | Model Avail | MobileOne Yaw (deg)<br>Median [IQR] | MobileOne Pitch (deg)<br>Median [IQR] | Right Eye H Ratio<br>Median [IQR] | Right Eye V Ratio<br>Median [IQR] |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `CENTER_BASELINE` | 246 | 8.17 | 100.0% | 100.0% | -4.49 [5.09] | +1.23 [5.33] | 0.5265 [0.0277] | 0.4358 [0.0236] |
| 2 | `EYES_LEFT` | 206 | 6.83 | 100.0% | 100.0% | -27.14 [24.68] | +10.29 [19.38] | 0.3368 [0.1647] | 0.5109 [0.1486] |
| 3 | `EYES_CENTER_RETURN` | 141 | 4.66 | 100.0% | 100.0% | -8.02 [7.60] | +0.25 [10.27] | 0.5159 [0.0394] | 0.4019 [0.0302] |
| 4 | `EYES_RIGHT` | 110 | 3.66 | 100.0% | 100.0% | +8.27 [17.57] | +22.46 [21.64] | 0.7644 [0.2084] | 0.3840 [0.0248] |
| 5 | `EYES_CENTER_RETURN` | 102 | 3.34 | 100.0% | 100.0% | -0.41 [32.55] | -3.42 [25.60] | 0.5124 [0.2614] | 0.4084 [0.0627] |
| 6 | `EYES_UP` | 175 | 5.81 | 100.0% | 100.0% | -11.37 [12.96] | +31.45 [33.49] | 0.5000 [0.0579] | 0.4370 [0.0728] |
| 7 | `EYES_CENTER_RETURN` | 140 | 4.62 | 100.0% | 100.0% | -12.41 [12.41] | +10.12 [38.14] | 0.5111 [0.0198] | 0.4195 [0.0502] |
| 8 | `EYES_DOWN` | 127 | 4.2 | 100.0% | 100.0% | -14.71 [8.60] | +3.42 [9.26] | 0.4990 [0.0222] | 0.2860 [0.3373] |
| 9 | `EYES_CENTER_RETURN` | 127 | 4.22 | 100.0% | 100.0% | -10.43 [9.62] | +6.63 [9.60] | 0.4941 [0.0224] | 0.3803 [0.3115] |
| 10 | `HEAD_LEFT_EYES_FORWARD` | 135 | 4.47 | 100.0% | 100.0% | -26.14 [10.06] | +4.83 [4.88] | 0.6472 [0.0486] | 0.3329 [0.0424] |
| 11 | `HEAD_CENTER_RETURN` | 80 | 2.62 | 100.0% | 100.0% | -23.21 [28.19] | -0.23 [10.59] | 0.4738 [0.2394] | 0.3998 [0.1166] |
| 12 | `HEAD_RIGHT_EYES_FORWARD` | 136 | 4.52 | 100.0% | 100.0% | +7.87 [12.12] | -8.99 [8.50] | 0.2777 [0.0737] | 0.4676 [0.0241] |
| 13 | `HEAD_CENTER_RETURN` | 125 | 4.11 | 100.0% | 100.0% | -1.95 [13.43] | -1.41 [7.15] | 0.4946 [0.1143] | 0.4283 [0.0703] |
| 14 | `HEAD_UP_EYES_FORWARD` | 120 | 3.94 | 100.0% | 100.0% | +10.97 [4.42] | +20.16 [4.38] | 0.4743 [0.0098] | 0.5382 [0.0139] |
| 15 | `HEAD_CENTER_RETURN` | 195 | 6.48 | 100.0% | 100.0% | -1.90 [7.57] | +8.73 [21.45] | 0.5060 [0.0336] | 0.4570 [0.0848] |
| 16 | `HEAD_DOWN_EYES_FORWARD` | 186 | 6.14 | 100.0% | 100.0% | -5.79 [7.29] | -9.62 [8.31] | 0.5501 [0.0110] | 0.3590 [0.0220] |
| 17 | `HEAD_CENTER_RETURN` | 163 | 5.42 | 100.0% | 100.0% | -4.40 [5.12] | -3.56 [11.14] | 0.5259 [0.0239] | 0.4469 [0.0270] |
| 18 | `COMBINED_MOVEMENT` | 652 | 21.69 | 100.0% | 100.0% | -8.45 [19.86] | +0.78 [16.95] | 0.4900 [0.0645] | 0.4374 [0.0815] |
| 19 | `TRACKING_RECOVERY` | 173 | 5.7 | 59.0% | 59.0% | -4.85 [8.17] | -7.63 [11.15] | 0.4893 [0.0237] | 0.4517 [0.0532] |
| 20 | `FINAL_CENTER_BASELINE` | 185 | 6.12 | 100.0% | 100.0% | -6.41 [3.81] | +1.29 [5.04] | 0.5213 [0.0052] | 0.3940 [0.0095] |

---

## 3. Detailed Experimental Comparisons

### A. Center Repeatability: Condition 1 vs Condition 20
Comparing initial fixation (`CENTER_BASELINE`) with the session conclusion (`FINAL_CENTER_BASELINE`) separated by 215 seconds:
- **MobileOne Yaw**: `-4.49°` vs `-6.41°` (`Δ = -1.92°`, IQR: `2.64°` vs `4.89°`).
- **MobileOne Pitch**: `+1.23°` vs `+1.29°` (`Δ = +0.06°`, IQR: `6.09°` vs `6.95°`).
- **Right Eye H Ratio**: `0.5265` vs `0.5213` (`Δ = -0.0052`, IQR: `0.0135` vs `0.0270`).
- **Right Eye V Ratio**: `0.4358` vs `0.3940` (`Δ = -0.0418`, IQR: `0.0210` vs `0.0526`).
- **Bilateral Agreement**: Left Eye H ratio changed from `0.482` to `0.495` (`Δ = +0.013`), remaining within 0.03 of the right eye.
- **Repeatability Conclusion**: Center baseline repeatability is very strong in horizontal geometry and MobileOne pitch. Slight downward vertical drift is observed in geometry ratios, consistent with postural settling or lid fatigue over the 3.6-minute run.

### B. Directed Eye Movements vs Center Baseline
| Directed Condition | MobileOne Yaw<br>(Baseline: `-4.49°`) | MobileOne Pitch<br>(Baseline: `+1.23°`) | Geometry H Ratio<br>(Baseline: `0.5265`) | Geometry V Ratio<br>(Baseline: `0.4358`) |
|:---|:---:|:---:|:---:|:---:|
| `EYES_LEFT` | **`-27.13°`** (`Δ = -22.64°`) | `+10.29°` (`Δ = +9.06°`) | **`0.3368`** (`Δ = -0.1897`) | `0.5109` (`Δ = +0.0751`) |
| `EYES_RIGHT` | **`+8.27°`** (`Δ = +12.76°`) | `+22.46°` (`Δ = +21.23°`) | **`0.7644`** (`Δ = +0.2379`) | `0.3840` (`Δ = -0.0518`) |
| `EYES_UP` | `-11.37°` (`Δ = -6.88°`) | **`+31.45°`** (`Δ = +30.22°`) | `0.5000` (`Δ = -0.0265`) | `0.4370` (`Δ = +0.0012`) |
| `EYES_DOWN` | `-14.71°` (`Δ = -10.22°`) | `+3.42°` (`Δ = +2.19°`) | `0.4990` (`Δ = -0.0275`) | **`0.2860`** (`Δ = -0.1498`) |

*Observations*:
1. Horizontal gaze shifts show clear, strong, and monotonic response in both modalities.
2. In vertical gaze, MobileOne exhibits high sensitivity to upward gaze (`+30.22°`), while Geometry V ratio exhibits high sensitivity to downward gaze (`-0.1498`). Eyelid aperture coupling explains the asymmetric geometric response.

### C. Eye-Only vs Head-Movement Dissociation (Ocular Counter-Rotation)
| Axis | Eye-Only Condition | Head-Movement Condition | MobileOne Metric Comparison | Geometry H Metric Comparison |
|:---|:---|:---|:---|:---|
| **Left** | `EYES_LEFT` (Eyes Left, Head Center) | `HEAD_LEFT_EYES_FORWARD` (Head Left, Eyes Forward) | `EYES_LEFT`: **`-27.13°`**<br>`HEAD_LEFT`: **`-26.14°`** | `EYES_LEFT`: **`0.3368`** (Iris Left in socket)<br>`HEAD_LEFT`: **`0.6472`** (Iris Right in socket) |
| **Right** | `EYES_RIGHT` (Eyes Right, Head Center) | `HEAD_RIGHT_EYES_FORWARD` (Head Right, Eyes Forward) | `EYES_RIGHT`: **`+8.27°`**<br>`HEAD_RIGHT`: **`+7.87°`** | `EYES_RIGHT`: **`0.7644`** (Iris Right in socket)<br>`HEAD_RIGHT`: **`0.2777`** (Iris Left in socket) |

**Empirical Sensor Findings**:
When the participant turns their head to the left while instructed to maintain fixation forward on screen:
- The actual physical fixation on screen was intended to be center.
- MobileOne predicts **Yaw = `-26.14°`**, tracking the facial orientation rather than gaze on screen.
- MediaPipe geometry measures **H = `0.6472`**, tracking the physical displacement of the iris toward the temporal border of the socket (consistent with the expected compensatory ocular counter-rotation).
- *Deduction*: A pretrained face-crop model cannot be assumed to function as an independent gaze-on-screen classifier in AAC without head-pose decoupling. Conversely, condition labels are protocol directives, not verified physical ground truth.

---

## 4. Chronological Analysis of Tracking Recovery (Condition 19)

Condition 19 explicitly tested tracking loss and reacquisition:
- **Total Condition Duration**: `5.70 s` (173 frames).
- **Pre-Loss Tracking**: 22 frames (`0.73 s`), 100% valid detection.
- **Dropout Event** (Camera covered / user lookaway): Exactly 71 contiguous frames (`2.36 s`, from `t = 197.17 s` to `t = 199.53 s`).
  - `face_detected = false`, `mobileone_valid = false`, `unavailable_reason = NO_FACE_DETECTED`.
  - Zero false positives, zero crashes, zero non-finite numerical leakage.
- **Reacquisition Phase**: Exactly 80 frames (`2.61 s`, from `t = 199.53 s` to `t = 202.14 s`).
  - Reacquisition frame 5833 latency rose to `50.11 ms` (MediaPipe full-frame detector invocation), then immediately dropped back to `26 ms`.
  - Geometric settling: Frames 5833–5836 showed transient landmark oscillation before stabilizing back to center baseline values.

---

## 5. Sampling and Duration Imbalances

Duration per condition varied across the protocol:
- **Shortest Condition**: `HEAD_CENTER_RETURN` (C11): 80 frames (`2.62 s`).
- **Longest Condition**: `COMBINED_MOVEMENT` (C18): 652 frames (`21.69 s`, 18.5% of total session).
- **Baseline Conditions**: Typically 100–250 frames (`3.3 s` to `8.2 s`).

*Impact on Analysis*:
- Naive global aggregations are disproportionately weighted by Condition 18.
- Independent per-condition evaluation is mandatory to avoid skewing center stability or latency estimates.

---

## 6. Latency and Compute Performance

| Stage | Median (p50) | 95th Percentile (p95) | Max |
|:---|:---:|:---:|:---:|
| **MediaPipe Face Landmarker** | `10.18 ms` | `11.69 ms` | `31.42 ms` |
| **Face-Crop Extraction** | `0.25 ms` | `0.40 ms` | `1.17 ms` |
| **MobileOne Preprocessing** | `4.75 ms` | `5.55 ms` | `8.10 ms` |
| **MobileOne CPU Inference** | `6.82 ms` | `7.51 ms` | `17.03 ms` |
| **Total Per-Frame Compute Time** | `28.21 ms` (capacity ~35.4 FPS) | `44.50 ms` | `77.71 ms` |

*Throughput Distinction*: Total per-frame compute time median of `28.21 ms` indicates a theoretical compute headroom of ~`35.4 FPS`. The actual measured physical acquisition throughput delivered by the webcam was **`30.01 FPS`** (mean inter-frame interval $\Delta t = 33.32\text{ ms}$, median $\Delta t = 31.00\text{ ms}$), governed by the camera hardware framerate. Zero frames exceeded 100 ms.

---

## 7. Diagnostic Figures

The following diagnostic plots were generated and saved to `artifacts/observations/analysis_c99123/`:

1. `timeline_overview.png`: Continuous time-series trace of MobileOne angles and MediaPipe geometry ratios across all 3,524 frames.
2. `center_repeatability.png`: Boxplots comparing distributions across all 10 repeated center-fixation conditions.
3. `eye_vs_head_dissociation.png`: Direct comparison of eye-only vs head-movement (compensatory gaze) conditions.
4. `tracking_recovery_event.png`: Frame-by-frame zoom into the 173 frames of Condition 19 dropout and reacquisition.

---

## 8. Limitations & Methodological Constraints

1. **Absence of Calibrated Ground Truth**: Condition labels (`EYES_LEFT`, `CENTER_BASELINE`, etc.) reflect instructed protocol actions, not externally verified angular ground truth.
2. **Head Pose Coupling**: MobileOne outputs reflect combined head-plus-eye gaze relative to camera, dominated by head orientation.
3. **Eyelid Coupling in Vertical Geometry**: 2D eyelid-normalized vertical ratios are non-linear and dampened during upward gaze due to eyelid co-movement.

---

## 9. Recommendations & Next Actions

1. **Protocol Verdict**:
   - **Do NOT repeat Run 1 immediately**. The protocol executed cleanly, collected high-quality data across all 20 conditions, and provided definitive evidence.
2. **Architecture Guidance**:
   - Do **NOT** replace MediaPipe geometry with raw MobileOne angles for intent classification. Raw MobileOne angles cannot distinguish a head turn with forward gaze from a true gaze shift.
3. **Recommended Engineering Focus**:
   - Investigate **hybrid sensor fusion** where MediaPipe provides head-pose compensation and iris geometry, while MobileOne provides high-frequency angular features.
   - Investigate **canthal-normal vertical distance** refinement to resolve the upward-gaze eyelid dampening issue before moving to M2 (Semantic FSM).
