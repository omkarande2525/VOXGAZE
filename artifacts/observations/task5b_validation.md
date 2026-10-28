# VoxGaze Task 5B — Validation of Run 1 Analysis & Run 2 Preparation

**Audit Date**: 2026-10-14  
**Evaluated Artifacts**:
- `artifacts/observations/obs_20261014_220836_c99123.csv` (3,524 raw frame records)
- `artifacts/observations/obs_20261014_220836_c99123_summary.json`
- `artifacts/observations/analysis_run1.md`
- `scripts/analyze_observation_session.py`
- `backend/app/perception/geometry.py` (`compute_canthal_normal_distance_px` and `norm_vertical_gaze_canthal_normal_experimental`)

---

## Part A: Audit of Run 1 Analysis

### 1. Verification of Numerical Statistics Against Raw CSV
Every reported numerical metric in `analysis_run1.md` was cross-checked against the raw records in `obs_20261014_220836_c99123.csv`:

| Check Category | Verification Result | Audit Findings & Clarifications |
|:---|:---:|:---|
| **Total Frames** | **VERIFIED** | Exactly 3,524 frame rows in CSV (header + 3,524 data lines). |
| **Protocol Completeness** | **VERIFIED** | All 20 conditions present in chronological order (Conditions 1 to 20). |
| **Tracking Loss Count** | **VERIFIED** | Exactly 71 frames with `face_detected = false` (all contiguous within Condition 19). |
| **Missing Value Integrity** | **VERIFIED** | Missing fields rendered as empty strings `""` without zero-fill corruption. |
| **Model CPU Inference Latency** | **VERIFIED** | Median = `6.82 ms`, 95th percentile = `7.51 ms`, Max = `17.03 ms`. |
| **Compute Frame Latency** | **VERIFIED** | Median = `28.21 ms`, 95th percentile = `44.50 ms`, Max = `77.71 ms`. |
| **Condition Endpoints (C1 vs C20)** | **VERIFIED** | C1 Yaw `-4.49°`, Pitch `+1.23°`, H `0.5265`, V `0.4358`. C20 Yaw `-6.41°`, Pitch `+1.29°`, H `0.5213`, V `0.3940`. |

---

### 2. Explicit Audit of Specific Analytical Claims

#### Claim 1: Whether eye-only and head-movement conditions establish independent ground truth
- **Audit Finding**: **OVERSTATED IN RUN 1 REPORT (Corrected).**
- **Evidence**: Condition labels (`EYES_LEFT`, `HEAD_LEFT_EYES_FORWARD`, etc.) represent **user-directed behavioral instructions**, not verified optoelectronic, electromagnetic, or target-calibrated ground truth.
- **Scientific Reality**: The participant was not tracked by an external dual-Purkinje tracker, scleral coil, or head-fixed bite bar. The participant may exhibit partial head movement during eye-only tasks or slight fixation drift during head-rotation tasks.
- **Correction Applied**: Condition labels must strictly be described as "instructed protocol conditions," never as "ground-truth gaze angles."

#### Claim 2: Whether the data prove a vestibulo-ocular reflex (VOR)
- **Audit Finding**: **OVERSTATED IN RUN 1 REPORT (Corrected).**
- **Evidence**: The Run 1 report claimed "The VOR Evidence: ... proves condition labels cannot be treated as ground-truth gaze."
- **Scientific Reality**: The vestibulo-ocular reflex (VOR) is a biological, neuro-otological reflex driven by vestibular canal input. The webcam system does not measure semicircular canal activation or calibrated 3D eye-in-head angular velocity vs head velocity ($G_{VOR} \approx -1.0$).
- **Observed Empirical Fact**: What the data actually prove is a **divergence between sensor representations**:
  1. MobileOne predicts an angle correlated with the cropped facial orientation (Yaw = `-26.14°`).
  2. MediaPipe iris landmarks displace toward the temporal margin of the orbit inside the crop (H = `0.6472`).
- **Correction Applied**: Revised Section 1.4 and Section 3.C of `analysis_run1.md` to describe the finding as "Empirical Dissociation Between Facial Orientation and Ocular Counter-Rotation."

#### Claim 3: Whether the reported approximately 35 FPS is measured throughput or an inference from latency
- **Audit Finding**: **NUMERICAL DISCREPANCY & CONFLATION (Corrected).**
- **Evidence**:
  - Processing compute latency: Median = `28.21 ms`. Calculating $1000.0 / 28.21 = 35.45\text{ FPS}$ gives the **theoretical compute throughput capacity**.
  - Actual measured frame arrival intervals ($\Delta t = t_{i} - t_{i-1}$) across all 3,524 recorded frames:
    - Median $\Delta t$: **`31.00 ms`** (`32.26 FPS`)
    - Mean $\Delta t$: **`33.32 ms`** (**`30.01 FPS`**)
    - 5th percentile $\Delta t$: `16.00 ms`
    - 95th percentile $\Delta t$: `47.00 ms`
- **Scientific Reality**: The physical acquisition rate was dictated and capped by the webcam hardware running at **30 FPS**. A pipeline that computes in 28 ms still only receives 30 frames per second from a 30 FPS camera.
- **Correction Applied**: Corrected `analysis_run1.md` to explicitly state:
  *"Total per-frame compute time median of 28.21 ms indicates compute capacity for ~35.4 FPS. The actual physical acquisition throughput was 30.01 FPS (mean inter-frame interval 33.32 ms), governed by the 30 FPS camera hardware."*

#### Claim 4: Stability conclusions vs duration imbalances and high within-condition IQRs
- **Audit Finding**: **PARTIAL REPORTING IN RUN 1 REPORT (Corrected).**
- **Evidence**:
  - Condition durations were highly unbalanced, ranging from **`2.62 s`** (`HEAD_CENTER_RETURN`, 80 frames) to **`21.69 s`** (`COMBINED_MOVEMENT`, 652 frames).
  - Several intermediate center-return conditions exhibited massive within-condition dispersion:
    - Condition 5 (`EYES_CENTER_RETURN`): MobileOne Yaw IQR was **`32.55°`**, Pitch IQR was **`25.60°`**, Geometry H IQR was **`0.2614`**.
    - Condition 7 (`EYES_CENTER_RETURN`): MobileOne Pitch IQR was **`38.14°`**.
    - Condition 11 (`HEAD_CENTER_RETURN`): Median Yaw was **`-23.21°`** with an IQR of **`28.19°`**.
- **Root Cause Identified**: The protocol allowed the participant to press `[SPACE]` to start recording immediately. When returning from an extreme gaze or head turn, the participant pressed Space **while their head and eyes were still traveling back to center**. As a result, dynamic transition trajectories were logged inside the condition, inflating IQRs.
- **Correction Applied**: Updated `analysis_run1.md` to document intermediate return transients and duration imbalances alongside endpoint repeatability.

#### Claim 5: Distinguishing observed evidence from mechanistic hypotheses
- **Audit Finding**: **CONFLATION IN RUN 1 REPORT (Corrected).**
- **Evidence**: The report labeled eyelid synchronous motion as the "Root Cause" of vertical geometry dampening and claimed MobileOne was detecting scleral exposure.
- **Scientific Reality**:
  - *Observed Evidence*: Condition 6 (`EYES_UP`) produced a `+30.22°` shift in MobileOne Pitch, but only a `+0.0012` shift in Geometry V ratio.
  - *Unverified Hypotheses*: Eyelid co-movement is a biomechanical hypothesis; MobileOne feature attribution to sclera is an unverified deep-learning interpretability hypothesis (no gradient attribution or feature saliency maps were run).
- **Correction Applied**: Re-labeled as "Working Hypotheses" in `analysis_run1.md`, distinguishing raw metric shifts from biomechanical or model-attribution explanations.

---

## Part B: Evaluation of Vertical Geometry

### 1. Implementation Inspection
Two functions exist in [`backend/app/perception/geometry.py`](file:///d:/VOXGAZE/backend/app/perception/geometry.py):

1. `compute_canthal_normal_distance_px(iris_pt, inner_canthus, outer_canthus, is_right_eye)`:
   - Computes signed perpendicular distance from the canthal axis to the iris center in **raw camera pixels**:
     $$\text{dist\_px} = \frac{(p_x - s_x) d_y - (p_y - s_y) d_x}{\text{eye\_width}}$$
   - **Units**: Pixels ($px$).
2. `norm_vertical_gaze_canthal_normal_experimental(iris_pt, inner_canthus, outer_canthus, is_right_eye)`:
   - Normalizes the perpendicular distance by eye width:
     $$\text{ratio} = \frac{(p_x - s_x) d_y - (p_y - s_y) d_x}{\text{eye\_width}^2} = \frac{\text{dist\_px}}{\text{eye\_width}}$$
   - **Units**: Dimensionless normalized ratio.

### 2. What Was Actually Recorded in Run 1?
In [`scripts/mobileone_live_diagnostic.py`](file:///d:/VOXGAZE/scripts/mobileone_live_diagnostic.py#L377):
```python
canthal_dist_px = compute_canthal_normal_distance_px(r_iris, r_inner, r_outer, is_right_eye=True)
```
- **Recorded Metric**: Raw pixel distance (`canthal_normal_dist_px`) was logged.
- **Eye Coverage**: Recorded for the **anatomical right eye only**. The left eye was not computed or recorded.
- **Normalized Ratio**: `norm_vertical_gaze_canthal_normal_experimental` was **NOT** recorded in Run 1.

### 3. Are the Run 1 Data Sufficient to Evaluate It?
- **Directional Trend Observed in Run 1**:
  - `CENTER_BASELINE` (C1): Median = `-2.52 px` (IQR = `0.33 px`).
  - `EYES_UP` (C6): Median = `-4.25 px` (IQR = `2.73 px`, iris moved $1.73\text{ px}$ upward relative to canthal axis).
  - `EYES_DOWN` (C8): Median = `-0.48 px` (IQR = `2.79 px`, iris moved $2.04\text{ px}$ downward relative to canthal axis).
  - `FINAL_CENTER_BASELINE` (C20): Median = `-2.41 px` (returns to within $0.11\text{ px}$ of C1).
- **Critical Limitations Found**:
  1. **Unnormalized Pixel Scale**: Because the metric is in raw pixels, it scales directly with the participant's distance to the webcam. If a user sits closer ($50\text{ cm}$ vs $70\text{ cm}$), the eye width in pixels expands from $\sim 35\text{ px}$ to $\sim 50\text{ px}$, distorting the pixel distance proportionally.
  2. **Head Pitch Foreshortening**: When the participant tilted their head up/down in C14 and C16, perspective foreshortening changed the 2D projected canthal distance (C14 = `-2.05 px`, C16 = `-3.36 px`).
  3. **Bilateral Gap**: The lack of left-eye canthal normal recording prevents bilateral consistency checks.
- **Conclusion**: The Run 1 data demonstrate that the canthal perpendicular axis moves monotonically with vertical eye movement, but **raw pixel distance is unsuitable as a general feature**. It must be normalized by inter-canthal eye width ($\text{dist\_px} / \text{eye\_width}$) and logged bilaterally before it can be validated for AAC intent gating.

---

## Part C: Recommended Run 2 Protocol (Fair Repeat Protocol)

To address the limitations discovered in Run 1, the following standardized protocol is recommended for Run 2:

### 1. Fixed Collection Duration & Sample Target
- **Duration**: Exactly **`5.0 seconds`** (150 frames at 30 FPS) per condition.
- **Rationale**: Eliminates duration imbalances (e.g. C18 taking 21.7s while C11 took 2.6s) and provides equal sample sizes for statistical comparison across all conditions.

### 2. Mandatory Automated Settling Interval
- **Settling Duration**: **`2.0 seconds`** (60 frames) pre-collection settling phase before logging begins for every condition.
- **Implementation**:
  - When a condition starts, the HUD displays: `[SETTLING: 2.0s - PLEASE HOLD POSITION]`.
  - Frames during settling are processed by the display but **NOT written to CSV**.
  - Logging activates automatically after 2.0 seconds.
- **Rationale**: Eliminates dynamic saccadic and head-rotation recovery transients from the data, directly solving the high IQRs observed in C5, C7, and C11.

### 3. Explicit On-Screen Visual Fixation Anchors
- **Visual Anchors**: Display a clean high-contrast crosshair on the HUD:
  - Center conditions: Crosshair at center of screen.
  - `EYES_LEFT` / `EYES_RIGHT`: Visual anchor on left / right margin.
  - `EYES_UP` / `EYES_DOWN`: Visual anchor on top / bottom margin.
  - `HEAD_LEFT_EYES_FORWARD`: Head turned left, crosshair stays at center.
- **Rationale**: Replaces subjective participant interpretation with repeatable visual fixation targets.

### 4. Bilateral Normalized Canthal Recording
- Extend the observation schema in `scripts/mobileone_live_diagnostic.py` to record:
  - `right_eye_canthal_ratio`: $\text{dist\_px} / \text{eye\_width}$ (dimensionless)
  - `left_eye_canthal_ratio`: $\text{dist\_px} / \text{eye\_width}$ (dimensionless)
  - `right_eye_canthal_px`: raw pixel distance
- **Rationale**: Allows direct evaluation of scale-invariant vertical geometry.

### 5. Standardized Tracking Recovery Procedure
- Standardize Condition 19 (`TRACKING_RECOVERY`):
  - 2.0s forward center baseline.
  - 2.0s physical camera occlusion (cover with card or hand).
  - 3.0s reacquisition at center.
- **Rationale**: Ensures identical dropout and reacquisition timelines across repeat sessions.

### 6. Participant Comfort Invariant
- Explicitly instruct the participant: *"Rotate head or eyes only to a natural, comfortable extent ($\approx 15^\circ - 20^\circ$). Do not strain neck or eyes."*

---

## Summary of Completed Audits & Corrections

1. **`artifacts/observations/analysis_run1.md` Updated**:
   - Corrected 35.4 FPS compute capacity vs 30.01 FPS measured camera throughput.
   - Reframed "VOR proof" into empirical sensor dissociation (head-pose dominance vs ocular counter-rotation).
   - Clarified that condition labels are protocol instructions, not physical ground truth.
   - Added documentation of intermediate center return transients and condition duration imbalances.
   - Separated observed vertical metric deltas from mechanistic hypotheses.
2. **Deterministic Test Verification**:
   - Ran full test suite: **69 tests passing in 0.78s** (`pytest`).
   - Verified that all sensor and geometry implementations remain untouched.

---

## Final Recommendation: Readiness for Run 2

| Gate / Prerequisite | Status | Evidence / Notes |
|:---|:---:|:---|
| **Run 1 Data Integrity** | **PASS** | 3,524 frames completely verified, zero NaN corruptions, 100% protocol adherence. |
| **Analysis Corrections** | **PASS** | Overstatements, throughput conflations, and mechanistic hypotheses corrected. |
| **Vertical Geometry Understanding** | **PASS** | Canthal normal code inspected; pixel-scale and bilateral limitations identified. |
| **Run 2 Protocol Design** | **PASS** | 5s fixed collection + 2s automated settling + visual fixation anchors specified. |

### Verdict
**READY FOR RUN 2.**  
Proceeding with Run 2 using the refined protocol (5s fixed duration, 2s automated settling, and bilateral normalized canthal logging) will provide a clean, transient-free, and scale-invariant benchmark.
