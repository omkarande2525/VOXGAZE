# VoxGaze M1.5-B1 — Physical Calibration Session Failure Audit (Run 2)

**Audited Artifacts**:
- CSV: `artifacts/observations/calib_v15_049638_20261021_205213_df2ec9.csv`
- Summary JSON: `artifacts/observations/calib_v15_049638_20261021_205213_df2ec9_summary.json`
- Session Timestamp: `2026-10-21T20:52:13`
- Evaluated Active Eye: Anatomical Right (nominal)
- Total Protocol Conditions: 27 / 27 (100% completed)
- Total Frames Recorded: 1,668

---

## Executive Summary

The M1.5-B1 physical calibration session completed all 27 conditions with 100.0% face detection availability and 99.88% geometry validity. However, **every non-center gaze direction (LEFT, RIGHT, UP, DOWN) and BLINK failed separation gates and were flagged UNSUPPORTED**. In Phase 2 independent testing, all non-center trials were classified as `CENTER`.

A rigorous code inspection and empirical trace identified **four compounding mechanisms**:
1. **Critical Pipeline Flaw — Premature Frame Mirroring (`cv2.flip`)**: `scripts/run_calibration_validation.py` (line 358) horizontally flipped `raw_frame` *before* passing it to MediaPipe FaceLandmarker and MobileOne. In a mirrored face, canonical right-eye landmarks (33, 133, 468) on image-left attach to the user's *physical left eye*. Normalizing the physical left eye with right-eye equations inverted horizontal directionality ($H_{left} > H_{center} > H_{right}$), producing negative separation deltas ($\Delta H_{left} = -0.0084$, $\Delta H_{right} = -0.0316$).
2. **Quality Gate Incompatibility with Closed Eyes**: `validate_eye_geometry` rejects $EAR < 0.18$ as `EYELID_CLOSED_OR_BLINK`. The evaluator required `geometry_valid=True` when filtering samples, creating an architectural contradiction: genuine closed-eye frames would have been discarded. Furthermore, lacking auditory cues, the operator did not close their eyes during `CALIB_CLOSED` ($EAR \approx 0.248$), yielding $\Delta EAR = 0.0081 < 0.05$.
3. **Eyelid Confound vs Canthal Normal Separation**: Eyelid-based vertical ratio ($V_{eyelid}$) inverted because upper eyelid retraction during upward gaze and eyelid ptosis during downward gaze distorted the upper lid landmark ($UP = 0.4384 > CENTER = 0.4171 > DOWN = 0.3951$). In contrast, the skeletal canthal normal metric ($V_{canthal}$) preserved the physical invariant ($UP = -0.0475 < CENTER = -0.0341 < DOWN = -0.0276$), but the calibration gate was hardcoded to evaluate $V_{eyelid}$.
4. **Evaluator Metric Distortion**: The evaluator omitted `UP` when unsupported, but failed to omit `LEFT`, `RIGHT`, or `DOWN`. It evaluated them against disabled thresholds where they defaulted to `CENTER`, while reporting an aggregate accuracy of 62.5% driven entirely by the 5 center-return trials (5/5 correct center vs 0/3 correct directional).

---

## 1. Frame Counts: Per-Condition vs Aggregate Target Statistics

### 1.1 Observed Facts
- **Total CSV rows**: Exactly 1,668 data rows (plus 1 header row).
- **Phase breakdown**:
  - Phase 1 (`CALIBRATION_ESTIMATION`, Conditions 1–10): 648 frames.
  - Phase 2 (`INDEPENDENT_EVALUATION`, Conditions 11–19): 540 frames.
  - Phase 3 (`HEAD_MOVEMENT_CONTRAST`, Conditions 20–27): 480 frames.
  - Sum: $648 + 540 + 480 = 1,668$ frames.
- **Condition-level frame counts**:
  - `CALIB_CENTER_1` (Condition 1): **107 frames**.
  - `CALIB_CENTER_2` through `CALIB_CENTER_5` (Conditions 3, 5, 7, 9): **60, 60, 61, 60 frames**.
  - All non-center conditions in Phase 1 (`CALIB_LEFT`, `CALIB_RIGHT`, `CALIB_UP`, `CALIB_DOWN`, `CALIB_CLOSED`): **60 frames each**.
  - All Phase 2 test trials (Conditions 11–19): **60 frames each**.
  - All Phase 3 contrast conditions (Conditions 20–27): **60 frames each**.
- **JSON Summary Target Statistics**:
  - `phase1_calibration.target_statistics["CENTER"]["total_frames"]`: **348 frames**.
  - `phase1_calibration.target_statistics["LEFT"]["total_frames"]`: **60 frames**.
  - `phase1_calibration.target_statistics["RIGHT"]["total_frames"]`: **60 frames**.
  - `phase1_calibration.target_statistics["UP"]["total_frames"]`: **60 frames**.
  - `phase1_calibration.target_statistics["DOWN"]["total_frames"]`: **60 frames**.
  - `phase1_calibration.target_statistics["CLOSED"]["total_frames"]`: **60 frames**.

### 1.2 Code-Level Cause Confirmed by Inspection
1. **Wall-clock vs Fixed Frame Control**: `CalibrationValidationHarness.tick()` controls condition transitions using monotonic elapsed time (`recording_duration_s = 4.0`), not frame counts. Frame count is therefore a function of loop throughput ($FPS \times 4.0\text{ s}$).
2. **Initial Loop Acceleration**: During Condition 1, frame processing latency was low (median total frame latency 29.5 ms, $\approx 27.3\text{ FPS}$) because MobileOne inference and display overlay caching had not yet begun. After Condition 1, total frame latency stabilized at $\approx 61.9\text{ ms}$ ($\approx 15.2\text{ FPS}$), yielding exactly 60 frames per 4.0s window.
3. **Aggregation by Target Direction**: In `backend/app/calibration/evaluator.py:141` (`compute_phase1_target_stats`), Phase 1 records are grouped by `target_direction`. The 5 interleaved center conditions (Conditions 1, 3, 5, 7, 9) all share `target_direction == "CENTER"`. Their frames aggregate to:
   $$107 + 60 + 60 + 61 + 60 = 348\text{ frames}$$
   Non-center targets only appear once in Phase 1, so their aggregate frame count is 60.

---

## 2. Screen Target Positions and Protocol Sequence Mapping

### 2.1 Observed Facts & Geometry Trace
Visual targets are defined in normalized display coordinates $(x, y) \in [0.0, 1.0]^2$ in `backend/app/calibration/protocol.py`:
- `TARGET_CENTER`: $(0.50, 0.50) \rightarrow (640, 360)\text{ px}$ on $1280 \times 720$.
- `TARGET_LEFT`: $(0.15, 0.50) \rightarrow (192, 360)\text{ px}$ (operator's physical left).
- `TARGET_RIGHT`: $(0.85, 0.50) \rightarrow (1088, 360)\text{ px}$ (operator's physical right).
- `TARGET_UP`: $(0.50, 0.15) \rightarrow (640, 108)\text{ px}$ (operator's upper visual field).
- `TARGET_DOWN`: $(0.50, 0.85) \rightarrow (640, 612)\text{ px}$ (operator's lower visual field).
- `CALIB_CLOSED`: `target_norm_pos = None`.

### 2.2 Display Rendering Trace
In `scripts/run_calibration_validation.py:252` (`draw_hud`):
- `active_key = cond.target_direction if cond.target_direction in targets else ("CENTER" if cond.is_center else None)`
- When `CALIB_LEFT` (Condition 2) or `EVAL_LEFT` (Condition 12) was active, `targets["LEFT"]` was illuminated at $(192, 360)\text{ px}$ (green circle, radius 22 px, outer ring 32 px).
- The operator correctly looked to their physical left side of the display.
- When `CALIB_CLOSED` (Condition 10) was active, no target circle was illuminated; only the top banner displayed the instruction: `"Gently close both eyes and hold until timer completes."`

---

## 3. The Root Cause of Horizontal Failure: Premature Frame Mirroring

### 3.1 Code Inspection (`scripts/run_calibration_validation.py:358`)
```python
ret, raw_frame = cap.read()
if not ret or raw_frame is None:
    continue
raw_frame = cv2.flip(raw_frame, 1)  # Mirror mode  <-- ROOT CAUSE BUG
rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
detection_result = detector.detect_for_video(mp_image, ts_ms)
```

### 3.2 Topological Mechanism Confirmed by Inspection
MediaPipe FaceLandmarker operates in the image frame coordinates $(u, v)$:
- In an **unmirrored camera image** of a person facing the lens:
  - The person's physical right eye is on the image-left ($u \approx 0.35$).
  - Canonical right-eye landmarks (33 outer canthus, 133 inner canthus, 468 iris) attach to image-left.
- In a **mirrored camera image** (`cv2.flip(raw_frame, 1)`):
  - The person's physical right eye is reflected to image-right ($u \approx 0.65$).
  - The person's physical left eye is reflected to image-left ($u \approx 0.35$).
  - MediaPipe FaceLandmarker assigns landmarks 33, 133, and 468 to the eye on **image-left** — which is **physically the person's LEFT eye**.

### 3.3 Mathematical Impact on Horizontal Ratio
In `backend/app/perception/geometry.py`:
```python
def norm_horizontal_gaze(iris_pt, inner_canthus, outer_canthus, is_right_eye=True):
    if is_right_eye:
        p_start = inner_canthus   # nasal (nose)
        p_end = outer_canthus     # temporal (ear)
    else:
        p_start = outer_canthus   # temporal (ear)
        p_end = inner_canthus     # nasal (nose)
    return project_ratio_2d(iris_pt, p_start, p_end)
```
When evaluating the nominal right eye on a horizontally mirrored frame:
1. MediaPipe places $p_{start}$ (landmark 133) on the nasal side of the physical left eye.
2. MediaPipe places $p_{end}$ (landmark 33) on the temporal side of the physical left eye.
3. The operator looks to the physical **LEFT** (toward their left ear).
4. In the physical left eye, looking left shifts the iris toward the **temporal canthus** ($p_{end}$).
5. Projecting toward $p_{end}$ causes $H$ to **INCREASE** ($H > H_{center}$).
6. When looking to the physical **RIGHT**, the left iris shifts nasal ($p_{start}$), causing $H$ to **DECREASE** ($H < H_{center}$).

### 3.4 Observed Empirical Data (Both Eyes Inverted)
| Condition | Target | Nominal Eye Tracked | Median H | Expected Unmirrored | Actual Behavior |
|:---|:---|:---:|:---:|:---:|:---:|
| `CALIB_CENTER` (1,3,5,7,9) | CENTER | Right (phys. Left) | **0.5402** | Baseline | Baseline |
| `CALIB_LEFT` (2) | LEFT | Right (phys. Left) | **0.5485** | $H < 0.5402$ | **Increased (+0.0083)** |
| `CALIB_RIGHT` (4) | RIGHT | Right (phys. Left) | **0.5086** | $H > 0.5402$ | **Decreased (-0.0316)** |
| `CALIB_CENTER` (1,3,5,7,9) | CENTER | Left (phys. Right) | **0.5151** | Baseline | Baseline |
| `CALIB_LEFT` (2) | LEFT | Left (phys. Right) | **0.5297** | $H < 0.5151$ | **Increased (+0.0146)** |
| `CALIB_RIGHT` (4) | RIGHT | Left (phys. Right) | **0.4710** | $H > 0.5151$ | **Decreased (-0.0441)** |

Because $H_{left} > H_{center}$ and $H_{right} < H_{center}$:
$$\Delta H_{left} = H_{center} - H_{left} = 0.5402 - 0.5485 = -0.0084 \quad (< 0.06)$$
$$\Delta H_{right} = H_{right} - H_{center} = 0.5086 - 0.5402 = -0.0316 \quad (< 0.06)$$
Both directional separation deltas became negative, failing both horizontal gates.

---

## 4. Vertical Gaze: Eyelid Confound vs Canthal Normal Separation

### 4.1 Observed Facts: Eyelid Normalization Inversion ($V_{eyelid}$)
In Phase 1 calibration:
- `CENTER`: median $V_{eyelid} = 0.4171$
- `UP`: median $V_{eyelid} = 0.4384$ (increased by $+0.0212$, opposite of expected $V \approx 0.0$)
- `DOWN`: median $V_{eyelid} = 0.3951$ (decreased by $-0.0221$, opposite of expected $V \approx 1.0$)
- Separation deltas in summary JSON:
  $$\Delta V_{eyelid\_up} = C_v - U_v = 0.4171 - 0.4383 = -0.0212 \quad (< 0.05)$$
  $$\Delta V_{eyelid\_down} = D_v - C_v = 0.3950 - 0.4171 = -0.0221 \quad (< 0.05)$$

### 4.2 Anatomical & Biomechanical Cause
In `backend/app/perception/geometry.py:113`, $V_{eyelid}$ projects the iris between upper lid (159) and lower lid (145):
1. **Upward Gaze**: The levator palpebrae superioris retracts the upper eyelid upward, widening the palpebral fissure ($EAR$ increased from 0.2561 at center to 0.2898 at UP). MediaPipe's upper lid landmark shifts superiorly along with the iris, preventing the iris-to-lid ratio from dropping toward 0.
2. **Downward Gaze (Lid Ptosis Synergy)**: When looking down, the upper eyelid naturally descends to follow the cornea, and the fissure narrows ($EAR$ dropped from 0.2561 to 0.2128). The upper lid landmark moves down faster than the iris center regressor, causing the projection ratio to decrease to 0.3951 rather than increasing toward 1.0.
3. Therefore, $V_{eyelid}$ is non-monotonic across natural vertical gaze.

### 4.3 Validation of Rigid Canthal Normal ($V_{canthal}$)
In `backend/app/perception/geometry.py:128`, `norm_vertical_gaze_canthal_normal_experimental` computes the perpendicular distance of the iris from the rigid canthal ligament line (landmarks 33 and 133):
- `UP` (Condition 6): median $V_{canthal} = \mathbf{-0.0475}$ (displaced superiorly)
- `CENTER` (Conditions 1,3,5,7,9): median $V_{canthal} = \mathbf{-0.0341}$
- `DOWN` (Condition 8): median $V_{canthal} = \mathbf{-0.0276}$ (displaced inferiorly)

For Left Eye:
- `UP`: median $V_{canthal} = \mathbf{-0.0548}$
- `CENTER`: median $V_{canthal} = \mathbf{-0.0410}$
- `DOWN`: median $V_{canthal} = \mathbf{-0.0345}$

**Invariant Preservation**: On both eyes, $V_{canthal}$ strictly obeyed the anatomical monotonicity invariant:
$$\text{UP } (-0.0475) < \text{CENTER } (-0.0341) < \text{DOWN } (-0.0276)$$
The canthal normal deltas were positive:
$$\Delta V_{canthal\_up} = C_{vc} - U_{vc} = -0.0341 - (-0.0475) = +0.0134$$
$$\Delta V_{canthal\_down} = D_{vc} - C_{vc} = -0.0276 - (-0.0341) = +0.0065$$

**Code Cause of Failure**: `backend/app/calibration/evaluator.py:280,292` evaluated vertical gates *strictly* against `delta_v_eyelid_up` and `delta_v_eyelid_down`. Because the eyelid metric inverted, both UP and DOWN were marked UNSUPPORTED despite the canthal normal metric succeeding.

---

## 5. Frozen Threshold Derivation and Disabled Direction Representation

### 5.1 Observed Facts: Frozen Thresholds in Summary JSON
```json
"frozen_thresholds": {
  "left": { "enter": 0.0, "exit": 0.0 },
  "right": { "enter": 1.0, "exit": 1.0 },
  "up": { "enter": -999.0, "exit": -999.0 },
  "down": { "enter": 999.0, "exit": 999.0 },
  "blink_closed_ear": 0.15,
  "blink_open_ear": 0.28
}
```

### 5.2 Code-Level Derivation Trace
In `backend/app/calibration/evaluator.py:257-301`:
- When `is_left_supported` is False: `l_enter, l_exit = 0.0, 0.0`
- When `is_right_supported` is False: `r_enter, r_exit = 1.0, 1.0`
- When `is_up_supported` is False: `u_enter, u_exit = -999.0, -999.0`
- When `is_down_supported` is False: `d_enter, d_exit = 999.0, 999.0`

In `classify_frame_discrete`:
- Left triggers if `h <= t.left.enter` (with `enter = 0.0`, requires $H \le 0.0$).
- Right triggers if `h >= t.right.enter` (with `enter = 1.0`, requires $H \ge 1.0$).
- Up triggers if `v <= t.up.enter` (requires $V \le -999.0$).
- Down triggers if `v >= t.down.enter` (requires $V \ge 999.0$).

### 5.3 Safety & Representation Risk
1. **Sentinel Inconsistency**: Left/Right use boundary floats $(0.0, 1.0)$, while Up/Down use large numbers $(-999.0, 999.0)$.
2. **Fragility Under Real Noise**: In real tracking sessions, unnormalized or extreme eyelid/iris coordinates CAN produce $H \ge 1.0$ (in `obs_v2`, `right_eye_h_ratio` reached a maximum of $1.5879$). If a consumer evaluates `h >= t.right.enter` without checking `is_right_supported`, extreme noise will trigger a false `RIGHT` activation.
3. **Contract Deficiency**: `DirectionThreshold` in `backend/app/core/contracts.py` only defines `enter: float` and `exit: float`. It lacks an explicit `is_enabled: bool` field. Relying on magic sentinel floats to encode disabled state violates strict defensive data contracts.

---

## 6. The CLOSED Condition and Blink Measurement Failure

### 6.1 Observed Facts in Condition 10 (`CALIB_CLOSED`)
- **Recorded frames**: 60 frames.
- **Right eye EAR**: min = 0.1965, max = 0.2718, median = **0.2480**, mean = 0.2436.
- **Left eye EAR**: min = 0.1712, max = 0.2739, median = **0.2593**, mean = 0.2516.
- **Open-eye Center baseline EAR**: median = **0.2561**.
- **Separation**: $\Delta EAR = 0.2561 - 0.2480 = \mathbf{0.0081} \ll 0.05$.
- **Conclusion**: The captured measurements **do NOT demonstrate actual blink separation**. The operator's eyes remained open throughout Condition 10.

### 6.2 Code-Level Causes Confirmed by Inspection
1. **Absence of Auditory Cueing**:
   - `scripts/run_calibration_validation.py` only provided visual feedback via the on-screen OpenCV window.
   - For `CALIB_CLOSED`, the user cannot see the screen while closing their eyes. Without an audible beep or speech cue to signal the start and end of settling (2s) and recording (4s), the operator could not time eye closure.
2. **Quality Gate Architectural Contradiction**:
   - In `backend/app/perception/geometry.py:498`:
     ```python
     if ear < min_ear_for_gaze:  # MIN_EAR_FOR_GAZE = 0.18
         reasons.append(GeometryQualityReason.EYELID_CLOSED_OR_BLINK)
     ```
     When $EAR < 0.18$, `validate_eye_geometry` marks `is_valid = False`.
   - In `backend/app/calibration/evaluator.py:84`:
     ```python
     valid_records = filter_valid_samples(records, active_eye=active_eye, require_geometry_valid=True)
     ```
   - If the operator *had* closed their eyes ($EAR \approx 0.05$), `validate_eye_geometry` would have marked every frame invalid, and `filter_valid_samples` would have discarded 100% of the closed-eye frames, resulting in `valid_samples_used = 0`!
   - This architectural contradiction must be decoupled: closed-eye baseline estimation requires validating landmarks without rejecting low EAR.

---

## 7. Trial-Level Medians vs Frame Statistics and Baseline Drift

### 7.1 Intra-Trial Tightness (Temporal Correlation)
Analyzing condition-by-condition medians and Median Absolute Deviation (MAD):
- `CALIB_LEFT` (Cond 2): $H = 0.5485 \pm 0.0040$
- `CALIB_CENTER_2` (Cond 3): $H = 0.5391 \pm 0.0031$
- `CALIB_RIGHT` (Cond 4): $H = 0.5086 \pm 0.0053$
- `CALIB_UP` (Cond 6): $H = 0.5341 \pm 0.0042$
- `CALIB_DOWN` (Cond 8): $H = 0.5290 \pm 0.0057$

Intra-trial MAD is minuscule ($\approx 0.003 - 0.005$). Frames within a 4-second fixation window are strongly temporally autocorrelated. Treating 60 consecutive frames as 60 independent observations is statistically invalid.

### 7.2 Inter-Trial Baseline Drift Across Session
Across the 14 center-fixation conditions recorded over 3 minutes:
- Initial Center Baseline: $H = 0.5447, \quad V = 0.3871$
- Maximum Center $H$ drift: span of $[0.5150, 0.5460]$ ($\Delta H = 0.0310$)
- Maximum Center $V$ drift: span of $[0.3821, 0.4450]$ ($\Delta V = 0.0629$)

### 7.3 Drift-to-Signal Ratio
Comfortable physical gaze shifts produce ratio excursions of $\Delta H \approx 0.04 - 0.08$. Postural drift over several minutes accounts for $\Delta H \approx 0.0310$.
Static absolute thresholds derived in Phase 1 become increasingly misaligned later in the session unless normalized relative to periodic return-to-center baselines.

---

## 8. Evaluator Accuracy Calculation and Confusion Matrix Flaws

### 8.1 Dissection of the Reported 62.5% Accuracy
In `phase2_independent_evaluation`:
- Reported `trial_level_accuracy`: **0.625 (62.5%)**
- Reported `frame_level_accuracy`: **0.625 (62.5%)**
- Trials evaluated: 8 (Conditions 11, 12, 13, 14, 15, 17, 18, 19). Condition 16 (`EVAL_UP`) was omitted.

| Condition Index | Condition Name | Target | Supported? | Trial Prediction | Correct? | Frame Correct / Total |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 11 | `EVAL_CENTER_1` | CENTER | Yes | CENTER | **True** | 60 / 60 |
| 12 | `EVAL_LEFT` | LEFT | **No** | CENTER | **False** | 0 / 60 |
| 13 | `EVAL_CENTER_2` | CENTER | Yes | CENTER | **True** | 60 / 60 |
| 14 | `EVAL_RIGHT` | RIGHT | **No** | CENTER | **False** | 0 / 60 |
| 15 | `EVAL_CENTER_3` | CENTER | Yes | CENTER | **True** | 60 / 60 |
| 16 | `EVAL_UP` | UP | **No** | *OMITTED* | *N/A* | 0 / 0 |
| 17 | `EVAL_CENTER_4` | CENTER | Yes | CENTER | **True** | 60 / 60 |
| 18 | `EVAL_DOWN` | DOWN | **No** | CENTER | **False** | 0 / 60 |
| 19 | `EVAL_CENTER_5` | CENTER | Yes | CENTER | **True** | 60 / 60 |
| **Total** | | | | | **5 / 8 (62.5%)** | **300 / 480 (62.5%)** |

### 8.2 Invalidation of the Evaluation Metric
1. **Asymmetric Omission**: `evaluator.py:430` skipped `UP` when unsupported (`status: OMITTED_UNSUPPORTED`), but failed to skip `LEFT`, `RIGHT`, or `DOWN`.
2. **Conflating Center Stability with Command Accuracy**: When all directional classifiers are disabled, the classifier defaults to `CENTER` for 100% of frames. Evaluating 5 center trials yields 5/5 correct center classifications, producing an aggregate accuracy of $5 / 8 = 62.5\%$.
3. **True Directional Command Sensitivity**:
   $$\text{Active Directional Sensitivity} = \frac{0 \text{ correct directional trials}}{3 \text{ directional trials}} = \mathbf{0.0\%}$$
   Reporting 62.5% accuracy masks complete directional non-functionality. When directional gates fail, directional accuracy must be reported as 0.0% or flagged as invalid.

---

## 9. Head-Contrast Results

### 9.1 Observed Facts in Phase 3
- `HEAD_LEFT_COMFORTABLE` (Cond 20): Mean MobileOne yaw $= +13.91^\circ$, pitch $= -16.73^\circ$, ocular $H = 0.5471$.
- `HEAD_CENTER_RETURN_1` (Cond 21): Mean MobileOne yaw $= +10.66^\circ$, pitch $= -19.60^\circ$, ocular $H = 0.5337$.
- `HEAD_RIGHT_COMFORTABLE` (Cond 22): Mean MobileOne yaw $= +9.59^\circ$, pitch $= -14.77^\circ$, ocular $H = 0.5504$.
- `HEAD_UP_COMFORTABLE` (Cond 24): Mean MobileOne yaw $= +16.08^\circ$, pitch $= -3.63^\circ$ (elevated from baseline $-19.6^\circ$), ocular $H = 0.4874$.
- `HEAD_DOWN_COMFORTABLE` (Cond 26): Mean MobileOne yaw $= -0.48^\circ$, pitch $= -18.31^\circ$, ocular $H = 0.5373$.

### 9.2 Inspection Findings
1. Pitch showed clear head elevation during `HEAD_UP` ($-3.63^\circ$ vs baseline $-19.60^\circ$).
2. Yaw estimates were distorted (+13.91° for left vs +9.59° for right) because the face crop was taken from the horizontally mirrored `raw_frame`. MobileOne is trained on unmirrored face images; mirroring inverts the horizontal gradient features.
3. **Architectural Non-Goal**: MobileOne estimates head pose, not ocular gaze angle. MobileOne outputs must strictly remain a diagnostic contrast to detect gross head rotation, and must NEVER be used to infer gaze accuracy or substitute for ocular tracking.

---

## 10. Recommended Experiments

To empirically verify fixes before any physical protocol re-run:

1. **Experiment E1 — Offline Replay on Mirrored vs Unmirrored Crops**:
   - Create a test script that takes a known facial image with left/right gaze, feeds both `frame` and `cv2.flip(frame, 1)` through MediaPipe FaceLandmarker, and verifies that landmark identities swap and horizontal ratios invert.
2. **Experiment E2 — Decoupled Closed-Eye EAR Quality Gate Test**:
   - Write unit tests verifying that `filter_valid_samples(require_geometry_valid=False)` or a dedicated closed-eye filter preserves frames with $EAR \in [0.01, 0.15]$ while rejecting non-finite coordinates.
3. **Experiment E3 — Canthal Normal Threshold Derivation Simulation**:
   - Replay Phase 1 records from `calib_v15_049638_20261021_205213_df2ec9.csv` using $V_{canthal}$ deltas ($\Delta V_{canthal\_up} = +0.0134$, $\Delta V_{canthal\_down} = +0.0065$) to verify threshold derivation and hysteresis stability.
4. **Experiment E4 — Symmetric Unsupported Direction Omission in Evaluator**:
   - Test that when `is_left_supported`, `is_right_supported`, `is_up_supported`, and `is_down_supported` are all False, all directional trials are marked `OMITTED_UNSUPPORTED`, and the evaluator reports directional command accuracy as 0.0% (or uncalibrated) rather than 62.5%.
5. **Experiment E5 — Audible Cues Verification**:
   - Verify that an audio tone / console bell triggers reliably at settling and recording phase transitions without introducing I/O latency to the frame processing loop.

---

## 11. Smallest Proposed Corrective Plan

Do not implement speculative changes. The corrective plan is strictly minimal and targeted:

1. **Move Mirroring to Presentation Only (`scripts/run_calibration_validation.py`)**:
   - Pass pristine, unmirrored `raw_frame` to MediaPipe and MobileOne.
   - Apply horizontal flip only when drawing the GUI canvas for user feedback: `cv2.imshow(window_name, cv2.flip(hud, 1))`.
   - Invert on-screen target $x$-coordinates for the flipped HUD canvas so targets match the mirrored video feed.
2. **Decouple Quality Gate for Closed Eyes**:
   - In `backend/app/calibration/evaluator.py`, allow $EAR < 0.18$ specifically during `CLOSED` condition filtering (`allow_closed_eyes=True`), while still verifying finite coordinates and face presence.
   - Add an audible console bell (`sys.stdout.write('\a')` or sound cue) at settling/recording phase transitions so the operator knows when to close/open eyes.
3. **Evaluate Rigid Canthal Normal Alongside Eyelid Vertical**:
   - In `backend/app/calibration/evaluator.py`, evaluate vertical separation on the bone-anchored canthal normal metric ($V_{canthal}$) when eyelid vertical delta fails due to lid synergy.
4. **Enforce Explicit Contract Semantics and Omission Consistency**:
   - Add explicit `is_enabled: bool` to `DirectionThreshold` in `backend/app/core/contracts.py` rather than relying on sentinel floats $(0.0, 1.0, \pm 999.0)$.
   - In `evaluator.py:evaluate_independent_phase2_trials`, omit ALL unsupported directions (`LEFT`, `RIGHT`, `UP`, `DOWN`) consistently, and report Directional Command Accuracy separately from Center Fixation Stability.

---

## 12. Implementation & Verification Summary (M1.5-B2)

### Changes Implemented

1. **Camera-Frame Mirroring Decoupled (`scripts/run_calibration_validation.py`)**:
   - Removed premature `raw_frame = cv2.flip(raw_frame, 1)` before perception. MediaPipe FaceLandmarker and MobileOne now receive pristine unmirrored camera frames.
   - Decoupled preview presentation: `display_frame = cv2.flip(raw_frame, 1)` is generated strictly for user GUI display. Target circles and HUD text are drawn cleanly on top of `display_frame`.
   - Verified that physical left gaze corresponds to medial iris motion toward nasal canthus (133) in the anatomical right eye, properly yielding $H \approx 0.0$ and positive separation $\Delta H_{left} > 0$.

2. **CLOSED-Condition Sample Handling & Audio Cues (`geometry.py`, `evaluator.py`, `run_calibration_validation.py`)**:
   - `validate_eye_geometry` and `filter_valid_samples` accept `allow_closed_eyes: bool = False`. When true, low EAR ($<0.18$) is permitted while strictly rejecting missing-face, non-finite, and zero/sub-threshold eye width.
   - `scripts/run_calibration_validation.py` passes `allow_closed_eyes=(target_direction == "CLOSED")` during live frame recording.
   - Implemented `emit_audio_cue` providing audible notifications (`PREPARE_CLOSE`, `CLOSE_NOW`, `EYES_OPEN`, `STEP_ADVANCE`) using `winsound.Beep` with stdout bell fallback.
   - Added high-contrast center alert box to `draw_hud` displaying explicit instructions when `target_direction == "CLOSED"`.
   - `derive_frozen_candidate_thresholds` verifies actual participant eye closure ($cl\_ear < 0.18$) before supporting blinks, producing explicit diagnostic reasons if the participant fails to close their eyes.

3. **Honest Vertical Feature Evaluation (`evaluator.py`)**:
   - Eyelid-based vertical ($V_{eyelid}$), canthal-normal vertical ($V_{canthal}$), and signed canthal normal distance ($d_{canthal}$ px) are tracked and reported with explicit definitions.
   - Monotonicity alone does not bypass provisional separation gates ($\Delta V \ge 0.05$). Unsupported vertical directions remain strictly disabled.

4. **Evaluator Reporting Integrity (`evaluator.py`, `run_calibration_validation.py`)**:
   - Implemented symmetrical omission for all unsupported directions (`LEFT`, `RIGHT`, `UP`, `DOWN`).
   - Separated metrics: `center_fixation_stability` and `directional_sensitivity` are evaluated and reported independently.
   - When all directions are unsupported, overall accuracy is reported as `None` / `UNAVAILABLE_NO_SUPPORTED_DIRECTIONS` rather than presenting center-only accuracy as overall accuracy.
   - Formatted summary printer handles `None` accuracies gracefully without raising exceptions.

5. **Defensive Unsupported Direction Threshold Contract (`contracts.py`, `gaze_engine.py`, `evaluator.py`)**:
   - Added `is_enabled: bool = True` to `DirectionThreshold` with complete backwards compatibility.
   - `GazeEngine` and `classify_frame_discrete` enforce `is_enabled`, immediately returning to `CENTER` on in-range ratios and returning `ABSTAIN` on out-of-range ($H \notin [0.10, 0.90]$) or non-finite inputs.

### Automated Test Suite Results
- Full test suite passed: **108 passed in 1.70s** (`pytest -v`).
- Zero warnings from `git diff --check`.
- CLI interface verified: `python scripts/run_calibration_validation.py --help` succeeded.
- Headless smoke test verified: executed 2 synthetic conditions with zero errors and generated valid structured artifacts.

### Remaining Physical Blockers & Risks
1. **Physical Gaze Separation Unmeasured**: Physical gaze separation has not yet been demonstrated on an unmirrored camera stream. Accuracy cannot be claimed until a physical trial is executed.
2. **Camera Positioning Sensitivity**: If the user's camera is significantly below or above eye level, resting center gaze may have an offset. The participant should ensure the webcam is positioned roughly level with the eyes.
3. **Participant Blink Timing**: The participant must listen for audio cues and keep eyes closed during the full recording duration of `CALIB_CLOSED`.

### Exact Commands for Next Physical Trial

```powershell
# 1. Activate environment
.venv\Scripts\Activate.ps1

# 2. Run full 27-condition physical calibration validation session (camera index 0, 1280x720)
python scripts/run_calibration_validation.py --camera 0 --width 1280 --height 720 --settling 2.0 --recording 4.0 --eye right
```
