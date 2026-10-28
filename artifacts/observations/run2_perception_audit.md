# VoxGaze Milestone M1.5-A: Run 2 Perception Audit & Eye-Geometry Hardening Report

**Report Artifact**: `artifacts/observations/run2_perception_audit.md`  
**Evaluation Scope**: Analysis of physical Run 2 dataset, root cause isolation of extreme gaze ratios, coordinate derivation of canthal-normal sign conventions, and deterministic implementation of eye-geometry quality gates.  
**Repository State**: Branch `feature/mobileone-s0-integration` | Milestone M1.5-A  

---

## 1. Dataset Provenance & Session Overview

The Run 2 protocol (`ObservationSessionV2`) enforces automated phase timing (2.0s settling countdown without logging + 5.0s recording streamed to disk) across 20 canonical guided conditions.

### Primary Dataset
* **Session ID**: `obs_v2_20261021_164127_e73263`
* **CSV File**: `artifacts/observations/obs_v2_20261021_164127_e73263.csv` (Size: 554,801 bytes)
* **Summary JSON**: `artifacts/observations/obs_v2_20261021_164127_e73263_summary.json` (Size: 19,129 bytes)
* **Date & Duration**: 2026-10-21 16:41:27 to 16:45:27 UTC+5:30 (Elapsed session: **239.22 s**, ~4.0 min)
* **Session Status**: `COMPLETED` (20 of 20 conditions completed, 0 skipped, 0 aborted)
* **Total Frames Recorded**: **3,005 frames**
* **Face Detection Availability**: 2,963 frames (98.6% tracking rate). Tracking loss occurred exclusively during Condition 19 (`TRACKING_RECOVERY`, 42 contiguous blackout frames).
* **Latency Profile**:
  * MobileOne ONNX inference: p50 = **7.08 ms**, p95 = **7.75 ms** ($N=2,963$).
  * Total frame processing: p50 = **28.24 ms**, p95 = **43.42 ms** ($N=3,005$).

### Excluded Artifact
* `obs_v2_20261021_164531_9e18c7.csv`: Aborted at 0.0s elapsed with 0 recorded frames. Preserved on disk but excluded from all numerical evaluations.

---

## 2. Quantitative Distributions & Root Cause Analysis of Outliers

### 2.1 Feature Distribution Summary ($N = 2,963$ detected frames)

| Feature Name | Minimum | 5th Pct (p05) | Median (p50) | 95th Pct (p95) | Maximum | Normal Physiological Expectation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `right_eye_h_ratio` | 0.0502 | 0.3065 | 0.4877 | 0.6913 | **1.5879** | $[0.10, 0.90]$ |
| `left_eye_h_ratio` | **-0.1827** | 0.2924 | 0.5083 | 0.6507 | 0.8387 | $[0.10, 0.90]$ |
| `right_eye_v_ratio` | **-8.6396** | -0.0103 | 0.4396 | 0.5683 | **11.6504** | $[0.20, 0.70]$ |
| `left_eye_v_ratio` | **-5.5686** | -0.0552 | 0.4358 | 0.5603 | **17.3821** | $[0.20, 0.70]$ |
| `canthal_normal_dist_px` | -4.14 px | -2.34 px | -1.70 px | +0.60 px | +1.64 px | $[-3.5\text{ px}, +2.0\text{ px}]$ |
| `right_eye_canthal_ratio`| **-0.5858** | -0.0779 | -0.0571 | +0.0191 | +0.0418 | $[-0.15, +0.10]$ |
| `left_eye_canthal_ratio` | -0.1823 | -0.0945 | -0.0598 | +0.0277 | +0.0655 | $[-0.15, +0.10]$ |
| `right_eye_ear` | 0.0046 | 0.0352 | 0.3065 | 0.5746 | **1.1866** | $[0.20, 0.45]$ (open) |
| `left_eye_ear` | 0.0028 | 0.0336 | 0.3080 | 0.4291 | 0.7065 | $[0.20, 0.45]$ (open) |
| `mobileone_yaw_deg` | -71.56° | -25.71° | -1.31° | +19.17° | +44.07° | Visual instruction: $[-25°, +25°]$ |
| `mobileone_pitch_deg` | -47.53° | -21.68° | -4.86° | +21.08° | +39.90° | Visual instruction: $[-15°, +20°]$ |

### 2.2 Root Causes of Outliers & Extreme Values

Exactly **225 frames** exhibited eye ratios departing from $[-0.5, 1.5]$. Three distinct mechanisms account for all observed departures:

#### Mechanism A: Eyelid Closure / Blink-Induced Division Blowup
In `backend/app/perception/geometry.py`, normalized vertical iris position is computed by projecting the iris center onto the vector connecting the upper and lower eyelid landmarks:
$$\text{proj} = \frac{(\vec{p}_{\text{iris}} - \vec{p}_{\text{upper}}) \cdot (\vec{p}_{\text{lower}} - \vec{p}_{\text{upper}})}{\|\vec{p}_{\text{lower}} - \vec{p}_{\text{upper}}\|^2 + \varepsilon}$$

* **Observed Evidence**:
  * Row 608 (frame 2034, `EYES_CENTER_RETURN`): `right_eye_ear = 0.0046`, `right_eye_v_ratio = -5.9224`.
  * Row 852 (frame 2443, `EYES_UP`): `left_eye_ear = 0.0028`, `left_eye_v_ratio = +17.3821`.
  * Row 1090 (frame 2908, `EYES_DOWN`): `right_eye_ear = 0.0291`, `right_eye_v_ratio = -0.6006`, `left_eye_v_ratio = -2.1396`.
* **Mechanical Explanation**: When the eyelids close during spontaneous blinks, the vertical distance $d_{\text{lid}} = \|\vec{p}_{\text{lower}} - \vec{p}_{\text{upper}}\|$ drops toward zero ($< 1.0\text{ px}$, squared length $< 1.0\text{ px}^2$). MediaPipe's iris regressor continues to output an estimated coordinate behind or near the eyelid seam. Dividing that small displacement by a near-zero denominator ($d_{\text{lid}}^2 < 1$) amplifies subpixel fitting noise into extreme values ($-8.6$ to $+17.4$). Furthermore, when upper and lower lids cross vertically ($dy < 0$), the orientation vector inverts, producing large negative ratios.
* **Engineering Policy**: When $d_{\text{lid}} < 2.0\text{ px}$ or $\text{EAR} < 0.08$, vertical eyelid projection is **mathematically degenerate / undefined**. The system must flag `INSUFFICIENT_LID_SEPARATION` or `EYELID_CLOSED_OR_BLINK`, mark `geometry_valid = False`, and preserve the raw number in diagnostic logs without inventing an artificial value or clamping.

#### Mechanism B: Orbital Translation During Head Movement
* **Observed Evidence**: In `HEAD_LEFT_EYES_FORWARD` (Rows 1354–1362), `left_eye_h_ratio` consistently reached **`-0.0815` to `-0.1827`**, while `right_eye_h_ratio` dropped to `0.0502`.
* **Mechanical Explanation**: When the participant turned their head to the left while keeping visual fixation on the screen center, the eyeball in the left socket rotated toward the temporal edge. In the 2D image plane, perspective foreshortening and orbital rotation caused the iris center to reach and slightly exceed the 2D canthus landmark segment.
* **Engineering Policy**: Negative horizontal ratios down to $-0.25$ during lateral head rotation are physiologically real physical projections, not calculation errors. They are preserved within the provisional plausible band $[-0.25, 1.25]$.

#### Mechanism C: Landmark Degradation During Tracking Recovery
* **Observed Evidence**: Row 2725 (frame 6441, `TRACKING_RECOVERY`): `right_eye_ear = 1.1866`, `right_eye_h_ratio = 1.5879`, `right_eye_canthal_ratio = -0.5858`.
* **Mechanical Explanation**: Immediately upon camera uncover, the MediaPipe Face Mesh model required 1–3 frames to stabilize feature points. During this transient frame, the upper and lower eyelids were regressed to an unnatural aperture wider than the eye width ($\text{EAR} > 1.0$).
* **Engineering Policy**: An $\text{EAR} > 0.70$ is anatomically implausible for human palpebral fissures. It is explicitly flagged as `IMPLAUSIBLE_EAR` with `geometry_valid = False`.

---

## 3. Mathematical Derivation of Canthal-Normal Sign Convention

### 3.1 Coordinate Geometry
In standard camera image space, $+x$ extends rightward and $+y$ extends downward:
* **Right Eye** (image left):
  * Nasal anchor: `p_start = inner_canthus` (landmark 133, larger $x$, nearer facial midline).
  * Temporal anchor: `p_end = outer_canthus` (landmark 33, smaller $x$, nearer image edge).
  * Axis vector: $\vec{v} = (dx, dy) = (\text{outer.x} - \text{inner.x}, \text{outer.y} - \text{inner.y})$.
  * Since outer canthus is to the left of inner canthus on the image, $dx < 0$.
* **Normal Vector**:
  * Rotating $\vec{v}$ by $90^\circ$ clockwise yields $\vec{n} = (dy, -dx)$.
  * Because $dx < 0$, the vertical component is $-dx > 0$ (**pointing downward** in $+y$).
* **Numerator Projection**:
  $$\text{proj\_num} = (\vec{p}_{\text{iris}} - \vec{p}_{\text{start}}) \cdot \vec{n} = (px - sx) \cdot dy - (py - sy) \cdot dx$$
  * When the eye moves **UP**, the iris coordinate $py$ decreases ($\Delta py < 0$).
  * The vertical term is $-(\Delta py) \cdot dx$. Since $\Delta py < 0$ and $dx < 0$, their product is positive, and the negative sign makes the term **negative**:
    $$\Delta \text{proj} < 0 \quad (\text{for upward gaze})$$
  * When the eye moves **DOWN**, $py$ increases ($\Delta py > 0$):
    $$-(\Delta py) \cdot dx > 0 \quad (\text{for downward gaze})$$

* **Left Eye** (image right):
  * `p_start = outer_canthus` (landmark 263, larger $x$, nearer image edge).
  * `p_end = inner_canthus` (landmark 362, smaller $x$, nearer facial midline).
  * Here also, $dx = \text{inner.x} - \text{outer.x} < 0$.
  * Normal vector $\vec{n} = (dy, -dx)$ has downward vertical component $-dx > 0$.
  * Upward movement yields $\Delta \text{proj} < 0$; downward movement yields $\Delta \text{proj} > 0$.

### 3.2 Verification Against Run 2 Empirical Medians

| Condition Name | Right Eye Canthal Ratio | Left Eye Canthal Ratio | Geometric Meaning |
| :--- | :---: | :---: | :--- |
| `EYES_UP` | **`-0.0652`** | **`-0.0701`** | Iris displaced superiorly above canthal line (most negative) |
| `CENTER_BASELINE` | **`-0.0584`** | **`-0.0608`** | Resting neutral position (superior to intercanthal line) |
| `EYES_DOWN` | **`-0.0039`** | **`+0.0068`** | Iris displaced inferiorly toward/below canthal line (most positive) |

**Derivation Verdict**: The coordinate equations are mathematically sound and consistent across both anatomical eyes:
$$\text{UP} < \text{CENTER} < \text{DOWN}$$

### 3.3 Upward Eye-Only Gaze Separation Limits
In Run 2, `right_eye_v_ratio` (eyelid normalized) measured `0.4311` on `EYES_UP` versus `0.4339` on `CENTER_BASELINE` (delta = `-0.0028`). Meanwhile, `canthal_normal_dist_px` shifted from `-1.80 px` to `-1.98 px` (delta = `-0.18 px`).
1. **Direct Observation**: Eyelid-normalized vertical ratio does not cleanly separate unconstrained upward eye gaze from center resting gaze.
2. **Kinematic Interpretation**: When looking up, the *levator palpebrae superioris* contracts synchronously with the *superior rectus*, retracting the upper eyelid upward in tandem with the cornea. Consequently, the distance from the upper lid to the iris center changes proportionally with total palpebral opening.
3. **Sensor Limitation Hypothesis**: Upward ocular rotation causes the superior limbus to occlude under the upper eyelid, which may bias MediaPipe's elliptical iris contour fitting downward, attenuating detected translation.

---

## 4. Distinction of Evidentiary Status

To maintain scientific integrity, findings are categorized by epistemic certainty:

* **Directly Observed Facts**:
  1. Spontaneous blinks and eye closures cause $d_{\text{lid}} < 2\text{ px}$ and produce extreme $V_{\text{eyelid}}$ values ($-8.6$ to $+17.4$).
  2. The sign convention for canthal-normal projection preserves $\text{UP} < \text{CENTER} < \text{DOWN}$ across both eyes.
  3. In `HEAD_LEFT_EYES_FORWARD`, MobileOne Yaw shifted to $-26.14^\circ$ while eye-in-socket geometry moved temporally ($H=0.6462$), proving MobileOne reflects head orientation rather than gaze on screen.
  4. Commodity webcam hardware achieved $30.01\text{ FPS}$ sustained throughput with $28.2\text{ ms}$ processing latency.
* **Interpretations Based on Physical Geometry**:
  1. Canthi are anchored to facial skeletal landmarks (palpebral ligaments) and provide more stable vertical reference axes than moving eyelids during squinting and partial blinks.
  2. Resting iris centers sit anatomically superior to the intercanthal line, explaining why neutral center canthal ratios are negative ($\approx -0.059$).
* **Unproven Working Hypotheses**:
  1. Upper eyelid occlusion of the superior limbus is hypothesized to attenuate detected upward iris translation in MediaPipe.
  2. Multi-point personalized calibration is hypothesized to compensate for small upward gaze separation if head tilt is stabilized.

---

## 5. Eye-Geometry Quality Validation Architecture

Implemented in `backend/app/perception/geometry.py`:

### 5.1 Provisional Quality Thresholds

```python
MIN_EYE_WIDTH_PX: float = 10.0         # Reject if canthal distance < 10 px
MIN_LID_SEPARATION_PX: float = 2.0     # Undefined eyelid projection if < 2 px
MIN_EAR_FOR_GAZE: float = 0.08         # Closed eye / blink threshold
MAX_PLAUSIBLE_EAR: float = 0.70        # Landmark distortion threshold
PLAUSIBLE_H_MIN: float = -0.25         # Allowable ocular counter-rotation limit
PLAUSIBLE_H_MAX: float = 1.25
PLAUSIBLE_V_MIN: float = -0.25
PLAUSIBLE_V_MAX: float = 1.25
PLAUSIBLE_CANTHAL_MIN: float = -0.25   # Canthal normal ratio range
PLAUSIBLE_CANTHAL_MAX: float = 0.25
```

### 5.2 Quality Contracts & Functions
* **`GeometryQualityReason`** (Enum):
  `VALID`, `NON_FINITE_COORDINATES`, `INSUFFICIENT_EYE_WIDTH`, `INSUFFICIENT_LID_SEPARATION`, `EYELID_CLOSED_OR_BLINK`, `IMPLAUSIBLE_EAR`, `IMPLAUSIBLE_H_RATIO`, `IMPLAUSIBLE_V_RATIO`, `IMPLAUSIBLE_CANTHAL_RATIO`.
* **`EyeQualityResult`** (frozen dataclass):
  Contains `is_valid: bool`, `rejection_reasons: Tuple[GeometryQualityReason, ...]`, and raw metrics (`eye_width_px`, `lid_separation_px`, `h_ratio`, `v_ratio`, `ear`, `canthal_ratio`).
* **`BilateralQualityResult`** (frozen dataclass):
  Contains combined bilateral `is_valid: bool`, individual `right_eye` and `left_eye` results, and human-readable `bilateral_reasons`.
* **`validate_eye_geometry(...)`**: Pure, deterministic function evaluating all quality gates.
* **`validate_bilateral_geometry(...)`**: Evaluates independent and combined bilateral quality.

### 5.3 Non-Clamping Invariant
Extreme values (e.g., $V = -5.92$) are **never silently clamped** to $0.0$ or $1.0$. The raw numbers remain intact in data structures and CSV logs for scientific auditability. The `geometry_valid` flag is marked `False`, and `unavailable_reason` explicitly records `EYELID_CLOSED_OR_BLINK` or `INSUFFICIENT_LID_SEPARATION`.

---

## 6. Verification & Automated Test Results

### 6.1 Targeted Geometry Hardening Tests (`tests/unit/test_gaze_geometry.py`)
Eight new unit tests were added, covering all required boundary and failure modes:
1. `test_validate_eye_geometry_valid_right_and_left`: Valid baseline passes for both anatomical eyes.
2. `test_validate_eye_geometry_non_finite`: NaN and Infinity coordinates rejected with `NON_FINITE_COORDINATES`.
3. `test_validate_eye_geometry_sub_threshold_and_zero_width`: Eyes with width $< 10\text{ px}$ and $0\text{ px}$ rejected with `INSUFFICIENT_EYE_WIDTH`.
4. `test_validate_eye_geometry_lid_separation_and_blink`: Closed lids ($< 2.0\text{ px}$, $\text{EAR} < 0.08$) rejected with `INSUFFICIENT_LID_SEPARATION` and `EYELID_CLOSED_OR_BLINK`.
5. `test_validate_eye_geometry_implausible_ear`: Landmark distortion ($\text{EAR} > 0.70$) rejected with `IMPLAUSIBLE_EAR`.
6. `test_validate_eye_geometry_plausible_boundary_ratios`: Validates ratios at exact provisional boundaries and rejects out-of-band ratios.
7. `test_validate_eye_geometry_preserves_raw_unclamped_measurements`: Confirms raw extreme ratios are preserved in result fields without clamping.
8. `test_validate_bilateral_geometry_asymmetric`: Verifies asymmetric conditions (one eye valid, other closed) under strict and permissive policies.

### 6.2 Full Test Suite Result
```text
pytest tests/ -v
============================= 87 passed in 4.49s ==============================
```
**Zero regressions**. All 87 tests passed across M0 governance, M1 perception, MobileOne S0 estimator, calibration, diagnostic, and structured observation modules.

---

## 7. Recommended Next Step for Milestone M1.5-B

With eye-geometry quality validation and Run 2 perception audit complete, the recommended next experimental task is **Milestone M1.5-B: Physical Target Calibration & Intent FSM Scaffolding**:

1. **Controlled Vertical Calibration Experiment**:
   Test whether personalized 5-point calibration (with the user fixating high-contrast targets on screen) generates distinct enter/exit thresholds that reliably separate `UP` from `CENTER` despite small eyelid deltas.
2. **Transition Beyond Scaffolding Gate**:
   Update `test_no_premature_future_modules` in `test_m0_gates.py` to allow Milestone M1.5 implementation of `backend/app/interaction/fsm.py` (the deterministic 4-command state machine: `IDLE -> PREVIEW -> CONFIRMED -> IDLE`).
3. **Explicit Confirmation & Speech Synthesis**:
   Wire deliberate blink confirmation ($300-800\text{ ms}$) to offline Piper TTS synthesis, ensuring no speech is emitted without visual preview and explicit user confirmation.
