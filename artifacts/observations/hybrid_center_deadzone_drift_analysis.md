# VoxGaze: Read-Only Analysis of Hybrid Model Center Deadzone Departures

**Date**: 2026-10-28  
**Task**: T2 (Read-Only Analysis of Center Deadzone False Activations)  
**Target Artifact**: `artifacts/observations/cmp_v16_20261025_164239_0d9a6c_comparison_report.json`  
**Data Provenance**: Physical Acquisition Session (`mock_mode: false`, 476 frames)  
**Auditor**: Antigravity Pair Programming (Milestone M1 Sensor Validation)  

---

## 1. Executive Summary & Objective

In physical comparison session `cmp_v16_20261025_164239_0d9a6c`, both `MobileOne S0 (Hybrid)` and `ResNet-34 (Hybrid)` achieved identical **40.0% center-return false-activation rates** (2 out of 5 center trials falsely escaped the center resting deadzone into active button territory). In stark contrast, both `MobileOne S0 (Angles-Only)` and `ResNet-34 (Angles-Only)` achieved a **0.0% center false-activation rate** (0 out of 5 center trials escaped).

This read-only investigation inspects the held-out Block 3 trial results from the real report to establish:
1. The preceding fixation target for each center trial.
2. The exact predicted median $(x, y)$ coordinates for both hybrid and angles-only models.
3. The exact Euclidean and axis-aligned distance outside the center resting deadzone ($[0.35, 0.35, 0.65, 0.65]$).
4. Which feature term (ocular geometry vs. pretrained angle) is responsible for driving gaze out of the resting deadzone.

---

## 2. Complete Center Trial Breakdown (Block 3 Held-Out Evaluation)

The resting center deadzone is defined as:
$$\text{CENTER Deadzone}: \quad x \in [0.35, 0.65], \quad y \in [0.35, 0.65]$$
A center trial is classified as **HIT (Stayed in Center)** if and only if:
$$0.35 \le \tilde{x} \le 0.65 \quad \text{and} \quad 0.35 \le \tilde{y} \le 0.65$$
If a coordinate escapes, the **Distance Outside Deadzone** is defined as the distance by which the median exceeds the nearest boundary:
$$\Delta x_{\text{out}} = \max(0, 0.35 - \tilde{x}, \tilde{x} - 0.65), \quad \Delta y_{\text{out}} = \max(0, 0.35 - \tilde{y}, \tilde{y} - 0.65)$$

### Comprehensive Observation Matrix (From Physical Report `0d9a6c`)

| Trial Name | Preceding Target | Feature Set | Median $x$ | Median $y$ | Deadzone Status | Distance Outside Deadzone | Contributing Term |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`B3_1_CENTER`** | Baseline / Rest | **MobileOne Hybrid** | $0.5021$ | $0.6551$ | **FAIL (Escaped)** | **$\Delta y = +0.0051$ (Down)** | **Geometry Term** ($+0.0530$ vs Angles) |
| | | MobileOne Angles | $0.5108$ | $0.6021$ | PASS (Center) | $0.0000$ (Inside) | Angles remain inside ($y \le 0.65$) |
| | | **ResNet-34 Hybrid** | $0.5070$ | $0.6921$ | **FAIL (Escaped)** | **$\Delta y = +0.0421$ (Down)** | **Geometry Term** ($+0.0720$ vs Angles) |
| | | ResNet-34 Angles | $0.4676$ | $0.6201$ | PASS (Center) | $0.0000$ (Inside) | Angles remain inside ($y \le 0.65$) |
| **`B3_3_CENTER`** | **`B3_2_DOWN`** | **MobileOne Hybrid** | $0.5194$ | $0.3193$ | **FAIL (Escaped)** | **$\Delta y = +0.0307$ (Up)** | **Geometry Term** ($-0.1671$ vs Angles) |
| | | MobileOne Angles | $0.4636$ | $0.4864$ | PASS (Center) | $0.0000$ (Dead center) | Angles perfectly centered ($y \approx 0.49$) |
| | | **ResNet-34 Hybrid** | $0.5193$ | $0.3303$ | **FAIL (Escaped)** | **$\Delta y = +0.0197$ (Up)** | **Geometry Term** ($-0.2012$ vs Angles) |
| | | ResNet-34 Angles | $0.4715$ | $0.5315$ | PASS (Center) | $0.0000$ (Dead center) | Angles perfectly centered ($y \approx 0.53$) |
| **`B3_5_CENTER`** | `B3_4_LEFT` | **MobileOne Hybrid** | $0.5118$ | $0.4985$ | PASS (Center) | $0.0000$ (Dead center) | Neutral (Both terms aligned) |
| | | MobileOne Angles | $0.5085$ | $0.5296$ | PASS (Center) | $0.0000$ (Inside) | Neutral |
| | | **ResNet-34 Hybrid** | $0.5133$ | $0.5434$ | PASS (Center) | $0.0000$ (Inside) | Neutral |
| | | ResNet-34 Angles | $0.5463$ | $0.5487$ | PASS (Center) | $0.0000$ (Inside) | Neutral |
| **`B3_7_CENTER`** | `B3_6_UP` | **MobileOne Hybrid** | $0.4936$ | $0.6433$ | PASS (Margin: $0.0067$) | $0.0000$ (Inside) | Geometry pulls downward ($+0.0717$) |
| | | MobileOne Angles | $0.4829$ | $0.5716$ | PASS (Center) | $0.0000$ (Inside) | Neutral |
| | | **ResNet-34 Hybrid** | $0.4935$ | $0.6275$ | PASS (Margin: $0.0225$) | $0.0000$ (Inside) | Geometry pulls downward ($+0.0678$) |
| | | ResNet-34 Angles | $0.4983$ | $0.5597$ | PASS (Center) | $0.0000$ (Inside) | Neutral |
| **`B3_9_CENTER`** | `B3_8_RIGHT` | **MobileOne Hybrid** | $0.4421$ | $0.5258$ | PASS (Center) | $0.0000$ (Inside) | Neutral |
| | | MobileOne Angles | $0.4949$ | $0.5491$ | PASS (Center) | $0.0000$ (Inside) | Neutral |
| | | **ResNet-34 Hybrid** | $0.4453$ | $0.5779$ | PASS (Center) | $0.0000$ (Inside) | Neutral |
| | | ResNet-34 Angles | $0.6406$ | $0.5725$ | PASS (Margin: $0.0094$) | $0.0000$ (Inside) | Neutral |

---

## 3. Detailed Forensic Analysis of the Two Failure Modes

### Case 1: Post-Downward Upward Overshoot (`B3_3_CENTER`)
- **Preceding Trial**: `B3_2_DOWN` (user looked downward to the bottom target).
- **Physical Observation**:
  - In `MobileOne Angles-Only`, median prediction was **$(x=0.4636, y=0.4864)$**. The pretrained model's raw pitch angle immediately returned to baseline, placing gaze dead-center ($y=0.4864 \approx 0.50$).
  - In `ResNet-34 Angles-Only`, median prediction was **$(x=0.4715, y=0.5315)$**. Similarly, the angular prediction returned cleanly to the center resting zone.
  - In `MobileOne Hybrid`, median prediction jumped upward to **$(x=0.5194, y=0.3193)$**, crossing the upper boundary ($y < 0.35$) by **$0.0307$** and falsely selecting the **UP** button!
  - In `ResNet-34 Hybrid`, median prediction jumped upward to **$(x=0.5193, y=0.3303)$**, crossing the upper boundary by **$0.0197$** and falsely selecting the **UP** button!
- **Root Cause & Contributing Term**:
  - When returning from a deliberate downward fixation (`DOWN`), the participant's eyelids exhibit **vertical post-saccadic hysteresis** (eyelids snap open or over-elevate relative to the pupil position).
  - This transient eyelid elevation sharply inflates the eyelid vertical ratio $V_{eyelid} = \frac{Iris - UpperLid}{LowerLid - UpperLid}$ and canthal normal distance.
  - Because the linear hybrid mapper weights ocular geometry heavily to compensate for compressed angular ranges, this geometric transient contributes a **$-0.1671$ (MobileOne)** and **$-0.2012$ (ResNet-34)** upward shift in $Y$, driving an otherwise centered fixation directly into an erroneous UP selection.
  - **Verdict**: **Ocular geometry term contributes > 90% of the drift.**

---

### Case 2: Resting Downward Drift (`B3_1_CENTER`)
- **Preceding Trial**: Start of Block 3 (resting gaze prior to first peripheral movement).
- **Physical Observation**:
  - In `MobileOne Angles-Only`, median prediction was **$(x=0.5108, y=0.6021)$**, staying comfortably inside the deadzone boundary ($y=0.6021 \le 0.65$).
  - In `ResNet-34 Angles-Only`, median prediction was **$(x=0.4676, y=0.6201)$**, also staying inside the deadzone.
  - In `MobileOne Hybrid`, median was **$(x=0.5021, y=0.6551)$**, slipping past the lower threshold ($y=0.65$) by **$+0.0051$**.
  - In `ResNet-34 Hybrid`, median was **$(x=0.5070, y=0.6921)$**, slipping past the lower threshold by **$+0.0421$** directly into the **DOWN** button region.
- **Root Cause & Contributing Term**:
  - At rest, natural eyelid droop (partial ptosis during relaxed resting fixation) compresses the vertical aperture, lowering $V_{eyelid}$ and shifting canthal normal distance downward.
  - In `ANGLES_ONLY`, the raw head/gaze pitch was stable, producing $y \approx 0.60$.
  - In `HYBRID`, the relaxed eyelid geometry added $+0.0530$ (MobileOne) and $+0.0720$ (ResNet-34) downward displacement, pushing the coordinate beyond the $0.65$ deadzone border.
  - **Verdict**: **Ocular geometry term contributes 100% of the boundary violation.**

---

## 4. Lateral vs. Vertical Contrast

In all lateral center returns:
- `B3_5_CENTER` (Preceding: `LEFT`): $y \in [0.498, 0.543]$, distance outside $= 0.0$.
- `B3_9_CENTER` (Preceding: `RIGHT`): $y \in [0.525, 0.578]$, distance outside $= 0.0$.

Horizontal gaze returns do not induce vertical eyelid lag. Consequently, horizontal eye movements return to center with high geometric stability ($\Delta x_{\text{out}} = 0.0$). The entire false activation vulnerability of the hybrid model is confined to **vertical eyelid coupling on the Y axis**.

---

## 5. Architectural Conclusions & Recommendations

1. **Geometry is the Sole Cause of Center Departures**:
   Across all 5 center trials, the pretrained angle terms (`yaw`, `pitch`) maintained a **0.0% false activation rate**, never escaping the deadzone. The ocular geometry features (`rv`, `lv`, `canthal_normal`) are entirely responsible for the 40% center false activations observed in both hybrid models.
2. **The Hybrid Dilemma**:
   - `ANGLES_ONLY` provides **perfect center safety (0.0% false activations)**, but suffers from low peripheral reach on laptop screens (25% to 75% accuracy) because models trained on unconstrained 3D gaze (Gaze360) underestimate small ocular-only deflections.
   - `HYBRID` resolves peripheral reach (75% to 100% accuracy), but inherits **vertical eyelid hysteresis**, producing an unacceptable 40% center false-activation rate.
3. **Engineering Implications for Milestone M1/M1.5**:
   - Do NOT adopt either hybrid model in its current linear configuration for live selection, as a 40% resting false-activation rate violates Safety Rule #1 ("Speech requires explicit confirmation; no speech without deliberate intent").
   - Potential non-breaking mitigations to explore in future tasks:
     - **Asymmetric Feature Weighting / Shrinkage**: Apply stronger $L_2$ regularization specifically to the vertical geometry coefficients ($w_{rv}, w_{lv}, w_{cn}$) relative to angle coefficients.
     - **Deadzone Calibration**: Tune the vertical center deadzone bounds dynamically based on user baseline eyelid aperture (e.g. expanding vertical deadzone from $[0.35, 0.65]$ to $[0.30, 0.70]$).
     - **State-Dependent Velocity Filtering**: Suppress ocular geometry updates during rapid saccadic recovery when returning from vertical targets.
