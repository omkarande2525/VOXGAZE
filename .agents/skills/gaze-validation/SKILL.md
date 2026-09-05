---
name: gaze-validation
description: Standardized procedures for camera qualification, 5-point calibration, hysteresis validation, and synthetic regression testing.
---

# Gaze Validation Skill

Use this skill to validate camera hardware, iris tracking stability, and gaze classification accuracy.

## Validation Workflow

### 1. Camera Capability Audit
- Query and record requested vs. actual resolution and FPS.
- Verify frame acquisition latency and hardware backend.

### 2. Landmark & Iris Extraction
- Confirm MediaPipe Face Landmarker runs in `RunningMode.VIDEO`.
- Verify monotonically increasing timestamps: `int(time.perf_counter() * 1000)`.
- Confirm iris landmarks (indices 468-477) are resolved.

### 3. Statistically Robust 5-Point Calibration
- Protocol: `CENTER`, `LEFT`, `RIGHT`, `UP`, `DOWN`, `CLOSED`.
- 0.5s warm-up + 1.5s collection window per target.
- Calculate Median, MAD (Median Absolute Deviation), and min/max spread.
- Reject calibration if sample count $< 30$ or MAD exceeds stability threshold.
- Save reproducible profile with versioning and quality metrics.

### 4. Synthetic Regression Verification
- Run synthetic suite (`scripts/generate_synthetic_gaze.py`) covering:
  - Clean directional saccades.
  - Gaussian jitter / tremor.
  - Slow head drift.
  - Boundary threshold oscillation.
  - Tracking dropouts / missing frames.
  - Natural blinks ($<250\text{ ms}$) vs deliberate closures ($300-800\text{ ms}$).

### 5. Empirical Benchmark Report
- Generate machine-readable reports under `artifacts/m1/`:
  - `camera_report.json`
  - `calibration_report.json`
  - `direction_confusion_matrix.csv`
  - `latency_summary.json` (mean, p50, p95).
