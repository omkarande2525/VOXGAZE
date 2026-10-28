# VoxGaze Task 5C: Controlled Live Run 2 Implementation Report

## 1. Executive Summary

Task 5C implements the refined **Run 2 Controlled Observation Protocol** designed in `artifacts/observations/task5b_validation.md`. 
The implementation adds a fully versioned, auto-timed measurement mode (`--measure-v2`) to the live diagnostic harness while strictly preserving backward compatibility for the Run 1 manual protocol (`--measure`).

All architectural and safety invariants from VoxGaze Milestone M1 remain intact:
- **Diagnostic isolation**: Gaze estimator and measurement routines remain completely decoupled from the AAC runtime, FSM, and speech pipeline.
- **Privacy invariant**: Only bounded scalar features are written to disk; zero raw webcam frames, face crops, or videos are recorded.
- **Non-destructive storage**: Unique timestamped filenames guarantee no existing observation files are overwritten.
- **Physical validation honesty**: Automated tests confirm headless correctness; physical webcam validation is deferred to the user.

---

## 2. Implemented Architecture & Protocol Design

### 2.1 Protocol Versioning & Backward Compatibility
- **Run 1 Mode (`--measure`)**: Preserved without modification. Uses manual `[SPACE]` or `[N]` start/stop transitions, manual durations, and the canonical 25-column Run 1 CSV schema.
- **Run 2 Mode (`--measure-v2`)**: Added as a separate, mutually-exclusive CLI flag. Enforces automatic phase timing, visual fixation targets, bilateral canthal ratios, and the extended 27-column Run 2 CSV schema.
- Running both flags simultaneously triggers an immediate mutual-exclusion error:
  `parser.error("--measure and --measure-v2 are mutually exclusive. Use one or the other.")`

### 2.2 Auto-Timed State Machine (`ObservationSessionV2`)
Each condition follows a strict four-phase deterministic lifecycle:
```text
WAITING ──[SPACE]──> SETTLING (2.0s) ──[Auto]──> RECORDING (5.0s) ──[Auto]──> DONE ──> Next Condition
                        │                            │
                     (Frames NOT logged)       (Frames logged to CSV)
```
- **Settling Phase (`V2_SETTLING_DURATION_S = 2.0s`)**:
  - Displays a countdown progress bar on the HUD with green background and text: `SETTLING Xs [HOLD POSITION]`.
  - Frames arriving during settling are counted for diagnostic telemetry (`frames_during_settling`) but are **strictly excluded** from condition samples and CSV logging (`record_frame()` returns `None`).
- **Recording Phase (`V2_RECORDING_DURATION_S = 5.0s`)**:
  - Automatically triggered when monotonic settling time reaches 2.0s.
  - Displays a recording progress bar with red accent and countdown: `REC Xs (N frames)`.
  - Streams numerical features frame-by-frame to the CSV log.
  - Automatically terminates at 5.0s elapsed recording time, finalizing the condition and advancing to `WAITING` for the next condition.
- **Interactive Controls**:
  - `[SPACE]`: Starts settling for the current waiting condition.
  - `[S]`: Skips the current condition (marks condition state as `"SKIPPED"` and advances immediately).
  - `[A]`: Safely aborts the entire session (marks status as `"ABORTED"`, preserves all streamed CSV rows, and writes the JSON summary).
  - `[Q]` / `ESC`: Exits the diagnostic immediately, safely finalizing and closing any active session.

### 2.3 Visual Fixation Targets (`V2_FIXATION_TARGETS`)
High-contrast crosshairs with central target circles are rendered on the live HUD canvas to provide standardized visual reference points:
| Condition Category | Condition Names | Normalized Position `(x, y)` | Description |
| :--- | :--- | :--- | :--- |
| **Center Baseline** | `CENTER_BASELINE`, `FINAL_CENTER_BASELINE`, all `*CENTER_RETURN` | `(0.50, 0.50)` | Exact screen center |
| **Gaze Left** | `EYES_LEFT` | `(0.08, 0.50)` | Horizontal left edge |
| **Gaze Right** | `EYES_RIGHT` | `(0.92, 0.50)` | Horizontal right edge |
| **Gaze Up** | `EYES_UP` | `(0.50, 0.08)` | Top vertical edge |
| **Gaze Down** | `EYES_DOWN` | `(0.50, 0.92)` | Bottom vertical edge |
| **Head Movement** | `HEAD_LEFT_*`, `HEAD_RIGHT_*`, `HEAD_UP_*`, `HEAD_DOWN_*` | `(0.50, 0.50)` | Fixation remains on center while head turns |
| **Unconstrained** | `COMBINED_MOVEMENT`, `TRACKING_RECOVERY` | None | No fixation target rendered |

**Dynamic Color States**:
- `WAITING`: Dim white crosshair (`(180, 180, 180)`, thickness 1)
- `SETTLING`: High-visibility yellow crosshair (`(0, 220, 255)`, thickness 2)
- `RECORDING`: High-visibility bright green crosshair (`(0, 255, 0)`, thickness 3)

### 2.4 Bilateral Normalized Canthal Projections
The Run 2 CSV schema (`CSV_HEADER_V2`) adds normalized bilateral canthal ratios:
- `right_eye_canthal_ratio`: Normalized perpendicular iris displacement from the right canthal axis (landmarks 33 & 133), divided by right eye width ($\|\vec{v}_{\text{canthal}}\|$).
- `left_eye_canthal_ratio`: Normalized perpendicular iris displacement from the left canthal axis (landmarks 263 & 362), divided by left eye width.
- `canthal_normal_dist_px`: Retained as the raw un-normalized pixel distance for direct empirical comparison.

**Sign Convention & Invariants**:
- Both anatomical eyes follow the invariant: $V_{\text{canthal}} < 0.0$ for **UP**, $0.0$ for **ON AXIS**, and $> 0.0$ for **DOWN**.
- Canthal ligaments provide fixed skeletal anchors that avoid eyelid movement artifacts.

### 2.5 Standardized Tracking Recovery Protocol
Condition 19 instructions standardized:
- Label: `TRACKING_RECOVERY`
- Instruction: `"Cover camera or look away briefly, then return to center."`
- Invariant: Never forces uncomfortable or extreme neck/eye strain.
- User can comfortably skip `[S]` or abort `[A]` at any time without data loss.

### 2.6 Extended Run 2 CSV Schema (27 Columns)
```text
 1. session_id
 2. monotonic_time_s
 3. elapsed_session_s
 4. condition_index
 5. condition_name
 6. frame_index
 7. face_detected
 8. face_crop_valid
 9. geometry_valid
10. right_eye_h_ratio
11. right_eye_v_ratio
12. right_eye_ear
13. left_eye_h_ratio
14. left_eye_v_ratio
15. left_eye_ear
16. canthal_normal_dist_px
17. right_eye_canthal_ratio   [NEW in Run 2]
18. left_eye_canthal_ratio    [NEW in Run 2]
19. mobileone_valid
20. mobileone_yaw_deg
21. mobileone_pitch_deg
22. mediapipe_latency_ms
23. crop_latency_ms
24. mobileone_preprocess_ms
25. mobileone_infer_ms
26. total_frame_latency_ms
27. unavailable_reason
```

---

## 3. Automated Test Suite Verification

All deterministic tests pass in the project environment (`.venv` on Windows):

### 3.1 Unit Test Suite (`tests/unit/test_structured_observation.py`)
18 passing tests covering:
- `test_observation_conditions_sequence_integrity`: 20 required conditions in canonical order.
- `test_measurement_schema_construction_and_csv_formatting`: Empty string missing values, correct column formatting.
- `test_unique_output_path_non_destructive`: Guaranteed unique timestamps + UUID collision prevention.
- `test_observation_session_lifecycle_and_streaming`: Run 1 start, record, skip, and finalize.
- `test_aborted_session_preserves_data`: Partial session abort safely flushes rows and saves summary.
- `test_summary_calculations_on_synthetic_records`: Robust stats (median, IQR, availability, latency).
- `test_circular_and_distribution_stats_edge_cases`: Boundary values, NaN/None handling, circular mean/variance.
- `test_offline_analysis_tool_logic`: Offline analysis aggregation.
- `test_v2_csv_header_has_bilateral_canthal_columns`: 27-column Run 2 header verification.
- `test_v2_frame_record_serialization_with_canthal_ratios`: Serialization with bilateral canthal ratios.
- `test_v2_session_auto_timed_settling_and_recording`: WAITING → SETTLING (2s) → RECORDING (5s) → DONE phase transitions.
- `test_v2_session_skip_and_abort`: Interactive skip and safe abort handling.
- `test_v2_session_csv_output_has_correct_header`: Correct CSV header on disk.
- `test_v2_session_summary_contains_timing_metadata`: Per-condition settling and recording timing metrics in JSON.
- `test_v2_fixation_targets_cover_all_condition_names`: Target coordinate definitions for all condition names.
- `test_v2_settling_and_recording_remaining_timers`: Countdown calculation correctness.
- `test_v2_constants`: Constant values match specification (2.0s settling, 5.0s recording).
- `test_headless_run2_diagnostic_execution`: End-to-end synthetic headless dry run producing valid CSV and JSON summary.

### 3.2 Full Repository Test Suite
```text
============================= 79 passed in 3.35s ==============================
```
Zero regressions across all existing modules (calibration, geometry, contracts, FSM, and MobileOne estimator).

---

## 4. Run 2 Launch Command & Manual Validation Checklist

### 4.1 Launch Command
Run the live diagnostic with the Run 2 protocol:
```powershell
.venv\Scripts\python scripts\mobileone_live_diagnostic.py --measure-v2
```

### 4.2 Manual Webcam Validation Checklist (For User Execution)

Before starting:
- [ ] Ensure room lighting is even (no strong backlighting or glare on glasses).
- [ ] Position camera at eye level, roughly 50–60 cm from your face.

During the session:
1. **Launch**: Verify the diagnostic window opens showing the live camera feed, side panels, and center guidance card.
2. **Readiness Check**: Verify top bar displays `[READY] -- COND 1/20: CENTER_BASELINE`.
3. **Condition 1 (Center Baseline)**:
   - Look at the white crosshair in the center of the screen.
   - Press `[SPACE]`.
   - Verify header turns green: `[SETTLING 2.0s - HOLD POSITION]`.
   - Hold steady. After 2.0s, verify header turns red: `[RECORDING 5.0s]`.
   - Keep looking at center crosshair. After 5.0s, verify recording auto-stops and condition advances to Condition 2 (`EYES_LEFT`).
4. **Conditions 2–9 (Directional Eye Gaze)**:
   - Notice the fixation crosshair moves to the requested edge (left, right, up, down).
   - Look at the target crosshair, press `[SPACE]`.
   - Hold gaze during the 2s settling countdown and 5s recording.
   - Verify returning to center for each `*_RETURN` condition.
5. **Conditions 10–17 (Head Movements)**:
   - Notice the fixation crosshair remains at screen center.
   - Turn or tilt your head as instructed while keeping eyes focused on the center target.
   - Press `[SPACE]`, hold during settling and recording.
6. **Condition 18 (Combined Movement)**:
   - Move head and eyes naturally, press `[SPACE]`, hold until auto-stop.
7. **Condition 19 (Tracking Recovery)**:
   - Cover camera lens with your hand or turn face away briefly, then face camera again.
   - Press `[SPACE]` to log recovery baseline.
   - *(Optional)* Press `[S]` if you prefer to skip this step.
8. **Condition 20 (Final Center Baseline)**:
   - Look naturally at center crosshair, press `[SPACE]`, hold 7s total.
9. **Session Finalization**:
   - Verify terminal prints the formatted `VOXGAZE V2 MEASUREMENT SESSION SUMMARY`.
   - Confirm unique output files exist in `artifacts/observations/obs_v2_*.csv` and `*_summary.json`.

---

## 5. Explicit Audit of Requirements Status

| Requirement | Implementation Status | Verification Status |
| :--- | :--- | :--- |
| **1. Backward compatibility** | Implemented (`--measure` preserved, `--measure-v2` added) | **Verified** by unit & headless tests |
| **2. 2s settling + 5s recording** | Implemented (`V2_SETTLING_DURATION_S=2.0`, `V2_RECORDING_DURATION_S=5.0`) | **Verified** by automated tests (settling frames excluded from CSV) |
| **3. HUD status & countdown** | Implemented in `draw_hud_v2` (progress bars, status, key hints) | **Verified** by headless HUD tests |
| **4. High-contrast fixation targets** | Implemented in `draw_fixation_target` & `V2_FIXATION_TARGETS` | **Verified** by target mapping tests |
| **5. Bilateral normalized canthal ratios** | Implemented in `CSV_HEADER_V2` & `FrameObservationRecordV2` | **Verified** by geometry tests & schema tests |
| **6. Comfortable recovery protocol** | Implemented (non-straining instructions, skip `[S]`, abort `[A]`) | **Verified** in code & state machine tests |
| **7. Start/end timing & latency telemetry** | Implemented in `ObservationSessionV2._finalize` & summary JSON | **Verified** in summary JSON tests |
| **8. Privacy invariant** | Implemented (scalars only, no image disk writes) | **Verified** by repository audit |
| **9. Non-destructive output path** | Implemented via `get_unique_output_path` | **Verified** by collision tests |
| **Physical webcam live validation** | Code implemented and ready for execution | **Unverified (Pending user live webcam run)** |
