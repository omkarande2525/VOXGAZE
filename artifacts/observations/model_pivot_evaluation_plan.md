# VoxGaze: Pivot to Pretrained Model-Based Gaze Estimation

**Date**: 2026-10-25  
**Milestone**: M1 / M1.5 (Sensor Validation & Perception Exploration)  
**Status**: Experimental Research Report & Controlled Evaluation Plan (Revision 2)  
**Authors**: Antigravity Pair Programming & VoxGaze Core Team  

---

## 1. Executive Summary & Problem Realignment

Initial VoxGaze milestones relied upon handcrafted geometric heuristics derived from MediaPipe facial mesh landmarks ($H_{ratio}$, $V_{ratio}$, and canthal-normal distance) with manual dual-threshold hysteresis. While this rule-based baseline requires zero external model weights, physical trials revealed structural limitations:
1. **Vertical Eyelid Occlusion**: Vertical iris movement moves eyelids along with the eyeball, compressing or inverting vertical ratios.
2. **Subject & Posture Bias**: Inter-subject morphological differences and webcam viewing angles introduce large static offsets in raw gaze angles (center yaw shifting from $-2.2^\circ$ to $-17.4^\circ$ across sessions).
3. **Threshold Tuning Saturation**: Fixed gates ($\Delta H \ge 0.06$, $\Delta V \ge 0.05$) fail on natural, comfortable eye movements, marking valid user gaze as unsupported.

To explore whether pretrained gaze estimation and personalized calibration can overcome these issues, we integrated the official `yakhyo/gaze-estimation` model family (MobileOne S0 and ResNet-34) and designed a regularized Ridge regression mapper.

### Core Insight from Revised Empirical Auditing
**Personalization alone does not guarantee reliability.** While MobileOne S0 executes with exceptional CPU efficiency ($\sim 7.3\text{ ms}$ inference $p50$, $\sim 78\text{ FPS}$), our retrospective evaluation demonstrates that:
- Frame-level rows cannot be treated as independent trials due to high temporal auto-correlation within 4-second fixations.
- Pretrained model angular output quality must be strictly separated from screen-coordinate calibration quality.
- Pretrained gaze models trained on unconstrained datasets (Gaze360) show compressed angular responses to small, ocular-only laptop screen fixations ($\Delta \text{Yaw} \approx 12^\circ - 14^\circ$), causing pure angle mappers to miss realistic AAC target bounding boxes unless combined with ocular geometry.
- Upward gaze (`UP`) remains universally compressed across models and geometry.
- Live selection gates must enforce **repeated-session trial-level sustained dwell** rather than single-session frame accuracy.

---

## 2. Technical Audit of Current Model Integration

### 2.1 Preprocessing Pipeline
MobileOne S0 and ResNet-34 share an identical preprocessing contract from `yakhyo/gaze-estimation`:
1. **Face Mesh Crop**: MediaPipe FaceLandmarker extracts 468 mesh points. In `backend/app/perception/face_crop.py`, a square bounding box centered on the face mesh is expanded by $20\%$ (`padding_scale = 1.20`), clamped to $[0, W] \times [0, H]$, and gated at $\ge 64 \times 64\text{ px}$.
2. **Color Space & Resizing**: BGR crop converted to RGB and bilinearly resized to $(448, 448)$ px.
3. **Standardization**: Scaled to $[0.0, 1.0]$ and standardized via ImageNet statistics:
   $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
4. **Tensor Layout**: Formatted as NCHW `float32` batch `(1, 3, 448, 448)`.

### 2.2 Output Decoding Formulation
Both models output two 90-bin classification logit vectors (`yaw`, `pitch`):
1. **Softmax Expectation**:
   $$p_i = \frac{\exp(z_i - \max(z))}{\sum_{j=0}^{89} \exp(z_j - \max(z))}, \quad \theta_{raw} = \sum_{i=0}^{89} (p_i \cdot i) \times 4.0^\circ$$
2. **Centering & Conversion**:
   $$\theta_{deg} = \theta_{raw} - 180.0^\circ, \quad \theta_{rad} = \text{deg2rad}(\theta_{deg})$$
   (Pitch positive = looking UP, yaw positive = looking RIGHT).

### 2.3 Integration State & Pipeline Placement
- **Telemetry Only**: In `backend/app/aac/gaze_input.py`, `LiveGazeProvider` streams `head_yaw_deg` and `head_pitch_deg` into telemetry, but sets `pointer_x = None`, `pointer_y = None`, and `is_selection_enabled = False`.
- **Zero Raw Image Leakage**: Pretrained models process in-memory crops during live capture; observation CSVs stream numerical features only. Zero raw webcam frames or face crops are persisted.

### 2.4 Reusable Components (Zero Modification Required)
- `backend/app/perception/face_crop.py`: MediaPipe-based crop extraction with clamping and minimum size gates.
- `backend/app/core/contracts.py`: Fundamental Pydantic contracts and safety state schemas.
- `backend/app/aac/`: AAC vocabulary board (11 core tiles), phrase composition buffer, dwell selector lifecycle, and offline Piper TTS.
- `backend/app/calibration/protocol.py`: Multi-phase calibration sequence (Phase 1 estimation, Phase 2 evaluation, Phase 3 head contrast).

---

## 3. Model-Comparison Interface & ResNet-34 Integration

We established a common abstraction isolating model backbones from the AAC FSM:
`BaseGazeEstimator` abstract base class $\to$ `MobileOneGazeEstimator` and `ResNet34GazeEstimator` returning standard `GazeModelOutput` contracts via `create_gaze_estimator()` in `backend/app/perception/model_registry.py`.

### 3.1 Model Checkpoint Specifications

| Specification | MobileOne S0 | ResNet-34 |
| :--- | :--- | :--- |
| **Model Checkpoint** | `mobileone_s0_gaze.onnx` | `resnet34_gaze.onnx` |
| **Provenance** | `yakhyo/gaze-estimation` | `yakhyo/gaze-estimation` |
| **License** | MIT License | MIT License |
| **File Size** | 4,974,521 bytes (~4.74 MB) | 85,491,644 bytes (~81.53 MB) |
| **SHA-256** | `e1d4b655a6...` | `c8e6b14f6095d2425241b9302aa663d9a23b7dfb9d43941352b718c91dc7f2cf` |
| **Input / Output** | `(1, 3, 448, 448)` $\to$ `[1, 90], [1, 90]` | `(1, 3, 448, 448)` $\to$ `[1, 90], [1, 90]` |
| **Provider** | ONNX Runtime `CPUExecutionProvider` | ONNX Runtime `CPUExecutionProvider` |

### 3.2 CPU Latency & Resource Profiling (50 Iterations on CPU)
| Metric | MobileOne S0 | ResNet-34 | Ratio (ResNet / MobileOne) |
| :--- | :--- | :--- | :--- |
| **Model Size** | 4.74 MB | 81.53 MB | **17.2x larger** |
| **Inference Latency (Mean)** | 7.37 ms | 70.42 ms | **9.5x slower** |
| **Inference Latency (p50)** | **7.31 ms** | **67.55 ms** | **9.2x slower** |
| **Inference Latency (p95)** | **8.05 ms** | **92.94 ms** | **11.5x slower** |
| **Preprocessing Latency (p50)** | 5.48 ms | 5.51 ms | 1.0x (identical) |
| **Total Frame Latency (p50)** | 12.85 ms | 74.07 ms | **5.8x slower** |
| **Estimated Throughput** | **~77.8 FPS** | **~13.5 FPS** | **-82.6% throughput** |

---

## 4. Addressing Core Evaluation Issues

### Issue 1: Evaluation Labels, Temporal Correlation & AAC Bounding Boxes
- **Temporal Auto-correlation**: A 4-second fixation trial produces $\sim 120$ frames. Treating each frame as an independent trial ($N=120$) severely inflates statistical confidence and conflates momentary microsaccadic noise with intentional target selection.
- **Trial-Level Command Decision Unit**: The evaluation framework now explicitly computes **trial-level metrics** by taking the median predicted coordinate $(\tilde{x}, \tilde{y})$ over post-settling frames of each independent trial.
- **Realistic AAC Bounding Boxes**: Rather than arbitrary Euclidean circular radiuses, selection accuracy is tested against rectangular UI target bounding boxes matching the assistive interface:
  $$\text{CENTER}: [0.35, 0.35, 0.65, 0.65], \quad \text{LEFT}: [0.00, 0.30, 0.35, 0.70]$$
  $$\text{RIGHT}: [0.65, 0.30, 1.00, 0.70], \quad \text{UP}: [0.30, 0.00, 0.70, 0.35], \quad \text{DOWN}: [0.30, 0.65, 0.70, 1.00]$$

### Issue 2: Rejection of Simultaneous Model Execution
- Running MobileOne S0 ($\sim 7\text{ ms}$) and ResNet-34 ($\sim 68\text{ ms}$) simultaneously on every video frame would require $\sim 75\text{ ms}$ processing time ($\sim 13\text{ FPS}$), dropping camera frames and disrupting MediaPipe mesh tracking.
- **Protocol Decision**: The next physical experiment will **not** execute simultaneous inference in the critical capture loop. Instead, captured frames will be evaluated **sequentially** on the same in-memory buffer, or across dedicated, alternating fixation blocks.

### Issue 3: Strict Separation of Model Quality (Layer 1) vs Calibration Quality (Layer 2)
We decompose evaluation into two distinct tiers:
1. **Layer 1: Pretrained Model Angular Metrics (Physical Truth)**:
   - Angular Separation: $\Delta \text{Yaw} = |\text{Yaw}_{RIGHT} - \text{Yaw}_{LEFT}|$, $\Delta \text{Pitch} = |\text{Pitch}_{DOWN} - \text{Pitch}_{UP}|$.
   - Fixation Jitter: Median Absolute Deviation ($\text{MAD}$) in yaw and pitch during stable gaze.
   - Center Baseline Drift: Shift in median $(\text{Yaw}, \text{Pitch})$ across repeated center return trials.
2. **Layer 2: Screen Calibration Mapping Metrics (Interface Control)**:
   - Trial-Level Selection Accuracy: % of independent peripheral trials where median gaze lands in the correct target bounding box.
   - Trial-Level Center False Activation Rate: % of center return trials where median gaze escapes the center resting box into an active button.
   - Frame-Level Hit Rate & Mean Target Error: Instantaneous jitter around the target.

---

## 5. Revised Empirical Results on Historical Physical Sessions

The offline evaluation script (`scripts/evaluate_observation_models.py`) processed 12 observation files (over 12,000 frames) with the revised two-layer evaluation architecture:

### 5.1 Layer 1: Pretrained Model Angular Quality (MobileOne S0)
| Session | Status / Condition | $\Delta \text{Yaw}$ Separation | $\Delta \text{Pitch}$ Separation | Fixation Jitter (Yaw MAD) | Fixation Jitter (Pitch MAD) | Center Baseline Drift |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0a3228** | Stable Posture, Clear Light | **$25.14^\circ$** | **$18.63^\circ$** | $2.4^\circ$ | $2.2^\circ$ | $\text{Yaw}: 3.59^\circ, \text{Pitch}: 10.23^\circ$ |
| **0a6b30** | Natural Posture, Today's Trial | **$12.61^\circ$** | **$10.28^\circ$** | $3.6^\circ$ | $2.7^\circ$ | $\text{Yaw}: 3.80^\circ, \text{Pitch}: 4.50^\circ$ |
| **0164a4** | Severe Head Drift & Posture Shift | **$10.65^\circ$** | **$4.75^\circ$** | $5.1^\circ$ | $4.2^\circ$ | $\text{Yaw}: 4.64^\circ, \text{Pitch}: 8.56^\circ$ |

### 5.2 Layer 2: Held-Out Screen Calibration Mapping Results

#### Session: `calib_v15_0a6b30` (Natural Posture, 2026-10-25)
| Feature Set | Trial Selection Accuracy (Peripheral Hit Rate) | Trial Center False Activation Rate | Frame Selection Accuracy | Frame Center False Activation Rate | Mean Target Error |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ANGLES_ONLY** | **0.0%** (0 / 4) | **0.0%** (0 / 5) | 7.5% | 0.8% | 0.1720 |
| **ANGLES_POLY** | **0.0%** (0 / 4) | **0.0%** (0 / 5) | 14.8% | 2.3% | 0.1709 |
| **GEOMETRY_ONLY** | **75.0%** (3 / 4) | **0.0%** (0 / 5) | 69.2% | 3.0% | **0.1454** |
| **HYBRID** | **75.0%** (3 / 4) | **0.0%** (0 / 5) | 68.4% | 3.5% | 0.1478 |

*Trial Breakdown on `0a6b30` (HYBRID)*:
- `EVAL_CENTER_1..5`: 5 of 5 center return trials stayed firmly in the center deadzone ($0.0\%$ false activation).
- `EVAL_LEFT`: Median $(x=0.281, y=0.604)$ $\to$ **PASS** (Inside Left button $[0.00, 0.35]$).
- `EVAL_RIGHT`: Median $(x=0.807, y=0.510)$ $\to$ **PASS** (Inside Right button $[0.65, 1.00]$).
- `EVAL_DOWN`: Median $(x=0.381, y=0.753)$ $\to$ **PASS** (Inside Down button $[0.65, 1.00]$).
- `EVAL_UP`: Median $(x=0.536, y=0.453)$ $\to$ **FAIL** (Upward vertical deflection insufficient to reach $[0.00, 0.35]$).

#### Session: `calib_v15_0a3228` (Stable Posture, 2026-10-22)
| Feature Set | Trial Selection Accuracy (Peripheral Hit Rate) | Trial Center False Activation Rate | Frame Selection Accuracy | Frame Center False Activation Rate | Mean Target Error |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ANGLES_ONLY** | **25.0%** (1 / 4) | **0.0%** (0 / 5) | 38.5% | 6.9% | 0.1569 |
| **ANGLES_POLY** | **50.0%** (2 / 4) | **0.0%** (0 / 5) | 51.7% | 21.6% | 0.1698 |
| **GEOMETRY_ONLY** | **75.0%** (3 / 4) | **0.0%** (0 / 5) | 70.0% | 23.3% | 0.1662 |
| **HYBRID** | **75.0%** (3 / 4) | **0.0%** (0 / 5) | 66.7% | 24.8% | **0.1632** |

#### Session: `calib_v15_0164a4` (Severe Drift Failure Case)
- All feature sets achieved $0.0\%$ trial selection accuracy, correctly triggering system-level `ABSTAIN`.

### Key Conclusions from Revised Evidence
1. **Pretrained model angles alone fail to reach peripheral AAC button bounds**: In `0a6b30`, `ANGLES_ONLY` achieved $0\%$ trial hit accuracy because raw angular separation ($\sim 12.6^\circ$) is compressed relative to the screen dimensions.
2. **Geometry and Hybrid reliably drive 3 of 4 directions**: Both `GEOMETRY_ONLY` and `HYBRID` achieve $75\%$ trial-level hit accuracy (`LEFT`, `RIGHT`, `DOWN`) and $0\%$ trial-level center false activations.
3. **Upward gaze (`UP`) is the persistent physiological blocker**: In all sessions and models, upward iris deflection is masked by eyelid elevation and camera pitch angles, failing to cross the upward threshold.
4. **Frame-level vs Trial-level reconciliation**: Frame-level center false rates of $\sim 20\%$ on `0a3228` were caused by single-frame instantaneous flutter; the sustained trial median never exited the center deadzone.

---

## 6. Comprehensive Architectural Comparison

| Architecture | Trial Selection Accuracy | Trial Center False Activation Rate | Layer 1 Angular Separation | CPU Inference Latency (p50) | Memory Footprint | Real-Time Viability (CPU) | Safety Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Handcrafted Baseline (Fixed Hysteresis)** | 55.6% – 85.7% (Fails Left/Right gates on `0a6b30`) | Variable (Fails on drift) | N/A (Geometric ratios) | **< 1.0 ms** | 0 MB | 100+ FPS | Word selection disabled |
| **MobileOne S0 + Personalized Hybrid** | **75.0%** (Passes Left, Right, Down on all clean runs) | **0.0%** (5/5 center returns held) | $\Delta \theta \approx 12^\circ - 25^\circ$ | **7.31 ms** | 4.74 MB | **~78 FPS** | Ready for Controlled Live Validation |
| **ResNet-34 + Personalized Hybrid** | *Requires live camera trial* | *Requires live camera trial* | Expected higher capacity | **67.55 ms** | 81.53 MB | **~13.5 FPS** (Sub-realtime) | Experimental Only (CPU bottleneck) |
| **Ensemble (MobileOne + ResNet-34)** | *Invalid without live data* | *Invalid without live data* | Redundant label space | **> 75 ms** | 86.3 MB | Infeasible (< 13 FPS) | Rejected for v1 |

---

## 7. Precise Safety Gate Definitions

Before live gaze-controlled word selection may ever be enabled, the system must empirically satisfy three non-negotiable gates:
1. **Held-Out Trial Selection Accuracy $\ge 85\%$**: Over at least 12 independent held-out peripheral trials across 4 directions, the trial-median gaze coordinate must fall within the designated button bounding box in $\ge 85\%$ of trials.
2. **Center-Return False Activation Rate $\le 5\%$**: Over at least 15 independent center return trials, the sustained trial-median gaze coordinate must remain within the center resting box ($[0.35, 0.35, 0.65, 0.65]$), with $\le 5\%$ false activations into actionable buttons.
3. **Cross-Session Generalization**: Gates 1 and 2 must pass across **at least two consecutive, independent sessions on separate days or camera repositionings**, proving portability beyond a single seated position.

---

## 8. Next Experiment: Controlled Sequential Comparison Protocol

Instead of an uncontrolled 27-condition test, the next physical trial will execute a **compact, sequential, 5-point comparison protocol**:

1. **Protocol Structure**:
   - 5 targets: `CENTER` (rest), `LEFT`, `RIGHT`, `UP`, `DOWN`.
   - 3 repeated blocks of fixations (3 seconds per target, 1.5 seconds settling), with minimal head movement.
   - Interleaved center returns between all peripheral targets.
2. **Sequential Model Execution (Zero Thread Starvation)**:
   - For each captured camera frame in memory, run MediaPipe face landmarker + crop extraction.
   - Run MobileOne S0 inference and record latency.
   - Run ResNet-34 inference sequentially on the same frame crop and record latency.
   - Zero raw frames or face crops written to disk (preserving Privacy Invariant #7).
3. **Decoupled Metric Logging**:
   - Log Layer 1 metrics (yaw, pitch, separation, jitter, drift) for both models simultaneously on identical physical photons.
   - Fit the identical `PersonalizedGazeMapper` on Block 1 & 2; evaluate held-out on Block 3.
4. **ResNet-34 Decision Rule**:
   - Adopt ResNet-34 only if it improves trial selection accuracy by $\ge 15\%$ over MobileOne S0. Otherwise, retain MobileOne S0 to preserve 60+ FPS CPU throughput and low battery/power consumption on assistive devices.
