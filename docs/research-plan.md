# VoxGaze Research & Evaluation Plan

## 1. Core Research Hypothesis
**Main Question**: Can a coarse, dwell-free gaze interaction protocol reduce communication selection time and physical/cognitive interaction burden compared with character-level dwell typing on standard commodity webcam hardware?

---

## 2. Measurement Policy & Scientific Integrity
- **No Fabricated Data**: Benchmark metrics must be measured on real hardware or reproducible synthetic datasets.
- **Event-Level vs. Frame-Level Accuracy**:
  - *Frame-Level Stability*: Percentage of individual video frames maintaining the classified state during sustained fixation.
  - *Event-Level Command Accuracy*: Binary success/failure of an intentional directional command trial (e.g. prompt "LOOK RIGHT" -> system transitions to `RIGHT`).
- **Latency Profiling**: Always record distribution metrics: mean, median (p50), and 95th percentile (p95).

---

## 3. Three-Tier Data Strategy
| Tier | Source | Role | Privacy & Scope Boundary |
| :--- | :--- | :--- | :--- |
| **Tier A (Reference)** | MPIIGaze, GazeCapture, Columbia Gaze | Benchmark generic perception robustness under illumination & pose variance | Perception reference only; does NOT contain AAC intent labels. |
| **Tier B (Personal Calibration)** | `data/local/*.user.json` | Normalize user eye geometry, eyelid baseline, and thresholds | Kept local; ignored by git. |
| **Tier C (Interaction Dataset)** | In-house controlled user trials | Measure WPM, task completion time, false activations, abort rate | Participant consent required; anonymized session IDs. |

---

## 4. Evaluation Tasks & Comparative Baselines
We benchmark VoxGaze against a standardized on-screen Character Dwell Keyboard:

| Task Class | Example Communication Target | Baseline (Dwell Keyboard) | VoxGaze Protocol |
| :--- | :--- | :--- | :--- |
| **Simple (Urgent)** | "I need water." | Dwell select letters: W-A-T-E-R | Fast path (1 direction + confirm) |
| **Medium (Daily)** | "Please turn on the light." | Dwell select 24 characters | Fast/deep path (2 selections + confirm) |
| **Complex (Medical)** | "My right leg is hurting. Please help." | Dwell select 36 characters | Deep path (3 selections + confirm) |

### Key Metrics:
1. **Task Completion Time (TCT)**: Time from task start to speech synthesis.
2. **Deliberate Actions**: Number of intentional physical eye movements required.
3. **False Activation Rate (FAR)**: Number of unintended triggers during 10-minute passive use (reading/talking).
4. **Abort / Cancellation Rate**: Number of initiated vs. cancelled previews.
5. **Subjective Fatigue**: 5-point NASA-TLX modified scale.
