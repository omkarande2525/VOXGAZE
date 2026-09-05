# VoxGaze AI

**A personalized, variable-depth semantic gaze interaction prototype for low-cost assistive communication.**

---

## Overview
VoxGaze converts deliberate, coarse physical eye movements (LEFT, RIGHT, UP, DOWN) into structured communication intents rather than requiring character-by-character dwell typing. It decouples gaze measurement from semantic intent mapping to provide high-speed, low-fatigue assistive communication on commodity consumer webcams.

### Core Architectural Pipeline
```text
Webcam (30 FPS)
  ↓
MediaPipe Face Landmarker (Iris Mesh)
  ↓
Quality Gate (Abstain on loss)
  ↓
User Calibration (5-point + blink baseline)
  ↓
Normalized Gaze Features (H, V ratios + EAR)
  ↓
Dual-Threshold Hysteresis
  ↓
Directional State (CENTER / LEFT / RIGHT / UP / DOWN)
  ↓
Semantic FSM (Fast & deep path navigation)
  ↓
Visual Preview (Persistent state)
  ↓
Deliberate Confirmation Blink
  ↓
Deterministic Phrase Registry (Local JSON)
  ↓
Offline Piper TTS → Audio Output
```

---

## Key Invariants
- **No Speech Without Confirmation**: Candidate selections require a visual preview and explicit confirmation blink.
- **Protocol-Defined Intent**: The eye acts as a physical control signal; the interaction state machine assigns semantic meaning.
- **Zero Cloud / Zero LLM**: Fully offline, deterministic, auditable, and low-latency critical path.
- **Abstention on Ambiguity**: Degraded tracking transitions to `ABSTAIN` instead of guessing.

---

## Development Milestones
1. **M0**: Repository Governance, Data Contracts, Scaffolding *(Current)*
2. **M1**: Webcam & Iris Sensor Validation Gate (MediaPipe, Calibration, Hysteresis, Synthetic Tests)
3. **M1.5**: Minimal 4-Command Interaction Loop (Gaze -> Preview -> Confirm -> Piper TTS)
4. **M2**: Hierarchical Variable-Depth Intent Traversal
5. **M3**: Personalized Intent Space Configuration
6. **M4**: Scientific Research Evaluation vs. Dwell Baseline

---

## Quickstart

### Prerequisites
- Python 3.10+
- Webcam (720p/1080p @ 30 FPS)

### Setup
```bash
# 1. Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run M0 verification tests
pytest tests/unit/ -v
```

### Documentation Map
- [docs/acceptance-gates.md](file:///d:/VOXGAZE/docs/acceptance-gates.md): Milestone pass/fail gates.
- [docs/data-contracts.md](file:///d:/VOXGAZE/docs/data-contracts.md): Pydantic data schemas.
- [docs/non-goals.md](file:///d:/VOXGAZE/docs/non-goals.md): Frozen v1 boundaries.
- [docs/DECISIONS.md](file:///d:/VOXGAZE/docs/DECISIONS.md): Architectural decision records.
- [docs/architecture.md](file:///d:/VOXGAZE/docs/architecture.md): Detailed system architecture.
- [docs/research-plan.md](file:///d:/VOXGAZE/docs/research-plan.md): Scientific benchmark methodology.
- [docs/ui-principles.md](file:///d:/VOXGAZE/docs/ui-principles.md): Assistive UI accessibility design.
