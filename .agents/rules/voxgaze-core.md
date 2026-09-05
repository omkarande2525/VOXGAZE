---
trigger: always_on
---

# VoxGaze Core Engineering Rules

VoxGaze v1 is a deterministic, offline-first assistive communication prototype.

## Critical Path
```text
Webcam → MediaPipe Face Landmarker → Personalized Calibration → Normalized Gaze → Dual-Threshold Hysteresis → Directional State → Semantic FSM → Visual Preview → Explicit Blink Confirmation → Deterministic Intent Registry → Offline Piper TTS
```

## Non-Negotiable Invariants
1. **Never bypass the pipeline**: Candidate gaze states MUST NOT directly trigger speech.
2. **Intent is protocol-defined**: The eye provides physical control signals; the interaction state determines meaning.
3. **No LLM / No Cloud in v1**: Zero external network calls, zero cloud models, no voice cloning, and no autonomous medical dispatch in the critical loop.
4. **Abstain on ambiguity**: Degraded tracking or missing landmarks must yield `ABSTAIN` / `TRACKING_PAUSED`.
5. **Never optimize before measuring**: Always measure baseline performance first.
6. **No premature ML**: Rule-based geometry and hysteresis are the baseline. ML is added only when empirical measurements justify it.
7. **Scientific integrity**: Never fabricate measurements. Report event-level accuracy separately from frame-level stability, and record p50/p95 latencies.
