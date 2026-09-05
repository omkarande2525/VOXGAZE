---
name: voxgaze-engineering
description: Engineering guidelines, architectural invariants, module boundaries, and testing discipline for VoxGaze development.
---

# VoxGaze Engineering Skill

Use this skill when modifying or developing code in the VoxGaze repository.

## Pre-Change Protocol
1. **Inspect Structure**: Verify module boundaries and check [AGENTS.md](file:///d:/VOXGAZE/AGENTS.md).
2. **Review Invariants**:
   - Zero LLM / Zero cloud in the loop.
   - Speech *always* requires preview + explicit confirmation blink.
   - Tracking ambiguity must trigger `ABSTAIN`.
   - Never replace deterministic solutions with ML without empirical justification.
   - Never optimize before measuring.
3. **Isolate Changes**: Implement the smallest correct change adhering to pure geometry separation.
4. **Test Before Claiming Success**: Run automated unit and synthetic regression suites.
5. **No Fabricated Benchmarks**: Always report real measurements (including p50 and p95 latency).

## Module Boundary Rules
- `geometry.py`: Pure math only. No OpenCV, no camera, no UI.
- `calibration/service.py`: Calibration data collection, MAD/median calculations, quality checks, profile serialization.
- `gaze_engine.py`: Takes features + calibration profile, runs hysteresis, outputs discrete gaze state.
- `fsm.py`: Manages state machine transitions.
- `registry.py`: Resolves deterministic phrases from `default_intents.json`.
- `tts_engine.py`: Invokes offline Piper TTS on validated text only.
