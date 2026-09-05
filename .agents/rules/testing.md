---
trigger: always_on
---

# VoxGaze Testing Rules

Every meaningful implementation change must include automated tests.

## Mandatory Test Categories
1. **Pure geometry unit tests**: $H, V$ eye axis normalization, $EAR$ calculation, and bilateral contracts with zero external dependencies.
2. **Robust calibration tests**: 0.5s warm-up + 1.5s window filtering, Median + MAD quality evaluation, and outlier rejection.
3. **Dual-threshold hysteresis tests**: Verification that boundary jitter does not cause rapid oscillation between states.
4. **State machine transition tests**: Invariant checks ensuring `IDLE -> PREVIEW -> CONFIRMED -> SPEAKING` order and zero unconfirmed speech transitions.
5. **Synthetic edge-case tests**: Verification against Gaussian jitter, slow head drift, tremor, and dropouts.

## Metric Reporting Standards
- Always report **event-level command accuracy** separately from **frame-level stability**.
- Report distribution latencies: mean, median (p50), and 95th percentile (p95).
- Report false activations per session/hour during passive testing.
