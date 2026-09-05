---
name: accessibility-ui
description: Design principles and verification guidelines for assistive, gaze-controlled user interfaces.
---

# Accessibility UI Skill

Use this skill when designing, building, or reviewing UI components for VoxGaze.

## Key Design Constraints

### 1. Spatial Layout & Dwell-Free Interaction
- Large directional targets organized into distinct quadrants/radial sectors.
- Never require pinpoint cursor targeting.

### 2. Multi-Modal State Feedback
- Never use color as the sole indicator of state.
- Always combine: **Icon + Text Label + High Contrast Color**.
  - `● TRACKING`
  - `✓ CALIBRATED`
  - `! ABSTAIN`
  - `✕ NO_FACE`

### 3. Persistent Preview
- Display current candidate intent, synthesized phrase preview, and confirmation/cancellation instructions prominently.
- Ensure the user always clearly sees what action is staged.

### 4. Obvious Cancel Mechanism
- Immediate one-action abort path (look-away or hold center) at every stage of navigation.

### 5. Motion Reduction
- Zero extraneous animations, parallax, or rapid movement. Preserve visual stability.
