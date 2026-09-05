---
trigger: always_on
---

# VoxGaze Safety Rules

1. **Speech requires explicit confirmation**: No speech may ever be synthesized without a valid visual preview and explicit confirmation blink ($300-800\text{ ms}$).
2. **Visual preview before speech**: The selected candidate phrase must be displayed on screen prior to commitment.
3. **Cancel path mandatory**: An obvious, single-step escape path (look-away or hold center) must be available at every state to abort without speech.
4. **Tracking failure must produce ABSTAIN**: Face loss, poor lighting, or landmark degradation must immediately pause interaction.
5. **No autonomous medical diagnosis**: VoxGaze is an assistive communication research prototype, not a diagnostic or therapeutic medical device.
6. **No autonomous emergency dispatch**: Emergency phrases must follow the exact same preview and confirmation loop.
7. **Privacy first**: Never record or store raw webcam frames by default without explicit user opt-in.
8. **Local data protection**: User-specific calibration profiles and sessions must remain local and ignored by Git.
9. **Never claim clinical validation**: The system must never be presented as clinically validated without formal regulatory trials.
