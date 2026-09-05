# VoxGaze Accessibility UI Principles

The VoxGaze interface is an assistive communication device, not a generic dashboard. The eye itself is the input mechanism; therefore, the visual environment directly impacts motor control and eye fatigue.

---

## 1. Core Principles

### 1. Large Dwell-Free Targets
- Prioritize broad spatial quadrants/radial zones over pinpoint cursor targets.
- Users look toward general screen regions (UP, DOWN, LEFT, RIGHT) rather than aiming at tiny buttons.

### 2. Multi-Modal State Signals (Never Color Alone)
- Status and selection indicators must never rely on color alone.
- Every state must combine an unambiguous icon, clear text label, and high-contrast color:
  - `● TRACKING` (Active tracking)
  - `✓ CALIBRATED` (Profile loaded)
  - `! ABSTAIN` (Degraded tracking / pause)
  - `✕ NO_FACE` (Subject out of frame)

### 3. Persistent Candidate Preview
- The interface must persistently answer the user's implicit question: *"What does the system think I just selected?"*
- Active selection, intended phrase, and available escape action must remain prominently visible during `PREVIEW`.

### 4. Obvious and Immediate Cancel Path
- Every stage of navigation must have a single-step escape mechanism (e.g. look-away or hold center) to abort without triggering speech.

### 5. Reduced Animation & Motion
- Eliminate decorative transitions, parallax effects, or rapid movement that could trigger involuntary saccades or visual fatigue.
