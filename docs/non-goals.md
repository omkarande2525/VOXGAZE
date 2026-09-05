# VoxGaze Non-Goals (Frozen v1 Scope)

To prevent scope creep, maintain focus on the core human-computer interaction hypothesis, and guarantee safety, the following elements are explicitly **OUT OF SCOPE** for VoxGaze v1:

1. **No Large Language Models (LLMs) in the Loop**:
   - Do not use LLMs, cloud APIs, or complex generative models to predict or hallucinate what the user wants to say.
   - All phrases are deterministically resolved from the local intent registry.

2. **No Cloud Dependencies**:
   - The entire perception, calibration, state machine, and TTS pipeline must run locally and offline on the user's laptop.
   - No external network requests during active communication.

3. **No Voice Cloning / Historical Audio Mimicry**:
   - Pretrained, lightweight local Piper TTS models are used. Voice cloning is out of scope for v1.

4. **No Medical Diagnosis or Clinical Efficacy Claims**:
   - VoxGaze is an assistive communication research prototype.
   - Never describe the system as a clinically validated diagnostic or therapeutic device.

5. **No Autonomous Medical Emergency Dispatch**:
   - Emergency or assistance requests must follow the exact same preview and deliberate confirmation loop.
   - The system must never trigger external 911/emergency dispatch based on passive or unconfirmed eye behavior.

6. **No Full Computer-Control / Free-Form Character Typing**:
   - The goal is coarse, dwell-free semantic intent communication, not replacing an operating system mouse or full QWERTY keyboard.

7. **No Complex Neural Gaze Models in Early Milestones**:
   - Do not train custom appearance models or deep networks from scratch for v1.
   - MediaPipe Face Landmarker + user calibration + geometric hysteresis is the baseline. Machine learning is considered only as an optional post-baseline research extension.

8. **No Premature Smartphone / Mobile Porting**:
   - Target development hardware is a standard laptop with built-in webcam and desktop OS.
