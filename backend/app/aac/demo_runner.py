"""
VoxGaze AAC Demo: Real-Time Runner and Bridge (Task D2b)

Connects the LiveGazeWorker perception daemon (Task D2a) to the pure
DemoStateMachine hierarchy (Task D1) and triggers offline speech synthesis
upon confirmed SPEAK intents.

INVARIANTS & SAFETY:
1. Gating Independence: Never sets is_selection_enabled to True and never
   bypasses PermissionError on AACService.append_tile. Uses phrase buffer directly.
2. Dual-Threshold Hysteresis: Uses horizontal mapped x coordinates only
   (LEFT enter < 0.35, exit > 0.40; RIGHT enter > 0.65, exit < 0.60).
   Ignores vertical y coordinates entirely.
3. Safety Abstention: Face loss or tracking loss immediately maps to NONE.
4. Non-Blocking Speech: Speech synthesis executes inside a dedicated daemon thread.
   During speech output, gaze input is ignored and UI reports 'SPEAKING...'.
5. Privacy & Ephemeral Data: Zero webcam frames or video written to disk.
"""

from collections import deque
from dataclasses import dataclass
import logging
from pathlib import Path
import threading
import time
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Union

from backend.app.aac.demo_perception import GazeSample, LiveGazeWorker, classify_x_hysteresis
from backend.app.aac.demo_tree import DemoScreen, DemoStateMachine, TreeEvent
from backend.app.aac.speech import OfflineTTSEngine, TTSResult

logger = logging.getLogger(__name__)


class DirectionConverter:
    """
    Converts continuous normalized horizontal screen coordinate x to discrete
    LEFT, RIGHT, or CENTER using dual-threshold hysteresis.
    Vertical coordinate y is completely ignored.
    """

    def __init__(
        self,
        left_enter: float = 0.35,
        left_exit: float = 0.40,
        right_enter: float = 0.65,
        right_exit: float = 0.60,
    ):
        self.left_enter = left_enter
        self.left_exit = left_exit
        self.right_enter = right_enter
        self.right_exit = right_exit
        self.state: str = "CENTER"

    def process(self, x: Optional[float], valid: bool = True, face_detected: bool = True) -> str:
        """Processes a single frame measurement into a directional control token."""
        if not valid or not face_detected or x is None:
            self.state = "CENTER"
            return "NONE"

        # Apply hysteresis using the proven helper
        self.state = classify_x_hysteresis(x, self.state)
        return self.state


class DemoRunner:
    """
    Coordinates perception worker, directional hysteresis converter, binary tree
    state machine, and background speech synthesis loop.
    """

    def __init__(
        self,
        worker: Optional[LiveGazeWorker] = None,
        fsm: Optional[DemoStateMachine] = None,
        speech_engine: Optional[Any] = None,
        allow_keys: bool = False,
        camera_idx: int = 0,
    ):
        self.worker = worker
        self.fsm = fsm or DemoStateMachine()
        self.speech_engine = speech_engine or OfflineTTSEngine()
        self.allow_keys = allow_keys
        self.camera_idx = camera_idx

        self.converter = DirectionConverter()
        self.current_direction: str = "CENTER"
        self.speaking: bool = False
        self.last_speech_error: Optional[str] = None
        self.paused: bool = False

        self.measured_fps: float = 0.0
        self.last_processed_timestamp: float = 0.0

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._runner_thread: Optional[threading.Thread] = None
        self._speech_thread: Optional[threading.Thread] = None

        self._last_fps_print_t: float = 0.0
        self._last_event_idx: int = 0

    def start(self) -> None:
        """Starts the live gaze worker (if present) and runner thread."""
        if self.worker is not None:
            self.worker.start()

        if self._runner_thread is not None and self._runner_thread.is_alive():
            return

        self._stop_event.clear()
        self._last_fps_print_t = time.monotonic()

        # Print initial FPS message
        print(f"[VoxGaze Demo] Runner started. Loop rate: initializing...")

        self._runner_thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="VoxGazeDemoRunner",
        )
        self._runner_thread.start()

    def stop(self) -> None:
        """Cleanly halts runner thread and worker."""
        self._stop_event.set()
        if self._runner_thread is not None and self._runner_thread.is_alive():
            self._runner_thread.join(timeout=2.0)
        self._runner_thread = None

        if self.worker is not None:
            self.worker.stop()

    def _run_loop(self) -> None:
        """Main background loop polling worker samples and updating FSM."""
        while not self._stop_event.is_set():
            now_mono = time.monotonic()

            # Periodic 30s console rate printout
            if (now_mono - self._last_fps_print_t) >= 30.0:
                self._last_fps_print_t = now_mono
                fps_val = self.measured_fps
                lat_val = getattr(self.worker, "per_frame_latency", 0.0) if self.worker else 0.0
                print(f"[VoxGaze Demo] Status: {fps_val:.1f} fps, latency: {lat_val:.1f} ms")

            if self.worker is not None:
                sample: Optional[GazeSample] = self.worker.latest_sample
                if sample is not None and sample.timestamp != self.last_processed_timestamp:
                    self.last_processed_timestamp = sample.timestamp
                    self.measured_fps = self.worker.fps
                    self.process_sample(
                        timestamp=sample.timestamp,
                        x=sample.x,
                        y=sample.y,
                        valid=sample.valid,
                        face_detected=sample.face_detected,
                    )

            time.sleep(0.01)

    def process_sample(
        self,
        timestamp: float,
        x: Optional[float],
        y: Optional[float] = None,
        valid: bool = True,
        face_detected: bool = True,
    ) -> str:
        """
        Processes a single gaze sample.
        Converts x to direction with hysteresis (y is ignored).
        Feeds FSM unless speaking or paused.
        Triggers speech if speak_request is emitted.
        """
        with self._lock:
            # 1. While speaking, ignore gaze decisions
            if self.speaking:
                return "SPEAKING"

            # 2. While paused, ignore gaze input
            if self.paused:
                return "PAUSED"

            # 3. Convert x to direction token (y is ignored)
            direction = self.converter.process(x, valid=valid, face_detected=face_detected)
            self.current_direction = direction

            # 4. Feed state machine
            start_event_count = len(self.fsm.events)
            self.fsm.on_sample(timestamp, direction)

            # 5. Check if a speak_request event was fired
            new_events = self.fsm.events[start_event_count:]
            for ev in new_events:
                if ev.type == "speak_request":
                    phrase_to_speak = " ".join(self.fsm.phrase)
                    self._trigger_speech(phrase_to_speak)
                    break

            return direction

    def _trigger_speech(self, phrase: str) -> None:
        """Executes speech synthesis on a dedicated background daemon thread."""
        if not phrase.strip():
            return

        self.speaking = True
        self.last_speech_error = None

        def _speech_worker():
            try:
                if hasattr(self.speech_engine, "speak"):
                    result = self.speech_engine.speak(phrase)
                    if hasattr(result, "success") and not result.success:
                        self.last_speech_error = getattr(result, "error_message", "Speech synthesis failed")
            except Exception as e:
                self.last_speech_error = str(e)
            finally:
                self.speaking = False

        self._speech_thread = threading.Thread(
            target=_speech_worker,
            daemon=True,
            name="DemoSpeechWorker",
        )
        self._speech_thread.start()

    def inject_key(self, key: str) -> None:
        """
        Injects a keyboard arrow direction (for fallback Plans B and C).
        Guarded strictly by self.allow_keys.
        """
        if not self.allow_keys:
            raise PermissionError("Key simulation disabled without --allow-keys.")

        with self._lock:
            if self.speaking or self.paused:
                return

            t = time.perf_counter()
            d = key.upper().strip()
            if d == "LEFT":
                self.current_direction = "LEFT"
                self.fsm.on_sample(t, "LEFT")
            elif d == "RIGHT":
                self.current_direction = "RIGHT"
                self.fsm.on_sample(t, "RIGHT")
            elif d == "CENTER":
                self.current_direction = "CENTER"
                self.fsm.on_sample(t, "CENTER")

            # Check for speech request
            for ev in self.fsm.events[self._last_event_idx:]:
                if ev.type == "speak_request":
                    phrase_to_speak = " ".join(self.fsm.phrase)
                    self._trigger_speech(phrase_to_speak)
                    break
            self._last_event_idx = len(self.fsm.events)

    def reset(self) -> None:
        """Escape action: resets hierarchy back to root."""
        with self._lock:
            self.fsm.reset(time.perf_counter())
            self.current_direction = "CENTER"
            self.converter.state = "CENTER"

    def toggle_pause(self) -> bool:
        """Toggles pause state for gaze input."""
        with self._lock:
            self.paused = not self.paused
            self.fsm.paused = self.paused
            if self.paused:
                self.current_direction = "NONE"
                self.converter.state = "CENTER"
            return self.paused

    def recalibrate(self) -> None:
        """Triggers recalibration sequence if supported."""
        logger.info("Recalibration requested on DemoRunner")

    def get_state(self) -> Dict[str, Any]:
        """Generates real-time state payload for UI /api/demo/state."""
        with self._lock:
            scr = self.fsm.current_screen

            # Tracking health status
            if self.paused or self.fsm.paused:
                tracking_status = "paused"
            elif self.worker is not None:
                latest = self.worker.latest_sample
                if latest and latest.valid and latest.face_detected:
                    tracking_status = "ok"
                else:
                    tracking_status = "lost"
            else:
                tracking_status = "ok"

            return {
                "banner_text": "EXPERIMENTAL - NOT VALIDATED - GATE NOT PASSED",
                "screen": {
                    "title": scr.title,
                    "left_label": scr.left_label,
                    "right_label": scr.right_label,
                    "kind": scr.kind,
                    "candidate": scr.candidate,
                    "candidate_type": scr.candidate_type,
                },
                "detected_direction": self.current_direction,
                "hold": {
                    "direction": self.fsm.hold_dir,
                    "progress": round(self.fsm.hold_progress, 3),
                },
                "armed": self.fsm.armed,
                "tracking": tracking_status,
                "phrase": list(self.fsm.phrase),
                "speaking": self.speaking,
                "speech_error": self.last_speech_error,
                "measured_fps": round(self.measured_fps, 1),
            }
