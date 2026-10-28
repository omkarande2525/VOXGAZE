"""
VoxGaze AAC MVP: Dwell & Confirmation Selection State Machine

Manages gaze dwell timing, candidate highlighting, visual preview state,
and safety pause/abort rules when tracking is degraded or pointer jumps.
"""

from enum import Enum
import math
import time
from typing import Optional, Tuple
from backend.app.aac.board import WordTile


class SelectionState(str, Enum):
    IDLE = "IDLE"
    DWELLING = "DWELLING"
    CANDIDATE_HIGHLIGHTED = "CANDIDATE_HIGHLIGHTED"
    CONFIRMED = "CONFIRMED"
    PAUSED_TRACKING_LOSS = "PAUSED_TRACKING_LOSS"


class DwellSelector:
    """
    Tracks dwell duration on AAC tiles and enforces safety invariants:
    1. Dwell pauses or aborts immediately on tracking loss or poor confidence.
    2. Large pointer jumps reset dwell to prevent accidental selection during saccades.
    3. Reaching dwell completion enters CANDIDATE_HIGHLIGHTED state.
    4. Adding a word requires explicit confirmation; dwell alone does not auto-commit speech.
    """

    def __init__(
        self,
        dwell_duration_s: float = 1.2,
        jump_threshold_norm: float = 0.20,
        min_confidence: float = 0.60,
    ):
        self.dwell_duration_s = max(0.5, dwell_duration_s)
        self.jump_threshold_norm = jump_threshold_norm
        self.min_confidence = min_confidence

        self.state = SelectionState.IDLE
        self.active_tile: Optional[WordTile] = None
        self.candidate_tile: Optional[WordTile] = None
        self.dwell_start_mono: Optional[float] = None
        self.last_pointer: Optional[Tuple[float, float]] = None
        self.dwell_progress: float = 0.0  # 0.0 to 1.0

    def update(
        self,
        current_tile: Optional[WordTile],
        pointer_norm: Optional[Tuple[float, float]],
        is_tracking_valid: bool = True,
        confidence: float = 1.0,
        now_mono: Optional[float] = None,
    ) -> SelectionState:
        """
        Advances dwell state given the current gaze pointer and tracking quality.
        Returns the updated SelectionState.
        """
        now = now_mono if now_mono is not None else time.monotonic()

        # 1. Quality gate: abort or pause dwell if tracking is degraded
        if not is_tracking_valid or confidence < self.min_confidence or pointer_norm is None:
            if self.state in (SelectionState.DWELLING, SelectionState.CANDIDATE_HIGHLIGHTED):
                self.state = SelectionState.PAUSED_TRACKING_LOSS
            self.dwell_progress = 0.0
            self.active_tile = None
            self.last_pointer = None
            return self.state

        px, py = pointer_norm
        if not (math.isfinite(px) and math.isfinite(py)):
            self.reset()
            return SelectionState.IDLE

        # 2. Check for sudden pointer jump / saccade
        if self.last_pointer is not None:
            dist = math.hypot(px - self.last_pointer[0], py - self.last_pointer[1])
            if dist > self.jump_threshold_norm:
                # Sudden jump resets dwell progress
                self.reset()
        self.last_pointer = (px, py)

        # 3. Tile target tracking
        if current_tile is None:
            # Gaze is not on any tile (e.g. margin or looking away)
            if self.state == SelectionState.DWELLING:
                self.reset()
            elif self.state == SelectionState.PAUSED_TRACKING_LOSS:
                self.reset()
            return self.state

        # Gaze is on a tile
        if self.active_tile is None or self.active_tile.id != current_tile.id:
            # Started hovering over a new tile
            self.active_tile = current_tile
            self.dwell_start_mono = now
            self.dwell_progress = 0.0
            self.state = SelectionState.DWELLING
            return self.state

        # Continuing dwell on the same active tile
        if self.state == SelectionState.DWELLING and self.dwell_start_mono is not None:
            elapsed = now - self.dwell_start_mono
            self.dwell_progress = min(1.0, elapsed / self.dwell_duration_s)

            if elapsed >= self.dwell_duration_s:
                self.state = SelectionState.CANDIDATE_HIGHLIGHTED
                self.candidate_tile = current_tile

        return self.state

    def confirm_candidate(self) -> Optional[WordTile]:
        """
        Explicitly confirms the highlighted candidate tile.
        Can only succeed if a candidate is actively highlighted.
        Returns the confirmed tile, or None.
        """
        if self.state != SelectionState.CANDIDATE_HIGHLIGHTED or self.candidate_tile is None:
            return None

        confirmed = self.candidate_tile
        self.state = SelectionState.CONFIRMED
        self.reset()
        return confirmed

    def cancel_candidate(self) -> None:
        """Cancels an active candidate selection without adding it."""
        self.reset()

    def reset(self) -> None:
        """Resets dwell tracking to IDLE state."""
        self.state = SelectionState.IDLE
        self.active_tile = None
        self.candidate_tile = None
        self.dwell_start_mono = None
        self.dwell_progress = 0.0
