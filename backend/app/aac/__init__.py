"""
VoxGaze AAC Package
Accessible AAC Word Board, Dwell State Machine, Offline Speech Synthesis, and Gaze Input.
"""

from backend.app.aac.board import AACBoard, PhraseBuilder, WordTile
from backend.app.aac.selection import DwellSelector, SelectionState
from backend.app.aac.speech import OfflineTTSEngine, TTSResult
from backend.app.aac.gaze_input import GazeInputProvider, GazeTelemetry, LiveGazeProvider, SimulatedGazeProvider

__all__ = [
    "AACBoard",
    "PhraseBuilder",
    "WordTile",
    "DwellSelector",
    "SelectionState",
    "OfflineTTSEngine",
    "TTSResult",
    "GazeInputProvider",
    "GazeTelemetry",
    "SimulatedGazeProvider",
    "LiveGazeProvider",
]
