"""
VoxGaze AAC MVP: FastAPI Server & Telemetry Router

Serves the accessible single-page AAC interface and exposes REST & WebSocket
endpoints for phrase building, gaze pointer tracking, dwell progress, and offline TTS.
"""

import asyncio
import json
from pathlib import Path
import time
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.app.aac.board import AACBoard, WordTile
from backend.app.aac.gaze_input import GazeTelemetry, LiveGazeProvider, SimulatedGazeProvider
from backend.app.aac.selection import DwellSelector, SelectionState
from backend.app.aac.speech import OfflineTTSEngine, TTSResult


class ModeRequest(BaseModel):
    mode: str = Field(..., pattern="^(DEMO|LIVE)$")


class MoveRequest(BaseModel):
    dx: Optional[float] = 0.0
    dy: Optional[float] = 0.0
    x: Optional[float] = None
    y: Optional[float] = None


class TileActionRequest(BaseModel):
    tile_id: str


class SpeakRequest(BaseModel):
    phrase: Optional[str] = None  # If None, speaks the current phrase from PhraseBuilder


class AACService:
    """
    Encapsulates all business logic for the AAC word board,
    operating modes, dwell selection, phrase management, and TTS.
    Fully testable independently of HTTP/ASGI layers.
    """

    def __init__(
        self,
        board: Optional[AACBoard] = None,
        selector: Optional[DwellSelector] = None,
        tts_engine: Optional[OfflineTTSEngine] = None,
        sim_provider: Optional[SimulatedGazeProvider] = None,
        live_provider: Optional[LiveGazeProvider] = None,
        active_mode: str = "DEMO",
    ):
        self.board = board or AACBoard()
        self.selector = selector or DwellSelector(dwell_duration_s=1.2)
        self.tts_engine = tts_engine or OfflineTTSEngine()
        self.sim_provider = sim_provider or SimulatedGazeProvider()
        self.live_provider = live_provider or LiveGazeProvider()
        self.active_mode = active_mode

    def get_board_tiles(self) -> Dict[str, Any]:
        """Returns the list of word tiles, categories, and normalized bounding boxes."""
        return {
            "tiles": [
                {
                    "id": t.id,
                    "text": t.text,
                    "label": t.label,
                    "icon": t.icon,
                    "category": t.category.value,
                    "bounds": {"x_min": t.x_min, "y_min": t.y_min, "x_max": t.x_max, "y_max": t.y_max},
                }
                for t in self.board.tiles
            ]
        }

    def get_status(self) -> Dict[str, Any]:
        """Returns system status, active mode, phrase, and current dwell state."""
        provider = self.sim_provider if self.active_mode == "DEMO" else self.live_provider
        telemetry = provider.get_telemetry()

        return {
            "mode": self.active_mode,
            "phrase": self.board.phrase_builder.phrase_text,
            "words": self.board.phrase_builder.words,
            "is_empty": self.board.phrase_builder.is_empty,
            "is_selection_enabled": telemetry.is_selection_enabled,
            "unsupported_reason": telemetry.unsupported_reason,
            "telemetry": {
                "timestamp_ms": telemetry.timestamp_ms,
                "pointer_x": telemetry.pointer_x,
                "pointer_y": telemetry.pointer_y,
                "is_tracking_valid": telemetry.is_tracking_valid,
                "confidence": telemetry.confidence,
                "status_label": telemetry.status_label,
                "is_simulated": telemetry.is_simulated,
                "h_ratio": telemetry.h_ratio,
                "v_ratio": telemetry.v_ratio,
                "ear": telemetry.ear,
                "head_yaw_deg": telemetry.head_yaw_deg,
                "head_pitch_deg": telemetry.head_pitch_deg,
            },
            "dwell": {
                "state": self.selector.state.value,
                "progress": round(self.selector.dwell_progress, 3),
                "active_tile": self.selector.active_tile.id if self.selector.active_tile else None,
                "candidate_tile": self.selector.candidate_tile.id if self.selector.candidate_tile else None,
            },
            "last_spoken": self.tts_engine.last_spoken_phrase,
        }

    def set_mode(self, mode: str) -> Dict[str, Any]:
        """Switches operating mode between DEMO and LIVE."""
        if mode not in ("DEMO", "LIVE"):
            raise ValueError(f"Invalid mode: {mode}")
        self.active_mode = mode
        self.selector.reset()
        if mode == "DEMO":
            self.sim_provider.reset()
        else:
            self.live_provider.reset()
        return {"status": "SUCCESS", "active_mode": self.active_mode}

    def append_tile(self, tile_id: str) -> Dict[str, Any]:
        """Appends a word tile to the phrase buffer."""
        # Enforce gate: Live selection is disabled if calibration gates have not passed
        if self.active_mode == "LIVE" and not self.live_provider.is_selection_enabled:
            raise PermissionError(self.live_provider.unsupported_reason or "Live word selection is disabled.")

        tile = self.board.get_tile_by_id(tile_id)
        if not tile:
            raise KeyError(f"Tile '{tile_id}' not found.")

        added = self.board.phrase_builder.add_word(tile.text)
        return {
            "success": added,
            "phrase": self.board.phrase_builder.phrase_text,
            "words": self.board.phrase_builder.words,
        }

    def backspace_phrase(self) -> Dict[str, Any]:
        """Removes the last word from the phrase (Undo)."""
        removed = self.board.phrase_builder.remove_last()
        return {
            "success": removed is not None,
            "removed": removed,
            "phrase": self.board.phrase_builder.phrase_text,
            "words": self.board.phrase_builder.words,
        }

    def clear_phrase(self) -> Dict[str, Any]:
        """Clears the entire phrase buffer."""
        self.board.phrase_builder.clear()
        self.selector.reset()
        return {
            "success": True,
            "phrase": "",
            "words": [],
        }

    def speak_phrase(self, phrase: Optional[str] = None) -> Dict[str, Any]:
        """
        Explicitly triggers speech synthesis for the assembled phrase.
        Never speaks automatically when words are highlighted or selected.
        """
        phrase_to_speak = phrase if phrase else self.board.phrase_builder.phrase_text
        if not phrase_to_speak:
            return {
                "success": False,
                "phrase": "",
                "engine_used": "NONE",
                "error_message": "EMPTY_PHRASE: No words in phrase to speak. Please select words first.",
            }

        result = self.tts_engine.speak(phrase_to_speak)
        return {
            "success": result.success,
            "phrase": result.phrase,
            "engine_used": result.engine_used,
            "error_message": result.error_message,
        }

    def demo_move(
        self,
        dx: Optional[float] = 0.0,
        dy: Optional[float] = 0.0,
        x: Optional[float] = None,
        y: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Moves simulated gaze reticle in Demo Mode and updates dwell."""
        if x is not None and y is not None:
            self.sim_provider.jump_to(x, y)
        elif dx is not None or dy is not None:
            self.sim_provider.move(dx or 0.0, dy or 0.0)

        # Hit-test against tiles
        curr_tile = self.board.find_tile_at(self.sim_provider.x, self.sim_provider.y)
        self.selector.update(
            current_tile=curr_tile,
            pointer_norm=(self.sim_provider.x, self.sim_provider.y),
            is_tracking_valid=True,
            confidence=1.0,
        )

        return {
            "pointer": {"x": self.sim_provider.x, "y": self.sim_provider.y},
            "hovered_tile": curr_tile.id if curr_tile else None,
            "dwell": {
                "state": self.selector.state.value,
                "progress": round(self.selector.dwell_progress, 3),
                "active_tile": self.selector.active_tile.id if self.selector.active_tile else None,
                "candidate_tile": self.selector.candidate_tile.id if self.selector.candidate_tile else None,
            },
        }

    def demo_confirm(self) -> Dict[str, Any]:
        """Explicitly confirms the active dwell candidate tile in Demo Mode."""
        confirmed_tile = self.selector.confirm_candidate()
        if confirmed_tile:
            self.board.phrase_builder.add_word(confirmed_tile.text)
            return {
                "success": True,
                "confirmed_tile": confirmed_tile.id,
                "phrase": self.board.phrase_builder.phrase_text,
                "words": self.board.phrase_builder.words,
            }
        return {"success": False, "message": "No candidate tile was highlighted for confirmation."}

    def get_calibration_status(self) -> Dict[str, Any]:
        """Returns honest calibration gate status loaded from empirical observation evidence."""
        gate_info = load_latest_calibration_gate_results()
        supported_dirs = [k for k in ("LEFT", "RIGHT", "UP", "DOWN", "BLINK") if gate_info.get(f"is_{k.lower()}_supported")]
        unsupported_dirs = [k for k in ("LEFT", "RIGHT", "UP", "DOWN", "BLINK") if not gate_info.get(f"is_{k.lower()}_supported")]

        if len(supported_dirs) == 5:
            msg = "All directional (LEFT, RIGHT, UP, DOWN) and blink gates passed empirical validation."
        elif "LEFT" in supported_dirs and "RIGHT" in supported_dirs and "BLINK" in supported_dirs:
            msg = (
                "Horizontal gaze gates (LEFT, RIGHT) and BLINK confirmation PASSED empirical validation. "
                "Vertical gates (UP, DOWN) remain unsupported."
            )
        elif supported_dirs:
            msg = f"Partially calibrated ({', '.join(supported_dirs)} passed). Unsupported: {', '.join(unsupported_dirs)}."
        else:
            msg = "Calibration directional separation gates have not passed. Live word activation remains disabled."

        v_metric = gate_info.get("vertical_metric", "eyelid")
        up_ok = gate_info.get("is_up_supported", False)
        down_ok = gate_info.get("is_down_supported", False)
        up_reason = (
            f"Passed (Delta V {'canthal' if v_metric == 'canthal_normal' else 'eyelid'} separation gate)"
            if up_ok else "Delta V separation gate not passed"
        )
        down_reason = (
            f"Passed (Delta V {'canthal' if v_metric == 'canthal_normal' else 'eyelid'} separation gate)"
            if down_ok else "Delta V separation gate not passed"
        )

        return {
            "evidence_source": gate_info.get("source_file"),
            "is_selection_enabled": self.live_provider.is_selection_enabled,
            "unsupported_reason": self.live_provider.unsupported_reason,
            "gates": {
                "LEFT": {
                    "supported": gate_info.get("is_left_supported", False),
                    "reason": "Passed (Delta H >= 0.06)" if gate_info.get("is_left_supported") else "Delta H separation gate < 0.06",
                },
                "RIGHT": {
                    "supported": gate_info.get("is_right_supported", False),
                    "reason": "Passed (Delta H >= 0.06)" if gate_info.get("is_right_supported") else "Delta H separation gate < 0.06",
                },
                "UP": {
                    "supported": up_ok,
                    "reason": up_reason,
                },
                "DOWN": {
                    "supported": down_ok,
                    "reason": down_reason,
                },
                "BLINK": {
                    "supported": gate_info.get("is_blink_supported", False),
                    "reason": "Passed (Delta EAR >= 0.05)" if gate_info.get("is_blink_supported") else "Delta EAR < 0.05",
                },
            },
            "unsupported_reasons": gate_info.get("unsupported_reasons", []),
            "message": msg,
            "frozen_thresholds": gate_info.get("frozen_thresholds", {}),
        }

    def start_calibration_session(self) -> Dict[str, Any]:
        """
        Initializes a structured 5-point calibration fixation sequence following
        the calibration protocol (TARGET_CENTER, TARGET_LEFT, TARGET_RIGHT, TARGET_UP, TARGET_DOWN).
        Uses warmup_sec=0.5s and collection_sec=1.5s per target.
        """
        targets = [
            {
                "id": "CENTER",
                "name": "CENTER",
                "x": 0.50,
                "y": 0.50,
                "instruction": "Look naturally at the CENTER target circle.",
                "settling_duration_ms": 500,
                "recording_duration_ms": 1500,
            },
            {
                "id": "LEFT",
                "name": "LEFT",
                "x": 0.15,
                "y": 0.50,
                "instruction": "Shift eyes to the LEFT circle without moving your head.",
                "settling_duration_ms": 500,
                "recording_duration_ms": 1500,
            },
            {
                "id": "RIGHT",
                "name": "RIGHT",
                "x": 0.85,
                "y": 0.50,
                "instruction": "Shift eyes to the RIGHT circle without moving your head.",
                "settling_duration_ms": 500,
                "recording_duration_ms": 1500,
            },
            {
                "id": "UP",
                "name": "UP",
                "x": 0.50,
                "y": 0.15,
                "instruction": "Shift eyes UP to the top circle without tilting your head.",
                "settling_duration_ms": 500,
                "recording_duration_ms": 1500,
            },
            {
                "id": "DOWN",
                "name": "DOWN",
                "x": 0.50,
                "y": 0.85,
                "instruction": "Shift eyes DOWN to the bottom circle without tilting your head.",
                "settling_duration_ms": 500,
                "recording_duration_ms": 1500,
            },
        ]
        self._active_calib_session = {
            "status": "RUNNING",
            "start_time_s": time.time(),
            "targets": targets,
            "current_target_index": 0,
        }
        return {
            "status": "STARTED",
            "targets": targets,
            "settling_ms": 500,
            "recording_ms": 1500,
            "total_targets": len(targets),
            "backend_perception_active": self.live_provider.is_tracking_valid,
            "instruction": "Follow each fixation target circle. Maintain head stability.",
        }

    def complete_calibration_session(self) -> Dict[str, Any]:
        """
        Completes the calibration session and reports honest gate status.
        Crucial Safety Invariant: Completing the UI sequence does NOT enable live word activation,
        because physical calibration directional separation gates have not passed.
        """
        self._active_calib_session = None
        status = self.get_calibration_status()
        supported_dirs = [k for k, v in status["gates"].items() if v.get("supported")]
        unsupported_dirs = [k for k, v in status["gates"].items() if not v.get("supported")]

        if len(supported_dirs) == 5:
            summary = "Calibration complete: All directional and blink gates PASSED."
        elif "LEFT" in supported_dirs and "RIGHT" in supported_dirs:
            summary = (
                "Calibration sequence finished. Horizontal gates (LEFT, RIGHT) and BLINK confirmation PASSED. "
                "Vertical gates (UP, DOWN) remain UNSUPPORTED. Horizontal navigation is available."
            )
        else:
            summary = (
                f"Calibration sequence finished. Supported: {', '.join(supported_dirs) if supported_dirs else 'None'}. "
                f"Unsupported: {', '.join(unsupported_dirs)}. Live word activation remains disabled."
            )

        return {
            "status": "COMPLETED",
            "is_selection_enabled": status["is_selection_enabled"],
            "gate_results": status["gates"],
            "unsupported_reasons": status["unsupported_reasons"],
            "summary_message": summary,
        }

    def abort_calibration_session(self) -> Dict[str, Any]:
        """Aborts any active calibration session and cleans up state."""
        self._active_calib_session = None
        return {"status": "ABORTED", "message": "Calibration session aborted."}


def load_latest_calibration_gate_results() -> Dict[str, Any]:
    """Helper loading the most recent calibration summary from artifacts/observations."""
    obs_dir = Path(__file__).resolve().parent.parent.parent.parent / "artifacts" / "observations"
    summaries = sorted(obs_dir.glob("calib_v15_*_summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if summaries:
        try:
            with open(summaries[0], "r", encoding="utf-8") as f:
                data = json.load(f)
            gate_res = data.get("phase1_calibration", {}).get("gate_results", {})
            return {
                "source_file": summaries[0].name,
                "is_left_supported": gate_res.get("is_left_supported", False),
                "is_right_supported": gate_res.get("is_right_supported", False),
                "is_up_supported": gate_res.get("is_up_supported", False),
                "is_down_supported": gate_res.get("is_down_supported", False),
                "is_blink_supported": gate_res.get("is_blink_supported", False),
                "vertical_metric": gate_res.get("vertical_metric", "eyelid"),
                "unsupported_reasons": gate_res.get("unsupported_reasons", []),
                "directional_separations": data.get("phase1_calibration", {}).get("directional_separations", {}),
                "frozen_thresholds": data.get("phase1_calibration", {}).get("frozen_thresholds", {}),
            }
        except Exception:
            pass
    return {
        "source_file": None,
        "is_left_supported": False,
        "is_right_supported": False,
        "is_up_supported": False,
        "is_down_supported": False,
        "is_blink_supported": False,
        "vertical_metric": "eyelid",
        "unsupported_reasons": ["No physical calibration session evaluated."],
        "directional_separations": {},
        "frozen_thresholds": {},
    }




def create_app(service: Optional[AACService] = None) -> FastAPI:
    """Factory creating the VoxGaze AAC FastAPI application."""
    app = FastAPI(
        title="VoxGaze AAC MVP",
        description="Working Gaze-to-Word AAC Demonstration Research Prototype",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    srv = service or AACService()
    app.state.service = srv

    # Connected WebSockets for real-time telemetry streaming
    active_connections = set()

    # -------------------------------------------------------------
    # REST API Routes
    # -------------------------------------------------------------

    @app.get("/api/board")
    def get_board_tiles() -> Dict[str, Any]:
        return srv.get_board_tiles()

    @app.get("/api/status")
    def get_status() -> Dict[str, Any]:
        return srv.get_status()

    @app.post("/api/mode")
    def set_mode(req: ModeRequest) -> Dict[str, Any]:
        return srv.set_mode(req.mode)

    @app.post("/api/phrase/append")
    def append_tile(req: TileActionRequest) -> Dict[str, Any]:
        try:
            return srv.append_tile(req.tile_id)
        except PermissionError as pe:
            raise HTTPException(status_code=403, detail=str(pe))
        except KeyError as ke:
            raise HTTPException(status_code=404, detail=str(ke))

    @app.post("/api/phrase/backspace")
    def backspace_phrase() -> Dict[str, Any]:
        return srv.backspace_phrase()

    @app.post("/api/phrase/clear")
    def clear_phrase() -> Dict[str, Any]:
        return srv.clear_phrase()

    @app.post("/api/speak")
    def speak_phrase(req: Optional[SpeakRequest] = None) -> Dict[str, Any]:
        phrase = req.phrase if req else None
        return srv.speak_phrase(phrase)

    @app.post("/api/demo/move")
    def demo_move(req: MoveRequest) -> Dict[str, Any]:
        return srv.demo_move(dx=req.dx, dy=req.dy, x=req.x, y=req.y)

    @app.post("/api/demo/confirm")
    def demo_confirm() -> Dict[str, Any]:
        return srv.demo_confirm()

    # -------------------------------------------------------------
    # Calibration Session & Gate Status Routes
    # -------------------------------------------------------------

    @app.get("/api/calibration/status")
    def get_calibration_status() -> Dict[str, Any]:
        return srv.get_calibration_status()

    @app.post("/api/calibration/session/start")
    def start_calibration_session() -> Dict[str, Any]:
        return srv.start_calibration_session()

    @app.post("/api/calibration/session/complete")
    def complete_calibration_session() -> Dict[str, Any]:
        return srv.complete_calibration_session()

    @app.post("/api/calibration/session/abort")
    def abort_calibration_session() -> Dict[str, Any]:
        return srv.abort_calibration_session()

    # -------------------------------------------------------------
    # WebSocket Streaming Telemetry
    # -------------------------------------------------------------

    @app.websocket("/ws/telemetry")
    async def websocket_telemetry(websocket: WebSocket):
        await websocket.accept()
        active_connections.add(websocket)
        try:
            while True:
                status_data = srv.get_status()
                await websocket.send_text(json.dumps(status_data))
                await asyncio.sleep(0.04)  # ~25 Hz update rate
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            active_connections.discard(websocket)

    # -------------------------------------------------------------
    # Static Assets & Web UI
    # -------------------------------------------------------------
    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/", response_class=HTMLResponse)
        def serve_index():
            index_path = static_dir / "index.html"
            if index_path.is_file():
                return FileResponse(index_path)
            return HTMLResponse("<h3>VoxGaze AAC UI loading...</h3>")

    return app
