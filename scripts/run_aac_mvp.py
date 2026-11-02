"""
VoxGaze AAC MVP: Working Gaze-to-Word AAC Demonstration Runner

Launches the local FastAPI web server hosting the accessible AAC word board,
simulated gaze demo mode, live research diagnostic pipeline, offline speech synthesis,
and the experimental horizontal gaze-tree demo (Task D2b).
"""

import argparse
from pathlib import Path
import sys
import threading
import time
from typing import Optional
import webbrowser

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import uvicorn
from backend.app.aac.server import create_app


def main():
    parser = argparse.ArgumentParser(description="VoxGaze Accessible AAC MVP Demonstration Runner.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically launch web browser")
    parser.add_argument("--camera", type=int, default=0, help="Camera index for live mode (default: 0)")
    parser.add_argument(
        "--experimental-demo",
        action="store_true",
        default=False,
        help="Launch experimental horizontal gaze-tree demo mode (Task D2b)",
    )
    parser.add_argument(
        "--allow-keys",
        action="store_true",
        default=False,
        help="Allow arrow keys to inject LEFT/RIGHT commands for fallback plans B and C",
    )
    args = parser.parse_args()

    demo_runner = None

    if args.experimental_demo:
        print("\n" + "*" * 65)
        print("  VOXGAZE EXPERIMENTAL DEMO MODE: GAZE-TO-SPEECH (TASK D2b)  ")
        print("  STATUS: EXPERIMENTAL - NOT VALIDATED - GATE NOT PASSED     ")
        print("*" * 65)
        print("Starting 18-trial calibration sequence. Look at targets...")
        print("Press ESC in the calibration window to abort.\n")

        from backend.app.aac.demo_perception import LiveGazeWorker, run_calibration
        from backend.app.aac.demo_runner import DemoRunner
        from backend.app.aac.demo_tree import DemoStateMachine
        from backend.app.aac.speech import OfflineTTSEngine

        mapper, cap = run_calibration(camera_idx=args.camera)
        if mapper is None or not getattr(mapper, "is_fitted", False):
            print("\n[VoxGaze Demo] Calibration aborted or failed. Exiting without launching demo.")
            return

        print("\n[VoxGaze Demo] Calibration succeeded! Starting LiveGazeWorker and DemoRunner...")
        worker = LiveGazeWorker(cap=cap, mapper=mapper)
        fsm = DemoStateMachine()
        speech_engine = OfflineTTSEngine()

        demo_runner = DemoRunner(
            worker=worker,
            fsm=fsm,
            speech_engine=speech_engine,
            allow_keys=args.allow_keys,
            camera_idx=args.camera,
        )
        demo_runner.start()

        app = create_app(
            experimental_demo=True,
            allow_keys=args.allow_keys,
            demo_runner=demo_runner,
        )
        target_path = "/demo"
    else:
        app = create_app(experimental_demo=False, allow_keys=False)
        target_path = "/"

    url = f"http://{args.host}:{args.port}{target_path}"

    print("\n" + "=" * 65)
    print("      VOXGAZE AAC MVP: WORKING GAZE-TO-WORD DEMONSTRATION     ")
    print("=" * 65)
    print(f" Web Interface URL     : {url}")
    print(" Operating Modes       :")
    if args.experimental_demo:
        print("   * EXPERIMENTAL DEMO : Horizontal Gaze-Tree Active (/demo)")
        print(f"   * KEY NAVIGATION    : {'ENABLED (--allow-keys)' if args.allow_keys else 'DISABLED'}")
    else:
        print("   1. DEMO MODE        : Simulated gaze reticle via Keyboard / WASD")
        print("                         Zero webcam required; reliable presentation.")
        print("   2. LIVE RESEARCH    : Real webcam perception pipeline.")
        print("                         Live selection DISABLED (calibration gates not passed).")
    print(" Offline Speech (TTS)  : Local Windows SAPI / Piper offline synthesis.")
    print(" Privacy & Safety      : Zero cloud calls, zero raw frame storage.")
    print(" Status                : Research prototype — Not clinically validated.")
    print("=" * 65 + "\n")

    if not args.no_browser:
        def _open_tab():
            time.sleep(1.2)
            try:
                webbrowser.open(url)
            except Exception:
                pass
        threading.Thread(target=_open_tab, daemon=True).start()

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        if demo_runner is not None:
            print("\n[VoxGaze Demo] Shutting down demo runner and releasing camera...")
            demo_runner.stop()


if __name__ == "__main__":
    main()
