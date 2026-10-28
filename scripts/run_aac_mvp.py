"""
VoxGaze AAC MVP: Working Gaze-to-Word AAC Demonstration Runner

Launches the local FastAPI web server hosting the accessible AAC word board,
simulated gaze demo mode, live research diagnostic pipeline, and offline speech synthesis.
"""

import argparse
from pathlib import Path
import sys
import threading
import time
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
    args = parser.parse_args()

    app = create_app()

    url = f"http://{args.host}:{args.port}"

    print("\n" + "=" * 65)
    print("      VOXGAZE AAC MVP: WORKING GAZE-TO-WORD DEMONSTRATION     ")
    print("=" * 65)
    print(f" Web Interface URL     : {url}")
    print(" Operating Modes       :")
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

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
