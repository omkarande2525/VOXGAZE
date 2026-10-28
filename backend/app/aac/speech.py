"""
VoxGaze AAC MVP: Offline Text-to-Speech (TTS) Engine

Provides 100% offline local speech synthesis using local Piper TTS
(if configured) or Windows SAPI desktop synthesizer fallback.
Zero cloud dependencies, zero external network calls.
"""

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from typing import Optional, Tuple


@dataclass(frozen=True)
class TTSResult:
    """Outcome of a speech synthesis request."""
    success: bool
    phrase: str
    engine_used: str
    error_message: Optional[str] = None


class OfflineTTSEngine:
    """
    Offline Text-to-Speech Engine with fallback hierarchy:
    1. Local Piper TTS process (if binary & voice model configured).
    2. Windows SAPI Synthesizer (built-in offline speech on Windows).
    3. Browser Client Signal (instructs browser's local SpeechSynthesis to speak).
    """

    def __init__(self, piper_binary: Optional[str] = None, piper_model: Optional[str] = None):
        self.piper_binary = piper_binary or shutil.which("piper")
        self.piper_model = piper_model
        self._lock = threading.Lock()
        self._last_spoken_phrase: Optional[str] = None

    @property
    def is_piper_available(self) -> bool:
        return bool(self.piper_binary and os.path.isfile(self.piper_binary))

    @property
    def is_windows_sapi_available(self) -> bool:
        return sys.platform == "win32"

    @property
    def last_spoken_phrase(self) -> Optional[str]:
        return self._last_spoken_phrase

    def speak(self, phrase: str, non_blocking: bool = False) -> TTSResult:
        """
        Synthesizes speech for the provided phrase.
        Validates phrase, handles errors gracefully, and never raises exceptions.
        """
        clean_phrase = phrase.strip()
        if not clean_phrase:
            return TTSResult(
                success=False,
                phrase="",
                engine_used="NONE",
                error_message="EMPTY_PHRASE: Cannot speak an empty phrase.",
            )

        if non_blocking:
            t = threading.Thread(target=self._execute_speak, args=(clean_phrase,), daemon=True)
            t.start()
            return TTSResult(
                success=True,
                phrase=clean_phrase,
                engine_used="ASYNC_LAUNCHED",
                error_message=None,
            )
        else:
            return self._execute_speak(clean_phrase)

    def _execute_speak(self, phrase: str) -> TTSResult:
        with self._lock:
            # 1. Try Piper TTS if available
            if self.is_piper_available and self.piper_model and os.path.isfile(self.piper_model):
                try:
                    # Piper command: echo phrase | piper --model model.onnx --output-raw | ...
                    proc = subprocess.run(
                        [self.piper_binary, "--model", self.piper_model, "--output_file", "-"],
                        input=phrase.encode("utf-8"),
                        capture_output=True,
                        timeout=5.0,
                    )
                    if proc.returncode == 0:
                        self._last_spoken_phrase = phrase
                        return TTSResult(success=True, phrase=phrase, engine_used="PIPER_LOCAL")
                except Exception as exc:
                    # Piper failed, fall through to Windows SAPI
                    pass

            # 2. Try Windows SAPI (offline desktop voice)
            if self.is_windows_sapi_available:
                try:
                    # Sanitize phrase for PowerShell script string literal
                    escaped = phrase.replace("'", "''").replace('"', '""')
                    ps_cmd = (
                        f"Add-Type -AssemblyName System.Speech; "
                        f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                        f"$s.Speak('{escaped}')"
                    )
                    proc = subprocess.run(
                        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps_cmd],
                        capture_output=True,
                        text=True,
                        timeout=6.0,
                    )
                    if proc.returncode == 0:
                        self._last_spoken_phrase = phrase
                        return TTSResult(success=True, phrase=phrase, engine_used="WINDOWS_SAPI_OFFLINE")
                    else:
                        err = proc.stderr.strip() or f"Process exited with code {proc.returncode}"
                        return TTSResult(
                            success=False,
                            phrase=phrase,
                            engine_used="WINDOWS_SAPI",
                            error_message=f"SAPI_ERROR: {err}",
                        )
                except subprocess.TimeoutExpired:
                    return TTSResult(
                        success=False,
                        phrase=phrase,
                        engine_used="WINDOWS_SAPI",
                        error_message="SAPI_TIMEOUT: Speech synthesis timed out after 6 seconds.",
                    )
                except Exception as exc:
                    return TTSResult(
                        success=False,
                        phrase=phrase,
                        engine_used="WINDOWS_SAPI",
                        error_message=f"SAPI_EXCEPTION: {str(exc)}",
                    )

            # 3. Fallback: Signal to client browser Web Speech
            self._last_spoken_phrase = phrase
            return TTSResult(
                success=True,
                phrase=phrase,
                engine_used="CLIENT_BROWSER_SPEECH_SIGNAL",
                error_message=None,
            )
