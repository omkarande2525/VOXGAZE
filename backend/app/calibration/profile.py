"""
VoxGaze Calibration Profile Schema & Persistence Helpers
Uses authoritative Pydantic models from backend.app.core.contracts.
"""

import json
from pathlib import Path
from typing import Optional, Union
from pydantic import ValidationError

from backend.app.core.contracts import (
    CalibrationProfile,
    CalibrationStats,
    CalibrationThresholds,
    DirectionThreshold,
)


def validate_thresholds(t: CalibrationThresholds) -> tuple[bool, Optional[str]]:
    """
    Validate that hysteresis thresholds have proper separation and ordering.
    - Left: enter < exit (looking left decreases H toward 0)
    - Right: enter > exit (looking right increases H toward 1)
    - Up: enter < exit (looking up decreases V toward 0)
    - Down: enter > exit (looking down increases V toward 1)
    - Blink: closed_ear < open_ear
    """
    if t.left.enter >= t.left.exit:
        return False, f"Left enter threshold ({t.left.enter}) must be < left exit threshold ({t.left.exit})"
    if t.right.enter <= t.right.exit:
        return False, f"Right enter threshold ({t.right.enter}) must be > right exit threshold ({t.right.exit})"
    if t.up.enter >= t.up.exit:
        return False, f"Up enter threshold ({t.up.enter}) must be < up exit threshold ({t.up.exit})"
    if t.down.enter <= t.down.exit:
        return False, f"Down enter threshold ({t.down.enter}) must be > down exit threshold ({t.down.exit})"
    if t.blink_closed_ear >= t.blink_open_ear:
        return False, f"Blink closed EAR ({t.blink_closed_ear}) must be < open EAR ({t.blink_open_ear})"

    return True, None


def load_calibration_profile(filepath_or_json: Union[str, Path]) -> CalibrationProfile:
    """Load and validate a CalibrationProfile from a JSON file path or JSON string."""
    if isinstance(filepath_or_json, Path) or (isinstance(filepath_or_json, str) and Path(filepath_or_json).is_file()):
        with open(filepath_or_json, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = json.loads(filepath_or_json)

    profile = CalibrationProfile.model_validate(data)
    is_valid, reason = validate_thresholds(profile.thresholds)
    if not is_valid:
        raise ValueError(f"Invalid calibration profile thresholds: {reason}")
    return profile


def save_calibration_profile(profile: CalibrationProfile, output_path: Union[str, Path]) -> Path:
    """Serialize a CalibrationProfile to disk as formatted JSON."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        f.write(profile.model_dump_json(indent=2))
    return out_p
