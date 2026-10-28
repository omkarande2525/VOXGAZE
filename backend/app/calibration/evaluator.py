"""
VoxGaze Calibration & Validation Evaluator (M1.5-B1)
Pure, deterministic analysis of calibration quality, directional separation,
baseline drift, frozen threshold derivation, and independent test trial evaluation.

RESEARCH & SCIENTIFIC INVARIANTS:
1. Strict Train/Test Separation: Thresholds are derived on Phase 1 ONLY and frozen before evaluating Phase 2.
2. Temporal Correlation: Frames within a 4-second trial are correlated. Both frame-level and trial-level metrics are computed.
3. Upward Gaze Gate: If vertical separation delta V < 0.05 (provisional engineering threshold), UP gaze is explicitly flagged as UP_GAZE_UNSUPPORTED.
4. Non-Destructive / Honest Data: Missing or invalid frames are accounted for in valid-sample rates and never imputed.
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from backend.app.calibration.protocol import (
    CalibrationValidationCondition,
    CalibrationValidationRecord,
    ProtocolPhase,
)
from backend.app.core.contracts import (
    CalibrationStats,
    CalibrationThresholds,
    Direction,
    DirectionThreshold,
)
from backend.app.perception.geometry import (
    PLAUSIBLE_H_MIN,
    PLAUSIBLE_H_MAX,
    PLAUSIBLE_V_MIN,
    PLAUSIBLE_V_MAX,
)


# Provisional engineering thresholds (documented as provisional, not universal clinical ground truth)
PROVISIONAL_MIN_SEPARATION_H: float = 0.06
PROVISIONAL_MIN_SEPARATION_V: float = 0.05
PROVISIONAL_MIN_SEPARATION_V_CANTHAL: float = 0.005
PROVISIONAL_MIN_SEPARATION_BLINK: float = 0.05
PROVISIONAL_MAX_MAD: float = 0.08


def filter_valid_samples(
    records: List[CalibrationValidationRecord],
    active_eye: str = "right",
    require_geometry_valid: bool = True,
    allow_closed_eyes: bool = False,
) -> List[CalibrationValidationRecord]:
    """
    Filters records that have face detected and valid active eye geometry.
    When allow_closed_eyes=True, permits low EAR (<0.18) for CLOSED condition calibration
    while strictly rejecting missing-face, non-finite, and zero-width geometry.
    """
    valid: List[CalibrationValidationRecord] = []
    for r in records:
        if not r.face_detected:
            continue

        if not allow_closed_eyes:
            if require_geometry_valid and not r.geometry_valid:
                continue

            # Check per-eye validity for open gaze
            if active_eye == "right":
                if r.right_eye_valid is False:
                    continue
                if r.right_eye_h_ratio is None or r.right_eye_v_ratio is None or r.right_eye_ear is None:
                    continue
                if not (math.isfinite(r.right_eye_h_ratio) and math.isfinite(r.right_eye_v_ratio) and math.isfinite(r.right_eye_ear)):
                    continue
            else:
                if r.left_eye_valid is False:
                    continue
                if r.left_eye_h_ratio is None or r.left_eye_v_ratio is None or r.left_eye_ear is None:
                    continue
                if not (math.isfinite(r.left_eye_h_ratio) and math.isfinite(r.left_eye_v_ratio) and math.isfinite(r.left_eye_ear)):
                    continue
        else:
            # Closed-eye condition: face must be detected and active eye EAR must be finite and plausible
            ear_val = r.right_eye_ear if active_eye == "right" else r.left_eye_ear
            if ear_val is None or not math.isfinite(ear_val) or ear_val < 0.0 or ear_val > 1.0:
                continue
            if r.unavailable_reason and ("NO_FACE" in r.unavailable_reason or "NON_FINITE" in r.unavailable_reason or "INSUFFICIENT_EYE_WIDTH" in r.unavailable_reason):
                continue

        valid.append(r)
    return valid


def compute_target_distribution(
    records: List[CalibrationValidationRecord],
    active_eye: str = "right",
    allow_closed_eyes: bool = False,
) -> Dict[str, Any]:
    """
    Computes distribution statistics (median, MAD, std, min, max, valid counts)
    for a list of records corresponding to a single target direction or condition.
    """
    total_frames = len(records)
    face_detected_count = sum(1 for r in records if r.face_detected)
    geom_valid_count = sum(1 for r in records if r.geometry_valid)
    r_eye_valid_count = sum(1 for r in records if r.right_eye_valid is True)
    l_eye_valid_count = sum(1 for r in records if r.left_eye_valid is True)

    valid_records = filter_valid_samples(
        records,
        active_eye=active_eye,
        require_geometry_valid=True,
        allow_closed_eyes=allow_closed_eyes,
    )
    valid_count = len(valid_records)

    if valid_count == 0:
        return {
            "total_frames": total_frames,
            "face_detected_count": face_detected_count,
            "geom_valid_count": geom_valid_count,
            "right_eye_valid_count": r_eye_valid_count,
            "left_eye_valid_count": l_eye_valid_count,
            "valid_samples_used": 0,
            "valid_sample_rate": 0.0,
            "h": None,
            "v_eyelid": None,
            "v_canthal": None,
            "canthal_dist_px": None,
            "ear": None,
        }

    if active_eye == "right":
        h_vals = np.array([r.right_eye_h_ratio for r in valid_records if r.right_eye_h_ratio is not None and math.isfinite(r.right_eye_h_ratio)], dtype=float)
        v_vals = np.array([r.right_eye_v_ratio for r in valid_records if r.right_eye_v_ratio is not None and math.isfinite(r.right_eye_v_ratio)], dtype=float)
        ear_vals = np.array([r.right_eye_ear for r in valid_records if r.right_eye_ear is not None and math.isfinite(r.right_eye_ear)], dtype=float)
        canthal_vals = np.array([r.right_eye_canthal_ratio for r in valid_records if r.right_eye_canthal_ratio is not None and math.isfinite(r.right_eye_canthal_ratio)], dtype=float)
    else:
        h_vals = np.array([r.left_eye_h_ratio for r in valid_records if r.left_eye_h_ratio is not None and math.isfinite(r.left_eye_h_ratio)], dtype=float)
        v_vals = np.array([r.left_eye_v_ratio for r in valid_records if r.left_eye_v_ratio is not None and math.isfinite(r.left_eye_v_ratio)], dtype=float)
        ear_vals = np.array([r.left_eye_ear for r in valid_records if r.left_eye_ear is not None and math.isfinite(r.left_eye_ear)], dtype=float)
        canthal_vals = np.array([r.left_eye_canthal_ratio for r in valid_records if r.left_eye_canthal_ratio is not None and math.isfinite(r.left_eye_canthal_ratio)], dtype=float)

    canthal_px_vals = np.array([r.canthal_normal_dist_px for r in valid_records if r.canthal_normal_dist_px is not None and math.isfinite(r.canthal_normal_dist_px)], dtype=float)

    def _stats(arr: np.ndarray) -> Optional[Dict[str, float]]:
        if len(arr) == 0:
            return None
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        return {
            "median": round(med, 4),
            "mad": round(mad, 4),
            "std": round(float(np.std(arr)), 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
        }

    return {
        "total_frames": total_frames,
        "face_detected_count": face_detected_count,
        "geom_valid_count": geom_valid_count,
        "right_eye_valid_count": r_eye_valid_count,
        "left_eye_valid_count": l_eye_valid_count,
        "valid_samples_used": valid_count,
        "valid_sample_rate": round(valid_count / total_frames, 4) if total_frames > 0 else 0.0,
        "h": _stats(h_vals),
        "v_eyelid": _stats(v_vals),
        "v_canthal": _stats(canthal_vals),
        "canthal_dist_px": _stats(canthal_px_vals),
        "ear": _stats(ear_vals),
    }


def compute_phase1_target_stats(
    phase1_records: List[CalibrationValidationRecord],
    active_eye: str = "right",
) -> Dict[str, Dict[str, Any]]:
    """
    Aggregates Phase 1 records by target direction: CENTER, LEFT, RIGHT, UP, DOWN, CLOSED.
    """
    targets: Dict[str, List[CalibrationValidationRecord]] = {
        "CENTER": [],
        "LEFT": [],
        "RIGHT": [],
        "UP": [],
        "DOWN": [],
        "CLOSED": [],
    }
    for r in phase1_records:
        if r.target_direction in targets:
            targets[r.target_direction].append(r)

    stats: Dict[str, Dict[str, Any]] = {}
    for t_dir, recs in targets.items():
        is_closed = (t_dir == "CLOSED")
        stats[t_dir] = compute_target_distribution(recs, active_eye=active_eye, allow_closed_eyes=is_closed)
    return stats


def compute_directional_separations(phase1_stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes directional separation metrics from Phase 1 medians.

    Sign Conventions:
    - Horizontal: LEFT < CENTER < RIGHT (H approx 0.0 is left, 1.0 is right).
      delta_left  = median(H_center) - median(H_left)   [positive indicates leftward shift]
      delta_right = median(H_right) - median(H_center)  [positive indicates rightward shift]
    - Eyelid Vertical: UP < CENTER < DOWN (V approx 0.0 is upper lid, 1.0 is lower lid).
      delta_up    = median(V_center) - median(V_up)     [positive indicates upward shift]
      delta_down  = median(V_down) - median(V_center)   [positive indicates downward shift]
    - Canthal-Normal Vertical: UP < CENTER < DOWN (V_canthal < 0.0 is superior/UP).
      delta_canthal_up   = median(V_canthal_center) - median(V_canthal_up)   [positive is UP]
      delta_canthal_down = median(V_canthal_down) - median(V_canthal_center) [positive is DOWN]
    - Blink EAR: CLOSED < OPEN (EAR approx 0.10 is closed, 0.30 is open).
      delta_blink = median(EAR_center) - median(EAR_closed) [positive is blink opening]
    """
    center = phase1_stats.get("CENTER", {})
    left = phase1_stats.get("LEFT", {})
    right = phase1_stats.get("RIGHT", {})
    up = phase1_stats.get("UP", {})
    down = phase1_stats.get("DOWN", {})
    closed = phase1_stats.get("CLOSED", {})

    def _med(d: Dict[str, Any], key: str) -> Optional[float]:
        metric = d.get(key)
        return metric.get("median") if metric else None

    c_h = _med(center, "h")
    c_v = _med(center, "v_eyelid")
    c_vc = _med(center, "v_canthal")
    c_ear = _med(center, "ear")

    l_h = _med(left, "h")
    r_h = _med(right, "h")
    u_v = _med(up, "v_eyelid")
    u_vc = _med(up, "v_canthal")
    d_v = _med(down, "v_eyelid")
    d_vc = _med(down, "v_canthal")
    cl_ear = _med(closed, "ear")

    sep: Dict[str, Any] = {}
    sep["delta_h_left"] = round(c_h - l_h, 4) if (c_h is not None and l_h is not None) else None
    sep["delta_h_right"] = round(r_h - c_h, 4) if (r_h is not None and c_h is not None) else None
    sep["delta_v_eyelid_up"] = round(c_v - u_v, 4) if (c_v is not None and u_v is not None) else None
    sep["delta_v_eyelid_down"] = round(d_v - c_v, 4) if (d_v is not None and c_v is not None) else None
    sep["delta_v_canthal_up"] = round(c_vc - u_vc, 4) if (c_vc is not None and u_vc is not None) else None
    sep["delta_v_canthal_down"] = round(d_vc - c_vc, 4) if (d_vc is not None and c_vc is not None) else None
    sep["delta_ear_blink"] = round(c_ear - cl_ear, 4) if (c_ear is not None and cl_ear is not None) else None

    return sep


def derive_frozen_candidate_thresholds(
    phase1_stats: Dict[str, Any],
    min_separation_h: float = PROVISIONAL_MIN_SEPARATION_H,
    min_separation_v: float = PROVISIONAL_MIN_SEPARATION_V,
    min_separation_v_canthal: float = PROVISIONAL_MIN_SEPARATION_V_CANTHAL,
    min_separation_blink: float = PROVISIONAL_MIN_SEPARATION_BLINK,
) -> Tuple[Optional[CalibrationThresholds], Dict[str, Any]]:
    """
    Derives candidate dual thresholds strictly from Phase 1 calibration data.
    Freezes candidate thresholds for independent Phase 2 evaluation.

    VERTICAL GAZE GATES (Dual Evaluation: Skeletal Canthal-Normal vs Palpebral Eyelid):
    1. Checks skeletal canthal-normal vertical separation against min_separation_v_canthal (0.005).
       Canthal axis landmarks (33 and 133 / 263 and 362) are anchored to facial bone/ligaments
       and immune to upper-eyelid levator muscle coupling.
    2. Checks palpebral eyelid vertical separation against min_separation_v (0.05).
    3. Prefers canthal normal when it exhibits valid physiological separation.
    4. If neither metric achieves physical separation, flags UP_GAZE_UNSUPPORTED / DOWN_GAZE_UNSUPPORTED.
    """
    sep = compute_directional_separations(phase1_stats)
    gate_results: Dict[str, Any] = {
        "is_left_supported": False,
        "is_right_supported": False,
        "is_up_supported": False,
        "is_down_supported": False,
        "is_blink_supported": False,
        "unsupported_reasons": [],
        "provisional_thresholds_used": {
            "min_separation_h": min_separation_h,
            "min_separation_v": min_separation_v,
            "min_separation_v_canthal": min_separation_v_canthal,
            "min_separation_blink": min_separation_blink,
        },
    }

    c_h = phase1_stats.get("CENTER", {}).get("h", {}).get("median") if phase1_stats.get("CENTER", {}).get("h") else None
    c_v = phase1_stats.get("CENTER", {}).get("v_eyelid", {}).get("median") if phase1_stats.get("CENTER", {}).get("v_eyelid") else None
    c_vc = phase1_stats.get("CENTER", {}).get("v_canthal", {}).get("median") if phase1_stats.get("CENTER", {}).get("v_canthal") else None
    c_ear = phase1_stats.get("CENTER", {}).get("ear", {}).get("median") if phase1_stats.get("CENTER", {}).get("ear") else None
    cl_ear = phase1_stats.get("CLOSED", {}).get("ear", {}).get("median") if phase1_stats.get("CLOSED", {}).get("ear") else None

    if c_h is None or c_ear is None or (c_v is None and c_vc is None):
        gate_results["unsupported_reasons"].append("INSUFFICIENT_CENTER_DATA: Missing center baseline medians")
        return None, gate_results

    # 1. Left Gaze Gate
    d_l = sep.get("delta_h_left")
    l_enabled = bool(d_l is not None and d_l >= min_separation_h)
    if l_enabled:
        gate_results["is_left_supported"] = True
        l_enter = c_h - 0.60 * d_l
        l_exit = c_h - 0.35 * d_l
    else:
        reason = f"LEFT_GAZE_UNSUPPORTED: Delta H ({d_l}) < {min_separation_h}"
        gate_results["unsupported_reasons"].append(reason)
        l_enter, l_exit = 0.0, 0.0

    # 2. Right Gaze Gate
    d_r = sep.get("delta_h_right")
    r_enabled = bool(d_r is not None and d_r >= min_separation_h)
    if r_enabled:
        gate_results["is_right_supported"] = True
        r_enter = c_h + 0.60 * d_r
        r_exit = c_h + 0.35 * d_r
    else:
        reason = f"RIGHT_GAZE_UNSUPPORTED: Delta H ({d_r}) < {min_separation_h}"
        gate_results["unsupported_reasons"].append(reason)
        r_enter, r_exit = 1.0, 1.0

    # 3 & 4. Vertical Gaze Gates (Dual Evaluation: Canthal Normal vs Eyelid)
    d_u_can = sep.get("delta_v_canthal_up")
    d_d_can = sep.get("delta_v_canthal_down")
    d_u_eye = sep.get("delta_v_eyelid_up")
    d_d_eye = sep.get("delta_v_eyelid_down")

    canthal_up_ok = bool(d_u_can is not None and d_u_can >= min_separation_v_canthal)
    canthal_down_ok = bool(d_d_can is not None and d_d_can >= min_separation_v_canthal)
    eyelid_up_ok = bool(d_u_eye is not None and d_u_eye >= min_separation_v)
    eyelid_down_ok = bool(d_d_eye is not None and d_d_eye >= min_separation_v)

    if (canthal_up_ok or canthal_down_ok) and c_vc is not None:
        vertical_metric = "canthal_normal"
        c_v_ref = c_vc

        if canthal_up_ok:
            gate_results["is_up_supported"] = True
            u_enter = c_v_ref - 0.60 * d_u_can
            u_exit = c_v_ref - 0.35 * d_u_can
            u_enabled = True
        else:
            reason = f"UP_GAZE_UNSUPPORTED: Delta V canthal ({d_u_can}) < {min_separation_v_canthal}"
            gate_results["unsupported_reasons"].append(reason)
            u_enter, u_exit = -999.0, -999.0
            u_enabled = False

        if canthal_down_ok:
            gate_results["is_down_supported"] = True
            d_enter = c_v_ref + 0.60 * d_d_can
            d_exit = c_v_ref + 0.35 * d_d_can
            d_enabled = True
        else:
            reason = f"DOWN_GAZE_UNSUPPORTED: Delta V canthal ({d_d_can}) < {min_separation_v_canthal}"
            gate_results["unsupported_reasons"].append(reason)
            d_enter, d_exit = 999.0, 999.0
            d_enabled = False

    elif (eyelid_up_ok or eyelid_down_ok) and c_v is not None:
        vertical_metric = "eyelid"
        c_v_ref = c_v

        if eyelid_up_ok:
            gate_results["is_up_supported"] = True
            u_enter = c_v_ref - 0.60 * d_u_eye
            u_exit = c_v_ref - 0.35 * d_u_eye
            u_enabled = True
        else:
            reason = f"UP_GAZE_UNSUPPORTED: Delta V eyelid ({d_u_eye}) < {min_separation_v}"
            gate_results["unsupported_reasons"].append(reason)
            u_enter, u_exit = -999.0, -999.0
            u_enabled = False

        if eyelid_down_ok:
            gate_results["is_down_supported"] = True
            d_enter = c_v_ref + 0.60 * d_d_eye
            d_exit = c_v_ref + 0.35 * d_d_eye
            d_enabled = True
        else:
            reason = f"DOWN_GAZE_UNSUPPORTED: Delta V eyelid ({d_d_eye}) < {min_separation_v}"
            gate_results["unsupported_reasons"].append(reason)
            d_enter, d_exit = 999.0, 999.0
            d_enabled = False

    else:
        vertical_metric = "eyelid"
        u_enter, u_exit = -999.0, -999.0
        d_enter, d_exit = 999.0, 999.0
        u_enabled = False
        d_enabled = False
        gate_results["unsupported_reasons"].append(
            f"UP_GAZE_UNSUPPORTED: Delta V eyelid ({d_u_eye}) < {min_separation_v} and Delta V canthal ({d_u_can}) < {min_separation_v_canthal}"
        )
        gate_results["unsupported_reasons"].append(
            f"DOWN_GAZE_UNSUPPORTED: Delta V eyelid ({d_d_eye}) < {min_separation_v} and Delta V canthal ({d_d_can}) < {min_separation_v_canthal}"
        )

    gate_results["vertical_metric"] = vertical_metric

    # 5. Blink Gate: participant must have actually closed eyes (closed EAR < 0.18)
    d_b = sep.get("delta_ear_blink")
    eyes_actually_closed = bool(cl_ear is not None and cl_ear < 0.18)
    if d_b is not None and d_b >= min_separation_blink and cl_ear is not None and eyes_actually_closed:
        gate_results["is_blink_supported"] = True
        b_closed = cl_ear + 0.40 * d_b
        b_open = c_ear - 0.25 * d_b
    else:
        gate_results["is_blink_supported"] = False
        if not eyes_actually_closed and cl_ear is not None:
            reason = f"BLINK_UNSUPPORTED: Participant eyes not closed during calibration (closed EAR {cl_ear:.4f} >= 0.18)"
        else:
            reason = f"BLINK_UNSUPPORTED: Delta EAR ({d_b}) < {min_separation_blink}"
        gate_results["unsupported_reasons"].append(reason)
        b_closed, b_open = 0.15, 0.28

    thresholds = CalibrationThresholds(
        left=DirectionThreshold(enter=l_enter, exit=l_exit, is_enabled=l_enabled),
        right=DirectionThreshold(enter=r_enter, exit=r_exit, is_enabled=r_enabled),
        up=DirectionThreshold(enter=u_enter, exit=u_exit, is_enabled=u_enabled),
        down=DirectionThreshold(enter=d_enter, exit=d_exit, is_enabled=d_enabled),
        blink_closed_ear=b_closed,
        blink_open_ear=b_open,
        vertical_metric=vertical_metric,
    )

    return thresholds, gate_results


def classify_frame_discrete(
    h: Optional[float],
    v: Optional[float],
    ear: Optional[float],
    current_dir: Direction,
    thresholds: CalibrationThresholds,
    is_up_supported: bool = True,
    is_down_supported: bool = True,
    is_left_supported: bool = True,
    is_right_supported: bool = True,
) -> Direction:
    """
    Applies frozen dual-threshold hysteresis to classify a single frame.
    Respects supported direction flags and is_enabled threshold status.
    Fails safely on non-finite, out-of-range, or disabled direction inputs.
    """
    # 1. Non-finite input safety gate
    if h is None or v is None or ear is None:
        return Direction.ABSTAIN
    if not (math.isfinite(h) and math.isfinite(v) and math.isfinite(ear)):
        return Direction.ABSTAIN

    # 2. Out-of-bounds ratio safety gate
    if not (PLAUSIBLE_H_MIN <= h <= PLAUSIBLE_H_MAX and PLAUSIBLE_V_MIN <= v <= PLAUSIBLE_V_MAX):
        return Direction.ABSTAIN

    # 3. Blink detection
    if ear <= thresholds.blink_closed_ear:
        return Direction.BLINK

    t = thresholds
    up_enabled = is_up_supported and getattr(t.up, "is_enabled", True)
    down_enabled = is_down_supported and getattr(t.down, "is_enabled", True)
    left_enabled = is_left_supported and getattr(t.left, "is_enabled", True)
    right_enabled = is_right_supported and getattr(t.right, "is_enabled", True)

    # Directional state transitions
    if current_dir == Direction.CENTER:
        candidates = []
        if left_enabled and h <= t.left.enter:
            depth = t.left.enter - h
            candidates.append((Direction.LEFT, depth))
        if right_enabled and h >= t.right.enter:
            depth = h - t.right.enter
            candidates.append((Direction.RIGHT, depth))
        if up_enabled and v <= t.up.enter:
            depth = t.up.enter - v
            candidates.append((Direction.UP, depth))
        if down_enabled and v >= t.down.enter:
            depth = v - t.down.enter
            candidates.append((Direction.DOWN, depth))

        if not candidates:
            return Direction.CENTER
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    elif current_dir == Direction.LEFT:
        if not left_enabled or h >= t.left.exit:
            return Direction.CENTER
        return Direction.LEFT

    elif current_dir == Direction.RIGHT:
        if not right_enabled or h <= t.right.exit:
            return Direction.CENTER
        return Direction.RIGHT

    elif current_dir == Direction.UP:
        if not up_enabled or v >= t.up.exit:
            return Direction.CENTER
        return Direction.UP

    elif current_dir == Direction.DOWN:
        if not down_enabled or v <= t.down.exit:
            return Direction.CENTER
        return Direction.DOWN

    return Direction.CENTER


def evaluate_independent_phase2_trials(
    phase2_records: List[CalibrationValidationRecord],
    frozen_thresholds: CalibrationThresholds,
    gate_results: Dict[str, Any],
    active_eye: str = "right",
) -> Dict[str, Any]:
    """
    Evaluates the frozen candidate thresholds on independent Phase 2 test trials.

    Computes:
    1. Frame-level classification accuracy and confusion counts.
    2. Trial-level evaluation (trial-aggregated median classification), respecting
       intra-trial temporal correlation instead of treating frames as independent.
    3. Center-fixation stability reported strictly separately from directional sensitivity.
    4. Symmetrical omission of all unsupported directions (never counts omitted trials as successes).
    5. When all directions are unsupported, directional accuracy is reported as unavailable/None.
    """
    is_up_supported = gate_results.get("is_up_supported", False)
    is_down_supported = gate_results.get("is_down_supported", False)
    is_left_supported = gate_results.get("is_left_supported", False)
    is_right_supported = gate_results.get("is_right_supported", False)

    # Group records by trial condition index
    trials_by_cond: Dict[int, List[CalibrationValidationRecord]] = {}
    for r in phase2_records:
        trials_by_cond.setdefault(r.condition_index, []).append(r)

    trial_evaluations: List[Dict[str, Any]] = []
    total_test_frames = 0
    correct_test_frames = 0

    confusion_matrix: Dict[str, Dict[str, int]] = {
        "CENTER": {"CENTER": 0, "LEFT": 0, "RIGHT": 0, "UP": 0, "DOWN": 0, "BLINK": 0, "ABSTAIN": 0},
        "LEFT": {"CENTER": 0, "LEFT": 0, "RIGHT": 0, "UP": 0, "DOWN": 0, "BLINK": 0, "ABSTAIN": 0},
        "RIGHT": {"CENTER": 0, "LEFT": 0, "RIGHT": 0, "UP": 0, "DOWN": 0, "BLINK": 0, "ABSTAIN": 0},
        "UP": {"CENTER": 0, "LEFT": 0, "RIGHT": 0, "UP": 0, "DOWN": 0, "BLINK": 0, "ABSTAIN": 0},
        "DOWN": {"CENTER": 0, "LEFT": 0, "RIGHT": 0, "UP": 0, "DOWN": 0, "BLINK": 0, "ABSTAIN": 0},
    }

    trial_correct_count = 0
    trial_total_count = 0

    center_trials_total = 0
    center_trials_correct = 0
    center_frames_total = 0
    center_frames_correct = 0

    dir_trials_total = 0
    dir_trials_correct = 0
    dir_frames_total = 0
    dir_frames_correct = 0

    for cond_idx in sorted(trials_by_cond.keys()):
        recs = trials_by_cond[cond_idx]
        if not recs:
            continue

        target_dir = recs[0].target_direction
        cond_name = recs[0].condition_name

        valid_recs = filter_valid_samples(recs, active_eye=active_eye, require_geometry_valid=True)
        valid_count = len(valid_recs)

        # Symmetrical omission: Skip evaluation if direction is explicitly unsupported
        is_dir_supported = (
            (target_dir == "CENTER")
            or (target_dir == "LEFT" and is_left_supported and getattr(frozen_thresholds.left, "is_enabled", True))
            or (target_dir == "RIGHT" and is_right_supported and getattr(frozen_thresholds.right, "is_enabled", True))
            or (target_dir == "UP" and is_up_supported and getattr(frozen_thresholds.up, "is_enabled", True))
            or (target_dir == "DOWN" and is_down_supported and getattr(frozen_thresholds.down, "is_enabled", True))
        )

        if not is_dir_supported:
            trial_evaluations.append({
                "condition_index": cond_idx,
                "condition_name": cond_name,
                "target_direction": target_dir,
                "status": "OMITTED_UNSUPPORTED",
                "total_frames": len(recs),
                "valid_frames": valid_count,
                "trial_predicted": None,
                "trial_correct": None,
                "frame_accuracy": None,
            })
            continue

        # Trial-level median calculation
        use_canthal_v = getattr(frozen_thresholds, "vertical_metric", "eyelid") == "canthal_normal"
        trial_predicted_dir: Optional[Direction] = None
        if valid_count > 0:
            if active_eye == "right":
                med_h = float(np.median([r.right_eye_h_ratio for r in valid_recs]))
                if use_canthal_v:
                    can_vals = [r.right_eye_canthal_ratio for r in valid_recs if r.right_eye_canthal_ratio is not None and math.isfinite(r.right_eye_canthal_ratio)]
                    med_v = float(np.median(can_vals)) if can_vals else float(np.median([r.right_eye_v_ratio for r in valid_recs]))
                else:
                    med_v = float(np.median([r.right_eye_v_ratio for r in valid_recs]))
                med_ear = float(np.median([r.right_eye_ear for r in valid_recs]))
            else:
                med_h = float(np.median([r.left_eye_h_ratio for r in valid_recs]))
                if use_canthal_v:
                    can_vals = [r.left_eye_canthal_ratio for r in valid_recs if r.left_eye_canthal_ratio is not None and math.isfinite(r.left_eye_canthal_ratio)]
                    med_v = float(np.median(can_vals)) if can_vals else float(np.median([r.left_eye_v_ratio for r in valid_recs]))
                else:
                    med_v = float(np.median([r.left_eye_v_ratio for r in valid_recs]))
                med_ear = float(np.median([r.left_eye_ear for r in valid_recs]))

            trial_predicted_dir = classify_frame_discrete(
                h=med_h,
                v=med_v,
                ear=med_ear,
                current_dir=Direction.CENTER,
                thresholds=frozen_thresholds,
                is_up_supported=is_up_supported,
                is_down_supported=is_down_supported,
                is_left_supported=is_left_supported,
                is_right_supported=is_right_supported,
            )

        trial_correct = (trial_predicted_dir.value == target_dir) if trial_predicted_dir else False
        if trial_predicted_dir is not None:
            trial_total_count += 1
            if trial_correct:
                trial_correct_count += 1

        # Frame-level evaluation within trial (stateful simulation)
        curr_sim_dir = Direction.CENTER
        cond_frame_correct = 0
        cond_eval_frames = 0

        for r in recs:
            if not r.face_detected or not r.geometry_valid:
                pred = "ABSTAIN"
            else:
                h = r.right_eye_h_ratio if active_eye == "right" else r.left_eye_h_ratio
                if use_canthal_v:
                    can_v = r.right_eye_canthal_ratio if active_eye == "right" else r.left_eye_canthal_ratio
                    v = can_v if (can_v is not None and math.isfinite(can_v)) else (r.right_eye_v_ratio if active_eye == "right" else r.left_eye_v_ratio)
                else:
                    v = r.right_eye_v_ratio if active_eye == "right" else r.left_eye_v_ratio
                ear = r.right_eye_ear if active_eye == "right" else r.left_eye_ear
                curr_sim_dir = classify_frame_discrete(
                    h=h,
                    v=v,
                    ear=ear,
                    current_dir=curr_sim_dir,
                    thresholds=frozen_thresholds,
                    is_up_supported=is_up_supported,
                    is_down_supported=is_down_supported,
                    is_left_supported=is_left_supported,
                    is_right_supported=is_right_supported,
                )
                pred = curr_sim_dir.value

            if target_dir in confusion_matrix and pred in confusion_matrix[target_dir]:
                confusion_matrix[target_dir][pred] += 1

            total_test_frames += 1
            cond_eval_frames += 1
            if pred == target_dir:
                correct_test_frames += 1
                cond_frame_correct += 1

        is_center_trial = (target_dir == "CENTER")
        if is_center_trial:
            center_trials_total += 1
            if trial_correct:
                center_trials_correct += 1
            center_frames_total += cond_eval_frames
            center_frames_correct += cond_frame_correct
        else:
            dir_trials_total += 1
            if trial_correct:
                dir_trials_correct += 1
            dir_frames_total += cond_eval_frames
            dir_frames_correct += cond_frame_correct

        trial_evaluations.append({
            "condition_index": cond_idx,
            "condition_name": cond_name,
            "target_direction": target_dir,
            "status": "EVALUATED",
            "total_frames": len(recs),
            "valid_frames": valid_count,
            "trial_predicted": trial_predicted_dir.value if trial_predicted_dir else None,
            "trial_correct": trial_correct,
            "frame_accuracy": round(cond_frame_correct / cond_eval_frames, 4) if cond_eval_frames > 0 else 0.0,
        })

    center_trial_acc = round(center_trials_correct / center_trials_total, 4) if center_trials_total > 0 else None
    center_frame_acc = round(center_frames_correct / center_frames_total, 4) if center_frames_total > 0 else None

    dir_trial_acc = round(dir_trials_correct / dir_trials_total, 4) if dir_trials_total > 0 else None
    dir_frame_acc = round(dir_frames_correct / dir_frames_total, 4) if dir_frames_total > 0 else None

    if dir_trials_total == 0:
        overall_trial_acc = None
        overall_frame_acc = None
        eval_status = "UNAVAILABLE_NO_SUPPORTED_DIRECTIONS"
    else:
        overall_trial_acc = round(trial_correct_count / trial_total_count, 4) if trial_total_count > 0 else 0.0
        overall_frame_acc = round(correct_test_frames / total_test_frames, 4) if total_test_frames > 0 else 0.0
        eval_status = "EVALUATED"

    return {
        "trials_evaluated_count": trial_total_count,
        "trials_correct_count": trial_correct_count,
        "trial_level_accuracy": overall_trial_acc,
        "total_test_frames": total_test_frames,
        "correct_test_frames": correct_test_frames,
        "frame_level_accuracy": overall_frame_acc,
        "evaluation_status": eval_status,
        "center_fixation_stability": {
            "trials_evaluated": center_trials_total,
            "trials_correct": center_trials_correct,
            "trial_accuracy": center_trial_acc,
            "frames_evaluated": center_frames_total,
            "frames_correct": center_frames_correct,
            "frame_accuracy": center_frame_acc,
        },
        "directional_sensitivity": {
            "trials_evaluated": dir_trials_total,
            "trials_correct": dir_trials_correct,
            "trial_accuracy": dir_trial_acc,
            "frames_evaluated": dir_frames_total,
            "frames_correct": dir_frames_correct,
            "frame_accuracy": dir_frame_acc,
            "status": "UNAVAILABLE" if dir_trials_total == 0 else "EVALUATED",
        },
        "trial_details": trial_evaluations,
        "confusion_matrix": confusion_matrix,
    }


def compute_center_drift(
    records: List[CalibrationValidationRecord],
    active_eye: str = "right",
) -> Dict[str, Any]:
    """
    Computes baseline drift across repeated center-return trials.
    Evaluates initial center baseline vs all subsequent center trials.
    """
    center_trials: Dict[int, List[CalibrationValidationRecord]] = {}
    for r in records:
        if r.target_direction == "CENTER":
            center_trials.setdefault(r.condition_index, []).append(r)

    if not center_trials:
        return {"error": "No center trials found"}

    sorted_conds = sorted(center_trials.keys())
    initial_recs = filter_valid_samples(center_trials[sorted_conds[0]], active_eye=active_eye)
    if not initial_recs:
        return {"error": "Initial center baseline has zero valid samples"}

    if active_eye == "right":
        init_med_h = float(np.median([r.right_eye_h_ratio for r in initial_recs]))
        init_med_v = float(np.median([r.right_eye_v_ratio for r in initial_recs]))
    else:
        init_med_h = float(np.median([r.left_eye_h_ratio for r in initial_recs]))
        init_med_v = float(np.median([r.left_eye_v_ratio for r in initial_recs]))

    drift_records: List[Dict[str, Any]] = []
    max_drift_h = 0.0
    max_drift_v = 0.0

    for cond_idx in sorted_conds:
        valid = filter_valid_samples(center_trials[cond_idx], active_eye=active_eye)
        if not valid:
            continue
        if active_eye == "right":
            curr_h = float(np.median([r.right_eye_h_ratio for r in valid]))
            curr_v = float(np.median([r.right_eye_v_ratio for r in valid]))
        else:
            curr_h = float(np.median([r.left_eye_h_ratio for r in valid]))
            curr_v = float(np.median([r.left_eye_v_ratio for r in valid]))

        dh = round(curr_h - init_med_h, 4)
        dv = round(curr_v - init_med_v, 4)
        max_drift_h = max(max_drift_h, abs(dh))
        max_drift_v = max(max_drift_v, abs(dv))

        drift_records.append({
            "condition_index": cond_idx,
            "condition_name": center_trials[cond_idx][0].condition_name,
            "median_h": round(curr_h, 4),
            "median_v": round(curr_v, 4),
            "delta_h_from_initial": dh,
            "delta_v_from_initial": dv,
        })

    return {
        "initial_baseline": {
            "condition_index": sorted_conds[0],
            "median_h": round(init_med_h, 4),
            "median_v": round(init_med_v, 4),
        },
        "max_absolute_drift_h": round(max_drift_h, 4),
        "max_absolute_drift_v": round(max_drift_v, 4),
        "center_returns_evaluated": len(drift_records),
        "drift_series": drift_records,
    }


def evaluate_calibration_validation_session(
    records: List[CalibrationValidationRecord],
    active_eye: str = "right",
) -> Dict[str, Any]:
    """
    Main entry point for full session audit.
    Separates Phase 1, Phase 2, and Phase 3 and computes all metrics.
    """
    p1_recs = [r for r in records if r.protocol_phase == ProtocolPhase.CALIBRATION_ESTIMATION.value]
    p2_recs = [r for r in records if r.protocol_phase == ProtocolPhase.INDEPENDENT_EVALUATION.value]
    p3_recs = [r for r in records if r.protocol_phase == ProtocolPhase.HEAD_MOVEMENT_CONTRAST.value]

    # 1. Phase 1 Calibration Statistics & Target Distributions
    p1_stats = compute_phase1_target_stats(p1_recs, active_eye=active_eye)
    p1_separations = compute_directional_separations(p1_stats)

    # 2. Derive Frozen Candidate Thresholds
    thresholds, gate_results = derive_frozen_candidate_thresholds(p1_stats)

    # 3. Independent Phase 2 Evaluation (Frozen thresholds applied to unseen Phase 2 data)
    p2_eval: Optional[Dict[str, Any]] = None
    if thresholds is not None and len(p2_recs) > 0:
        p2_eval = evaluate_independent_phase2_trials(
            phase2_records=p2_recs,
            frozen_thresholds=thresholds,
            gate_results=gate_results,
            active_eye=active_eye,
        )

    # 4. Center-Return Baseline Drift
    drift = compute_center_drift(records, active_eye=active_eye)

    # 5. Exploratory Phase 3 Head Contrast Summary
    p3_summary: Dict[str, Any] = {}
    if p3_recs:
        p3_by_dir: Dict[str, List[CalibrationValidationRecord]] = {}
        for r in p3_recs:
            p3_by_dir.setdefault(r.target_direction, []).append(r)
        for d, recs in p3_by_dir.items():
            valid = filter_valid_samples(recs, active_eye=active_eye)
            if valid:
                yaws = [r.mobileone_yaw_deg for r in valid if r.mobileone_yaw_deg is not None]
                pitches = [r.mobileone_pitch_deg for r in valid if r.mobileone_pitch_deg is not None]
                hs = [r.right_eye_h_ratio if active_eye == "right" else r.left_eye_h_ratio for r in valid]
                p3_summary[d] = {
                    "valid_samples": len(valid),
                    "mean_mobileone_yaw_deg": round(float(np.mean(yaws)), 2) if yaws else None,
                    "mean_mobileone_pitch_deg": round(float(np.mean(pitches)), 2) if pitches else None,
                    "median_ocular_h": round(float(np.median(hs)), 4) if hs else None,
                }

    # Aggregate overall report
    total_frames = len(records)
    face_count = sum(1 for r in records if r.face_detected)
    geom_count = sum(1 for r in records if r.geometry_valid)

    report = {
        "metadata": {
            "total_frames_recorded": total_frames,
            "overall_face_availability_pct": round(100.0 * face_count / total_frames, 2) if total_frames > 0 else 0.0,
            "overall_geometry_valid_pct": round(100.0 * geom_count / total_frames, 2) if total_frames > 0 else 0.0,
            "active_eye_evaluated": active_eye,
        },
        "phase1_calibration": {
            "target_statistics": p1_stats,
            "directional_separations": p1_separations,
            "gate_results": gate_results,
            "frozen_thresholds": thresholds.model_dump() if thresholds else None,
        },
        "phase2_independent_evaluation": p2_eval,
        "baseline_center_drift": drift,
        "phase3_exploratory_head_contrast": p3_summary,
    }

    return report
