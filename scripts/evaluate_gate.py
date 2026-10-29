"""
VoxGaze Milestone M1 / Task T5: Cross-Session Gate Evaluator (scripts/evaluate_gate.py)

Evaluates whether two or more comparison sessions pass the non-negotiable VoxGaze adoption gate:
1. Physical Data Verification: mock_mode is strictly False in each session report.
2. Layout & Protocol Consistency: Identical target layouts and held-out block structure.
3. Multi-Session Independence: Separate calendar days OR explicit declared camera repositioning.
4. Per-Session Acceptance Gates (EACH session must pass independently):
   - Peripheral Target Accuracy: >= 85.0% on >= 12 held-out trials
   - Center False Activation Rate: <= 5.0% on >= 15 held-out trials
5. Output Status: PASS, FAIL, or INVALID with explicit denominators, counts, and reasons.

READ-ONLY on all input report files.
"""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

MIN_PERIPHERAL_TRIALS: int = 12
MIN_CENTER_TRIALS: int = 15
PERIPHERAL_ACCURACY_THRESHOLD: float = 0.85
CENTER_FALSE_ACTIVATION_MAX: float = 0.05
MIN_INDEPENDENT_SESSIONS: int = 2

# Supported model identifier aliases
MODEL_ALIASES: Dict[str, str] = {
    "mobileone": "mobileone_s0_hybrid",
    "mobileone_s0": "mobileone_s0_hybrid",
    "mobileone_s0_hybrid": "mobileone_s0_hybrid",
    "mobileone_s0_angles": "mobileone_s0_angles",
    "resnet34": "resnet34_hybrid",
    "resnet34_hybrid": "resnet34_hybrid",
    "resnet34_angles": "resnet34_angles",
}


def parse_session_date(report: Dict[str, Any]) -> Optional[str]:
    """Extracts calendar date string (YYYY-MM-DD) from timestamp_utc or session_id."""
    ts = report.get("timestamp_utc")
    if ts and isinstance(ts, str):
        try:
            # Handles '2026-10-29T07:27:35Z' or '2026-10-29 07:27:35'
            clean_ts = ts.replace("Z", "").replace(" ", "T")
            dt = datetime.fromisoformat(clean_ts)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass

    # Fallback to session ID prefix 'cmp_v16_YYYYMMDD_...'
    sid = report.get("session_id", "")
    if sid.startswith("cmp_v16_") and len(sid) >= 16:
        raw_date = sid[8:16]
        if len(raw_date) == 8 and raw_date.isdigit():
            return f"{raw_date[0:4]}-{raw_date[4:6]}-{raw_date[6:8]}"

    return None


def extract_session_layout_fingerprint(report: Dict[str, Any]) -> str:
    """Builds a layout fingerprint string to verify identical block/target layout across sessions."""
    protocol = report.get("protocol", {})
    if not isinstance(protocol, dict):
        return "UNKNOWN_LAYOUT"

    blocks_cnt = protocol.get("blocks_count", 0)
    targets_per_block = protocol.get("targets_per_block", 0)
    held_out_blocks = tuple(protocol.get("held_out_test_blocks", []))
    balanced_orders = protocol.get("balanced_orders", {})

    # Generate deterministic string representation of layout sequence
    orders_repr = json.dumps(balanced_orders, sort_keys=True)
    return f"blocks={blocks_cnt}|targets_per_block={targets_per_block}|held_out={held_out_blocks}|orders={orders_repr}"


def evaluate_cross_session_gate(
    report_paths: List[Path],
    model_name: str = "mobileone_s0_hybrid",
    declared_camera_repositioning: bool = False,
) -> Dict[str, Any]:
    """
    Evaluates multi-session reports against the VoxGaze acceptance gate.
    All files are opened in read-only mode.
    """
    reasons: List[str] = []
    canonical_model = MODEL_ALIASES.get(model_name.lower().strip(), model_name.lower().strip())

    # 1. Check report count
    if len(report_paths) < MIN_INDEPENDENT_SESSIONS:
        reasons.append(
            f"Insufficient sessions provided: {len(report_paths)} report(s) given, "
            f"minimum required for cross-session gate is {MIN_INDEPENDENT_SESSIONS}."
        )
        return {
            "status": "INVALID",
            "model": canonical_model,
            "sessions_evaluated": len(report_paths),
            "reasons": reasons,
            "session_summaries": [],
        }

    loaded_reports: List[Tuple[Path, Dict[str, Any]]] = []
    for path in report_paths:
        p = Path(path).resolve()
        if not p.is_file():
            reasons.append(f"Report file not found: {p}")
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            loaded_reports.append((p, data))
        except Exception as exc:
            reasons.append(f"Failed to parse JSON report at {p}: {exc}")

    if len(loaded_reports) < len(report_paths):
        return {
            "status": "INVALID",
            "model": canonical_model,
            "sessions_evaluated": len(loaded_reports),
            "reasons": reasons,
            "session_summaries": [],
        }

    # 2. Verify all sessions are real physical data (mock_mode == False)
    for path, rep in loaded_reports:
        sid = rep.get("session_id", path.name)
        if rep.get("mock_mode", True) is not False:
            reasons.append(
                f"Session '{sid}' is a MOCK run (mock_mode=True). "
                f"Cross-session gate strictly requires real physical acquisition."
            )
        if rep.get("is_physical_validation", False) is not True:
            reasons.append(f"Session '{sid}' does not have is_physical_validation=True.")

        # Check for aborted / incomplete reports
        if rep.get("status") in ("INCOMPLETE", "ABORTED_INSUFFICIENT_FRAMES") or rep.get("aborted", False):
            reasons.append(f"Session '{sid}' is incomplete or was aborted early.")

    # 3. Verify same model exists across all sessions
    for path, rep in loaded_reports:
        sid = rep.get("session_id", path.name)
        models_dict = rep.get("models", {})
        if canonical_model not in models_dict:
            available = ", ".join(models_dict.keys())
            reasons.append(
                f"Model '{canonical_model}' missing from session '{sid}'. Available models: [{available}]."
            )

    # 4. Verify same layout across all sessions
    fingerprints = [extract_session_layout_fingerprint(rep) for _, rep in loaded_reports]
    if len(set(fingerprints)) > 1:
        reasons.append(
            "Mismatched layout or block protocol between sessions. All evaluated sessions must share identical layouts."
        )

    # 5. Verify independence: separate calendar days OR declared repositioning
    session_dates = [parse_session_date(rep) for _, rep in loaded_reports]
    unique_dates = set([d for d in session_dates if d is not None])
    same_day = len(unique_dates) <= 1

    if same_day and not declared_camera_repositioning:
        date_str = session_dates[0] if session_dates else "unknown"
        reasons.append(
            f"All evaluated sessions occurred on the same day ({date_str}) without a declared "
            f"camera/laptop repositioning. Physical independence requires separate calendar days "
            f"or an explicit declared camera repositioning (--declared-camera-repositioning)."
        )

    # If any structural validity checks failed, return INVALID immediately
    if reasons:
        return {
            "status": "INVALID",
            "model": canonical_model,
            "sessions_evaluated": len(loaded_reports),
            "reasons": reasons,
            "session_summaries": [],
        }

    # 6. Apply numerical acceptance gates in EACH session
    session_summaries: List[Dict[str, Any]] = []
    any_session_invalid = False
    any_session_failed = False

    for path, rep in loaded_reports:
        sid = rep.get("session_id", path.name)
        date_str = parse_session_date(rep) or "unknown"
        m_data = rep["models"][canonical_model]
        eval_5dir = m_data.get("evaluation_5dir", {})

        if not isinstance(eval_5dir, dict) or eval_5dir.get("status") not in ("VALID", None):
            reasons.append(f"Session '{sid}' model evaluation status is invalid: {eval_5dir.get('status')}")
            any_session_invalid = True
            continue

        peri_passed = eval_5dir.get("peripheral_trials_passed")
        peri_total = eval_5dir.get("peripheral_trials_total")
        peri_acc = eval_5dir.get("peripheral_trial_accuracy")

        center_fa = eval_5dir.get("center_trials_false_act")
        center_stayed = eval_5dir.get("center_trials_stayed")
        center_total = eval_5dir.get("center_trials_total")
        center_fa_rate = eval_5dir.get("center_return_false_activation_rate")

        # Sanity check values
        if any(v is None for v in (peri_passed, peri_total, peri_acc, center_fa, center_total, center_fa_rate)):
            reasons.append(f"Session '{sid}' is missing required numerical trial metrics.")
            any_session_invalid = True
            continue

        # Check for NaN / non-finite
        if not (math.isfinite(peri_acc) and math.isfinite(center_fa_rate)):
            reasons.append(f"Session '{sid}' contains non-finite metric values.")
            any_session_invalid = True
            continue

        # Check sample sizes
        peri_sample_ok = peri_total >= MIN_PERIPHERAL_TRIALS
        center_sample_ok = center_total >= MIN_CENTER_TRIALS

        if not peri_sample_ok:
            reasons.append(
                f"Session '{sid}' peripheral sample size insufficient: {peri_total}/{MIN_PERIPHERAL_TRIALS} required."
            )
            any_session_invalid = True

        if not center_sample_ok:
            reasons.append(
                f"Session '{sid}' center sample size insufficient: {center_total}/{MIN_CENTER_TRIALS} required."
            )
            any_session_invalid = True

        # Check thresholds
        peri_pass = peri_acc >= PERIPHERAL_ACCURACY_THRESHOLD
        center_pass = center_fa_rate <= CENTER_FALSE_ACTIVATION_MAX

        if not peri_pass:
            reasons.append(
                f"Session '{sid}' failed peripheral accuracy gate: {peri_passed}/{peri_total} "
                f"({peri_acc*100:.1f}%) < required {PERIPHERAL_ACCURACY_THRESHOLD*100:.1f}%."
            )
            any_session_failed = True

        if not center_pass:
            reasons.append(
                f"Session '{sid}' failed center false-activation gate: {center_fa}/{center_total} "
                f"({center_fa_rate*100:.1f}%) > allowed {CENTER_FALSE_ACTIVATION_MAX*100:.1f}%."
            )
            any_session_failed = True

        session_passed = bool(peri_sample_ok and center_sample_ok and peri_pass and center_pass)

        # Ancillary metrics
        lat_dict = m_data.get("end_to_end_latency_ms", {})
        lat_p50 = lat_dict.get("p50") if isinstance(lat_dict, dict) else None
        trans_dict = m_data.get("transient_center_escapes", {})
        trans_escapes = trans_dict.get("trials_with_transient_escapes") if isinstance(trans_dict, dict) else None

        session_summaries.append({
            "session_id": sid,
            "date": date_str,
            "report_path": str(path),
            "passed": session_passed,
            "peripheral": {
                "passed": peri_passed,
                "total": peri_total,
                "accuracy": round(peri_acc, 4),
                "display": f"{peri_passed}/{peri_total} ({peri_acc*100:.1f}%)",
                "gate_passed": peri_pass,
            },
            "center": {
                "false_activations": center_fa,
                "stayed": center_stayed,
                "total": center_total,
                "false_activation_rate": round(center_fa_rate, 4),
                "display": f"{center_fa}/{center_total} ({center_fa_rate*100:.1f}%)",
                "gate_passed": center_pass,
            },
            "latency_p50_ms": lat_p50,
            "transient_escape_trials": f"{trans_escapes}/{center_total}" if trans_escapes is not None else "N/A",
        })

    if any_session_invalid:
        final_status = "INVALID"
    elif any_session_failed:
        final_status = "FAIL"
    else:
        final_status = "PASS"

    return {
        "status": final_status,
        "model": canonical_model,
        "sessions_evaluated": len(loaded_reports),
        "unique_dates_count": len(unique_dates),
        "dates_evaluated": sorted(list(unique_dates)),
        "declared_camera_repositioning": declared_camera_repositioning,
        "reasons": reasons,
        "session_summaries": session_summaries,
    }


def print_evaluation_summary(result: Dict[str, Any]) -> None:
    """Formats and prints the cross-session gate evaluation in human-readable markdown-like text."""
    status = result["status"]
    model = result["model"]
    sess_count = result["sessions_evaluated"]

    print("\n" + "=" * 80)
    print(f" VOXGAZE MULTI-SESSION ACCEPTANCE GATE EVALUATOR")
    print(f" Evaluated Model: {model} | Sessions Count: {sess_count}")
    print("=" * 80)

    print(f"\n[FINAL STATUS]: >>> {status} <<<")

    if result.get("dates_evaluated"):
        dates_str = ", ".join(result["dates_evaluated"])
        print(f" Dates Evaluated: [{dates_str}] (Unique Days: {result.get('unique_dates_count')})")
    if result.get("declared_camera_repositioning"):
        print(f" Declared Camera Repositioning: TRUE (Portability waiver applied)")

    print("\n[PER-SESSION RESULTS (EACH SESSION MUST PASS)]:")
    for idx, s in enumerate(result.get("session_summaries", []), 1):
        pass_tag = "PASS" if s["passed"] else "FAIL"
        print(f"  Session {idx}: {s['session_id']} ({s['date']}) -> {pass_tag}")
        print(f"    - Peripheral Target Accuracy:     {s['peripheral']['display']} (Required: >= 85.0% on >= 12 trials)")
        print(f"    - Center False Activation (Med):  {s['center']['display']} (Required: <= 5.0% on >= 15 trials)")
        print(f"    - Center Transient Escapes (Any): {s['transient_escape_trials']}")
        print(f"    - Pipeline Latency (p50):         {s['latency_p50_ms']} ms")

    if result.get("reasons"):
        print("\n[DECISION REASONS & CRITERIA LOG]:")
        for r in result["reasons"]:
            print(f"  * {r}")
    else:
        print("\n[DECISION REASONS & CRITERIA LOG]:")
        print("  * All sessions satisfied >=85% peripheral accuracy on >=12 trials.")
        print("  * All sessions satisfied <=5% center false activation rate on >=15 trials.")
        print("  * Multi-session independence verified.")

    print("=" * 80 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluates whether two or more physical comparison sessions pass the VoxGaze acceptance gate."
    )
    parser.add_argument(
        "reports",
        nargs="+",
        type=Path,
        help="Paths to 2 or more session comparison report JSON files.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mobileone_s0_hybrid",
        help="Model identifier to evaluate (default: mobileone_s0_hybrid).",
    )
    parser.add_argument(
        "--declared-camera-repositioning",
        "--repositioned",
        action="store_true",
        help="Explicitly declare that camera/laptop was repositioned for same-day sessions.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON result to stdout instead of formatted text.",
    )

    args = parser.parse_args()

    result = evaluate_cross_session_gate(
        report_paths=args.reports,
        model_name=args.model,
        declared_camera_repositioning=args.declared_camera_repositioning,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_evaluation_summary(result)

    # Return exit codes: 0 for PASS, 1 for FAIL, 2 for INVALID
    if result["status"] == "PASS":
        return 0
    elif result["status"] == "FAIL":
        return 1
    else:
        return 2


if __name__ == "__main__":
    sys.exit(main())
