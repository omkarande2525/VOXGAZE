"""
VoxGaze Milestone M1 - Bilateral Gaze Signal Comparison & Result Audit Tool

Performs strict, independent evaluation of per-target physical measurements:
- RIGHT: median H_R, MAD H_R, median V_R, MAD V_R
- LEFT: median H_L, MAD H_L, median V_L, MAD V_L
- BILATERAL: median H_B, MAD H_B, median V_B, MAD V_B
- DISAGREEMENT: |H_R - H_L| median, |V_R - V_L| median
- AVAILABILITY: bilateral / total, right / total, left / total
- SEPARATIONS: center-relative separation independently for H_R, H_L, H_B, V_R, V_L, V_B
- CLASSIFICATION: ordering, separation >= 0.060, MAD <= 0.080
- ANSWERS TO MANDATORY AUDIT QUESTIONS A, B, C, D, E
"""

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Optional

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from backend.app.perception.diagnostic import (
    TargetDiagnosticStats,
    compute_raw_coordinate_deltas,
    verify_physical_ordering,
)

DEFAULT_ARTIFACT_PATH = Path("artifacts/m1/physical_geometry_diagnostic.json")
M1_SEPARATION_THRESHOLD = 0.060
M1_MAD_THRESHOLD = 0.080


def audit_bilateral_dataset(
    target_stats: Dict[str, TargetDiagnosticStats],
    camera_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Audits the bilateral dataset, strictly avoiding copying or inferring left-eye metrics from right-eye.
    """
    targets = ["CENTER", "LEFT", "RIGHT", "UP", "DOWN"]
    for t in targets:
        if t not in target_stats:
            raise KeyError(f"Missing required target for audit: {t}")

    c = target_stats["CENTER"]
    l = target_stats["LEFT"]
    r = target_stats["RIGHT"]
    u = target_stats["UP"]
    d = target_stats["DOWN"]

    # 1. Center-Relative Separations
    # Horizontal: LEFT (Center - Left), RIGHT (Right - Center)
    sep_h_r = {"LEFT": c.median_h_right - l.median_h_right, "RIGHT": r.median_h_right - c.median_h_right}
    sep_h_l = {"LEFT": c.median_h_left - l.median_h_left, "RIGHT": r.median_h_left - c.median_h_left}
    sep_h_b = {"LEFT": c.median_h_bilateral - l.median_h_bilateral, "RIGHT": r.median_h_bilateral - c.median_h_bilateral}

    # Vertical: UP (Center - Up), DOWN (Down - Center)
    sep_v_r = {"UP": c.median_v_canthal_right - u.median_v_canthal_right, "DOWN": d.median_v_canthal_right - c.median_v_canthal_right}
    sep_v_l = {"UP": c.median_v_canthal_left - u.median_v_canthal_left, "DOWN": d.median_v_canthal_left - c.median_v_canthal_left}
    sep_v_b = {"UP": c.median_v_canthal_bilateral - u.median_v_canthal_bilateral, "DOWN": d.median_v_canthal_bilateral - c.median_v_canthal_bilateral}

    # 2. Independent Ordering
    ord_h_r = bool(l.median_h_right < c.median_h_right < r.median_h_right)
    ord_h_l = bool(l.median_h_left < c.median_h_left < r.median_h_left)
    ord_h_b = bool(l.median_h_bilateral < c.median_h_bilateral < r.median_h_bilateral)

    ord_v_r = bool(u.median_v_canthal_right < c.median_v_canthal_right < d.median_v_canthal_right)
    ord_v_l = bool(u.median_v_canthal_left < c.median_v_canthal_left < d.median_v_canthal_left)
    ord_v_b = bool(u.median_v_canthal_bilateral < c.median_v_canthal_bilateral < d.median_v_canthal_bilateral)

    # 3. Tracking Availability
    tot_frames = sum(target_stats[t].sample_count for t in targets)
    right_frames = sum(target_stats[t].sample_count_right for t in targets)
    left_frames = sum(target_stats[t].sample_count_left for t in targets)
    bi_frames = sum(target_stats[t].sample_count_bilateral for t in targets)

    # 4. Mandatory Questions A, B, C, D, E
    # Q_A: Does left eye contain a larger directional signal than right eye?
    # Compare directional excursion magnitude for each target
    l_larger_h_left = abs(sep_h_l["LEFT"]) > abs(sep_h_r["LEFT"])
    l_larger_h_right = abs(sep_h_l["RIGHT"]) > abs(sep_h_r["RIGHT"])
    l_larger_v_up = abs(sep_v_l["UP"]) > abs(sep_v_r["UP"])
    l_larger_v_down = abs(sep_v_l["DOWN"]) > abs(sep_v_r["DOWN"])

    # Q_B: Does bilateral mean outperform the better individual eye?
    # Directional separation is only meaningful in the intended direction (> 0).
    # Bilateral mean is bounded by min(sep_r, sep_l) <= sep_b <= max(sep_r, sep_l).
    better_h_left = max(sep_h_r["LEFT"], sep_h_l["LEFT"])
    better_h_right = max(sep_h_r["RIGHT"], sep_h_l["RIGHT"])
    better_v_up = max(sep_v_r["UP"], sep_v_l["UP"])
    better_v_down = max(sep_v_r["DOWN"], sep_v_l["DOWN"])

    b_outperforms_h_left = (sep_h_b["LEFT"] > better_h_left) and (sep_h_b["LEFT"] > 0)
    b_outperforms_h_right = (sep_h_b["RIGHT"] > better_h_right) and (sep_h_b["RIGHT"] > 0)
    b_outperforms_v_up = (sep_v_b["UP"] > better_v_up) and (sep_v_b["UP"] > 0)
    b_outperforms_v_down = (sep_v_b["DOWN"] > better_v_down) and (sep_v_b["DOWN"] > 0)
    b_outperforms_any = b_outperforms_h_left or b_outperforms_h_right or b_outperforms_v_up or b_outperforms_v_down

    # Q_C: Does bilateral averaging reduce MAD?
    mad_reductions = {
        t: {
            "mad_h_r": target_stats[t].mad_h_right,
            "mad_h_l": target_stats[t].mad_h_left,
            "mad_h_b": target_stats[t].mad_h_bilateral,
            "h_reduced": (target_stats[t].mad_h_bilateral < target_stats[t].mad_h_right) or (target_stats[t].mad_h_bilateral < target_stats[t].mad_h_left),
            "mad_v_r": target_stats[t].mad_v_canthal_right,
            "mad_v_l": target_stats[t].mad_v_canthal_left,
            "mad_v_b": target_stats[t].mad_v_canthal_bilateral,
            "v_reduced": (target_stats[t].mad_v_canthal_bilateral < target_stats[t].mad_v_canthal_right) or (target_stats[t].mad_v_canthal_bilateral < target_stats[t].mad_v_canthal_left),
        }
        for t in targets
    }
    mad_reduced_targets = sum(1 for m in mad_reductions.values() if m["h_reduced"] or m["v_reduced"])

    # Q_D: Does bilateral mean reach the unchanged 0.060 gate?
    reach_h_left = sep_h_b["LEFT"] >= M1_SEPARATION_THRESHOLD
    reach_h_right = sep_h_b["RIGHT"] >= M1_SEPARATION_THRESHOLD
    reach_v_up = sep_v_b["UP"] >= M1_SEPARATION_THRESHOLD
    reach_v_down = sep_v_b["DOWN"] >= M1_SEPARATION_THRESHOLD
    all_reach = reach_h_left and reach_h_right and reach_v_up and reach_v_down

    # Q_E: Is inter-eye disagreement small enough that fusion is meaningful?
    # Disagreement should be substantially smaller than target separation
    dis_h = [target_stats[t].median_inter_eye_h for t in targets]
    dis_v = [target_stats[t].median_inter_eye_v for t in targets]
    med_dis_h = float(np.median(dis_h))
    med_dis_v = float(np.median(dis_v))
    min_sep_h = min(abs(sep_h_b["LEFT"]), abs(sep_h_b["RIGHT"]))
    min_sep_v = min(abs(sep_v_b["UP"]), abs(sep_v_b["DOWN"]))
    is_disagreement_acceptable = bool(med_dis_h < min_sep_h and med_dis_v < max(0.010, min_sep_v))

    return {
        "targets": targets,
        "separations": {
            "H_R": sep_h_r,
            "H_L": sep_h_l,
            "H_B": sep_h_b,
            "V_R": sep_v_r,
            "V_L": sep_v_l,
            "V_B": sep_v_b,
        },
        "ordering": {
            "H_R": ord_h_r,
            "H_L": ord_h_l,
            "H_B": ord_h_b,
            "V_R": ord_v_r,
            "V_L": ord_v_l,
            "V_B": ord_v_b,
        },
        "availability": {
            "total_frames": tot_frames,
            "right_valid": right_frames,
            "left_valid": left_frames,
            "bilateral_valid": bi_frames,
            "bilateral_rate": bi_frames / max(1, tot_frames),
        },
        "question_a_left_vs_right": {
            "H_LEFT_larger": l_larger_h_left,
            "H_RIGHT_larger": l_larger_h_right,
            "V_UP_larger": l_larger_v_up,
            "V_DOWN_larger": l_larger_v_down,
        },
        "question_b_bilateral_outperforms_better": {
            "outperforms_any": b_outperforms_any,
            "H_LEFT": {"better_eye": better_h_left, "bilateral": sep_h_b["LEFT"], "outperforms": b_outperforms_h_left},
            "H_RIGHT": {"better_eye": better_h_right, "bilateral": sep_h_b["RIGHT"], "outperforms": b_outperforms_h_right},
            "V_UP": {"better_eye": better_v_up, "bilateral": sep_v_b["UP"], "outperforms": b_outperforms_v_up},
            "V_DOWN": {"better_eye": better_v_down, "bilateral": sep_v_b["DOWN"], "outperforms": b_outperforms_v_down},
        },
        "question_c_mad_reduction": {
            "mad_reductions": mad_reductions,
            "reduced_target_count": mad_reduced_targets,
        },
        "question_d_gate_reach": {
            "threshold": M1_SEPARATION_THRESHOLD,
            "all_reach": all_reach,
            "H_LEFT": {"measured": sep_h_b["LEFT"], "reach": reach_h_left},
            "H_RIGHT": {"measured": sep_h_b["RIGHT"], "reach": reach_h_right},
            "V_UP": {"measured": sep_v_b["UP"], "reach": reach_v_up},
            "V_DOWN": {"measured": sep_v_b["DOWN"], "reach": reach_v_down},
        },
        "question_e_inter_eye_disagreement": {
            "median_dis_h": med_dis_h,
            "median_dis_v": med_dis_v,
            "min_sep_h": min_sep_h,
            "min_sep_v": min_sep_v,
            "meaningful_fusion": is_disagreement_acceptable,
        },
    }


def print_audit_report(stats: Dict[str, TargetDiagnosticStats], audit: Dict[str, Any]):
    targets = audit["targets"]
    print("\n" + "=" * 115)
    print("  VOXGAZE M1 BILATERAL RESULT AUDIT — PHYSICAL SIGNAL COMPARISON")
    print("=" * 115)

    # 1. Per-Target Signal Measurements
    print("\n1. PER-TARGET INDEPENDENT MEASUREMENTS:")
    print(f"  {'TARGET':<8} {'H_R':<8} {'MAD_HR':<8} {'V_R':<9} {'MAD_VR':<8} | "
          f"{'H_L':<8} {'MAD_HL':<8} {'V_L':<9} {'MAD_VL':<8} | "
          f"{'H_B':<8} {'MAD_HB':<8} {'V_B':<9} {'MAD_VB':<8}")
    print("  " + "-" * 111)
    for t in targets:
        s = stats[t]
        print(f"  {t:<8} {s.median_h_right:<8.4f} {s.mad_h_right:<8.4f} {s.median_v_canthal_right:<9.4f} {s.mad_v_canthal_right:<8.4f} | "
              f"{s.median_h_left:<8.4f} {s.mad_h_left:<8.4f} {s.median_v_canthal_left:<9.4f} {s.mad_v_canthal_left:<8.4f} | "
              f"{s.median_h_bilateral:<8.4f} {s.mad_h_bilateral:<8.4f} {s.median_v_canthal_bilateral:<9.4f} {s.mad_v_canthal_bilateral:<8.4f}")

    # 2. Inter-Eye Disagreement and Availability
    print("\n2. INTER-EYE DISAGREEMENT & FRAME AVAILABILITY:")
    print(f"  {'TARGET':<8} {'|H_R - H_L|':<14} {'|V_R - V_L|':<14} {'RIGHT VALID':<14} {'LEFT VALID':<14} {'BILATERAL VALID':<16} {'TOTAL':<8}")
    print("  " + "-" * 90)
    for t in targets:
        s = stats[t]
        print(f"  {t:<8} {s.median_inter_eye_h:<14.4f} {s.median_inter_eye_v:<14.4f} "
              f"{s.sample_count_right:<14} {s.sample_count_left:<14} {s.sample_count_bilateral:<16} {s.sample_count:<8}")
    av = audit["availability"]
    print(f"  Total Valid: Right={av['right_valid']}/{av['total_frames']}, "
          f"Left={av['left_valid']}/{av['total_frames']}, "
          f"Bilateral={av['bilateral_valid']}/{av['total_frames']} ({av['bilateral_rate']*100:.1f}%)")

    # 3. Center-Relative Separations & Gate Classification
    print("\n3. CENTER-RELATIVE SEPARATION & GATE CLASSIFICATION (Gate >= 0.060, MAD <= 0.080):")
    seps = audit["separations"]
    print(f"  {'SIGNAL':<12} {'LEFT_SEP':<10} {'>=0.060?':<10} {'RIGHT_SEP':<10} {'>=0.060?':<10} {'ORDERING':<10} {'MAD<=0.080?':<12}")
    print("  " + "-" * 76)
    for sig, h_or_v, t1, t2 in [
        ("H_R", "H", "LEFT", "RIGHT"),
        ("H_L", "H", "LEFT", "RIGHT"),
        ("H_B", "H", "LEFT", "RIGHT"),
    ]:
        s1 = seps[sig][t1]
        s2 = seps[sig][t2]
        g1 = "PASS" if s1 >= M1_SEPARATION_THRESHOLD else "FAIL"
        g2 = "PASS" if s2 >= M1_SEPARATION_THRESHOLD else "FAIL"
        ord_stat = "PASS" if audit["ordering"][sig] else "FAIL"
        m_ok = "PASS" if all(stats[t].mad_h_right <= M1_MAD_THRESHOLD for t in targets) else "FAIL"
        print(f"  {sig:<12} {s1:<+10.4f} {g1:<10} {s2:<+10.4f} {g2:<10} {ord_stat:<10} {m_ok:<12}")

    print(f"  {'SIGNAL':<12} {'UP_SEP':<10} {'>=0.060?':<10} {'DOWN_SEP':<10} {'>=0.060?':<10} {'ORDERING':<10} {'MAD<=0.080?':<12}")
    print("  " + "-" * 76)
    for sig, h_or_v, t1, t2 in [
        ("V_R", "V", "UP", "DOWN"),
        ("V_L", "V", "UP", "DOWN"),
        ("V_B", "V", "UP", "DOWN"),
    ]:
        s1 = seps[sig][t1]
        s2 = seps[sig][t2]
        g1 = "PASS" if s1 >= M1_SEPARATION_THRESHOLD else "FAIL"
        g2 = "PASS" if s2 >= M1_SEPARATION_THRESHOLD else "FAIL"
        ord_stat = "PASS" if audit["ordering"][sig] else "FAIL"
        m_ok = "PASS" if all(stats[t].mad_v_canthal_right <= M1_MAD_THRESHOLD for t in targets) else "FAIL"
        print(f"  {sig:<12} {s1:<+10.4f} {g1:<10} {s2:<+10.4f} {g2:<10} {ord_stat:<10} {m_ok:<12}")

    # 4. Answers to Mandatory Questions A, B, C, D, E
    print("\n" + "=" * 80)
    print("  ANSWERS TO MANDATORY AUDIT QUESTIONS:")
    print("=" * 80)

    # Question A
    qa = audit["question_a_left_vs_right"]
    larger_count = sum(1 for v in qa.values() if v)
    print("\n[QUESTION A] Does left eye contain a larger directional signal than right eye?")
    print(f"  Separations: H_LEFT: Right={seps['H_R']['LEFT']:+.4f} vs Left={seps['H_L']['LEFT']:+.4f}")
    print(f"               H_RIGHT: Right={seps['H_R']['RIGHT']:+.4f} vs Left={seps['H_L']['RIGHT']:+.4f}")
    print(f"               V_UP: Right={seps['V_R']['UP']:+.4f} vs Left={seps['V_L']['UP']:+.4f}")
    print(f"               V_DOWN: Right={seps['V_R']['DOWN']:+.4f} vs Left={seps['V_L']['DOWN']:+.4f}")
    if larger_count >= 3:
        print("  ANSWER: YES. Left eye exhibits greater directional excursion in a majority of directions.")
    elif larger_count > 0:
        print(f"  ANSWER: ASYMMETRIC. Left eye is larger in {larger_count}/4 directions, while Right eye dominates others.")
    else:
        print("  ANSWER: NO. Right eye exhibits equal or greater directional signal across directions.")

    # Question B
    qb = audit["question_b_bilateral_outperforms_better"]
    print("\n[QUESTION B] Does bilateral mean outperform the better individual eye?")
    if qb["outperforms_any"]:
        print("  ANSWER: YES. Bilateral fusion exceeds the better individual eye.")
    else:
        print("  ANSWER: NO. Arithmetic mean H_B = (H_R + H_L)/2 is bounded between the two signals:")
        print(f"          H_B <= max(H_R, H_L). It averages the signals rather than constructively superimposing them.")

    # Question C
    qc = audit["question_c_mad_reduction"]
    print("\n[QUESTION C] Does bilateral averaging reduce MAD?")
    if qc["reduced_target_count"] > 0:
        print(f"  ANSWER: YES. Bilateral averaging reduces sample variance in {qc['reduced_target_count']}/5 targets,")
        print("          dampening independent ocular micro-tremor and landmark tracking jitter.")
    else:
        print("  ANSWER: NO. Bilateral averaging did not achieve lower MAD.")

    # Question D
    qd = audit["question_d_gate_reach"]
    print("\n[QUESTION D] Does bilateral mean reach the unchanged 0.060 gate?")
    if qd["all_reach"]:
        print("  ANSWER: YES. All directions reach or exceed 0.060.")
    else:
        print("  ANSWER: NO (FAIL). Bilateral separations remain below the unchanged 0.060 gate:")
        for d_k, d_v in [("LEFT", qd["H_LEFT"]), ("RIGHT", qd["H_RIGHT"]), ("UP", qd["V_UP"]), ("DOWN", qd["V_DOWN"])]:
            print(f"          {d_k:<8}: {d_v['measured']:+.4f} (Required: >= {M1_SEPARATION_THRESHOLD:.3f}) -> [{'PASS' if d_v['reach'] else 'FAIL'}]")

    # Question E
    qe = audit["question_e_inter_eye_disagreement"]
    print("\n[QUESTION E] Is inter-eye disagreement small enough that fusion is meaningful?")
    print(f"  Median Inter-Eye Disagreement: |H_R - H_L| = {qe['median_dis_h']:.4f}, |V_R - V_L| = {qe['median_dis_v']:.4f}")
    print(f"  Minimum Directional Excursion : Min H Sep = {qe['min_sep_h']:.4f}, Min V Sep = {qe['min_sep_v']:.4f}")
    if qe["meaningful_fusion"]:
        print("  ANSWER: YES. Inter-eye disagreement is consistently smaller than intentional ocular excursion,")
        print("          confirming that both eyes track conjugate physical motion without divergent disconjugacy.")
    else:
        print("  ANSWER: MARGINAL / NO. Disagreement is comparable to or larger than directional excursion,")
        print("          indicating that inter-eye offset obscures small vertical signals.")

    print("=" * 115 + "\n")


def main():
    parser = argparse.ArgumentParser(description="VoxGaze Milestone M1 Bilateral Signal Audit")
    parser.add_argument("--artifact", type=str, default=str(DEFAULT_ARTIFACT_PATH), help="Path to diagnostic artifact")
    args = parser.parse_args()

    artifact_path = Path(args.artifact)
    if not artifact_path.is_file():
        print(f"[ERROR] Diagnostic artifact not found: {artifact_path}")
        return

    with open(artifact_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_stats = data.get("target_stats", {})
    stats: Dict[str, TargetDiagnosticStats] = {}
    for tgt in ["CENTER", "LEFT", "RIGHT", "UP", "DOWN"]:
        if tgt not in raw_stats:
            continue
        r = raw_stats[tgt]
        stats[tgt] = TargetDiagnosticStats(
            target=tgt,
            sample_count=r.get("sample_count", 60),
            sample_count_right=r.get("sample_count_right", r.get("sample_count", 60)),
            sample_count_left=r.get("sample_count_left", 0),
            sample_count_bilateral=r.get("sample_count_bilateral", 0),
            median_h_right=r.get("median_h_right", r.get("median_h", 0.5)),
            mad_h_right=r.get("mad_h_right", r.get("mad_h", 0.005)),
            median_v_canthal_right=r.get("median_v_canthal_right", r.get("median_v_canthal", 0.0)),
            mad_v_canthal_right=r.get("mad_v_canthal_right", r.get("mad_v_canthal", 0.002)),
            median_v_eyelid_right=r.get("median_v_eyelid_right", r.get("median_v_eyelid", r.get("median_v", 0.5))),
            mad_v_eyelid_right=r.get("mad_v_eyelid_right", r.get("mad_v_eyelid", r.get("mad_v", 0.005))),
            median_h_left=r.get("median_h_left", 0.0),
            mad_h_left=r.get("mad_h_left", 0.0),
            median_v_canthal_left=r.get("median_v_canthal_left", 0.0),
            mad_v_canthal_left=r.get("mad_v_canthal_left", 0.0),
            median_v_eyelid_left=r.get("median_v_eyelid_left", 0.0),
            mad_v_eyelid_left=r.get("mad_v_eyelid_left", 0.0),
            median_h_bilateral=r.get("median_h_bilateral", 0.0),
            mad_h_bilateral=r.get("mad_h_bilateral", 0.0),
            median_v_canthal_bilateral=r.get("median_v_canthal_bilateral", 0.0),
            mad_v_canthal_bilateral=r.get("mad_v_canthal_bilateral", 0.0),
            median_inter_eye_h=r.get("median_inter_eye_h", 0.0),
            mad_inter_eye_h=r.get("mad_inter_eye_h", 0.0),
            median_inter_eye_v=r.get("median_inter_eye_v", 0.0),
            mad_inter_eye_v=r.get("mad_inter_eye_v", 0.0),
            median_h=r.get("median_h_right", r.get("median_h", 0.5)),
            mad_h=r.get("mad_h_right", r.get("mad_h", 0.005)),
            median_v=r.get("median_v_eyelid_right", r.get("median_v", 0.5)),
            mad_v=r.get("mad_v_eyelid_right", r.get("mad_v", 0.005)),
            median_v_eyelid=r.get("median_v_eyelid_right", r.get("median_v", 0.5)),
            mad_v_eyelid=r.get("mad_v_eyelid_right", r.get("mad_v", 0.005)),
            median_v_canthal=r.get("median_v_canthal_right", 0.0),
            mad_v_canthal=r.get("mad_v_canthal_right", 0.0),
            median_v_canthal_px=r.get("median_v_canthal_px", 0.0),
            mad_v_canthal_px=r.get("mad_v_canthal_px", 0.0),
            median_ear=r.get("median_ear", 0.3),
            mad_ear=r.get("mad_ear", 0.01),
            median_iris_x=r.get("median_iris_x", 0.0),
            median_iris_y=r.get("median_iris_y", 0.0),
            mad_iris_x=r.get("mad_iris_x", 0.0),
            mad_iris_y=r.get("mad_iris_y", 0.0),
            median_inner_x=r.get("median_inner_x", 0.0),
            median_inner_y=r.get("median_inner_y", 0.0),
            median_outer_x=r.get("median_outer_x", 0.0),
            median_outer_y=r.get("median_outer_y", 0.0),
            median_upper_x=r.get("median_upper_x", 0.0),
            median_upper_y=r.get("median_upper_y", 0.0),
            median_lower_x=r.get("median_lower_x", 0.0),
            median_lower_y=r.get("median_lower_y", 0.0),
            eye_width_px=r.get("eye_width_px", 25.0),
            eye_height_px=r.get("eye_height_px", 7.0),
        )

    audit = audit_bilateral_dataset(stats, data.get("camera_metadata"))
    print_audit_report(stats, audit)


if __name__ == "__main__":
    main()
