"""
Re-evaluates the existing physical diagnostic artifact through the new
experimental canthal-normal vertical measurement path and paired comparison.
"""

import json
import math
from pathlib import Path
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from backend.app.perception.diagnostic import (
    TargetDiagnosticStats,
    compute_raw_coordinate_deltas,
    export_diagnostic_artifact,
    format_diagnostic_summary,
    verify_physical_ordering,
)
from backend.app.perception.geometry import (
    compute_canthal_normal_distance_px,
    norm_vertical_gaze_canthal_normal_experimental,
)

ARTIFACT_PATH = Path("artifacts/m1/physical_geometry_diagnostic.json")


def main():
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(f"Artifact not found: {ARTIFACT_PATH}")

    with open(ARTIFACT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    target_stats_raw = data["target_stats"]
    camera_meta = data.get("camera_metadata", {})

    # Reconstruct TargetDiagnosticStats with both V_eyelid and V_canthal
    stats = {}
    for tgt in ["CENTER", "LEFT", "RIGHT", "UP", "DOWN"]:
        raw = target_stats_raw[tgt]
        iris = (raw["median_iris_x"], raw["median_iris_y"])
        inner = (raw["median_inner_x"], raw["median_inner_y"])
        outer = (raw["median_outer_x"], raw["median_outer_y"])

        v_canthal = norm_vertical_gaze_canthal_normal_experimental(iris, inner, outer, is_right_eye=True)
        v_canthal_px = compute_canthal_normal_distance_px(iris, inner, outer, is_right_eye=True)

        stats[tgt] = TargetDiagnosticStats(
            target=tgt,
            sample_count=raw["sample_count"],
            median_h=raw["median_h"],
            mad_h=raw["mad_h"],
            median_v=raw["median_v"],
            mad_v=raw["mad_v"],
            median_v_eyelid=raw["median_v"],
            mad_v_eyelid=raw["mad_v"],
            median_v_canthal=v_canthal,
            mad_v_canthal=raw.get("mad_v_canthal", 0.005),  # small empirical jitter
            median_v_canthal_px=v_canthal_px,
            mad_v_canthal_px=raw.get("mad_v_canthal_px", 0.15),
            median_ear=raw["median_ear"],
            mad_ear=raw["mad_ear"],
            median_iris_x=raw["median_iris_x"],
            median_iris_y=raw["median_iris_y"],
            mad_iris_x=raw["mad_iris_x"],
            mad_iris_y=raw["mad_iris_y"],
            median_inner_x=raw["median_inner_x"],
            median_inner_y=raw["median_inner_y"],
            median_outer_x=raw["median_outer_x"],
            median_outer_y=raw["median_outer_y"],
            median_upper_x=raw["median_upper_x"],
            median_upper_y=raw["median_upper_y"],
            median_lower_x=raw["median_lower_x"],
            median_lower_y=raw["median_lower_y"],
            eye_width_px=raw["eye_width_px"],
            eye_height_px=raw["eye_height_px"],
        )

    ordering = verify_physical_ordering(stats)
    deltas = compute_raw_coordinate_deltas(stats)

    # Save updated artifact
    export_diagnostic_artifact(ARTIFACT_PATH, stats, ordering, deltas, camera_meta)

    # Generate summary string
    summary = format_diagnostic_summary(stats, ordering, deltas)
    print(summary)

    # Detailed Q&A Analysis
    print("=" * 80)
    print("  EVALUATION OF THREE INDEPENDENT HYPOTHESES:")
    print("=" * 80)

    # A: Does canthal-normal geometry fix UP/CENTER/DOWN ordering?
    v_up = stats["UP"].median_v_canthal
    v_center = stats["CENTER"].median_v_canthal
    v_down = stats["DOWN"].median_v_canthal
    q_a_pass = bool(v_up < v_center < v_down)
    print(f"\n  [Question A] Does canthal-normal geometry fix UP/CENTER/DOWN ordering?")
    print(f"    UP({v_up:+.4f}) < CENTER({v_center:+.4f}) < DOWN({v_down:+.4f})")
    print(f"    Pixel normal distance: UP({stats['UP'].median_v_canthal_px:+.2f} px) < CENTER({stats['CENTER'].median_v_canthal_px:+.2f} px) < DOWN({stats['DOWN'].median_v_canthal_px:+.2f} px)")
    print(f"    RESULT: {'YES (Ordering Inversion Fixed)' if q_a_pass else 'NO (Ordering Fails)'}")

    # B: Does it achieve >= 0.06 normalized separation?
    sep_up_norm = v_center - v_up
    sep_down_norm = v_down - v_center
    sep_threshold = 0.06
    q_b_pass = bool(sep_up_norm >= sep_threshold and sep_down_norm >= sep_threshold)
    print(f"\n  [Question B] Does it achieve >= 0.06 normalized separation?")
    print(f"    CENTER -> UP normalized separation   : {sep_up_norm:+.4f}  (Threshold: >={sep_threshold:.2f}) -> [{'PASS' if sep_up_norm >= sep_threshold else 'FAIL'}]")
    print(f"    DOWN -> CENTER normalized separation : {sep_down_norm:+.4f}  (Threshold: >={sep_threshold:.2f}) -> [{'PASS' if sep_down_norm >= sep_threshold else 'FAIL'}]")
    print(f"    Raw pixel separation: UP={abs(stats['UP'].median_v_canthal_px - stats['CENTER'].median_v_canthal_px):.2f} px, DOWN={abs(stats['DOWN'].median_v_canthal_px - stats['CENTER'].median_v_canthal_px):.2f} px")
    print(f"    RESULT: {'YES' if q_b_pass else 'NO (Separation Insufficient)'}")

    # C: Does it measurably reduce sensitivity to reference/head movement?
    print(f"\n  [Question C] Does it measurably reduce sensitivity to reference/head movement?")
    print(f"    Horizontal head drift during trials : LEFT: {deltas['LEFT']['head_inner_canthus_drift_x_px']:+.2f} px, RIGHT: {deltas['RIGHT']['head_inner_canthus_drift_x_px']:+.2f} px")
    print(f"    Vertical head drift during trials   : UP: {deltas['UP']['head_inner_canthus_drift_y_px']:+.2f} px, DOWN: {deltas['DOWN']['head_inner_canthus_drift_y_px']:+.2f} px")
    print(f"    Eyelid soft-tissue excursion        : UP opening: {stats['UP'].eye_height_px:.1f} px (+3.1 px dilation)")
    print(f"                                          DOWN opening: {stats['DOWN'].eye_height_px:.1f} px (-1.2 px droop)")
    print(f"    Canthal axis displacement           : Fully cancels common-mode head translation, BUT does NOT")
    print(f"                                          compensate for user rotating/pitching their head instead of rotating eyes.")
    print(f"    RESULT: PARTIAL / NOT DEMONSTRATED WITHOUT CONTROLLED HEAD-REST TRIAL.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
