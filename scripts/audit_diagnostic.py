import json
import math
import numpy as np

with open("artifacts/m1/physical_geometry_diagnostic.json", "r", encoding="utf-8") as f:
    d = json.load(f)

stats = d["target_stats"]

print("=" * 115)
print(f"{'TARGET':<8} {'H':<7} {'V':<7} {'IRIS (x, y)':<18} {'INNER (x, y)':<18} {'OUTER (x, y)':<18} {'UPPER (x, y)':<18} {'LOWER (x, y)':<18} {'WIDTH':<6} {'OPEN':<6}")
print("-" * 115)

for tgt in ["CENTER", "LEFT", "RIGHT", "UP", "DOWN"]:
    s = stats[tgt]
    print(f"{tgt:<8} {s['median_h']:<7.3f} {s['median_v']:<7.3f} "
          f"({s['median_iris_x']:.1f}, {s['median_iris_y']:.1f})      "
          f"({s['median_inner_x']:.1f}, {s['median_inner_y']:.1f})      "
          f"({s['median_outer_x']:.1f}, {s['median_outer_y']:.1f})      "
          f"({s['median_upper_x']:.1f}, {s['median_upper_y']:.1f})      "
          f"({s['median_lower_x']:.1f}, {s['median_lower_y']:.1f})      "
          f"{s['eye_width_px']:<6.1f} {s['eye_height_px']:<6.1f}")

print("\n" + "=" * 80)
print("  DETAILED VECTOR & PROJECTION ANALYSIS")
print("=" * 80)

c = stats["CENTER"]
c_inn = np.array([c["median_inner_x"], c["median_inner_y"]])
c_out = np.array([c["median_outer_x"], c["median_outer_y"]])
c_upp = np.array([c["median_upper_x"], c["median_upper_y"]])
c_low = np.array([c["median_lower_x"], c["median_lower_y"]])
c_iris = np.array([c["median_iris_x"], c["median_iris_y"]])

for tgt in ["CENTER", "LEFT", "RIGHT", "UP", "DOWN"]:
    s = stats[tgt]
    inn = np.array([s["median_inner_x"], s["median_inner_y"]])
    out = np.array([s["median_outer_x"], s["median_outer_y"]])
    upp = np.array([s["median_upper_x"], s["median_upper_y"]])
    low = np.array([s["median_lower_x"], s["median_lower_y"]])
    iris = np.array([s["median_iris_x"], s["median_iris_y"]])

    # H axis vector (inner -> outer)
    ch = out - inn
    # V axis vector (upper -> lower)
    cv = low - upp

    # Unit vectors
    len_h = np.linalg.norm(ch)
    len_v = np.linalg.norm(cv)
    u_h = ch / len_h
    u_v = cv / len_v

    # Projections
    h_ratio = np.dot(iris - inn, ch) / (len_h ** 2 + 1e-7)
    v_ratio = np.dot(iris - upp, cv) / (len_v ** 2 + 1e-7)

    proj_h = inn + h_ratio * ch
    proj_v = upp + v_ratio * cv

    # Displacement of iris from segment start along unit axis in pixels
    disp_h_px = np.dot(iris - inn, u_h)
    disp_v_px = np.dot(iris - upp, u_v)

    # Relative to CENTER:
    delta_iris_raw = iris - c_iris
    delta_inn_raw = inn - c_inn
    delta_iris_relative_to_canthi = (iris - inn) - (c_iris - c_inn)

    print(f"\n--- [{tgt}] ---")
    print(f"  Eye Width (Canthal axis length) : {len_h:.2f} px")
    print(f"  Eye Opening (Palpebral height)  : {len_v:.2f} px")
    print(f"  Eye Axis Vector (inner -> outer): ({ch[0]:+.2f}, {ch[1]:+.2f})")
    print(f"  Vertical Vector (upper -> lower): ({cv[0]:+.2f}, {cv[1]:+.2f})")
    print(f"  Iris distance along H axis      : {disp_h_px:.2f} px (out of {len_h:.2f} px) -> H = {h_ratio:.3f}")
    print(f"  Iris distance along V axis      : {disp_v_px:.2f} px (out of {len_v:.2f} px) -> V = {v_ratio:.3f}")
    print(f"  Raw Iris shift from CENTER      : d_x = {delta_iris_raw[0]:+.2f} px, d_y = {delta_iris_raw[1]:+.2f} px")
    print(f"  Raw Inner Canthus shift (Head)  : d_x = {delta_inn_raw[0]:+.2f} px, d_y = {delta_inn_raw[1]:+.2f} px")
    print(f"  Iris shift relative to Canthi   : d_x = {delta_iris_relative_to_canthi[0]:+.2f} px, d_y = {delta_iris_relative_to_canthi[1]:+.2f} px")
