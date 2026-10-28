import json
import numpy as np

with open("artifacts/m1/physical_geometry_diagnostic.json", "r", encoding="utf-8") as f:
    data = json.load(f)["target_stats"]

targets = ["CENTER", "LEFT", "RIGHT", "UP", "DOWN"]

for t in targets:
    s = data[t]
    iris = np.array([s["median_iris_x"], s["median_iris_y"]])
    inner = np.array([s["median_inner_x"], s["median_inner_y"]])
    outer = np.array([s["median_outer_x"], s["median_outer_y"]])
    upper = np.array([s["median_upper_x"], s["median_upper_y"]])
    lower = np.array([s["median_lower_x"], s["median_lower_y"]])

    # 1. eye-axis vector
    v_h = outer - inner
    eye_width = np.linalg.norm(v_h)
    u_h = v_h / eye_width

    # 2. perpendicular axis vector (orthogonal to eye axis, pointing downward +y)
    v_perp = np.array([v_h[1], -v_h[0]])
    u_perp = v_perp / np.linalg.norm(v_perp)

    # 3. V-axis vector (upper to lower lid)
    v_v = lower - upper
    eye_opening = np.linalg.norm(v_v)
    u_v = v_v / eye_opening

    # 4. Projected iris coordinate on H axis
    h_ratio = float(np.dot(iris - inner, v_h) / (eye_width**2))
    proj_iris_h = inner + h_ratio * v_h

    # 5. Projected iris coordinate on V axis (lid-to-lid)
    v_ratio = float(np.dot(iris - upper, v_v) / (eye_opening**2))
    proj_iris_v = upper + v_ratio * v_v

    # 6. Projected iris coordinate on perpendicular axis (passing through eye midpoint)
    eye_mid = (inner + outer) / 2.0
    perp_dist = float(np.dot(iris - eye_mid, u_perp))
    proj_iris_perp = eye_mid + perp_dist * u_perp

    print(f"=== {t} ===")
    print(f"  Median Raw Iris     : ({iris[0]:.2f}, {iris[1]:.2f})")
    print(f"  Median Inner (133)  : ({inner[0]:.2f}, {inner[1]:.2f})")
    print(f"  Median Outer (33)   : ({outer[0]:.2f}, {outer[1]:.2f})")
    print(f"  Median Upper (159)  : ({upper[0]:.2f}, {upper[1]:.2f})")
    print(f"  Median Lower (145)  : ({lower[0]:.2f}, {lower[1]:.2f})")
    print(f"  Eye Width           : {eye_width:.2f} px")
    print(f"  Eye Opening         : {eye_opening:.2f} px")
    print(f"  Eye-Axis Vector     : ({v_h[0]:+.2f}, {v_h[1]:+.2f})")
    print(f"  Perp-Axis Vector    : ({v_perp[0]:+.2f}, {v_perp[1]:+.2f})  [unit: ({u_perp[0]:+.4f}, {u_perp[1]:+.4f})]")
    print(f"  V-Axis Vector (lids): ({v_v[0]:+.2f}, {v_v[1]:+.2f})")
    print(f"  Projected Iris (H)  : ({proj_iris_h[0]:.2f}, {proj_iris_h[1]:.2f})")
    print(f"  Projected Iris (V)  : ({proj_iris_v[0]:.2f}, {proj_iris_v[1]:.2f})")
    print(f"  Normalized H        : {h_ratio:.4f} (stored median: {s['median_h']:.3f})")
    print(f"  Normalized V        : {v_ratio:.4f} (stored median: {s['median_v']:.3f})")
    print(f"  Perp dist from mid  : {perp_dist:+.2f} px")
