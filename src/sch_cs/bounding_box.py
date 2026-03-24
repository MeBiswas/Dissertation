# sch_cs/bounding_box.py

import numpy as np

from src.utils import section

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4C — Bounding Box Correction  (Algorithm 1 from 2018 paper)
# ─────────────────────────────────────────────────────────────────────────────
 
"""
    Corrects centroids that fall outside their region boundary.
    Happens when a region is concave (C-shaped or horseshoe-shaped).

    Algorithm 1:
        For each region k where centroid is outside the mask:
            1. Compute bounding box BB(k) — 4 corner points
            2. D_q = Euclidean distance from centroid to each corner q
            3. New centroid = corner with minimum D_q
"""
def apply_bounding_box_correction(regions: list, labeled_image: np.ndarray) -> list:
    section("STEP 4C — Bounding Box Correction  (Algorithm 1)")
 
    for reg in regions:
        X, Y   = reg["centroid"]
        xi, yj = int(round(X)), int(round(Y))
        h, w   = labeled_image.shape
 
        in_bounds = (0 <= xi < h) and (0 <= yj < w)
        in_region = in_bounds and (labeled_image[xi, yj] == reg["label"])
 
        if in_region:
            reg["centroid_corrected"] = False
            print(f"  R({reg['label']}): ({X:.2f}, {Y:.2f}) "
                  f"→ inside region ✓")
        else:
            coords = reg["coords"]
            i_min, j_min = coords[:, 0].min(), coords[:, 1].min()
            i_max, j_max = coords[:, 0].max(), coords[:, 1].max()
 
            corners = np.array([
                [i_min, j_min],  # top-left
                [i_min, j_max],  # top-right
                [i_max, j_min],  # bottom-left
                [i_max, j_max],  # bottom-right
            ], dtype=float)
 
            distances   = np.linalg.norm(corners - np.array([X, Y]), axis=1)
            nearest_idx = int(np.argmin(distances))
            new_centroid = (float(corners[nearest_idx, 0]),
                            float(corners[nearest_idx, 1]))
 
            corner_names = ["top-left", "top-right",
                            "bottom-left", "bottom-right"]
            print(f"  R({reg['label']}): ({X:.2f}, {Y:.2f}) "
                  f"→ OUTSIDE region ⚠  applying correction")
            for idx, (c, d, n) in enumerate(
                    zip(corners, distances, corner_names)):
                mark = " ← nearest" if idx == nearest_idx else ""
                print(f"    {n:<15} ({c[0]:.1f},{c[1]:.1f})  "
                      f"dist={d:.2f}{mark}")
            print(f"    Corrected centroid: {new_centroid}")
 
            reg["centroid"]           = new_centroid
            reg["centroid_corrected"] = True
 
    return regions