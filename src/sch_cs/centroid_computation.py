# sch_cs/centroid_computation.py

import numpy as np

from src.utils import section

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4B — Weighted Centroid Computation  (Equation 4)
# ─────────────────────────────────────────────────────────────────────────────
 
"""
    Computes weighted centroid for each region using Equation (4):
        X = sum( R(k)(i,j) * i ) / sum( R(k)(i,j) )
        Y = sum( R(k)(i,j) * j ) / sum( R(k)(i,j) )

    Weight = pixel intensity. Brighter pixels pull the centroid more
    strongly — like centre of mass in physics.
"""
def compute_centroids(pb: np.ndarray, regions: list) -> list:
    section("STEP 4B — Weighted Centroid Computation  (Equation 4)")
    print(f"  X = sum(R(k)(i,j) * i) / sum(R(k)(i,j))")
    print(f"  Y = sum(R(k)(i,j) * j) / sum(R(k)(i,j))")
    print(f"  where R(k)(i,j) = pixel intensity at coordinate (i,j)")
 
    print(f"\n  {'Region':<10}  {'Centroid (X,Y)':<22}  "
          f"{'sum(R*i)':<15}  {'sum(R*j)':<15}  {'sum(R)'}")
    print(f"  {'-'*10}  {'-'*22}  {'-'*15}  {'-'*15}  {'-'*12}")
 
    for reg in regions:
        coords  = reg["coords"]
        i_vals  = coords[:, 0].astype(float)
        j_vals  = coords[:, 1].astype(float)
        weights = pb[coords[:, 0], coords[:, 1]].astype(float)
 
        sum_w  = float(np.sum(weights))
        sum_wi = float(np.sum(weights * i_vals))
        sum_wj = float(np.sum(weights * j_vals))
 
        X = sum_wi / sum_w
        Y = sum_wj / sum_w
 
        reg["centroid"] = (X, Y)
        reg["sum_w"]    = sum_w
        reg["sum_wi"]   = sum_wi
        reg["sum_wj"]   = sum_wj
 
        print(f"  R({reg['label']:<7})  "
              f"({X:.2f}, {Y:.2f}){'':10}  "
              f"{sum_wi:<15.1f}  {sum_wj:<15.1f}  {sum_w:.1f}")
 
    return regions