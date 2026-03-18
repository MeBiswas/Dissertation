# sch_cs/cs_isolation.py

import numpy as np

from src.utils import section

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4D — CS Isolation Algorithm
# ─────────────────────────────────────────────────────────────────────────────
 
"""
    CS (Centroid-knowledge of SRs) isolation algorithm.

    Iteratively eliminates non-SR regions based on X-coordinate of centroids.
    SRs lie horizontally near the breast centre — regions far above/below
    (neck, armpit, fold) have very different X-coordinates and get eliminated.

    Round 1,3,5...: eliminate regions where centroid X < C_avg(X)
    Round 2,4,6...: eliminate regions where centroid X > C_avg(X)
    Stop when: max|R(k)_c(X) - C_avg(X)| <= epsilon  (epsilon=35)
"""
def cs_isolation(regions: list, epsilon: float = 35.0) -> list:
    section("STEP 4D — CS Isolation Algorithm")
    print(f"  epsilon = {epsilon}")
    print(f"  Starting with {len(regions)} region(s)")
    print(f"\n  Initial centroids (X=row, Y=col):")
    for r in regions:
        X, Y = r["centroid"]
        print(f"    R({r['label']}): X={X:.2f}, Y={Y:.2f}")
 
    active    = regions.copy()
    round_num = 0
 
    while len(active) > 1:
        round_num += 1
        X_vals  = [r["centroid"][0] for r in active]
        C_avg_X = float(np.mean(X_vals))
 
        print(f"\n  ── Round {round_num} ─────────────────────────────────")
        print(f"  C_avg({round_num})(X) = {C_avg_X:.4f}")
 
        max_dev = max(abs(r["centroid"][0] - C_avg_X) for r in active)
        print(f"  max|R(k)_c(X) - C_avg(X)| = {max_dev:.4f}  "
              f"(need <= {epsilon} to stop)")
 
        if max_dev <= epsilon:
            print(f"  ✓ Converged — stopping")
            break
 
        if round_num % 2 == 1:
            surviving  = [r for r in active
                          if r["centroid"][0] >= C_avg_X]
            eliminated = [r for r in active
                          if r["centroid"][0] <  C_avg_X]
            rule = f"X < {C_avg_X:.2f}"
        else:
            surviving  = [r for r in active
                          if r["centroid"][0] <= C_avg_X]
            eliminated = [r for r in active
                          if r["centroid"][0] >  C_avg_X]
            rule = f"X > {C_avg_X:.2f}"
 
        print(f"  Eliminate where {rule}:")
        for r in eliminated:
            X, Y = r["centroid"]
            print(f"    ✗ R({r['label']}): X={X:.2f}  eliminated")
        for r in surviving:
            X, Y = r["centroid"]
            print(f"    ✓ R({r['label']}): X={X:.2f}  kept")
 
        active = surviving
        if len(active) <= 1:
            break
 
    section("CS ISOLATION — FINAL RESULT")
    print(f"  {len(active)} SR(s) identified:")
    for r in active:
        X, Y = r["centroid"]
        print(f"    R({r['label']}): centroid=({X:.2f},{Y:.2f}), "
              f"size={r['size']}px")
 
    return active