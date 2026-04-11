# sch_cs/step_7.py

# ── Standard library ──────────────────────────────────────────────────────────
from typing import Dict, List

# ── Numeric ───────────────────────────────────────────────────────────
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2.7 — Weighted centroids  [Equation 4]
# ─────────────────────────────────────────────────────────────────────────────

def compute_centroids(pb: np.ndarray, regions: List[Dict]) -> List[Dict]:
    print(f'\n[CS 2.7] Weighted centroids for {len(regions)} regions...')
    for reg in regions:
        coords  = reg['coords']
        weights = pb[coords[:, 0], coords[:, 1]].astype(float)
        sw      = weights.sum()
        X       = float((weights * coords[:, 0]).sum() / sw)
        Y       = float((weights * coords[:, 1]).sum() / sw)
        reg['centroid'] = (X, Y)
        print(f'  R({reg["label"]:>4}): centroid=({X:.1f},{Y:.1f}), size={reg["size"]}px')
    return regions