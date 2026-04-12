# src/features_extraction/step_2.py

import numpy as np
from skimage.feature import graycomatrix

# ═════════════════════════════════════════════════════════════════════════════
# PART 1 — GLCM
# ═════════════════════════════════════════════════════════════════════════════
def compute_glcm_normalized(image: np.ndarray) -> np.ndarray:
    # ── Ensure uint8 ──────────────────────────────────────────────────────────
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
 
    # ── Get the number of gray levels present in the image ───────────────────
    # Using full 256 levels is standard but computationally heavy.
    # The paper uses L = number of gray levels in p_b^s (the segmented image).
    # We reduce to levels actually present for efficiency.
    n_levels = int(image.max()) + 1
    n_levels = max(n_levels, 2)             # need at least 2 levels
 
    # ── Compute raw GLCM  G(u,v / distance=1, angle=0°) ─────────────────────
    # skimage returns shape (L, L, n_distances, n_angles)
    # We use distance=1, angle=0 (horizontal) as the paper specifies
    glcm_raw = graycomatrix(
        image,
        distances=[1],
        angles=[0], # 0° = horizontal pairs
        levels=n_levels,
        symmetric=False, # we manually symmetrize below
        normed=False # we manually normalize below
    )
    G = glcm_raw[:, :, 0, 0].astype(np.float64)    # shape: (L, L)
 
    # ── Symmetrize: G_sym = G + G^T  (as in Eq 22) ───────────────────────────
    G_sym = G + G.T
 
    # ── Normalize: g = G_sym / ΣΣ G_sym ─────────────────────────────────────
    total = G_sym.sum()
    if total == 0:
        raise ValueError("GLCM is all zeros. Is the segmented SR empty?")
    g = G_sym / total # now g sums to 1.0
 
    return g # shape: (L, L), float64