# src/features_extraction/step_2.py

import numpy as np
from skimage.feature import graycomatrix

# ═════════════════════════════════════════════════════════════════════════════
# PART 1 — GLCM
# ═════════════════════════════════════════════════════════════════════════════
def compute_glcm_normalized(image: np.ndarray, n_levels: int = 64) -> np.ndarray:
    if image.max() == 0:
        return np.ones((n_levels, n_levels), dtype=np.float64) / (n_levels ** 2)

    img_scaled = (image.astype(np.float64) / 255.0 * (n_levels - 1)).astype(np.uint8)

    G = graycomatrix(
        img_scaled,
        distances=[1],
        angles=[0],
        levels=n_levels,
        symmetric=False,
        normed=False
    )[:, :, 0, 0].astype(np.float64)

    G_sym  = G + G.T
    total  = G_sym.sum()
    if total == 0:
        raise ValueError("GLCM is all zeros — SR region may be empty.")

    return G_sym / total