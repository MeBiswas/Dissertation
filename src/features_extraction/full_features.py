# src/features_extraction/full_features.py

import numpy as np
from skimage.feature import graycomatrix

from .hu_moments import compute_hu_moments
from .haralick_features import compute_glcm_normalized, compute_haralick_features

# ═════════════════════════════════════════════════════════════════════════════
# COMBINED: Full 21-element Feature Vector for one breast
# ═════════════════════════════════════════════════════════════════════════════
 
def extract_feature_vector(segmented_sr: np.ndarray, original_gray: np.ndarray) -> np.ndarray:
    """
    Extracts the full 21-element feature vector [f_v]_{1×21} from one
    breast's segmented SR, as described in Section III-A of the paper.
 
    Pipeline:
        segmented_sr (binary mask)
            → mask applied to original grayscale  → SR patch
            → GLCM (Eq 22)                        → normalized g
            → 14 Haralick features
            → 7  Hu's moment invariants
            → concatenate                          → [f_v]_{1×21}
 
    WHY apply mask to original_gray and not use segmented_sr directly?
        The segmented_sr is a binary mask (0/1). Using it directly for
        GLCM would only give 2 gray levels (no texture information).
        We need the ACTUAL intensity values within the SR region.
        So we extract only the pixels inside the SR from original_gray.
 
    Args:
        segmented_sr  : Binary mask from DLPE (1=SR, 0=background), uint8
        original_gray : The preprocessed grayscale TBI (p_b), uint8
 
    Returns:
        f_v : 21-element feature vector [14 Haralick | 7 Hu], float64
    """
    # ── Apply SR mask to get intensity values within the SR ───────────────────
    sr_region = original_gray * segmented_sr    # zero outside SR
 
    # ── Safety check: make sure SR is not empty ───────────────────────────────
    if sr_region.max() == 0 or segmented_sr.sum() == 0:
        print("[Warn]  SR region is empty! Returning zero feature vector.")
        return np.zeros(21, dtype=np.float64)
 
    # ── Part 1: 14 Haralick features ─────────────────────────────────────────
    print("[Feat]  Computing GLCM and 14 Haralick features...")
    g              = compute_glcm_normalized(sr_region)
    haralick_feats = compute_haralick_features(g)
    print(f"        Haralick features: {haralick_feats.round(4)}")
 
    # ── Part 2: 7 Hu's moment invariants ─────────────────────────────────────
    print("[Feat]  Computing 7 Hu's moment invariants...")
    hu_feats = compute_hu_moments(sr_region)
    print(f"        Hu's moments:      {hu_feats.round(4)}")
 
    # ── Concatenate to 21-element vector ─────────────────────────────────────
    f_v = np.concatenate([haralick_feats, hu_feats])   # shape: (21,)
    print(f"[Feat]  Feature vector shape: {f_v.shape}  ← ready for asymmetry step")
 
    return f_v                              # shape: (21,)