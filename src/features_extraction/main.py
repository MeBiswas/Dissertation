# src/features_extraction/main.py

import cv2
import numpy as np

from .hu_moments import compute_hu_moments
from .full_features import extract_feature_vector
from .visualization import visualize_feature_extraction
from .haralick_features import compute_glcm_normalized, compute_haralick_features

# ═════════════════════════════════════════════════════════════════════════════
# MAIN — Run feature extraction for both breasts
# ═════════════════════════════════════════════════════════════════════════════
 
def run_feature_extraction(preprocessed_path: str,
                           sr_left_path: str,
                           sr_right_path: str,
                           save_output: bool = True
                           ) -> tuple[np.ndarray, np.ndarray]:
    """
    Runs the full feature extraction for both left and right breasts.
    Returns f_v_left and f_v_right — inputs to the asymmetry step.
 
    Args:
        preprocessed_path : Path to preprocessed grayscale TBI (p_b)
        sr_left_path      : Path to segmented SR mask — LEFT  breast
        sr_right_path     : Path to segmented SR mask — RIGHT breast
        save_output       : Save visualization
 
    Returns:
        f_v_left  : 21-element feature vector for left  breast
        f_v_right : 21-element feature vector for right breast
    """
    # ── Load images ───────────────────────────────────────────────────────────
    p_b       = cv2.imread(preprocessed_path, cv2.IMREAD_GRAYSCALE)
    sr_left   = cv2.imread(sr_left_path,      cv2.IMREAD_GRAYSCALE)
    sr_right  = cv2.imread(sr_right_path,     cv2.IMREAD_GRAYSCALE)
 
    for name, img in [("preprocessed", p_b),
                      ("SR left",  sr_left),
                      ("SR right", sr_right)]:
        if img is None:
            raise FileNotFoundError(f"Could not load {name} image.")
 
    # Binarize SR masks (0 or 1)
    _, sr_left  = cv2.threshold(sr_left,  127, 1, cv2.THRESH_BINARY)
    _, sr_right = cv2.threshold(sr_right, 127, 1, cv2.THRESH_BINARY)
 
    print("=" * 60)
    print("LEFT BREAST")
    print("=" * 60)
    f_v_left = extract_feature_vector(sr_left, p_b)
 
    print("\n" + "=" * 60)
    print("RIGHT BREAST")
    print("=" * 60)
    f_v_right = extract_feature_vector(sr_right, p_b)
 
    # ── Optional: visualize one breast ───────────────────────────────────────
    if save_output:
        sr_region = p_b * sr_left
        g_viz = compute_glcm_normalized(sr_region)
        h_viz = compute_haralick_features(g_viz)
        hu_viz = compute_hu_moments(sr_region)
        visualize_feature_extraction(
            original_gray   = p_b,
            segmented_sr    = sr_left,
            g               = g_viz,
            haralick_feats  = h_viz,
            hu_feats        = hu_viz,
            save_path       = "feature_extraction.png"
        )
 
    print("\n[Ready] Both feature vectors extracted.")
    print("        → Next: Asymmetry vector F = |f_v_left - f_v_right|")
 
    return f_v_left, f_v_right