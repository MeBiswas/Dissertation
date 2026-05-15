# src/features_extraction/main.py

import numpy as np

from .step_1 import extract_sr_region
from .step_5 import compute_feature_vector
from .step_6 import save_all
from .step_7 import visualize_feature_extraction
from src.utils import create_run_folder

"""
Stage 3 — Step 1: Feature Extraction from Segmented SRs
Implements Section III-A of Pramanik et al. (2018)
 
Extracts a 21-element feature vector from each breast's segmented SR:
    - 14 Haralick features  (texture, from GLCM — Equation 22)
    - 7  Hu's moment invariants (shape)
    Concatenated → [f_v]_{1×21}
 
Paper reference: Section III-A "Feature Extraction and Classifier Design"
 
    Eq 22: g_{p_b^s}(u,v) = G_sym(u,v / 1, 0°)
                             ────────────────────────────────
                             ΣΣ G_sym(u,v / 1, 0°)
 
    where G_sym(u,v/1,0°) = G(u,v/1,0°) + G^T(u,v/1,0°)
"""

# ═════════════════════════════════════════════════════════════════════════════
# MAIN — Run feature extraction for both breasts
# ═════════════════════════════════════════════════════════════════════════════
def run_feature_pipeline(
    preprocessed_img,
    sr_left,
    sr_right,
    config,
    image_name="image"
):
    print(f"[Process] Feature Extraction → {image_name}")

    run_dir = create_run_folder(config.output_dir, image_name)

    # --- LEFT ---
    print("\n[LEFT]")
    sr_left_region = extract_sr_region(sr_left, preprocessed_img)

    if sr_left_region is None:
        f_left = np.zeros(21)
        g_left = np.zeros((2,2))
        h_left = np.zeros(14)
        hu_left = np.zeros(7)
    else:
        f_left, g_left, h_left, hu_left = compute_feature_vector(sr_left_region)

    # --- RIGHT ---
    print("\n[RIGHT]")
    sr_right_region = extract_sr_region(sr_right, preprocessed_img)

    if sr_right_region is None:
        f_right = np.zeros(21)
        g_right = np.zeros((2,2))
        h_right = np.zeros(14)
        hu_right = np.zeros(7)
    else:
        f_right, g_right, h_right, hu_right = compute_feature_vector(sr_right_region)

    # --- SAVE ---
    if config.save_results:
        save_all(run_dir, f_left, f_right, g_left, g_right)

    # --- VISUALIZE (only LEFT for consistency with your code) ---
    if config.save_visualization or config.show_visualization:
        visualize_feature_extraction(
            preprocessed_img,
            sr_left,
            g_left,
            h_left,
            hu_left,
            run_dir,
            show=config.show_visualization
        )

    return {
        "f_v_left": f_left,
        "f_v_right": f_right,
        "run_dir": run_dir
    }