# src/sr_segmentation/step_1.py

import numpy as np

"""
Utility: Split Segmented SR into Left and Right Breast Regions
--------------------------------------------------------------
Bridges the gap between dlpe_iteration_loop.py (outputs one full-image SR mask)
and feature_extraction.py (needs separate left/right SR masks).
 
The frontal TBI always has both breasts visible. After DLPE segmentation,
the SR mask contains blobs from BOTH breasts. This script:
    1. Finds the vertical centre of the breast region
    2. Splits the SR mask into left and right halves
    3. Validates each half has at least one SR blob
    4. Saves them as sr_left.npy and sr_right.npy
 
Note on left/right convention:
    Image LEFT  half → patient's RIGHT breast
    Image RIGHT half → patient's LEFT  breast
    The paper uses patient's perspective (f_v^L = patient's left breast).
    We follow image coordinates and clearly label which is which.
"""
# ─────────────────────────────────────────────────────────────────────────────
# FIND VERTICAL CENTRE OF BREAST REGION
# ─────────────────────────────────────────────────────────────────────────────
def find_vertical_centre(preprocessed_img: np.ndarray) -> int:
    h, w = preprocessed_img.shape
    nonzero = np.argwhere(preprocessed_img > 0)

    if len(nonzero) == 0:
        centre_col = w // 2
        print(f"[Split] No body pixels — using image centre {centre_col}")
        return centre_col

    col_min = int(nonzero[:, 1].min())
    col_max = int(nonzero[:, 1].max())
    centre_col = (col_min + col_max) // 2

    print(f"[Split] Body cols [{col_min},{col_max}] → centre = {centre_col} (width={w})")
    return centre_col