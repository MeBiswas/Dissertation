# src/sr_segmentation/vertical_centre.py

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
 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

# ─────────────────────────────────────────────────────────────────────────────
# FIND VERTICAL CENTRE OF BREAST REGION
# ─────────────────────────────────────────────────────────────────────────────
 
def find_vertical_centre(preprocessed_img: np.ndarray) -> int:
    h, w = preprocessed_img.shape
 
    # Sum intensities along each column
    col_sums = preprocessed_img.sum(axis=0).astype(np.float64)
 
    # Search only in the middle third of the image
    left_bound  = w // 3
    right_bound = 2 * w // 3
    middle_sums = col_sums[left_bound:right_bound]
 
    # The sternum appears as a local minimum (dark vertical gap)
    centre_col = left_bound + int(np.argmin(middle_sums))
 
    print(f"[Split] Vertical centre found at column {centre_col} "
          f"(image width={w})")
 
    return centre_col