# src/sr_segmentation/split_mask.py

import numpy as np

from .vertical_centre import find_vertical_centre

# ─────────────────────────────────────────────────────────────────────────────
# SPLIT SR MASK INTO LEFT AND RIGHT
# ─────────────────────────────────────────────────────────────────────────────
 
def split_sr_left_right(segmented_sr: np.ndarray, preprocessed_img: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    centre_col = find_vertical_centre(preprocessed_img)
 
    H, W = segmented_sr.shape
 
    # Create full-size masks (same shape as original) with zeros elsewhere
    sr_img_left  = np.zeros((H, W), dtype=np.uint8)
    sr_img_right = np.zeros((H, W), dtype=np.uint8)
 
    # Copy SR blobs from each half
    sr_img_left[:, :centre_col]  = segmented_sr[:, :centre_col]
    sr_img_right[:, centre_col:] = segmented_sr[:, centre_col:]
 
    # ── Validate: each half should have at least some SR pixels ──────────────
    n_left  = sr_img_left.sum()
    n_right = sr_img_right.sum()
 
    print(f"[Split] SR pixels — image left  half: {n_left}")
    print(f"[Split] SR pixels — image right half: {n_right}")
 
    if n_left == 0:
        print("[Warn]  No SR found in image LEFT half  "
              "(patient's right breast). "
              "This patient may only have SR in one breast.")
    if n_right == 0:
        print("[Warn]  No SR found in image RIGHT half "
              "(patient's left breast). "
              "This patient may only have SR in one breast.")
 
    return sr_img_left, sr_img_right, centre_col