# src/sr_segmentation/step_2.py

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# SPLIT SR MASK INTO LEFT AND RIGHT
# ─────────────────────────────────────────────────────────────────────────────
def split_sr(segmented_sr, centre_col):
    H, W = segmented_sr.shape
    
    if centre_col is None:
        centre_col = W//2

    sr_left  = np.zeros((H, W), dtype=np.uint8)
    sr_right = np.zeros((H, W), dtype=np.uint8)

    sr_left[:, :centre_col]  = segmented_sr[:, :centre_col]
    sr_right[:, centre_col:] = segmented_sr[:, centre_col:]

    n_left = sr_left.sum()
    n_right = sr_right.sum()

    print(f"[Split] Left pixels : {n_left}")
    print(f"[Split] Right pixels : {n_right}")

    if n_left == 0:
        print("[Warn] No SR in LEFT half")
    if n_right == 0:
        print("[Warn] No SR in RIGHT half")

    return sr_left, sr_right