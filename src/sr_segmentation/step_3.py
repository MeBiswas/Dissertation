# src/sr_segmentation/step_3.py

import os
import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────
def save_all(run_dir, sr_left, sr_right, centre_col):
    
    # Images
    cv2.imwrite(os.path.join(run_dir, "sr_left.png"),  sr_left * 255)
    cv2.imwrite(os.path.join(run_dir, "sr_right.png"), sr_right * 255)

    # Arrays
    np.save(os.path.join(run_dir, "sr_left.npy"),  sr_left)
    np.save(os.path.join(run_dir, "sr_right.npy"), sr_right)

    # Metadata
    with open(os.path.join(run_dir, "metadata.txt"), "w") as f:
        f.write(f"centre_col: {centre_col}\n")

    print(f"[Save] Results saved → {run_dir}")