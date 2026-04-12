# src/chm_bias_correction/step_4.py

import os
import cv2
import numpy as np

def save_all_outputs(run_dir, p_bar, chm_local, Nc, eps=1e-10):
    
    # --- Save corrected image ---
    img_path = os.path.join(run_dir, "corrected.png")
    p_scaled = p_bar / (p_bar.max() + eps) * 255
    cv2.imwrite(img_path, p_scaled.astype(np.uint8))

    # --- Save numpy arrays ---
    np.save(os.path.join(run_dir, "p_bar.npy"), p_bar)
    np.save(os.path.join(run_dir, "chm_local.npy"), chm_local)

    # --- Save scalar ---
    with open(os.path.join(run_dir, "nc.txt"), "w") as f:
        f.write(f"Nc: {Nc}\n")

    return {
        "image": img_path,
        "folder": run_dir
    }