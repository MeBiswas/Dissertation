# src/features_extraction/step_6.py

import os
import numpy as np

def save_all(run_dir, f_left, f_right, g_left, g_right):

    np.save(os.path.join(run_dir, "f_v_left.npy"), f_left)
    np.save(os.path.join(run_dir, "f_v_right.npy"), f_right)

    np.save(os.path.join(run_dir, "glcm_left.npy"), g_left)
    np.save(os.path.join(run_dir, "glcm_right.npy"), g_right)

    # readable file
    with open(os.path.join(run_dir, "features.txt"), "w") as f:
        f.write("LEFT:\n")
        f.write(str(f_left) + "\n\n")
        f.write("RIGHT:\n")
        f.write(str(f_right) + "\n")

    print(f"[Save] Features saved → {run_dir}")