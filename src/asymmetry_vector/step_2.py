# src/asymmetry_vector/step_2.py

import os
import numpy as np

# ═════════════════════════════════════════════════════════════════════════════
# SAVE
# ═════════════════════════════════════════════════════════════════════════════
def save_asymmetry(F: np.ndarray, run_dir: str):
    os.makedirs(run_dir, exist_ok=True)

    # Save numpy
    np.save(os.path.join(run_dir, "F.npy"), F)

    # Save readable txt
    with open(os.path.join(run_dir, "asymmetry.txt"), "w") as f:
        f.write("Asymmetry Feature Vector (F)\n")
        f.write("="*40 + "\n")
        f.write(np.array2string(F, precision=4))

    print(f"[Save] Asymmetry saved → {run_dir}")