# src/level_set_initialization/step_1.py

import numpy as np

def prepare_binary(schcs_binary: np.ndarray):
    p_th_b = schcs_binary.astype(np.float64)

    if p_th_b.max() > 1.0:
        p_th_b = p_th_b / 255.0
        print("[Init] Binary normalized [0,255] → [0,1]")

    unique_vals = np.unique(p_th_b)

    if not np.all(np.isin(unique_vals, [0.0, 1.0])):
        print(f"[Warn] Unexpected values: {unique_vals} → thresholding @0.5")
        p_th_b = (p_th_b >= 0.5).astype(np.float64)

    return p_th_b