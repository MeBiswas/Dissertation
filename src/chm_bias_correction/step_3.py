# src/chm_bias_correction/step_3.py

import numpy as np

def compute_p_bar(image, chm_local, Nc, eps):
    p = image.astype(np.float64)

    chm_safe = np.where(chm_local < eps, eps, chm_local)
    p_bar = (p * Nc) / chm_safe

    p_bar[p == 0] = 0.0
    return p_bar