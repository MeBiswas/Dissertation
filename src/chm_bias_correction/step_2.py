# src/chm_bias_correction/step_2.py

import numpy as np

def compute_nc(image: np.ndarray, order: int, eps: float):
    p = image.astype(np.float64)
    n = order

    mask = p > 0
    numerator = np.sum(p[mask] ** (n + 1))
    denominator = np.sum(p[mask] ** n) + eps

    Nc = numerator / denominator
    return Nc