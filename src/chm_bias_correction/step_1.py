# src/chm_bias_correction/step_1.py

import cv2
import numpy as np

def compute_chm_local(image: np.ndarray, window_size: int = 9, order: int = 1) -> np.ndarray:
    p = image.astype(np.float64)
    k = window_size
    kernel = np.ones((k, k), dtype=np.float64) / (k * k)

    sum_num = cv2.filter2D(p**(order+1), -1, kernel, borderType=cv2.BORDER_REFLECT) * (k*k)
    sum_den = cv2.filter2D(p**order, -1, kernel, borderType=cv2.BORDER_REFLECT) * (k*k)

    safe_den = np.where(sum_den < 1e-10, 1e-10, sum_den)
    chm = sum_num / safe_den
    chm[p == 0] = 0.0
    return chm