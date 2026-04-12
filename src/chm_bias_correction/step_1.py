# src/chm_bias_correction/step_1.py

import cv2
import numpy as np
from scipy.ndimage import uniform_filter

# def compute_chm_local(image: np.ndarray, window_size: int, order: int, eps: float):
#     p = image.astype(np.float64)
#     w = window_size
#     n = order

#     sum_num = uniform_filter(p ** (n + 1), size=w, mode='reflect') * (w ** 2)
#     sum_den = uniform_filter(p ** n, size=w, mode='reflect') * (w ** 2)

#     sum_den = np.where(sum_den == 0, eps, sum_den)

#     chm_local = sum_num / sum_den
#     return chm_local

def compute_chm_local(image, window_size=9, order=1):
    p      = image.astype(np.float64)
    k      = window_size
    kernel = np.ones((k, k), dtype=np.float64) / (k * k)

    sum_num = cv2.filter2D(p**(order+1), -1, kernel,
                           borderType=cv2.BORDER_CONSTANT) * (k*k)
    sum_den = cv2.filter2D(p**order,     -1, kernel,
                           borderType=cv2.BORDER_CONSTANT) * (k*k)

    safe_den = np.where(sum_den < 1e-6, 1e-6, sum_den)
    chm      = sum_num / safe_den
    chm[p == 0] = 0.0          # keep background clean
    return chm