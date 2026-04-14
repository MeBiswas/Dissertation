# src/chm_bias_correction/step_1.py

import cv2
import numpy as np

def compute_chm_local(image, window_size=9, order=1):
    p = image.astype(np.float64)
    k = window_size
    kernel = np.ones((k, k), dtype=np.float64) / (k * k)

    # Use BORDER_REPLICATE or BORDER_REFLECT instead of CONSTANT
    sum_num = cv2.filter2D(p**(order+1), -1, kernel, borderType=cv2.BORDER_REPLICATE) * (k*k)
    sum_den = cv2.filter2D(p**order, -1, kernel, borderType=cv2.BORDER_REPLICATE) * (k*k)

    chm = sum_num / (sum_den + 1e-10)
    chm[p == 0] = 0.0
    return chm