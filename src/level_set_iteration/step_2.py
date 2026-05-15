# src/level_set_iteration/step_2.py

import numpy as np
from typing import Tuple

from .step_1 import heaviside_smooth

# =========================================================================
# STEP 2: Region Means (Equations 19-20)
# =========================================================================
def update_region_means(p_bar: np.ndarray, phi: np.ndarray, epsilon: float) -> Tuple[float, float]:
    H = heaviside_smooth(phi, epsilon)

    numerator_l1 = np.sum(p_bar * H)
    denominator_l1 = np.sum(H)
    l1 = numerator_l1 / (denominator_l1 + 1e-10)

    one_minus_H = 1.0 - H
    numerator_l2 = np.sum(p_bar * one_minus_H)
    denominator_l2 = np.sum(one_minus_H)
    l2 = numerator_l2 / (denominator_l2 + 1e-10)

    return l1, l2