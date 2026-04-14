# src/level_set_iteration/step_4.py

import numpy as np
from typing import Tuple
from scipy.stats import pearsonr

# =========================================================================
# STEP 4: Stopping Criterion (Equation 21)
# =========================================================================
def check_stopping(
    phi_prev: np.ndarray,
    phi_curr: np.ndarray,
    phi_next: np.ndarray,
    iteration: int,
    t_stop: float
) -> Tuple[bool, float, float]:
    if iteration <= 5:
        return False, 1.0, 1.0
    
    # ── Guard: don't stop if contour has collapsed (no positive phi pixels) ──
    if (phi_curr > 0).sum() == 0:
        return False, 1.0, 1.0
    
    r_prev, _ = pearsonr(phi_prev.ravel(), phi_curr.ravel())
    r_curr, _ = pearsonr(phi_curr.ravel(), phi_next.ravel())
    
    dev_prev = abs(1.0 - r_prev)
    dev_curr = abs(1.0 - r_curr)
    
    stop = (dev_prev <= t_stop) and (dev_curr <= t_stop)
    
    return stop, dev_prev, dev_curr