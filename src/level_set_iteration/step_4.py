# src/level_set_iteration/step_4.py

from scipy.stats import pearsonr

# =========================================================================
# STEP 4: Stopping Criterion
# =========================================================================

def check_stopping(phi_prev, phi_curr, phi_next, iteration: int, t_stop: float):

    if iteration <= 5:
        return False, 1.0, 1.0

    r_prev, _ = pearsonr(phi_prev.ravel(), phi_curr.ravel())
    r_curr, _ = pearsonr(phi_curr.ravel(), phi_next.ravel())

    dev_prev = abs(1.0 - r_prev)
    dev_curr = abs(1.0 - r_curr)

    stop = (dev_prev <= t_stop) and (dev_curr <= t_stop)

    return stop, dev_prev, dev_curr