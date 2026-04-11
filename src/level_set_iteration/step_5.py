# src/level_set_iteration/step_5.py

import numpy as np

from .step_2 import update_region_means
from .step_3 import update_phi
from .step_4 import check_stopping

# =========================================================================
# STEP 5: Iteration Loop
# =========================================================================

def compute_vartheta(n_sr: int) -> float:
    return (0.1 * 255.0) / max(n_sr, 1)


def iterate_level_set(p_bar: np.ndarray, phi_init: np.ndarray, n_sr: int, config, verbose: bool = True):

    vartheta = compute_vartheta(n_sr)

    phi = phi_init.copy().astype(np.float64)
    phi_prev = phi.copy()

    history = {
        'l1': [],
        'l2': [],
        'dev_prev': [],
        'dev_curr': [],
        'iteration': [],
    }

    final_iteration = config.max_iterations

    for n in range(config.max_iterations):

        l1, l2 = update_region_means(p_bar, phi, config.epsilon)

        phi_curr = phi.copy()
        phi_next = update_phi(phi_curr, p_bar, l1, l2, vartheta, config)

        stop, dev_prev, dev_curr = check_stopping(
            phi_prev, phi_curr, phi_next, n, config.t_stop
        )

        history['l1'].append(l1)
        history['l2'].append(l2)
        history['dev_prev'].append(dev_prev)
        history['dev_curr'].append(dev_curr)
        history['iteration'].append(n)

        if verbose and (n % config.verbose_every == 0 or stop):
            print(f"Iter {n:4d} | l1={l1:.3f} l2={l2:.3f} | "
                  f"|1-r_prev|={dev_prev:.4f} |1-r_curr|={dev_curr:.4f}")

        phi_prev = phi_curr
        phi = phi_next

        if stop:
            final_iteration = n
            break

    segmented_sr = (phi > 0).astype(np.uint8)

    return phi, segmented_sr, history