# src/level_set_iteration/step_5.py

import numpy as np
from typing import Dict, Tuple

from .storage_config import LevelSetConfig
from .step_2 import update_region_means
from .step_3 import update_phi
from .step_4 import check_stopping

# =========================================================================
# STEP 5: Adaptive ϑ (Section IV)
# =========================================================================
def compute_vartheta(n_sr: int) -> float:
    return (0.1 * 255.0) / max(n_sr, 1)

# =========================================================================
# STEP 5: Main Iteration Loop
# =========================================================================
def iterate_level_set(
    p_bar: np.ndarray,
    phi_init: np.ndarray,
    n_sr: int,
    config: LevelSetConfig,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    # Adaptive ϑ
    vartheta = compute_vartheta(n_sr)

    p = p_bar.astype(np.float64)
    p_max = p.max()
    if p_max > 1.0:
        p = p / p_max

    if verbose:
        print(f"\n[Level Set] Initializing iteration loop")
        print(f"n_sr = {n_sr} → ϑ = (0.1×255)/{n_sr} = {vartheta:.4f}")
        print(f"α1={config.alpha1}, α2={config.alpha2}, θ={config.theta}")
        print(f"ε={config.epsilon}, dt={config.dt}, t_stop={config.t_stop}")
        print(f"max_iterations={config.max_iterations}")
        print("-" * 60)

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
    l1 = l2 = 0.0

    for n in range(config.max_iterations):

        # Step A: Update region means
        l1, l2 = update_region_means(p, phi, config.epsilon)

        # Step B: Update φ
        phi_curr = phi.copy()
        phi_next = update_phi(
            phi_curr, p, l1, l2, vartheta,
            config.alpha1, config.alpha2, config.theta,
            config.epsilon, config.dt
        )

        # Check stopping criterion
        stop, dev_prev, dev_curr = check_stopping(
            phi_prev, phi_curr, phi_next, n, config.t_stop
        )

        # Record history
        history['l1'].append(l1)
        history['l2'].append(l2)
        history['dev_prev'].append(dev_prev)
        history['dev_curr'].append(dev_curr)
        history['iteration'].append(n)

        # Verbose output
        if verbose and (n % config.verbose_every == 0 or stop):
            print(f"Iter {n:4d} | l1={l1:8.3f}  l2={l2:8.3f} | "
                  f"diff={l1-l2:7.2f} | |1-r|={dev_curr:.4f}")

        phi_prev = phi_curr
        phi = phi_next

        if stop:
            final_n = n
            if verbose:
                print(f"\n[Stop] Criterion met at iteration {n}.")
            break
    else:
        if verbose:
            print(f"\n[Warn] Max iterations ({config.max_iterations}) reached.")

    segmented_sr = (phi > 0).astype(np.uint8)

    if verbose:
        print(f"Final l1={l1:.3f}, l2={l2:.3f}, diff={l1-l2:.3f}")
        print(f"SR pixels: {segmented_sr.sum()} ({100*segmented_sr.mean():.2f}%)")

    return phi, segmented_sr, history