# src/level_set_iteration/step_3.py

import numpy as np

from .step_1 import dirac_smooth, compute_curvature, compute_laplacian

# =========================================================================
# STEP 3: Update φ
# =========================================================================

def update_phi(phi: np.ndarray, p_bar: np.ndarray, l1: float, l2: float, vartheta: float, config) -> np.ndarray:

    delta = dirac_smooth(phi, config.epsilon)
    kappa = compute_curvature(phi)
    lap = compute_laplacian(phi)

    data_term = delta * (
        -config.alpha1 * (p_bar - l1) ** 2
        + config.alpha2 * (p_bar - l2) ** 2
    )

    curvature_term = vartheta * delta * kappa
    drpt_term = config.theta * (lap - kappa)

    dphi_dt = data_term + curvature_term + drpt_term
    return phi + config.dt * dphi_dt