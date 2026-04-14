# src/level_set_iteration/step_3.py

import numpy as np

from .step_1 import dirac_smooth, compute_curvature, compute_laplacian

# =========================================================================
# STEP 3: φ Update (Equation 18)
# =========================================================================
def update_phi(
    phi: np.ndarray,
    p_bar: np.ndarray,
    l1: float,
    l2: float,
    vartheta: float,
    alpha1: float,
    alpha2: float,
    theta: float,
    epsilon: float,
    dt: float
) -> np.ndarray:
    delta = dirac_smooth(phi, epsilon)
    kappa = compute_curvature(phi)
    lap = compute_laplacian(phi)
    
    # Term 1: Data fitting
    data_term = delta * (
        -alpha1 * (p_bar - l1) ** 2
        + alpha2 * (p_bar - l2) ** 2
    )
    
    # Term 2: Curvature smoothing
    curvature_term = vartheta * delta * kappa
    
    # Term 3: DRPT (distance regularization)
    drpt_term = theta * (lap - kappa)
    
    dphi_dt = data_term + curvature_term + drpt_term
    return phi + dt * dphi_dt