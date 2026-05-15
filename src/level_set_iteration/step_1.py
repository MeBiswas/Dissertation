# src/level_set_iteration/step_1.py

import numpy as np

# =========================================================================
# STEP 1: Mathematical Building Blocks (Equations 13-16)
# =========================================================================
def heaviside_smooth(phi: np.ndarray, epsilon: float) -> np.ndarray:
    return 0.5 * (1.0 + (2.0 / np.pi) * np.arctan(phi / epsilon))

def dirac_smooth(phi: np.ndarray, epsilon: float) -> np.ndarray:
    return (1.0 / np.pi) * (epsilon / (epsilon**2 + phi**2))

def compute_curvature(phi: np.ndarray) -> np.ndarray:
    phi_y, phi_x = np.gradient(phi)
    grad_mag = np.sqrt(phi_x**2 + phi_y**2 + 1e-10)

    nx = phi_x / grad_mag
    ny = phi_y / grad_mag

    _, dnx_dx = np.gradient(nx)
    dny_dy, _ = np.gradient(ny)

    return dnx_dx + dny_dy

def compute_laplacian(phi: np.ndarray) -> np.ndarray:
    phi_y, phi_x = np.gradient(phi)
    phi_yy, _ = np.gradient(phi_y)
    _, phi_xx = np.gradient(phi_x)
    return phi_xx + phi_yy