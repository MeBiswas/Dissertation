# src/level_set_initialization/step_2.py

import numpy as np

# ----------------------------------------------------------------------------
# Core: Initialize φ (Equation 17)
# ----------------------------------------------------------------------------
def initialize_phi(
    p_th_b: np.ndarray,
    inside_value: float = 4.0,
    outside_value: float = -4.0
) -> np.ndarray:
    b = p_th_b.astype(np.float64)
    phi = inside_value * b - (1.0 - b) * abs(outside_value)
    
    n_inside = int(np.sum(phi > 0))
    n_outside = int(np.sum(phi < 0))
    n_total = phi.size
    
    print(f"\n[Eq 17] φ initialized:")
    print(f"  Shape: {phi.shape}")
    print(f"  Inside (+{inside_value}): {n_inside} px ({100*n_inside/n_total:.2f}%)")
    print(f"  Outside ({outside_value}): {n_outside} px ({100*n_outside/n_total:.2f}%)")
    print(f"  Unique values: {np.unique(phi)}")
    
    return phi