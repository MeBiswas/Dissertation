# src/level_set_initialization/step_2.py

import numpy as np

def initialize_phi(p_th_b, inside_value, outside_value):
    
    phi = inside_value * p_th_b - (1.0 - p_th_b) * abs(outside_value)

    # --- SAME SANITY CHECKS ---
    n_inside = np.sum(phi > 0)
    n_outside = np.sum(phi < 0)
    n_total = phi.size

    print(f"[Eq 17] φ initialized")
    print(f"Shape: {phi.shape}")
    print(f"Inside: {n_inside} ({100*n_inside/n_total:.1f}%)")
    print(f"Outside: {n_outside} ({100*n_outside/n_total:.1f}%)")
    print(f"Values: {np.unique(phi)}")

    return phi