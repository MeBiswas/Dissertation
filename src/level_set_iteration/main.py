# src/level_set_iteration/main.py

"""
Stage 2 — Step 3: DLPE Level Set Iteration Loop
Implements Equations 18, 19, 20, 21 from Pramanik et al. (2018)

Paper reference: Section II-C, subsections 2) "Energy minimization and numerical implementation"

Equations implemented:
    Eq 13 : DLPE energy functional in level set form    (used implicitly)
    Eq 19 : l1 — mean intensity INSIDE  contour         (update step A)
    Eq 20 : l2 — mean intensity OUTSIDE contour         (update step A)
    Eq 18 : dφ/dt — gradient descent update for φ       (update step B)
    Eq 21 : Stopping criterion using correlation        (convergence check)

Supporting functions (not explicit equations but required numerically):
    H_ε(φ)     — Smooth Heaviside function
    δ_ε(φ)     — Smooth Dirac delta (derivative of H_ε)
    div(∇φ/|∇φ|) — Mean curvature of φ (curvature term)
    ∇²φ        — Laplacian of φ (regularization term)

Parameters (from paper Section IV):
    alpha1 = alpha2 = 1       (data fitting weights)
    theta  = 0.2              (DRPT regularization weight)
    vartheta = (0.1*255)/k    (smoothness weight, adaptive to num SRs k)
    t_stop = 0.05             (stopping threshold)
    epsilon = 1.5             (smoothness of Heaviside/Dirac approximation)
    dt = 0.1                  (time step for gradient descent)
"""
import os
import numpy as np
from typing import Dict

from .storage_config import LevelSetConfig
from .step_5 import iterate_level_set
from .step_6 import save_results
from .step_7 import visualize_results

# =========================================================================
# Pipeline
# =========================================================================
def run_level_set(
    p_bar: np.ndarray,
    phi_init: np.ndarray,
    n_sr: int,
    image_name: str,
    preprocessed_img: np.ndarray,
    config: LevelSetConfig,
    verbose: bool = True,
    do_visualize: bool = True,
    do_save: bool = True
) -> Dict:
    
    # Input validation
    assert p_bar.shape == phi_init.shape, "Shape mismatch: p_bar vs phi_init"
    assert p_bar.dtype == np.float64, "p_bar must be float64"
    assert phi_init.dtype == np.float64, "phi_init must be float64"
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"[Level Set] Processing: {image_name}")
        print(f"{'='*60}")
        print(f"  p_bar shape: {p_bar.shape}, dtype: {p_bar.dtype}")
        print(f"  phi_init shape: {phi_init.shape}, values: {np.unique(phi_init)}")
        print(f"  n_sr: {n_sr}")
    
    # Run iteration
    phi_final, segmented_sr, history = iterate_level_set(
        p_bar, phi_init, n_sr, config, verbose
    )
    
    # Save results
    saved_paths, run_dir = {}, None
    if do_save:
        saved_paths, run_dir = save_results(
            phi_final, segmented_sr, history,
            image_name, p_bar, phi_init, config
        )
        if verbose:
            print(f"\n[Save] Results saved to: {run_dir}")
    
    # Visualize
    if do_visualize:
        viz_path = None
        if do_save and run_dir:
            viz_path = os.path.join(run_dir, f"{image_name}_level_set_results.png")
        
        visualize_results(
            preprocessed_img, phi_init, phi_final, segmented_sr,
            history, config, save_path=viz_path, show=True
        )
    
    return {
        'phi_final': phi_final,
        'segmented_sr': segmented_sr,
        'history': history,
        'saved_paths': saved_paths,
        'run_dir': run_dir,
        'image_name': image_name
    }