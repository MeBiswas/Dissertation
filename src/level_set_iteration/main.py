# src/level_set_iteration/main.py

import os
import numpy as np
from typing import Dict

from .step_6 import save_results
from .step_5 import iterate_level_set
from .step_7 import visualize_results
from .storage_config import LevelSetConfig

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

# =========================================================================
# LEVEL SET ITERATION
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
    
# =========================================================================
# LEVEL SET ITERATION PER SR
# =========================================================================
def run_level_set_per_sr(
    p_bar : np.ndarray,
    sr_regions : list,
    image_name : str,
    preprocessed_img : np.ndarray,
    config : LevelSetConfig,
    margin : int  = 40,
    verbose : bool = True,
    do_visualize : bool = True,
    do_save : bool = True,
) -> dict:
    H, W = p_bar.shape
    combined_segmented = np.zeros((H, W), dtype=np.uint8)
    combined_phi = np.full((H, W), -4.0, dtype=np.float64)
    all_histories = []

    for idx, sr in enumerate(sr_regions):
        coords = sr['coords']
        r0 = max(0, int(coords[:, 0].min()) - margin)
        r1 = min(H, int(coords[:, 0].max()) + margin)
        c0 = max(0, int(coords[:, 1].min()) - margin)
        c1 = min(W, int(coords[:, 1].max()) + margin)

        # Crop both p_bar and phi to this window
        pb_crop = p_bar[r0:r1, c0:c1].copy()

        # Build a phi seed for just this SR in the crop coordinate space
        phi_crop = np.full_like(pb_crop, -4.0, dtype=np.float64)
        phi_crop[sr['mask'][r0:r1, c0:c1]] = 4.0

        if verbose:
            print(f"\n[SR {idx+1}/{len(sr_regions)}] "
                  f"label={sr['label']}, size={sr['size']}px, "
                  f"crop=[{r0}:{r1},{c0}:{c1}]")

        phi_final_crop, seg_crop, history = iterate_level_set(
            pb_crop, phi_crop, n_sr=1, config=config, verbose=verbose)

        # Paste result back into full image
        combined_segmented[r0:r1, c0:c1] |= seg_crop
        combined_phi[r0:r1, c0:c1] = np.maximum(
            combined_phi[r0:r1, c0:c1], phi_final_crop
        )
        all_histories.append(history)

    # Merge histories for plotting
    merged = {k: [] for k in all_histories[0]}
    for h in all_histories:
        for k in h:
            merged[k].extend(h[k])

    # Save / visualise using existing run_level_set infrastructure
    saved_paths, run_dir = {}, None
    if do_save:
        saved_paths, run_dir = save_results(
            combined_phi, combined_segmented, merged,
            image_name, p_bar, None, config
        )
    if do_visualize:
        from scipy.ndimage import label as scipy_label
        # Build a dummy phi_init for visualisation
        phi_init_display = np.full((H,W), -4.0)
        for sr in sr_regions:
            phi_init_display[sr['mask']] = 4.0
        visualize_results(
            preprocessed_img, phi_init_display, combined_phi,
            combined_segmented, merged, config,
            save_path=None, show=True
        )

    return {
        'phi_final' : combined_phi,
        'segmented_sr' : combined_segmented,
        'history' : merged,
        'saved_paths' : saved_paths,
        'run_dir' : run_dir,
        'image_name' : image_name,
    }