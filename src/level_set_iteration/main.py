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

from .step_5 import iterate_level_set
from .step_6 import save_results
from .step_7 import visualize_results

# =========================================================================
# Pipeline
# =========================================================================
def run_level_set(p_bar, phi_init, n_sr, image_name, preprocessed_img, config, verbose=True, do_visualize=True, do_save=True):

    phi_final, segmented_sr, history = iterate_level_set(
        p_bar, phi_init, n_sr, config, verbose
    )

    saved_paths, run_dir = {}, None

    if do_save:
        saved_paths, run_dir = save_results(
            phi_final, segmented_sr, history,
            image_name, p_bar, phi_init, config
        )

    if do_visualize:
        visualize_results(
            preprocessed_img, phi_init,
            phi_final, segmented_sr,
            history, config
        )

    return {
        'phi_final': phi_final,
        'segmented_sr': segmented_sr,
        'history': history,
        'saved_paths': saved_paths,
        'run_dir': run_dir,
        'image_name': image_name
    }