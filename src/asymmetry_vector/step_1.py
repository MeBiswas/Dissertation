# src/asymmetry_vector/step_1.py

"""
Stage 3 — Step 2: Asymmetry Feature Vector
Implements Section III-A of Pramanik et al. (2018)
 
    F = |f_v^(L) - f_v^(R)|
 
    where:
        f_v^(L) = 21-element feature vector of LEFT  breast SR
        f_v^(R) = 21-element feature vector of RIGHT breast SR
        F       = 21-element ASYMMETRY vector (input to classifier)
 
Paper reference: Section III-A
    "Then, the asymmetry feature vector [F]_{1×21} is calculated for each
     patient breast thermogram by taking the absolute difference between
     corresponding elements of f_v^(L) and f_v^(R),
     i.e., F = |f_v^(L) - f_v^(R)|"
 
Clinical insight:
    A healthy person's left and right breasts should have SIMILAR thermal
    patterns → their feature vectors should be similar → F ≈ 0 everywhere.
    An abnormal breast will show asymmetry → F will have large values.
    This asymmetry signal is what the neural network learns to classify.
"""

import numpy as np

# ═════════════════════════════════════════════════════════════════════════════
# CORE FUNCTION — Asymmetry Feature Vector
# ═════════════════════════════════════════════════════════════════════════════
 
def compute_asymmetry_vector(f_v_left: np.ndarray, f_v_right: np.ndarray) -> np.ndarray:
    # ── Input validation ──────────────────────────────────────────────────────
    if f_v_left.shape != (21,) or f_v_right.shape != (21,):
        raise ValueError(
            f"Expected feature vectors of shape (21,). "
            f"Got left={f_v_left.shape}, right={f_v_right.shape}.\n"
            "Make sure feature_extraction.py ran correctly."
        )
 
    # ── Core computation: F = |f_v^(L) - f_v^(R)| ────────────────────────────
    F = np.abs(f_v_left - f_v_right) # element-wise absolute difference
 
    return F # shape: (21,), all values ≥ 0