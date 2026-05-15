# src/asymmetry_vector/step_1.py

import numpy as np

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

# ═════════════════════════════════════════════════════════════════════════════
# CORE FUNCTION — Asymmetry Feature Vector
# ═════════════════════════════════════════════════════════════════════════════
def compute_asymmetry(f_v_left: np.ndarray, f_v_right: np.ndarray) -> np.ndarray:
    if f_v_left.shape != (21,) or f_v_right.shape != (21,):
        raise ValueError(
            f"Expected feature vectors of shape (21,). "
            f"Got left={f_v_left.shape}, right={f_v_right.shape}."
        )

    F = np.abs(f_v_left - f_v_right)

    print(f"[Asym] Computed F vector → shape: {F.shape}")
    return F