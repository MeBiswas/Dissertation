# src/asymmetry_vector/step_2.py

import numpy as np

# ═════════════════════════════════════════════════════════════════════════════
# DATASET-LEVEL — Build F for all patients
# ═════════════════════════════════════════════════════════════════════════════
 
def build_asymmetry_dataset(
    feature_vectors_left: np.ndarray,
    feature_vectors_right: np.ndarray,
    labels: np.ndarray
) -> tuple[np.ndarray,np.ndarray]:
    if feature_vectors_left.shape != feature_vectors_right.shape:
        raise ValueError("Left and right feature matrix shapes must match.")
 
    N = feature_vectors_left.shape[0]
    F_dataset = np.abs(feature_vectors_left - feature_vectors_right)
 
    # ── Print summary statistics separated by class ───────────────────────────
    print(f"[Dataset] {N} patients processed → F_dataset shape: {F_dataset.shape}")
 
    normal_mask   = (labels == 0)
    abnormal_mask = (labels == 1)
 
    if normal_mask.any() and abnormal_mask.any():
        mean_normal   = F_dataset[normal_mask].mean(axis=0)
        mean_abnormal = F_dataset[abnormal_mask].mean(axis=0)
 
        print(f"\n  Mean asymmetry — Normal   patients: "
              f"{mean_normal.round(4)}")
        print(f"  Mean asymmetry — Abnormal patients: "
              f"{mean_abnormal.round(4)}")
        print(f"\n  Overall: abnormal patients show "
              f"{(mean_abnormal > mean_normal).sum()}/21 features "
              f"with higher asymmetry (expected: most features)")
 
    return F_dataset, labels