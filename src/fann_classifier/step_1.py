# src/fann_classifier/step_1.py

import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

# ═════════════════════════════════════════════════════════════════════════════
# PART 1 — PERFORMANCE METRICS
# ═════════════════════════════════════════════════════════════════════════════
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()

    eps = 1e-10
    sensitivity = TP / (TP + FN + eps)
    specificity = TN / (TN + FP + eps)
    ppv         = TP / (TP + FP + eps)
    npv         = TN / (TN + FN + eps)
    accuracy    = (TP + TN) / (TP + TN + FP + FN + eps)

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float('nan')

    return {
        'accuracy'   : float(accuracy    * 100),
        'sensitivity': float(sensitivity * 100),
        'specificity': float(specificity * 100),
        'ppv'        : float(ppv         * 100),
        'npv'        : float(npv         * 100),
        'auc'        : auc,
        'TP': int(TP), 'TN': int(TN),
        'FP': int(FP), 'FN': int(FN),
    }