# src/fann_classifier/step_1.py

"""
Stage 3 — Step 3: Feed-Forward Artificial Neural Network (FANN) Classifier
Implements Section III-A (Classifier Design) of Pramanik et al. (2018)
 
Network Architecture (exactly as described in paper):
    Input  layer : 21 neurons  — one per element of F
                   Transfer fn : Linear
    Hidden layer : 42 neurons  — experimentally chosen (2× input)
                   Transfer fn : Hyperbolic tangent (tanh)
    Output layer :  1 neuron   — 0=Normal, 1=Abnormal
                   Transfer fn : Softmax (sigmoid for binary)
 
Training:
    Algorithm    : Levenberg-Marquardt back-propagation
    Learning rate: 0.1
    Evaluation   : 5-fold cross-validation
 
Performance metrics (Table IV in paper):
    Sensitivity, Specificity, PPV, NPV, Accuracy, AUC
 
Paper's reported results:
    Sensitivity=87%, Specificity=89%, PPV=87%, NPV=89%,
    Accuracy=88.5%, AUC=0.939
"""

import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

# ═════════════════════════════════════════════════════════════════════════════
# PART 1 — PERFORMANCE METRICS
# ═════════════════════════════════════════════════════════════════════════════
 
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    eps = 1e-10
 
    # ── Confusion matrix ──────────────────────────────────────────────────────
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
 
    sensitivity = tp / (tp + fn + eps) # Recall
    specificity = tn / (tn + fp + eps)
    ppv         = tp / (tp + fp + eps) # Precision
    npv         = tn / (tn + fn + eps)
    accuracy    = (tp + tn) / (tp + tn + fp + fn + eps)
    auc_score   = roc_auc_score(y_true, y_prob)
 
    return {
        'sensitivity' : sensitivity,
        'specificity' : specificity,
        'ppv'         : ppv,
        'npv'         : npv,
        'accuracy'    : accuracy,
        'auc'         : auc_score,
        # Keep raw counts for reporting
        'tp': int(tp), 'tn': int(tn),
        'fp': int(fp), 'fn': int(fn)
    }