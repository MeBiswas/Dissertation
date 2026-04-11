# src/fann_classifier/step_3.py

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from .step_2 import build_fann
from .step_1 import compute_metrics

# ═════════════════════════════════════════════════════════════════════════════
# PART 3 — 5-FOLD CROSS-VALIDATION
# ═════════════════════════════════════════════════════════════════════════════
 
def run_cross_validation(F_dataset: np.ndarray, labels: np.ndarray, n_splits: int = 5) -> dict:
    # ── Setup ─────────────────────────────────────────────────────────────────
    skf      = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    per_fold = []
 
    # Accumulate predictions across all folds for the overall ROC curve
    all_y_true = []
    all_y_prob = []
 
    print("=" * 65)
    print(f"5-FOLD STRATIFIED CROSS-VALIDATION")
    print(f"Total patients: {len(labels)} "
          f"(Normal={int((labels==0).sum())}, "
          f"Abnormal={int((labels==1).sum())})")
    print("=" * 65)
 
    # ── Fold loop ─────────────────────────────────────────────────────────────
    for fold_idx, (train_idx, test_idx) in enumerate(
            skf.split(F_dataset, labels)):
 
        # Split data
        X_train, X_test = F_dataset[train_idx], F_dataset[test_idx]
        y_train, y_test = labels[train_idx],    labels[test_idx]
 
        # ── Normalize: fit on train, transform both ───────────────────────────
        # This is critical — never fit the scaler on test data
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)
 
        # ── Train FANN ────────────────────────────────────────────────────────
        model = build_fann()
        model.fit(X_train, y_train)
 
        # ── Predict ───────────────────────────────────────────────────────────
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]  # P(Abnormal)
 
        # ── Compute metrics ───────────────────────────────────────────────────
        fold_metrics = compute_metrics(y_test, y_pred, y_prob)
        fold_metrics['fold'] = fold_idx + 1
        fold_metrics['n_train_normal']   = int((y_train == 0).sum())
        fold_metrics['n_train_abnormal'] = int((y_train == 1).sum())
        fold_metrics['n_test_normal']    = int((y_test  == 0).sum())
        fold_metrics['n_test_abnormal']  = int((y_test  == 1).sum())
        per_fold.append(fold_metrics)
 
        # Accumulate for overall ROC
        all_y_true.extend(y_test.tolist())
        all_y_prob.extend(y_prob.tolist())
 
        # Print fold summary
        print(f"\nFold {fold_idx+1}/{n_splits}")
        print(f"  Train: {len(train_idx)} "
              f"(N={fold_metrics['n_train_normal']}, "
              f"Ab={fold_metrics['n_train_abnormal']})")
        print(f"  Test : {len(test_idx)} "
              f"(N={fold_metrics['n_test_normal']}, "
              f"Ab={fold_metrics['n_test_abnormal']})")
        print(f"  Confusion: TP={fold_metrics['tp']}, TN={fold_metrics['tn']}, "
              f"FP={fold_metrics['fp']}, FN={fold_metrics['fn']}")
        print(f"  Sensitivity={fold_metrics['sensitivity']*100:.1f}%  "
              f"Specificity={fold_metrics['specificity']*100:.1f}%  "
              f"Accuracy={fold_metrics['accuracy']*100:.1f}%  "
              f"AUC={fold_metrics['auc']:.3f}")
 
    # ── Aggregate across folds ─────────────────────────────────────────────
    metric_keys = ['sensitivity', 'specificity', 'ppv',
                   'npv', 'accuracy', 'auc']
    mean_metrics = {k: np.mean([f[k] for f in per_fold]) for k in metric_keys}
    std_metrics  = {k: np.std( [f[k] for f in per_fold]) for k in metric_keys}
 
    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)
 
    # ── Final summary table ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("FINAL RESULTS (mean ± std across 5 folds)")
    print("=" * 65)
    print(f"{'Metric':<15} {'Our Result':>12} {'Paper Target':>14}")
    print("-" * 45)
    targets = {
        'sensitivity': '87%',
        'specificity': '89%',
        'ppv'        : '87%',
        'npv'        : '89%',
        'accuracy'   : '88.5%',
        'auc'        : '0.939'
    }
    for k in metric_keys:
        val = mean_metrics[k]
        std = std_metrics[k]
        if k == 'auc':
            print(f"  {k:<13} {val:.3f} ± {std:.3f}    {targets[k]:>8}")
        else:
            print(f"  {k:<13} {val*100:.1f}% ± {std*100:.1f}%   {targets[k]:>8}")
    print("=" * 65)
 
    return {
        'per_fold'   : per_fold,
        'mean'       : mean_metrics,
        'std'        : std_metrics,
        'all_y_true' : all_y_true,
        'all_y_prob' : all_y_prob,
    }