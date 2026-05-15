# src/fann_classifier/step_3.py

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from .step_2 import build_fann
from .step_1 import compute_metrics

# ═════════════════════════════════════════════════════════════════════════════
# PART 3 — 5-FOLD CROSS-VALIDATION
# ═════════════════════════════════════════════════════════════════════════════
def run_fann_5fold(
    X            : np.ndarray,
    y            : np.ndarray,
    n_folds      : int   = 5,
    epochs       : int   = 1000,
    batch_size   : int   = 32,
    learning_rate: float = 0.1,
    patience     : int   = 100,
    random_state : int   = 42,
    results_dir  : str   = 'fann_results',
    verbose      : bool  = True,
) -> dict:
    os.makedirs(results_dir, exist_ok=True)

    skf = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=random_state
    )

    fold_metrics  = []
    fold_scalers  = []
    fold_models   = []
    fold_histories= []
    all_y_true    = []
    all_y_pred    = []
    all_y_prob    = []

    if verbose:
        n_normal   = int((y == 0).sum())
        n_abnormal = int((y == 1).sum())
        print(f"\n{'='*60}")
        print(f"  FANN  ·  5-Fold Stratified Cross-Validation")
        print(f"  Architecture : 21 -> 42 (tanh) -> 1 (sigmoid)")
        print(f"  Optimiser    : Adam  lr={learning_rate}")
        print(f"  Max epochs   : {epochs}   patience={patience}")
        print(f"  Total samples: {len(y)}"
              f"  (normal={n_normal}, abnormal={n_abnormal})")
        print(f"{'='*60}")

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # -- Feature scaling (train only) ------------------------------------
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)
        fold_scalers.append(scaler)

        # -- Build model -----------------------------------------------------
        tf.random.set_seed(random_state + fold)
        np.random.seed(random_state + fold)
        model = build_fann(learning_rate=learning_rate)

        # -- Train -----------------------------------------------------------
        # Split 10 samples manually for early stopping -- no validation_split
        val_size  = 10
        idx_perm  = np.random.permutation(len(X_train))
        X_val_es  = X_train[idx_perm[:val_size]]
        y_val_es  = y_train[idx_perm[:val_size]]
        X_tr_es   = X_train[idx_perm[val_size:]]
        y_tr_es   = y_train[idx_perm[val_size:]]

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=150,
                restore_best_weights=True, verbose=0,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5,
                patience=50, min_lr=1e-6, verbose=0,
            ),
        ]

        hist = model.fit(
            X_tr_es, y_tr_es,
            epochs=2000,
            batch_size=len(X_tr_es),   # full-batch -- all ~95 samples per update
            validation_data=(X_val_es, y_val_es),
            callbacks=callbacks,
            verbose=0,
        )
        fold_histories.append(hist.history)

        # -- Evaluate --------------------------------------------------------
        y_prob = model.predict(X_test, verbose=0).ravel()
        y_pred = (y_prob >= 0.5).astype(int)

        m = compute_metrics(y_test, y_pred, y_prob)
        fold_metrics.append(m)
        fold_models.append(model)

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_y_prob.extend(y_prob.tolist())

        if verbose:
            used = len(hist.history['loss'])
            print(f"\n  Fold {fold}/{n_folds}  "
                  f"(train={len(train_idx)}, test={len(test_idx)}, "
                  f"epochs={used})")
            print(f"    Acc={m['accuracy']:.1f}%  "
                  f"Sens={m['sensitivity']:.1f}%  "
                  f"Spec={m['specificity']:.1f}%  "
                  f"PPV={m['ppv']:.1f}%  "
                  f"NPV={m['npv']:.1f}%  "
                  f"AUC={m['auc']:.3f}")

    # -- Aggregate metrics ---------------------------------------------------
    KEYS = ['accuracy', 'sensitivity', 'specificity', 'ppv', 'npv', 'auc']
    avg = {k: float(np.mean([f[k] for f in fold_metrics])) for k in KEYS}
    std = {k: float(np.std ([f[k] for f in fold_metrics])) for k in KEYS}

    # Pooled = all fold predictions combined into one global evaluation
    pooled = compute_metrics(
        np.array(all_y_true),
        np.array(all_y_pred),
        np.array(all_y_prob),
    )

    # Paper reference values (Table IV)
    PAPER = {
        'accuracy': 88.5, 'sensitivity': 87.0,
        'specificity': 89.0, 'ppv': 87.0, 'npv': 89.0, 'auc': 0.939,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"  {'Metric':<14} {'Avg':>8}  {'+-Std':>6}  {'Paper':>8}")
        print(f"  {'-'*44}")
        for k in KEYS:
            unit  = '' if k == 'auc' else '%'
            paper = PAPER[k]
            print(f"  {k:<14} {avg[k]:7.2f}{unit}  "
                  f"+-{std[k]:5.2f}  {paper:7.3f}{unit}")
        print(f"{'='*60}\n")

    results = {
        'fold_metrics'  : fold_metrics,
        'avg'           : avg,
        'std'           : std,
        'pooled'        : pooled,
        'all_y_true'    : all_y_true,
        'all_y_pred'    : all_y_pred,
        'all_y_prob'    : all_y_prob,
        'fold_histories': fold_histories,
        'fold_models'   : fold_models,
        'fold_scalers'  : fold_scalers,
        'n_folds'       : n_folds,
        'n_samples'     : len(y),
    }

    # -- Save JSON results (models excluded -- not JSON-serializable) ----------
    serializable = {
        'avg': avg, 'std': std, 'pooled': pooled,
        'fold_metrics': fold_metrics,
        'paper_reference': PAPER,
        'n_folds': n_folds,
        'n_samples': int(len(y)),
        'class_dist': {
            'normal'  : int((y == 0).sum()),
            'abnormal': int((y == 1).sum()),
        },
    }
    json_path = os.path.join(results_dir, 'fann_cv_results.json')
    with open(json_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    if verbose:
        print(f"[Saved] {json_path}")

    return results