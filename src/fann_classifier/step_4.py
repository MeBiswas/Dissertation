# src/fann_classifier/step_4.py

import os
import numpy as np
from typing import List
from tensorflow import keras
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    accuracy_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler

# ═════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════
def visualize_fann_results(
    results    : dict,
    results_dir: str = 'fann_results',
    show       : bool = True,
) -> None:
    os.makedirs(results_dir, exist_ok=True)

    PAPER = {
        'accuracy': 88.5, 'sensitivity': 87.0,
        'specificity': 89.0, 'ppv': 87.0, 'npv': 89.0, 'auc': 93.9,
    }

    fold_metrics = results['fold_metrics']
    avg          = results['avg']
    n_folds      = results['n_folds']
    all_y_true   = np.array(results['all_y_true'])
    all_y_prob   = np.array(results['all_y_prob'])
    all_y_pred   = np.array(results['all_y_pred'])

    KEYS   = ['accuracy', 'sensitivity', 'specificity', 'ppv', 'npv']
    LABELS = ['Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV']
    COLORS = ['#378ADD', '#1D9E75', '#D85A30', '#7F77DD', '#BA7517']

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        'STBIA — FANN 5-Fold Cross-Validation Results',
        fontsize=14, fontweight='bold', y=0.98
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

    # ── (a) Per-fold metrics ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    x   = np.arange(n_folds)
    w   = 0.15
    offsets = np.linspace(-(len(KEYS)-1)/2*w, (len(KEYS)-1)/2*w, len(KEYS))

    for ki, (key, lbl, col, off) in enumerate(
        zip(KEYS, LABELS, COLORS, offsets)
    ):
        vals = [f[key] for f in fold_metrics]
        ax1.bar(x + off, vals, width=w, label=lbl, color=col, alpha=0.85)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Fold {i+1}' for i in range(n_folds)], fontsize=10)
    ax1.set_ylim(0, 110)
    ax1.set_ylabel('Score (%)', fontsize=10)
    ax1.set_title('(a) Per-fold metrics', fontsize=11)
    ax1.legend(fontsize=8, loc='lower right')
    ax1.axhline(88.5, color='gray', lw=1, ls='--', alpha=0.5,
                label='paper Acc')
    ax1.grid(axis='y', alpha=0.25)

    # ── (b) ROC curve ─────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    try:
        fpr, tpr, _ = roc_curve(all_y_true, all_y_prob)
        auc_val     = roc_auc_score(all_y_true, all_y_prob)
        ax2.plot(fpr, tpr, color='#378ADD', lw=2,
                 label=f'STBIA  AUC={auc_val:.3f}')
        ax2.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4, label='Random')
        # Paper reference point (sens=0.87, spec=0.89 → fpr=0.11, tpr=0.87)
        ax2.scatter([0.11], [0.87], color='#D85A30', zorder=5, s=80,
                    label='Paper (0.939)')
    except Exception:
        ax2.text(0.5, 0.5, 'Insufficient data\nfor ROC',
                 ha='center', va='center', transform=ax2.transAxes)

    ax2.set_xlabel('False Positive Rate', fontsize=10)
    ax2.set_ylabel('True Positive Rate', fontsize=10)
    ax2.set_title('(b) ROC curve (pooled folds)', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.2)

    # ── (c) Confusion matrix (pooled) ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    cm  = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1])
    im  = ax3.imshow(cm, cmap='Blues', vmin=0)

    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = 'white' if val > cm.max() * 0.6 else 'black'
            ax3.text(j, i, str(val), ha='center', va='center',
                     fontsize=16, color=color, fontweight='bold')

    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(['Normal\n(pred)', 'Abnormal\n(pred)'], fontsize=10)
    ax3.set_yticklabels(['Normal\n(true)', 'Abnormal\n(true)'], fontsize=10)
    ax3.set_title('(c) Confusion matrix (pooled folds)', fontsize=11)
    plt.colorbar(im, ax=ax3, fraction=0.046)

    # ── (d) Comparison table ──────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    col_labels = ['Metric', 'Yours', 'Paper', 'Δ']
    rows       = []
    for k, lbl in zip(KEYS + ['auc'], LABELS + ['AUC']):
        unit  = '' if k == 'auc' else '%'
        yours = avg[k]
        paper = PAPER[k] if k != 'auc' else 93.9
        delta = yours - paper
        sign  = '+' if delta >= 0 else ''
        rows.append([
            lbl,
            f"{yours:.1f}{unit}",
            f"{paper:.1f}{unit}",
            f"{sign}{delta:.1f}{unit}",
        ])

    tbl = ax4.table(
        cellText=col_labels[0:0] + rows,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)

    # Colour header
    for j in range(4):
        tbl[0, j].set_facecolor('#378ADD')
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    # Colour delta column: green if positive, red if negative
    for i, row in enumerate(rows, 1):
        delta_str = row[3]
        fc = '#EAF3DE' if delta_str.startswith('+') else '#FCEBEB'
        tbl[i, 3].set_facecolor(fc)

    ax4.set_title('(d) Your results vs paper (Table IV)', fontsize=11, pad=14)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = os.path.join(results_dir, 'fann_results.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[Saved] {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
        
# ═════════════════════════════════════════════════════════════════════════════
# SINGLE IMAGE CLASSIFICATION VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════
def visualize_single_image_classification(
    F_vector: np.ndarray,
    trained_model: keras.Model,
    scaler: StandardScaler,
    image_name: str,
    class_labels: List[str] = ['Normal', 'Abnormal']
) -> None:
    # Reshape F_vector for single prediction (1, 21)
    F_reshaped = F_vector.reshape(1, -1)

    # Scale the input feature vector
    F_scaled = scaler.transform(F_reshaped)

    # Predict probability
    probability = trained_model.predict(F_scaled, verbose=0).item()

    # Determine class label
    predicted_class_idx = int(probability >= 0.5)
    predicted_label = class_labels[predicted_class_idx]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(class_labels, [1-probability, probability], color=['lightblue', 'lightcoral'])
    ax.set_xlim(0, 1) # Probability range
    ax.set_title(f'Classification Output for {image_name}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Probability', fontsize=10)
    ax.set_ylabel('Class', fontsize=10)
    ax.tick_params(axis='y', labelsize=10)

    # Annotate the prediction
    ax.text(0.5, 1.05, f'Predicted: {predicted_label} (Prob: {probability:.2f})',
            horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes,
            fontsize=12, fontweight='bold', color='darkgreen' if predicted_label == 'Normal' else 'darkred')

    plt.tight_layout()
    plt.show()

    print(f"\n[Classification] Image: {image_name}")
    print(f"  Predicted Class: {predicted_label}")
    print(f"  Probability: {probability:.4f}")