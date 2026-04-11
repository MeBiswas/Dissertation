# src/fann_classifier/step_4.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import auc, roc_curve

# ═════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════
 
def visualize_results(results: dict, save_path: str = "classification_results.png"):
    fig = plt.figure(figsize=(22, 5))
    gs  = gridspec.GridSpec(1, 4, figure=fig)
    fig.suptitle("Stage 3 — Step 3: FANN Classification Results  "
                 "(Section III-A, Table IV)",
                 fontsize=13, fontweight='bold')
 
    # ── Panel 1: ROC Curve ────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    fpr, tpr, _ = roc_curve(results['all_y_true'], results['all_y_prob'])
    roc_auc     = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.5)')
    ax1.axhline(y=0.939, color='green', lw=1, linestyle=':',
                label='Paper AUC = 0.939')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.02])
    ax1.set_xlabel('False Positive Rate (1 - Specificity)')
    ax1.set_ylabel('True Positive Rate (Sensitivity)')
    ax1.set_title('(1) ROC Curve\n(all folds combined)')
    ax1.legend(loc='lower right', fontsize=8)
    ax1.grid(True, alpha=0.3)
 
    # ── Panel 2: Per-fold metrics ─────────────────────────────────────────────
    ax2   = fig.add_subplot(gs[1])
    folds = [f['fold'] for f in results['per_fold']]
    keys  = ['sensitivity', 'specificity', 'ppv', 'npv', 'accuracy']
    cols  = ['steelblue', 'orangered', 'seagreen', 'purple', 'gold']
    x     = np.arange(len(folds))
    bar_w = 0.15
    for i, (k, c) in enumerate(zip(keys, cols)):
        vals = [f[k]*100 for f in results['per_fold']]
        ax2.bar(x + i*bar_w, vals, bar_w, label=k.capitalize(),
                color=c, alpha=0.8)
    ax2.set_xticks(x + bar_w*2)
    ax2.set_xticklabels([f"Fold {i}" for i in folds])
    ax2.set_ylim([0, 110])
    ax2.set_ylabel("Score (%)")
    ax2.set_title("(2) Metrics per Fold")
    ax2.legend(fontsize=7, loc='lower right')
    ax2.grid(True, alpha=0.3, axis='y')
 
    # ── Panel 3: Aggregated confusion matrix ──────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    total_tp = sum(f['tp'] for f in results['per_fold'])
    total_tn = sum(f['tn'] for f in results['per_fold'])
    total_fp = sum(f['fp'] for f in results['per_fold'])
    total_fn = sum(f['fn'] for f in results['per_fold'])
    cm_array = np.array([[total_tp, total_fn],
                          [total_fp, total_tn]])
    im = ax3.imshow(cm_array, cmap='Blues')
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(['Predicted\nAbnormal', 'Predicted\nNormal'])
    ax3.set_yticklabels(['Actual\nAbnormal', 'Actual\nNormal'])
    ax3.set_title("(3) Aggregated\nConfusion Matrix")
    labels_cm = [[f'TP={total_tp}', f'FN={total_fn}'],
                 [f'FP={total_fp}', f'TN={total_tn}']]
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, labels_cm[i][j], ha='center',
                     va='center', fontsize=11, fontweight='bold',
                     color='white' if cm_array[i, j] > cm_array.max()/2
                     else 'black')
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
 
    # ── Panel 4: Our results vs Paper targets ─────────────────────────────────
    ax4 = fig.add_subplot(gs[3])
    metric_labels = ['Sensitivity', 'Specificity', 'PPV',
                     'NPV', 'Accuracy', 'AUC×100']
    paper_vals = [87, 89, 87, 89, 88.5, 93.9]
    our_vals   = [
        results['mean']['sensitivity'] * 100,
        results['mean']['specificity'] * 100,
        results['mean']['ppv']         * 100,
        results['mean']['npv']         * 100,
        results['mean']['accuracy']    * 100,
        results['mean']['auc']         * 100,
    ]
    x4    = np.arange(len(metric_labels))
    bar_w = 0.35
    ax4.bar(x4 - bar_w/2, paper_vals, bar_w, label='Paper (Table IV)',
            color='steelblue', alpha=0.8)
    ax4.bar(x4 + bar_w/2, our_vals,   bar_w, label='Our Result',
            color='orangered', alpha=0.8)
    ax4.set_xticks(x4)
    ax4.set_xticklabels(metric_labels, rotation=30, ha='right', fontsize=8)
    ax4.set_ylim([0, 110])
    ax4.set_ylabel("Score (%)")
    ax4.set_title("(4) Our Results vs\nPaper Targets (Table IV)")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[Save]  Visualization saved → {save_path}")
    plt.show()