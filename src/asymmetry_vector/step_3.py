# src/asymmetry_vector/step_3.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ═════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════
 
def visualize_asymmetry(
    f_v_left: np.ndarray,
    f_v_right: np.ndarray,
    F: np.ndarray,
    label: str = "Unknown",
    save_path: str = "asymmetry_result.png"
):
    feature_names = (
        [f"H{i+1}"  for i in range(14)] +
        [f"Hu{i+1}" for i in range(7)]
    )
    x = np.arange(21)
    colors = ['steelblue'] * 14 + ['orangered'] * 7
 
    fig, axes = plt.subplots(1, 3, figsize=(22, 5))
    fig.suptitle(
        f"Stage 3 — Step 2: Asymmetry Feature Vector  |  Patient: {label}",
        fontsize=13, fontweight='bold'
    )
 
    # ── Panel 1: Left vs Right features side by side ─────────────────────────
    bar_w = 0.35
    axes[0].bar(x - bar_w/2, f_v_left,  bar_w, label='Left breast',
                color='mediumseagreen', alpha=0.8)
    axes[0].bar(x + bar_w/2, f_v_right, bar_w, label='Right breast',
                color='salmon', alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(feature_names, rotation=90, fontsize=8)
    axes[0].set_title("(1) Left vs Right Feature Vectors\n"
                      "f_v^(L)  and  f_v^(R)")
    axes[0].set_ylabel("Feature value")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
 
    # ── Panel 2: Asymmetry vector F ───────────────────────────────────────────
    bars = axes[1].bar(x, F, color=colors, alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(feature_names, rotation=90, fontsize=8)
    axes[1].set_title("(2) Asymmetry Vector F\n"
                      "F = |f_v^(L) - f_v^(R)|")
    axes[1].set_ylabel("|Left - Right|")
    axes[1].grid(True, alpha=0.3, axis='y')
 
    blue_patch  = mpatches.Patch(color='steelblue',  label='Haralick (H1-H14)')
    red_patch   = mpatches.Patch(color='orangered',  label="Hu's moments (Hu1-Hu7)")
    axes[1].legend(handles=[blue_patch, red_patch], loc='upper right')
 
    # ── Panel 3: Which features differ most? (sorted bar) ────────────────────
    sorted_idx   = np.argsort(F)[::-1]     # descending order
    sorted_F     = F[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_colors = [colors[i] for i in sorted_idx]
 
    axes[2].barh(range(21), sorted_F[::-1], color=sorted_colors[::-1],
                 alpha=0.85)
    axes[2].set_yticks(range(21))
    axes[2].set_yticklabels(sorted_names[::-1], fontsize=8)
    axes[2].set_title("(3) Features Ranked by Asymmetry\n"
                      "(top = most asymmetric)")
    axes[2].set_xlabel("|Left - Right|")
    axes[2].grid(True, alpha=0.3, axis='x')
    axes[2].legend(handles=[blue_patch, red_patch], loc='lower right')
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[Save]  Visualization saved → {save_path}")
    plt.show()
 
 
def visualize_dataset_asymmetry(F_dataset: np.ndarray, labels: np.ndarray, save_path: str = "dataset_asymmetry.png"):
    feature_names = (
        [f"H{i+1}"  for i in range(14)] +
        [f"Hu{i+1}" for i in range(7)]
    )
    x = np.arange(21)
 
    normal_mean   = F_dataset[labels == 0].mean(axis=0)
    abnormal_mean = F_dataset[labels == 1].mean(axis=0)
    normal_std    = F_dataset[labels == 0].std(axis=0)
    abnormal_std  = F_dataset[labels == 1].std(axis=0)
 
    fig, ax = plt.subplots(figsize=(16, 5))
    bar_w = 0.35
    ax.bar(x - bar_w/2, normal_mean,   bar_w, yerr=normal_std,
           label='Normal',   color='steelblue', alpha=0.8, capsize=3)
    ax.bar(x + bar_w/2, abnormal_mean, bar_w, yerr=abnormal_std,
           label='Abnormal', color='salmon',    alpha=0.8, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=90)
    ax.set_title("Mean Asymmetry (F) per Feature — Normal vs Abnormal\n"
                 "(Error bars = std dev)")
    ax.set_ylabel("Mean |Left - Right|")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[Save]  Dataset asymmetry plot saved → {save_path}")
    plt.show() 