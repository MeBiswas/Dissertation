# src/asymmetry_vector/step_3.py

import os
import numpy as np
import matplotlib.pyplot as plt

# ═════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════
def visualize_asymmetry(
    f_v_left: np.ndarray,
    f_v_right: np.ndarray,
    F: np.ndarray,
    run_dir: str,
    label: str = "Unknown",
    save: bool = True,
    show: bool = True
):
    feature_names = (
        [f"H{i+1}" for i in range(14)] +
        [f"Hu{i+1}" for i in range(7)]
    )

    x = np.arange(21)
    colors = ['steelblue'] * 14 + ['orangered'] * 7

    fig, axes = plt.subplots(1, 3, figsize=(22, 5))
    fig.suptitle(f"Asymmetry Feature Vector | {label}", fontsize=13, fontweight='bold')

    # Panel 1
    axes[0].bar(x - 0.2, f_v_left, 0.4, label='Left', color='green')
    axes[0].bar(x + 0.2, f_v_right, 0.4, label='Right', color='red')
    axes[0].set_title("Left vs Right")
    axes[0].legend()

    # Panel 2
    axes[1].bar(x, F, color=colors)
    axes[1].set_title("Asymmetry F")

    # Panel 3
    sorted_idx = np.argsort(F)[::-1]
    axes[2].barh(range(21), F[sorted_idx][::-1], color=[colors[i] for i in sorted_idx][::-1])
    axes[2].set_title("Ranked Features")

    plt.tight_layout()

    if save:
        os.makedirs(run_dir, exist_ok=True)
        save_path = os.path.join(run_dir, "visualization.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Save] Visualization → {save_path}")

    if show:
        plt.show()

    plt.close(fig)