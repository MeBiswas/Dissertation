# src/features_extraction/visualization.py

import numpy as np
import matplotlib.pyplot as plt

# ═════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════
 
def visualize_feature_extraction(original_gray: np.ndarray,
                                  segmented_sr: np.ndarray,
                                  g: np.ndarray,
                                  haralick_feats: np.ndarray,
                                  hu_feats: np.ndarray,
                                  save_path: str = "feature_extraction.png"):
    """
    4-panel visualization:
        1. Original grayscale with SR contour overlaid
        2. Extracted SR region (masked)
        3. GLCM heatmap
        4. Bar chart of all 21 features
    """
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle("Stage 3 — Step 1: Feature Extraction (Section III-A)",
                 fontsize=13, fontweight='bold')
 
    # Panel 1: Grayscale with SR boundary
    axes[0].imshow(original_gray, cmap='gray')
    axes[0].contour(segmented_sr, levels=[0.5], colors=['red'], linewidths=[2])
    axes[0].set_title("(1) SR Boundary on TBI")
    axes[0].axis('off')
 
    # Panel 2: Extracted SR region
    sr_region = original_gray * segmented_sr
    axes[1].imshow(sr_region, cmap='hot')
    axes[1].set_title("(2) SR Region\n(mask applied to p_b)")
    axes[1].axis('off')
 
    # Panel 3: GLCM heatmap (log scale for visibility)
    g_log = np.log(g + 1e-10)
    glcm_img = axes[2].imshow(g_log, cmap='viridis')
    axes[2].set_title("(3) Normalized GLCM g\n(log scale, Eq 22)")
    axes[2].set_xlabel("Gray level v")
    axes[2].set_ylabel("Gray level u")
    plt.colorbar(glcm_img, ax=axes[2], fraction=0.046, pad=0.04)
 
    # Panel 4: All 21 feature values as bar chart
    all_features = np.concatenate([haralick_feats, hu_feats])
    feature_names = (
        [f"H{i+1}" for i in range(14)] +       # H1..H14: Haralick
        [f"Hu{i+1}" for i in range(7)]          # Hu1..Hu7: Hu's moments
    )
    colors = ['steelblue'] * 14 + ['orangered'] * 7
    axes[3].bar(feature_names, all_features, color=colors)
    axes[3].set_title("(4) 21-element Feature Vector\n"
                      "(Blue=Haralick | Red=Hu)")
    axes[3].set_xlabel("Feature")
    axes[3].set_ylabel("Value")
    axes[3].tick_params(axis='x', rotation=90)
    axes[3].grid(True, alpha=0.3, axis='y')
 
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='steelblue', label='14 Haralick'),
                       Patch(facecolor='orangered', label='7 Hu Moments')]
    axes[3].legend(handles=legend_elements, loc='upper right')
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[Save]  Visualization saved → {save_path}")
    plt.show()