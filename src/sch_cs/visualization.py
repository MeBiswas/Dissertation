# sch_cs/visualization.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────────────────────
# VISUAL OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
 
"""
    4-panel figure matching paper's Fig. 3:
    (a) p_b   (b) binary   (c) all regions + centroids   (d) final SRs
"""
def visualize_results(pb, binary_image, all_regions, sr_regions, th, save_path):
    sr_labels = {r["label"] for r in sr_regions}
 
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    fig.suptitle("SCH-CS Steps 3 & 4 — Suspicious Region Identification\n"
                 f"th = {th:.2f}", fontsize=13, fontweight='bold')
 
    # (a) background-removed grayscale
    axes[0].imshow(pb, cmap='gray')
    axes[0].set_title("(a) p_b\nBackground-removed")
    axes[0].axis('off')
 
    # (b) binary thresholded
    axes[1].imshow(binary_image, cmap='gray')
    axes[1].set_title(f"(b) Binary  th={th:.1f}\nWhite = SR candidates")
    axes[1].axis('off')
 
    # (c) all regions with colour overlay and centroid markers
    overlay = cv2.cvtColor((pb * 0.6).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    for reg in all_regions:
        color = (0, 180, 0) if reg["label"] in sr_labels else (180, 0, 0)
        overlay[reg["mask"]] = (
            overlay[reg["mask"]] * 0.5 + np.array(color) * 0.5
        ).astype(np.uint8)
 
    axes[2].imshow(overlay)
    for reg in all_regions:
        X, Y  = reg["centroid"]
        c     = "lime"  if reg["label"] in sr_labels else "red"
        mk    = "o"     if reg["label"] in sr_labels else "x"
        axes[2].plot(Y, X, mk, color=c, markersize=9, markeredgewidth=2)
        axes[2].annotate(
            f"R({reg['label']})",
            xy=(Y, X), xytext=(Y + 2, X - 3),
            fontsize=7, color=c,
            bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.5))
 
    axes[2].legend(handles=[
        mpatches.Patch(color='lime', label='SR kept'),
        mpatches.Patch(color='red',  label='Eliminated')],
        fontsize=8, loc='lower right')
    axes[2].set_title("(c) All regions\nGreen=SR, Red=eliminated")
    axes[2].axis('off')
 
    # (d) final SRs only — equivalent to paper's Fig. 3(d)
    final = np.zeros_like(pb, dtype=np.uint8)
    for reg in sr_regions:
        final[reg["mask"]] = pb[reg["mask"]]
 
    axes[3].imshow(final, cmap='gray')
    for reg in sr_regions:
        X, Y   = reg["centroid"]
        coords = reg["coords"]
        i_min, j_min = coords[:, 0].min(), coords[:, 1].min()
        i_max, j_max = coords[:, 0].max(), coords[:, 1].max()
        axes[3].plot(Y, X, 'o', color='lime', markersize=9,
                     markeredgewidth=2)
        rect = mpatches.Rectangle(
            (j_min, i_min), j_max - j_min, i_max - i_min,
            linewidth=1.5, edgecolor='lime', facecolor='none')
        axes[3].add_patch(rect)
        axes[3].annotate(
            f"SR({reg['label']})",
            xy=(Y, X), xytext=(Y + 2, X - 4),
            fontsize=7, color='lime',
            bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.6))
 
    axes[3].set_title(f"(d) Final SRs\n{len(sr_regions)} SR(s) found")
    axes[3].axis('off')
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n  [Plot saved to {save_path}]")