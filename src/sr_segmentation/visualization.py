# src/src/sr_segmentation/visualization.py

import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────
 
def visualize_split(
    preprocessed_img: np.ndarray,
    segmented_sr: np.ndarray,
    sr_img_left: np.ndarray,
    sr_img_right: np.ndarray,
    centre_col: int,
    save_path: str = "sr_split_result.png"
):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("SR Split: Left and Right Breast Regions",
                 fontsize=13, fontweight='bold')
 
    # Panel 1: Full SR on image
    axes[0].imshow(preprocessed_img, cmap='gray')
    axes[0].contour(segmented_sr, levels=[0.5],
                    colors=['red'], linewidths=[2])
    axes[0].axvline(x=centre_col, color='yellow',
                    linewidth=2, linestyle='--', label='Split line')
    axes[0].set_title("(1) Full Segmented SR\n(both breasts)")
    axes[0].legend(fontsize=8)
    axes[0].axis('off')
 
    # Panel 2: Image left half (patient RIGHT breast)
    axes[1].imshow(preprocessed_img, cmap='gray')
    axes[1].contour(sr_img_left, levels=[0.5],
                    colors=['cyan'], linewidths=[2])
    axes[1].axvline(x=centre_col, color='yellow',
                    linewidth=1, linestyle='--')
    axes[1].set_title("(2) Image LEFT half\n(Patient's RIGHT breast)")
    axes[1].axis('off')
 
    # Panel 3: Image right half (patient LEFT breast)
    axes[2].imshow(preprocessed_img, cmap='gray')
    axes[2].contour(sr_img_right, levels=[0.5],
                    colors=['lime'], linewidths=[2])
    axes[2].axvline(x=centre_col, color='yellow',
                    linewidth=1, linestyle='--')
    axes[2].set_title("(3) Image RIGHT half\n(Patient's LEFT breast)")
    axes[2].axis('off')
 
    # Panel 4: Both halves overlaid with colour coding
    overlay = np.stack([preprocessed_img]*3, axis=-1).astype(np.float32)
    overlay /= (overlay.max() + 1e-10)         # normalize to [0,1]
 
    # Tint left SR cyan, right SR green
    cyan_mask  = sr_img_left  > 0
    green_mask = sr_img_right > 0
    overlay[cyan_mask,  0] = 0.0
    overlay[cyan_mask,  2] = 1.0               # blue channel → cyan
    overlay[green_mask, 1] = 1.0               # green channel
    overlay[green_mask, 0] = 0.0
    overlay[green_mask, 2] = 0.0
 
    axes[3].imshow(overlay)
    axes[3].axvline(x=centre_col, color='yellow',
                    linewidth=2, linestyle='--')
    axes[3].set_title("(4) Colour Overlay\nCyan=Img Left | Green=Img Right")
    axes[3].axis('off')
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[Save]  Visualization saved → {save_path}")
    plt.show()