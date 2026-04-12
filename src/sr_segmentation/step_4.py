# src/sr_segmentation/step_4.py

import os
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────
def visualize(preprocessed_img, segmented_sr, sr_left, sr_right, centre_col, run_dir, show=True):
    
    save_path = os.path.join(run_dir, "visualization.png")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(preprocessed_img, cmap='gray')
    axes[0].contour(segmented_sr, levels=[0.5], colors=['red'])
    axes[0].axvline(x=centre_col, color='yellow', linestyle='--')

    axes[1].imshow(preprocessed_img, cmap='gray')
    axes[1].contour(sr_left, levels=[0.5], colors=['cyan'])

    axes[2].imshow(preprocessed_img, cmap='gray')
    axes[2].contour(sr_right, levels=[0.5], colors=['lime'])

    overlay = np.stack([preprocessed_img]*3, axis=-1).astype(np.float32)
    overlay /= (overlay.max() + 1e-10)

    overlay[sr_left > 0]  = [0, 1, 1]
    overlay[sr_right > 0] = [0, 1, 0]

    axes[3].imshow(overlay)
    axes[3].axvline(x=centre_col, color='yellow', linestyle='--')

    plt.tight_layout()
    plt.savefig(save_path)

    if show:
        plt.show()
    else:
        plt.close()