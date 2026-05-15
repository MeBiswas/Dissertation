# src/chm_bias_correction/step_5.py

import numpy as np
import matplotlib.pyplot as plt

def visualize_chm(pb, chm_local, p_bar, eps=1e-10, save_path=None, show=True):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(pb, cmap='hot')
    axes[0].set_title("Input")

    axes[1].imshow(chm_local, cmap='hot')
    axes[1].set_title("CHM Local")

    disp = p_bar / (p_bar.max() + eps) * 255
    axes[2].imshow(disp, cmap='hot')
    axes[2].set_title("Corrected")

    mid = pb.shape[0] // 2

    def norm01(x):
        return (x - x.min()) / (x.max() - x.min() + eps)

    axes[3].plot(norm01(pb[mid].astype(np.float64)), label="original", color='steelblue', lw=1.5)
    axes[3].plot(norm01(p_bar[mid]), label="corrected", color='orangered', lw=1.5)
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    axes[3].set_title('Intensity profile (mid row)')

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()