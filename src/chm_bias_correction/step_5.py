# src/chm_bias_correction/step_5.py

import matplotlib.pyplot as plt

def visualize(pb, chm_local, p_bar, eps=1e-10, save_path=None, show=True):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(pb, cmap='hot')
    axes[0].set_title("Input")

    axes[1].imshow(chm_local, cmap='hot')
    axes[1].set_title("CHM Local")

    disp = p_bar / (p_bar.max() + eps) * 255
    axes[2].imshow(disp, cmap='hot')
    axes[2].set_title("Corrected")

    mid = pb.shape[0] // 2
    axes[3].plot(pb[mid], label="original")
    axes[3].plot(p_bar[mid], label="corrected")
    axes[3].legend()

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()