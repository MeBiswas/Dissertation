# src/level_set_initialization/step_4.py

import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def visualize(preprocessed_img, schcs_binary, phi, run_dir, show=True):
    
    viz_path = os.path.join(run_dir, "phi_visualization.png")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(preprocessed_img, cmap='gray')
    axes[1].imshow(schcs_binary, cmap='gray')

    phi_img = axes[2].imshow(
        phi,
        cmap='RdBu_r',
        norm=mcolors.TwoSlopeNorm(vmin=-4, vcenter=0, vmax=4)
    )

    axes[3].imshow(preprocessed_img, cmap='gray')
    axes[3].contour(phi, levels=[0], colors=['red'], linewidths=[2])

    plt.colorbar(phi_img, ax=axes[2])

    plt.tight_layout()
    plt.savefig(viz_path)

    if show:
        plt.show()
    else:
        plt.close()

    return viz_path