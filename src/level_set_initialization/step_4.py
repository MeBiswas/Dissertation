# src/level_set_initialization/step_4.py

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ----------------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------------
def visualize_phi_init(
    preprocessed_img: np.ndarray,
    sr_mask: np.ndarray,
    phi: np.ndarray,
    run_dir: str,
    show: bool = True
) -> str:
    viz_path = os.path.join(run_dir, "phi_initialization.png")
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Equation 17: Level Set Function φ Initialization\n'
                 '(Contour starts from validated SCH-CS SR regions only)',
                 fontsize=12, fontweight='bold')
    
    # Panel 1: Preprocessed image
    axes[0].imshow(preprocessed_img, cmap='gray')
    axes[0].set_title('(1) Preprocessed p_b', fontsize=10)
    axes[0].axis('off')
    
    # Panel 2: SR mask (from validated regions)
    axes[1].imshow(sr_mask, cmap='gray')
    axes[1].set_title(f'(2) SR Mask (p_th_b)\n{int(sr_mask.sum())} validated SR pixels',
                      fontsize=10)
    axes[1].axis('off')
    
    # Panel 3: φ values
    phi_img = axes[2].imshow(
        phi,
        cmap='RdBu_r',
        norm=mcolors.TwoSlopeNorm(vmin=-4, vcenter=0, vmax=4)
    )
    axes[2].set_title('(3) Initialized φ (Eq 17)\nRed=+4 (SR) | Blue=-4 (outside)',
                      fontsize=10)
    axes[2].axis('off')
    plt.colorbar(phi_img, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Panel 4: φ=0 contour overlay
    axes[3].imshow(preprocessed_img, cmap='gray')
    axes[3].contour(phi, levels=[0], colors=['red'], linewidths=[2])
    axes[3].set_title('(4) φ=0 Initial Contour on p_b\n(will evolve to SR boundary)',
                      fontsize=10)
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    print(f"\n[Viz] Saved: {viz_path}")
    return viz_path