# src/level_set_iteration/step_7.py

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional

from .storage_config import LevelSetConfig

# =========================================================================
# STEP 7: Visualization
# =========================================================================
def visualize_results(
    preprocessed_img: np.ndarray,
    phi_init: np.ndarray,
    phi_final: np.ndarray,
    segmented_sr: np.ndarray,
    history: Dict,
    config: LevelSetConfig,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """Visualize level set results with proper labels."""
    
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle('DLPE Level Set — Results (Equations 18–21)',
                 fontsize=13, fontweight='bold')
    
    # Panel 1: Initial contour
    axes[0].imshow(preprocessed_img, cmap='gray')
    axes[0].contour(phi_init, levels=[0], colors=['cyan'], linewidths=[2])
    axes[0].set_title('(1) Initial φ=0 contour\n(from SCH-CS Eq 17)')
    axes[0].axis('off')
    
    # Panel 2: Final contour
    axes[1].imshow(preprocessed_img, cmap='gray')
    axes[1].contour(phi_final, levels=[0], colors=['red'], linewidths=[2])
    axes[1].set_title('(2) Final φ=0 contour\n(after DLPE Eq 18)')
    axes[1].axis('off')
    
    # Panel 3: Segmented SR mask
    axes[2].imshow(segmented_sr, cmap='gray')
    axes[2].set_title(f'(3) Segmented SR\n({int(segmented_sr.sum())}px, φ>0)')
    axes[2].axis('off')
    
    # Panel 4: l1, l2 convergence
    if history and history['iteration']:
        iters = history['iteration']
        axes[3].plot(iters, history['l1'], color='red', lw=1.5, label='l1 (inside SR)')
        axes[3].plot(iters, history['l2'], color='blue', lw=1.5, label='l2 (outside SR)')
        axes[3].set_title('(4) l1, l2 convergence\n(Eq 19, 20)')
        axes[3].set_xlabel('Iteration')
        axes[3].legend(fontsize=8)
        axes[3].grid(True, alpha=0.3)
        
        # Panel 5: Stopping criterion
        axes[4].plot(iters, history['dev_prev'], color='purple', lw=1.5,
                     label='|1-r(N-1,N)|')
        axes[4].plot(iters, history['dev_curr'], color='orange', lw=1.5,
                     label='|1-r(N,N+1)|')
        axes[4].axhline(config.t_stop, color='red', ls='--', lw=1,
                        label=f't_stop={config.t_stop}')
        axes[4].set_title('(5) Stopping criterion\n(Eq 21)')
        axes[4].set_xlabel('Iteration')
        axes[4].legend(fontsize=8)
        axes[4].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Viz] Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()