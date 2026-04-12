import os
import numpy as np
from datetime import datetime

from .step_1 import compute_asymmetry
from .step_2 import save_asymmetry
from .step_3 import visualize_asymmetry

# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def run_asymmetry_pipeline(
    f_v_left: np.ndarray,
    f_v_right: np.ndarray,
    image_name: str,
    base_output_dir: str,
    label: str = "Unknown",
    save_visualization: bool = True,
    show_visualization: bool = True
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_output_dir, f"{image_name}_{timestamp}")

    print(f"[Asymmetry] Processing → {run_dir}")

    # Step 1
    F = compute_asymmetry(f_v_left, f_v_right)

    # Step 2
    save_asymmetry(F, run_dir)

    # Step 3
    if save_visualization or show_visualization:
        visualize_asymmetry(
            f_v_left, f_v_right, F,
            run_dir=run_dir,
            label=label,
            save=save_visualization,
            show=show_visualization
        )

    return {
        "F": F,
        "run_dir": run_dir
    }