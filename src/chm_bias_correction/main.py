# src/chm_bias_correction/main.py

import os
from .step_1 import compute_chm_local
from .step_2 import compute_nc
from .step_3 import compute_p_bar
from .step_4 import save_all_outputs
from .step_5 import visualize_chm
from src.utils import create_run_folder

def run_chm_pipeline(image, config, image_name="image"):
    
    print("[Step 0] Creating run folder...")
    run_dir = create_run_folder(config.output_dir, image_name)
    
    print(f"[INFO] Outputs will be saved in: {run_dir}")
    
    print("[Step 1] Computing CHM local...")
    chm_local = compute_chm_local(image, config.window_size, config.order)

    print("[Step 2] Computing Nc...")
    Nc = compute_nc(image, config.order, config.eps)

    print("[Step 3] Computing p_bar...")
    p_bar = compute_p_bar(image, chm_local, Nc, config.eps)

    if config.save_corrected_image:
        print("[Step 4] Saving all outputs...")
        save_paths = save_all_outputs(run_dir, p_bar, chm_local, Nc)

    if config.show_visualization or config.save_visualization:
        print("[Step 5] Visualizing...")
        viz_path = None

        if config.save_visualization:
            viz_path = os.path.join(run_dir, "visualization.png")

        visualize_chm(
            image,
            chm_local,
            p_bar,
            eps=config.eps,
            save_path=viz_path,
            show=config.show_visualization
        )

    return {
        "Nc": Nc,
        "p_bar": p_bar,
        "run_dir": run_dir,
        "chm_local": chm_local,
    }