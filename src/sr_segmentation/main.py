# src/sr_segmentation/main.py

from .step_1 import find_vertical_centre
from .step_2 import split_sr
from .step_3 import save_all
from .step_4 import visualize
from src.utils import create_run_folder

# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Full split pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_sr_split_pipeline(
    preprocessed_img,
    segmented_sr,
    config,
    image_name="image",
    centre_col_override=None
):
    
    print(f"[Process] SR Split → {image_name}")

    # --- Step 0: folder ---
    run_dir = create_run_folder(config.output_dir, image_name)

    # --- Step 1: centre ---
    if centre_col_override is not None:
        centre_col = centre_col_override
        print(f"[Split] Using manual centre: {centre_col}")
    else:
        centre_col = find_vertical_centre(preprocessed_img)

    # --- Step 2: split ---
    sr_left, sr_right = split_sr(segmented_sr, centre_col)

    # --- Step 3: save ---
    if config.save_results:
        save_all(run_dir, sr_left, sr_right, centre_col)

    # --- Step 4: visualize ---
    if config.save_visualization or config.show_visualization:
        visualize(
            preprocessed_img,
            segmented_sr,
            sr_left,
            sr_right,
            centre_col,
            run_dir,
            show=config.show_visualization
        )

    return {
        "sr_left": sr_left,
        "sr_right": sr_right,
        "centre_col": centre_col,
        "run_dir": run_dir
    }