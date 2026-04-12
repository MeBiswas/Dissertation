# src/level_set_initialization/main.py

import os
import cv2

from .step_1 import prepare_binary
from .step_2 import initialize_phi
from .step_3 import save_all
from .step_4 import visualize
from src.utils import create_run_folder

def run_phi_init_pipeline(
    preprocessed_image_path,
    schcs_binary_path,
    config,
    image_name=None
):
    # --- Load ---
    p_b = cv2.imread(preprocessed_image_path, cv2.IMREAD_GRAYSCALE)
    schcs_binary = cv2.imread(schcs_binary_path, cv2.IMREAD_GRAYSCALE)

    if p_b is None or schcs_binary is None:
        raise FileNotFoundError("Image loading failed")

    if image_name is None:
        image_name = os.path.splitext(os.path.basename(preprocessed_image_path))[0]

    if p_b.shape != schcs_binary.shape:
        raise ValueError("Shape mismatch")

    print(f"[Process] {image_name}")

    # --- Step 0: create run folder ---
    run_dir = create_run_folder(config.output_dir, image_name)

    # --- Step 1 ---
    p_th_b = prepare_binary(schcs_binary)

    # --- Step 2 ---
    phi = initialize_phi(
        p_th_b,
        config.inside_value,
        config.outside_value
    )

    # --- Step 3 ---
    if config.save_phi_array:
        save_paths = save_all(run_dir, phi)

    # --- Step 4 ---
    if config.save_visualization or config.show_visualization:
        visualize(
            p_b,
            p_th_b,
            phi,
            run_dir,
            show=config.show_visualization
        )

    return {
        "phi": phi,
        "run_dir": run_dir
    }