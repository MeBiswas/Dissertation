# src/level_set_initialization/main.py

import os
import cv2
import numpy as np

from .config import PhiInitConfig
from .step_1 import build_sr_mask_from_regions
from .step_2 import initialize_phi
from .step_3 import save_phi_outputs
from .step_4 import visualize_phi_init
from src.utils import create_run_folder

from typing import List, Dict, Optional, Union

# ----------------------------------------------------------------------------
# Main Pipeline Function
# ----------------------------------------------------------------------------
def run_phi_init_pipeline(
    preprocessed_image: Union[str, np.ndarray],
    sr_regions: List[Dict],
    config: PhiInitConfig,
    image_name: Optional[str] = None
) -> Dict:
    # --- Load preprocessed image ---
    if isinstance(preprocessed_image, str):
        p_b = cv2.imread(preprocessed_image, cv2.IMREAD_GRAYSCALE)
        if p_b is None:
            raise FileNotFoundError(f"Cannot load: {preprocessed_image}")
        if image_name is None:
            image_name = os.path.splitext(os.path.basename(preprocessed_image))[0]
    else:
        p_b = preprocessed_image
        if image_name is None:
            image_name = "image"
    
    print(f"\n{'='*60}")
    print(f"[φ Init] Processing: {image_name}")
    print(f"{'='*60}")
    print(f"  Preprocessed shape: {p_b.shape}")
    print(f"  SR regions count: {len(sr_regions)}")
    
    # --- Step 0: Create run folder ---
    run_dir = create_run_folder(config.output_dir, image_name)
    print(f"\n[Run Dir] {run_dir}")
    
    # --- Step 1: Build SR mask from VALIDATED regions (THE FIX) ---
    sr_mask = build_sr_mask_from_regions(sr_regions, shape=p_b.shape)
    
    # --- Step 2: Initialize φ (Equation 17) ---
    phi = initialize_phi(
        sr_mask,
        inside_value=config.inside_value,
        outside_value=config.outside_value
    )
    
    # --- Step 3: Save φ array ---
    save_paths = {}
    if config.save_phi_array:
        save_paths = save_phi_outputs(run_dir, phi)
    
    # --- Step 4: Visualize ---
    if config.save_visualization or config.show_visualization:
        visualize_phi_init(
            p_b,
            sr_mask,
            phi,
            run_dir,
            show=config.show_visualization
        )
    
    # --- Return results ---
    return {
        "phi": phi,
        "sr_mask": sr_mask,
        "run_dir": run_dir,
        "saved_paths": save_paths
    }