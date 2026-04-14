# src/level_set_iteration/step_6.py

import os
import cv2
import json
import pickle
import numpy as np
from datetime import datetime
from typing import Dict, Tuple

from .storage_config import LevelSetConfig

# =========================================================================
# STEP 6: Save Results
# =========================================================================
def save_results(
    phi_final: np.ndarray,
    segmented_sr: np.ndarray,
    history: Dict,
    image_name: str,
    p_bar: np.ndarray,
    phi_init: np.ndarray,
    config: LevelSetConfig
) -> Tuple[Dict, str]:
    """Save all outputs to run directory."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config.output_dir, f"{image_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    saved_paths = {}
    
    if config.save_phi_final:
        path = os.path.join(run_dir, f"{image_name}_phi_final.npy")
        np.save(path, phi_final)
        saved_paths['phi_final'] = path
    
    if config.save_segmented_sr:
        npy_path = os.path.join(run_dir, f"{image_name}_segmented_sr.npy")
        np.save(npy_path, segmented_sr)
        saved_paths['segmented_sr_npy'] = npy_path
        
        png_path = os.path.join(run_dir, f"{image_name}_segmented_sr.png")
        cv2.imwrite(png_path, segmented_sr * 255)
        saved_paths['segmented_sr_png'] = png_path
    
    if config.save_history:
        path = os.path.join(run_dir, f"{image_name}_history.pkl")
        with open(path, 'wb') as f:
            pickle.dump(history, f)
        saved_paths['history'] = path
    
    if config.save_metadata:
        metadata = {
            'image_name': image_name,
            'timestamp': timestamp,
            'config': {
                'alpha1': config.alpha1,
                'alpha2': config.alpha2,
                'theta': config.theta,
                'epsilon': config.epsilon,
                'dt': config.dt,
                't_stop': config.t_stop,
                'max_iterations': config.max_iterations,
            },
            'p_bar_shape': p_bar.shape,
            'phi_init_shape': phi_init.shape,
            'phi_final_shape': phi_final.shape,
            'segmented_sr_pixels': int(segmented_sr.sum()),
            'segmented_sr_percent': float(100 * segmented_sr.mean()),
            'final_l1': float(history['l1'][-1]) if history['l1'] else None,
            'final_l2': float(history['l2'][-1]) if history['l2'] else None,
            'iterations': len(history['iteration']),
            'converged': len(history['iteration']) < config.max_iterations,
        }
        
        metadata_path = os.path.join(run_dir, f"{image_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        saved_paths['metadata'] = metadata_path
    
    return saved_paths, run_dir