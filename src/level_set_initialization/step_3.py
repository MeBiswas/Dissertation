# src/level_set_initialization/step_3.py

import os
import numpy as np

def save_all(run_dir, phi):
    
    phi_path = os.path.join(run_dir, "phi.npy")
    np.save(phi_path, phi)

    return {
        "phi_path": phi_path
    }