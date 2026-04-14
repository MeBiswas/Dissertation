# src/level_set_initialization/step_3.py

import os
import numpy as np
from typing import Dict

# ----------------------------------------------------------------------------
# Save outputs
# ----------------------------------------------------------------------------
def save_phi_outputs(run_dir: str, phi: np.ndarray) -> Dict[str, str]:
    phi_path = os.path.join(run_dir, "phi_initial.npy")
    np.save(phi_path, phi)
    
    print(f"\n[Save] φ saved: {phi_path}")
    
    return {
        "phi_path": phi_path,
        "run_dir": run_dir
    }