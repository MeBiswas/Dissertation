# src/level_set_initialization/config.py

from dataclasses import dataclass

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
@dataclass
class PhiInitConfig:
    inside_value: float = 4.0
    outside_value: float = -4.0
    
    save_visualization: bool = True
    show_visualization: bool = True
    
    save_phi_array: bool = True
    output_dir: str = "phi_initialized"