# src/level_set_iteration/storage_config.py

from dataclasses import dataclass

# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class LevelSetConfig:
    # Mathematical parameters
    alpha1: float = 1.0
    alpha2: float = 1.0
    theta: float = 0.2
    epsilon: float = 1.5
    dt: float = 0.1
    t_stop: float = 0.05
    max_iterations: int = 1000
    verbose_every: int = 50
    
    # Storage settings
    output_dir: str = "data/level_set_results"
    save_visualization: bool = True
    save_phi_final: bool = True
    save_segmented_sr: bool = True
    save_history: bool = True
    save_metadata: bool = True