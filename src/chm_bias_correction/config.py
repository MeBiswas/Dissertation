# src/chm_bias_correction/config.py

from dataclasses import dataclass

@dataclass
class CHMConfig:
    window_size: int = 9
    order: int = 1
    eps: float = 1e-10
    
    save_visualization: bool = False
    show_visualization: bool = True
    visualization_path: str = "chm_result.png"
    
    output_dir: str = "outputs"
    save_corrected_image: bool = True