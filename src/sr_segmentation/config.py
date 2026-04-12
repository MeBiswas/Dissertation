# src/sr_segmentation/config.py

from dataclasses import dataclass

@dataclass
class SRSplitConfig:
    output_dir: str = "sr_splits"
    
    save_results: bool = True
    save_visualization: bool = True
    show_visualization: bool = True