# src/features_extraction/config.py

from dataclasses import dataclass

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class FeatureConfig:
    output_dir: str = "feature_extraction"

    save_results: bool = True
    save_visualization: bool = True
    show_visualization: bool = True