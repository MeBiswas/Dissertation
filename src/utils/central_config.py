# utils/experiment_config.py

# ── Standard library ──────────────────────────────────────────────────────────
from typing import Dict
from dataclasses import dataclass, field

# ── Imports──────────────────────────────────────────────────────────
from .paths import base_path, bcd_dataset, dmr_ir_o

config_1 = {
    # "dataset_path": base_path + bcd_dataset['sick'],
    "dataset_path": base_path + dmr_ir_o,
    
    #processing mode
    "process_all": False,
    
    # used iff process_all=False
    "image_index": 6,
    
    # Cropping settings
    "enable_cropping": True,
    "crop_neck_percent": 0.26,      # Crop 18% from top (neck region)
    "crop_stomach_percent": 0.04,   # Crop 12% from bottom (stomach region)
    "crop_armpit_percent": 0.10,    # Crop 22% from sides (armpit regions)

    "hot_region_percentile": 95,
    "num_hot_regions": 2,
    
    # visualization
    "show_visualizations": True,
    
    # saving results
    "save_results": False,
    "output_dir": "outputs"
}

# ─────────────────────────────────────────────────────────────────────────────
#  CENTRAL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SchCsConfig:
    # ── CS isolation convergence threshold (paper: epsilon = 35) ──────────────
    epsilon : float = 35.0

    # ── Minimum region size (pixels) — noise removal ──────────────────────────
    min_region_px : int = 20

    # ── Edge-column guard ─────────────────────────────────────────────────────
    # Regions whose centroid column is within this fraction of either image edge
    # are almost certainly FLIR artefacts (text, calibration square), not SRs.
    edge_col_pct : float = 0.08

@dataclass
class PreprocessConfig:
    # ── FLIR overlay text bands (fraction of image dimension) ─────────────────
    # FLIR cameras burn parameter readouts into fixed pixel rows/columns.
    # These bands are zeroed BEFORE any other processing.
    overlay_top_pct   : float = 0.08   # top 8%  — temperature / date readouts
    overlay_left_pct  : float = 0.00   # left side (0 = none for this dataset)
    overlay_right_pct : float = 0.00   # right side colour bar removed separately

    # ── Colour scale bar detection (right-side vertical strip) ─────────────────
    colorbar_search_pct  : float = 0.15   # search rightmost 15% of image width
    colorbar_blue_thresh : int   = 200    # column blue-mean threshold

    # ── Background removal ────────────────────────────────────────────────────
    # Strategy: Otsu threshold on the GREEN channel.
    # WHY GREEN: In FLIR JPEG exports the background is near-black in all
    # channels, while body tissue is always non-zero in green.  Green gives
    # a clean bimodal histogram that Otsu splits correctly.
    bg_channel : str = 'green'   # 'green' | 'sum'  (sum = original v1 method)
    bg_sum_thresh : int = 30     # only used when bg_channel == 'sum'

    # ── Anatomical crop percentages (of bounding-box height / width) ──────────
    crop_neck_pct    : float = 0.16
    crop_stomach_pct : float = 0.14
    crop_armpit_pct  : float = 0.10

    # ── DMR-IR Format-B image dimensions ──────────────────────────────────────
    format_b_h : int = 120
    format_b_w : int = 160
    format_b_crop : Dict = field(default_factory=lambda: {
        'top': 18, 'bottom': 100, 'left': 0, 'right': 134
    })

SCH_CFG = SchCsConfig()
PRE_CFG = PreprocessConfig()