# preprocessing/main.py

# ── Standard library ──────────────────────────────────────────────────────────
import os
from typing import Dict

# ── Numeric / image ───────────────────────────────────────────────────────────
import cv2

# ── Utils Import ───────────────────────────────────────────────────────────
from src.utils import PRE_CFG, PreprocessConfig

# ── Preprocessing Import ───────────────────────────────────────────────────────────
from .step_4 import remove_background
from .step_1 import strip_flir_overlay
from .step_2 import remove_color_scale
from .step_3 import extract_blue_channel
from .step_6 import visualize_preprocessing
from .step_5 import (crop_anatomical_regions,
                    gray_level_reconstruction)

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ENTRY — run_preprocessing()
# ─────────────────────────────────────────────────────────────────────────────

def run_preprocessing(
    image_path : str,
    cfg        : PreprocessConfig = PRE_CFG,
    visualize  : bool = True
) -> Dict:
    """
    Full pre-processing pipeline.

    Returns dict with keys:
        'pb'             — final p_b (uint8) ready for SCH-CS
        'grayscale'      — cropped blue channel
        'bg_removed'     — cropped background-removed image
        'without_scale'  — colour image after overlay strip + bar removal
        'original_color' — raw BGR image
        'image_name'     — filename
    """
    print('=' * 60)
    print(f'  PRE-PROCESSING: {os.path.basename(image_path)}')
    print('=' * 60)

    original_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if original_color is None:
        raise FileNotFoundError(f'Cannot read: {image_path}')
    print(f'[1.0] Loaded {original_color.shape[0]}×{original_color.shape[1]}.')

    # Steps 1.1 → 1.2 → 1.3 → 1.4 → 1.5
    clean_color   = strip_flir_overlay(original_color, cfg)
    without_scale = remove_color_scale(clean_color, cfg)
    grayscale     = extract_blue_channel(without_scale)
    bg_removed, body_mask = remove_background(grayscale, without_scale, cfg)

    cropped_bg, (r0, r1, c0, c1) = crop_anatomical_regions(
        bg_removed, body_mask, cfg
    )
    grayscale_cropped = grayscale[r0:r1, c0:c1]
    pb = gray_level_reconstruction(cropped_bg, grayscale_cropped)

    print(f'\n[Done] p_b shape={pb.shape}, '
          f'non-zero={int((pb>0).sum())} pixels.')

    if visualize:
        visualize_preprocessing(
            original_color[r0:r1, c0:c1],
            without_scale[r0:r1,  c0:c1],
            grayscale_cropped,
            bg_removed[r0:r1, c0:c1],
            pb,
            os.path.basename(image_path)
        )

    return {
        'pb'            : pb,
        'grayscale'     : grayscale_cropped,
        'bg_removed'    : bg_removed[r0:r1, c0:c1],
        'without_scale' : without_scale[r0:r1, c0:c1],
        'original_color': original_color[r0:r1, c0:c1],
        'image_name'    : os.path.basename(image_path),
    }