# preprocessing/step_5.py

# ── Standard library ──────────────────────────────────────────────────────────
from typing import Tuple

# ── Numeric / Image ────────────────────────────────────────────────────────
import numpy as np

# ── Utils Import ───────────────────────────────────────────────────────────
from src.utils import PRE_CFG, PreprocessConfig

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1.5 — Anatomical crop 
# ─────────────────────────────────────────────────────────────────────────────
#
# CROP: Remove neck (top), stomach (bottom), armpits (sides).
# These are warm regions that would push the threshold upward and cause
# genuine breast SRs to be missed.
#
# ─────────────────────────────────────────────────────────────────────────────
def crop_anatomical_regions(
    bg_removed : np.ndarray,
    body_mask : np.ndarray,
    cfg : PreprocessConfig = PRE_CFG
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Crop neck / stomach / armpit bands. Returns (cropped, (r0,r1,c0,c1))."""
    coords = np.argwhere(body_mask > 0)
    if len(coords) == 0:
        h, w = bg_removed.shape
        return bg_removed.copy(), (0, h, 0, w)

    min_row, min_col = coords.min(axis=0)
    max_row, max_col = coords.max(axis=0)
    height = max_row - min_row
    width = max_col - min_col

    top_cut = min(int(height * cfg.crop_neck_pct), height // 4)
    bottom_cut = min(int(height * cfg.crop_stomach_pct), height // 4)
    side_cut = min(int(width  * cfg.crop_armpit_pct), width  // 4)

    r0 = min_row + top_cut
    r1 = max_row - bottom_cut
    c0 = min_col + side_cut
    c1 = max_col - side_cut

    cropped = bg_removed[r0:r1, c0:c1].copy()
    print(f'[1.5a] Crop: neck={top_cut}px, stomach={bottom_cut}px, '
        f'sides={side_cut}px. {bg_removed.shape} → {cropped.shape}')
    return cropped, (r0, r1, c0, c1)