# sch_cs/step-5.py

# ── Standard library ──────────────────────────────────────────────────────────
from typing import Dict 

# ── Numeric / image ───────────────────────────────────────────────────────────
import numpy as np
from scipy.ndimage import label

# ── Utils Import ───────────────────────────────────────────────────────────
from src.utils import SCH_CFG, SchCsConfig

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2.5 — Threshold + label connected regions
# ─────────────────────────────────────────────────────────────────────────────

def threshold_and_label(
    pb  : np.ndarray,
    th  : float,
    cfg : SchCsConfig = SCH_CFG
) -> Dict:
    print(f'\n[SCH 2.5] Thresholding (th={th:.2f}) and labelling...')
    binary = (pb > th).astype(np.uint8)
    struct = np.ones((3, 3), dtype=int)
    labeled_image, n = label(binary, structure=struct)

    regions = []
    for k in range(1, n + 1):
        mask   = labeled_image == k
        coords = np.argwhere(mask)
        size   = int(mask.sum())
        if size < cfg.min_region_px:
            continue
        regions.append({'label': k, 'mask': mask, 'coords': coords, 'size': size})

    print(f'  SR pixels={int(binary.sum())}, '
          f'regions total={n}, after size filter={len(regions)}')
    return {'binary_image': binary, 'labeled_image': labeled_image, 'regions': regions}