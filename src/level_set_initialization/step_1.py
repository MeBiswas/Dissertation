# src/level_set_initialization/step_1.py

import numpy as np
from typing import List, Dict, Tuple

def prepare_binary(schcs_binary: np.ndarray):
    p_th_b = schcs_binary.astype(np.float64)

    if p_th_b.max() > 1.0:
        p_th_b = p_th_b / 255.0
        print("[Init] Binary normalized [0,255] → [0,1]")

    unique_vals = np.unique(p_th_b)

    if not np.all(np.isin(unique_vals, [0.0, 1.0])):
        print(f"[Warn] Unexpected values: {unique_vals} → thresholding @0.5")
        p_th_b = (p_th_b >= 0.5).astype(np.float64)

    return p_th_b

# ----------------------------------------------------------------------------
# Core: Build SR mask from validated regions (THE FIX)
# ----------------------------------------------------------------------------
def build_sr_mask_from_regions(
    sr_regions: List[Dict],
    shape: Tuple[int, int]
) -> np.ndarray:
    p_th_b = np.zeros(shape, dtype=np.uint8)
    total_pixels = 0
    
    print("\n[SR Mask] Building from validated SR regions:")
    for sr in sr_regions:
        mask = sr['mask']
        p_th_b[mask] = 1
        total_pixels += int(mask.sum())
        
        # Print region info if available
        label = sr.get('label', '?')
        size = sr.get('size', mask.sum())
        cent = sr.get('centroid', (None, None))
        if cent[0] is not None:
            print(f"  SR {label}: {size}px, centroid=({cent[0]:.1f}, {cent[1]:.1f})")
        else:
            print(f"  SR {label}: {size}px")
    
    print(f"  Total SR pixels: {total_pixels} ({100*total_pixels/p_th_b.size:.2f}% of image)")
    return p_th_b