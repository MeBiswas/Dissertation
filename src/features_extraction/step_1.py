# src/features_extraction/step_1.py

import numpy as np

# ═════════════════════════════════════════════════════════════════════════════
# EXTRACT SR REGION
# ═════════════════════════════════════════════════════════════════════════════
def extract_sr_region(segmented_sr, original_gray):
    coords = np.argwhere(segmented_sr > 0)
    if len(coords) == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    
    # Crop to bounding box, then apply mask within it
    sr_crop  = original_gray[r0:r1+1, c0:c1+1].copy()
    mask_crop = segmented_sr[r0:r1+1, c0:c1+1]
    sr_crop[mask_crop == 0] = 0
    return sr_crop