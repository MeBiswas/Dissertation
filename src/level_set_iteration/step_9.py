# src/level_set_iteration/step_9.py

import numpy as np

from .step_8 import validate_segmentation

# =========================================================================
# STEP 9: Return the DLPE segmentation if valid, else fall back to SCH-CS masks.
# =========================================================================
def get_segmented_sr_with_fallback(
    level_set_result : dict,
    sr_regions : list,
    image_shape : tuple,
) -> tuple:
    seg = level_set_result['segmented_sr']
    ok, reason = validate_segmentation(seg, sr_regions, image_shape)

    if ok:
        print(f"[Validation] DLPE segmentation accepted: {reason}")
        return seg, 'dlpe'

    print(f"[Validation] DLPE rejected: {reason}")
    print("[Validation] Falling back to SCH-CS binary mask.")

    # Build fallback from SCH-CS sr_regions
    fallback = np.zeros(image_shape, dtype=np.uint8)
    for sr in sr_regions:
        fallback[sr['mask']] = 1
    return fallback, 'schcs_fallback'