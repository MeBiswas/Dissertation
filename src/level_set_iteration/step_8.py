# src/level_set_iteration/step_8.py

import numpy as np

# =========================================================================
# STEP 8: Validate that DLPE produced a plausible segmentation.
# =========================================================================
def validate_segmentation(
    segmented_sr : np.ndarray,
    sr_regions : list,
    image_shape : tuple,
    max_sr_pct : float = 0.20,
    min_sr_px : int = 50,
) -> tuple:
    total_px = segmented_sr.size
    sr_px    = int(segmented_sr.sum())
    sr_pct   = sr_px / total_px

    if sr_px < min_sr_px:
        return False, f"SR too small: {sr_px}px < {min_sr_px}px minimum"

    if sr_pct > max_sr_pct:
        return False, (
            f"SR too large: {sr_pct*100:.1f}% > {max_sr_pct*100:.0f}% max "
            f"(contour likely exploded)"
        )

    return True, f"OK: {sr_px}px ({sr_pct*100:.2f}%)"