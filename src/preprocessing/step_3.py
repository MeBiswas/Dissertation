# preprocessing/step_3.py

# ── Numeric ──────────────────────────────────────────────────────────────────
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1.3 — Extract blue channel  →  grayscale  [Figure 2c]
# ─────────────────────────────────────────────────────────────────────────────
#
# In FLIR pseudo-colour JPEG images the Blue channel (index 0 in BGR)
# encodes temperature monotonically: hot = high value, cool = low value.
# This matches Figure 2(c) in the paper directly.
#
# NO normalisation, NO CLAHE, NO inversion — raw pixel values only.
# ─────────────────────────────────────────────────────────────────────────────

def extract_blue_channel(color_img: np.ndarray) -> np.ndarray:
    if color_img.ndim == 2:
        print('[1.3] Already grayscale — returned as-is.')
        return color_img.astype(np.uint8)

    gray = color_img[:, :, 0].copy()   # BGR → B = index 0
    print(f'[1.3] Blue channel extracted. Range: [{gray.min()}, {gray.max()}].')
    return gray