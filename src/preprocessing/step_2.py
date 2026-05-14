# preprocessing/step_2.py

# ── Numeric / Image ────────────────────────────────────────────────────────
import numpy as np
from scipy.ndimage import label

# ── Utils Import ───────────────────────────────────────────────────────────
from src.utils import PRE_CFG, PreprocessConfig

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1.2 — Remove colour scale bar + calibration square
# ─────────────────────────────────────────────────────────────────────────────
#
# TWO artefacts are handled here:
#
# A) RIGHT-SIDE COLOUR BAR
#    Detected by scanning columns from the right: if a column's mean blue
#    value >= colorbar_blue_thresh it is part of the bar and is zeroed.
#    Scanning stops at the first column that falls below the threshold.
#
# B) BLUE CALIBRATION SQUARE
#    A small, compact, near-255-blue rectangle that FLIR places in the
#    lower portion of the image as a temperature reference.
#    Detected as: very bright blue (> 240), small (< 0.5% of image),
#    located below the top 40%, and roughly square (aspect ratio < 3).
# ─────────────────────────────────────────────────────────────────────────────
def remove_color_scale(
    color_img : np.ndarray,
    cfg : PreprocessConfig = PRE_CFG
) -> np.ndarray:
    h, w = color_img.shape[:2]
    out = color_img.copy()

    # ── Format B: fixed crop ──────────────────────────────────────────────────
    if h == cfg.format_b_h and w == cfg.format_b_w:
        c = cfg.format_b_crop
        result = out[c['top']:c['bottom'], c['left']:c['right']].copy()
        print(f'[1.2] Format-B ({h}×{w}). Hard-cropped to '
            f'{result.shape[0]}×{result.shape[1]}.')
        return result

    # ── A) Right-side colour bar ───────────────────────────────────────────────
    blue_col_means = out[:, :, 0].mean(axis=0)   # B in BGR = index 0
    bar_start = w   # will move left while bar columns are found

    for col in range(w - 1, w - 1 - int(w * cfg.colorbar_search_pct), -1):
        if blue_col_means[col] >= cfg.colorbar_blue_thresh:
            bar_start = col
        else:
            break

    if bar_start < w:
        out[:, bar_start:] = 0
        print(f'[1.2] Colour bar zeroed: cols {bar_start}–{w-1}.')
    else:
        print('[1.2] No colour bar detected.')

    # ── B) Calibration square ─────────────────────────────────────────────────
    blue_ch = out[:, :, 0]
    bright_mask = (blue_ch > 240).astype(np.uint8)
    bright_mask[:int(h * 0.40), :] = 0   # only look below top 40%

    lbl, n = label(bright_mask)
    total = h * w
    n_removed = 0

    for k in range(1, n + 1):
        m = lbl == k
        size = int(m.sum())
        if size > 0.005 * total:
            continue   # too large — genuine tissue
        coords = np.argwhere(m)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        ar = max(r1-r0, c1-c0) / (min(r1-r0, c1-c0) + 1e-6)
        if ar > 3.0:
            continue   # too elongated — not a square
        out[m] = 0
        n_removed += 1
        print(f'[1.2] Calibration square removed: '
            f'size={size}px, bbox=({r1-r0}×{c1-c0}), aspect={ar:.2f}.')

    if n_removed == 0:
        print('[1.2] No calibration square detected.')

    return out
