# preprocessing/step_1.py

# ── Numeric ───────────────────────────────────────────────────────────
import numpy as np

# ── Utils Import ───────────────────────────────────────────────────────────
from src.utils import PRE_CFG, PreprocessConfig

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1.1 — Strip FLIR overlay bands
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY THIS IS NEEDED (and why v1 missed it):
#   FLIR cameras burn a HUD directly into the JPEG pixel data:
#     - Top rows   : temperature range, date, emissivity readout
#     - Crosshair  : centre measurement marker
#     - Blue square: calibration reference target
#
#   In v1 only the right-side colour bar was stripped.  The top HUD text
#   survived into p_b and produced high-intensity pixels that:
#     a) polluted the histogram used for rho / threshold computation
#     b) created false-positive regions in the thresholded binary image
#
# STRATEGY:
#   Zero out the top `overlay_top_pct` rows unconditionally —
#   the HUD is always in that band for DMR-IR images.
#   The calibration square (bright blue rectangle, lower half) is handled
#   separately in remove_color_scale because it is a distinct artefact.
# ─────────────────────────────────────────────────────────────────────────────
def strip_flir_overlay(
    color_img : np.ndarray,
    cfg       : PreprocessConfig = PRE_CFG
) -> np.ndarray:
    h, w  = color_img.shape[:2]
    out   = color_img.copy()

    top_px   = int(h * cfg.overlay_top_pct)
    left_px  = int(w * cfg.overlay_left_pct)
    right_px = int(w * cfg.overlay_right_pct)

    zeroed = []
    if top_px > 0:
        out[:top_px, :] = 0
        zeroed.append(f'top {top_px}px')
    if left_px > 0:
        out[:, :left_px] = 0
        zeroed.append(f'left {left_px}px')
    if right_px > 0:
        out[:, w-right_px:] = 0
        zeroed.append(f'right {right_px}px')

    if zeroed:
        print(f'[1.1] FLIR overlay zeroed: {", ".join(zeroed)}.')
    else:
        print('[1.1] No overlay bands configured — skipped.')
    return out