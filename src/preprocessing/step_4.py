# preprocessing/step_4.py

# ── Standard library ──────────────────────────────────────────────────────────
from typing import Tuple

# ── Numeric / Image ────────────────────────────────────────────────────────
import cv2
import numpy as np
from scipy.ndimage import label

# ── Utils Import ───────────────────────────────────────────────────────────
from src.utils import PRE_CFG, PreprocessConfig

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1.4 — Remove background  →  bg_removed  [Figure 2d]
# ─────────────────────────────────────────────────────────────────────────────
#
# BUG IN V1:
#   channel_sum > 15 kept ALL 307 200 pixels (entire image).
#   This happened because this particular FLIR image has a warm thermal
#   gradient across the whole frame — no pixel is truly dark in all channels.
#
# FIX — Otsu on the GREEN channel:
#   WHY GREEN: In FLIR JPEG exports the background region (outside the body)
#   is encoded as cold blue (high B, low G, low R).  The Green channel
#   therefore gives the cleanest separation between background (near-zero G)
#   and body tissue (non-zero G across all temperatures).
#   Otsu automatically finds the optimal threshold without any hand-tuning.
#
#   After Otsu we:
#     1. Apply morphological closing (fills tiny interior holes)
#     2. Keep only the largest connected component (= the body)
# ─────────────────────────────────────────────────────────────────────────────

def remove_background(
    grayscale    : np.ndarray,
    color_no_scale: np.ndarray,
    cfg          : PreprocessConfig = PRE_CFG
) -> Tuple[np.ndarray, np.ndarray]:
    if cfg.bg_channel == 'green':
        # ── Otsu on green channel (index 1 in BGR) ────────────────────────────
        green    = color_no_scale[:, :, 1]
        otsu_thr, _ = cv2.threshold(
            green, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        body_mask_u8 = (green > otsu_thr).astype(np.uint8)
        print(f'[1.4] Otsu on green channel: threshold = {otsu_thr:.1f}.')
    else:
        # ── Fallback: channel sum ─────────────────────────────────────────────
        channel_sum  = color_no_scale.astype(np.float32).sum(axis=2)
        body_mask_u8 = (channel_sum > cfg.bg_sum_thresh).astype(np.uint8)
        print(f'[1.4] Channel-sum threshold = {cfg.bg_sum_thresh}.')

    # Close small interior holes
    kernel       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    body_mask_u8 = cv2.morphologyEx(body_mask_u8, cv2.MORPH_CLOSE, kernel)

    # Keep largest connected component
    labeled_arr, n = label(body_mask_u8)
    if n > 1:
        sizes        = np.bincount(labeled_arr.ravel())
        sizes[0]     = 0
        largest      = sizes.argmax()
        body_mask_u8 = (labeled_arr == largest).astype(np.uint8)
        print(f'[1.4] {n} components found — kept largest (body).')
    else:
        print(f'[1.4] Single component detected.')

    n_body = int(body_mask_u8.sum())
    pct    = 100 * n_body / body_mask_u8.size
    print(f'[1.4] Body pixels: {n_body} ({pct:.1f}% of image).')

    bg_removed = (grayscale * body_mask_u8).astype(np.uint8)
    return bg_removed, body_mask_u8