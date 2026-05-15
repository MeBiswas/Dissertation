# sch_cs/step_8.py

# ── Standard library ──────────────────────────────────────────────────────────
from typing import Dict, List

# ── Numeric / image ───────────────────────────────────────────────────────────
import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2.8 — Boundary centroid correction  [Algorithm 1]  ← FIXED
# ─────────────────────────────────────────────────────────────────────────────
#
# BUG IN V1:
#   When the centroid fell outside the region, v1 snapped it to the nearest
#   BOUNDING-BOX CORNER.  For large or elongated regions the nearest corner
#   can be far from the actual region pixels — e.g. R(6) jumped from
#   (163, 100) to (10, 121) which is a corner of the BB but nowhere near
#   the actual region.
#
# FIX:
#   Build the set of all pixels ON THE REGION BOUNDARY and snap to the
#   nearest one.  The boundary is defined as any region pixel that has at
#   least one 4-connected neighbour that is NOT in the region.
#   This guarantees the corrected centroid is always a valid region pixel.
# ─────────────────────────────────────────────────────────────────────────────
def bounding_box_correction(
    regions : List[Dict],
    labeled_image : np.ndarray
) -> List[Dict]:
    print(f'\n[CS 2.8] Centroid boundary correction...')
    H, W = labeled_image.shape
    n_corrected = 0

    for reg in regions:
        X, Y = reg['centroid']
        xi, yj = int(round(X)), int(round(Y))

        in_bounds = (0 <= xi < H) and (0 <= yj < W)
        in_region = in_bounds and (labeled_image[xi, yj] == reg['label'])

        if in_region:
            reg['centroid_corrected'] = False
            continue

        # ── Build boundary pixel set ──────────────────────────────────────────
        # Use erosion: boundary = mask AND NOT eroded_mask
        mask_u8 = reg['mask'].astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(mask_u8, kernel, iterations=1)
        boundary = np.argwhere((mask_u8 - eroded) > 0)   # shape (N, 2)

        if len(boundary) == 0:
            # Region is a single pixel — use its coordinates directly
            boundary = reg['coords']

        # Nearest boundary pixel to original centroid
        dists = np.linalg.norm(boundary - np.array([X, Y]), axis=1)
        nearest = boundary[int(np.argmin(dists))]
        new_c = (float(nearest[0]), float(nearest[1]))

        print(f'  R({reg["label"]:>4}): ({X:.1f},{Y:.1f}) outside → '
              f'snapped to boundary pixel ({new_c[0]:.1f},{new_c[1]:.1f})')

        reg['centroid'] = new_c
        reg['centroid_corrected'] = True
        n_corrected += 1

    print(f'  {n_corrected} corrected, {len(regions)-n_corrected} already inside.')
    return regions