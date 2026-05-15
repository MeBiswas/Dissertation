# src/features_extraction/step_4.py

import cv2
import numpy as np

# ═════════════════════════════════════════════════════════════════════════════
# PART 2 — 7 HU'S MOMENT INVARIANTS  (Hu, 1962 — paper reference [29])
# ═════════════════════════════════════════════════════════════════════════════
def compute_hu_moments(image: np.ndarray) -> np.ndarray:
    img_float = image.astype(np.float32)

    # ── Compute image moments using OpenCV ───────────────────────────────────
    moments = cv2.moments(img_float)

    # ── Get normalized central moments η_{pq} ────────────────────────────────
    η20 = moments['nu20']
    η02 = moments['nu02']
    η11 = moments['nu11']
    η30 = moments['nu30']
    η12 = moments['nu12']
    η21 = moments['nu21']
    η03 = moments['nu03']

    # ── Compute the 7 Hu invariants ───────────────────────────────────────────
    phi = np.zeros(7, dtype=np.float64)

    phi[0] = η20 + η02

    phi[1] = (η20 - η02)**2 + 4 * η11**2

    phi[2] = (η30 - 3*η12)**2 + (3*η21 - η03)**2

    phi[3] = (η30 + η12)**2 + (η21 + η03)**2

    phi[4] = (  (η30 - 3*η12) * (η30 + η12)
              * ((η30 + η12)**2 - 3*(η21 + η03)**2)
              + (3*η21 - η03) * (η21 + η03)
              * (3*(η30 + η12)**2 - (η21 + η03)**2) )

    phi[5] = (  (η20 - η02)
              * ((η30 + η12)**2 - (η21 + η03)**2)
              + 4 * η11 * (η30 + η12) * (η21 + η03) )

    phi[6] = (  (3*η21 - η03) * (η30 + η12)
              * ((η30 + η12)**2 - 3*(η21 + η03)**2)
              - (η30 - 3*η12) * (η21 + η03)
              * (3*(η30 + η12)**2 - (η21 + η03)**2) )

    # ── Log transform to compress dynamic range ───────────────────────────────
    hu_features = np.array([
        np.sign(p) * np.log10(abs(p) + 1e-10)
        for p in phi
    ], dtype=np.float64)

    hu_features = np.clip(hu_features, -6, 6)

    return hu_features