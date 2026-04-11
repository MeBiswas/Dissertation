# src/features_extraction/hu_moments.py

import cv2
import numpy as np

# ═════════════════════════════════════════════════════════════════════════════
# PART 2 — 7 HU'S MOMENT INVARIANTS  (Hu, 1962 — paper reference [29])
# ═════════════════════════════════════════════════════════════════════════════
 
def compute_hu_moments(image: np.ndarray) -> np.ndarray:
    """
    Computes Hu's 7 moment invariants from the segmented SR image.
    Reference: Hu (1962) — paper reference [29].
 
    What are moments?
        Image moments are weighted sums of pixel intensities that capture
        the "shape" of a region — similar to how mean and variance
        summarize a distribution.
 
        Raw moment M_{pq} = ΣΣ x^p * y^q * I(x,y)
        Central moment μ_{pq} = ΣΣ (x-x̄)^p * (y-ȳ)^q * I(x,y)
        Normalized central moment η_{pq} = μ_{pq} / μ_{00}^γ
 
    What makes Hu's moments special?
        Hu derived 7 combinations of normalized central moments that
        are INVARIANT to:
            ✓ Translation  (shifting the image)
            ✓ Rotation     (rotating the image)
            ✓ Scale        (resizing the image)
 
        This is crucial for TBIs because:
            - Patients are positioned slightly differently each scan
            - Breasts vary in size between patients
            - We need features that capture SHAPE, not position/size
 
    The 7 invariants (φ1 ... φ7):
        φ1 = η20 + η02
        φ2 = (η20 - η02)² + 4η11²
        φ3 = (η30 - 3η12)² + (3η21 - η03)²
        φ4 = (η30 + η12)² + (η21 + η03)²
        φ5 = (η30-3η12)(η30+η12)[(η30+η12)²-3(η21+η03)²]
             + (3η21-η03)(η21+η03)[3(η30+η12)²-(η21+η03)²]
        φ6 = (η20-η02)[(η30+η12)²-(η21+η03)²]
             + 4η11(η30+η12)(η21+η03)
        φ7 = (3η21-η03)(η30+η12)[(η30+η12)²-3(η21+η03)²]
             - (η30-3η12)(η21+η03)[3(η30+η12)²-(η21+η03)²]
 
    We use log transform: log|φi| to compress the wide dynamic range.
 
    Args:
        image : Segmented SR image (uint8 grayscale).
                Background pixels should be 0.
 
    Returns:
        hu_features : 1D float64 array of 7 Hu's moment invariants
                      Values are log-transformed: log|φi|
    """
    # Ensure float32 for OpenCV moments computation
    img_float = image.astype(np.float32)
 
    # ── Compute image moments using OpenCV ───────────────────────────────────
    # cv2.moments() returns a dict with raw, central, and normalized moments
    moments = cv2.moments(img_float)
 
    # ── Get normalized central moments η_{pq} ────────────────────────────────
    # OpenCV provides nu20, nu02, nu11, nu30, nu12, nu21, nu03 directly
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
    # Hu's moments span many orders of magnitude (e.g. 1e-2 to 1e-20).
    # Log transform brings them to a comparable scale for the neural network.
    # sign(φi) * log|φi| preserves the sign while compressing magnitude.
    hu_features = np.array([
        np.sign(p) * np.log10(abs(p) + 1e-10)
        for p in phi
    ], dtype=np.float64)
 
    return hu_features                      # shape: (7,)