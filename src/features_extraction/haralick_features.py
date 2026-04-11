# src/features_extraction/haralick_features.py

"""
Stage 3 — Step 1: Feature Extraction from Segmented SRs
Implements Section III-A of Pramanik et al. (2018)
 
Extracts a 21-element feature vector from each breast's segmented SR:
    - 14 Haralick features  (texture, from GLCM — Equation 22)
    - 7  Hu's moment invariants (shape)
    Concatenated → [f_v]_{1×21}
 
Paper reference: Section III-A "Feature Extraction and Classifier Design"
 
    Eq 22: g_{p_b^s}(u,v) = G_sym(u,v / 1, 0°)
                             ────────────────────────────────
                             ΣΣ G_sym(u,v / 1, 0°)
 
    where G_sym(u,v/1,0°) = G(u,v/1,0°) + G^T(u,v/1,0°)
"""

import cv2
import numpy as np
from typing import Union
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix

# ═════════════════════════════════════════════════════════════════════════════
# PART 1 — GLCM + 14 HARALICK FEATURES  (Equation 22)
# ═════════════════════════════════════════════════════════════════════════════
 
def compute_glcm_normalized(image: np.ndarray) -> np.ndarray:
    """
    Computes the normalized symmetric GLCM as described in Equation 22.
 
    What is a GLCM?
        The Gray Level Co-occurrence Matrix (GLCM) counts how often pairs
        of pixel intensities appear next to each other in an image.
        G(u, v / distance=1, angle=0°) means:
            "How many times does gray level u appear directly to the LEFT
             of gray level v in the image?"
 
        Example in a tiny 3×3 image:
            [1, 2, 1]
            [2, 1, 2]
            [1, 2, 1]
            G(1,2) = 4  (1 appears 4 times to the left of 2)
            G(2,1) = 4  (2 appears 4 times to the left of 1)
 
    Why symmetric (G + G^T)?
        The raw GLCM is directional — G(u,v) ≠ G(v,u) in general.
        Adding the transpose makes it symmetric so texture is measured
        equally in both left→right and right→left directions.
 
    Why normalize?
        Dividing by the total count converts raw frequencies into
        probabilities: g(u,v) = probability that gray levels u and v
        are adjacent. This makes features scale-invariant.
 
    Paper details (Eq 22):
        - Distance = 1  (immediate neighbours)
        - Angle    = 0° (horizontal pairs only)
        - G_sym    = G + G^T   (symmetrized)
        - g        = G_sym / ΣΣG_sym  (normalized to probabilities)
 
    Args:
        image : Segmented SR region as uint8 grayscale array.
                Can be the full image with SR pixels intact and
                background = 0, OR just the cropped SR patch.
 
    Returns:
        g : Normalized symmetric GLCM as 2D float64 array
            Shape: (L, L) where L = number of gray levels in image
    """
    # ── Ensure uint8 ──────────────────────────────────────────────────────────
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
 
    # ── Get the number of gray levels present in the image ───────────────────
    # Using full 256 levels is standard but computationally heavy.
    # The paper uses L = number of gray levels in p_b^s (the segmented image).
    # We reduce to levels actually present for efficiency.
    n_levels = int(image.max()) + 1
    n_levels = max(n_levels, 2)             # need at least 2 levels
 
    # ── Compute raw GLCM  G(u,v / distance=1, angle=0°) ─────────────────────
    # skimage returns shape (L, L, n_distances, n_angles)
    # We use distance=1, angle=0 (horizontal) as the paper specifies
    glcm_raw = graycomatrix(
        image,
        distances=[1],
        angles=[0], # 0° = horizontal pairs
        levels=n_levels,
        symmetric=False, # we manually symmetrize below
        normed=False # we manually normalize below
    )
    G = glcm_raw[:, :, 0, 0].astype(np.float64)    # shape: (L, L)
 
    # ── Symmetrize: G_sym = G + G^T  (as in Eq 22) ───────────────────────────
    G_sym = G + G.T
 
    # ── Normalize: g = G_sym / ΣΣ G_sym ─────────────────────────────────────
    total = G_sym.sum()
    if total == 0:
        raise ValueError("GLCM is all zeros. Is the segmented SR empty?")
    g = G_sym / total # now g sums to 1.0
 
    return g # shape: (L, L), float64
 
 
def compute_haralick_features(g: np.ndarray) -> np.ndarray:
    """
    Computes all 14 Haralick texture features from normalized GLCM g.
    Reference: Haralick et al. (1973) — paper reference [28].
 
    The 14 features and their intuition:
 
    f1  — Angular Second Moment (Energy / Uniformity)
          High when image has very uniform texture (few gray-level pairs)
          f1 = ΣΣ g(u,v)²
 
    f2  — Contrast
          Measures intensity difference between a pixel and its neighbour.
          High for images with sharp edges / large local variations.
          f2 = ΣΣ (u-v)² * g(u,v)
 
    f3  — Correlation
          Measures linear dependency of gray levels across the image.
          High when rows/columns have consistent intensity trends.
          f3 = ΣΣ [(u-μ_u)(v-μ_v) * g(u,v)] / (σ_u * σ_v)
 
    f4  — Sum of Squares (Variance)
          Variance of gray level distribution. High = more spread out.
          f4 = ΣΣ (u - mean_g)² * g(u,v)
 
    f5  — Inverse Difference Moment (Homogeneity / Local Homogeneity)
          High when image is locally homogeneous (nearby pixels similar).
          f5 = ΣΣ g(u,v) / (1 + (u-v)²)
 
    f6  — Sum Average
          Mean of the sum distribution p_{x+y}.
          f6 = Σ i * p_{x+y}(i)   where i ∈ {2,...,2L}
 
    f7  — Sum Variance
          Variance of the sum distribution p_{x+y}.
          f7 = Σ (i - f8)² * p_{x+y}(i)
 
    f8  — Sum Entropy
          Entropy (randomness) of the sum distribution.
          f8 = -Σ p_{x+y}(i) * log(p_{x+y}(i))
 
    f9  — Entropy
          Overall randomness of the GLCM. High = complex texture.
          f9 = -ΣΣ g(u,v) * log(g(u,v))
 
    f10 — Difference Variance
          Variance of the difference distribution p_{x-y}.
          f10 = variance of p_{x-y}
 
    f11 — Difference Entropy
          Entropy of the difference distribution.
          f11 = -Σ p_{x-y}(i) * log(p_{x-y}(i))
 
    f12 — Information Measure of Correlation 1
          Compares actual correlation to maximum possible correlation.
          f12 = (f9 - HXY1) / max(HX, HY)
 
    f13 — Information Measure of Correlation 2
          Alternative correlation information measure.
          f13 = sqrt(1 - exp(-2(HXY2 - f9)))
 
    f14 — Maximal Correlation Coefficient
          Largest eigenvalue-based measure of correlation.
          f14 = sqrt(second_largest_eigenvalue(Q))
          where Q(u,v) = Σ_w [g(u,w)*g(v,w)] / (p_x(u)*p_y(w))
 
    Args:
        g : Normalized symmetric GLCM, shape (L, L), values sum to 1.0
 
    Returns:
        features : 1D float64 array of 14 Haralick features
    """
    L = g.shape[0]
    eps = 1e-10                             # avoid log(0)
 
    # ── Index arrays for vectorized computation ───────────────────────────────
    # u, v are (L, L) arrays where u[i,j]=i and v[i,j]=j
    u = np.arange(L).reshape(-1, 1) * np.ones((1, L))   # rows
    v = np.arange(L).reshape(1, -1) * np.ones((L, 1))   # cols
 
    # ── Marginal probabilities ────────────────────────────────────────────────
    # p_x(u) = Σ_v g(u,v)  — probability of gray level u in first position
    # p_y(v) = Σ_u g(u,v)  — probability of gray level v in second position
    p_x = g.sum(axis=1) # shape: (L,)
    p_y = g.sum(axis=0) # shape: (L,)
 
    # Means and standard deviations of marginal distributions
    idx  = np.arange(L, dtype=np.float64)
    mu_x = np.sum(idx * p_x)
    mu_y = np.sum(idx * p_y)
    sg_x = np.sqrt(np.sum((idx - mu_x)**2 * p_x) + eps)
    sg_y = np.sqrt(np.sum((idx - mu_y)**2 * p_y) + eps)
 
    # Overall mean of g
    mean_g = np.sum(u * g)
 
    # ── Sum and difference distributions ─────────────────────────────────────
    # p_{x+y}(k) = Σ_{u+v=k} g(u,v)  for k = 0, 1, ..., 2(L-1)
    # p_{x-y}(k) = Σ_{|u-v|=k} g(u,v) for k = 0, 1, ..., L-1
    p_xy  = np.zeros(2 * L) # sum distribution
    p_xmy = np.zeros(L) # difference distribution
 
    for i in range(L):
        for j in range(L):
            p_xy[i + j]     += g[i, j]
            p_xmy[abs(i-j)] += g[i, j]
 
    # ── Entropy terms needed for f12, f13 ────────────────────────────────────
    # HX = entropy of p_x, HY = entropy of p_y
    HX  = -np.sum(p_x[p_x > eps] * np.log(p_x[p_x > eps] + eps))
    HY  = -np.sum(p_y[p_y > eps] * np.log(p_y[p_y > eps] + eps))
 
    # HXY = entropy of g itself (= f9)
    g_nz = g[g > eps]
    HXY = -np.sum(g_nz * np.log(g_nz + eps))
 
    # HXY1 = -ΣΣ g(u,v) * log(p_x(u) * p_y(v))
    px_mat = p_x.reshape(-1, 1) * np.ones((1, L))  # broadcast to (L,L)
    py_mat = p_y.reshape(1, -1) * np.ones((L, 1))
    prod = px_mat * py_mat
    HXY1 = -np.sum(g * np.log(prod + eps))
 
    # HXY2 = -ΣΣ p_x(u)*p_y(v) * log(p_x(u)*p_y(v))
    HXY2 = -np.sum(prod * np.log(prod + eps))
 
    # ─────────────────────────────────────────────────────────────────────────
    # THE 14 HARALICK FEATURES
    # ─────────────────────────────────────────────────────────────────────────
 
    # f1: Angular Second Moment (Energy)
    f1 = np.sum(g ** 2)
 
    # f2: Contrast
    f2 = np.sum((u - v) ** 2 * g)
 
    # f3: Correlation
    f3 = np.sum((u - mu_x) * (v - mu_y) * g) / (sg_x * sg_y + eps)
 
    # f4: Sum of Squares (Variance)
    f4 = np.sum((u - mean_g) ** 2 * g)
 
    # f5: Inverse Difference Moment (Homogeneity)
    f5 = np.sum(g / (1.0 + (u - v) ** 2))
 
    # f6: Sum Average
    k_arr = np.arange(2 * L, dtype=np.float64)
    f6 = np.sum(k_arr * p_xy)
 
    # f7: Sum Variance (uses f8 — Sum Entropy — so compute f8 first)
    p_xy_nz = p_xy[p_xy > eps]
    f8 = -np.sum(p_xy_nz * np.log(p_xy_nz + eps)) # Sum Entropy
    f7 = np.sum((k_arr - f8) ** 2 * p_xy) # Sum Variance
 
    # f8 already computed above (Sum Entropy)
 
    # f9: Entropy
    f9 = HXY
 
    # f10: Difference Variance
    k_diff = np.arange(L, dtype=np.float64)
    f10 = np.var(p_xmy)
 
    # f11: Difference Entropy
    p_xmy_nz = p_xmy[p_xmy > eps]
    f11 = -np.sum(p_xmy_nz * np.log(p_xmy_nz + eps))
 
    # f12: Information Measure of Correlation 1
    f12 = (HXY - HXY1) / (max(HX, HY) + eps)
 
    # f13: Information Measure of Correlation 2
    val_f13 = 1.0 - np.exp(-2.0 * (HXY2 - HXY))
    f13 = np.sqrt(max(val_f13, 0.0))
 
    # f14: Maximal Correlation Coefficient
    # Q(u,v) = Σ_w g(u,w)*g(v,w) / (p_x(u) * p_y(w))
    Q = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            denom = p_x[i] * p_y + eps # shape (L,)
            Q[i, j] = np.sum(g[i, :] * g[j, :] / denom)
 
    eigenvalues = np.linalg.eigvals(Q)
    eigenvalues = np.sort(np.abs(eigenvalues.real))[::-1]   # descending
    f14 = np.sqrt(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
 
    features = np.array(
        [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14],
        dtype=np.float64
    )
 
    return features # shape: (14,)