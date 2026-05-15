# src/features_extraction/step_3.py

import numpy as np

# ═════════════════════════════════════════════════════════════════════════════
# PART 2 — 14 HARALICK FEATURES
# ═════════════════════════════════════════════════════════════════════════════
def compute_haralick_features(g: np.ndarray) -> np.ndarray:
    L = g.shape[0]
    eps = 1e-10

    # ── Index arrays for vectorized computation ───────────────────────────────
    u = np.arange(L).reshape(-1, 1) * np.ones((1, L))   # rows
    v = np.arange(L).reshape(1, -1) * np.ones((L, 1))   # cols

    # ── Marginal probabilities ────────────────────────────────────────────────
    p_x = g.sum(axis=1)
    p_y = g.sum(axis=0)

    # Means and standard deviations of marginal distributions
    idx  = np.arange(L, dtype=np.float64)
    mu_x = np.sum(idx * p_x)
    mu_y = np.sum(idx * p_y)
    sg_x = np.sqrt(np.sum((idx - mu_x)**2 * p_x) + eps)
    sg_y = np.sqrt(np.sum((idx - mu_y)**2 * p_y) + eps)

    # Overall mean of g
    mean_g = np.sum(u * g)

    # ── Sum and difference distributions ─────────────────────────────────────
    p_xy  = np.zeros(2 * L) # sum distribution
    p_xmy = np.zeros(L) # difference distribution

    for i in range(L):
        for j in range(L):
            p_xy[i + j]     += g[i, j]
            p_xmy[abs(i-j)] += g[i, j]

    # ── Entropy terms needed for f12, f13 ────────────────────────────────────
    HX  = -np.sum(p_x[p_x > eps] * np.log(p_x[p_x > eps] + eps))
    HY  = -np.sum(p_y[p_y > eps] * np.log(p_y[p_y > eps] + eps))

    g_nz = g[g > eps]
    HXY = -np.sum(g_nz * np.log(g_nz + eps))

    px_mat = p_x.reshape(-1, 1) * np.ones((1, L))
    py_mat = p_y.reshape(1, -1) * np.ones((L, 1))
    prod = px_mat * py_mat
    HXY1 = -np.sum(g * np.log(prod + eps))

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
    mean_diff = np.sum(k_diff * p_xmy)
    f10 = np.sum((k_diff - mean_diff)**2 * p_xmy)

    # f11: Difference Entropy
    p_xmy_nz = p_xmy[p_xmy > eps]
    f11 = -np.sum(p_xmy_nz * np.log(p_xmy_nz + eps))

    # f12: Information Measure of Correlation 1
    f12 = (HXY - HXY1) / (max(HX, HY) + eps)

    # f13: Information Measure of Correlation 2
    val_f13 = 1.0 - np.exp(-2.0 * (HXY2 - HXY))
    f13 = np.sqrt(max(val_f13, 0.0))

    # f14: Maximal Correlation Coefficient
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

    return features