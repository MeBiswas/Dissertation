# src/features_extraction/step_3.py

import numpy as np

# ═════════════════════════════════════════════════════════════════════════════
# PART 2 — 14 HARALICK FEATURES
# ═════════════════════════════════════════════════════════════════════════════
def compute_haralick_features(g: np.ndarray) -> np.ndarray:
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