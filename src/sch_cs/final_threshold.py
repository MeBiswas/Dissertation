# sch_cs/final_threshold.py

import numpy as np

from src.utils import section

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Final Threshold th  (Equation 3)
# ─────────────────────────────────────────────────────────────────────────────
 
"""
    Computes the final threshold th from Equation (3).

    Args:
        pb    : Background-removed grayscale image (p_b).
        step2 : Dict from compute_initial_threshold() — needs 't_star'.

    Returns:
        dict with 'm_p', 't_star', 'th', 'th_reason'.
"""
def compute_final_threshold(pb: np.ndarray, step2: dict) -> dict:
    t_star = step2["t_star"]
 
    section("STEP 3 — Final Threshold th  (Equation 3)")
 
    # ── m(p) — mean pixel intensity of pb, excluding background zeros
    nonzero_pixels = pb[pb > 0]
    m_p = float(np.mean(nonzero_pixels))
 
    print(f"  m(p) = mean intensity of all non-zero pixels in p_b")
    print(f"  m(p) = {m_p:.4f}")
    print(f"\n  t*   = {t_star:.4f}  (carried from Step 2)")
 
    # ── Apply Equation 3 ─────────────────────────────────────────────────
    if t_star < m_p:
        th = m_p
        reason = f"t* ({t_star:.4f}) < m(p) ({m_p:.4f})  →  th = m(p)"
    else:
        th = t_star
        reason = f"t* ({t_star:.4f}) >= m(p) ({m_p:.4f})  →  th = t*"
 
    print(f"\n  Equation 3 decision:")
    print(f"  {reason}")
    print(f"\n  th = {th:.4f}")
    print(f"\n  Pixels with intensity > {th:.1f} → classified as SR")
    print(f"  Pixels with intensity <= {th:.1f} → classified as TBR/background")
 
    section("STEP 3 SUMMARY")
    
    print(f"  {'Variable':<12}  {'Formula':<35}  {'Value'}")
    print(f"  {'-'*12}  {'-'*35}  {'-'*15}")
    print(f"  {'m(p)':<12}  {'mean of non-zero pixels in pb':<35}  {m_p:.4f}")
    print(f"  {'t*':<12}  {'from Step 2':<35}  {t_star:.4f}")
    print(f"  {'th':<12}  {'max(t*, m(p))':<35}  {th:.4f}")
 
    return {"m_p": m_p, "t_star": t_star, "th": th, "th_reason": reason}

# def compute_final_threshold(pb, step2):
#     nonzero = pb[pb > 0]
#     m_p     = float(np.mean(nonzero))

#     section("STEP 3 — Final Threshold th")
#     print(f"  m(p) = {m_p:.4f}")
#     print(f"  t*   = {step2['t_star']:.4f}")

#     if m_p > 200:
#         th     = float(np.percentile(nonzero, 85))
#         reason = (f"m(p)={m_p:.2f} > 200 → bimodal histogram detected. "
#                   f"Fallback: th = 85th percentile = {th:.2f}")
#     elif step2["t_star"] < m_p:
#         th     = m_p
#         reason = f"t* < m(p) → th = m(p) = {m_p:.2f}"
#     else:
#         th     = step2["t_star"]
#         reason = f"t* >= m(p) → th = t* = {th:.2f}"

#     print(f"\n  Decision: {reason}")
#     print(f"  th = {th:.4f}")

#     return {"m_p": m_p, "t_star": step2["t_star"],
#             "th": th, "th_reason": reason}