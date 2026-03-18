# sch_cs/initial_threshold.py

import numpy as np
import matplotlib.pyplot as plt

from src.utils import section

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Compute Initial Threshold t*  (Equation 2)
# ─────────────────────────────────────────────────────────────────────────────
 
def compute_initial_threshold(step1: dict, plot: bool = True) -> dict:
 
    # ── Retrieve from Step 1 ─────────────────────────────────────────────────
    gray_levels   = step1["gray_levels"]
    h             = step1["h"]
    rho           = step1["rho"]
    rho_effective = step1["rho_effective"]
    m_freq        = step1["m"]
 
    section("RECAP FROM STEP 1")
    print(f"  rho (raw from formula)  = {rho:.4f}")
    if rho != rho_effective:
        print(f"  rho_effective (fallback)= {rho_effective:.4f}  "
              f"← used here because raw rho was negative")
    else:
        print(f"  rho_effective           = {rho_effective:.4f}  (same as raw rho)")
    print(f"  m (mean frequency)      = {m_freq:.4f}")
    print(f"  Total available gray levels = {len(gray_levels)}")
 
    # ── 1. Build Array A ──────────────────────────────────────────────────────
    # A = gray level VALUES where h(ni) < rho_effective
    section("VARIABLE A  (small-peak gray levels where h(ni) < rho_effective)")
 
    small_peak_mask  = h < rho_effective
    A_levels         = gray_levels[small_peak_mask]
    A_freqs          = h[small_peak_mask]
 
    print(f"  Rule: select ni where h(ni) < rho_effective ({rho_effective:.4f})")
    print(f"\n  {'Gray Level (ni)':<20}  {'h(ni)':<20}  h(ni) < {rho_effective:.2f}?")
    print(f"  {'-'*20}  {'-'*20}  {'-'*20}")
    for gl, freq in zip(A_levels, A_freqs):
        print(f"  {gl:<20}  {int(freq):<20}  YES")
 
    print(f"\n  A = {A_levels.tolist()}")
    print(f"\n  IMPORTANT: A holds gray level VALUES (e.g. 200, 210 ...),")
    print(f"  NOT their frequencies. Frequencies above are shown only to")
    print(f"  confirm the selection criterion was met.")
 
    # ── 2. j ──────────────────────────────────────────────────────────────────
    section("VARIABLE j  (number of elements in A)")
    j = len(A_levels)
    print(f"  j = {j}")
 
    if j == 0:
        print(f"\n  ERROR: A is still empty even with rho_effective = "
              f"{rho_effective:.4f}.")
        print(f"  This should not happen with the fallback to m.")
        print(f"  Please check your input image and preprocessing output.")
        return {"A": A_levels, "j": 0, "m_A": None,
                "alpha_A": None, "t_star": None}
 
    # ── 3. m(A) ───────────────────────────────────────────────────────────────
    section("VARIABLE m(A)  (mean of gray level VALUES in A)")
    m_A = float(np.mean(A_levels))
    print(f"  m(A) = (1/j) * sum(ni for ni in A)")
    print(f"  m(A) = (1/{j}) * {int(np.sum(A_levels))}")
    print(f"  m(A) = {m_A:.4f}")
    print(f"\n  This is the average INTENSITY VALUE of the SR-candidate")
    print(f"  gray levels — not how often they appear.")
    print(f"  Suspicious regions are centred around intensity ≈ {m_A:.1f}")
 
    # ── 4. alpha(A) ───────────────────────────────────────────────────────────
    section("VARIABLE alpha(A)  (std dev of gray level VALUES in A)")
 
    # Show per-element squared deviations only if A is small enough to print
    if j <= 30:
        print(f"  Per-element squared deviations (ni - m(A))^2:")
        for gl in A_levels:
            dev = gl - m_A
            print(f"    ni={gl:>3d}:  ({gl} - {m_A:.2f})^2 = "
                  f"({dev:.2f})^2 = {dev**2:.4f}")
    else:
        print(f"  (A has {j} elements — skipping per-element printout)")
 
    variance = float(np.var(A_levels))
    alpha_A  = float(np.std(A_levels))
 
    print(f"\n  Variance  = {variance:.4f}")
    print(f"  alpha(A)  = sqrt({variance:.4f}) = {alpha_A:.4f}")
    print(f"\n  The SR-candidate gray levels are spread ±{alpha_A:.1f}")
    print(f"  intensity units around their mean {m_A:.1f}.")
 
    # ── 5. t* ─────────────────────────────────────────────────────────────────
    section("FINAL RESULT: t*  (Initial Threshold)")
    t_star = m_A - alpha_A
    print(f"  t* = m(A) - alpha(A)")
    print(f"  t* = {m_A:.4f} - {alpha_A:.4f}")
    print(f"  t* = {t_star:.4f}")
    print(f"\n  WHY subtract alpha?")
    print(f"  Shifting one std deviation BELOW the mean is conservative.")
    print(f"  It ensures we don't accidentally cut off lower-intensity")
    print(f"  boundary pixels of a suspicious region.")
    print(f"  Step 3 will apply a final safety check on top of this value.")
 
    # ── Summary ───────────────────────────────────────────────────────────────
    section("SUMMARY TABLE  (all Step 2 variables)")
    print(f"  {'Variable':<12}  {'Formula':<40}  {'Value'}")
    print(f"  {'-'*12}  {'-'*40}  {'-'*20}")
    print(f"  {'A':<12}  {'ni where h(ni) < rho_effective':<40}  "
          f"{A_levels.tolist()}")
    print(f"  {'j':<12}  {'len(A)':<40}  {j}")
    print(f"  {'m(A)':<12}  {'(1/j)*sum(ni in A)':<40}  {m_A:.4f}")
    print(f"  {'alpha(A)':<12}  {'std(A)':<40}  {alpha_A:.4f}")
    print(f"  {'t*':<12}  {'m(A) - alpha(A)':<40}  {t_star:.4f}")
 
    # ── Plot ──────────────────────────────────────────────────────────────────
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("SCH-CS Step 2 — Initial Threshold t*",
                     fontsize=13, fontweight='bold')
 
        # Left: full histogram with rho_effective and A highlighted
        ax = axes[0]
        full_hist = step1["histogram"].copy()
        full_hist[0] = 0
        ax.bar(range(256), full_hist, color='steelblue', alpha=0.5,
               width=1.0, label='h(ni)')
        ax.axhline(y=rho_effective, color='red', linewidth=2,
                   linestyle='-.', label=f'rho_effective = {rho_effective:.1f}')
        highlight = np.zeros(256)
        for gl, freq in zip(A_levels, A_freqs):
            highlight[gl] = freq
        ax.bar(range(256), highlight, color='orange', alpha=0.8,
               width=1.0, label=f'Array A ({j} levels)')
        ax.set_xlabel("Gray Level", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title("Full Histogram — orange = array A", fontsize=11)
        ax.legend(fontsize=9)
        ax.set_xlim([0, 255])
 
        # Right: zoom into A, mark m(A), alpha, t*
        ax2 = axes[1]
        ax2.bar(A_levels, A_freqs, color='orange', alpha=0.8,
                width=1.0, label='A gray levels')
        ax2.axvline(x=m_A, color='green', linewidth=2, linestyle='--',
                    label=f'm(A) = {m_A:.2f}')
        ax2.axvline(x=t_star, color='purple', linewidth=2, linestyle='-.',
                    label=f't* = {t_star:.2f}')
        ax2.axvline(x=m_A + alpha_A, color='green', linewidth=1,
                    linestyle=':', alpha=0.5,
                    label=f'm(A)+alpha = {m_A + alpha_A:.2f}')
        ax2.axvspan(t_star, m_A + alpha_A, alpha=0.1, color='green',
                    label='±1 std range')
        ax2.set_xlabel("Gray Level (ni)", fontsize=11)
        ax2.set_ylabel("Frequency", fontsize=11)
        ax2.set_title("Zoomed: Array A with m(A), alpha(A), t*", fontsize=11)
        ax2.legend(fontsize=9)
 
        plt.tight_layout()
        plt.savefig("step2_initial_threshold.png", dpi=150,
                    bbox_inches='tight')
        plt.show()
        print("\n  [Plot saved to step2_initial_threshold.png]")
 
    return {
        "A"       : A_levels,
        "A_freqs" : A_freqs,
        "j"       : j,
        "m_A"     : m_A,
        "alpha_A" : alpha_A,
        "t_star"  : t_star,
    }