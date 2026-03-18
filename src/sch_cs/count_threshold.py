# sch_cs/count_threshold.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

from src.utils import section

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Compute Count Threshold rho  (Equation 1)
# ─────────────────────────────────────────────────────────────────────────────
 
def compute_count_threshold(pb: np.ndarray, plot_histogram: bool = True) -> dict:
 
    # ── 0. Compute histogram (exclude background bin 0) ──────────────────────
    section("HISTOGRAM COMPUTATION")
 
    full_hist = cv2.calcHist([pb], [0], None, [256], [0, 256]).flatten()
    full_hist[0] = 0 # exclude background pixels
 
    gray_levels = np.where(full_hist > 0)[0]
    h = full_hist[gray_levels]
 
    print(f"  Total histogram bins (0-255)      : 256")
    print(f"  Non-empty bins (excluding bin 0)  : {len(gray_levels)}")
    print(f"\n  First 10 gray levels present : {gray_levels[:10].tolist()}")
    print(f"  Last  10 gray levels present : {gray_levels[-10:].tolist()}")
    print(f"\n  Sample frequencies (first 15 non-empty bins):")
    print(f"  {'Gray Level':>12}  {'Frequency':>12}")
    print(f"  {'-'*12}  {'-'*12}")
    
    for gl, freq in zip(gray_levels[:15], h[:15]):
        print(f"  {gl:>12}  {int(freq):>12}")
        
    if len(gray_levels) > 15:
        print(f"  ... ({len(gray_levels) - 15} more levels not shown)")
 
    # ── 1. R ──────────────────────────────────────────────────────────────────
    section("VARIABLE R  (number of available gray levels)")
    R = len(gray_levels)
    print(f"  R = {R}")
 
    # ── 2. N ──────────────────────────────────────────────────────────────────
    section("VARIABLE N  (total pixel count)")
    N = int(np.sum(h))
    print(f"  N = sum of all h(ni) = {N}")
    print(f"  Cross-check — non-zero pixels in image: "
          f"{int(np.count_nonzero(pb))}")
 
    # ── 3. m ──────────────────────────────────────────────────────────────────
    section("VARIABLE m  (mean frequency  =  N / R)")
    m = N / R
    print(f"  m = {N} / {R} = {m:.4f}")
    print(f"\n  Peaks TALLER  than m ({m:.2f}) → dominant peaks")
    print(f"  Peaks SHORTER than m ({m:.2f}) → small peaks (SR candidates)")
 
    # ── 4. r ──────────────────────────────────────────────────────────────────
    section("VARIABLE r  (number of tall peaks where h(ni) > m)")
    tall_mask = h > m
    tall_peak_levels = gray_levels[tall_mask]
    tall_peak_freqs = h[tall_mask]
    r = int(np.sum(tall_mask))
    print(f"  r = {r}")
    print(f"\n  Tall peaks:")
    for gl, freq in zip(tall_peak_levels, tall_peak_freqs):
        marker = "  <-- MAX" if int(freq) == int(np.max(tall_peak_freqs)) else ""
        print(f"    gray level {gl:>3d}  →  frequency {int(freq):>8d}{marker}")
 
    # ── 5. V ──────────────────────────────────────────────────────────────────
    section("VARIABLE V  (average height of tall peaks)")
    V = float(np.sum(tall_peak_freqs)) / r
    print(f"  V = {int(np.sum(tall_peak_freqs))} / {r} = {V:.4f}")
 
    # ── 6. C ──────────────────────────────────────────────────────────────────
    section("VARIABLE C  (normalising constant  =  N / V)")
    C = N / V
    print(f"  C = {N} / {V:.4f} = {C:.4f}")
 
    # ── 7. max_h ──────────────────────────────────────────────────────────────
    section("VARIABLE max_h  (maximum histogram frequency)")
    max_h = float(np.max(h))
    max_h_level = int(gray_levels[np.argmax(h)])
    print(f"  max_h = {int(max_h)}  (at gray level {max_h_level})")
 
    # ── 8. rho ────────────────────────────────────────────────────────────────
    section("FINAL RESULT: rho  (Count Threshold)")
    rho = max_h / C - m
    print(f"  rho = max_h / C  -  m")
    print(f"  rho = {int(max_h)} / {C:.4f}  -  {m:.4f}")
    print(f"  rho = {max_h / C:.4f}  -  {m:.4f}")
    print(f"  rho = {rho:.4f}")
 
    # ── 9. NEGATIVE rho HANDLING ──────────────────────────────────────────────
    section("NEGATIVE rho CHECK")
    if rho <= 0:
        rho_effective = m      # fall back to mean frequency
        print(f"  ⚠  rho = {rho:.4f}  →  NEGATIVE (or zero)")
        print(f"\n  WHY THIS HAPPENS:")
        print(f"  rho = max_h * V / N  -  N/R")
        print(f"  When the histogram is heavily concentrated (most pixels")
        print(f"  cluster around a few gray levels), the dominant peaks are")
        print(f"  so large that subtracting m pulls rho below zero.")
        print(f"  The paper's formula assumes a multimodal histogram with")
        print(f"  clearly separated small and large peaks (DMR-IR database).")
        print(f"\n  FIX APPLIED:")
        print(f"  The paper's stated intent is to find gray levels that")
        print(f"  appear RARELY (below average frequency).")
        print(f"  → Falling back to:  rho_effective = m = {m:.4f}")
        print(f"  This directly captures the paper's intent:")
        print(f"  'select gray levels whose frequency is below average'")
    else:
        rho_effective = rho
        print(f"  ✓  rho = {rho:.4f}  →  POSITIVE, no fallback needed.")
        print(f"  rho_effective = rho = {rho:.4f}")
 
    print(f"\n  rho_effective (used for selecting array A in Step 2)")
    print(f"  = {rho_effective:.4f}")
 
    # ── Summary ───────────────────────────────────────────────────────────────
    section("SUMMARY TABLE")
    print(f"  {'Variable':<16}  {'Formula':<35}  {'Value'}")
    print(f"  {'-'*16}  {'-'*35}  {'-'*15}")
    print(f"  {'R':<16}  {'distinct gray levels in p_b':<35}  {R}")
    print(f"  {'N':<16}  {'sum of all h(ni)':<35}  {N}")
    print(f"  {'m':<16}  {'N / R':<35}  {m:.4f}")
    print(f"  {'r':<16}  {'count where h(ni) > m':<35}  {r}")
    print(f"  {'V':<16}  {'sum(tall peaks) / r':<35}  {V:.4f}")
    print(f"  {'C':<16}  {'N / V':<35}  {C:.4f}")
    print(f"  {'max_h':<16}  {'max(h)':<35}  {int(max_h)}  (level {max_h_level})")
    print(f"  {'rho (raw)':<16}  {'max_h / C  -  m':<35}  {rho:.4f}")
    print(f"  {'rho_effective':<16}  {'rho if rho>0, else m':<35}  "
          f"{rho_effective:.4f}  ← used in Step 2")
 
    # ── Plot ──────────────────────────────────────────────────────────────────
    if plot_histogram:
        fig, ax = plt.subplots(figsize=(12, 5))
        plot_hist = full_hist.copy()
        ax.bar(range(256), plot_hist, color='steelblue', alpha=0.6,
               width=1.0, label='h(ni)')
        ax.axhline(y=m, color='green', linewidth=2, linestyle='--',
                   label=f'm = {m:.1f}  (mean freq)')
        if rho > 0:
            ax.axhline(y=rho, color='red', linewidth=2, linestyle='-.',
                       label=f'rho = {rho:.1f}')
        else:
            ax.axhline(y=rho_effective, color='orange', linewidth=2,
                       linestyle='-.',
                       label=f'rho_effective = m = {rho_effective:.1f}'
                             f'  (fallback, raw rho={rho:.1f})')
        ax.fill_between(range(256), 0, rho_effective, color='orange',
                        alpha=0.12, label='Small peaks zone (SR candidates)')
        ax.set_xlabel("Gray Level", fontsize=12)
        ax.set_ylabel("Frequency h(ni)", fontsize=12)
        ax.set_title(
            f"Histogram of p_b — Step 1 annotations\n"
            f"R={R}, N={N}, m={m:.1f}, r={r}, V={V:.1f}, "
            f"C={C:.1f}, rho={rho:.2f}, rho_eff={rho_effective:.2f}",
            fontsize=11)
        ax.legend(fontsize=10)
        ax.set_xlim([0, 255])
        plt.tight_layout()
        plt.savefig("step1_histogram.png", dpi=150, bbox_inches='tight')
        plt.show()
        print("\n  [Plot saved to step1_histogram.png]")
 
    return {
        "histogram"        : full_hist,
        "gray_levels"      : gray_levels,
        "h"                : h,
        "R"                : R,
        "N"                : N,
        "m"                : m,
        "tall_peak_levels" : tall_peak_levels,
        "tall_peak_freqs"  : tall_peak_freqs,
        "r"                : r,
        "V"                : V,
        "C"                : C,
        "max_h"            : max_h,
        "max_h_level"      : max_h_level,
        "rho"              : rho,
        "rho_effective"    : rho_effective,
    }