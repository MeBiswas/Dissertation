# sch_cs/count_threshold.py

import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import section

# STEP 1 — Compute Count Threshold rho  (Equation 1)
def compute_count_threshold(pb: np.ndarray, plot_histogram: bool = True) -> dict:
    section("HISTOGRAM COMPUTATION")

    full_hist = cv2.calcHist([pb], [0], None, [256], [0, 256]).flatten()

    full_hist[0] = 0

    gray_levels = np.where(full_hist > 0)[0]

    h = full_hist[gray_levels]

    print(f"  Total histogram bins (0-255)     : 256")
    print(f"  Bins with zero frequency (empty) : {np.sum(full_hist == 0)}")
    print(f"  Non-empty bins (available levels): {len(gray_levels)}")
    print(f"\n  First 10 gray levels present : {gray_levels[:10]}")
    print(f"  Last  10 gray levels present : {gray_levels[-10:]}")
    print(f"\n  Histogram frequencies for those levels:")
    print(f"  (gray_level : frequency)")
    for gl, freq in zip(gray_levels[:15], h[:15]):
        print(f"    {gl:>3d} : {int(freq)}")
    if len(gray_levels) > 15:
        print(f"    ... ({len(gray_levels) - 15} more levels not shown)")

    # ── 1. R — Number of available gray levels ────────────────────────────────
    section("VARIABLE R  (number of available gray levels)")

    R = len(gray_levels)

    print(f"  R = number of distinct gray levels present in p_b")
    print(f"  R = {R}")
    print(f"\n  NOTE: 'Available' means gray levels that actually appear")
    print(f"        in the image (non-zero histogram bins, excluding 0).")

    # ── 2. N — Total pixel count ──────────────────────────────────────────────
    section("VARIABLE N  (total pixel count)")

    N = int(np.sum(h))

    print(f"  N = sum of all h(ni)  for i = 0 to L-1")
    print(f"  N = {N}")
    print(f"\n  Cross-check: non-zero pixels in image = "
          f"{int(np.count_nonzero(pb))}")
    print(f"  (should match N if background bin is correctly excluded)")

    # ── 3. m — Mean frequency ─────────────────────────────────────────────────
    section("VARIABLE m  (mean frequency)")

    m = N / R

    print(f"  m = (1/R) * sum(h(ni))  =  N / R")
    print(f"  m = {N} / {R}")
    print(f"  m = {m:.4f}")
    print(f"\n  Interpretation: on average, each gray level appears")
    print(f"  {m:.1f} times in the image.")
    print(f"\n  Peaks TALLER  than m ({m:.2f}) are considered 'dominant' peaks.")
    print(f"  Peaks SHORTER than m ({m:.2f}) are 'small' peaks -> SR candidates.")

    # ── 4. r — Number of tall peaks (peaks taller than m) ────────────────────
    section("VARIABLE r  (number of tall peaks where h(ni) > m)")

    tall_mask       = h > m
    tall_peak_levels = gray_levels[tall_mask]
    tall_peak_freqs  = h[tall_mask]

    r = int(np.sum(tall_mask))

    print(f"  r = count of bins where h(ni) > m ({m:.4f})")
    print(f"  r = {r}")
    print(f"\n  These {r} tall peaks correspond to gray levels:")
    for gl, freq in zip(tall_peak_levels, tall_peak_freqs):
        print(f"    gray level {gl:>3d}  ->  frequency {int(freq):>8d}"
              f"  {'<-- MAX' if int(freq) == int(np.max(tall_peak_freqs)) else ''}")

    # ── 5. V — Average height of tall peaks ───────────────────────────────────
    section("VARIABLE V  (average height of tall peaks)")

    V = float(np.sum(tall_peak_freqs)) / r

    print(f"  V = sum(h(ni) for h(ni) > m)  /  r")
    print(f"  V = {int(np.sum(tall_peak_freqs))} / {r}")
    print(f"  V = {V:.4f}")
    print(f"\n  Interpretation: the average frequency of the dominant")
    print(f"  (tall) histogram peaks = {V:.1f} pixels per gray level.")

    # ── 6. C — Normalising constant ───────────────────────────────────────────
    section("VARIABLE C  (normalising constant)")

    C = N / V

    print(f"  C = N / V")
    print(f"  C = {N} / {V:.4f}")
    print(f"  C = {C:.4f}")
    print(f"\n  Interpretation: C scales the maximum peak height down to")
    print(f"  a 'typical large-peak' unit, so rho is meaningful regardless")
    print(f"  of image size or overall brightness.")

    # ── 7. max_h — Maximum histogram frequency ────────────────────────────────
    section("VARIABLE max_h  (maximum histogram frequency)")

    max_h      = float(np.max(h))
    max_h_level = int(gray_levels[np.argmax(h)])

    print(f"  max_h = max[h(n0), h(n1), ..., h(nL-1)]")
    print(f"  max_h = {int(max_h)}  (at gray level {max_h_level})")

    # ── 8. rho — Count Threshold ──────────────────────────────────────────────
    section("FINAL RESULT: rho  (Count Threshold)")

    rho = max_h / C - m

    print(f"  rho = max_h / C  -  m")
    print(f"  rho = {int(max_h)} / {C:.4f}  -  {m:.4f}")
    print(f"  rho = {max_h / C:.4f}  -  {m:.4f}")
    print(f"  rho = {rho:.4f}")
    print(f"\n  Interpretation:")
    print(f"  Any histogram peak with frequency < {rho:.2f} is considered a")
    print(f"  'small peak' -> its gray levels are SR candidates.")
    print(f"  Any peak with frequency >= {rho:.2f} is a dominant tissue peak.")

    # ── Summary Table ─────────────────────────────────────────────────────────
    section("SUMMARY TABLE  (all variables at a glance)")
    print(f"  {'Variable':<10}  {'Formula':<35}  {'Value'}")
    print(f"  {'-'*10}  {'-'*35}  {'-'*15}")
    print(f"  {'R':<10}  {'distinct gray levels in p_b':<35}  {R}")
    print(f"  {'N':<10}  {'sum of all h(ni)':<35}  {N}")
    print(f"  {'m':<10}  {'N / R':<35}  {m:.4f}")
    print(f"  {'r':<10}  {'count of bins where h(ni) > m':<35}  {r}")
    print(f"  {'V':<10}  {'sum(tall peaks) / r':<35}  {V:.4f}")
    print(f"  {'C':<10}  {'N / V':<35}  {C:.4f}")
    print(f"  {'max_h':<10}  {'max(h)':<35}  {int(max_h)}  (at level {max_h_level})")
    print(f"  {'rho':<10}  {'max_h / C  -  m':<35}  {rho:.4f}")

    # ── Histogram Plot ────────────────────────────────────────────────────────
    if plot_histogram:
        fig, ax = plt.subplots(figsize=(12, 5))

        # Plot full histogram (excluding bin 0)
        plot_hist = full_hist.copy()
        plot_hist[0] = 0
        ax.bar(range(256), plot_hist, color='steelblue', alpha=0.6,
               width=1.0, label='h(ni) — histogram')

        # Mark the mean frequency line m
        ax.axhline(y=m, color='green', linewidth=2,
                   linestyle='--', label=f'm = {m:.1f}  (mean frequency)')

        # Mark the rho line
        ax.axhline(y=rho, color='red', linewidth=2,
                   linestyle='-.',
                   label=f'rho = {rho:.1f}  (count threshold)')

        # Shade the "small peaks" region (below rho)
        ax.fill_between(range(256), 0, rho,
                        color='orange', alpha=0.15,
                        label='Small peaks zone (SR candidates)')

        ax.set_xlabel("Gray Level", fontsize=12)
        ax.set_ylabel("Frequency h(ni)", fontsize=12)
        ax.set_title("Histogram of p_b with SCH-CS Step 1 annotations\n"
                     f"R={R}, N={N}, m={m:.1f}, r={r}, "
                     f"V={V:.1f}, C={C:.1f}, rho={rho:.2f}",
                     fontsize=11)
        ax.legend(fontsize=10)
        ax.set_xlim([0, 255])
        plt.tight_layout()
        plt.savefig("step1_histogram.png", dpi=150, bbox_inches='tight')
        plt.show()
        print("\n  [Plot saved to step1_histogram.png]")

    # Return all variables for use in next steps
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
    }