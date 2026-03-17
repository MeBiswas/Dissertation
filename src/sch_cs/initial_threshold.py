# sch_cs/initial_threshold.py

import numpy as np
import matplotlib.pyplot as plt

from src.utils import section

# STEP 2 — Compute Initial Threshold t*  (Equation 2)
def compute_initial_threshold(step1: dict, plot: bool = True) -> dict:
    # ── Retrieve what we need from Step 1 ────────────────────────────────────
    gray_levels = step1["gray_levels"]
    h           = step1["h"]
    rho         = step1["rho"]
    m_freq      = step1["m"]

    section("RECAP FROM STEP 1")
    print(f"  rho (count threshold) = {rho:.4f}")
    print(f"  m   (mean frequency)  = {m_freq:.4f}")
    print(f"  Total available gray levels = {len(gray_levels)}")

    # ── 1. Build Array A ──────────────────────────────────────────────────────
    section("VARIABLE A  (small-peak gray levels — SR candidates)")

    small_peak_mask  = h < rho
    A_levels         = gray_levels[small_peak_mask]
    A_freqs          = h[small_peak_mask]

    print(f"  Rule: select gray levels ni where h(ni) < rho ({rho:.4f})")
    print(f"\n  All gray levels where h(ni) < rho:")
    print(f"  {'Gray Level (ni)':<20}  {'Frequency h(ni)':<20}  {'h(ni) < rho?'}")
    print(f"  {'-'*20}  {'-'*20}  {'-'*12}")
    for gl, freq in zip(A_levels, A_freqs):
        print(f"  {gl:<20}  {int(freq):<20}  YES")

    print(f"\n  A = {A_levels.tolist()}")
    print(f"\n  NOTE: A contains the gray level VALUES (e.g. 210, 225 ...),")
    print(f"        NOT their frequencies. The frequencies are shown above")
    print(f"        only for reference to confirm they are all < rho.")

    # ── 2. j — Size of A ─────────────────────────────────────────────────────
    section("VARIABLE j  (number of elements in A)")

    j = len(A_levels)

    print(f"  j = number of gray levels in A")
    print(f"  j = {j}")

    if j == 0:
        print("\n  WARNING: A is empty — no gray levels have h(ni) < rho.")
        print("  This means rho may be too small.")
        print("  Check your Step 1 results and your image's histogram.")
        return {"A": A_levels, "j": 0, "m_A": None, "alpha_A": None,
                "t_star": None}

    # ── 3. m(A) — Mean of gray level VALUES in A ─────────────────────────────
    section("VARIABLE m(A)  (mean of gray level values in A)")

    m_A = float(np.mean(A_levels))

    print(f"  m(A) = (1/j) * sum(ni for ni in A)")
    print(f"  m(A) = (1/{j}) * {int(np.sum(A_levels))}")
    print(f"  m(A) = {m_A:.4f}")
    print(f"\n  This is the AVERAGE of the small-peak GRAY LEVEL VALUES.")
    print(f"  It tells us: 'the suspicious region gray levels are")
    print(f"  centred around intensity {m_A:.1f}'")

    # ── 4. alpha(A) — Std deviation of gray level VALUES in A ────────────────
    section("VARIABLE alpha(A)  (std deviation of gray level values in A)")

    alpha_A = float(np.std(A_levels))

    print(f"  alpha(A) = sqrt( (1/j) * sum( (ni - m(A))^2 ) )")
    print(f"\n  Per-element squared deviations (ni - m(A))^2:")
    for gl in A_levels:
        dev = gl - m_A
        print(f"    ni={gl:>3d}:  ({gl} - {m_A:.2f})^2  =  "
              f"({dev:.2f})^2  =  {dev**2:.4f}")

    variance = float(np.var(A_levels))
    print(f"\n  Mean of squared deviations (variance) = {variance:.4f}")
    print(f"  alpha(A) = sqrt({variance:.4f})")
    print(f"  alpha(A) = {alpha_A:.4f}")
    print(f"\n  Interpretation: the small-peak gray levels are spread")
    print(f"  +/- {alpha_A:.1f} intensity units around their mean {m_A:.1f}.")

    # ── 5. t* — Initial Threshold ─────────────────────────────────────────────
    section("FINAL RESULT: t*  (Initial Threshold)")

    t_star = m_A - alpha_A

    print(f"  t* = m(A) - alpha(A)")
    print(f"  t* = {m_A:.4f} - {alpha_A:.4f}")
    print(f"  t* = {t_star:.4f}")
    print(f"\n  WHY subtract alpha?")
    print(f"  We shift the threshold BELOW the mean of A by one std deviation.")
    print(f"  This is conservative — it ensures we don't accidentally cut off")
    print(f"  the lower-intensity boundary pixels of a suspicious region.")

    # ── Summary Table ─────────────────────────────────────────────────────────
    section("SUMMARY TABLE  (all Step 2 variables at a glance)")
    print(f"  {'Variable':<12}  {'Formula':<40}  {'Value'}")
    print(f"  {'-'*12}  {'-'*40}  {'-'*15}")
    print(f"  {'A':<12}  {'gray levels where h(ni) < rho':<40}  "
          f"{A_levels.tolist()}")
    print(f"  {'j':<12}  {'len(A)':<40}  {j}")
    print(f"  {'m(A)':<12}  {'(1/j)*sum(ni in A)':<40}  {m_A:.4f}")
    print(f"  {'alpha(A)':<12}  {'std(A)':<40}  {alpha_A:.4f}")
    print(f"  {'t*':<12}  {'m(A) - alpha(A)':<40}  {t_star:.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("SCH-CS Step 2 — Initial Threshold t*", fontsize=13,
                     fontweight='bold')

        # Left plot: full histogram with rho line and A region highlighted
        ax = axes[0]
        full_hist = step1["histogram"].copy()
        full_hist[0] = 0
        ax.bar(range(256), full_hist, color='steelblue', alpha=0.5,
               width=1.0, label='h(ni)')
        ax.axhline(y=rho, color='red', linewidth=2, linestyle='-.',
                   label=f'rho = {rho:.2f}')
        # Highlight bars below rho (the A candidates)
        highlight = np.zeros(256)
        for gl, freq in zip(A_levels, A_freqs):
            highlight[gl] = freq
        ax.bar(range(256), highlight, color='orange', alpha=0.8,
               width=1.0, label=f'A: h(ni) < rho  ({j} levels)')
        ax.set_xlabel("Gray Level", fontsize=11)
        ax.set_ylabel("Frequency h(ni)", fontsize=11)
        ax.set_title("Full Histogram — orange bars = array A", fontsize=11)
        ax.legend(fontsize=9)
        ax.set_xlim([0, 255])

        # Right plot: zoom into A gray levels, mark m(A), alpha(A), t*
        ax2 = axes[1]
        if len(A_levels) > 0:
            ax2.bar(A_levels, A_freqs, color='orange', alpha=0.8,
                    width=1.0, label='A gray levels')
            ax2.axvline(x=m_A, color='green', linewidth=2,
                        linestyle='--', label=f'm(A) = {m_A:.2f}')
            ax2.axvline(x=m_A - alpha_A, color='purple', linewidth=2,
                        linestyle='-.',
                        label=f't* = m(A)-alpha(A) = {t_star:.2f}')
            ax2.axvline(x=m_A + alpha_A, color='green', linewidth=1,
                        linestyle=':', alpha=0.5,
                        label=f'm(A)+alpha(A) = {m_A+alpha_A:.2f}')
            # Shade the +/- 1 std region
            ax2.axvspan(m_A - alpha_A, m_A + alpha_A,
                        alpha=0.1, color='green', label='±1 std range')
        ax2.set_xlabel("Gray Level (ni)", fontsize=11)
        ax2.set_ylabel("Frequency h(ni)", fontsize=11)
        ax2.set_title("Zoomed: Array A with m(A), alpha(A), t*", fontsize=11)
        ax2.legend(fontsize=9)

        plt.tight_layout()
        plt.savefig("step2_initial_threshold.png", dpi=150, bbox_inches='tight')
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