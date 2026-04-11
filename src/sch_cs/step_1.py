# sch_cs/step_1.py

# ── Standard library ──────────────────────────────────────────────────────────
from typing import Dict

# ── Numeric / image ───────────────────────────────────────────────────────────
import numpy as np
from scipy.signal import find_peaks

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2.1 — Histogram Peak Calculation
# ─────────────────────────────────────────────────────────────────────────────
def compute_histogram(pb: np.ndarray) -> Dict:
    print('\n[SCH 2.1] Building histogram...')
    full_hist = np.zeros(256, dtype=np.float64)
    flat = pb[pb > 0].ravel()
    for v in flat:
        full_hist[int(v)] += 1

    N = int(full_hist.sum())
    R = int((full_hist > 0).sum())
    m = N / R

    peak_indices, _ = find_peaks(full_hist, height=1)
    peak_freqs      = full_hist[peak_indices]

    print(f'  N={N}, R={R}, m=N/R={m:.2f}, peaks found={len(peak_indices)}')
    return {'full_hist': full_hist, 'peak_indices': peak_indices, 'peak_freqs': peak_freqs, 'R': R, 'N': N, 'm': m}