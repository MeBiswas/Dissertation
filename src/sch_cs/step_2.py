# sch_cs/step_2.py

# ── Standard library ──────────────────────────────────────────────────────────
from typing import Dict 

# ── Numeric / image ───────────────────────────────────────────────────────────
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2.2 — Initial Threshold
# ─────────────────────────────────────────────────────────────────────────────
def compute_rho(hist_data: Dict) -> Dict:
    print('\n[SCH 2.2] Computing rho...')
    peak_freqs = hist_data['peak_freqs']
    peak_idx = hist_data['peak_indices']
    N, m = hist_data['N'], hist_data['m']

    tall_mask = peak_freqs > m
    r = int(tall_mask.sum())
    if r == 0:
        r = len(peak_freqs)
        tall_mask = np.ones(len(peak_freqs), dtype=bool)

    V = float(peak_freqs[tall_mask].sum()) / r
    C = N / V
    max_h = float(peak_freqs.max())
    max_lv = int(peak_idx[peak_freqs.argmax()])
    rho = max_h / C - m

    rho_eff = m if rho <= 0 else rho
    status = 'fallback to m' if rho <= 0 else 'positive'
    print(f'  r={r}, V={V:.1f}, C={C:.1f}, max_h={int(max_h)} at level {max_lv}')
    print(f'  rho={rho:.2f} ({status}) → rho_eff={rho_eff:.2f}')

    return {
        'rho': rho,
        'rho_effective': rho_eff,
        'V': V,
        'C': C,
        'r': r,
        'max_h': max_h,
        'max_h_level': max_lv
    }