# sch_cs/step_3.py

# ── Standard library ──────────────────────────────────────────────────────────
from typing import Dict 

# ── Numeric / image ───────────────────────────────────────────────────────────
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2.3 — t-star
# ─────────────────────────────────────────────────────────────────────────────
def compute_t_star(hist_data: Dict, rho_data: Dict) -> Dict:
    print('\n[SCH 2.3] Computing t*...')
    small_mask = hist_data['peak_freqs'] < rho_data['rho_effective']
    A = hist_data['peak_indices'][small_mask].astype(float)
    A_freqs = hist_data['peak_freqs'][small_mask]

    if len(A) == 0:
        raise ValueError('Array A is empty — check preprocessing output.')

    m_A, alpha_A = float(np.mean(A)), float(np.std(A))
    t_star = m_A - alpha_A
    print(f'  |A|={len(A)}, A range=[{int(A.min())},{int(A.max())}], '
          f'm(A)={m_A:.2f}, alpha(A)={alpha_A:.2f}, t*={t_star:.2f}')
    return {'A': A, 'A_freqs': A_freqs, 'm_A': m_A, 'alpha_A': alpha_A, 't_star': t_star}