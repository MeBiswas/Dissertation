# sch_cs/step_4.py

# ── Standard library ──────────────────────────────────────────────────────────
from typing import Dict 

# ── Numeric / image ───────────────────────────────────────────────────────────
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2.4 — Final Threshold
# ─────────────────────────────────────────────────────────────────────────────
def compute_threshold(pb: np.ndarray, tstar_data: Dict) -> Dict:
    """Apply Equation 3 to get final threshold th."""
    print('\n[SCH 2.4] Computing th...')
    t_star = tstar_data['t_star']
    m_p    = float(pb[pb > 0].mean())
    th     = m_p if t_star < m_p else t_star
    reason = ('th = m(p)' if t_star < m_p else 'th = t*')
    print(f'  m(p)={m_p:.2f}, t*={t_star:.2f} → {reason} = {th:.2f}')
    return {'th': th, 'm_p': m_p, 't_star': t_star}