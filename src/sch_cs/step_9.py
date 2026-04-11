# sch_cs/step_9.py

# ── Standard library ──────────────────────────────────────────────────────────
from typing import Dict, List 

# ── Numeric / image ───────────────────────────────────────────────────────────
import numpy as np
from scipy.ndimage import label

# ── Utils Import ───────────────────────────────────────────────────────────
from src.utils import SCH_CFG, SchCsConfig

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2.9 — CS isolation  [Algorithm from paper]
# ─────────────────────────────────────────────────────────────────────────────
#
# Identical to v1 (already correct).  Reproduced here for completeness.
# ─────────────────────────────────────────────────────────────────────────────

def cs_isolation(
    regions : List[Dict],
    cfg     : SchCsConfig = SCH_CFG
) -> List[Dict]:
    print(f'\n[CS 2.9] CS isolation (epsilon={cfg.epsilon})...')
    print(f'  Starting with {len(regions)} regions.')

    if len(regions) <= 1:
        print('  One or zero regions — isolation skipped.')
        return regions

    active = regions.copy()
    rnd    = 0

    while len(active) > 1:
        rnd    += 1
        X_vals  = [r['centroid'][0] for r in active]
        C_avg   = float(np.mean(X_vals))

        if rnd % 2 == 1:
            surviving = [r for r in active if r['centroid'][0] >= C_avg]
            direction = 'X < C_avg eliminated'
        else:
            surviving = [r for r in active if r['centroid'][0] <= C_avg]
            direction = 'X > C_avg eliminated'

        if not surviving:
            print(f'  Round {rnd}: all eliminated — reverting.')
            break

        new_X   = [r['centroid'][0] for r in surviving]
        new_avg = float(np.mean(new_X))
        max_dev = max(abs(x - new_avg) for x in new_X)
        active  = surviving

        print(f'  Round {rnd}: C_avg={C_avg:.1f}, {direction}. '
              f'Survivors={len(active)}, max_dev={max_dev:.2f}')

        if max_dev <= cfg.epsilon:
            print(f'  Converged (max_dev={max_dev:.2f} <= eps={cfg.epsilon}).')
            break

    print(f'  Final SRs: {len(active)} — labels {[r["label"] for r in active]}')
    return active