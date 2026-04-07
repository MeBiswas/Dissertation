# sch_cs/step_6.py

# ── Standard library ──────────────────────────────────────────────────────────
from typing import Dict, List

# ── Utils Import ───────────────────────────────────────────────────────────
from src.utils import SCH_CFG, SchCsConfig

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2.6  ← NEW — Column-edge guard
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY THIS IS NEEDED:
#   After thresholding, regions near the left and right edges of the image
#   are virtually always FLIR artefacts:
#     - Left edge  : residual HUD text, crosshair tick marks
#     - Right edge : colour bar bleed, calibration square
#
#   Genuine breast SRs are always located in the CENTRAL portion of the
#   image because the crop step already removed the armpit columns.
#   Any region whose centroid column is within edge_col_pct of either edge
#   can be safely discarded BEFORE running the expensive CS isolation loop.
#
# The paper does not state this explicitly but it is implied by the dataset
# structure and confirmed by inspection of Figure 3(c).
# ─────────────────────────────────────────────────────────────────────────────

def column_edge_filter(
    regions    : List[Dict],
    image_width: int,
    cfg        : SchCsConfig = SCH_CFG
) -> List[Dict]:
    """
    Discard regions whose centroid column is within edge_col_pct of either edge.

    Parameters
    ----------
    regions     : list from threshold_and_label() — must already have 'centroid'
    image_width : int  — width of p_b

    Returns
    -------
    Filtered list of region dicts.
    """
    print(f'\n[SCH 2.6] Column-edge guard (edge_col_pct={cfg.edge_col_pct})...')
    margin  = int(image_width * cfg.edge_col_pct)
    col_min = margin
    col_max = image_width - margin

    kept     = []
    rejected = []
    for reg in regions:
        _, Y = reg['centroid']
        if col_min <= Y <= col_max:
            kept.append(reg)
        else:
            rejected.append(reg['label'])

    print(f'  Kept {len(kept)} / {len(regions)} regions '
          f'(rejected labels near edges: {rejected})')
    return kept