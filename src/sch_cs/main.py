# sch_cs/main.py

# ── Standard library ──────────────────────────────────────────────────────────
from typing import Dict

# ── Numeric ────────────────────────────────────────────────────────────────
import numpy as np

# ── Utils Import ───────────────────────────────────────────────────────────
from src.utils import SCH_CFG, SchCsConfig

# ── SCH-CS Import ───────────────────────────────────────────────────────────
from .step_1 import compute_histogram
from .step_2 import compute_rho
from .step_3 import compute_t_star
from .step_4 import compute_threshold
from .step_5 import threshold_and_label
from .step_6 import column_edge_filter
from .step_7 import compute_centroids
from .step_8 import bounding_box_correction
from .step_9 import cs_isolation
from .step_10 import visualize_schcs

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ENTRY — run_schcs()
# ─────────────────────────────────────────────────────────────────────────────

def run_schcs(
    pb         : np.ndarray,
    cfg        : SchCsConfig = SCH_CFG,
    visualize  : bool = True,
    image_name : str  = ''
) -> Dict:
    """
    Full SCH-CS pipeline.

    Returns dict with keys:
        'sr_regions'    — final SRs ready for DLPE LSM initialisation
        'all_regions'   — all regions after edge filter (before CS isolation)
        'binary_image'  — thresholded binary image
        'labeled_image' — label map
        'th'            — threshold used
        (+ intermediate dicts from each step)
    """
    print('\n' + '=' * 60)
    print(f'  SCH-CS: {image_name}')
    print('=' * 60)

    # ── SCH: compute threshold ────────────────────────────────────────────────
    hist_data  = compute_histogram(pb)
    rho_data   = compute_rho(hist_data)
    tstar_data = compute_t_star(hist_data, rho_data)
    th_data    = compute_threshold(pb, tstar_data)

    # ── Segment + initial filter ──────────────────────────────────────────────
    label_data = threshold_and_label(pb, th_data['th'], cfg)

    if not label_data['regions']:
        print('\n[Warning] No regions after thresholding.')
        return {'sr_regions': [], 'all_regions': [],
                'binary_image': label_data['binary_image'],
                'labeled_image': label_data['labeled_image'],
                'th': th_data['th']}

    # ── CS: isolate genuine SRs ───────────────────────────────────────────────
    regions = compute_centroids(pb, label_data['regions'])

    # Step 2.6: remove edge-column artefacts BEFORE bounding-box correction
    regions = column_edge_filter(regions, pb.shape[1], cfg)

    if not regions:
        print('\n[Warning] All regions removed by edge filter.')
        return {'sr_regions': [], 'all_regions': [],
                'binary_image': label_data['binary_image'],
                'labeled_image': label_data['labeled_image'],
                'th': th_data['th']}

    regions    = bounding_box_correction(regions, label_data['labeled_image'])
    sr_regions = cs_isolation(regions, cfg)

    if visualize:
        visualize_schcs(pb, hist_data, rho_data, th_data,
                        label_data, sr_regions, regions, image_name)

    print(f'\n[Done] {len(sr_regions)} SR(s) ready for DLPE Level Set.')

    return {
        'sr_regions'   : sr_regions,
        'all_regions'  : regions,
        'binary_image' : label_data['binary_image'],
        'labeled_image': label_data['labeled_image'],
        'th'           : th_data['th'],
        'hist_data'    : hist_data,
        'rho_data'     : rho_data,
        'tstar_data'   : tstar_data,
        'th_data'      : th_data,
    }