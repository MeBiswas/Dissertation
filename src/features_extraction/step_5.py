# src/features_extraction/step_5.py

# ═════════════════════════════════════════════════════════════════════════════
# COMBINED: Full 21-element Feature Vector for one breast
# ═════════════════════════════════════════════════════════════════════════════
import numpy as np
from .step_2 import compute_glcm_normalized
from .step_3 import compute_haralick_features
from .step_4 import compute_hu_moments

def compute_feature_vector(sr_region):

    g = compute_glcm_normalized(sr_region)
    haralick = compute_haralick_features(g)
    hu = compute_hu_moments(sr_region)

    f_v = np.concatenate([haralick, hu])

    return f_v, g, haralick, hu