# sch_cs/connected_regions.py

import numpy as np

from src.utils import section
from scipy.ndimage import label

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4A — Threshold and find connected regions
# ─────────────────────────────────────────────────────────────────────────────
 
"""
    Applies threshold th to pb and finds all connected regions.

    Returns:
        binary_image  : Binary image (1 = SR candidate, 0 = TBR/background).
        labeled_image : Each connected region has a unique integer label.
        regions       : List of dicts per region with keys:
                        'label', 'mask', 'coords', 'pixel_vals', 'size'
"""
def apply_threshold_and_find_regions(pb: np.ndarray, th: float) -> tuple:
    section("STEP 4A — Apply Threshold and Find Connected Regions")
 
    binary_image = (pb > th).astype(np.uint8)
    total_sr_pixels = int(np.sum(binary_image))
 
    print(f"  Threshold applied: th = {th:.4f}")
    print(f"  Pixels > th (SR candidates) : {total_sr_pixels}")
    print(f"  Pixels <= th (TBR/BG)       : {pb.size - total_sr_pixels}")
 
    # 8-connectivity: diagonal neighbours count as connected
    struct = np.ones((3, 3), dtype=int)
    labeled_image, num_regions = label(binary_image, structure=struct)
 
    print(f"\n  Connected regions found: {num_regions}")
    print(f"\n  {'Region':<10}  {'Size (px)':<12}  {'% of SR pixels'}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*15}")
 
    regions = []
    for k in range(1, num_regions + 1):
        mask       = (labeled_image == k)
        coords     = np.argwhere(mask)
        pixel_vals = pb[mask]
        size       = int(np.sum(mask))
        pct        = 100.0 * size / total_sr_pixels if total_sr_pixels > 0 else 0
 
        regions.append({
            "label"     : k,
            "mask"      : mask,
            "coords"    : coords,
            "pixel_vals": pixel_vals,
            "size"      : size,
        })
        print(f"  R({k:<7})  {size:<12}  {pct:.2f}%")
 
    return binary_image, labeled_image, regions