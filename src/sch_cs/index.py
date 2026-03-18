# sch_cs/index.py
import numpy as np

from src.utils import section
from .cs_isolation import cs_isolation
from .visualization import visualize_results
from .centroid_computation import compute_centroids
from .final_threshold import compute_final_threshold
from .bounding_box import apply_bounding_box_correction
from .connected_regions import apply_threshold_and_find_regions

# ─────────────────────────────────────────────────────────────────────────────
# MASTER FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
 
"""
    Runs Steps 3 and 4 of SCH-CS in sequence and returns all results.

    Args:
        pb        : Background-removed grayscale image (p_b).
        step2     : Dict from compute_initial_threshold() (Step 2).
        epsilon   : CS convergence threshold — paper uses 35.
        plot      : Whether to produce the 4-panel visual output.
        save_path : Where to save the figure.

    Returns:
        dict containing all outputs. Key field is 'sr_regions' — these
        are the approximate SR contours used to initialise the LSM.
"""
def run_steps_3_and_4(
    pb: np.ndarray,
    step2: dict,
    epsilon: float = 35.0,
    plot: bool = True,
    save_path: str = "schcs_step3_step4_result.png"
) -> dict:
    # Step 3
    step3 = compute_final_threshold(pb, step2)
    th    = step3["th"]
 
    # Step 4A
    binary_image, labeled_image, regions = \
        apply_threshold_and_find_regions(pb, th)
 
    if len(regions) == 0:
        print(f"\n  [Warning] No regions found. th={th:.2f} may be too high.")
        return {"th": th, "sr_regions": [], "all_regions": [],
                "binary_image": binary_image, **step3}
 
    # Step 4B
    regions = compute_centroids(pb, regions)
 
    # Step 4C
    regions = apply_bounding_box_correction(regions, labeled_image)
 
    # Step 4D
    sr_regions = cs_isolation(regions, epsilon=epsilon)
 
    # Visual
    if plot:
        visualize_results(pb, binary_image, regions,
                          sr_regions, th, save_path)
 
    # Final summary
    section("COMPLETE SCH-CS SUMMARY  (Steps 1 → 4)")
    print(f"  Step 1  rho_effective    = "
          f"{step2.get('rho_effective', 'N/A')}")
    print(f"  Step 2  t*               = {step2['t_star']:.4f}")
    print(f"  Step 3  th               = {th:.4f}")
    print(f"           reason          : {step3['th_reason']}")
    print(f"  Step 4  Total regions    = {len(regions)}")
    print(f"          SRs after CS     = {len(sr_regions)}")
    for r in sr_regions:
        X, Y = r["centroid"]
        print(f"          SR R({r['label']}): "
              f"centroid=({X:.2f},{Y:.2f}), "
              f"size={r['size']}px, "
              f"bbox_corrected={r['centroid_corrected']}")
 
    return {
        **step3,
        "binary_image" : binary_image,
        "labeled_image": labeled_image,
        "all_regions"  : regions,
        "sr_regions"   : sr_regions,
    }