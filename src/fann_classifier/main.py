# src/fann_classifier/main.py

import os
import numpy as np
from datetime import datetime
from .step_4 import visualize_results
from .step_3 import run_cross_validation

# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def run_fann_classification(
    F_dataset: np.ndarray,
    labels: np.ndarray,
    base_output_dir: str = "fann_classification",
    run_name: str = "image1",
    save_output: bool = True
) -> dict:

    # ── Create timestamped folder ────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_output_dir, f"{run_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"[SaveDir] {output_dir}")

    # ── Run cross-validation ─────────────────────────────────────────────────
    results = run_cross_validation(F_dataset, labels)

    # ── Save outputs ─────────────────────────────────────────────────────────
    if save_output:

        # 1. Save raw arrays
        np.save(os.path.join(output_dir, "all_y_true.npy"), results['all_y_true'])
        np.save(os.path.join(output_dir, "all_y_prob.npy"), results['all_y_prob'])

        # 2. Save metrics summary
        metrics_path = os.path.join(output_dir, "metrics.txt")
        with open(metrics_path, "w") as f:
            f.write("FINAL RESULTS (mean ± std)\n")
            f.write("=" * 50 + "\n")

            for k in results['mean'].keys():
                mean = results['mean'][k]
                std  = results['std'][k]

                if k == 'auc':
                    f.write(f"{k:15}: {mean:.4f} ± {std:.4f}\n")
                else:
                    f.write(f"{k:15}: {mean*100:.2f}% ± {std*100:.2f}%\n")

        print(f"[Save] metrics.txt")

        # 3. Save per-fold details (optional but useful)
        np.save(os.path.join(output_dir, "per_fold.npy"), results['per_fold'])

        # 4. Visualization (THIS FIXES YOUR MAIN ISSUE PATTERN)
        visualize_results(
            results,
            save_path=os.path.join(output_dir, "classification_results.png")
        )

    print("\n[Ready] Classification complete.")
    return results