# preprocessing/step_6.py

# ── Numeric / Image ────────────────────────────────────────────────────────
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1.6 — Gray-level reconstruction  →  p_b
# ─────────────────────────────────────────────────────────────────────────────
#
# RECONSTRUCTION: Background removal can leave thin zero gaps inside the
# breast boundary.  Row-by-row propagation fills them from the original
# pixel values.
#
# RESULT: p_b — the image the paper feeds directly into SCH-CS.
# ─────────────────────────────────────────────────────────────────────────────
def gray_level_reconstruction(
    bg_removed : np.ndarray,
    grayscale : np.ndarray
) -> np.ndarray:
    """Fill interior zero gaps by row-wise propagation from original values."""
    assert bg_removed.shape == grayscale.shape
    rows, cols = bg_removed.shape
    mid = cols // 2
    pb = bg_removed.copy().astype(np.float32)

    for r in range(rows):
        for c in range(1, mid):
            if pb[r, c] == 0 and pb[r, c-1] != 0:
                pb[r, c] = grayscale[r, c]

    for r in range(rows):
        for c in range(cols - 2, mid - 1, -1):
            if pb[r, c] == 0 and pb[r, c+1] != 0:
                pb[r, c] = grayscale[r, c]

    pb = pb.astype(np.uint8)
    nz = int((pb > 0).sum())
    print(f'[1.5b] Reconstruction done. Non-zero pixels: {nz} / {pb.size}.')
    return pb