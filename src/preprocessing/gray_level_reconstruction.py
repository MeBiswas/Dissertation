# preprocessing/gray_level_reconstruction

import numpy as np

def gray_level_reconstruction(background_removed: np.ndarray, original: np.ndarray) -> np.ndarray:
    rows, cols = background_removed.shape
    mid = cols // 2

    I_c = background_removed.copy()

    for r in range(rows):
        for c in range(1, mid):
            if I_c[r, c] == 0 and I_c[r, c - 1] != 0:
                I_c[r, c] = original[r, c]

    for r in range(rows):
        for c in range(cols - 2, mid - 1, -1):
            if I_c[r, c] == 0 and I_c[r, c + 1] != 0:
                I_c[r, c] = original[r, c]

    return I_c