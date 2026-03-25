# preprocessing/gray_level_reconstruction

import numpy as np

"""
Reconstruct missing gray levels from left and right sides.

Args:
    background_removed: Image with background removed
    original: Original grayscale image
    
Returns: Reconstructed image
"""
def gray_level_reconstruction(background_removed: np.ndarray, original: np.ndarray) -> np.ndarray:
    rows, cols = background_removed.shape
    mid = cols // 2
    reconstructed = background_removed.copy()
    
    # Left-to-right reconstruction
    for r in range(rows):
        for c in range(1, mid):
            if reconstructed[r, c] == 0 and reconstructed[r, c - 1] != 0:
                reconstructed[r, c] = original[r, c]
    
    # Right-to-left reconstruction
    for r in range(rows):
        for c in range(cols - 2, mid - 1, -1):
            if reconstructed[r, c] == 0 and reconstructed[r, c + 1] != 0:
                reconstructed[r, c] = original[r, c]
    
    return reconstructed