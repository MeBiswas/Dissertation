# preprocessing/otsu_thresholding.py

import cv2
import numpy as np

from typing import Tuple
from scipy.ndimage import label

"""
    Remove background using Otsu thresholding and keep largest connected component.

    Args: gray_image: Input grayscale image
        
    Returns: Tuple of (background_removed_image, breast_mask)
"""
def remove_background(gray_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Otsu thresholding
    otsu_thresh, binary_mask = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    print(f"[Otsu] Threshold: {otsu_thresh:.2f}")
    
    # Keep only largest connected component
    labeled_array, num_features = label(binary_mask)
    if num_features > 1:
        # Get sizes of all components (excluding background)
        component_sizes = np.bincount(labeled_array.ravel())
        component_sizes[0] = 0  # Ignore background
        largest_label = component_sizes.argmax()
        binary_mask = (labeled_array == largest_label).astype(np.uint8) * 255
        print(f"[Otsu] Kept largest of {num_features} components.")
    
    breast_mask = (binary_mask > 0).astype(np.uint8)
    background_removed = gray_image * breast_mask
    
    return background_removed.astype(np.uint8), breast_mask