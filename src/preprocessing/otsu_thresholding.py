# preprocessing/otsu_thresholding.py

import cv2
import numpy as np

from scipy.ndimage import label

def remove_background(gray_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    otsu_thresh_value, binary_mask = cv2.threshold(
        gray_image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    print(f"[Otsu] Automatically selected threshold value: {otsu_thresh_value}")

    labeled_array, num_features = label(binary_mask)
    if num_features > 1:
        component_sizes = np.bincount(labeled_array.ravel())
        component_sizes[0] = 0
        largest_label = component_sizes.argmax()
        binary_mask = (labeled_array == largest_label).astype(np.uint8) * 255
        print(f"[Otsu] Found {num_features} components — kept only the largest.")

    breast_mask = (binary_mask > 0).astype(np.uint8)
    background_removed = gray_image * breast_mask

    return background_removed.astype(np.uint8), breast_mask