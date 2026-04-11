# src/sr_segmentation/main.py

import cv2
import numpy as np

from .visualization import visualize_split
from .split_mask import split_sr_left_right

# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Full split pipeline
# ─────────────────────────────────────────────────────────────────────────────
 
def run_sr_split(preprocessed_path: str, segmented_sr_path: str, save_output: bool = True) -> tuple[np.ndarray, np.ndarray]:
    # ── Load preprocessed TBI ─────────────────────────────────────────────────
    p_b = cv2.imread(preprocessed_path, cv2.IMREAD_GRAYSCALE)
    if p_b is None:
        raise FileNotFoundError(f"Could not load: {preprocessed_path}")
 
    # ── Load segmented SR (supports .npy and image files) ────────────────────
    if segmented_sr_path.endswith('.npy'):
        segmented_sr = np.load(segmented_sr_path).astype(np.uint8)
    else:
        segmented_sr = cv2.imread(segmented_sr_path, cv2.IMREAD_GRAYSCALE)
        if segmented_sr is None:
            raise FileNotFoundError(f"Could not load: {segmented_sr_path}")
        _, segmented_sr = cv2.threshold(segmented_sr, 127, 1,
                                        cv2.THRESH_BINARY)
 
    print(f"[Load]  Preprocessed TBI shape : {p_b.shape}")
    print(f"[Load]  Segmented SR shape     : {segmented_sr.shape}")
 
    # ── Split ─────────────────────────────────────────────────────────────────
    sr_img_left, sr_img_right, centre_col = split_sr_left_right(
        segmented_sr, p_b
    )
 
    # ── Save ──────────────────────────────────────────────────────────────────
    cv2.imwrite("sr_left.png",  sr_img_left  * 255)
    cv2.imwrite("sr_right.png", sr_img_right * 255)
    np.save("sr_left.npy",  sr_img_left)
    np.save("sr_right.npy", sr_img_right)
    print("[Save]  sr_left.png / sr_left.npy")
    print("[Save]  sr_right.png / sr_right.npy")
 
    # ── Visualize ─────────────────────────────────────────────────────────────
    if save_output:
        visualize_split(p_b, segmented_sr,
                        sr_img_left, sr_img_right,
                        centre_col,
                        save_path="sr_split_result.png")
 
    print("\n[Ready] Pass these into feature_extraction.py:")
    print("          sr_left_path  = 'sr_left.png'")
    print("          sr_right_path = 'sr_right.png'")
 
    return sr_img_left, sr_img_right