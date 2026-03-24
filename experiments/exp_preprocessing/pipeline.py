# experiments/exp_preprocessing/pipeline.py

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, List, Any
from src.preprocessing import ImageProcessor, remove_background, gray_level_reconstruction

# ─────────────────────────────────────────────────────────────────────────────
# PRE-PROCESSING PIPELINE
# 
# ─────────────────────────────────────────────────────────────────────────────
"""
    Initialize preprocessing pipeline with configuration.
    
    Args:
        config: Configuration dictionary with keys:
            - dataset_path: Path to image dataset
            - process_all: If True, process all images; else process single image
            - image_index: Index of image to process (if process_all=False)
            - enable_cropping: Whether to crop anatomical regions
            - crop_neck_percent: Percentage to crop from top (neck region)
            - crop_stomach_percent: Percentage to crop from bottom (stomach region)
            - crop_armpit_percent: Percentage to crop from sides (armpit regions)
            - show_visualizations: Whether to display visualizations
            - save_results: Whether to save results to disk
            - output_dir: Directory for saving results
"""
class PreprocessingPipeline:
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dataset_path = config["dataset_path"]
        self.processor = ImageProcessor()
        
        # Cropping configuration
        self.enable_cropping = config.get("enable_cropping", True)
        self.crop_neck_percent = config.get("crop_neck_percent", 0.18)
        self.crop_stomach_percent = config.get("crop_stomach_percent", 0.12)
        self.crop_armpit_percent = config.get("crop_armpit_percent", 0.22)
        
        # Get and validate image files
        self.image_files = self._get_image_files()
        
        if not self.image_files:
            raise ValueError(f"[Error] No images found in {self.dataset_path}")
        
        print(f"[Pipeline] Found {len(self.image_files)} images")
    
    def _get_image_files(self) -> List[str]:
        """Get sorted list of image files from dataset path."""
        extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
        files = [
            f for f in os.listdir(self.dataset_path)
            if f.lower().endswith(extensions)
        ]
        return sorted(files)
    
    def _crop_anatomical_regions(self, gray_image: np.ndarray, breast_mask: np.ndarray) -> np.ndarray:
        """
        Crop out neck, armpit, and stomach regions to focus on breast tissue only.
        
        Args:
            gray_image: Grayscale image
            breast_mask: Binary mask of breast region
            
        Returns:
            Cropped image
        """
        if not self.enable_cropping:
            return gray_image
        
        # Get breast region bounds
        coords = np.argwhere(breast_mask > 0)
        if len(coords) == 0:
            return gray_image
        
        min_row, min_col = coords.min(axis=0)
        max_row, max_col = coords.max(axis=0)
        
        height = max_row - min_row
        width = max_col - min_col
        
        # Calculate crop boundaries with safety checks
        top_crop = int(height * self.crop_neck_percent)
        bottom_crop = int(height * self.crop_stomach_percent)
        left_crop = int(width * self.crop_armpit_percent)
        right_crop = int(width * self.crop_armpit_percent)
        
        # Safety: ensure we don't crop too much
        max_top_bottom = height - 10  # Leave at least 10 pixels
        max_left_right = width - 10
        
        if top_crop + bottom_crop >= max_top_bottom:
            top_crop = height // 4
            bottom_crop = height // 6
            print(f"[Warning] Adjusted cropping: top={top_crop}, bottom={bottom_crop}")
        
        if left_crop + right_crop >= max_left_right:
            left_crop = width // 4
            right_crop = width // 4
            print(f"[Warning] Adjusted cropping: left={left_crop}, right={right_crop}")
        
        # Apply cropping
        cropped = gray_image[
            min_row + top_crop : max_row - bottom_crop,
            min_col + left_crop : max_col - right_crop
        ]
        
        print(f"[Cropping] Removed neck({top_crop}px), stomach({bottom_crop}px), "
              f"armpits(L:{left_crop},R:{right_crop})")
        print(f"[Cropping] Shape: {gray_image.shape} → {cropped.shape}")
        
        return cropped
    
    def _visualize_cropping(self, before: np.ndarray, after: np.ndarray, enhanced: np.ndarray) -> None:
        """Visualize cropping and enhancement results."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        titles = [
            "Before Cropping\n(Original Breast)",
            "After Cropping\n(Neck/Armpits/Stomach Removed)",
            "After Enhancement"
        ]
        
        for ax, img, title in zip(axes, [before, after, enhanced], titles):
            ax.imshow(img, cmap='gray')
            ax.set_title(title)
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def process_image(self, image_index: int) -> Dict[str, Any]:
        """
        Process a single image through the complete pipeline.
        
        Args:
            image_index: Index of image to process
            
        Returns:
            Dictionary containing processed results
        """
        image_name = self.image_files[image_index]
        image_path = os.path.join(self.dataset_path, image_name)
        
        print(f'\n[Pipeline] Processing: {image_name}')
        
        # Step 1 & 2: Load and convert to grayscale
        final_grayscale = self.processor._load_grayscale(image_path)
        
        # Step 3: Background removal
        bg_removed, breast_mask = remove_background(final_grayscale)
        
        # Step 4: Crop anatomical regions
        cropped_bg = self._crop_anatomical_regions(bg_removed, breast_mask)
        
        # Update mask for cropped region
        if self.enable_cropping:
            breast_mask = (cropped_bg > 0).astype(np.uint8)
        
        # Step 5: Gray-level reconstruction
        pb_raw = gray_level_reconstruction(cropped_bg, cropped_bg)
        
        # Step 6: Thermal enhancement
        pb_enhanced = self.processor.enhance_thermal(pb_raw)
        
        # Prepare results
        result = {
            "pb": pb_enhanced,
            "pb_raw": pb_raw,
            "mask": breast_mask,
            "image_name": image_name,
            "original": final_grayscale,
            "background_removed": bg_removed,
            "cropped": cropped_bg
        }
        
        # Visualize if requested
        if self.config.get("show_visualizations", False):
            self._visualize_cropping(bg_removed, cropped_bg, pb_enhanced)
            self.processor.visualize_original_processed_and_histogram(
                final_grayscale, pb_enhanced,
                "Original", "After Reconstruction & Cropping", "Processed Histogram"
            )
        
        # Save results if requested
        if self.config.get("save_results", False):
            self._save_results(result)
        
        return result
    
    def _save_results(self, result: Dict[str, Any]) -> None:
        """Save processing results to disk."""
        output_dir = self.config.get("output_dir", "outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(result["image_name"])[0]
        
        # Save processed images
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_pb.png"), result["pb"])
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_mask.png"), result["mask"] * 255)
        
        print(f"[Save] Results saved to {output_dir}/")
    
    def run(self) -> Dict[str, Any]:
        """
        Run the preprocessing pipeline.
        
        Returns:
            Processed results (single image or list of images)
        """
        if self.config.get("process_all", False):
            print("[Pipeline] Running batch processing...")
            results = []
            for i in range(len(self.image_files)):
                results.append(self.process_image(i))
            print("[Pipeline] Batch processing complete")
            return results
        else:
            print("[Pipeline] Running single image processing")
            index = self.config.get("image_index", 0)
            return self.process_image(index)