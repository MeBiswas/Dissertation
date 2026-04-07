# experiments/exp_preprocessing/pipeline.py

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, List, Any, Tuple
from src.preprocessing import ImageProcessor, gray_level_reconstruction

# ─────────────────────────────────────────────────────────────────────────────
# PRE-PROCESSING PIPELINE
# 
# ─────────────────────────────────────────────────────────────────────────────
"""Pipeline for preprocessing thermal breast images."""
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
        
        # Detection configuration
        self.hot_region_percentile = config.get("hot_region_percentile", 95)
        self.num_hot_regions = config.get("num_hot_regions", 2)
        
        # Get and validate image files
        self.image_files = self._get_image_files()
        
        if not self.image_files:
            raise ValueError(f"[Error] No images found in {self.dataset_path}")
        
        print(f"[Pipeline] Found {len(self.image_files)} images")
    
    def _get_image_files(self) -> List[str]:
        """Get sorted list of image files from dataset path."""
        extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
        return sorted([f for f in os.listdir(self.dataset_path) if f.lower().endswith(extensions)])
    
    
    """
        Crop neck, armpit, and stomach regions.
        
        Returns:
            Tuple of (cropped_image, crop_coords) where crop_coords = (row_start, row_end, col_start, col_end)
    """
    def _crop_anatomical_regions(self, gray_image: np.ndarray, breast_mask: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        if not self.enable_cropping:
            h, w = gray_image.shape[:2]
            return gray_image, (0, h, 0, w)
        
        coords = np.argwhere(breast_mask > 0)
        if len(coords) == 0:
            h, w = gray_image.shape[:2]
            return gray_image, (0, h, 0, w)
        
        min_row, min_col = coords.min(axis=0)
        max_row, max_col = coords.max(axis=0)
        
        height = max_row - min_row
        width = max_col - min_col
        
        # Calculate crop boundaries
        top_crop = int(height * self.crop_neck_percent)
        bottom_crop = int(height * self.crop_stomach_percent)
        left_crop = int(width * self.crop_armpit_percent)
        right_crop = int(width * self.crop_armpit_percent)
        
        # Safety checks
        if top_crop + bottom_crop >= height - 10:
            top_crop, bottom_crop = height // 4, height // 6
            print(f"[Warning] Adjusted cropping: top={top_crop}, bottom={bottom_crop}")
        
        if left_crop + right_crop >= width - 10:
            left_crop = right_crop = width // 4
            print(f"[Warning] Adjusted cropping: left={left_crop}, right={right_crop}")
        
        # Apply cropping
        row_start, row_end = min_row + top_crop, max_row - bottom_crop
        col_start, col_end = min_col + left_crop, max_col - right_crop
        
        cropped = gray_image[row_start:row_end, col_start:col_end]
        
        print(f"[Cropping] Removed neck({top_crop}px), stomach({bottom_crop}px), "
              f"armpits(L:{left_crop},R:{right_crop}) → {gray_image.shape} → {cropped.shape}")
        
        return cropped, (row_start, row_end, col_start, col_end)
    
    
    """
        Detect hottest regions using adaptive thresholding and contour detection.
        
        Args:
            grayscale_image: Grayscale image of the breast
            
        Returns:
            List of tuples (center_x, center_y, radius) for each circle
    """
    def _detect_hottest_regions(self, grayscale_image: np.ndarray) -> List[Tuple[int, int, int]]:
        # Get non-zero pixels only
        nonzero_pixels = grayscale_image[grayscale_image > 0]
        if len(nonzero_pixels) == 0:
            return []
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(grayscale_image, self.processor.GAUSSIAN_BLUR_KERNEL, 0)
        
        # Dynamic threshold based on percentile
        threshold = np.percentile(nonzero_pixels, self.hot_region_percentile)
        hot_mask = (blurred >= threshold).astype(np.uint8) * 255
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.processor.MORPH_HOT_KERNEL)
        hot_mask = cv2.morphologyEx(hot_mask, cv2.MORPH_CLOSE, kernel)
        hot_mask = cv2.morphologyEx(hot_mask, cv2.MORPH_OPEN, kernel)
        
        # Apply original mask to ensure we only consider breast region
        breast_mask = (grayscale_image > 0).astype(np.uint8)
        hot_mask = cv2.bitwise_and(hot_mask, hot_mask, mask=breast_mask)
        
        # Find contours
        contours, _ = cv2.findContours(hot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        circles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.processor.CONTOUR_AREA_MIN:
                # Get minimum enclosing circle
                (x, y), radius = cv2.minEnclosingCircle(contour)
                radius = int(radius)
                
                if radius > self.processor.HOT_REGION_RADIUS_MIN:
                    # Calculate average intensity for this region
                    region_mask = np.zeros_like(grayscale_image, dtype=np.uint8)
                    cv2.drawContours(region_mask, [contour], -1, 255, -1)
                    avg_intensity = np.mean(grayscale_image[region_mask > 0])
                    circles.append((int(x), int(y), radius, area, avg_intensity))
        
        # Sort by average intensity (hottest first) and take top regions
        circles.sort(key=lambda x: x[4], reverse=True)
        return [(c[0], c[1], c[2]) for c in circles[:self.num_hot_regions]]
    
    """
        Create visualization matching Figure 2 from the paper with black circles marking hottest regions.
    """
    def _visualize_paper_figure(self, original_color: np.ndarray, without_scale: np.ndarray,
                                grayscale: np.ndarray, background_removed: np.ndarray,
                                reconstructed: np.ndarray, image_name: str) -> None:
        # Detect hottest regions
        hot_regions = self._detect_hottest_regions(grayscale)
        
        # Draw circles on original image
        original_with_circles = original_color.copy()
        for cx, cy, radius in hot_regions:
            cv2.circle(original_with_circles, (cx, cy), radius, (0, 0, 0), 3)
            cv2.circle(original_with_circles, (cx, cy), radius, (255, 255, 255), 1)
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Preprocessing Pipeline - {image_name}\nBlack circles mark hottest regions",
                     fontsize=14, fontweight='bold')
        
        # (a) Original with circles
        axes[0, 0].imshow(cv2.cvtColor(original_with_circles, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title(f"(a) Original Pseudo-Color TBI\n{len(hot_regions)} hottest regions", fontsize=10)
        axes[0, 0].axis('off')
        
        if hot_regions:
            info_text = "\n".join([f"R{i+1}: ({cx},{cy}) r={r}" for i, (cx, cy, r) in enumerate(hot_regions)])
            axes[0, 0].text(5, 5, info_text, fontsize=8, color='white',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))
        
        # (b) Without color scale
        axes[0, 1].imshow(cv2.cvtColor(without_scale, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title("(b) Without Color-Scale Bar", fontsize=10)
        axes[0, 1].axis('off')
        
        # (c) Grayscale image
        axes[0, 2].imshow(grayscale, cmap='gray')
        axes[0, 2].set_title("(c) Grayscale (Inverted Blue)\nhot regions = bright", fontsize=10)
        axes[0, 2].axis('off')
        
        # (d) Background removed
        axes[1, 0].imshow(background_removed, cmap='gray')
        axes[1, 0].set_title("(d) Background Removed\n(Otsu + Largest Component)", fontsize=10)
        axes[1, 0].axis('off')
        
        # (e) After reconstruction and cropping
        axes[1, 1].imshow(reconstructed, cmap='gray')
        cropping_status = " (Cropped)" if self.enable_cropping else ""
        axes[1, 1].set_title(f"(e) After Reconstruction & Cropping{cropping_status}", fontsize=10)
        axes[1, 1].axis('off')
        
        # (g) Processed histogram
        mean_val = np.mean(reconstructed[reconstructed > 0]) if np.any(reconstructed > 0) else 0
        axes[1, 2].hist(reconstructed.ravel(), 256, range=[0, 256], color='steelblue', alpha=0.7)
        axes[1, 2].axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.1f}')
        axes[1, 2].set_title("(g) Processed Histogram", fontsize=10)
        axes[1, 2].set_xlabel("Pixel Intensity", fontsize=9)
        axes[1, 2].set_ylabel("Frequency", fontsize=9)
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].legend(fontsize=8)
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print(f"\n[Statistics] {image_name}")
        print(f"  - Hot regions: {len(hot_regions)}")
        for i, (cx, cy, r) in enumerate(hot_regions):
            print(f"    Region {i+1}: center=({cx},{cy}), radius={r}px")
        print(f"  - Grayscale range: [{grayscale.min()}, {grayscale.max()}]")
        print(f"  - Background removed: {np.sum(background_removed > 0)} pixels")
        print(f"  - Reconstructed shape: {reconstructed.shape}")
        print(f"  - Histogram mean: {mean_val:.2f}")
    
    """Save processing results to disk."""
    def _save_results(self, result: Dict[str, Any]) -> None:
        output_dir = self.config.get("output_dir", "outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(result["image_name"])[0]
        files = [
            ("01_original_color", result["original_color"]),
            ("02_without_scale", result["without_scale"]),
            ("03_grayscale", result["grayscale"]),
            ("04_background_removed", result["background_removed"]),
            ("05_cropped", result["cropped"]),
            ("06_final_enhanced", result["pb"]),
            ("07_mask", result["mask"] * 255)
        ]
        
        for suffix, img in files:
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_{suffix}.png"), img)
        
        print(f"[Save] All results saved to {output_dir}/")
    
    """Process a single image through the complete pipeline."""
    def process_image(self, image_index: int) -> Dict[str, Any]:
        image_name = self.image_files[image_index]
        image_path = os.path.join(self.dataset_path, image_name)
        print(f'\n[Pipeline] Processing: {image_name}')
        
        # Load original color image
        original_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if original_color is None:
            raise ValueError(f"Could not load: {image_path}")
        
        # Remove color scale bar → Figure 2(b)
        without_scale, crop_info = self.processor.remove_color_scale(original_color.copy())
        
        # Convert to inverted blue channel → Figure 2(c)
        grayscale = self.processor.to_grayscale(without_scale)
        
        # Background removal
        bg_removed, breast_mask = self.processor.remove_background(grayscale, color_image=without_scale)
        
        # Crop anatomical regions
        cropped_bg, crop_coords = self._crop_anatomical_regions(bg_removed, breast_mask)
        
        # Crop grayscale with exact same coordinates
        row_start, row_end, col_start, col_end = crop_coords
        grayscale_cropped = grayscale[row_start:row_end, col_start:col_end]
        
        # Gray-level reconstruction
        pb = gray_level_reconstruction(cropped_bg, grayscale_cropped)
        
        if self.enable_cropping:
            breast_mask = (cropped_bg > 0).astype(np.uint8)
        
        result = {
            "pb": pb,
            "mask": breast_mask,
            "image_name": image_name,
            "original_color": original_color,
            "without_scale": without_scale,
            "grayscale": grayscale,
            "background_removed": bg_removed,
            "cropped": cropped_bg,
            "crop_info": crop_info,
        }
        
        if self.config.get("show_visualizations", False):
            self._visualize_paper_figure(
                original_color=original_color,
                without_scale=without_scale,
                grayscale=grayscale,
                background_removed=bg_removed,
                reconstructed=pb,
                image_name=image_name
            )
        
        if self.config.get("save_results", False):
            self._save_results(result)
        
        return result
    
    """Run the preprocessing pipeline."""
    def run(self) -> Any:
        if self.config.get("process_all", False):
            print("[Pipeline] Running batch processing...")
            results = [self.process_image(i) for i in range(len(self.image_files))]
            print("[Pipeline] Batch processing complete")
            return results
        else:
            print("[Pipeline] Running single image processing")
            return self.process_image(self.config.get("image_index", 0))