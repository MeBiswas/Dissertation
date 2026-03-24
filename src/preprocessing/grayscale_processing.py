# preprocessing/grayscale_processing.py
import cv2
import warnings
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, Dict

"""
    Optimized image processing class for thermal breast images.
"""
class ImageProcessor:
    
    # Class constants for better maintainability
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_TILE_SIZE = (8, 8)
    MEDIAN_BLUR_KERNEL = 3
    
    # DMR-IR format B dimensions (FLIR camera with overlay)
    FORMAT_B_DIMS = (120, 160)
    FORMAT_B_CROP = {
        'top': 18,
        'bottom': 100,
        'left': 0,
        'right': 134
    }
    
    def __init__(self):
        self._clahe = cv2.createCLAHE(
            clipLimit=self.CLAHE_CLIP_LIMIT,
            tileGridSize=self.CLAHE_TILE_SIZE
        )
    
    def enhance_thermal(self, image_array: np.ndarray) -> np.ndarray:
        """
        Cleans noise and enhances thermal contrast for better SR detection.
        
        Args:
            image_array: Input grayscale image
            
        Returns:
            Enhanced image
        """
        # 1. Remove 1px noise using median blur
        denoised = cv2.medianBlur(image_array, self.MEDIAN_BLUR_KERNEL)
        
        # 2. Normalize to use full 0-255 range (non-destructive)
        mask = denoised > 0
        if np.any(mask):
            min_val = denoised[mask].min()
            max_val = denoised.max()
            if max_val > min_val:
                # Vectorized normalization
                normalized = ((denoised - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                # Apply mask to preserve background
                denoised = np.where(mask, normalized, denoised)
        
        # 3. CLAHE for local contrast enhancement
        enhanced = self._clahe.apply(denoised)
        
        return enhanced
    
    def _load_grayscale(self, image_path: str) -> np.ndarray:
        """
        Load image and convert to grayscale (MSB/R channel).
        
        Args:
            image_path: Path to image file
            
        Returns:
            Grayscale image array
        """
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Remove color scale bar before MSB extraction
        img_no_scale, crop_info = self.remove_color_scale(img)
        
        # Extract R channel (MSB grayscale)
        msb_gray = self.to_grayscale(img_no_scale)
        
        print(f"[Log] Loaded {image_path}, removed color scale, extracted MSB grayscale.")
        return msb_gray
    
    def visualize(self, img: np.ndarray, title: str = "Image") -> None:
        """Display image with title."""
        plt.figure(figsize=(8, 6))
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def visualize_original_processed_and_histogram(
        self,
        original_img: np.ndarray,
        processed_img: np.ndarray,
        original_title: str,
        processed_title: str,
        histogram_title: str
    ) -> None:
        """Display original image, processed image, and histogram."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original Image
        axes[0].imshow(original_img, cmap='gray')
        axes[0].set_title(original_title)
        axes[0].axis('off')
        
        # Processed Image
        axes[1].imshow(processed_img, cmap='gray')
        axes[1].set_title(processed_title)
        axes[1].axis('off')
        
        # Histogram
        axes[2].hist(processed_img.ravel(), 256, range=[0, 256], color='gray')
        axes[2].set_title(histogram_title)
        axes[2].set_xlabel('Pixel Intensity')
        axes[2].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
    
    def to_grayscale(self, image_array: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale. For BGR images, extract R channel (MSB).
        
        Args:
            image_array: Input image array
            
        Returns:
            Grayscale image array
        """
        if len(image_array.shape) == 2 or (len(image_array.shape) == 3 and image_array.shape[2] == 1):
            return image_array
        elif len(image_array.shape) == 3 and image_array.shape[2] == 3:
            # Extract R channel (index 2 in BGR)
            return image_array[:, :, 2].copy()
        else:
            warnings.warn(f"Unsupported channel count: {image_array.shape}, returning original")
            return image_array
    
    def remove_color_scale(self, color_image: np.ndarray, image_name: str = "") -> Tuple[np.ndarray, Dict]:
        """
        Remove color scale bar from FLIR thermal images.
        
        Args:
            color_image: Input BGR image
            image_name: Optional image name for logging
            
        Returns:
            Tuple of (cropped_image, crop_info_dict)
        """
        h, w = color_image.shape[:2]
        
        # Check if this is Format B (older FLIR images with overlay)
        if h == self.FORMAT_B_DIMS[0] and w == self.FORMAT_B_DIMS[1]:
            crop = self.FORMAT_B_CROP
            cropped = color_image[
                crop['top']:crop['bottom'],
                crop['left']:crop['right']
            ]
            
            crop_info = {
                "format": "B (2013-2015, FLIR overlay)",
                "original_shape": (h, w),
                "cropped_shape": cropped.shape[:2],
                "removed_top": crop['top'],
                "removed_bottom": h - crop['bottom'],
                "removed_right": w - crop['right'],
            }
            print(f"[Log] Format B detected ({h}x{w}). Cropped to {cropped.shape[0]}x{cropped.shape[1]}.")
        else:
            # Format A — clean images, no cropping needed
            cropped = color_image.copy()
            crop_info = {
                "format": "A (2018-2020, clean)",
                "original_shape": (h, w),
                "cropped_shape": (h, w),
                "removed_top": 0,
                "removed_bottom": 0,
                "removed_right": 0,
            }
            print(f"[Log] Format A detected ({h}x{w}). No cropping needed.")
        
        return cropped, crop_info