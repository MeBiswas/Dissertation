# preprocessing/grayscale_processing.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageProcessor:
    def __init__(self):
        pass

    def _load_grayscale(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[Error] ImageProcessor: Could not load image from {image_path}")
            raise ValueError(f"Could not load image from {image_path}")

        # Step 1: Remove color scale bar BEFORE MSB extraction
        img_no_scale, crop_col = self.remove_color_scale(img)
        
        # Step 2: Extract R channel (MSB grayscale)
        # msb_gray = self.to_grayscale(img)
        msb_gray = self.to_grayscale(img_no_scale)

        print(f"[Log] ImageProcessor: Loaded {image_path}, "
          f"removed color scale at col {crop_col}, "
          f"extracted MSB grayscale.")
        return msb_gray

    def visualize(self, img, title="Image"):
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()
        print(f"[Log] ImageProcessor: Displayed image with title '{title}'.")

    def visualize_original_processed_and_histogram(
        self,
        original_img,
        processed_img,
        original_title,
        processed_title,
        histogram_title
    ):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot Original Image
        axes[0].imshow(original_img, cmap='gray')
        axes[0].set_title(original_title)
        axes[0].axis('off')

        # Plot Processed Image
        axes[1].imshow(processed_img, cmap='gray')
        axes[1].set_title(processed_title)
        axes[1].axis('off')

        # Plot Histogram of Processed Image
        axes[2].hist(processed_img.ravel(), 256, range=[0, 256], color='gray')
        axes[2].set_title(histogram_title)
        axes[2].set_xlabel('Pixel Intensity')
        axes[2].set_ylabel('Frequency')

        plt.tight_layout()
        plt.show()
        print(f"[Log] ImageProcessor: Displayed original, processed image and its histogram for '{original_title}'.")

    def to_grayscale(self, image_array):
        # Already grayscale — return as-is
        if len(image_array.shape) == 2 or (
                len(image_array.shape) == 3 and image_array.shape[2] == 1):
            print("[Log] ImageProcessor: Image is already grayscale.")
            return image_array

        elif len(image_array.shape) == 3 and image_array.shape[2] == 3:
            # cv2 loads images as BGR — index 2 is the R channel
            msb_gray = image_array[:, :, 2].copy()
            print("[Log] ImageProcessor: Extracted MSB grayscale "
                  "(R channel) from BGR image.")
            return msb_gray

        else:
            print("[Warning] ImageProcessor: Unsupported number of channels. "
                  "Returning original image.")
            return image_array

    def remove_color_scale(self, color_image, image_name=""):
        h, w = color_image.shape[:2]

        if h == 120 and w == 160:
            
            top    = 18     # remove top parameter overlay (rows 0-17)
            bottom = 100    # remove bottom FLIR logo strip (rows 105-119)
            left   = 0      # no overlay on left side
            right  = 134    # remove color bar (cols 134-159)

            cropped = color_image[top:bottom, left:right]

            crop_info = {
                "format"         : "B (2013-2015, FLIR overlay)",
                "original_shape" : (h, w),
                "cropped_shape"  : cropped.shape[:2],
                "removed_top"    : top,
                "removed_bottom" : h - bottom,
                "removed_right"  : w - right,
            }
            print(f"[Log] ImageProcessor: Format B detected ({h}x{w}). "
                f"Removed top={top}px, bottom={h-bottom}px, "
                f"right={w-right}px. "
                f"New size: {cropped.shape[0]}x{cropped.shape[1]}.")

        else:
            # FORMAT A — 2018-2020 clean image, no cropping needed
            cropped = color_image.copy()

            crop_info = {
                "format"         : "A (2018-2020, clean)",
                "original_shape" : (h, w),
                "cropped_shape"  : (h, w),
                "removed_top"    : 0,
                "removed_bottom" : 0,
                "removed_right"  : 0,
            }
            print(f"[Log] ImageProcessor: Format A detected ({h}x{w}). "
                f"No cropping needed.")

        return cropped, crop_info