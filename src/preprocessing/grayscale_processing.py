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
        msb_gray = self.to_grayscale(img)

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

    def visualize_original_processed_and_histogram(self, original_img, processed_img, original_title, processed_title, histogram_title):
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
        """
        Converts a pseudo-color TBI to grayscale using MSB extraction
        as described in Section II-A of Pramanik et al. (2018):

            'only the most significant byte of each pixel value is
             extracted to form a grayscale image of breast thermogram'

        In a 24-bit pseudo-color TBI, temperature is encoded such that
        the R channel carries the primary thermal intensity information
        with the highest contrast between hot and cool regions.
        Extracting R directly gives a sparse, multimodal histogram —
        which is exactly what the SCH-CS step requires.

        Standard cv2.COLOR_BGR2GRAY (0.299R + 0.587G + 0.114B) blends
        all channels together, producing a dense concentrated histogram
        where the SCH-CS count threshold rho goes negative.
        """
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
        
    def remove_color_scale(self, color_image):
        """
        Removes the color scale bar from a pseudo-color TBI.
        
        As described in Section II-A of Pramanik et al. (2018):
            'a color scale of fixed width appears alongside the image
            at a fixed position... the color scale is first automatically
            removed from the TBI using the width and positional
            information of it'

        The FLIR color scale bar is a vertical strip on the RIGHT edge.
        This method detects its width automatically by scanning from the
        right edge leftward and finding where image content begins.

        Args:
            color_image : Original BGR image as loaded by cv2.

        Returns:
            cropped : BGR image with color scale bar removed.
            crop_col : The column index where cropping was applied.
                    Useful for debugging/logging.
        """
        h, w = color_image.shape[:2]

        # Convert to grayscale just for edge detection
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # The color scale bar is typically a uniform vertical gradient
        # with a sharp vertical edge where it meets the main image.
        # Strategy: compute column-wise std deviation.
        # The color bar columns have LOW std (uniform gradient per column).
        # The image columns have HIGH std (varied tissue texture).
        col_std = np.std(gray, axis=0)   # shape: (width,)

        # Scan from right edge leftward — find first column with
        # std above a threshold (= real image content starts here)
        threshold = 10.0    # experimentally reasonable for FLIR images
        crop_col  = w       # default: no cropping

        for c in range(w - 1, w // 2, -1):
            if col_std[c] > threshold:
                crop_col = c + 1
                break

        cropped = color_image[:, :crop_col, :]

        print(f"[Log] ImageProcessor: Color scale removed. "
            f"Cropped at column {crop_col} (original width={w}, "
            f"new width={crop_col}).")

        return cropped, crop_col