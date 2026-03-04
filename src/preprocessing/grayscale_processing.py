# preprocessing/grayscale_processing.py
import cv2
import matplotlib.pyplot as plt

class ImageProcessor:
    def __init__(self):
        pass

    def _load_grayscale(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[Error] ImageProcessor: Could not load image from {image_path}")
            raise ValueError(f"Could not load image from {image_path}")
        print(f"[Log] ImageProcessor: Loaded image {image_path} in grayscale.")
        return img

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

    # Checks if an image is grayscale; if not, converts it to grayscale.
    def to_grayscale(self, image_array):
        if len(image_array.shape) == 2 or (len(image_array.shape) == 3 and image_array.shape[2] == 1):
            print("[Log] ImageProcessor: Image is already grayscale.")
            return image_array
        elif len(image_array.shape) == 3 and image_array.shape[2] == 3:
            grayscale_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            print("[Log] ImageProcessor: Converted color image to grayscale.")
            return grayscale_image
        else:
            print("[Warning] ImageProcessor: Image has an unsupported number of channels. Returning original image.")
            return image_array