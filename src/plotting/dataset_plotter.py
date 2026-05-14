# plotting/dataset_plotting.py

import os
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ═════════════════════════════════════════════════════════════════════════════
# DATASET PLOT
# ═════════════════════════════════════════════════════════════════════════════
class DatasetPlot:
    # Constructor
    def __init__(self, cols, path):
        self.cols = cols
        self.path = path
        self.image_files = []

    # Images
    def imageFiles(self):
        self.image_files = [
            f for f in os.listdir(self.path)
            if f.lower().endswith(('.png', '.jpg'))
        ]

    # Plotting
    def imagePlotting(self):
        image_count = len(self.image_files)
        rows = math.ceil(image_count/self.cols)

        plt.figure(figsize=(self.cols*4, rows*4))

        for i, filename in enumerate(self.image_files):
            img_path = os.path.join(self.path, filename)
            img = mpimg.imread(img_path)

            plt.subplot(rows, self.cols, i+1)
            plt.imshow(img)
            plt.title(filename, fontsize=8)
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    def main(self):
        self.imageFiles()
        self.imagePlotting()