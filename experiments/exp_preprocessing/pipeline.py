# experiments/exp_preprocessing/pipeline.py

import os

from src.preprocessing import ImageProcessor, remove_background, gray_level_reconstruction

class PreprocessingPipeline:
    def __init__(self, config):
        self.config = config
        self.dataset_path = config["dataset_path"]
        self.processor = ImageProcessor()
        
        self.image_files = [
            f for f in os.listdir(self.dataset_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.image_files.sort()
        
        if not self.image_files:
            raise ValueError("[Error] No images found.")
        
        print(f"[Pipeline] Found {len(self.image_files)}")
    
    def process_image(self, image_index):
        image_name = self.image_files[image_index]
        image_path = os.path.join(self.dataset_path, image_name)
        
        print(f'\n[Pipeline] Processing: {image_name}')
        
        # Step 1 - Load
        grayscale_img_loaded = self.processor._load_grayscale(image_path)
        
        # Step 2 - Ensure grayscale
        final_grayscale_img = self.processor.to_grayscale(grayscale_img_loaded)
        
        # Step 3 - Background removal
        bg_removed, breast_mask = remove_background(final_grayscale_img)
        
        # Step 4 - Gray-level reconstruction
        pb = gray_level_reconstruction(bg_removed, final_grayscale_img)
        
        result = {
            "pb": pb,
            "mask": breast_mask,
            "image_name": image_name,
            "original": final_grayscale_img,
            "background_removed": bg_removed
        }
        
        if self.config["show_visualizations"]:
            self.processor.visualize_original_processed_and_histogram(
                final_grayscale_img,
                pb,
                "Original",
                "After Reconstruction",
                "Processed Histogram"
            )
        
        return result
    
    def run(self):
        if self.config["process_all"]:
            print("[Pipeline] Running batch processing")
            results = []
            
            for i in range(len(self.image_files)):
                result = self.process_image(i)
                results.append(result)
            
            print("[Pipeline] Batch processing complete")
            return results
        
        else:
            print("[Pipeline] Running single image processing")
            index = self.config["image_index"]
            return self.process_image(index)