# src/preprocessing/storage_config.py

import os
import cv2
import json
from datetime import datetime
from typing import Dict, Optional

# =============================================================================
# STORAGE CONFIGURATION
# =============================================================================
class StorageConfig:
    """Configuration for storing preprocessing results."""
    def __init__(self, output_dir: str = "data/preprocessed"):
        self.output_dir = output_dir
        self.save_pb = True           # Save p_b (final preprocessed image)
        self.save_grayscale = True    # Save grayscale image
        self.save_bg_removed = True   # Save background removed image
        self.save_metadata = True     # Save metadata as JSON
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)


def save_preprocessing_results(
    results: Dict,
    storage_config: Optional[StorageConfig] = None
) -> Dict:
    if storage_config is None:
        storage_config = StorageConfig()
    
    # Extract image name without extension
    original_name = results['image_name']
    base_name = os.path.splitext(original_name)[0]
    
    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a run-specific subdirectory
    run_dir = os.path.join(storage_config.output_dir, f"{base_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    saved_paths = {}
    
    # Save p_b (most important - needed for CHM correction)
    if storage_config.save_pb:
        pb_path = os.path.join(run_dir, f"{base_name}_pb.png")
        cv2.imwrite(pb_path, results['pb'])
        saved_paths['pb'] = pb_path
        print(f"[Save] p_b saved to: {pb_path}")
    
    # Save grayscale image
    if storage_config.save_grayscale:
        gray_path = os.path.join(run_dir, f"{base_name}_grayscale.png")
        cv2.imwrite(gray_path, results['grayscale'])
        saved_paths['grayscale'] = gray_path
        print(f"[Save] Grayscale saved to: {gray_path}")
    
    # Save background removed image
    if storage_config.save_bg_removed:
        bg_path = os.path.join(run_dir, f"{base_name}_bg_removed.png")
        cv2.imwrite(bg_path, results['bg_removed'])
        saved_paths['bg_removed'] = bg_path
        print(f"[Save] Background removed saved to: {bg_path}")
    
    # Save original color (cropped)
    color_path = os.path.join(run_dir, f"{base_name}_original_color.png")
    cv2.imwrite(color_path, results['original_color'])
    saved_paths['original_color'] = color_path
    print(f"[Save] Original color saved to: {color_path}")
    
    # Save metadata as JSON
    if storage_config.save_metadata:
        metadata = {
            'image_name': results['image_name'],
            'base_name': base_name,
            'timestamp': timestamp,
            'pb_shape': results['pb'].shape,
            'pb_dtype': str(results['pb'].dtype),
            'pb_min': int(results['pb'].min()),
            'pb_max': int(results['pb'].max()),
            'pb_nonzero': int((results['pb'] > 0).sum()),
            'saved_files': saved_paths
        }
        
        metadata_path = os.path.join(run_dir, f"{base_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        saved_paths['metadata'] = metadata_path
        print(f"[Save] Metadata saved to: {metadata_path}")
    
    # Add saved paths to results
    results['saved_paths'] = saved_paths
    results['run_dir'] = run_dir
    results['base_name'] = base_name
    results['timestamp'] = timestamp
    
    return results