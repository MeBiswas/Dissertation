# src/sch_cs/storage_config.py

import os
import cv2
import json
import pickle
import numpy as np
from datetime import datetime
from typing import Dict, Optional

# =============================================================================
# STORAGE CONFIGURATION
# =============================================================================

class SchCsStorageConfig:
    """Configuration for storing SCH-CS results."""
    def __init__(self, output_dir: str = "data/schcs_results"):
        self.output_dir = output_dir
        self.save_binary_image = True      # Save thresholded binary image
        self.save_labeled_image = True     # Save labeled image (visualization)
        self.save_sr_regions_mask = True   # Save final SR regions mask
        self.save_metadata = True          # Save metadata as JSON
        self.save_regions_data = True      # Save region data as pickle (for debugging)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)


def save_schcs_results(
    results: Dict,
    pb: np.ndarray,
    image_name: str,
    storage_config: Optional[SchCsStorageConfig] = None
) -> Dict:
    if storage_config is None:
        storage_config = SchCsStorageConfig()
    
    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a run-specific subdirectory
    run_dir = os.path.join(storage_config.output_dir, f"{image_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    saved_paths = {}
    
    # Save binary image (p_th_b) - MOST IMPORTANT for Level Set initialization
    if storage_config.save_binary_image:
        binary_path = os.path.join(run_dir, f"{image_name}_binary.png")
        # binary_image is already 0/1, scale to 0-255 for saving
        cv2.imwrite(binary_path, results['binary_image'] * 255)
        saved_paths['binary_image'] = binary_path
        print(f"[Save] Binary image saved to: {binary_path}")
    
    # Save final SR regions mask (for visualization/debugging)
    if storage_config.save_sr_regions_mask:
        sr_mask = np.zeros_like(pb, dtype=np.uint8)
        for region in results['sr_regions']:
            sr_mask[region['mask']] = 255
        sr_mask_path = os.path.join(run_dir, f"{image_name}_sr_mask.png")
        cv2.imwrite(sr_mask_path, sr_mask)
        saved_paths['sr_mask'] = sr_mask_path
        print(f"[Save] SR mask saved to: {sr_mask_path}")
    
    # Save labeled image (for debugging)
    if storage_config.save_labeled_image:
        # Normalize labeled image for visualization
        labeled_max = results['labeled_image'].max()
        if labeled_max > 0:
            labeled_viz = (results['labeled_image'] * (255 / labeled_max)).astype(np.uint8)
        else:
            labeled_viz = results['labeled_image'].astype(np.uint8)
        labeled_path = os.path.join(run_dir, f"{image_name}_labeled.png")
        cv2.imwrite(labeled_path, labeled_viz)
        saved_paths['labeled_image'] = labeled_path
        print(f"[Save] Labeled image saved to: {labeled_path}")
    
    # Save region data as pickle (for detailed debugging)
    if storage_config.save_regions_data:
        regions_data = {
            'sr_regions': [],
            'all_regions': []
        }
        
        # Convert SR regions to serializable format
        for region in results['sr_regions']:
            regions_data['sr_regions'].append({
                'label': region['label'],
                'size': region['size'],
                'centroid': region['centroid'],
                'centroid_corrected': region.get('centroid_corrected', False),
                'mask_indices': np.argwhere(region['mask']).tolist()  # Store coordinates
            })
        
        # Convert all regions to serializable format
        for region in results['all_regions']:
            regions_data['all_regions'].append({
                'label': region['label'],
                'size': region['size'],
                'centroid': region['centroid'],
                'centroid_corrected': region.get('centroid_corrected', False)
            })
        
        regions_path = os.path.join(run_dir, f"{image_name}_regions.pkl")
        with open(regions_path, 'wb') as f:
            pickle.dump(regions_data, f)
        saved_paths['regions_data'] = regions_path
        print(f"[Save] Regions data saved to: {regions_path}")
    
    # Save metadata as JSON
    if storage_config.save_metadata:
        metadata = {
            'image_name': image_name,
            'timestamp': timestamp,
            'pb_shape': pb.shape,
            'pb_dtype': str(pb.dtype),
            'pb_min': int(pb.min()),
            'pb_max': int(pb.max()),
            'threshold_used': float(results['th']),
            'num_sr_regions': len(results['sr_regions']),
            'num_all_regions': len(results['all_regions']),
            'sr_region_details': [
                {
                    'label': r['label'],
                    'size': r['size'],
                    'centroid': r['centroid'],
                    'centroid_corrected': r.get('centroid_corrected', False)
                }
                for r in results['sr_regions']
            ],
            'all_region_details': [
                {
                    'label': r['label'],
                    'size': r['size'],
                    'centroid': r['centroid']
                }
                for r in results['all_regions']
            ],
            'saved_files': saved_paths
        }
        
        metadata_path = os.path.join(run_dir, f"{image_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        saved_paths['metadata'] = metadata_path
        print(f"[Save] Metadata saved to: {metadata_path}")
    
    # Add saved paths to results
    results['saved_paths'] = saved_paths
    results['run_dir'] = run_dir
    results['timestamp'] = timestamp
    results['image_name'] = image_name
    
    return results