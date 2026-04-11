# utils/helper.py
import os
import cv2
import json
import pickle
import numpy as np

from typing import Dict

# ─────────────────────────────────────────────────────────────────────────────
# Helper: pretty divider for console output
# ─────────────────────────────────────────────────────────────────────────────
def section(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# PRE-PROCESSING: FUNCTION TO LOAD PREPROCESSED RESULTS FOR NEXT STAGES
# ─────────────────────────────────────────────────────────────────────────────
def load_preprocessing_results(run_dir: str) -> Dict:
    # Load metadata
    metadata_path = None
    for f in os.listdir(run_dir):
        if f.endswith('_metadata.json'):
            metadata_path = os.path.join(run_dir, f)
            break
    
    if metadata_path is None:
        raise FileNotFoundError(f"No metadata found in {run_dir}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load images
    results = {
        'pb': cv2.imread(metadata['saved_files']['pb'], cv2.IMREAD_GRAYSCALE),
        'grayscale': cv2.imread(metadata['saved_files']['grayscale'], cv2.IMREAD_GRAYSCALE),
        'bg_removed': cv2.imread(metadata['saved_files']['bg_removed'], cv2.IMREAD_GRAYSCALE),
        'original_color': cv2.imread(metadata['saved_files']['original_color'], cv2.IMREAD_COLOR),
        'image_name': metadata['image_name'],
        'base_name': metadata['base_name'],
        'run_dir': run_dir,
        'saved_paths': metadata['saved_files']
    }
    
    print(f"[Load] Loaded preprocessing results for: {metadata['base_name']}")
    print(f"pb shape: {results['pb'].shape}")
    
    return results

# =============================================================================
# LOAD SCH-CS RESULTS FOR NEXT STAGES
# =============================================================================
def load_schcs_results(run_dir: str) -> Dict:
    # Find metadata file
    metadata_path = None
    for f in os.listdir(run_dir):
        if f.endswith('_metadata.json'):
            metadata_path = os.path.join(run_dir, f)
            break
    
    if metadata_path is None:
        raise FileNotFoundError(f"No metadata found in {run_dir}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    results = {}
    
    # Load binary image (most important for Level Set)
    if 'binary_image' in metadata['saved_files']:
        binary_image = cv2.imread(metadata['saved_files']['binary_image'], cv2.IMREAD_GRAYSCALE)
        results['binary_image'] = (binary_image > 127).astype(np.uint8)  # Ensure binary
        print(f"[Load] Binary image loaded: {results['binary_image'].shape}")
    
    # Load SR mask if available
    if 'sr_mask' in metadata['saved_files']:
        results['sr_mask'] = cv2.imread(metadata['saved_files']['sr_mask'], cv2.IMREAD_GRAYSCALE)
        print(f"[Load] SR mask loaded: {results['sr_mask'].shape}")
    
    # Load regions data if available
    if 'regions_data' in metadata['saved_files']:
        with open(metadata['saved_files']['regions_data'], 'rb') as f:
            results['regions_data'] = pickle.load(f)
        print(f"[Load] Regions data loaded")
    
    results['metadata'] = metadata
    results['run_dir'] = run_dir
    results['image_name'] = metadata['image_name']
    results['threshold_used'] = metadata['threshold_used']
    results['num_sr_regions'] = metadata['num_sr_regions']
    
    print(f"[Load] Loaded SCH-CS results for: {metadata['image_name']}")
    print(f"       Number of SR regions: {metadata['num_sr_regions']}")
    
    return results