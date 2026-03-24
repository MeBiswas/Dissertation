# utils/experiment_config.py

from .paths import base_path, bcd_dataset, dmr_ir_o

config_1 = {
    "dataset_path": base_path + bcd_dataset['sick'],
    # "dataset_path": base_path + dmr_ir_o,
    
    #processing mode
    "process_all": False,
    
    # used iff process_all=False
    "image_index": 95,
    
    # Cropping settings
    "enable_cropping": True,
    "crop_neck_percent": 0.26,      # Crop 18% from top (neck region)
    "crop_stomach_percent": 0.04,   # Crop 12% from bottom (stomach region)
    "crop_armpit_percent": 0.10,    # Crop 22% from sides (armpit regions)

    # visualization
    "show_visualizations": True,
    
    # saving results
    "save_results": False,
    "output_dir": "outputs"
}