# utils/experiment_config.py

from .paths import base_path, bcd_dataset

config_1 = {
    "dataset_path": base_path + bcd_dataset['sick'],
    
    #processing mode
    "process_all": False,
    
    # used iff process_all=False
    "image_index": 0,
    
    # visualization
    "show_visualizations": True,
    
    # saving results
    "save_results": False,
    "output_dir": "outputs"
}