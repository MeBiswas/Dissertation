# utils/experiment_config.py

from .paths import base_path, bcd_dataset, dmr_ir_o

config_1 = {
    "dataset_path": base_path + bcd_dataset['sick'],
    # "dataset_path": base_path + dmr_ir_o,
    
    #processing mode
    "process_all": False,
    
    # used iff process_all=False
    "image_index": 95,
    
    # visualization
    "show_visualizations": True,
    
    # saving results
    "save_results": False,
    "output_dir": "outputs"
}