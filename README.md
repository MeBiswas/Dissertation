# Dissertation - Breast Cancer Detection using Deep Learning (Fuzzy)

This repository contains an implementation of the method proposed in:

"S. Pramanik et al., Suspicious-region segmentation from breast thermogram using DLPE-based level set method,
IEEE Transactions on Medical Imaging, 2018."

## Project Structure

### Directory Overview
- `data/` : Raw and processed thermogram datasets
- `src/`  : Core implementation organized by pipeline stages
- `experiments/` : Reproducible experiment configurations
- `results/` : Segmentation and classification outputs
- `notebooks/` : Interactive exploration and debugging
- `docs/` : Notes, equations, and dissertation mappings

### **Pipeline Implementation**

#### **PART 0 — Imports & Config**
- `src/utils/experiment_config.py` : Configuration management
- `src/utils/paths.py` : Dataset and output path handling
- `src/utils/helper.py` : Utility functions

#### **PART 1 — Pre-processing Pipeline** *(Paper Section II-A)*
Located in `src/preprocessing/`

- **Step 1.1** Remove color scale bar  
- **Step 1.2** Extract blue channel (grayscale)  
  → `image_processing.py : extract_blue_channel()`
- **Step 1.3** Remove background (Otsu + largest component)  
  → `otsu_thresholding.py`, `image_processing.py`
- **Step 1.4** Gray-level reconstruction  
  → `gray_level_reconstruction.py : morphological_reconstruction()`
- **Step 1.5** Visualise → Figure 2 of paper  
  → `eda.py : plot_preprocessing_stages()`

#### **PART 2 — SCH-CS Pipeline** *(Paper Section II-B)*
Located in `src/sch_cs/`

- **Step 2.1** Compute histogram of p_b  
  → `index.py : compute_histogram()`
- **Step 2.2** Compute rho (Count Threshold) — Equation 1  
  → `count_threshold.py : compute_count_threshold()`
- **Step 2.3** Build Array A & compute t* — Equation 2  
  → `index.py : compute_threshold_array()`
- **Step 2.4** Compute final threshold th — Equation 3  
  → `final_threshold.py : compute_final_threshold()`
- **Step 2.5** Threshold p_b → binary image  
  → `initial_threshold.py : apply_threshold()`
- **Step 2.6** Label connected regions  
  → `connected_regions.py : label_connected_components()`
- **Step 2.7** Compute weighted centroids — Equation 4  
  → `centroid_computation.py : compute_centroids()`
- **Step 2.8** Bounding-box centroid fix — Algorithm 1  
  → `bounding_box.py : fix_bounding_boxes()`
- **Step 2.9** CS isolation loop  
  → `cs_isolation.py : isolate_suspicious_regions()`
- **Step 2.10** Visualise → Figure 3 of paper  
  → `visualization.py : plot_sch_cs_pipeline()`

#### **PART 3 — Run & Inspect Results**
- `src/run_experiment.py` : Main experiment orchestration
- `experiments/` : Experiment-specific configurations and parameter sets
- `notebooks/` : Analysis and result inspection notebooks

## Status
🚧 Initial repository structure (no experiments executed yet)

## Author
<Abhipriyo Biswas>
MSc Data Science
