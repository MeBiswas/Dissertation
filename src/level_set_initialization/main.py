# src/level_set_initialization/main.py

import os
import warnings
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

warnings.filterwarnings('ignore')

@dataclass
class PhiInitConfig:
    """Configuration parameters for Level Set Function initialization."""
    inside_value: float = 4.0
    outside_value: float = -4.0
    save_visualization: bool = True
    show_visualization: bool = True
    visualization_path: str = "phi_init_result.png"
    output_dir: str = "data/phi_initialized"
    save_phi_array: bool = True

# Initializes the Level Set Function φ from SCH-CS binary blobs.
class LevelSetInitializer:
    def __init__(self, config: Optional[PhiInitConfig] = None):
        self.config = config or PhiInitConfig()
        self._phi: Optional[np.ndarray] = None
        self._image_name: Optional[str] = None
        self._saved_filepath: Optional[str] = None
        
        # Create output directory if it doesn't exist
        if self.config.save_phi_array:
            os.makedirs(self.config.output_dir, exist_ok=True)
            print(f"[Init] Output directory ready: {self.config.output_dir}")
    
    # =========================================================================
    # Core Equation 17 Implementation
    # =========================================================================
    
    def initialize_phi(self, schcs_binary: np.ndarray) -> np.ndarray:
        # Normalize input to strict 0/1 binary
        p_th_b = schcs_binary.astype(np.float64)
        if p_th_b.max() > 1.0:
            p_th_b = p_th_b / 255.0
            print("[Init]  Binary image normalized from [0,255] → [0.0, 1.0]")

        # Verify it's actually binary
        unique_vals = np.unique(p_th_b)
        if not np.all(np.isin(unique_vals, [0.0, 1.0])):
            print(f"[Warn]  Binary image has unexpected values: {unique_vals}. "
                  "Thresholding at 0.5...")
            p_th_b = (p_th_b >= 0.5).astype(np.float64)

        # Equation 17
        #   φ = 4 * p_th_b  -  (1 - p_th_b) * 4
        #     = 4 * p_th_b  -  4 + 4 * p_th_b
        #     = 8 * p_th_b  -  4
        phi = self.config.inside_value * p_th_b - (1.0 - p_th_b) * abs(self.config.outside_value)

        # Sanity checks
        n_inside = np.sum(phi > 0)
        n_outside = np.sum(phi < 0)
        n_total = phi.size
        
        print(f"[Eq 17] φ initialized successfully.")
        print(f"        Shape      : {phi.shape}")
        print(f"        Inside SR  : {n_inside} pixels "
              f"({100*n_inside/n_total:.1f}%) → φ = +{self.config.inside_value}")
        print(f"        Outside SR : {n_outside} pixels "
              f"({100*n_outside/n_total:.1f}%) → φ = -{abs(self.config.outside_value)}")
        print(f"        φ unique values: {np.unique(phi)}")

        return phi
    
    # =========================================================================
    # Saving Methods
    # =========================================================================
    
    def _generate_filename(self, original_name: str, suffix: str = "phi") -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{original_name}_{suffix}_{timestamp}"
        return filename
    
    def save_phi_array(self, phi: np.ndarray, image_name: str) -> str:
        if not self.config.save_phi_array:
            print("[Save] Saving φ array disabled in config")
            return ""
        
        # Generate filename with timestamp
        base_name = self._generate_filename(image_name, "phi")
        filepath = os.path.join(self.config.output_dir, f"{base_name}.npy")
        
        # Save numpy array
        np.save(filepath, phi)
        self._saved_filepath = filepath
        print(f"[Save] φ array saved → {filepath}")
        
        return filepath
    
    def save_phi_visualization(
        self,
        preprocessed_img: np.ndarray,
        schcs_binary: np.ndarray,
        phi: np.ndarray,
        image_name: str
    ) -> str:
        if not self.config.save_visualization:
            return ""
        
        # Generate filename with timestamp
        base_name = self._generate_filename(image_name, "phi_viz")
        viz_path = os.path.join(self.config.output_dir, f"{base_name}.png")
        
        # Create visualization
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f"Equation 17: Level Set Function φ Initialization\nImage: {image_name}",
                     fontsize=13, fontweight='bold')

        # Panel 1: Preprocessed grayscale image
        axes[0].imshow(preprocessed_img, cmap='gray')
        axes[0].set_title("(1) Preprocessed TBI\n(input p_b)")
        axes[0].axis('off')

        # Panel 2: SCH-CS binary blobs
        axes[1].imshow(schcs_binary, cmap='gray')
        axes[1].set_title("(2) SCH-CS Binary Blobs\n(p_th_b — input to Eq 17)")
        axes[1].axis('off')

        # Panel 3: φ heatmap
        phi_img = axes[2].imshow(phi, cmap='RdBu_r',
                                  norm=mcolors.TwoSlopeNorm(vmin=-4, vcenter=0, vmax=4))
        axes[2].set_title("(3) Initialized φ (Eq 17)\nRed=+4 (SR) | Blue=−4 (outside)")
        axes[2].axis('off')
        plt.colorbar(phi_img, ax=axes[2], fraction=0.046, pad=0.04)

        # Panel 4: φ=0 contour overlaid on grayscale
        axes[3].imshow(preprocessed_img, cmap='gray')
        axes[3].contour(phi, levels=[0], colors=['red'], linewidths=[2])
        axes[3].set_title("(4) φ=0 Contour on TBI\n(initial contour — will evolve next)")
        axes[3].axis('off')

        plt.tight_layout()
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        print(f"[Save] Visualization saved → {viz_path}")
        
        if self.config.show_visualization:
            plt.show()
        else:
            plt.close()
        
        return viz_path
    
    # =========================================================================
    # Main Processing Pipeline
    # =========================================================================
    
    def process(
        self,
        schcs_binary: np.ndarray,
        preprocessed_img: np.ndarray,
        image_name: str,
        visualize: bool = True,
        save_results: bool = True
    ) -> np.ndarray:
        # Store image name
        self._image_name = image_name
        
        # Shape check
        if preprocessed_img.shape != schcs_binary.shape:
            raise ValueError(
                f"Shape mismatch! Preprocessed TBI: {preprocessed_img.shape}, "
                f"SCH-CS binary: {schcs_binary.shape}.\n"
                "Both must have the same dimensions."
            )
        
        print(f"\n[Process] Initializing φ for: {image_name}")
        print(f"          Input shape: {schcs_binary.shape}")
        
        # Compute φ using Equation 17
        phi = self.initialize_phi(schcs_binary)
        
        # Save φ array for next steps
        if save_results and self.config.save_phi_array:
            self.save_phi_array(phi, image_name)
        
        # Create and save visualization
        if visualize:
            self.save_phi_visualization(preprocessed_img, schcs_binary, phi, image_name)
        
        self._phi = phi
        return phi
    
    def process_from_arrays(
        self,
        schcs_binary: np.ndarray,
        preprocessed_img: np.ndarray,
        image_name: str,
        visualize: bool = True
    ) -> np.ndarray:
        return self.process(schcs_binary, preprocessed_img, image_name, visualize)
    
    def process_from_files(
        self,
        preprocessed_image_path: str,
        schcs_binary_path: str,
        image_name: Optional[str] = None,
        visualize: bool = True
    ) -> np.ndarray:
        # Load preprocessed TBI
        p_b = cv2.imread(preprocessed_image_path, cv2.IMREAD_GRAYSCALE)
        if p_b is None:
            raise FileNotFoundError(f"Could not load: {preprocessed_image_path}")
        print(f"[Load]  Preprocessed TBI shape  : {p_b.shape}")

        # Load SCH-CS binary blobs
        schcs_binary = cv2.imread(schcs_binary_path, cv2.IMREAD_GRAYSCALE)
        if schcs_binary is None:
            raise FileNotFoundError(f"Could not load: {schcs_binary_path}")
        print(f"[Load]  SCH-CS binary blob shape: {schcs_binary.shape}")
        
        # Extract image name if not provided
        if image_name is None:
            image_name = os.path.splitext(os.path.basename(preprocessed_image_path))[0]
        
        return self.process(schcs_binary, p_b, image_name, visualize)
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_results(self) -> Dict[str, Any]:
        return {
            'phi': self._phi,
            'image_name': self._image_name,
            'saved_filepath': self._saved_filepath
        }


# =============================================================================
# Convenience Functions (Backward Compatible)
# =============================================================================

def initialize_phi(schcs_binary: np.ndarray) -> np.ndarray:
    initializer = LevelSetInitializer()
    return initializer.initialize_phi(schcs_binary)


def visualize_phi_initialization(
    preprocessed_img: np.ndarray,
    schcs_binary: np.ndarray,
    phi: np.ndarray,
    save_path: str = "phi_init_result.png"
):
    config = PhiInitConfig(visualization_path=save_path)
    initializer = LevelSetInitializer(config)
    # Use a dummy image name for standalone function
    initializer.save_phi_visualization(preprocessed_img, schcs_binary, phi, "temp")


def run_phi_initialization(
    preprocessed_image_path: str,
    schcs_binary_path: str,
    save_output: bool = True,
    output_dir: str = "data/phi_initialized"
) -> np.ndarray:
    config = PhiInitConfig(
        save_visualization=save_output,
        show_visualization=False,
        output_dir=output_dir,
        save_phi_array=True
    )
    initializer = LevelSetInitializer(config)
    return initializer.process_from_files(preprocessed_image_path, schcs_binary_path)