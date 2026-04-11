# src/chm_bias_correction/main.py

import os
import cv2
import warnings
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Dict, Any
from scipy.ndimage import uniform_filter

from src.utils import chm_corrected_results_path

warnings.filterwarnings('ignore')

# Configuration parameters for CHM correction.
@dataclass
class CHMConfig:
    window_size: int = 9
    order: int = 1
    eps: float = 1e-10
    
    # Visualization settings
    save_visualization: bool = False  # Default to False for production
    show_visualization: bool = True   # Separate flag for showing vs saving
    visualization_path: str = "chm_correction_result.png"
    
    # Output settings
    output_dir: str = chm_corrected_results_path
    save_corrected_image: bool = True

# Contraharmonic Mean (CHM) filter for bias field correction.
class CHMCorrector:
    def __init__(self, config: Optional[CHMConfig] = None):
        self.config = config or CHMConfig()
        self._p_bar: Optional[np.ndarray] = None
        self._chm_local: Optional[np.ndarray] = None
        self._nc: Optional[float] = None
        self._image_name: Optional[str] = None
        
        # Create output directory if it doesn't exist
        if self.config.save_corrected_image:
            os.makedirs(self.config.output_dir, exist_ok=True)
            print(f"[Init] Output directory ready: {self.config.output_dir}")
    
    def _generate_filename(self, base_name: str = "corrected") -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{base_name}_{timestamp}.png"
        return filename

    def _get_image_name_from_path(self, image_path: str) -> str:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        return base_name
    
    def save_corrected_image(self, p_bar: np.ndarray, base_name: Optional[str] = None) -> str:
        if not self.config.save_corrected_image:
            print("[Save] Saving disabled in config")
            return ""
        
        # Determine base name
        if base_name is None:
            base_name = self._image_name if self._image_name else "corrected"
        
        # Generate filename with timestamp
        filename = self._generate_filename(base_name)
        filepath = os.path.join(self.config.output_dir, filename)
        
        # Scale to 0-255 and convert to uint8 for saving
        p_bar_scaled = p_bar / (p_bar.max() + self.config.eps) * 255
        p_bar_uint8 = p_bar_scaled.astype(np.uint8)
        
        # Save image
        cv2.imwrite(filepath, p_bar_uint8)
        print(f"[Save] Corrected image saved → {filepath}")
        
        return filepath
    
    # =========================================================================
    # Core CHM Filter Implementations (Equations 8, 9, 10)
    # =========================================================================
    
    # Equation 8: Local CHM Filter.
    def compute_chm_local(self, image: np.ndarray) -> np.ndarray:
        p = image.astype(np.float64)
        w = self.config.window_size
        n = self.config.order
        
        # Compute sums over local window
        sum_num = uniform_filter(p ** (n + 1), size=w, mode='reflect') * (w ** 2)
        sum_den = uniform_filter(p ** n, size=w, mode='reflect') * (w ** 2)
        
        # Avoid division by zero
        sum_den = np.where(sum_den == 0, self.config.eps, sum_den)
        
        return sum_num / sum_den
    
    # Equation 9: Global Normalizing Constant Nc.
    def compute_nc(self, image: np.ndarray) -> float:
        p = image.astype(np.float64)
        n = self.config.order
        mask = p > 0
        
        numerator = np.sum(p[mask] ** (n + 1))
        denominator = np.sum(p[mask] ** n) + self.config.eps
        
        Nc = numerator / denominator
        print(f'[Eq 9]  Nc = {Nc:.4f}')
        
        return Nc
    
    # Equation 10: Corrected Image p_bar.
    def compute_p_bar(self, image: np.ndarray) -> np.ndarray:
        p = image.astype(np.float64)
        
        # Compute local CHM filter (Eq 8)
        chm_local = self.compute_chm_local(p)
        
        # Compute global normalizer (Eq 9)
        Nc = self.compute_nc(p)
        
        # Store for later use
        self._chm_local = chm_local
        self._nc = Nc
        
        # Apply correction (Eq 10)
        chm_safe = np.where(chm_local < self.config.eps, self.config.eps, chm_local)
        p_bar = (p * Nc) / chm_safe
        
        # Keep background pixels zero
        p_bar[p == 0] = 0.0
        
        print(f'[Eq 10] p_bar range: [{p_bar.min():.3f}, {p_bar.max():.3f}]')
        
        return p_bar
    
    # =========================================================================
    # Visualization Methods
    # =========================================================================
    
    def visualize_correction(
        self,
        pb: np.ndarray,
        p_bar: np.ndarray
    ) -> None:
        pb_float = pb.astype(np.float64)
        p_bar_float = p_bar.astype(np.float64)
        
        # Compute CHM for visualization (if not already computed)
        if self._chm_local is None:
            chm_local = self.compute_chm_local(pb_float)
        else:
            chm_local = self._chm_local
        
        # Create figure
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle("Equations 8–10: CHM Bias Field Correction", fontsize=13, fontweight='bold')
        
        # Panel 1: Input p_b
        axes[0].imshow(pb_float, cmap='hot')
        axes[0].set_title("Input p_b\n(Stage 1 output)")
        axes[0].axis('off')
        
        # Panel 2: CHM_W (bias field)
        axes[1].imshow(chm_local, cmap='hot')
        axes[1].set_title("Eq 8: CHM_W\n(local bias field)")
        axes[1].axis('off')
        
        # Panel 3: Corrected p_bar
        p_bar_disp = p_bar_float / (p_bar_float.max() + self.config.eps) * 255
        axes[2].imshow(p_bar_disp, cmap='hot')
        axes[2].set_title("Eq 10: p_bar\n(corrected)")
        axes[2].axis('off')
        
        # Panel 4: Intensity profile
        mid_row = pb_float.shape[0] // 2
        
        def normalize(x):
            return (x - x.min()) / (x.max() - x.min() + self.config.eps)
        
        axes[3].plot(normalize(pb_float[mid_row]), color='steelblue', lw=1.5, label='p_b (original)')
        axes[3].plot(normalize(p_bar_float[mid_row]), color='orangered', lw=1.5, label='p_bar (corrected)')
        
        axes[3].set_title("Intensity profile (mid row)\n(hill → plateau expected)")
        axes[3].set_xlabel("Pixel column")
        axes[3].set_ylabel("Normalized intensity")
        axes[3].legend(fontsize=8)
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.config.save_visualization:
            plt.savefig(self.config.visualization_path, dpi=150, bbox_inches='tight')
            print(f"[Save] Visualization saved → {self.config.visualization_path}")
        
        plt.show()
    
    # =========================================================================
    # Main Processing Pipeline
    # =========================================================================
    
    def process(
        self,
        image: np.ndarray,
        image_name: Optional[str] = None,
        visualize: bool = True,
        save_image: bool = True
    ) -> np.ndarray:
        # Store image name for saving
        self._image_name = image_name
        
        print(f'[CHM] Input pb: shape={image.shape}, dtype={image.dtype}, '
              f'range=[{image.min()},{image.max()}]')
        
        # Compute corrected image
        p_bar = self.compute_p_bar(image)
        
        print(f'[CHM] p_bar : shape={p_bar.shape}, dtype={p_bar.dtype}')
        assert p_bar.shape == image.shape, 'Shape mismatch!'
        assert p_bar.dtype == np.float64, 'p_bar must be float64 for level set'
        print('[CHM] ✓ p_bar ready.')
        
        # Save corrected image
        if save_image and self.config.save_corrected_image:
            saved_path = self.save_corrected_image(p_bar, image_name)
            print(f'[CHM] Corrected image saved to: {saved_path}')
        
        # Visualize results
        if visualize:
            self.visualize_correction(image, p_bar)
        
        self._p_bar = p_bar
        return p_bar
    
    def process_from_file(
        self,
        image_path: str,
        visualize: bool = True,
        save_image: bool = True
    ) -> np.ndarray:
        p_b = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if p_b is None:
            raise FileNotFoundError(
                f"Could not load: {image_path}\n"
                "Make sure this is the OUTPUT of preprocessing_tbi.py"
            )
        
        # Extract image name from path
        image_name = self._get_image_name_from_path(image_path)
        
        print(f"[Load]  Image shape: {p_b.shape}, dtype: {p_b.dtype}")
        print(f"[Load]  Image name: {image_name}")
        print(f"Intensity range: [{p_b.min()}, {p_b.max()}]")
        
        return self.process(p_b, image_name=image_name, visualize=visualize, save_image=save_image)
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_results(self) -> Dict[str, Any]:
        return {
            'p_bar': self._p_bar,
            'chm_local': self._chm_local,
            'nc': self._nc
        }

# =============================================================================
# Convenience Functions (Backward Compatible)
# =============================================================================

def compute_chm_local(image: np.ndarray, window_size: int = 9, order: int = 1) -> np.ndarray:
    """Equation 8: Local CHM Filter (standalone function)."""
    corrector = CHMCorrector(CHMConfig(window_size=window_size, order=order))
    return corrector.compute_chm_local(image)


def compute_nc(image: np.ndarray, order: int = 1) -> float:
    """Equation 9: Global Normalizing Constant (standalone function)."""
    corrector = CHMCorrector(CHMConfig(order=order))
    return corrector.compute_nc(image)


def compute_p_bar(image: np.ndarray, window_size: int = 9, order: int = 1) -> np.ndarray:
    """Equation 10: Corrected Image p_bar (standalone function)."""
    corrector = CHMCorrector(CHMConfig(window_size=window_size, order=order))
    return corrector.compute_p_bar(image)


def visualize_chm_correction(
    pb: np.ndarray,
    p_bar: np.ndarray,
    compute_chm_local_func,
    window_size: int = 9,
    order: int = 1,
    save_path: str = "chm_correction_result.png"
):
    """Visualization function (standalone, backward compatible)."""
    config = CHMConfig(window_size=window_size, order=order, visualization_path=save_path)
    corrector = CHMCorrector(config)
    corrector.visualize_correction(pb, p_bar)


def run_chm_correction(
    preprocessed_image_path: str,
    window_size: int = 9,
    order: int = 1,
    save_output: bool = True,
    output_dir: str = chm_corrected_results_path
) -> np.ndarray:
    config = CHMConfig(
        window_size=window_size,
        order=order,
        save_visualization=save_output,
        output_dir=output_dir,
        save_corrected_image=True
    )
    corrector = CHMCorrector(config)
    return corrector.process_from_file(preprocessed_image_path, visualize=save_output)
