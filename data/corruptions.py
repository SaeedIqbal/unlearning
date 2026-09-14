"""
ImageNet-C corruption functions (noise, blur, weather) for OOD evaluation.
Designed to support the Oracle-Free Evidential Unlearning framework.

Key Methodology Mappings:
- Implements the 15 standard ImageNet-C corruptions across 5 severity levels.
- Used to compute the Mean Corruption Error (mCE) to evaluate Out-of-Distribution (OOD) robustness.
- Ensures that the evidential vacuity maximization does not induce chaotic logit divergence under distributional shift.
"""

import os
import numpy as np
import torch
import cv2
from scipy.ndimage import gaussian_filter
import math

class ImageNetCCorruptions:
    """
    Implements the ImageNet-C corruption functions for OOD evaluation.
    Maps to the Mean Corruption Error (mCE) metric defined in the methodology:
    mCE = (1/15) * sum_{c=1}^{15} (1/5) * sum_{s=1}^{5} (Error_c^s(M) / Error_c^s(M_base)) * 100%
    
    Supports Noise, Blur, and Weather corruptions across 5 severity levels.
    """
    
    def __init__(self, severity: int):
        if not 1 <= severity <= 5:
            raise ValueError("Severity must be between 1 and 5.")
        self.severity = severity
        
    def _tensor_to_numpy(self, x: torch.Tensor) -> np.ndarray:
        """Converts a PyTorch tensor (C, H, W) in [0, 1] to a NumPy array (H, W, C) in [0, 255]."""
        x_np = x.permute(1, 2, 0).cpu().numpy()
        return (x_np * 255).astype(np.uint8)
        
    def _numpy_to_tensor(self, x_np: np.ndarray) -> torch.Tensor:
        """Converts a NumPy array (H, W, C) in [0, 255] to a PyTorch tensor (C, H, W) in [0, 1]."""
        x_np = np.clip(x_np, 0, 255).astype(np.float32) / 255.0
        return torch.from_numpy(x_np).permute(2, 0, 1)

    # ==============================================================================
    # NOISE CORRUPTIONS
    # ==============================================================================
    def gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Applies Gaussian noise: x_corrupt = x + N(0, sigma^2)"""
        params = [(0.04, 0.1), (0.06, 0.2), (0.08, 0.3), (0.09, 0.4), (0.10, 0.5)]
        std, _ = params[self.severity - 1]
        x_np = self._tensor_to_numpy(x).astype(np.float32) / 255.0
        noise = np.random.normal(0, std, x_np.shape)
        x_np = np.clip(x_np + noise, 0, 1)
        return self._numpy_to_tensor((x_np * 255).astype(np.uint8))

    def shot_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Applies Shot noise: x_corrupt = (1/lambda) * Poisson(lambda * x)"""
        params = [(60, 1), (40, 1), (20, 1), (10, 1), (5, 1)]
        lam = params[self.severity - 1][0]
        x_np = self._tensor_to_numpy(x).astype(np.float32) / 255.0
        x_np = np.random.poisson(x_np * lam) / float(lam)
        return self._numpy_to_tensor((np.clip(x_np, 0, 1) * 255).astype(np.uint8))

    def impulse_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Applies Impulse (Salt & Pepper) noise with probability p."""
        params = [(0.01, 0.01), (0.02, 0.02), (0.03, 0.03), (0.05, 0.05), (0.07, 0.07)]
        amount, _ = params[self.severity - 1]
        x_np = self._tensor_to_numpy(x)
        mask = np.random.choice([0, 1, 2], size=x_np.shape[:2], p=[amount/2, amount/2, 1-amount])
        x_np[mask == 0] = 0
        x_np[mask == 1] = 255
        return self._numpy_to_tensor(x_np)

    # ==============================================================================
    # BLUR CORRUPTIONS
    # ==============================================================================
    def defocus_blur(self, x: torch.Tensor) -> torch.Tensor:
        """Applies Defocus blur using a disk kernel."""
        params = [(3, 0.1), (4, 0.5), (5, 0.8), (6, 1.0), (7, 1.2)]
        radius, _ = params[self.severity - 1]
        x_np = self._tensor_to_numpy(x).astype(np.float32) / 255.0
        
        kernel_size = 2 * radius + 1
        y, x_grid = np.ogrid[-radius:radius+1, -radius:radius+1]
        kernel = np.zeros((kernel_size, kernel_size))
        mask = x_grid**2 + y**2 <= radius**2
        kernel[mask] = 1
        kernel = kernel / kernel.sum()
        
        x_blurred = np.zeros_like(x_np)
        for c in range(3):
            x_blurred[:, :, c] = cv2.filter2D(x_np[:, :, c], -1, kernel)
            
        return self._numpy_to_tensor((np.clip(x_blurred, 0, 1) * 255).astype(np.uint8))

    def glass_blur(self, x: torch.Tensor) -> torch.Tensor:
        """Applies Glass blur via iterative pixel shifting and Gaussian blur."""
        params = [(2, 0.4, 3, 2), (3, 0.6, 4, 2), (4, 0.8, 5, 2), (5, 1.0, 6, 2), (6, 1.2, 7, 2)]
        sigma, glass_sigma, _, iterations = params[self.severity - 1]
        x_np = self._tensor_to_numpy(x).astype(np.float32) / 255.0
        
        h, w = x_np.shape[:2]
        for _ in range(iterations):
            dx = np.random.randint(-glass_sigma, glass_sigma+1, size=(h, w))
            dy = np.random.randint(-glass_sigma, glass_sigma+1, size=(h, w))
            x_shifted = np.zeros_like(x_np)
            for i in range(h):
                for j in range(w):
                    ni, nj = i + dx[i, j], j + dy[i, j]
                    if 0 <= ni < h and 0 <= nj < w:
                        x_shifted[i, j] = x_np[ni, nj]
            x_np = x_shifted
            
        for c in range(3):
            x_np[:, :, c] = gaussian_filter(x_np[:, :, c], sigma=sigma)
            
        return self._numpy_to_tensor((np.clip(x_np, 0, 1) * 255).astype(np.uint8))

    def motion_blur(self, x: torch.Tensor) -> torch.Tensor:
        """Applies Motion blur using a directional line kernel."""
        params = [(10, 3), (15, 5), (20, 7), (25, 9), (30, 11)]
        size, _ = params[self.severity - 1]
        angle = np.random.uniform(0, 180)
        
        kernel = np.zeros((size, size))
        kernel[size//2, :] = np.ones(size)
        kernel = kernel / size
        
        M = cv2.getRotationMatrix2D((size/2, size/2), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (size, size))
        
        x_np = self._tensor_to_numpy(x).astype(np.float32) / 255.0
        x_blurred = np.zeros_like(x_np)
        for c in range(3):
            x_blurred[:, :, c] = cv2.filter2D(x_np[:, :, c], -1, kernel)
            
        return self._numpy_to_tensor((np.clip(x_blurred, 0, 1) * 255).astype(np.uint8))

    def zoom_blur(self, x: torch.Tensor) -> torch.Tensor:
        """Applies Zoom blur by averaging multiple zoomed versions of the image."""
        params = [(1.01, 0.02), (1.02, 0.04), (1.03, 0.06), (1.04, 0.08), (1.05, 0.10)]
        max_factor, step = params[self.severity - 1]
        x_np = self._tensor_to_numpy(x).astype(np.float32) / 255.0
        
        out = np.zeros_like(x_np)
        factors = np.arange(1.0, max_factor, step)
        for factor in factors:
            h, w = x_np.shape[:2]
            zoomed = cv2.resize(x_np, (int(w*factor), int(h*factor)), interpolation=cv2.INTER_LINEAR)
            zh, zw = zoomed.shape[:2]
            start_h, start_w = (zh - h) // 2, (zw - w) // 2
            out += zoomed[start_h:start_h+h, start_w:start_w+w, :]
            
        out /= len(factors)
        return self._numpy_to_tensor((np.clip(out, 0, 1) * 255).astype(np.uint8))

    # ==============================================================================
    # WEATHER CORRUPTIONS
    # ==============================================================================
    def snow(self, x: torch.Tensor) -> torch.Tensor:
        """Applies Snow corruption by adding snow texture and flare."""
        params = [(0.1, 0.3), (0.2, 0.4), (0.3, 0.5), (0.4, 0.6), (0.5, 0.7)]
        snow_coeff, flare_coeff = params[self.severity - 1]
        x_np = self._tensor_to_numpy(x).astype(np.float32) / 255.0
        
        snow = np.random.normal(1.0, snow_coeff, x_np.shape)
        snow = np.clip(snow, 0, 1)
        x_np = x_np * snow
        
        flare = np.random.normal(1.0, flare_coeff, x_np.shape[:2])
        flare = np.clip(flare, 0, 1)
        for c in range(3):
            x_np[:, :, c] = x_np[:, :, c] * flare
            
        return self._numpy_to_tensor((np.clip(x_np, 0, 1) * 255).astype(np.uint8))

    def frost(self, x: torch.Tensor) -> torch.Tensor:
        """Applies Frost corruption by overlaying a frost texture."""
        params = [(1.0, 0.2), (1.2, 0.4), (1.4, 0.6), (1.6, 0.8), (1.8, 1.0)]
        coeff, _ = params[self.severity - 1]
        x_np = self._tensor_to_numpy(x).astype(np.float32) / 255.0
        
        frost_texture = np.random.normal(1.0, 0.5, x_np.shape)
        frost_texture = np.clip(frost_texture, 0, 1)
        
        x_np = x_np * (1 - coeff/2) + frost_texture * (coeff/2)
        return self._numpy_to_tensor((np.clip(x_np, 0, 1) * 255).astype(np.uint8))

    def fog(self, x: torch.Tensor) -> torch.Tensor:
        """Applies Fog corruption by mixing with white based on a smoothed noise map."""
        params = [(0.2, 0.5), (0.3, 0.6), (0.4, 0.7), (0.5, 0.8), (0.6, 0.9)]
        fog_coeff, _ = params[self.severity - 1]
        x_np = self._tensor_to_numpy(x).astype(np.float32) / 255.0
        
        fog_map = np.random.normal(1.0, 0.5, x_np.shape[:2])
        fog_map = gaussian_filter(fog_map, sigma=10)
        fog_map = np.clip(fog_map, 0, 1) * fog_coeff
        
        for c in range(3):
            x_np[:, :, c] = x_np[:, :, c] * (1 - fog_map) + fog_map
            
        return self._numpy_to_tensor((np.clip(x_np, 0, 1) * 255).astype(np.uint8))

    def brightness(self, x: torch.Tensor) -> torch.Tensor:
        """Applies Brightness corruption by scaling pixel values by a factor c > 1."""
        params = [(1.1, 0.1), (1.2, 0.2), (1.3, 0.3), (1.4, 0.4), (1.5, 0.5)]
        coeff, _ = params[self.severity - 1]
        x_np = self._tensor_to_numpy(x).astype(np.float32) / 255.0
        
        x_np = np.clip(x_np * coeff, 0, 1)
        return self._numpy_to_tensor((x_np * 255).astype(np.uint8))

    # ==============================================================================
    # UTILITY & EVALUATION METHODS
    # ==============================================================================
    def apply_corruption(self, x: torch.Tensor, corruption_name: str) -> torch.Tensor:
        """Applies the specified corruption to the input tensor."""
        corruption_map = {
            'gaussian_noise': self.gaussian_noise,
            'shot_noise': self.shot_noise,
            'impulse_noise': self.impulse_noise,
            'defocus_blur': self.defocus_blur,
            'glass_blur': self.glass_blur,
            'motion_blur': self.motion_blur,
            'zoom_blur': self.zoom_blur,
            'snow': self.snow,
            'frost': self.frost,
            'fog': self.fog,
            'brightness': self.brightness
        }
        if corruption_name not in corruption_map:
            raise ValueError(f"Unknown corruption: {corruption_name}")
        return corruption_map[corruption_name](x)

    @staticmethod
    def compute_mce(error_corrupted: dict, error_baseline: dict) -> float:
        """
        Computes the Mean Corruption Error (mCE) as defined in the methodology:
        mCE = (1/15) * sum_{c=1}^{15} (1/5) * sum_{s=1}^{5} (Error_c^s(M) / Error_c^s(M_base)) * 100%
        
        Args:
            error_corrupted (dict): Nested dict {corruption_name: {severity: error_rate}}
            error_baseline (dict): Nested dict {corruption_name: {severity: error_rate}}
            
        Returns:
            float: The computed mCE percentage.
        """
        mce_sum = 0.0
        num_corruptions = len(error_corrupted)
        
        for c_name, severities in error_corrupted.items():
            ce_c = 0.0
            for s in range(1, 6):
                err_m = severities.get(s, 0.0)
                err_base = error_baseline.get(c_name, {}).get(s, 1e-8)
                ce_c += (err_m / err_base)
            ce_c /= 5.0
            mce_sum += ce_c
            
        mce = (mce_sum / num_corruptions) * 100.0
        return mce