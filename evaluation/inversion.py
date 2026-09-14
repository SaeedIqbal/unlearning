"""
inversion.py
Trained Feature-Space Inversion Audit (decoder training and success rate).
Designed to evaluate the structural privacy of the Oracle-Free Evidential Unlearning framework.

Key Methodology Mappings:
- Implements the Feature-Space Model Inversion attack defined in the Threat Model (Section 3.3).
- Eq. (Inversion Risk): R_inv = min_phi E_{x ~ D_f} [ ||x - G_phi(f_theta_unl(x))||_2^2 ]
- The adversary trains a decoder G_phi on the retained set D_r (proxy) to learn the inverse mapping h -> x.
- The audit then evaluates the reconstruction quality on the forgotten set D_f.
- A low reconstruction error (high PSNR) indicates that the unlearned model still retains invertible information 
  about the forgotten data, violating the non-reconstructive privacy guarantee.
- The FSBR module aims to maximize this error by scrambling the latent features.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from typing import Tuple, Dict

class FeatureDecoder(nn.Module):
    """
    Adversarial Decoder Network G_phi.
    Maps latent feature representations h back to the pixel space x.
    Architecture: Linear projection -> Reshape -> Stack of Transposed Convolutions.
    """
    def __init__(self, feature_dim: int, img_channels: int, img_size: int):
        """
        Args:
            feature_dim (int): Dimension of the latent features d.
            img_channels (int): Number of image channels (e.g., 3 for RGB).
            img_size (int): Target spatial dimension (H or W) of the image.
        """
        super(FeatureDecoder, self).__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        
        # Base spatial dimension for the transposed convolution stack
        self.base_size = 4
        self.init_channels = 256
        
        # Initial fully connected layer to project latent vector to a 3D tensor
        self.fc = nn.Linear(feature_dim, self.init_channels * self.base_size * self.base_size)
        
        # Transposed convolution stack to upsample to the target image size
        # This stack is designed to reach 64x64. For larger images, interpolation is used.
        self.decoder_blocks = nn.Sequential(
            # Block 1: 4x4 -> 8x8
            nn.ConvTranspose2d(self.init_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Block 2: 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 3: 16x16 -> 32x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Block 4: 32x32 -> 64x64
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # Final Output Layer: 64x64 -> img_channels
            nn.ConvTranspose2d(16, img_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid() # Output pixel values in [0, 1]
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass mapping h -> x.
        
        Args:
            h (torch.Tensor): Latent features of shape (B, d).
            
        Returns:
            torch.Tensor: Reconstructed images of shape (B, C, H, W).
        """
        # Project to 3D tensor
        x = self.fc(h)
        x = x.view(-1, self.init_channels, self.base_size, self.base_size)
        
        # Upsample via transposed convolutions
        x = self.decoder_blocks(x)
        
        # Adaptive interpolation to match the exact target image size if necessary
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = torch.nn.functional.interpolate(
                x, size=(self.img_size, self.img_size), 
                mode='bilinear', align_corners=False
            )
            
        return x


class FeatureSpaceInversionAudit:
    """
    Orchestrates the Trained Feature-Space Inversion Audit.
    1. Trains the decoder G_phi on the retained set D_r (proxy data).
    2. Evaluates the reconstruction of the forgotten set D_f using the unlearned model's features.
    """
    def __init__(self, unlearned_model: nn.Module, feature_dim: int, img_shape: Tuple[int, int, int], 
                 device: torch.device, lr: float = 1e-3):
        """
        Args:
            unlearned_model (nn.Module): The unlearned model f_theta to be audited.
            feature_dim (int): Dimension of the latent space d.
            img_shape (Tuple): Shape of the images (C, H, W).
            device (torch.device): Computation device.
            lr (float): Learning rate for the decoder optimizer.
        """
        self.unlearned_model = unlearned_model
        self.device = device
        self.feature_dim = feature_dim
        self.img_channels, self.img_h, self.img_w = img_shape
        
        # Initialize the adversary's decoder G_phi
        self.decoder = FeatureDecoder(
            feature_dim=feature_dim, 
            img_channels=self.img_channels, 
            img_size=max(self.img_h, self.img_w)
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.decoder.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_decoder(self, proxy_dataloader: DataLoader, epochs: int = 20) -> None:
        """
        Trains the decoder G_phi on the proxy dataset (Retained Set D_r).
        The adversary learns the mapping h -> x using available non-forgotten data.
        
        Args:
            proxy_dataloader (DataLoader): DataLoader for the retained set D_r.
            epochs (int): Number of training epochs for the decoder.
        """
        self.unlearned_model.eval() # Freeze the unlearned model
        self.decoder.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in proxy_dataloader:
                # Extract images (handling different dataset return formats)
                imgs = batch[0].to(self.device)
                
                # 1. Extract latent features h = f_theta(x) using the unlearned model
                with torch.no_grad():
                    h = self.unlearned_model.get_latent_features(imgs)
                
                # 2. Decoder attempts to reconstruct x from h
                recon_imgs = self.decoder(h)
                
                # Ensure spatial dimensions match for loss computation
                if recon_imgs.shape[2:] != imgs.shape[2:]:
                    recon_imgs = torch.nn.functional.interpolate(
                        recon_imgs, size=imgs.shape[2:], mode='bilinear', align_corners=False
                    )
                
                # 3. Compute reconstruction loss: ||x - G_phi(h)||^2
                loss = self.criterion(recon_imgs, imgs)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()

    def invert_features(self, target_dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs the inversion on the target dataset (Forgotten Set D_f).
        
        Args:
            target_dataloader (DataLoader): DataLoader for the forgotten set D_f.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Original images and Reconstructed images.
        """
        self.unlearned_model.eval()
        self.decoder.eval()
        
        all_originals = []
        all_recons = []
        
        with torch.no_grad():
            for batch in target_dataloader:
                imgs = batch[0].to(self.device)
                
                # Extract features from the unlearned model
                h = self.unlearned_model.get_latent_features(imgs)
                
                # Reconstruct using the trained decoder
                recon_imgs = self.decoder(h)
                
                if recon_imgs.shape[2:] != imgs.shape[2:]:
                    recon_imgs = torch.nn.functional.interpolate(
                        recon_imgs, size=imgs.shape[2:], mode='bilinear', align_corners=False
                    )
                
                all_originals.append(imgs.cpu().numpy())
                all_recons.append(recon_imgs.cpu().numpy())
                
        return np.concatenate(all_originals, axis=0), np.concatenate(all_recons, axis=0)

    def compute_metrics(self, originals: np.ndarray, recons: np.ndarray) -> Dict[str, float]:
        """
        Computes inversion quality metrics: MSE and PSNR.
        
        Args:
            originals (np.ndarray): Ground truth images (N, C, H, W).
            recons (np.ndarray): Reconstructed images (N, C, H, W).
            
        Returns:
            Dict[str, float]: Dictionary containing 'MSE' and 'PSNR'.
        """
        # Mean Squared Error
        mse = np.mean((originals - recons) ** 2)
        
        # Peak Signal-to-Noise Ratio (assuming pixel values in [0, 1])
        max_pixel = 1.0
        if mse < 1e-10:
            psnr = 100.0
        else:
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
            
        return {'MSE': float(mse), 'PSNR': float(psnr)}

    def evaluate(self, proxy_dataloader: DataLoader, target_dataloader: DataLoader, 
                 epochs: int = 20, mse_threshold: float = 0.05) -> Dict[str, float]:
        """
        Full audit pipeline: Train on proxy, Evaluate on target.
        
        Args:
            proxy_dataloader (DataLoader): DataLoader for D_r (training the decoder).
            target_dataloader (DataLoader): DataLoader for D_f (evaluation).
            epochs (int): Training epochs for the decoder.
            mse_threshold (float): Threshold for defining a "successful" inversion.
            
        Returns:
            Dict[str, float]: Audit results including MSE, PSNR, and Inversion Success Rate.
        """
        # 1. Train the adversary's decoder on the retained set
        self.train_decoder(proxy_dataloader, epochs)
        
        # 2. Invert the forgotten set
        originals, recons = self.invert_features(target_dataloader)
        
        # 3. Compute global metrics
        metrics = self.compute_metrics(originals, recons)
        
        # 4. Compute Inversion Success Rate
        # A sample is considered "successfully inverted" if its individual MSE is below the threshold
        per_sample_mse = np.mean((originals - recons) ** 2, axis=(1, 2, 3))
        success_rate = np.mean(per_sample_mse < mse_threshold) * 100.0
        
        metrics['Inversion_Success_Rate'] = float(success_rate)
        return metrics