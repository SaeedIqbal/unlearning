"""
Linear Centered Kernel Alignment for deep representation extraction audit.
Designed to evaluate the structural privacy of the Oracle-Free Evidential Unlearning framework.

Key Methodology Mappings:
- Implements the Linear Centered Kernel Alignment (CKA) metric to audit deep representation extraction.
- Eq. (CKA): CKA(H_orig, H_unl) = ||H_unl^T H_orig||_F^2 / (||H_orig^T H_orig||_F * ||H_unl^T H_unl||_F)
- H_orig, H_unl are centered feature matrices of the original and unlearned models on D_f.
- A high CKA score indicates that the unlearned model retains deep architectural traces of the forgotten data.
- The FSBR module explicitly minimizes this metric by scrambling hidden layer representations along the principal axes of the forgetting manifold.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

class LinearCKAAuditor:
    """
    Audits deep representation extraction using Linear Centered Kernel Alignment (CKA).
    Measures the residual similarity between the original and unlearned models on the forgetting set D_f.
    """
    def __init__(self, model_orig: nn.Module, model_unl: nn.Module, device: torch.device):
        """
        Args:
            model_orig (nn.Module): The original, fully trained model.
            model_unl (nn.Module): The unlearned model.
            device (torch.device): The computation device.
        """
        self.model_orig = model_orig
        self.model_unl = model_unl
        self.device = device
        
    def extract_features(self, model: nn.Module, dataloader: DataLoader) -> np.ndarray:
        """
        Extracts the latent feature representations h = f_theta(x) for all samples in the dataloader.
        
        Args:
            model (nn.Module): The model to extract features from.
            dataloader (DataLoader): DataLoader for the target dataset (e.g., D_f).
            
        Returns:
            np.ndarray: Feature matrix H of shape (N, d).
        """
        model.eval()
        features = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Handle datasets returning (img, target, ...) or (img, target, demo_attrs, ...)
                imgs = batch[0].to(self.device)
                
                # Extract latent features using the unified model's method
                h = model.get_latent_features(imgs)
                features.append(h.cpu().numpy())
                
        return np.concatenate(features, axis=0)

    def center_features(self, H: np.ndarray) -> np.ndarray:
        """
        Centers the feature matrix H by subtracting the mean of each column (feature dimension).
        H_centered = H - 1/N * 1 1^T H
        
        Args:
            H (np.ndarray): Feature matrix of shape (N, d).
            
        Returns:
            np.ndarray: Centered feature matrix of shape (N, d).
        """
        mean = np.mean(H, axis=0, keepdims=True)
        H_centered = H - mean
        return H_centered

    def compute_cka(self, H1: np.ndarray, H2: np.ndarray) -> float:
        """
        Computes the Linear Centered Kernel Alignment (CKA) between two centered feature matrices.
        CKA(H1, H2) = ||H2^T H1||_F^2 / (||H1^T H1||_F * ||H2^T H2||_F)
        
        Args:
            H1 (np.ndarray): First centered feature matrix (N, d1).
            H2 (np.ndarray): Second centered feature matrix (N, d2).
            
        Returns:
            float: CKA score in [0, 1].
        """
        # Cross-covariance matrix (unnormalized)
        cross_cov = H2.T @ H1  # (d2, d1)
        
        # Frobenius norm squared of cross-covariance
        numerator = np.linalg.norm(cross_cov, ord='fro') ** 2
        
        # Auto-covariance matrices
        auto_cov1 = H1.T @ H1  # (d1, d1)
        auto_cov2 = H2.T @ H2  # (d2, d2)
        
        # Frobenius norms of auto-covariances
        denom1 = np.linalg.norm(auto_cov1, ord='fro')
        denom2 = np.linalg.norm(auto_cov2, ord='fro')
        
        denominator = denom1 * denom2
        
        if denominator < 1e-12:
            return 0.0
            
        cka_score = numerator / denominator
        return float(cka_score)

    def evaluate(self, dataloader: DataLoader) -> float:
        """
        Evaluates the CKA score between the original and unlearned models on the given dataset.
        
        Args:
            dataloader (DataLoader): DataLoader for the target dataset (e.g., D_f).
            
        Returns:
            float: CKA score.
        """
        # Extract features from both models
        H_orig = self.extract_features(self.model_orig, dataloader)
        H_unl = self.extract_features(self.model_unl, dataloader)
        
        # Center the feature matrices
        H_orig_centered = self.center_features(H_orig)
        H_unl_centered = self.center_features(H_unl)
        
        # Compute and return CKA
        cka_score = self.compute_cka(H_orig_centered, H_unl_centered)
        return cka_score