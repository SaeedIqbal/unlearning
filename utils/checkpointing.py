"""
checkpointing.py
Model saving/loading, EMA updates for μ_f^(t) and Σ_f^(t) (Eq. 10').
Designed to support the Oracle-Free Evidential Unlearning framework.

Key Methodology Mappings:
- Eq. 10': EMA updates for the forgetting manifold statistics:
           μ_f^(t) = γ μ_f^(t-1) + (1-γ) μ_hat_f^(t)
           Σ_f^(t) = γ Σ_f^(t-1) + (1-γ) Σ_hat_f^(t) + λ_reg I
- Handles saving and loading of the Oracle-Free Evidential Model, optimizer states, 
  and EMA statistics to ensure reproducibility and support sequential unlearning rounds.
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple

class ForgettingManifoldEMA:
    """
    Manages the Exponential Moving Average (EMA) updates for the forgetting manifold 
    statistics μ_f and Σ_f (Eq. 10').
    """
    def __init__(self, feature_dim: int, momentum: float = 0.9, lambda_reg: float = 1e-4):
        """
        Args:
            feature_dim (int): Dimension of the latent features d.
            momentum (float): EMA momentum γ.
            lambda_reg (float): Tikhonov regularization λ_reg for covariance stability.
        """
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.lambda_reg = lambda_reg
        
        # Initialize statistics
        self.mu_f = torch.zeros(feature_dim)
        self.Sigma_f = torch.eye(feature_dim) * lambda_reg
        self.is_initialized = False
        
    def update(self, batch_mu: torch.Tensor, batch_cov: torch.Tensor) -> None:
        """
        Updates the empirical mean and covariance of the forgetting set D_f using EMA.
        Eq. 10': 
        μ_f^(t) = γ μ_f^(t-1) + (1-γ) μ_hat_f^(t)
        Σ_f^(t) = γ Σ_f^(t-1) + (1-γ) Σ_hat_f^(t) + λ_reg I
        
        Args:
            batch_mu (torch.Tensor): Batch empirical mean μ_hat_f^(t) of shape (d,).
            batch_cov (torch.Tensor): Batch empirical covariance Σ_hat_f^(t) of shape (d, d).
        """
        with torch.no_grad():
            if not self.is_initialized:
                self.mu_f = batch_mu.clone()
                self.Sigma_f = batch_cov + self.lambda_reg * torch.eye(self.feature_dim)
                self.is_initialized = True
            else:
                self.mu_f = self.momentum * self.mu_f + (1.0 - self.momentum) * batch_mu
                self.Sigma_f = self.momentum * self.Sigma_f + (1.0 - self.momentum) * batch_cov + self.lambda_reg * torch.eye(self.feature_dim)

    def get_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the current EMA statistics.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: μ_f and Σ_f.
        """
        return self.mu_f, self.Sigma_f

    def state_dict(self) -> Dict[str, Any]:
        """
        Returns the state dictionary for checkpointing.
        """
        return {
            'mu_f': self.mu_f.cpu(),
            'Sigma_f': self.Sigma_f.cpu(),
            'is_initialized': self.is_initialized,
            'feature_dim': self.feature_dim,
            'momentum': self.momentum,
            'lambda_reg': self.lambda_reg
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Loads the state dictionary from a checkpoint.
        """
        self.mu_f = state_dict['mu_f'].clone()
        self.Sigma_f = state_dict['Sigma_f'].clone()
        self.is_initialized = state_dict['is_initialized']
        self.feature_dim = state_dict['feature_dim']
        self.momentum = state_dict['momentum']
        self.lambda_reg = state_dict['lambda_reg']


class CheckpointManager:
    """
    Handles saving and loading of the Oracle-Free Evidential Model, optimizer states, 
    and EMA statistics.
    """
    def __init__(self, save_dir: str):
        """
        Args:
            save_dir (str): Directory to save checkpoints.
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def save_checkpoint(self, epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer, 
                        ema_updater: ForgettingManifoldEMA, metrics: Optional[Dict[str, float]] = None, 
                        filename: str = 'checkpoint.pth') -> None:
        """
        Saves the model, optimizer, EMA stats, and metrics to a checkpoint file.
        
        Args:
            epoch (int): Current epoch or unlearning round.
            model (nn.Module): The Oracle-Free Evidential Model.
            optimizer (torch.optim.Optimizer): The optimizer.
            ema_updater (ForgettingManifoldEMA): The EMA updater for μ_f and Σ_f.
            metrics (Optional[Dict[str, float]]): Dictionary of current metrics (e.g., CKA, MIA).
            filename (str): Name of the checkpoint file.
        """
        checkpoint_path = os.path.join(self.save_dir, filename)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'ema_state_dict': ema_updater.state_dict(),
            'metrics': metrics if metrics is not None else {}
        }
        
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, 
                        ema_updater: Optional[ForgettingManifoldEMA] = None, 
                        filename: str = 'checkpoint.pth', device: str = 'cpu') -> Tuple[int, Dict[str, float]]:
        """
        Loads the model, optimizer, EMA stats, and metrics from a checkpoint file.
        
        Args:
            model (nn.Module): The model to load weights into.
            optimizer (Optional[torch.optim.Optimizer]): The optimizer to load state into.
            ema_updater (Optional[ForgettingManifoldEMA]): The EMA updater to load state into.
            filename (str): Name of the checkpoint file.
            device (str): Device to map the loaded tensors to.
            
        Returns:
            Tuple[int, Dict[str, float]]: The epoch/round and the metrics dictionary.
        """
        checkpoint_path = os.path.join(self.save_dir, filename)
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        # Load EMA state if provided
        if ema_updater is not None and 'ema_state_dict' in checkpoint:
            ema_updater.load_state_dict(checkpoint['ema_state_dict'])
            # Move EMA stats to the correct device
            ema_updater.mu_f = ema_updater.mu_f.to(device)
            ema_updater.Sigma_f = ema_updater.Sigma_f.to(device)
            
        epoch = checkpoint.get('epoch', 0)
        metrics = checkpoint.get('metrics', {})
        
        return epoch, metrics