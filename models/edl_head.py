"""
Evidential Deep Learning head g_ψ mapping h → e(x) via Softplus (Eq. 1).
Designed to support the Oracle-Free Evidential Unlearning framework.

Key Methodology Mappings:
- Implements the evidence mapping g_ψ: R^d -> R^K_{>=0}.
- Eq. 1: e(x) = Softplus(g_ψ(h)) = log(1 + exp(g_ψ(h))).
- Computes Dirichlet concentration parameters α_k(x) = e_k(x) + 1 (Context of Eq. 2).
- Computes total Dirichlet strength S(x) = sum(α_k(x)).
- Computes epistemic vacuity u(x) = K / S(x) (Eq. 3).
- Computes expected probability p_k(x) = α_k(x) / S(x) (Eq. 4).
- Computes predictive variance Var[p_k] = p_k(x)(1 - p_k(x)) / (S(x) + 1) (Eq. 5).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class EDLHead(nn.Module):
    """
    Evidential Deep Learning (EDL) head that maps latent features h to non-negative evidence e(x).
    """
    def __init__(self, feature_dim: int, num_classes: int, hidden_dims: list = None, dropout: float = 0.1):
        """
        Args:
            feature_dim (int): Dimension of the input latent features h (d).
            num_classes (int): Number of classes K.
            hidden_dims (list): List of hidden layer dimensions. Default is [256, 128].
            dropout (float): Dropout rate for regularization.
        """
        super(EDLHead, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        if hidden_dims is None:
            hidden_dims = [256, 128]
            
        layers = []
        in_dim = feature_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
            
        # Final layer mapping to K logits
        layers.append(nn.Linear(in_dim, num_classes))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Maps latent features h to evidence vector e(x) via Softplus activation (Eq. 1).
        e(x) = log(1 + exp(g_ψ(h)))
        
        Args:
            h (torch.Tensor): Latent features of shape (B, d).
            
        Returns:
            evidence (torch.Tensor): Non-negative evidence vector of shape (B, K).
        """
        logits = self.mlp(h)
        # Eq. 1: Softplus activation to ensure e(x) >= 0
        evidence = F.softplus(logits)
        return evidence

    def compute_dirichlet_parameters(self, h: torch.Tensor) -> torch.Tensor:
        """
        Computes the Dirichlet concentration parameters α(x).
        α_k(x) = e_k(x) + 1 (Context of Eq. 2)
        
        Args:
            h (torch.Tensor): Latent features of shape (B, d).
            
        Returns:
            alpha (torch.Tensor): Concentration parameters of shape (B, K).
        """
        evidence = self.forward(h)
        alpha = evidence + 1.0
        return alpha

    def compute_vacuity(self, h: torch.Tensor) -> torch.Tensor:
        """
        Computes the epistemic vacuity u(x).
        u(x) = K / S(x), where S(x) = sum_{k=1}^K α_k(x) (Eq. 3)
        
        Args:
            h (torch.Tensor): Latent features of shape (B, d).
            
        Returns:
            vacuity (torch.Tensor): Epistemic vacuity of shape (B,).
        """
        alpha = self.compute_dirichlet_parameters(h)
        S = alpha.sum(dim=1)
        vacuity = self.num_classes / S
        return vacuity

    def compute_predictive_moments(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the expected probability (mean) and variance of the predictive distribution.
        p_k(x) = α_k(x) / S(x) (Eq. 4)
        Var[p_k] = p_k(x)(1 - p_k(x)) / (S(x) + 1) (Eq. 5)
        
        Args:
            h (torch.Tensor): Latent features of shape (B, d).
            
        Returns:
            p_mean (torch.Tensor): Expected probability of shape (B, K).
            p_var (torch.Tensor): Predictive variance of shape (B, K).
        """
        alpha = self.compute_dirichlet_parameters(h)
        S = alpha.sum(dim=1, keepdim=True)
        
        # Eq. 4
        p_mean = alpha / S
        
        # Eq. 5
        p_var = (p_mean * (1.0 - p_mean)) / (S + 1.0)
        
        return p_mean, p_var

    def get_evidence_and_moments(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Comprehensive forward pass returning evidence, vacuity, and predictive moments.
        Highly optimized for Evidential Uncertainty Partitioning (EUP, Eq. 9) to avoid redundant forward passes.
        
        Args:
            h (torch.Tensor): Latent features of shape (B, d).
            
        Returns:
            evidence (torch.Tensor): Evidence vector e(x) (B, K).
            vacuity (torch.Tensor): Epistemic vacuity u(x) (B,).
            p_mean (torch.Tensor): Expected probability (B, K).
            p_var (torch.Tensor): Predictive variance (B, K).
        """
        evidence = self.forward(h)
        alpha = evidence + 1.0
        S = alpha.sum(dim=1)
        
        # Eq. 3
        vacuity = self.num_classes / S
        
        S_keepdim = S.unsqueeze(1)
        # Eq. 4
        p_mean = alpha / S_keepdim
        # Eq. 5
        p_var = (p_mean * (1.0 - p_mean)) / (S_keepdim + 1.0)
        
        return evidence, vacuity, p_mean, p_var