"""
Unified model computing Dirichlet moments, vacuity, and variance (Eqs. 2-5).
Designed to support the Oracle-Free Evidential Unlearning framework.

Key Methodology Mappings:
- Unifies the feature extractor f_theta (Encoder) and the evidential head g_psi (EDL Head).
- Forward pass maps x -> h -> e(x) -> alpha(x) -> {u(x), p(x), Var[p(x)]}.
- Eq. 2: alpha_k(x) = e_k(x) + 1
- Eq. 3: u(x) = K / S(x), where S(x) = sum_{k=1}^K alpha_k(x)
- Eq. 4: p_k(x) = alpha_k(x) / S(x)
- Eq. 5: Var[p_k] = p_k(x)(1 - p_k(x)) / (S(x) + 1)
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any

# Importing from the previously defined modules in the models/ directory
from encoder import get_encoder
from edl_head import EDLHead


class OracleFreeEvidentialModel(nn.Module):
    """
    Unified Oracle-Free Evidential Unlearning Model.
    Combines the backbone encoder f_theta and the Evidential Deep Learning head g_psi.
    """
    def __init__(self, encoder_name: str, num_classes: int, pretrained: bool = True, 
                 hidden_dims: list = None, dropout: float = 0.1, **encoder_kwargs):
        """
        Args:
            encoder_name (str): Name of the backbone encoder ('resnet18', 'resnet34', 'vit_s').
            num_classes (int): Number of classes K.
            pretrained (bool): Whether to initialize the encoder with pre-trained weights.
            hidden_dims (list): List of hidden layer dimensions for the EDL head.
            dropout (float): Dropout rate for the EDL head.
            **encoder_kwargs: Additional keyword arguments for the encoder (e.g., img_size).
        """
        super(OracleFreeEvidentialModel, self).__init__()
        
        self.num_classes = num_classes
        
        # Initialize the feature extractor f_theta (Encoder)
        self.encoder = get_encoder(encoder_name, pretrained=pretrained, **encoder_kwargs)
        
        # Initialize the evidential head g_psi (EDL Head)
        self.edl_head = EDLHead(
            feature_dim=self.encoder.feature_dim, 
            num_classes=num_classes, 
            hidden_dims=hidden_dims, 
            dropout=dropout
        )
        
    def get_latent_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts latent feature representation h = f_theta(x).
        
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
            
        Returns:
            h (torch.Tensor): Latent features of shape (B, d).
        """
        return self.encoder(x)

    def get_dirichlet_parameters(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the Dirichlet concentration parameters alpha(x).
        Eq. 2: alpha_k(x) = e_k(x) + 1
        
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
            
        Returns:
            alpha (torch.Tensor): Concentration parameters of shape (B, K).
        """
        h = self.get_latent_features(x)
        return self.edl_head.compute_dirichlet_parameters(h)

    def get_total_strength(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Computes the total Dirichlet strength S(x).
        S(x) = sum_{k=1}^K alpha_k(x)
        
        Args:
            alpha (torch.Tensor): Concentration parameters of shape (B, K).
            
        Returns:
            S (torch.Tensor): Total strength of shape (B,).
        """
        return alpha.sum(dim=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass computing evidence, vacuity, expected probability, and variance.
        
        Maps x -> h -> e(x) -> {alpha, S, u, p, Var}.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
            
        Returns:
            evidence (torch.Tensor): Evidence vector e(x) of shape (B, K).
            vacuity (torch.Tensor): Epistemic vacuity u(x) of shape (B,). (Eq. 3)
            p_mean (torch.Tensor): Expected probability p_k(x) of shape (B, K). (Eq. 4)
            p_var (torch.Tensor): Predictive variance Var[p_k] of shape (B, K). (Eq. 5)
        """
        # 1. Extract latent features h = f_theta(x)
        h = self.get_latent_features(x)
        
        # 2. Compute evidence e(x) = Softplus(g_psi(h)) (Eq. 1) and all moments
        evidence, vacuity, p_mean, p_var = self.edl_head.get_evidence_and_moments(h)
        
        return evidence, vacuity, p_mean, p_var

    def compute_pfc_bound_components(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Computes all components required for the Probabilistic Forgetting Certification (PFC) bound.
        Useful for the PFC module to evaluate the theoretical upper bounds (Eqs. 27-28).
        
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
            
        Returns:
            dict: Dictionary containing 'alpha', 'S', 'vacuity', 'p_mean', 'p_var'.
        """
        evidence, vacuity, p_mean, p_var = self.forward(x)
        alpha = evidence + 1.0
        S = self.get_total_strength(alpha)
        
        return {
            'evidence': evidence,
            'alpha': alpha,
            'S': S,
            'vacuity': vacuity,
            'p_mean': p_mean,
            'p_var': p_var
        }


# ==============================================================================
# Factory Function for Reusability
# ==============================================================================
def get_full_model(encoder_name: str, num_classes: int, pretrained: bool = True, 
                   hidden_dims: list = None, dropout: float = 0.1, **encoder_kwargs) -> OracleFreeEvidentialModel:
    """
    Factory function to instantiate the unified Oracle-Free Evidential Model.
    
    Args:
        encoder_name (str): Name of the backbone encoder.
        num_classes (int): Number of classes K.
        pretrained (bool): Whether to use pre-trained weights for the encoder.
        hidden_dims (list): Hidden dimensions for the EDL head.
        dropout (float): Dropout rate.
        **encoder_kwargs: Additional arguments for the encoder.
        
    Returns:
        OracleFreeEvidentialModel: The instantiated unified model.
    """
    return OracleFreeEvidentialModel(
        encoder_name=encoder_name,
        num_classes=num_classes,
        pretrained=pretrained,
        hidden_dims=hidden_dims,
        dropout=dropout,
        **encoder_kwargs
    )