"""
Backbone networks (ResNet-18/34, ViT-S) mapping x -> h.
Designed to support the Oracle-Free Evidential Unlearning framework.

Key Methodology Mappings:
- Implements the feature extractor f_theta: X -> R^d.
- The output latent representation h is passed to the EDL head g_psi to compute evidence e(x) (Eq. 1).
- The encoder parameters theta are updated via gradient descent on L_total (Eq. 24).
- Supports gradient flow for Feature-Space Boundary Repulsion (Eq. 16) and 
  Generalized Gauss-Newton (GGN) matrix approximation for LIAV (Eq. 17).
"""

import torch
import torch.nn as nn
from torchvision import models
import math
from typing import List, Iterator

class BaseEncoder(nn.Module):
    """
    Abstract base class for all feature extractors f_theta.
    Defines the standard interface for mapping input images x to latent representations h.
    """
    def __init__(self):
        super().__init__()
        self.feature_dim: int = 0
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Maps input image x to latent feature representation h.
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
        Returns:
            h (torch.Tensor): Latent feature representation of shape (B, feature_dim).
        """
        raise NotImplementedError("Subclasses must implement the forward method.")
        
    def get_parameters(self) -> Iterator[nn.Parameter]:
        """Returns an iterator over the encoder parameters theta."""
        return self.parameters()


# ==============================================================================
# 1. ResNet Encoders (ResNet-18, ResNet-34)
# ==============================================================================
class ResNetEncoder(BaseEncoder):
    """
    ResNet-based feature extractor.
    Modifies the standard torchvision ResNet to remove the final classification layer (fc),
    outputting the 512-dimensional pooled feature vector h.
    """
    def __init__(self, variant: str = 'resnet18', pretrained: bool = True):
        """
        Args:
            variant (str): 'resnet18' or 'resnet34'.
            pretrained (bool): Whether to initialize with ImageNet pre-trained weights.
        """
        super().__init__()
        
        if variant == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base_model = models.resnet18(weights=weights)
            self.feature_dim = 512
        elif variant == 'resnet34':
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            base_model = models.resnet34(weights=weights)
            self.feature_dim = 512
        else:
            raise ValueError(f"Unsupported ResNet variant: {variant}. Choose 'resnet18' or 'resnet34'.")
            
        # Extract layers up to avgpool, discarding the fc layer
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass mapping x -> h.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        h = torch.flatten(x, 1)
        return h


# ==============================================================================
# 2. Vision Transformer Encoder (ViT-Small)
# ==============================================================================
class PatchEmbedding(nn.Module):
    """Converts an image into a sequence of projected patches."""
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)             # Shape: (B, E, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2) # Shape: (B, N, E)
        return x

class TransformerBlock(nn.Module):
    """Standard Transformer Encoder Block with Multi-Head Self-Attention and MLP."""
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class ViTEncoder(BaseEncoder):
    """
    Vision Transformer Small (ViT-S) feature extractor.
    Outputs the CLS token representation as the latent feature vector h.
    Config: patch_size=16, embed_dim=384, depth=12, num_heads=6.
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, embed_dim: int = 384, 
                 depth: int = 12, num_heads: int = 6, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        self.feature_dim = embed_dim
        
        # Initialize weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass mapping x -> h.
        """
        B = x.shape[0]
        x = self.patch_embed(x) # Shape: (B, N, E)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # Shape: (B, 1+N, E)
        x = x + self.pos_embed
        x = self.dropout(x)
        
        x = self.blocks(x)
        x = self.norm(x)
        
        h = x[:, 0] # Extract CLS token representation
        return h


# ==============================================================================
# Factory Function for Reusability
# ==============================================================================
def get_encoder(encoder_name: str, pretrained: bool = True, **kwargs) -> BaseEncoder:
    """
    Factory function to instantiate the appropriate encoder network.
    
    Args:
        encoder_name (str): Name of the encoder ('resnet18', 'resnet34', 'vit_s').
        pretrained (bool): Whether to use pre-trained weights (applicable to ResNet).
        
    Returns:
        BaseEncoder: The instantiated encoder model.
    """
    encoder_name = encoder_name.lower()
    
    if encoder_name == 'resnet18':
        return ResNetEncoder(variant='resnet18', pretrained=pretrained)
    elif encoder_name == 'resnet34':
        return ResNetEncoder(variant='resnet34', pretrained=pretrained)
    elif encoder_name == 'vit_s':
        # ViT-Small default configuration
        return ViTEncoder(
            img_size=kwargs.get('img_size', 224),
            patch_size=kwargs.get('patch_size', 16),
            embed_dim=kwargs.get('embed_dim', 384),
            depth=kwargs.get('depth', 12),
            num_heads=kwargs.get('num_heads', 6),
            mlp_ratio=kwargs.get('mlp_ratio', 4.0),
            dropout=kwargs.get('dropout', 0.0)
        )
    else:
        raise ValueError(f"Unknown encoder: {encoder_name}. Choose from 'resnet18', 'resnet34', 'vit_s'.")