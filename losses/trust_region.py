"""
L2 trust-region penalty λ4||θ - θ0||^2 (Eq. 24').
Designed to support the Oracle-Free Evidential Unlearning framework.

Key Methodology Mappings:
- Eq. 24': L_trust_region = lambda_4 * ||theta_T - theta_0||_2^2
- Bounds the encoder drift to ensure that the Probabilistic Forgetting Certification (PFC) 
  theoretical guarantees (Eqs. 27-28) remain mathematically valid for real D_f samples 
  evaluated at inference time.
- Prevents catastrophic forgetting of the retained set's underlying feature distribution 
  by restricting the optimization trajectory to a local neighborhood of the original weights.
"""

import torch
import torch.nn as nn

class TrustRegionPenalty(nn.Module):
    """
    Computes the L2 trust-region penalty to bound encoder drift.
    This penalty is added to the total training objective (Eq. 24) to ensure that the 
    encoder parameters do not drift too far from the original pre-trained weights.
    """
    def __init__(self, original_encoder: nn.Module, lambda_4: float = 0.01):
        """
        Args:
            original_encoder (nn.Module): The original, unmodified encoder f_theta_0.
            lambda_4 (float): Weighting factor lambda_4 for the trust-region penalty.
        """
        super(TrustRegionPenalty, self).__init__()
        self.lambda_4 = lambda_4
        
        # Store the original parameters as buffers so they automatically move with the 
        # module to GPU/CPU via .to(device), but are excluded from optimizer updates.
        self.num_params = 0
        with torch.no_grad():
            for i, param in enumerate(original_encoder.parameters()):
                self.register_buffer(f'orig_param_{i}', param.clone().detach())
                self.num_params += 1

    def forward(self, current_encoder: nn.Module) -> torch.Tensor:
        """
        Computes the squared L2 distance between current and original encoder parameters.
        Eq. 24': L_trust_region = lambda_4 * ||theta_T - theta_0||_2^2
        
        Args:
            current_encoder (nn.Module): The current encoder f_theta_T.
            
        Returns:
            torch.Tensor: Scalar trust-region penalty loss.
        """
        # Initialize drift accumulator on the same device as the current encoder
        device = next(current_encoder.parameters()).device
        drift_sq = torch.tensor(0.0, device=device)
        
        # Iterate through all parameters and accumulate the squared L2 difference
        for i, current_param in enumerate(current_encoder.parameters()):
            orig_param = getattr(self, f'orig_param_{i}')
            diff = current_param - orig_param
            drift_sq = drift_sq + torch.sum(diff ** 2)
            
        # Apply the weighting factor lambda_4
        loss = self.lambda_4 * drift_sq
        return loss