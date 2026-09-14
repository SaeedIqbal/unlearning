"""
L_forget (Eq. 25) and its exact gradient via reparameterization (Eq. 26).
Designed to support the Oracle-Free Evidential Unlearning framework.

Key Methodology Mappings:
- Eq. 25: L_forget = - E_{z ~ N(mu_f, sigma^2 I_d)} [u(z)]
- Eq. 26: Exact gradient of L_forget with respect to EDL head parameters psi via the reparameterization trick.
"""

import torch
import torch.nn as nn

class EvidentialForgettingLoss(nn.Module):
    """
    Computes the Evidential Forgetting Loss (L_forget) to explicitly drive the model's 
    epistemic vacuity to its theoretical maximum over the forgetting manifold's neighborhood.
    """
    def __init__(self, manifold_stats, num_classes: int, feature_dim: int, 
                 sigma: float = 0.1, num_mc_samples: int = 4, device: str = 'cuda'):
        """
        Args:
            manifold_stats: ForgettingManifoldStatistics object containing the empirical mean mu_f (Eq. 10).
            num_classes (int): Number of classes K.
            feature_dim (int): Dimension of the latent features d.
            sigma (float): Standard deviation for the Gaussian neighborhood N(mu_f, sigma^2 I_d).
            num_mc_samples (int): Number of Monte Carlo samples to estimate the expectation in Eq. 25.
            device (str): Computation device ('cuda' or 'cpu').
        """
        super(EvidentialForgettingLoss, self).__init__()
        self.manifold_stats = manifold_stats
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.sigma = sigma
        self.num_mc_samples = num_mc_samples
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

    def forward(self, edl_head: nn.Module) -> torch.Tensor:
        """
        Computes the Evidential Forgetting Loss (Eq. 25).
        L_forget = - E_{z ~ N(mu_f, sigma^2 I_d)} [u(z)]
        
        The exact gradient with respect to the EDL head parameters psi is computed 
        automatically via PyTorch's autograd using the reparameterization trick (Eq. 26):
        z = mu_f + sigma * epsilon, where epsilon ~ N(0, I_d).
        
        Args:
            edl_head (nn.Module): The Evidential Deep Learning head g_psi.
            
        Returns:
            torch.Tensor: Scalar L_forget loss.
        """
        # Retrieve the empirical mean of the forgetting manifold (Eq. 10)
        mu_f = self.manifold_stats.mu_f.to(self.device)
        
        # Eq. 26: Reparameterization trick for differentiable sampling
        # epsilon ~ N(0, I_d)
        # Note: torch.randn is used here strictly for the Monte Carlo estimation 
        # of the expectation in Eq. 25, not for synthetic dataset generation.
        epsilon = torch.randn(self.num_mc_samples, self.feature_dim, device=self.device)
        
        # z = mu_f + sigma * epsilon
        # mu_f is detached from the computation graph to ensure gradients only flow to psi
        z = mu_f.detach().unsqueeze(0) + self.sigma * epsilon
        
        # Forward pass through the EDL head to compute evidence e(z) (Eq. 1)
        evidence = edl_head(z)
        
        # Compute Dirichlet concentration parameters alpha(z) = e(z) + 1 (Eq. 2)
        alpha = evidence + 1.0
        
        # Compute total Dirichlet strength S(z) = sum(alpha_k(z))
        S = alpha.sum(dim=1)
        
        # Compute epistemic vacuity u(z) = K / S(z) (Eq. 3)
        vacuity = self.num_classes / S
        
        # Eq. 25: L_forget = - E[u(z)]
        # The negative sign ensures that minimizing the loss maximizes the vacuity
        loss = -vacuity.mean()
        
        return loss