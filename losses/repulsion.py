"""
L_FSBR with Mahalanobis norm and squared hinge (Eq. 15).
Designed to support the Oracle-Free Evidential Unlearning framework.

Key Methodology Mappings:
- Eq. 15: L_FSBR = E[ max(0, gamma - ||h(x) - z_adv*||_{Sigma_f^{-1}})^2 ]
- Computes the Mahalanobis distance using the Cholesky decomposition of the regularized covariance Sigma_f (Eq. 10).
- Applies the squared hinge loss to push the retained edge features h(x) away from the adversarial centroid z_adv*.
"""

import torch
import torch.nn as nn

class FSBRMarginLoss(nn.Module):
    """
    Computes the Feature-Space Boundary Repulsion (FSBR) loss.
    This loss explicitly shapes the decision boundary by repelling the latent features 
    of the ambiguous retained set D_r^e away from the synthesized adversarial centroid z_adv*.
    """
    def __init__(self, manifold_stats, margin: float = 1.0):
        """
        Args:
            manifold_stats: ForgettingManifoldStatistics object containing the regularized covariance Sigma_f (Eq. 10).
            margin (float): Margin gamma for the squared hinge loss.
        """
        super(FSBRMarginLoss, self).__init__()
        self.manifold_stats = manifold_stats
        self.margin = margin

    def compute_mahalanobis_distance(self, h_r_edge: torch.Tensor, z_adv: torch.Tensor) -> torch.Tensor:
        """
        Computes the Mahalanobis distance between retained edge features and the adversarial centroid.
        d_M(h, z_adv) = sqrt( (h - z_adv)^T Sigma_f^{-1} (h - z_adv) )
        
        Args:
            h_r_edge (torch.Tensor): Latent features of the edge retained set D_r^e (B, d).
            z_adv (torch.Tensor): Synthesized adversarial centroid z_adv* (1, d) or (d,).
            
        Returns:
            torch.Tensor: Mahalanobis distances (B,).
        """
        device = h_r_edge.device
        Sigma_f = self.manifold_stats.Sigma_f.to(device)
        
        if z_adv.dim() == 1:
            z_adv = z_adv.unsqueeze(0)
            
        # Compute difference between retained edge features and adversarial centroid
        diff = h_r_edge - z_adv # (B, d)
        
        # Compute Mahalanobis distance using Cholesky decomposition for numerical stability
        # Sigma_f = L L^T => Sigma_f^{-1} = L^{-T} L^{-1}
        # d_M^2 = diff^T L^{-T} L^{-1} diff = || L^{-1} diff^T ||_2^2
        L = torch.linalg.cholesky(Sigma_f)
        
        # Solve L y = diff^T for y => y = L^{-1} diff^T
        # diff is (B, d), diff.t() is (d, B)
        y = torch.linalg.solve_triangular(L, diff.t(), upper=False) # (d, B)
        
        # Squared Mahalanobis distance
        dist_sq = (y ** 2).sum(dim=0) # (B,)
        
        # Add small epsilon for numerical stability before sqrt
        dist = torch.sqrt(dist_sq + 1e-8)
        return dist

    def forward(self, h_r_edge: torch.Tensor, z_adv: torch.Tensor) -> torch.Tensor:
        """
        Computes the FSBR loss (Eq. 15).
        L_FSBR = E[ max(0, gamma - d_M(h(x), z_adv*))^2 ]
        
        Args:
            h_r_edge (torch.Tensor): Latent features of the edge retained set D_r^e (B, d).
            z_adv (torch.Tensor): Synthesized adversarial centroid z_adv* (1, d) or (d,).
            
        Returns:
            torch.Tensor: Scalar FSBR loss.
        """
        # Compute Mahalanobis distance
        dist = self.compute_mahalanobis_distance(h_r_edge, z_adv)
        
        # Eq. 15: Squared hinge loss
        # Penalizes retained features that fall within the margin gamma of the adversarial centroid
        hinge_loss = torch.clamp(self.margin - dist, min=0.0) ** 2
        
        # Expectation over the batch
        loss = hinge_loss.mean()
        
        return loss