"""
Feature-Space Boundary Repulsion: Computes μ_f, Σ_f (Eq. 10), synthesizes z_adv* (Eqs. 11-14), 
and repulsion loss (Eqs. 15-16).
Designed to support the Oracle-Free Evidential Unlearning framework.

Key Methodology Mappings:
- Eq. 10: Empirical mean μ_f and regularized covariance Σ_f of the forgetting manifold.
- Eq. 11: Exact gradient of epistemic vacuity ∇_z u(z) via Jacobian of evidence mapping.
- Eq. 12: Constrained maximization of vacuity within Mahalanobis ellipsoidal neighborhood E.
- Eq. 13: Projected gradient ascent update rule for z_adv.
- Eq. 14: Projection operator Π_E(v) as a strictly convex quadratic program.
- Eq. 15: Repulsion loss L_FSBR with squared hinge and Mahalanobis norm.
- Eq. 16: Gradient of L_FSBR with respect to encoder parameters θ (handled via PyTorch autograd).
"""

import torch
import torch.nn as nn
from typing import Tuple

class ForgettingManifoldStatistics:
    """
    Computes and maintains the zero-order summary statistics of the forgetting manifold D_f.
    Implements Eq. 10 using Exponential Moving Average (EMA) for stability.
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
        
        self.mu_f = torch.zeros(feature_dim)
        self.Sigma_f = torch.eye(feature_dim) * lambda_reg
        self.is_initialized = False
        
    def update(self, h_f: torch.Tensor) -> None:
        """
        Updates the empirical mean and covariance of the forgetting set D_f.
        Eq. 10: μ_f^(t) = γ μ_f^(t-1) + (1-γ) E[h_f]
                Σ_f^(t) = γ Σ_f^(t-1) + (1-γ) E[(h_f - μ_f)(h_f - μ_f)^T] + λ_reg I
        
        Args:
            h_f (torch.Tensor): Latent features of the forgetting set (N, d).
        """
        with torch.no_grad():
            batch_mu = h_f.mean(dim=0)
            diff = h_f - batch_mu.unsqueeze(0)
            batch_cov = torch.matmul(diff.t(), diff) / h_f.size(0)
            
            if not self.is_initialized:
                self.mu_f = batch_mu.clone()
                self.Sigma_f = batch_cov + self.lambda_reg * torch.eye(self.feature_dim)
                self.is_initialized = True
            else:
                self.mu_f = self.momentum * self.mu_f + (1.0 - self.momentum) * batch_mu
                self.Sigma_f = self.momentum * self.Sigma_f + (1.0 - self.momentum) * batch_cov + self.lambda_reg * torch.eye(self.feature_dim)

    def get_mahalanobis_distance_sq(self, z: torch.Tensor) -> torch.Tensor:
        """
        Computes the squared Mahalanobis distance of points z from the mean μ_f.
        d_M^2(z) = (z - μ_f)^T Σ_f^{-1} (z - μ_f)
        
        Args:
            z (torch.Tensor): Points to evaluate (B, d).
            
        Returns:
            torch.Tensor: Squared Mahalanobis distances (B,).
        """
        device = z.device
        mu_f = self.mu_f.to(device)
        Sigma_f = self.Sigma_f.to(device)
        
        diff = z - mu_f
        # Use Cholesky decomposition for numerical stability: Σ_f = L L^T
        L = torch.linalg.cholesky(Sigma_f)
        # Solve L y = diff^T for y
        y = torch.linalg.solve_triangular(L, diff.t(), upper=False) # (d, B)
        dist_sq = (y ** 2).sum(dim=0) # (B,)
        return dist_sq


class AdversarialCentroidSynthesizer:
    """
    Synthesizes the adversarial centroid z_adv* by maximizing epistemic vacuity 
    within the Mahalanobis ellipsoidal neighborhood E (Eqs. 11-14).
    """
    def __init__(self, manifold_stats: ForgettingManifoldStatistics, edl_head: nn.Module, 
                 num_classes: int, num_steps: int = 10, step_size: float = 0.1, rho: float = 1.5):
        """
        Args:
            manifold_stats (ForgettingManifoldStatistics): Statistics of the forgetting manifold.
            edl_head (nn.Module): The Evidential Deep Learning head g_ψ.
            num_classes (int): Number of classes K.
            num_steps (int): Number of projected gradient ascent steps.
            step_size (float): Step size η for gradient ascent.
            rho (float): Radius of the Mahalanobis ellipsoid E.
        """
        self.manifold_stats = manifold_stats
        self.edl_head = edl_head
        self.num_classes = num_classes
        self.num_steps = num_steps
        self.step_size = step_size
        self.rho = rho

    def compute_vacuity_gradient(self, z: torch.Tensor) -> torch.Tensor:
        """
        Computes the exact gradient of epistemic vacuity with respect to z.
        Eq. 11: ∇_z u(z) = ∇_z (K / S(z))
        
        Args:
            z (torch.Tensor): Latent features (B, d).
            
        Returns:
            torch.Tensor: Gradient of vacuity (B, d).
        """
        z.requires_grad_(True)
        evidence = self.edl_head(z)
        alpha = evidence + 1.0
        S = alpha.sum(dim=1)
        vacuity = self.num_classes / S
        
        # Compute gradient via autograd
        grad_vacuity = torch.autograd.grad(vacuity.sum(), z, create_graph=False)[0]
        return grad_vacuity

    def project_to_ellipsoid(self, v: torch.Tensor) -> torch.Tensor:
        """
        Projects points v onto the Mahalanobis ellipsoid E.
        Eq. 14: Π_E(v) = argmin_z ||z - v||_2^2 s.t. (z - μ_f)^T Σ_f^{-1} (z - μ_f) <= ρ^2
        Solved via eigendecomposition and bisection for the Lagrange multiplier.
        
        Args:
            v (torch.Tensor): Points to project (B, d).
            
        Returns:
            torch.Tensor: Projected points (B, d).
        """
        device = v.device
        mu_f = self.manifold_stats.mu_f.to(device)
        Sigma_f = self.manifold_stats.Sigma_f.to(device)
        
        # Check if points are already inside the ellipsoid
        dist_sq = self.manifold_stats.get_mahalanobis_distance_sq(v)
        inside_mask = dist_sq <= self.rho ** 2
        
        if inside_mask.all():
            return v
            
        # Eigendecomposition of Σ_f for exact projection
        eigenvalues, eigenvectors = torch.linalg.eigh(Sigma_f)
        eigenvalues = torch.clamp(eigenvalues, min=1e-6) # Numerical stability
        
        diff = v - mu_f
        w = torch.matmul(diff, eigenvectors) # Transform to eigenbasis (B, d)
        
        # Bisection method to find Lagrange multiplier ν >= 0
        nu_min = torch.zeros(v.size(0), device=device)
        nu_max = torch.ones(v.size(0), device=device) * 1e6
        
        lam = eigenvalues.unsqueeze(0) # (1, d)
        
        for _ in range(40): # 40 iterations for high precision
            nu_mid = (nu_min + nu_max) / 2.0
            denom = 1.0 + nu_mid.unsqueeze(1) / lam
            u = w / denom
            constraint_val = (u ** 2 / lam).sum(dim=1)
            
            outside = constraint_val > self.rho ** 2
            nu_min = torch.where(outside, nu_mid, nu_min)
            nu_max = torch.where(outside, nu_max, nu_mid)
            
        nu_final = (nu_min + nu_max) / 2.0
        denom_final = 1.0 + nu_final.unsqueeze(1) / lam
        u_proj = w / denom_final
        
        # Transform back to original basis
        diff_proj = torch.matmul(u_proj, eigenvectors.t())
        z_proj = diff_proj + mu_f
        
        # Combine inside and projected points
        z_final = torch.where(inside_mask.unsqueeze(1), v, z_proj)
        return z_final

    def synthesize(self, h_f_mean: torch.Tensor) -> torch.Tensor:
        """
        Synthesizes the adversarial centroid z_adv* via projected gradient ascent.
        Eqs. 12-13: z^(i+1) = Π_E(z^(i) + η ∇_z u(z^(i)))
        
        Args:
            h_f_mean (torch.Tensor): Initial mean of the forgetting manifold (d,).
            
        Returns:
            torch.Tensor: Synthesized adversarial centroid z_adv* (1, d).
        """
        # Initialize z at the mean of the forgetting manifold
        z = h_f_mean.detach().clone().unsqueeze(0) # (1, d)
        
        for _ in range(self.num_steps):
            grad_u = self.compute_vacuity_gradient(z)
            z_ascend = z + self.step_size * grad_u
            z = self.project_to_ellipsoid(z_ascend)
            
        # Detach to prevent gradient flow back to the manifold statistics during loss computation
        return z.detach()


class FeatureSpaceBoundaryRepulsion:
    """
    Computes the FSBR loss and handles gradient flow (Eqs. 15-16).
    """
    def __init__(self, manifold_stats: ForgettingManifoldStatistics, margin: float = 1.0):
        """
        Args:
            manifold_stats (ForgettingManifoldStatistics): Statistics of the forgetting manifold.
            margin (float): Margin γ for the squared hinge loss.
        """
        self.manifold_stats = manifold_stats
        self.margin = margin

    def compute_loss(self, h_r_edge: torch.Tensor, z_adv: torch.Tensor) -> torch.Tensor:
        """
        Computes the Feature-Space Boundary Repulsion loss.
        Eq. 15: L_FSBR = E[ max(0, γ - ||h(x) - z_adv*||_{Σ_f^{-1}})^2 ]
        Eq. 16: The gradient with respect to encoder parameters θ is handled automatically 
                by PyTorch autograd, as z_adv* is treated as a constant (stop-gradient).
        
        Args:
            h_r_edge (torch.Tensor): Latent features of the edge retained set D_r^e (B, d).
            z_adv (torch.Tensor): Synthesized adversarial centroid z_adv* (1, d).
            
        Returns:
            torch.Tensor: Scalar FSBR loss.
        """
        device = h_r_edge.device
        Sigma_f = self.manifold_stats.Sigma_f.to(device)
        
        # Compute difference between retained edge features and adversarial centroid
        diff = h_r_edge - z_adv # (B, d)
        
        # Compute Mahalanobis distance: d_M = sqrt( diff^T Σ_f^{-1} diff )
        L = torch.linalg.cholesky(Sigma_f)
        y = torch.linalg.solve_triangular(L, diff.t(), upper=False) # (d, B)
        dist_sq = (y ** 2).sum(dim=0) # (B,)
        dist = torch.sqrt(dist_sq + 1e-8)
        
        # Eq. 15: Squared hinge loss
        loss = torch.clamp(self.margin - dist, min=0.0) ** 2
        return loss.mean()