"""
Hessian-Vector Products (Pearlmutter's trick), Conjugate Gradient solver, Mahalanobis distance.
Designed as reusable mathematical utilities for the Oracle-Free Evidential Unlearning framework.

Key Methodology Mappings:
- HVP via Pearlmutter's trick: Enables O(p) implicit Hessian computation for the Generalized 
  Gauss-Newton (GGN) approximation in LIAV (Eq. 17).
- Conjugate Gradient Solver: Iteratively solves the damped linear system (H + λI)x = b to compute 
  influence vectors I_θ(x_i) without explicit matrix inversion (Eq. 18).
- Mahalanobis Distance: Computes the statistical distance d_M(z) = sqrt((z - μ_f)^T Σ_f^{-1} (z - μ_f)) 
  used in the FSBR ellipsoidal constraint (Eq. 12), projection operator (Eq. 14), and repulsion loss (Eq. 15).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Callable, List, Tuple, Iterator


# ==============================================================================
# 1. Parameter Flattening Utilities
# ==============================================================================
class ParameterFlattener:
    """
    Utility class to flatten and unflatten model parameters into a single 1D vector.
    Required for HVP and CG operations that operate on the full parameter space R^p.
    """

    @staticmethod
    def flatten_parameters(parameters: Iterator[nn.Parameter]) -> torch.Tensor:
        """
        Concatenates all parameter tensors into a single flat 1D vector.
        
        Args:
            parameters: Iterator over model parameters (e.g., model.parameters()).
            
        Returns:
            torch.Tensor: Flattened parameter vector of shape (P,).
        """
        return torch.cat([p.flatten() for p in parameters])

    @staticmethod
    def unflatten_to_parameters(flat_vector: torch.Tensor, reference_parameters: Iterator[nn.Parameter]) -> List[torch.Tensor]:
        """
        Reshapes a flat 1D vector back into a list of tensors matching the shapes of reference parameters.
        
        Args:
            flat_vector (torch.Tensor): Flat vector of shape (P,).
            reference_parameters: Iterator over reference model parameters providing target shapes.
            
        Returns:
            List[torch.Tensor]: List of reshaped tensors.
        """
        unflattened = []
        offset = 0
        for p in reference_parameters:
            numel = p.numel()
            unflattened.append(flat_vector[offset:offset + numel].view_as(p))
            offset += numel
        return unflattened


# ==============================================================================
# 2. Hessian-Vector Product via Pearlmutter's Trick
# ==============================================================================
class HessianVectorProduct:
    """
    Computes Hessian-Vector Products (HVP) using Pearlmutter's trick.
    Avoids explicit construction of the p x p Hessian matrix, maintaining O(p) time and space complexity.
    
    Pearlmutter's trick: H @ v = d/dt [nabla L(theta + t*v)]|_{t=0}
    Implemented via two nested autograd passes:
      1. Compute g = nabla_theta L(theta) with create_graph=True
      2. Compute H @ v = nabla_theta (g^T v)
    """

    def __init__(self, model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device):
        """
        Args:
            model (nn.Module): The model whose Hessian is being approximated.
            dataloader (DataLoader): DataLoader over the retained set D_r for computing the empirical Hessian.
            device (torch.device): Computation device.
        """
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.flattener = ParameterFlattener()

    def compute(self, v: torch.Tensor) -> torch.Tensor:
        """
        Computes the Hessian-Vector Product H @ v, where H is the empirical Hessian 
        (GGN approximation) averaged over the retained set D_r.
        
        This implements Eq. 17's requirement for implicit Hessian access:
        H = (1/|D_r|) sum_{x in D_r} nabla^2_theta ell(x, theta)
        
        Args:
            v (torch.Tensor): Flattened vector of shape (P,) to multiply with the Hessian.
            
        Returns:
            torch.Tensor: The resulting HVP vector of shape (P,).
        """
        Hv = torch.zeros_like(v)
        num_batches = 0

        for batch in self.dataloader:
            imgs = batch[0].to(self.device)
            targets = batch[1].to(self.device)

            self.model.zero_grad()

            # Forward pass
            outputs = self.model(imgs)
            if isinstance(outputs, tuple) and len(outputs) == 4:
                p_mean = outputs[2]  # Expected probability (Eq. 4)
            else:
                p_mean = outputs

            # Compute scalar loss for Hessian approximation
            loss = F.nll_loss(torch.log(p_mean + 1e-8), targets)

            # First-order gradients with computational graph retained for second-order differentiation
            grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            flat_grads = self.flattener.flatten_parameters(grads)

            # Compute the dot product g^T v (scalar)
            grad_v_dot = torch.dot(flat_grads, v)

            # Second-order gradients: nabla_theta (g^T v) = H @ v
            hvp_grads = torch.autograd.grad(grad_v_dot, self.model.parameters(), 
                                            retain_graph=False, allow_unused=True)
            
            # Handle unused parameters (e.g., buffers that don't require grad)
            flat_hvp_parts = []
            for g in hvp_grads:
                if g is not None:
                    flat_hvp_parts.append(g.flatten())
                else:
                    flat_hvp_parts.append(torch.zeros(0, device=self.device))
            flat_hvp = torch.cat(flat_hvp_parts)

            Hv += flat_hvp
            num_batches += 1

        # Average over all batches
        if num_batches > 0:
            Hv /= num_batches

        return Hv


# ==============================================================================
# 3. Conjugate Gradient Solver
# ==============================================================================
class ConjugateGradientSolver:
    """
    Iteratively solves the damped linear system (H + lambda * I) x = b using the 
    Conjugate Gradient (CG) method.
    
    Used in LIAV (Eq. 18) to compute the influence vector:
    I_theta(x_i) = (H + lambda * I)^{-1} nabla_theta ell(x_i, theta)
    without explicitly forming or inverting the p x p Hessian matrix.
    
    The CG method only requires Hessian-Vector Products (provided by HessianVectorProduct),
    maintaining O(p) memory and O(k*p) time for k iterations.
    """

    def __init__(self, hvp_fn: Callable[[torch.Tensor], torch.Tensor], damping: float = 1e-3,
                 max_iter: int = 100, tol: float = 1e-6):
        """
        Args:
            hvp_fn (Callable): Function that computes H @ v for any vector v.
            damping (float): Tikhonov damping coefficient lambda (Eq. 17).
            max_iter (int): Maximum number of CG iterations.
            tol (float): Convergence tolerance on the residual norm.
        """
        self.hvp_fn = hvp_fn
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol

    def _damped_hvp(self, v: torch.Tensor) -> torch.Tensor:
        """
        Computes (H + lambda * I) @ v = H @ v + lambda * v.
        
        Args:
            v (torch.Tensor): Input vector of shape (P,).
            
        Returns:
            torch.Tensor: Damped HVP result of shape (P,).
        """
        return self.hvp_fn(v) + self.damping * v

    def solve(self, b: torch.Tensor) -> torch.Tensor:
        """
        Solves (H + lambda * I) x = b using the Conjugate Gradient method.
        
        Algorithm:
          x_0 = 0
          r_0 = b - (H + lambda*I) x_0 = b
          p_0 = r_0
          For k = 0, 1, ...:
            alpha_k = (r_k^T r_k) / (p_k^T (H + lambda*I) p_k)
            x_{k+1} = x_k + alpha_k * p_k
            r_{k+1} = r_k - alpha_k * (H + lambda*I) p_k
            if ||r_{k+1}|| < tol: break
            beta_k = (r_{k+1}^T r_{k+1}) / (r_k^T r_k)
            p_{k+1} = r_{k+1} + beta_k * p_k
        
        Args:
            b (torch.Tensor): Right-hand side vector of shape (P,).
            
        Returns:
            torch.Tensor: Solution vector x of shape (P,).
        """
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rs_old = torch.dot(r, r).item()

        for iteration in range(self.max_iter):
            Ap = self._damped_hvp(p)
            pAp = torch.dot(p, Ap).item()

            # Guard against numerical breakdown
            if abs(pAp) < 1e-12:
                break

            alpha = rs_old / pAp
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = torch.dot(r, r).item()

            # Check convergence
            if np.sqrt(rs_new) < self.tol:
                break

            beta = rs_new / rs_old
            p = r + beta * p
            rs_old = rs_new

        return x


# ==============================================================================
# 4. Mahalanobis Distance
# ==============================================================================
class MahalanobisDistance:
    """
    Computes the Mahalanobis distance using the Cholesky decomposition of the covariance matrix.
    
    Used in FSBR for:
    - Eq. 12: Defining the ellipsoidal neighborhood E = {z : d_M(z, mu_f) <= rho}
    - Eq. 14: Projection operator Pi_E onto the Mahalanobis ellipsoid
    - Eq. 15: Repulsion loss L_FSBR with Mahalanobis norm
    
    The Mahalanobis distance is defined as:
    d_M(z, mu) = sqrt((z - mu)^T Sigma^{-1} (z - mu))
    
    Using Cholesky decomposition Sigma = L L^T:
    d_M^2 = || L^{-1} (z - mu)^T ||_2^2
    """

    def __init__(self, mu: torch.Tensor, Sigma: torch.Tensor):
        """
        Args:
            mu (torch.Tensor): Mean vector of shape (d,).
            Sigma (torch.Tensor): Covariance matrix of shape (d, d). Must be symmetric positive definite.
        """
        self.mu = mu
        self.Sigma = Sigma
        self._L = None  # Cached Cholesky factor

    def _get_cholesky(self, device: torch.device) -> torch.Tensor:
        """
        Computes and caches the Cholesky decomposition of Sigma.
        Sigma = L L^T where L is lower triangular.
        
        Args:
            device (torch.device): Target device.
            
        Returns:
            torch.Tensor: Lower triangular Cholesky factor L of shape (d, d).
        """
        if self._L is None or self._L.device != device:
            Sigma_device = self.Sigma.to(device)
            self._L = torch.linalg.cholesky(Sigma_device)
        return self._L

    def compute_squared_distance(self, z: torch.Tensor) -> torch.Tensor:
        """
        Computes the squared Mahalanobis distance for a batch of points.
        d_M^2(z) = (z - mu)^T Sigma^{-1} (z - mu) = || L^{-1} (z - mu)^T ||_2^2
        
        Args:
            z (torch.Tensor): Points to evaluate, shape (B, d) or (d,).
            
        Returns:
            torch.Tensor: Squared Mahalanobis distances, shape (B,) or scalar.
        """
        squeeze = False
        if z.dim() == 1:
            z = z.unsqueeze(0)
            squeeze = True

        device = z.device
        mu = self.mu.to(device)
        L = self._get_cholesky(device)

        # diff: (B, d)
        diff = z - mu

        # Solve L y = diff^T for y => y = L^{-1} diff^T
        # diff^T is (d, B), result y is (d, B)
        y = torch.linalg.solve_triangular(L, diff.t(), upper=False)

        # Squared distance: sum of squares along dimension 0
        dist_sq = (y ** 2).sum(dim=0)  # (B,)

        if squeeze:
            dist_sq = dist_sq.squeeze(0)

        return dist_sq

    def compute_distance(self, z: torch.Tensor) -> torch.Tensor:
        """
        Computes the Mahalanobis distance (square root of squared distance).
        d_M(z) = sqrt(d_M^2(z))
        
        Args:
            z (torch.Tensor): Points to evaluate, shape (B, d) or (d,).
            
        Returns:
            torch.Tensor: Mahalanobis distances.
        """
        dist_sq = self.compute_squared_distance(z)
        return torch.sqrt(dist_sq + 1e-8)

    def compute_inverse_transform(self, diff: torch.Tensor) -> torch.Tensor:
        """
        Computes Sigma^{-1} @ diff using the Cholesky factor.
        Sigma^{-1} diff = L^{-T} L^{-1} diff
        
        This is useful for computing gradients involving the Mahalanobis metric.
        
        Args:
            diff (torch.Tensor): Difference vectors, shape (B, d) or (d,).
            
        Returns:
            torch.Tensor: Transformed vectors Sigma^{-1} @ diff.
        """
        squeeze = False
        if diff.dim() == 1:
            diff = diff.unsqueeze(0)
            squeeze = True

        device = diff.device
        L = self._get_cholesky(device)

        # Step 1: Solve L y = diff^T => y = L^{-1} diff^T
        y = torch.linalg.solve_triangular(L, diff.t(), upper=False)  # (d, B)

        # Step 2: Solve L^T x = y => x = L^{-T} y = Sigma^{-1} diff^T
        x = torch.linalg.solve_triangular(L.t(), y, upper=True)  # (d, B)

        result = x.t()  # (B, d)

        if squeeze:
            result = result.squeeze(0)

        return result