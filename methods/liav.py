"""
Localized Influence-Aware Vacuity: Krylov subspace isolation (Eq. 17-20) and spectral filtering (Eq. 23).
Designed to support the Oracle-Free Evidential Unlearning framework.

Key Methodology Mappings:
- Eq. 17: Generalized Gauss-Newton (GGN) matrix approximation with Tikhonov damping.
- Eq. 18: Updatable influence vector I_theta(x_i) = (H + lambda I)^{-1} nabla_theta ell(x_i, theta).
- Eq. 19 & 20: Orthogonal projector Pi_null and rank-m Krylov subspace projector P_{x_i} via Arnoldi iteration.
- Eq. 21: Gradient of vacuity nabla_theta u(x_i; theta) via chain rule.
- Eq. 22: Cosine similarity rho(x_i) between projected vacuity gradient and LOO shift.
- Eq. 23: Final regularized parameter update Delta theta_final via spectral filtering matrix M(rho).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Callable

class HessianVectorProduct:
    """
    Utility class to compute Hessian-Vector Products (HVP) using Pearlmutter's trick.
    Avoids explicit construction of the Hessian matrix, maintaining O(p) time complexity.
    """
    def __init__(self, model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device):
        self.model = model
        self.dataloader = dataloader
        self.device = device

    def compute(self, v: torch.Tensor) -> torch.Tensor:
        """
        Computes H @ v, where H is the empirical Hessian (GGN approximation) over the retained set D_r.
        
        Args:
            v (torch.Tensor): Flattened parameter vector to multiply with the Hessian.
            
        Returns:
            torch.Tensor: The resulting HVP vector, flattened.
        """
        Hv = torch.zeros_like(v)
        
        for batch in self.dataloader:
            imgs = batch[0].to(self.device)
            targets = batch[1].to(self.device)
            
            self.model.zero_grad()
            outputs = self.model(imgs)
            p_mean = outputs[2]  # Expected probability (Eq. 4)
            
            # Compute standard cross-entropy loss on expected probabilities for Hessian approximation
            loss = F.nll_loss(torch.log(p_mean + 1e-8), targets)
            
            # First-order gradients
            grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            flat_grads = torch.cat([g.flatten() for g in grads])
            
            # Dot product of gradients and vector v
            grad_v_dot = torch.dot(flat_grads, v)
            
            # Second-order gradients (HVP)
            hvp_grads = torch.autograd.grad(grad_v_dot, self.model.parameters())
            flat_hvp = torch.cat([g.flatten() for g in hvp_grads])
            
            Hv += flat_hvp
            
        # Average over the dataset
        Hv /= len(self.dataloader)
        return Hv


class ConjugateGradientSolver:
    """
    Solves the linear system (H + lambda I) x = b using the Conjugate Gradient method.
    Used to compute the influence vector without explicit matrix inversion.
    """
    def __init__(self, hvp_fn: Callable[[torch.Tensor], torch.Tensor], damping: float = 1e-3, 
                 max_iter: int = 100, tol: float = 1e-6):
        self.hvp_fn = hvp_fn
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol

    def solve(self, b: torch.Tensor) -> torch.Tensor:
        """
        Solves (H + lambda I) x = b.
        
        Args:
            b (torch.Tensor): The right-hand side vector (gradient of loss for a specific instance).
            
        Returns:
            torch.Tensor: The solution vector x (influence vector).
        """
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rs_old = torch.dot(r, r).item()

        for i in range(self.max_iter):
            Ap = self.hvp_fn(p) + self.damping * p
            pAp = torch.dot(p, Ap).item()
            
            if pAp < 1e-10:
                break
                
            alpha = rs_old / pAp
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = torch.dot(r, r).item()
            
            if np.sqrt(rs_new) < self.tol:
                break
                
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
            
        return x


class LocalizedInfluenceAwareVacuity:
    """
    Implements the Localized Influence-Aware Vacuity (LIAV) mechanism for precise instance-level deletion.
    Restricts updates to validated Krylov subspaces and applies spectral filtering to prevent collateral over-forgetting.
    """
    def __init__(self, model: nn.Module, retained_dataloader: torch.utils.data.DataLoader, 
                 device: torch.device, damping: float = 1e-3, krylov_dim: int = 10, 
                 loo_threshold: float = 0.85):
        """
        Args:
            model (nn.Module): The Oracle-Free Evidential Model.
            retained_dataloader (DataLoader): DataLoader for the retained set D_r.
            device (torch.device): Computation device.
            damping (float): Tikhonov damping parameter lambda for Hessian regularization (Eq. 17).
            krylov_dim (int): Dimension m of the Krylov subspace (Eq. 20).
            loo_threshold (float): Threshold tau_val for Leave-One-Out alignment validation (Eq. 22).
        """
        self.model = model
        self.retained_dataloader = retained_dataloader
        self.device = device
        self.damping = damping
        self.krylov_dim = krylov_dim
        self.loo_threshold = loo_threshold
        
        # Initialize HVP utility and CG solver
        self.hvp_util = HessianVectorProduct(model, retained_dataloader, device)
        self.cg_solver = ConjugateGradientSolver(self.hvp_util.compute, damping=self.damping)

    def compute_influence_vector(self, x_i: torch.Tensor, y_i: torch.Tensor) -> torch.Tensor:
        """
        Computes the influence vector for a specific instance x_i.
        Eq. 18: I_theta(x_i) = (H + lambda I)^{-1} nabla_theta ell(x_i, theta)
        
        Args:
            x_i (torch.Tensor): Input image for the instance to be forgotten.
            y_i (torch.Tensor): Target label for the instance.
            
        Returns:
            torch.Tensor: Flattened influence vector.
        """
        self.model.zero_grad()
        outputs = self.model(x_i.unsqueeze(0).to(self.device))
        p_mean = outputs[2]
        loss = F.nll_loss(torch.log(p_mean + 1e-8), y_i.unsqueeze(0).to(self.device))
        
        grads = torch.autograd.grad(loss, self.model.parameters())
        flat_grads = torch.cat([g.flatten() for g in grads])
        
        # Solve the damped linear system using CG
        influence_vec = self.cg_solver.solve(flat_grads)
        return influence_vec

    def compute_vacuity_gradient(self, x_i: torch.Tensor) -> torch.Tensor:
        """
        Computes the gradient of epistemic vacuity with respect to model parameters.
        Eq. 21: nabla_theta u(x_i; theta)
        
        Args:
            x_i (torch.Tensor): Input image for the instance.
            
        Returns:
            torch.Tensor: Flattened vacuity gradient vector.
        """
        self.model.zero_grad()
        outputs = self.model(x_i.unsqueeze(0).to(self.device))
        vacuity = outputs[1]  # u(x) (Eq. 3)
        
        grads = torch.autograd.grad(vacuity, self.model.parameters())
        flat_grads = torch.cat([g.flatten() for g in grads])
        return flat_grads

    def compute_krylov_projector(self, influence_vec: torch.Tensor) -> Tuple[Callable, Callable]:
        """
        Constructs the rank-m Krylov subspace projector P_{x_i} and its complement P_{x_i}^perp.
        Eq. 19 & 20: Uses Arnoldi iteration with modified Gram-Schmidt orthogonalization.
        
        Args:
            influence_vec (torch.Tensor): The influence vector to seed the Krylov subspace.
            
        Returns:
            Tuple[Callable, Callable]: Projection functions for the subspace and its orthogonal complement.
        """
        # Initialize basis with normalized influence vector
        q = influence_vec / (torch.norm(influence_vec) + 1e-8)
        Q = [q]
        
        # Arnoldi iteration to build orthonormal basis for Krylov subspace
        for j in range(1, self.krylov_dim):
            v = self.hvp_util.compute(Q[-1])
            # Orthogonalize against existing basis vectors
            for i in range(j):
                v = v - torch.dot(v, Q[i]) * Q[i]
            
            norm_v = torch.norm(v)
            if norm_v < 1e-8:
                break  # Subspace is fully spanned
                
            Q.append(v / norm_v)
            
        # Stack basis vectors into a matrix of shape (m, P)
        Q_mat = torch.stack(Q)
        
        def project(v: torch.Tensor) -> torch.Tensor:
            """Projects v onto the Krylov subspace."""
            coeffs = torch.matmul(Q_mat, v)
            return torch.matmul(coeffs, Q_mat)
            
        def project_complement(v: torch.Tensor) -> torch.Tensor:
            """Projects v onto the orthogonal complement of the Krylov subspace."""
            return v - project(v)
            
        return project, project_complement

    def compute_cosine_similarity(self, vacuity_grad: torch.Tensor, influence_vec: torch.Tensor, 
                                  project_fn: Callable) -> float:
        """
        Computes the cosine similarity between the projected vacuity gradient and the influence vector.
        Eq. 22: rho(x_i) = < P_{x_i} nabla_theta u, I_theta(x_i) > / ( || P_{x_i} nabla_theta u || || I_theta(x_i) || )
        
        Args:
            vacuity_grad (torch.Tensor): Gradient of vacuity.
            influence_vec (torch.Tensor): Influence vector (LOO shift).
            project_fn (Callable): Function to project onto the Krylov subspace.
            
        Returns:
            float: Cosine similarity rho(x_i).
        """
        proj_vacuity_grad = project_fn(vacuity_grad)
        
        dot_prod = torch.dot(proj_vacuity_grad, influence_vec)
        norm_proj = torch.norm(proj_vacuity_grad)
        norm_inf = torch.norm(influence_vec)
        
        if norm_proj < 1e-8 or norm_inf < 1e-8:
            return 0.0
            
        rho = (dot_prod / (norm_proj * norm_inf)).item()
        return rho

    def compute_spectral_filter(self, rho: float) -> float:
        """
        Computes the spectral filtering scalar M(rho) to dampen updates if LOO alignment is poor.
        Eq. 23: M(rho) = max(0, (rho - tau_val) / (1 - tau_val))
        
        Args:
            rho (float): Cosine similarity rho(x_i).
            
        Returns:
            float: Scaling factor M(rho) in [0, 1].
        """
        if rho < self.loo_threshold:
            return max(0.0, (rho - self.loo_threshold) / (1.0 - self.loo_threshold + 1e-8))
        return 1.0

    def compute_instance_update(self, x_i: torch.Tensor, y_i: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Main orchestration method to compute the final parameter update Delta theta_final for a specific instance.
        
        Args:
            x_i (torch.Tensor): Input image for the instance to be forgotten.
            y_i (torch.Tensor): Target label for the instance.
            
        Returns:
            Tuple[torch.Tensor, float]: The flattened parameter update direction and the LOO alignment score rho.
        """
        # 1. Compute influence vector I_theta(x_i) (Eq. 18)
        influence_vec = self.compute_influence_vector(x_i, y_i)
        
        # 2. Compute vacuity gradient nabla_theta u(x_i) (Eq. 21)
        vacuity_grad = self.compute_vacuity_gradient(x_i)
        
        # 3. Compute Krylov subspace projector (Eq. 20)
        project_fn, _ = self.compute_krylov_projector(influence_vec)
        
        # 4. Compute cosine similarity rho(x_i) (Eq. 22)
        rho = self.compute_cosine_similarity(vacuity_grad, influence_vec, project_fn)
        
        # 5. Compute spectral filter M(rho) (Eq. 23)
        M_rho = self.compute_spectral_filter(rho)
        
        # 6. Final update direction Delta theta_final = - M(rho) * I_theta(x_i)
        # (The actual learning rate eta is applied by the external training loop)
        delta_theta = -M_rho * influence_vec
        
        return delta_theta, rho