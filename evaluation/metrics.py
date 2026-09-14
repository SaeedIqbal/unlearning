"""
Standard accuracy, Avg. Gap, and LOO alignment score ρ (Eq. 22).
Designed to evaluate the utility and instance-level unlearning efficacy of the Oracle-Free Evidential Unlearning framework.

Key Methodology Mappings:
- Standard Accuracy: Top-1 accuracy on the retained set D_r using the expected predictive probability p_mean (Eq. 4).
- Average Performance Gap: Avg. Gap = Acc(theta_retrain) - Acc(theta_unl) on D_r.
- LOO Alignment Score ρ (Eq. 22): Cosine similarity between the vacuity gradient ∇_θ u(x_i; θ_unl) 
  and the true Leave-One-Out parameter shift Δθ_LOO = θ_LOO - θ_orig.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict

class UnlearningMetricsEvaluator:
    """
    Evaluates standard utility metrics (Accuracy, Avg. Gap) and instance-level unlearning 
    efficacy (LOO alignment score ρ) for the Oracle-Free Evidential Unlearning framework.
    """
    def __init__(self, model_unl: nn.Module, model_retrain: nn.Module, device: torch.device):
        """
        Args:
            model_unl (nn.Module): The unlearned model.
            model_retrain (nn.Module): The exact retrained baseline model (trained on D_r).
            device (torch.device): Computation device.
        """
        self.model_unl = model_unl
        self.model_retrain = model_retrain
        self.device = device

    def compute_accuracy(self, model: nn.Module, dataloader: DataLoader) -> float:
        """
        Computes the top-1 accuracy on the given dataset using the expected predictive probability (Eq. 4).
        
        Args:
            model (nn.Module): The model to evaluate.
            dataloader (DataLoader): DataLoader for the target dataset (e.g., D_r).
            
        Returns:
            float: Top-1 accuracy in [0, 1].
        """
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Handle datasets returning (img, target, ...) or (img, target, demo_attrs, ...)
                imgs = batch[0].to(self.device)
                targets = batch[1].to(self.device)
                
                outputs = model(imgs)
                
                # Handle Oracle-Free Evidential Model output format
                if isinstance(outputs, tuple) and len(outputs) == 4:
                    p_mean = outputs[2]  # Expected probability (Eq. 4)
                else:
                    p_mean = outputs
                    
                _, predicted = torch.max(p_mean, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
        return correct / total if total > 0 else 0.0

    def compute_average_gap(self, dr_dataloader: DataLoader) -> float:
        """
        Computes the Average Performance Gap between the retrained baseline and the unlearned model on D_r.
        Avg. Gap = Acc(theta_retrain) - Acc(theta_unl)
        
        Args:
            dr_dataloader (DataLoader): DataLoader for the retained set D_r.
            
        Returns:
            float: Average performance gap (absolute difference).
        """
        acc_retrain = self.compute_accuracy(self.model_retrain, dr_dataloader)
        acc_unl = self.compute_accuracy(self.model_unl, dr_dataloader)
        
        gap = acc_retrain - acc_unl
        return float(gap)

    def compute_loo_alignment_score(self, x_i: torch.Tensor, y_i: torch.Tensor, 
                                    model_orig: nn.Module, model_loo: nn.Module) -> float:
        """
        Computes the empirical Leave-One-Out (LOO) alignment score ρ(x_i) (Eq. 22).
        ρ(x_i) = < ∇_θ u(x_i; θ_unl), Δθ_LOO > / ( || ∇_θ u(x_i; θ_unl) || || Δθ_LOO || )
        where Δθ_LOO = θ_LOO - θ_orig.
        
        Note: The Krylov subspace projection P_{x_i} is applied during training in the LIAV module. 
        Here, we evaluate the direct alignment between the unlearned model's vacuity gradient 
        and the true LOO parameter shift to validate the efficacy of the instance-level deletion.
        
        Args:
            x_i (torch.Tensor): Input image for the instance to be forgotten.
            y_i (torch.Tensor): Target label for the instance.
            model_orig (nn.Module): The original fully trained model.
            model_loo (nn.Module): The true LOO model (trained on D \ {x_i}).
            
        Returns:
            float: Cosine similarity ρ(x_i) in [-1, 1].
        """
        # 1. Compute Δθ_LOO = θ_LOO - θ_orig
        delta_theta_loo = []
        for p_loo, p_orig in zip(model_loo.parameters(), model_orig.parameters()):
            delta_theta_loo.append((p_loo - p_orig).flatten())
        delta_theta_loo = torch.cat(delta_theta_loo).to(self.device)
        
        # 2. Compute vacuity gradient ∇_θ u(x_i; θ_unl) (Eq. 21)
        self.model_unl.zero_grad()
        outputs = self.model_unl(x_i.unsqueeze(0).to(self.device))
        vacuity = outputs[1]  # u(x) (Eq. 3)
        
        grads = torch.autograd.grad(vacuity, self.model_unl.parameters())
        vacuity_grad = torch.cat([g.flatten() for g in grads])
        
        # 3. Compute cosine similarity between vacuity_grad and delta_theta_loo
        dot_prod = torch.dot(vacuity_grad, delta_theta_loo)
        norm_vac = torch.norm(vacuity_grad)
        norm_loo = torch.norm(delta_theta_loo)
        
        if norm_vac < 1e-8 or norm_loo < 1e-8:
            return 0.0
            
        rho = (dot_prod / (norm_vac * norm_loo)).item()
        return float(rho)

    def evaluate_utility(self, dr_dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluates standard utility metrics on the retained set D_r.
        
        Args:
            dr_dataloader (DataLoader): DataLoader for the retained set D_r.
            
        Returns:
            Dict[str, float]: Dictionary containing 'Acc_Retrain', 'Acc_Unl', and 'Avg_Gap'.
        """
        acc_retrain = self.compute_accuracy(self.model_retrain, dr_dataloader)
        acc_unl = self.compute_accuracy(self.model_unl, dr_dataloader)
        avg_gap = acc_retrain - acc_unl
        
        return {
            'Acc_Retrain': float(acc_retrain * 100.0),
            'Acc_Unl': float(acc_unl * 100.0),
            'Avg_Gap': float(avg_gap * 100.0) # Converted to percentage points
        }