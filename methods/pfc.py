"""
Probabilistic Forgetting Certification: Tracks theoretical bounds (Eqs. 27-28) and encoder drift.
Designed to support the Oracle-Free Evidential Unlearning framework.

Key Methodology Mappings:
- Eq. 27: PFC strict upper bound on expected predictive probability:
          p̂_c(z) <= (epsilon + 1) / (epsilon + K)
- Eq. 28: Theoretical maximum variance of predictive distribution as epsilon -> 0:
          max Var[p_c] = (1/K)(1 - 1/K) / (K + 1)
- Eq. 24 (Trust-Region): ||theta_T - theta_0||_2^2 <= delta^2
  The PFC guarantees are only valid when the encoder drift is bounded.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, List, Any
from torch.utils.data import DataLoader


class EncoderDriftMonitor:
    """
    Monitors the L2 drift of the encoder parameters from the initial checkpoint.
    Ensures the Trust-Region constraint (Eq. 24) is satisfied so that the 
    PFC theoretical guarantees remain mathematically valid at inference time.
    """
    def __init__(self, model: nn.Module, drift_threshold: float = 10.0):
        """
        Args:
            model (nn.Module): The Oracle-Free Evidential Model.
            drift_threshold (float): Maximum allowed L2 drift delta (Eq. 24).
        """
        self.drift_threshold = drift_threshold
        self.initial_params: Dict[str, torch.Tensor] = {}
        self._snapshot_initial_params(model)

    def _snapshot_initial_params(self, model: nn.Module) -> None:
        """
        Takes a deep copy of the encoder parameters theta_0 at initialization.
        Only the encoder parameters are tracked, as the EDL head is expected to change.
        
        Args:
            model (nn.Module): The model at initialization (theta_0).
        """
        for name, param in model.encoder.named_parameters():
            self.initial_params[name] = param.data.clone().detach().cpu()

    def compute_drift(self, model: nn.Module) -> float:
        """
        Computes the L2 norm of the encoder parameter drift.
        Eq. 24: ||theta_T - theta_0||_2
        
        Args:
            model (nn.Module): The current model state (theta_T).
            
        Returns:
            float: The L2 drift magnitude.
        """
        drift_sq = 0.0
        for name, param in model.encoder.named_parameters():
            if name in self.initial_params:
                diff = param.data.cpu() - self.initial_params[name]
                drift_sq += (diff ** 2).sum().item()
        return float(np.sqrt(drift_sq))

    def is_drift_bounded(self, model: nn.Module) -> bool:
        """
        Checks whether the encoder drift satisfies the Trust-Region constraint.
        Eq. 24: ||theta_T - theta_0||_2^2 <= delta^2
        
        Args:
            model (nn.Module): The current model state.
            
        Returns:
            bool: True if drift is within the threshold, False otherwise.
        """
        return self.compute_drift(model) <= self.drift_threshold


class ProbabilisticForgettingCertification:
    """
    Implements the Probabilistic Forgetting Certification (PFC) framework.
    Derives formal upper bounds on residual confidence via Dirichlet concentration parameters
    and validates them empirically against actual model outputs on the forgetting set.
    """
    def __init__(self, num_classes: int, epsilon: float = 0.1):
        """
        Args:
            num_classes (int): Number of classes K.
            epsilon (float): The residual evidence mass epsilon. When the model has 
                             fully forgotten, alpha_k -> 1 for all k, so e_k -> 0.
                             epsilon represents the maximum allowable residual evidence.
        """
        self.K = num_classes
        self.epsilon = epsilon

    def compute_theoretical_upper_bound(self) -> float:
        """
        Computes the PFC strict upper bound on the expected predictive probability.
        Eq. 27: p̂_c(z) <= (epsilon + 1) / (epsilon + K)
        
        When epsilon -> 0 (perfect forgetting), this bound approaches 1/K (uniform).
        
        Returns:
            float: The theoretical upper bound on p̂_c.
        """
        return (self.epsilon + 1.0) / (self.epsilon + self.K)

    def compute_theoretical_max_variance(self) -> float:
        """
        Computes the theoretical maximum variance of the predictive distribution.
        Eq. 28: max Var[p_c] = (1/K)(1 - 1/K) / (K + 1)
        This is achieved when alpha_k = 1 for all k (uniform Dirichlet).
        
        Returns:
            float: The theoretical maximum variance.
        """
        p_uniform = 1.0 / self.K
        S_uniform = float(self.K)  # When alpha_k = 1, S = K
        max_var = (p_uniform * (1.0 - p_uniform)) / (S_uniform + 1.0)
        return max_var

    def compute_empirical_bounds(self, model: nn.Module, forget_dataloader: DataLoader, 
                                 device: torch.device) -> Dict[str, float]:
        """
        Computes empirical statistics on the forgetting set D_f to validate the PFC bounds.
        Measures the actual maximum predictive probability, mean vacuity, and mean variance 
        on D_f samples and compares them against the theoretical bounds.
        
        Args:
            model (nn.Module): The unlearned model.
            forget_dataloader (DataLoader): DataLoader for the forgetting set D_f.
            device (torch.device): Computation device.
            
        Returns:
            Dict[str, float]: Dictionary containing empirical and theoretical metrics.
        """
        model.eval()
        
        all_max_probs = []
        all_vacuities = []
        all_variances = []
        all_evidences = []

        with torch.no_grad():
            for batch in forget_dataloader:
                imgs = batch[0].to(device)
                
                # Forward pass: evidence, vacuity, p_mean, p_var
                evidence, vacuity, p_mean, p_var = model(imgs)
                
                # Maximum predictive probability across all classes for each sample
                max_probs, _ = p_mean.max(dim=1)
                all_max_probs.append(max_probs.cpu().numpy())
                
                all_vacuities.append(vacuity.cpu().numpy())
                
                # Mean variance across all classes for each sample
                mean_var = p_var.mean(dim=1)
                all_variances.append(mean_var.cpu().numpy())
                
                # Mean total evidence for each sample
                mean_evidence = evidence.mean(dim=1)
                all_evidences.append(mean_evidence.cpu().numpy())

        all_max_probs = np.concatenate(all_max_probs)
        all_vacuities = np.concatenate(all_vacuities)
        all_variances = np.concatenate(all_variances)
        all_evidences = np.concatenate(all_evidences)

        theoretical_bound = self.compute_theoretical_upper_bound()
        theoretical_max_var = self.compute_theoretical_max_variance()

        return {
            'theoretical_p_bound': theoretical_bound,
            'empirical_max_p_mean': float(np.mean(all_max_probs)),
            'empirical_max_p_95th': float(np.percentile(all_max_probs, 95)),
            'empirical_max_p_max': float(np.max(all_max_probs)),
            'theoretical_max_var': theoretical_max_var,
            'empirical_mean_var': float(np.mean(all_variances)),
            'empirical_mean_vacuity': float(np.mean(all_vacuities)),
            'empirical_mean_evidence': float(np.mean(all_evidences)),
            'p_bound_satisfied': bool(np.all(all_max_probs <= theoretical_bound + 1e-4)),
            'num_samples': len(all_max_probs)
        }

    def compute_certification_report(self, model: nn.Module, forget_dataloader: DataLoader,
                                     drift_monitor: EncoderDriftMonitor, 
                                     device: torch.device) -> Dict[str, Any]:
        """
        Generates a comprehensive PFC certification report combining theoretical bounds, 
        empirical validation, and encoder drift verification.
        
        The PFC guarantees are only valid when:
        1. The encoder drift is bounded (Eq. 24).
        2. The empirical predictive probabilities on D_f are below the theoretical bound (Eq. 27).
        3. The empirical variance approaches the theoretical maximum (Eq. 28).
        
        Args:
            model (nn.Module): The unlearned model.
            forget_dataloader (DataLoader): DataLoader for D_f.
            drift_monitor (EncoderDriftMonitor): The drift monitoring instance.
            device (torch.device): Computation device.
            
        Returns:
            Dict[str, Any]: Complete certification report.
        """
        empirical = self.compute_empirical_bounds(model, forget_dataloader, device)
        drift = drift_monitor.compute_drift(model)
        drift_bounded = drift_monitor.is_drift_bounded(model)

        # Certification is valid only if drift is bounded AND empirical bounds are satisfied
        is_certified = drift_bounded and empirical['p_bound_satisfied']

        report = {
            'is_certified': is_certified,
            'drift_bounded': drift_bounded,
            'encoder_drift_l2': drift,
            'drift_threshold': drift_monitor.drift_threshold,
            'theoretical_p_bound': empirical['theoretical_p_bound'],
            'empirical_max_p_mean': empirical['empirical_max_p_mean'],
            'empirical_max_p_95th': empirical['empirical_max_p_95th'],
            'theoretical_max_var': empirical['theoretical_max_var'],
            'empirical_mean_var': empirical['empirical_mean_var'],
            'empirical_mean_vacuity': empirical['empirical_mean_vacuity'],
            'empirical_mean_evidence': empirical['empirical_mean_evidence'],
            'p_bound_satisfied': empirical['p_bound_satisfied'],
            'num_samples_evaluated': empirical['num_samples'],
            'num_classes_K': self.K,
            'epsilon': self.epsilon
        }
        return report

    def track_bounds_over_training(self, model: nn.Module, forget_dataloader: DataLoader,
                                   drift_monitor: EncoderDriftMonitor, 
                                   device: torch.device, 
                                   history: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Tracks the PFC bounds and encoder drift over the course of training.
        Should be called at the end of each epoch to monitor convergence.
        
        Args:
            model (nn.Module): The current model state.
            forget_dataloader (DataLoader): DataLoader for D_f.
            drift_monitor (EncoderDriftMonitor): The drift monitoring instance.
            device (torch.device): Computation device.
            history (List[Dict]): Existing history to append to. If None, creates new list.
            
        Returns:
            List[Dict[str, Any]]: Updated history of PFC metrics.
        """
        if history is None:
            history = []

        report = self.compute_certification_report(model, forget_dataloader, drift_monitor, device)
        report['epoch'] = len(history) + 1
        history.append(report)
        return history