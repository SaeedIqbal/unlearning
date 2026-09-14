"""
Adaptive Membership Inference Attack (loss-based ASR computation).
Designed to evaluate the structural privacy of the Oracle-Free Evidential Unlearning framework.

Key Methodology Mappings:
- Implements the Adaptive Membership Inference Attack (MIA) as defined in the Threat Model (Section 3.3).
- Eq. (MIA Advantage): Adv_MIA = | P_{x ~ D_f}[A(x) = 1] - P_{x ~ D_out}[A(x) = 1] |
- Computes the empirical cross-entropy loss for each sample using the expected predictive distribution (Eq. 4) 
  to distinguish between members of the forgetting set (D_f) and non-members (D_out).
- Determines the optimal decision threshold to maximize the attack advantage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict

class AdaptiveMembershipInferenceAttack:
    """
    Implements the Adaptive Membership Inference Attack (MIA) to evaluate the privacy 
    guarantees of the unlearned model. The attack uses the model's loss distribution 
    to distinguish between members of the forgetting set (D_f) and non-members (D_out).
    """
    def __init__(self, model: nn.Module, device: torch.device):
        """
        Args:
            model (nn.Module): The Oracle-Free Evidential Unlearning model.
            device (torch.device): The computation device ('cuda' or 'cpu').
        """
        self.model = model
        self.device = device
        
    def compute_sample_losses(self, dataloader: DataLoader) -> np.ndarray:
        """
        Computes the empirical cross-entropy loss for each sample in the dataloader.
        Uses the expected predictive probability p_mean (Eq. 4) to compute the loss.
        
        Args:
            dataloader (DataLoader): DataLoader for the target set (either D_f or D_out).
            
        Returns:
            np.ndarray: Array of shape (N,) containing the loss for each sample.
        """
        self.model.eval()
        losses = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Handle datasets returning (img, target, ...) or (img, target, demo_attrs, ...)
                imgs = batch[0].to(self.device)
                targets = batch[1].to(self.device)
                
                # Forward pass to get expected probabilities (Eq. 4)
                outputs = self.model(imgs)
                p_mean = outputs[2] 
                
                # Compute cross-entropy loss for each sample using the expected probability
                # Adding a small epsilon to prevent log(0)
                log_p = torch.log(p_mean + 1e-8)
                loss = F.nll_loss(log_p, targets, reduction='none')
                
                losses.append(loss.cpu().numpy())
                
        return np.concatenate(losses, axis=0)

    def find_optimal_threshold(self, member_losses: np.ndarray, non_member_losses: np.ndarray) -> float:
        """
        Finds the optimal loss threshold that maximizes the MIA Advantage.
        Members (D_f) typically have lower loss than non-members (D_out).
        The attack predicts "member" (A(x) = 1) if loss <= threshold.
        
        Args:
            member_losses (np.ndarray): Losses for the forgetting set D_f.
            non_member_losses (np.ndarray): Losses for the non-member set D_out.
            
        Returns:
            float: The optimal threshold.
        """
        # Combine losses and sort to evaluate all possible thresholds
        all_losses = np.concatenate([member_losses, non_member_losses])
        thresholds = np.unique(all_losses)
        
        best_advantage = -1.0
        best_threshold = 0.0
        
        # To maintain computational efficiency on large datasets, evaluate at percentiles if needed
        if len(thresholds) > 1000:
            thresholds = np.percentile(all_losses, np.linspace(0, 100, 1000))
            
        for thresh in thresholds:
            # True Positive Rate (TPR): P(A(x) = 1 | x in D_f)
            tpr = np.mean(member_losses <= thresh)
            # False Positive Rate (FPR): P(A(x) = 1 | x in D_out)
            fpr = np.mean(non_member_losses <= thresh)
            
            # MIA Advantage = |TPR - FPR|
            advantage = abs(tpr - fpr)
            if advantage > best_advantage:
                best_advantage = advantage
                best_threshold = thresh
                
        return float(best_threshold)

    def evaluate(self, df_dataloader: DataLoader, dout_dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluates the MIA Attack Success Rate (ASR) and Advantage on the unlearned model.
        
        Args:
            df_dataloader (DataLoader): DataLoader for the forgetting set D_f (members).
            dout_dataloader (DataLoader): DataLoader for the non-member set D_out.
            
        Returns:
            Dict[str, float]: Dictionary containing 'ASR' (%), 'MIA_Advantage', 'TPR', 'FPR', and 'Optimal_Threshold'.
        """
        # Compute losses for members and non-members
        member_losses = self.compute_sample_losses(df_dataloader)
        non_member_losses = self.compute_sample_losses(dout_dataloader)
        
        # Find the optimal threshold that maximizes the attack advantage
        optimal_threshold = self.find_optimal_threshold(member_losses, non_member_losses)
        
        # Compute attack predictions based on the optimal threshold
        # Predict 1 (member) if loss <= threshold
        member_preds = (member_losses <= optimal_threshold).astype(float)
        non_member_preds = (non_member_losses <= optimal_threshold).astype(float)
        
        # True Positive Rate (TPR) and False Positive Rate (FPR)
        tpr = np.mean(member_preds)
        fpr = np.mean(non_member_preds)
        
        # Attack Success Rate (ASR) = (TPR + 1 - FPR) / 2
        asr = (tpr + 1.0 - fpr) / 2.0
        
        # MIA Advantage = |TPR - FPR|
        mia_advantage = abs(tpr - fpr)
        
        return {
            'ASR': float(asr * 100.0),  # Converted to percentage
            'MIA_Advantage': float(mia_advantage),
            'TPR': float(tpr),
            'FPR': float(fpr),
            'Optimal_Threshold': optimal_threshold
        }