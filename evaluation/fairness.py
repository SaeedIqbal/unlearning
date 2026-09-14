"""
Demographic Parity Difference (DPD) for FairFace and UTKFace.
Designed to evaluate the demographic fairness of the Oracle-Free Evidential Unlearning framework.

Key Methodology Mappings:
- Implements the Demographic Parity Difference (DPD) metric to ensure that the unlearning 
  process does not introduce or exacerbate biases across protected demographic attributes 
  (e.g., race, gender, age).
- DPD measures the maximum difference in the expected probability of predicting a specific 
  class across different demographic groups.
- Eq. (DPD): DPD_c = max_{a, a'} | P(Y_hat=c | A=a) - P(Y_hat=c | A=a') |
- Overall DPD = max_c DPD_c
- Uses the expected predictive probability p_mean (Eq. 4) to compute the group-wise expectations.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Any

class DemographicFairnessEvaluator:
    """
    Evaluates the demographic fairness of the unlearned model using the Demographic Parity Difference (DPD).
    Ensures that the Evidential Uncertainty Partitioning (EUP) and Feature-Space Boundary Repulsion (FSBR) 
    do not disproportionately affect specific demographic subgroups.
    """
    def __init__(self, model: nn.Module, device: torch.device):
        """
        Args:
            model (nn.Module): The Oracle-Free Evidential Unlearning model.
            device (torch.device): The computation device ('cuda' or 'cpu').
        """
        self.model = model
        self.device = device

    def extract_predictions_and_attributes(self, dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Extracts the expected predictive probabilities (Eq. 4) and demographic attributes for all samples.
        
        Args:
            dataloader (DataLoader): DataLoader for the target dataset (FairFace or UTKFace).
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]: 
                - p_mean: Expected probabilities (N, K).
                - pred_classes: Predicted class indices (N,).
                - demo_attrs: Dictionary mapping attribute names to tensors of group indices (N,).
        """
        self.model.eval()
        all_p_mean = []
        all_pred_classes = []
        all_demo_attrs = {'race': [], 'gender': [], 'age': []}
        
        with torch.no_grad():
            for batch in dataloader:
                # Handle datasets returning (img, target, demo_attrs, idx)
                imgs = batch[0].to(self.device)
                demo_attrs = batch[2] # Dict of tensors collated by PyTorch
                
                # Forward pass to get expected probabilities (Eq. 4)
                outputs = self.model(imgs)
                p_mean = outputs[2] 
                pred_classes = torch.argmax(p_mean, dim=1)
                
                all_p_mean.append(p_mean.cpu())
                all_pred_classes.append(pred_classes.cpu())
                
                for key in all_demo_attrs:
                    if key in demo_attrs:
                        all_demo_attrs[key].append(demo_attrs[key].cpu())
                        
        all_p_mean = torch.cat(all_p_mean, dim=0)
        all_pred_classes = torch.cat(all_pred_classes, dim=0)
        for key in all_demo_attrs:
            if all_demo_attrs[key]:
                all_demo_attrs[key] = torch.cat(all_demo_attrs[key], dim=0)
            else:
                all_demo_attrs[key] = torch.tensor([])
                
        return all_p_mean, all_pred_classes, all_demo_attrs

    def compute_group_probabilities(self, p_mean: torch.Tensor, demo_attrs: Dict[str, torch.Tensor], attr_name: str) -> Dict[Any, torch.Tensor]:
        """
        Computes P(Y_hat=c | A=a) for each group 'a' in the specified demographic attribute.
        
        Args:
            p_mean (torch.Tensor): Expected probabilities (N, K).
            demo_attrs (Dict[str, torch.Tensor]): Dictionary of demographic attributes.
            attr_name (str): The demographic attribute to evaluate ('race', 'gender', or 'age').
            
        Returns:
            Dict[Any, torch.Tensor]: Mapping from group index to a tensor of shape (K,) 
                                     containing the mean expected probability for each class.
        """
        attr_values = demo_attrs[attr_name]
        unique_groups = torch.unique(attr_values)
        
        # Filter out invalid/missing values (e.g., -1 used for missing data)
        unique_groups = unique_groups[unique_groups >= 0]
        
        group_probs = {}
        for g in unique_groups:
            mask = (attr_values == g)
            if mask.sum() == 0:
                continue
                
            # Mean expected probability for this demographic group
            # p_mean is (N, K), so mean over dim=0 gives (K,)
            probs = p_mean[mask].mean(dim=0)
            group_probs[g.item()] = probs
            
        return group_probs

    def compute_dpd(self, group_probs: Dict[Any, torch.Tensor]) -> Tuple[float, np.ndarray]:
        """
        Computes the Demographic Parity Difference (DPD).
        DPD_c = max_{a, a'} | P(Y_hat=c | A=a) - P(Y_hat=c | A=a') |
        Overall DPD = max_c DPD_c
        
        Args:
            group_probs (Dict[Any, torch.Tensor]): Group-wise expected probabilities.
            
        Returns:
            Tuple[float, np.ndarray]: 
                - overall_dpd: The maximum DPD across all classes.
                - dpd_per_class: Array of DPD values for each class.
        """
        if len(group_probs) < 2:
            # Cannot compute parity difference with fewer than 2 groups
            num_classes = next(iter(group_probs.values())).shape[0]
            return 0.0, np.zeros(num_classes)
            
        groups = list(group_probs.keys())
        # Stack probabilities: (num_groups, K)
        probs_tensor = torch.stack([group_probs[g] for g in groups]) 
        
        # Compute pairwise absolute differences for each class
        # Resulting shape: (num_groups, num_groups, K)
        diffs = torch.abs(probs_tensor[:, None, :] - probs_tensor[None, :, :])
        
        # Max difference across groups for each class -> (K,)
        # First max over the second group (dim=1), then max over the first group (dim=0)
        dpd_per_class = diffs.max(dim=1)[0].max(dim=0)[0] 
        
        # Overall DPD is the maximum across all classes
        overall_dpd = dpd_per_class.max().item()
        
        return overall_dpd, dpd_per_class.cpu().numpy()

    def evaluate(self, dataloader: DataLoader, attr_names: List[str] = ['race', 'gender', 'age']) -> Dict[str, Any]:
        """
        Main evaluation pipeline to compute DPD for the specified demographic attributes.
        
        Args:
            dataloader (DataLoader): DataLoader for the target dataset.
            attr_names (List[str]): List of demographic attributes to evaluate.
            
        Returns:
            Dict[str, Any]: Dictionary containing DPD results for each attribute.
        """
        p_mean, pred_classes, demo_attrs = self.extract_predictions_and_attributes(dataloader)
        
        results = {}
        for attr in attr_names:
            if attr not in demo_attrs or demo_attrs[attr].numel() == 0:
                continue
                
            group_probs = self.compute_group_probabilities(p_mean, demo_attrs, attr)
            overall_dpd, dpd_per_class = self.compute_dpd(group_probs)
            
            # Compute group sizes for context
            group_sizes = {}
            for g in group_probs:
                group_sizes[g] = (demo_attrs[attr] == g).sum().item()
            
            results[attr] = {
                'overall_dpd': overall_dpd,
                'dpd_per_class': dpd_per_class.tolist(),
                'num_groups': len(group_probs),
                'group_sizes': group_sizes
            }
            
        return results