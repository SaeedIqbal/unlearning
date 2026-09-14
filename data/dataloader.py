"""
DataLoaders for D_r, D_f, D_r^e, D_r^c, and D_out (for MIA).
Designed to support the Oracle-Free Evidential Unlearning framework.

Key Methodology Mappings:
- UnlearningDataPartitioner: Handles the initial split of D into D_r and D_f (Eq. 1).
- EvidentialPartitioner: Implements Evidential Uncertainty Partitioning (EUP, Eq. 9), 
  computing expected probability (Eq. 4), vacuity (Eq. 3), and variance (Eq. 5) to 
  separate D_r into D_r^e (edge) and D_r^c (confident).
- UnlearningDataLoaderFactory: Centralized factory to generate and manage all required 
  PyTorch DataLoaders, including D_out for Membership Inference Attack (MIA) evaluation.
"""

import torch
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np

class UnlearningDataPartitioner:
    """
    Handles the initial partitioning of the full dataset D into D_r (retained) and D_f (forgotten).
    Supports both class-level and instance-level unlearning protocols.
    """
    def __init__(self, full_dataset, forget_targets, target_type='class'):
        """
        Args:
            full_dataset (Dataset): The original PyTorch Dataset (D = D_r U D_f).
            forget_targets (list): List of class indices (for class-level) or sample indices (for instance-level).
            target_type (str): 'class' or 'instance'.
        """
        self.full_dataset = full_dataset
        self.forget_targets = set(forget_targets)
        self.target_type = target_type
        
        self.dr_indices = []
        self.df_indices = []
        self._partition()
        
    def _partition(self):
        """
        Iterates through the dataset to separate D_r and D_f based on the target_type.
        """
        for idx in range(len(self.full_dataset)):
            # Assuming dataset returns (img, target, actual_idx) or (img, target)
            sample = self.full_dataset[idx]
            target = sample[1]
            
            if self.target_type == 'class':
                if target in self.forget_targets:
                    self.df_indices.append(idx)
                else:
                    self.dr_indices.append(idx)
            elif self.target_type == 'instance':
                # For instance-level, we check the actual_idx if available, or the dataset index
                actual_idx = sample[2] if len(sample) > 2 else idx
                if actual_idx in self.forget_targets:
                    self.df_indices.append(idx)
                else:
                    self.dr_indices.append(idx)
            else:
                raise ValueError("target_type must be 'class' or 'instance'")
                
    def get_dr_subset(self):
        """Returns the retained subset D_r."""
        return Subset(self.full_dataset, self.dr_indices)
        
    def get_df_subset(self):
        """Returns the forgetting subset D_f."""
        return Subset(self.full_dataset, self.df_indices)


class EvidentialPartitioner:
    """
    Implements Evidential Uncertainty Partitioning (EUP, Eq. 9).
    Partitions D_r into D_r^e (edge/ambiguous) and D_r^c (confident).
    """
    def __init__(self, thresholds):
        """
        Args:
            thresholds (dict): Dictionary containing 'p' (tau_p), 'u' (tau_u), 'v' (tau_v).
        """
        self.thresholds = thresholds
        self.dre_indices = []
        self.drc_indices = []
        
    def compute_evidential_metrics(self, model, dataloader, device):
        """
        Computes expected probability (Eq. 4), vacuity (Eq. 3), and variance (Eq. 5) for each sample.
        
        Returns:
            list of tuples: (index, p_y, u_x, var_y)
        """
        model.eval()
        metrics = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Handle datasets returning (img, target, idx) or (img, target)
                imgs = batch[0]
                targets = batch[1]
                indices = batch[2] if len(batch) > 2 else torch.arange(imgs.size(0))
                
                imgs = imgs.to(device)
                
                # Forward pass through the evidential model
                outputs = model(imgs)
                
                if len(outputs) == 4:
                    _, evidence, vacuity, variance = outputs
                else:
                    evidence = outputs
                    # Compute S(x) = sum(e_k) + K
                    K = evidence.size(1)
                    S = evidence.sum(dim=1) + K
                    vacuity = K / S  # Eq. 3
                    
                    # Compute expected probability for the true class (Eq. 4)
                    p_y = (evidence.gather(1, targets.unsqueeze(1)).squeeze(1) + 1) / S
                    
                    # Compute variance (Eq. 5)
                    variance = (p_y * (1 - p_y)) / (S + 1)
                
                for i in range(imgs.size(0)):
                    idx = indices[i].item() if torch.is_tensor(indices[i]) else indices[i]
                    metrics.append((idx, p_y[i].item(), vacuity[i].item(), variance[i].item()))
                    
        return metrics
        
    def partition_retained_set(self, model, retained_dataset, batch_size, device, num_workers=4):
        """
        Splits D_r into D_r^e and D_r^c based on Eq. 9:
        D_r^e = {x in D_r | p_y(x) < tau_p OR u(x) > tau_u OR Var[p_y] > tau_v}
        D_r^c = D_r \ D_r^e
        """
        temp_loader = DataLoader(retained_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        metrics = self.compute_evidential_metrics(model, temp_loader, device)
        
        self.dre_indices = []
        self.drc_indices = []
        
        tau_p = self.thresholds.get('p', 0.5)
        tau_u = self.thresholds.get('u', 0.5)
        tau_v = self.thresholds.get('v', 0.05)
        
        for idx, p_y, u_x, var_y in metrics:
            if p_y < tau_p or u_x > tau_u or var_y > tau_v:
                self.dre_indices.append(idx)
            else:
                self.drc_indices.append(idx)
                
        return self.dre_indices, self.drc_indices
        
    def get_dre_subset(self, retained_dataset):
        """Returns the edge/ambiguous retained subset D_r^e."""
        return Subset(retained_dataset, self.dre_indices)
        
    def get_drc_subset(self, retained_dataset):
        """Returns the confident retained subset D_r^c."""
        return Subset(retained_dataset, self.drc_indices)


class UnlearningDataLoaderFactory:
    """
    Factory class to create and manage DataLoaders for D_r, D_f, D_r^e, D_r^c, and D_out.
    """
    def __init__(self, full_dataset, forget_targets, target_type='class', 
                 edl_thresholds=None, batch_size=128, num_workers=4, dout_dataset=None):
        """
        Args:
            full_dataset (Dataset): The original training dataset D.
            forget_targets (list): Classes or instances to forget.
            target_type (str): 'class' or 'instance'.
            edl_thresholds (dict): Thresholds for EUP (Eq. 9).
            batch_size (int): Batch size for DataLoaders.
            num_workers (int): Number of workers for DataLoaders.
            dout_dataset (Dataset): Optional dataset for D_out (MIA non-members).
        """
        self.full_dataset = full_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dout_dataset = dout_dataset
        
        # Initialize partitioners
        self.base_partitioner = UnlearningDataPartitioner(full_dataset, forget_targets, target_type)
        self.eup_partitioner = EvidentialPartitioner(edl_thresholds if edl_thresholds else {'p': 0.5, 'u': 0.5, 'v': 0.05})
        
        self.dr_subset = self.base_partitioner.get_dr_subset()
        self.df_subset = self.base_partitioner.get_df_subset()
        
        self._dre_subset = None
        self._drc_subset = None
        
    def _create_loader(self, subset, shuffle=True, drop_last=False):
        """Helper method to create a DataLoader from a Subset."""
        return DataLoader(subset, batch_size=self.batch_size, shuffle=shuffle, 
                          num_workers=self.num_workers, drop_last=drop_last, pin_memory=True)
        
    def get_dr_loader(self):
        """Returns DataLoader for D_r."""
        return self._create_loader(self.dr_subset, shuffle=True)
        
    def get_df_loader(self):
        """Returns DataLoader for D_f."""
        return self._create_loader(self.df_subset, shuffle=False)
        
    def get_dre_loader(self, model, device):
        """
        Returns DataLoader for D_r^e. Triggers EUP partitioning if not already done.
        """
        if self._dre_subset is None:
            self.eup_partitioner.partition_retained_set(model, self.dr_subset, self.batch_size, device, self.num_workers)
            self._dre_subset = self.eup_partitioner.get_dre_subset(self.dr_subset)
        return self._create_loader(self._dre_subset, shuffle=True)
        
    def get_drc_loader(self, model, device):
        """
        Returns DataLoader for D_r^c. Triggers EUP partitioning if not already done.
        """
        if self._drc_subset is None:
            self.eup_partitioner.partition_retained_set(model, self.dr_subset, self.batch_size, device, self.num_workers)
            self._drc_subset = self.eup_partitioner.get_drc_subset(self.dr_subset)
        return self._create_loader(self._drc_subset, shuffle=True)
        
    def get_dout_loader(self):
        """
        Returns DataLoader for D_out (MIA non-members).
        If dout_dataset is not provided, raises an error.
        """
        if self.dout_dataset is None:
            raise ValueError("dout_dataset must be provided to create D_out loader for MIA evaluation.")
        return self._create_loader(self.dout_dataset, shuffle=False)