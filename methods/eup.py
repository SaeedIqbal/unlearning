"""
Evidential Uncertainty Partitioning: Computes u(x) and extracts D_r^e (Eq. 9).
Designed to support the Oracle-Free Evidential Unlearning framework.

Key Methodology Mappings:
- Implements Evidential Uncertainty Partitioning (EUP) to dynamically isolate ambiguous boundary samples.
- Eq. 9: D_r^e = {x in D_r | p_y(x) < tau_p OR u(x) > tau_u OR Var[p_y] > tau_v}
- D_r^c = D_r \ D_r^e
- Eliminates reliance on frozen teachers by using the student's internal Dirichlet vacuity.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset
from typing import Tuple, List, Dict, Any

class EvidentialUncertaintyPartitioner:
    """
    Implements Evidential Uncertainty Partitioning (EUP) to separate the retained set D_r 
    into edge/ambiguous samples D_r^e and confident samples D_r^c based on Eq. 9.
    """
    def __init__(self, thresholds: Dict[str, float]):
        """
        Args:
            thresholds (Dict[str, float]): Dictionary containing the partitioning thresholds.
                - 'tau_p' (float): Threshold for expected probability p_y(x) (Eq. 4).
                - 'tau_u' (float): Threshold for epistemic vacuity u(x) (Eq. 3).
                - 'tau_v' (float): Threshold for predictive variance Var[p_y] (Eq. 5).
        """
        self.tau_p = thresholds.get('tau_p', 0.5)
        self.tau_u = thresholds.get('tau_u', 0.5)
        self.tau_v = thresholds.get('tau_v', 0.05)
        
        self.dre_indices: List[int] = []
        self.drc_indices: List[int] = []
        self.metrics: Dict[int, Dict[str, float]] = {}

    def compute_evidential_metrics(self, model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> None:
        """
        Computes the evidential metrics for all samples in the dataloader.
        Extracts expected probability (Eq. 4), vacuity (Eq. 3), and variance (Eq. 5).
        
        Args:
            model (torch.nn.Module): The Oracle-Free Evidential Model.
            dataloader (DataLoader): DataLoader for the retained set D_r.
            device (torch.device): The device to run the computation on.
        """
        model.eval()
        self.metrics = {}
        
        with torch.no_grad():
            for batch in dataloader:
                # Handle datasets returning (img, target, idx) or (img, target, demo_attrs, idx)
                imgs = batch[0].to(device)
                targets = batch[1].to(device)
                
                # Determine the index based on dataset structure
                if len(batch) == 4:  # FairFace/UTKFace: img, target, demo_attrs, idx
                    indices = batch[3]
                elif len(batch) == 3:  # Standard: img, target, idx
                    indices = batch[2]
                else:
                    indices = torch.arange(imgs.size(0))
                
                # Forward pass through the unified model
                evidence, vacuity, p_mean, p_var = model(imgs)
                
                # Extract metrics for the true class
                for i in range(imgs.size(0)):
                    idx = indices[i].item() if torch.is_tensor(indices[i]) else int(indices[i])
                    target = targets[i].item()
                    
                    # Eq. 4: Expected probability for the true class
                    p_y = p_mean[i, target].item()
                    
                    # Eq. 3: Epistemic vacuity
                    u_x = vacuity[i].item()
                    
                    # Eq. 5: Predictive variance for the true class
                    var_y = p_var[i, target].item()
                    
                    self.metrics[idx] = {
                        'p_y': p_y,
                        'u_x': u_x,
                        'var_y': var_y
                    }

    def partition_retained_set(self, model: torch.nn.Module, retained_dataset: Dataset, 
                               batch_size: int = 128, num_workers: int = 4, device: str = 'cuda') -> Tuple[List[int], List[int]]:
        """
        Partitions the retained set D_r into D_r^e and D_r^c based on Eq. 9:
        D_r^e = {x in D_r | p_y(x) < tau_p OR u(x) > tau_u OR Var[p_y] > tau_v}
        D_r^c = D_r \ D_r^e
        
        Args:
            model (torch.nn.Module): The Oracle-Free Evidential Model.
            retained_dataset (Dataset): The retained dataset D_r.
            batch_size (int): Batch size for the DataLoader.
            num_workers (int): Number of workers for the DataLoader.
            device (str): Device to use ('cuda' or 'cpu').
            
        Returns:
            Tuple[List[int], List[int]]: Indices for D_r^e and D_r^c.
        """
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        dataloader = DataLoader(retained_dataset, batch_size=batch_size, shuffle=False, 
                                num_workers=num_workers, pin_memory=True)
        
        self.compute_evidential_metrics(model, dataloader, device)
        
        self.dre_indices = []
        self.drc_indices = []
        
        for idx, metric_dict in self.metrics.items():
            p_y = metric_dict['p_y']
            u_x = metric_dict['u_x']
            var_y = metric_dict['var_y']
            
            # Apply Eq. 9 conditions
            if p_y < self.tau_p or u_x > self.tau_u or var_y > self.tau_v:
                self.dre_indices.append(idx)
            else:
                self.drc_indices.append(idx)
                
        return self.dre_indices, self.drc_indices

    def get_dre_subset(self, retained_dataset: Dataset) -> Subset:
        """
        Returns the PyTorch Subset for the edge/ambiguous retained set D_r^e.
        
        Args:
            retained_dataset (Dataset): The original retained dataset D_r.
            
        Returns:
            Subset: The D_r^e subset.
        """
        if not self.dre_indices:
            raise ValueError("EUP partitioning has not been performed yet. Call partition_retained_set() first.")
        return Subset(retained_dataset, self.dre_indices)

    def get_drc_subset(self, retained_dataset: Dataset) -> Subset:
        """
        Returns the PyTorch Subset for the confident retained set D_r^c.
        
        Args:
            retained_dataset (Dataset): The original retained dataset D_r.
            
        Returns:
            Subset: The D_r^c subset.
        """
        if not self.drc_indices:
            raise ValueError("EUP partitioning has not been performed yet. Call partition_retained_set() first.")
        return Subset(retained_dataset, self.drc_indices)

    def get_partition_statistics(self) -> Dict[str, Any]:
        """
        Computes and returns statistics about the partitioning.
        
        Returns:
            Dict[str, Any]: Dictionary containing counts and average metrics for D_r^e and D_r^c.
        """
        if not self.metrics:
            return {}
            
        dre_p_y = [self.metrics[idx]['p_y'] for idx in self.dre_indices]
        dre_u_x = [self.metrics[idx]['u_x'] for idx in self.dre_indices]
        dre_var_y = [self.metrics[idx]['var_y'] for idx in self.dre_indices]
        
        drc_p_y = [self.metrics[idx]['p_y'] for idx in self.drc_indices]
        drc_u_x = [self.metrics[idx]['u_x'] for idx in self.drc_indices]
        drc_var_y = [self.metrics[idx]['var_y'] for idx in self.drc_indices]
        
        return {
            'dre_count': len(self.dre_indices),
            'drc_count': len(self.drc_indices),
            'dre_avg_p_y': np.mean(dre_p_y) if dre_p_y else 0.0,
            'dre_avg_u_x': np.mean(dre_u_x) if dre_u_x else 0.0,
            'dre_avg_var_y': np.mean(dre_var_y) if dre_var_y else 0.0,
            'drc_avg_p_y': np.mean(drc_p_y) if drc_p_y else 0.0,
            'drc_avg_u_x': np.mean(drc_u_x) if drc_u_x else 0.0,
            'drc_avg_var_y': np.mean(drc_var_y) if drc_var_y else 0.0
        }