"""
Logic for class-level, instance-level, and sequential unlearning splits.
Designed to support the Oracle-Free Evidential Unlearning framework.

Key Methodology Mappings:
- ClassLevelPartitioner: Implements Eq. 1 for class-level unlearning, where D_f consists of all samples 
  belonging to the specified forget_classes.
- InstanceLevelPartitioner: Implements Eq. 1 for instance-level unlearning (e.g., GDPR Right to be Forgotten), 
  where D_f consists of specific instances (e.g., specific identities in CelebA/LFW).
- SequentialPartitioner: Implements the sequential unlearning protocol over T rounds, 
  updating D_r^(t) = D \ U_{i=1}^t D_f^(i) and ensuring warm-start initialization theta^(t)_init = theta^(t-1)_final.
"""

import os
import numpy as np
import torch
from torch.utils.data import Subset, Dataset
from typing import List, Dict, Any, Union, Tuple, Set

class BasePartitioner:
    """
    Base class for unlearning partitioners.
    Provides foundational methods for subset extraction and index management.
    """
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.dr_indices: List[int] = []
        self.df_indices: List[int] = []
        
    def get_dr_subset(self) -> Subset:
        """Returns the PyTorch Subset for the retained set D_r."""
        return Subset(self.dataset, self.dr_indices)
        
    def get_df_subset(self) -> Subset:
        """Returns the PyTorch Subset for the forgetting set D_f."""
        return Subset(self.dataset, self.df_indices)
        
    def get_indices(self) -> Tuple[List[int], List[int]]:
        """Returns the raw index lists for D_r and D_f."""
        return self.dr_indices, self.df_indices


class ClassLevelPartitioner(BasePartitioner):
    """
    Partitions the dataset for class-level unlearning.
    Implements Eq. 1: D = D_r U D_f, where D_f contains all samples belonging to the specified forget_classes.
    """
    def __init__(self, dataset: Dataset, forget_classes: List[int]):
        super().__init__(dataset)
        self.forget_classes = set(forget_classes)
        self._partition()
        
    def _partition(self):
        """
        Iterates through the dataset to separate D_r and D_f based on class labels.
        """
        self.dr_indices = []
        self.df_indices = []
        
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            # Assuming dataset returns (img, target, ...) or (img, target, demo_attrs, ...)
            target = sample[1]
            
            if target in self.forget_classes:
                self.df_indices.append(idx)
            else:
                self.dr_indices.append(idx)


class InstanceLevelPartitioner(BasePartitioner):
    """
    Partitions the dataset for instance-level unlearning (e.g., GDPR compliance).
    Implements Eq. 1: D = D_r U D_f, where D_f contains specific instances (e.g., specific identities).
    """
    def __init__(self, dataset: Dataset, forget_instances: List[int]):
        super().__init__(dataset)
        self.forget_instances = set(forget_instances)
        self._partition()
        
    def _partition(self):
        """
        Iterates through the dataset to separate D_r and D_f based on specific instance indices.
        """
        self.dr_indices = []
        self.df_indices = []
        
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            # For instance-level, the actual_idx is typically the 3rd element (index 2) 
            # as defined in our custom datasets.py
            actual_idx = sample[2] if len(sample) > 2 else idx
            
            if actual_idx in self.forget_instances:
                self.df_indices.append(idx)
            else:
                self.dr_indices.append(idx)


class SequentialPartitioner:
    """
    Manages sequential unlearning over T rounds.
    Implements the sequential protocol where in round t, the model is warm-started from theta^(t-1)_final.
    The retained set is updated as: D_r^(t) = D \ U_{i=1}^t D_f^(i).
    """
    def __init__(self, dataset: Dataset, forget_targets_sequence: List[List[int]], target_type: str = 'class'):
        """
        Args:
            dataset (Dataset): The original training dataset D.
            forget_targets_sequence (List[List[int]]): A list of lists, where each inner list contains 
                                                       the targets (classes or instances) to forget in round t.
            target_type (str): 'class' or 'instance'.
        """
        self.dataset = dataset
        self.forget_targets_sequence = forget_targets_sequence
        self.target_type = target_type
        self.num_rounds = len(forget_targets_sequence)
        
        # Store partitioners for each round
        self.round_partitioners: List[BasePartitioner] = []
        self.cumulative_forgotten_indices: Set[int] = set()
        
        self._compute_sequential_partitions()
        
    def _compute_sequential_partitions(self):
        """
        Computes the D_r^(t) and D_f^(t) for each round t.
        Ensures that once an instance/class is forgotten in round t, it remains excluded in subsequent rounds.
        """
        all_indices = set(range(len(self.dataset)))
        
        for t in range(self.num_rounds):
            forget_targets = set(self.forget_targets_sequence[t])
            
            # The true D_r^(t-1) before this round's forgetting
            current_dr_indices = all_indices - self.cumulative_forgotten_indices
            
            round_df_indices = []
            round_dr_indices = []
            
            for idx in current_dr_indices:
                sample = self.dataset[idx]
                target = sample[1]
                actual_idx = sample[2] if len(sample) > 2 else idx
                
                is_forgotten = False
                if self.target_type == 'class' and target in forget_targets:
                    is_forgotten = True
                elif self.target_type == 'instance' and actual_idx in forget_targets:
                    is_forgotten = True
                    
                if is_forgotten:
                    round_df_indices.append(idx)
                    self.cumulative_forgotten_indices.add(idx)
                else:
                    round_dr_indices.append(idx)
                    
            # Create a custom partitioner object to store these specific indices for round t
            custom_partitioner = BasePartitioner(self.dataset)
            custom_partitioner.dr_indices = round_dr_indices
            custom_partitioner.df_indices = round_df_indices
            self.round_partitioners.append(custom_partitioner)
            
    def get_round_partitioner(self, round_idx: int) -> BasePartitioner:
        """Returns the partitioner for a specific round t."""
        if round_idx < 0 or round_idx >= self.num_rounds:
            raise IndexError(f"Round index {round_idx} out of bounds. Total rounds: {self.num_rounds}")
        return self.round_partitioners[round_idx]
        
    def get_dr_subset_for_round(self, round_idx: int) -> Subset:
        """Returns the retained subset D_r^(t) for round t."""
        return self.get_round_partitioner(round_idx).get_dr_subset()
        
    def get_df_subset_for_round(self, round_idx: int) -> Subset:
        """Returns the forgetting subset D_f^(t) for round t."""
        return self.get_round_partitioner(round_idx).get_df_subset()
        
    def get_cumulative_forgotten_indices(self) -> Set[int]:
        """Returns the set of all indices forgotten up to the current state."""
        return self.cumulative_forgotten_indices