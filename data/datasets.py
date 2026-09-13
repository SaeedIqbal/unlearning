"""
PyTorch Dataset classes for CIFAR-10/100, TinyImageNet, CelebA, LFW, FairFace, UTKFace.
Designed to support the Oracle-Free Evidential Unlearning framework.

Key Methodology Mappings:
- All datasets return the `actual_idx` to support Evidential Uncertainty Partitioning (EUP, Eq. 9), 
  allowing the training loop to map computed vacuity/variance back to specific samples.
- FairFace and UTKFace return demographic attributes to support Demographic Parity Difference (DPD) evaluation.
- CelebA and LFW support instance-level unlearning via identity/attribute filtering.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from PIL import Image

class BaseUnlearningDataset(Dataset):
    """
    Base class for all unlearning datasets.
    Supports dynamic partitioning into D_r (retained), D_f (forgotten), and D_r^e (edge).
    """
    def __init__(self, root_dir, split='train', transform=None, target_transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # To be populated by subclasses: List of tuples containing sample info
        self.samples = []
        self.active_indices = None
        
    def set_partition(self, indices):
        """Sets the active subset of the dataset (e.g., D_r, D_f, or D_r^e)."""
        self.active_indices = np.array(indices)
        
    def reset_partition(self):
        """Resets to the full dataset."""
        self.active_indices = np.arange(len(self.samples))
        
    def __len__(self):
        if self.active_indices is not None:
            return len(self.active_indices)
        return len(self.samples)
        
    def _load_image(self, img_path):
        """Helper to load and convert image to RGB."""
        return Image.open(img_path).convert('RGB')


# ==============================================================================
# 1. CIFAR-10 & CIFAR-100
# ==============================================================================
class CIFAR10UnlearningDataset(BaseUnlearningDataset):
    def __init__(self, root_dir, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root_dir, 'train' if train else 'test', transform, target_transform)
        self.dataset = datasets.CIFAR10(root=root_dir, train=train, download=download)
        self.samples = list(zip(self.dataset.data, self.dataset.targets))
        self.reset_partition()
        
    def __getitem__(self, idx):
        actual_idx = self.active_indices[idx] if self.active_indices is not None else idx
        img_array, target = self.samples[actual_idx]
        
        img = Image.fromarray(img_array)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
            
        return img, target, actual_idx

class CIFAR100UnlearningDataset(BaseUnlearningDataset):
    def __init__(self, root_dir, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root_dir, 'train' if train else 'test', transform, target_transform)
        self.dataset = datasets.CIFAR100(root=root_dir, train=train, download=download)
        self.samples = list(zip(self.dataset.data, self.dataset.targets))
        self.reset_partition()
        
    def __getitem__(self, idx):
        actual_idx = self.active_indices[idx] if self.active_indices is not None else idx
        img_array, target = self.samples[actual_idx]
        
        img = Image.fromarray(img_array)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
            
        return img, target, actual_idx


# ==============================================================================
# 2. TinyImageNet
# ==============================================================================
class TinyImageNetDataset(BaseUnlearningDataset):
    def __init__(self, root_dir, split='train', transform=None, target_transform=None):
        super().__init__(root_dir, split, transform, target_transform)
        self.split_dir = os.path.join(root_dir, split)
        
        self.class_to_idx = {}
        self.samples = []
        
        if split == 'train':
            class_dirs = sorted([d for d in os.listdir(self.split_dir) if os.path.isdir(os.path.join(self.split_dir, d))])
            for idx, c_name in enumerate(class_dirs):
                self.class_to_idx[c_name] = idx
                img_dir = os.path.join(self.split_dir, c_name, 'images')
                for img_name in os.listdir(img_dir):
                    if img_name.endswith('.JPEG'):
                        self.samples.append((os.path.join(img_dir, img_name), idx))
        elif split == 'val':
            val_img_dir = os.path.join(self.split_dir, 'images')
            val_anno_file = os.path.join(self.split_dir, 'val_annotations.txt')
            
            # Load class mapping from train to ensure consistent indices
            train_dir = os.path.join(root_dir, 'train')
            class_dirs = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
            for idx, c_name in enumerate(class_dirs):
                self.class_to_idx[c_name] = idx
                
            with open(val_anno_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    img_name = parts[0]
                    c_name = parts[1]
                    if c_name in self.class_to_idx:
                        self.samples.append((os.path.join(val_img_dir, img_name), self.class_to_idx[c_name]))
                        
        self.reset_partition()

    def __getitem__(self, idx):
        actual_idx = self.active_indices[idx] if self.active_indices is not None else idx
        img_path, target = self.samples[actual_idx]
        
        img = self._load_image(img_path)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
            
        return img, target, actual_idx


# ==============================================================================
# 3. CelebA (Supports Instance-Level Unlearning via Identity)
# ==============================================================================
class CelebADataset(BaseUnlearningDataset):
    def __init__(self, root_dir, split='train', transform=None, target_transform=None, attr_type='identity'):
        super().__init__(root_dir, split, transform, target_transform)
        self.attr_type = attr_type # 'identity' for instance-level, or specific attribute name
        
        img_dir = os.path.join(root_dir, 'img_align_celeba')
        attr_file = os.path.join(root_dir, 'list_attr_celeba.txt')
        identity_file = os.path.join(root_dir, 'identity_CelebA.txt')
        eval_file = os.path.join(root_dir, 'list_eval_partition.txt')
        
        # Load partition
        partition_map = {}
        with open(eval_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                partition_map[parts[0]] = int(parts[1]) # 0: train, 1: val, 2: test
                
        split_idx = {'train': 0, 'val': 1, 'test': 2}[split]
        
        # Load identities
        identity_map = {}
        with open(identity_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                identity_map[parts[0]] = int(parts[1])
                
        # Load attributes
        self.attr_names = []
        self.attr_map = {}
        with open(attr_file, 'r') as f:
            num_samples = int(f.readline().strip())
            self.attr_names = f.readline().strip().split()
            for line in f:
                parts = line.strip().split()
                img_name = parts[0]
                attrs = [int(x) for x in parts[1:]]
                # Convert -1 to 0 for binary classification
                attrs = [1 if x == 1 else 0 for x in attrs]
                self.attr_map[img_name] = attrs
                
        # Build samples
        self.samples = []
        for img_name, p_idx in partition_map.items():
            if p_idx == split_idx:
                img_path = os.path.join(img_dir, img_name)
                if self.attr_type == 'identity':
                    target = identity_map.get(img_name, -1)
                else:
                    if self.attr_type in self.attr_names:
                        attr_idx = self.attr_names.index(self.attr_type)
                        target = self.attr_map[img_name][attr_idx]
                    else:
                        target = 0
                self.samples.append((img_path, target))
                
        self.reset_partition()

    def __getitem__(self, idx):
        actual_idx = self.active_indices[idx] if self.active_indices is not None else idx
        img_path, target = self.samples[actual_idx]
        
        img = self._load_image(img_path)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
            
        return img, target, actual_idx


# ==============================================================================
# 4. LFW (Labeled Faces in the Wild)
# ==============================================================================
class LFWDataset(BaseUnlearningDataset):
    def __init__(self, root_dir, split='train', transform=None, target_transform=None, min_images_per_person=5):
        super().__init__(root_dir, split, transform, target_transform)
        self.split_dir = os.path.join(root_dir, 'lfw')
        
        self.class_to_idx = {}
        self.samples = []
        
        person_dirs = sorted([d for d in os.listdir(self.split_dir) if os.path.isdir(os.path.join(self.split_dir, d))])
        
        for idx, person in enumerate(person_dirs):
            person_dir = os.path.join(self.split_dir, person)
            images = [f for f in os.listdir(person_dir) if f.endswith('.jpg')]
            if len(images) >= min_images_per_person:
                self.class_to_idx[person] = idx
                for img_name in images:
                    self.samples.append((os.path.join(person_dir, img_name), idx))
                    
        self.reset_partition()

    def __getitem__(self, idx):
        actual_idx = self.active_indices[idx] if self.active_indices is not None else idx
        img_path, target = self.samples[actual_idx]
        
        img = self._load_image(img_path)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
            
        return img, target, actual_idx


# ==============================================================================
# 5. FairFace (Supports Demographic Parity Difference - DPD)
# ==============================================================================
class FairFaceDataset(BaseUnlearningDataset):
    def __init__(self, root_dir, split='train', transform=None, target_transform=None, target_attr='race'):
        super().__init__(root_dir, split, transform, target_transform)
        self.target_attr = target_attr
        
        csv_file = os.path.join(root_dir, f'fairface_label_{split}.csv')
        df = pd.read_csv(csv_file)
        
        # Map attributes to indices
        self.attr_to_idx = {}
        if target_attr == 'race':
            races = ['East Asian', 'Indian', 'Black', 'White', 'Middle Eastern', 'Latino_Hispanic', 'Southeast Asian']
            self.attr_to_idx = {r: i for i, r in enumerate(races)}
        elif target_attr == 'gender':
            genders = ['Male', 'Female']
            self.attr_to_idx = {g: i for i, g in enumerate(genders)}
        elif target_attr == 'age':
            ages = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
            self.attr_to_idx = {a: i for i, a in enumerate(ages)}
            
        self.samples = []
        for _, row in df.iterrows():
            img_path = os.path.join(root_dir, row['file'])
            target = self.attr_to_idx.get(row[target_attr], -1)
            
            # Extract all demographic attributes for DPD calculation
            demo_attrs = {
                'race': self.attr_to_idx.get(row.get('race', ''), -1) if target_attr != 'race' else target,
                'gender': self.attr_to_idx.get(row.get('gender', ''), -1) if target_attr != 'gender' else target,
                'age': self.attr_to_idx.get(row.get('age', ''), -1) if target_attr != 'age' else target
            }
            self.samples.append((img_path, target, demo_attrs))
            
        self.reset_partition()
        
    def __getitem__(self, idx):
        actual_idx = self.active_indices[idx] if self.active_indices is not None else idx
        img_path, target, demo_attrs = self.samples[actual_idx]
        
        img = self._load_image(img_path)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
            
        # Return image, target, demographic attributes, and index
        return img, target, demo_attrs, actual_idx


# ==============================================================================
# 6. UTKFace (Supports Demographic Parity Difference - DPD)
# ==============================================================================
class UTKFaceDataset(BaseUnlearningDataset):
    def __init__(self, root_dir, split='train', transform=None, target_transform=None, target_attr='race', train_ratio=0.8):
        super().__init__(root_dir, split, transform, target_transform)
        self.target_attr = target_attr
        
        img_dir = os.path.join(root_dir, 'UTKFace')
        all_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        
        # Sort for reproducible split
        all_files.sort()
        split_idx = int(len(all_files) * train_ratio)
        
        if split == 'train':
            files = all_files[:split_idx]
        else:
            files = all_files[split_idx:]
            
        self.samples = []
        for f in files:
            parts = f.split('_')
            if len(parts) >= 3:
                try:
                    age = int(parts[0])
                    gender = int(parts[1])
                    race = int(parts[2])
                    
                    # Bin age into groups for classification
                    if age < 10: age_bin = 0
                    elif age < 20: age_bin = 1
                    elif age < 30: age_bin = 2
                    elif age < 40: age_bin = 3
                    elif age < 50: age_bin = 4
                    elif age < 60: age_bin = 5
                    else: age_bin = 6
                    
                    demo_attrs = {'age': age_bin, 'gender': gender, 'race': race}
                    
                    if target_attr == 'race': target = race
                    elif target_attr == 'gender': target = gender
                    elif target_attr == 'age': target = age_bin
                    else: target = race
                    
                    self.samples.append((os.path.join(img_dir, f), target, demo_attrs))
                except ValueError:
                    continue
                    
        self.reset_partition()
        
    def __getitem__(self, idx):
        actual_idx = self.active_indices[idx] if self.active_indices is not None else idx
        img_path, target, demo_attrs = self.samples[actual_idx]
        
        img = self._load_image(img_path)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
            
        return img, target, demo_attrs, actual_idx


# ==============================================================================
# Factory Function for Reusability
# ==============================================================================
def get_dataset(dataset_name, root_dir='/home/phd/datasets', split='train', transform=None, target_transform=None, **kwargs):
    """Factory function to instantiate the appropriate unlearning dataset."""
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'cifar10':
        return CIFAR10UnlearningDataset(root_dir=os.path.join(root_dir, 'cifar10'), train=(split=='train'), transform=transform, target_transform=target_transform, **kwargs)
    elif dataset_name == 'cifar100':
        return CIFAR100UnlearningDataset(root_dir=os.path.join(root_dir, 'cifar100'), train=(split=='train'), transform=transform, target_transform=target_transform, **kwargs)
    elif dataset_name == 'tinyimagenet':
        return TinyImageNetDataset(root_dir=os.path.join(root_dir, 'tiny-imagenet-200'), split=split, transform=transform, target_transform=target_transform)
    elif dataset_name == 'celeba':
        return CelebADataset(root_dir=os.path.join(root_dir, 'celeba'), split=split, transform=transform, target_transform=target_transform, **kwargs)
    elif dataset_name == 'lfw':
        return LFWDataset(root_dir=os.path.join(root_dir, 'lfw'), split=split, transform=transform, target_transform=target_transform, **kwargs)
    elif dataset_name == 'fairface':
        return FairFaceDataset(root_dir=os.path.join(root_dir, 'fairface'), split=split, transform=transform, target_transform=target_transform, **kwargs)
    elif dataset_name == 'utkface':
        return UTKFaceDataset(root_dir=os.path.join(root_dir, 'utkface'), split=split, transform=transform, target_transform=target_transform, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")