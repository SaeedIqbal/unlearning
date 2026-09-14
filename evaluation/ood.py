"""
Mean Corruption Error (mCE) computation for ImageNet-C.
Designed to evaluate the Out-of-Distribution (OOD) robustness of the Oracle-Free Evidential Unlearning framework.

Key Methodology Mappings:
- Implements the Mean Corruption Error (mCE) metric to evaluate OOD robustness across 15 corruptions 
  and 5 severity levels, ensuring that the evidential vacuity maximization does not induce chaotic 
  logit divergence under distributional shift.
- Eq. (mCE): mCE = (1/15) * sum_{c=1}^{15} (1/5) * sum_{s=1}^{5} (Error_c^s(M) / Error_c^s(M_base)) * 100%
- Error_c^s(M) is the top-1 error rate of the unlearned model M on corruption c at severity s.
- Error_c^s(M_base) is the top-1 error rate of the baseline model (e.g., exact retraining) on the same corruption and severity.
- Uses the original ImageNet-C dataset saved in /home/phd/datasets/imagenet-c.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Dict, List, Tuple, Any

# Standard 15 ImageNet-C corruptions
IMAGENET_C_CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
]

class OODRobustnessEvaluator:
    """
    Evaluates the Out-of-Distribution (OOD) robustness of the unlearned model using the 
    Mean Corruption Error (mCE) metric on the ImageNet-C dataset.
    """
    def __init__(self, model: nn.Module, baseline_model: nn.Module, device: torch.device, 
                 corruption_root: str = '/home/phd/datasets/imagenet-c', 
                 transform: transforms.Compose = None, batch_size: int = 64, num_workers: int = 4):
        """
        Args:
            model (nn.Module): The unlearned Oracle-Free Evidential Model.
            baseline_model (nn.Module): The baseline model (e.g., exact retraining) for normalization.
            device (torch.device): Computation device.
            corruption_root (str): Root directory of the ImageNet-C dataset.
            transform (transforms.Compose): Image transformations (must match the model's expected input).
            batch_size (int): Batch size for evaluation.
            num_workers (int): Number of workers for the DataLoader.
        """
        self.model = model
        self.baseline_model = baseline_model
        self.device = device
        self.corruption_root = corruption_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        if transform is None:
            # Default ImageNet evaluation transform
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def _get_dataloader(self, corruption_name: str, severity: int) -> DataLoader:
        """
        Creates a DataLoader for a specific corruption and severity level.
        Assumes the standard ImageNet-C directory structure: root/corruption_name/severity/
        
        Args:
            corruption_name (str): Name of the corruption (e.g., 'gaussian_noise').
            severity (int): Severity level (1 to 5).
            
        Returns:
            DataLoader: PyTorch DataLoader for the specified corruption and severity.
        """
        path = os.path.join(self.corruption_root, corruption_name, str(severity))
        if not os.path.exists(path):
            raise FileNotFoundError(f"Corruption path not found: {path}. Ensure ImageNet-C is correctly structured.")
        
        dataset = datasets.ImageFolder(path, transform=self.transform)
        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            pin_memory=True
        )

    def compute_top1_error(self, model: nn.Module, dataloader: DataLoader) -> float:
        """
        Computes the top-1 error rate for a given model and dataloader.
        Error = 1 - (Correct Predictions / Total Samples)
        
        Args:
            model (nn.Module): The model to evaluate.
            dataloader (DataLoader): DataLoader for the target dataset.
            
        Returns:
            float: Top-1 error rate in [0, 1].
        """
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for imgs, targets in dataloader:
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = model(imgs)
                
                # Handle the Oracle-Free Evidential Model output format
                if isinstance(outputs, tuple) and len(outputs) == 4:
                    p_mean = outputs[2]  # Expected probability (Eq. 4)
                else:
                    p_mean = outputs
                    
                _, predicted = torch.max(p_mean, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
        error_rate = 1.0 - (correct / total) if total > 0 else 0.0
        return error_rate

    def evaluate_corruption(self, corruption_name: str, severity: int) -> Tuple[float, float, float]:
        """
        Evaluates a specific corruption at a specific severity level.
        
        Args:
            corruption_name (str): Name of the corruption.
            severity (int): Severity level (1 to 5).
            
        Returns:
            Tuple[float, float, float]: (Error_M, Error_M_base, CE_c_s)
        """
        dataloader = self._get_dataloader(corruption_name, severity)
        
        # Compute error for the unlearned model
        error_m = self.compute_top1_error(self.model, dataloader)
        
        # Compute error for the baseline model
        error_base = self.compute_top1_error(self.baseline_model, dataloader)
        
        # Compute Corruption Error for this severity: Error_c^s(M) / Error_c^s(M_base)
        ce_c_s = error_m / (error_base + 1e-8)
        
        return error_m, error_base, ce_c_s

    def evaluate_mce(self) -> Dict[str, Any]:
        """
        Computes the Mean Corruption Error (mCE) across all 15 corruptions and 5 severities.
        Eq: mCE = (1/15) * sum_{c=1}^{15} (1/5) * sum_{s=1}^{5} (Error_c^s(M) / Error_c^s(M_base)) * 100%
        
        Returns:
            Dict[str, Any]: Detailed results including per-severity errors, per-corruption CE, and final mCE.
        """
        mce_sum = 0.0
        detailed_results = {}
        
        for c_name in IMAGENET_C_CORRUPTIONS:
            ce_c_sum = 0.0
            corruption_details = {}
            
            for s in range(1, 6):
                error_m, error_base, ce_c_s = self.evaluate_corruption(c_name, s)
                ce_c_sum += ce_c_s
                
                corruption_details[f'severity_{s}'] = {
                    'error_M': error_m,
                    'error_M_base': error_base,
                    'CE_c_s': ce_c_s
                }
                
            # Average Corruption Error for this corruption: (1/5) * sum_{s=1}^{5} CE_c_s
            ce_c = ce_c_sum / 5.0
            mce_sum += ce_c
            
            detailed_results[c_name] = {
                'severities': corruption_details,
                'CE_c': ce_c
            }
            
        # Final mCE: (1/15) * sum_{c=1}^{15} CE_c * 100%
        mce = (mce_sum / len(IMAGENET_C_CORRUPTIONS)) * 100.0
        
        detailed_results['mCE'] = mce
        return detailed_results