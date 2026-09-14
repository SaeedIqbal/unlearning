"""
Executes all auditing protocols (MIA, CKA, Inversion, DPD, mCE).
Designed to support the Oracle-Free Evidential Unlearning framework.

Key Methodology Mappings:
- MIA (Threat Model 1): Adaptive Membership Inference Attack using loss-based ASR.
- CKA (Threat Model 2): Linear Centered Kernel Alignment for deep representation extraction.
- Inversion (Threat Model 3): Trained Feature-Space Inversion Audit.
- DPD: Demographic Parity Difference for FairFace and UTKFace.
- mCE: Mean Corruption Error for ImageNet-C OOD robustness.
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

# Add project root to path to allow imports from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets import get_dataset
from data.partition import ClassLevelPartitioner, InstanceLevelPartitioner
from models.full_model import get_full_model
from evaluation.mia import AdaptiveMembershipInferenceAttack
from evaluation.cka import LinearCKAAuditor
from evaluation.inversion import FeatureSpaceInversionAudit
from evaluation.fairness import DemographicFairnessEvaluator
from evaluation.ood import OODRobustnessEvaluator
from evaluation.metrics import UnlearningMetricsEvaluator


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy and torch tensor types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return super(NumpyEncoder, self).default(obj)


class ComprehensiveAuditor:
    """
    Orchestrates the comprehensive auditing protocols for the Oracle-Free Evidential Unlearning framework.
    Evaluates utility, privacy (Threat Models 1-3), fairness, and OOD robustness.
    """
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
        # 1. Data Transformations (Evaluation mode: no random augmentations)
        self.eval_transform = self._get_eval_transforms()
        
        # 2. Load Models
        print("Loading models...")
        self.model_unl = self._load_model(args.unlearned_ckpt)
        self.model_orig = self._load_model(args.original_ckpt)
        self.model_retrain = self._load_model(args.retrain_ckpt) if os.path.exists(args.retrain_ckpt) else None
        
        # 3. Datasets and Partitioning (Eq. 1)
        full_dataset = get_dataset(args.dataset, root_dir='/home/phd/datasets', split='train', transform=self.eval_transform)
        
        if args.forget_classes:
            forget_targets = [int(x) for x in args.forget_classes.split(',')]
            partitioner = ClassLevelPartitioner(full_dataset, forget_targets)
            print(f"Class-level audit: Forgetting classes {forget_targets}")
        elif args.forget_instances:
            forget_targets = [int(x) for x in args.forget_instances.split(',')]
            partitioner = InstanceLevelPartitioner(full_dataset, forget_targets)
            print(f"Instance-level audit: Forgetting {len(forget_targets)} instances")
        else:
            raise ValueError("Must specify either --forget_classes or --forget_instances")
            
        self.dr_subset = partitioner.get_dr_subset()
        self.df_subset = partitioner.get_df_subset()
        
        # D_out for MIA (using test/val split as non-members)
        eval_split = 'val' if args.dataset not in ['celeba', 'lfw'] else 'test'
        self.dout_dataset = get_dataset(args.dataset, root_dir='/home/phd/datasets', split=eval_split, transform=self.eval_transform)
        
        # 4. DataLoaders
        self.dr_loader = DataLoader(self.dr_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        self.df_loader = DataLoader(self.df_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        self.dout_loader = DataLoader(self.dout_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        
        # Fairness dataset (if applicable)
        if args.dataset in ['fairface', 'utkface']:
            self.fairness_dataset = get_dataset(args.dataset, root_dir='/home/phd/datasets', split=eval_split, transform=self.eval_transform)
            self.fairness_loader = DataLoader(self.fairness_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        else:
            self.fairness_loader = None

    def _get_eval_transforms(self):
        """Returns deterministic evaluation transformations."""
        if self.args.img_size <= 32:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.args.img_size, self.args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

    def _load_model(self, ckpt_path):
        """Loads a model from a checkpoint file."""
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model checkpoint not found at {ckpt_path}")
        
        model = get_full_model(
            encoder_name=self.args.encoder, 
            num_classes=self.args.num_classes, 
            pretrained=False, 
            img_size=self.args.img_size
        ).to(self.device)
        
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def run_utility(self):
        """Evaluates standard utility metrics (Accuracy, Avg. Gap)."""
        print("\n[1/6] Running Utility Evaluation (Accuracy, Avg. Gap)...")
        if self.model_retrain:
            evaluator = UnlearningMetricsEvaluator(self.model_unl, self.model_retrain, self.device)
            metrics = evaluator.evaluate_utility(self.dr_loader)
            self.results['utility'] = metrics
            print(f"Utility Results: Acc_Unl={metrics['Acc_Unl']:.2f}%, Avg_Gap={metrics['Avg_Gap']:.2f}%")
        else:
            print("Retrain baseline not found. Skipping Avg. Gap computation.")
            evaluator = UnlearningMetricsEvaluator(self.model_unl, self.model_unl, self.device)
            acc = evaluator.compute_accuracy(self.model_unl, self.dr_loader)
            self.results['utility'] = {'Acc_Unl': acc * 100.0}
            print(f"Unlearned Accuracy: {acc * 100.0:.2f}%")

    def run_mia(self):
        """Executes Threat Model 1: Adaptive Membership Inference Attack."""
        print("\n[2/6] Running Membership Inference Attack (MIA)...")
        attack = AdaptiveMembershipInferenceAttack(self.model_unl, self.device)
        results = attack.evaluate(self.df_loader, self.dout_loader)
        self.results['mia'] = results
        print(f"MIA Results: ASR={results['ASR']:.2f}%, Advantage={results['MIA_Advantage']:.4f}")

    def run_cka(self):
        """Executes Threat Model 2: Deep Representation Extraction Audit (CKA)."""
        print("\n[3/6] Running Deep Representation Extraction Audit (CKA)...")
        auditor = LinearCKAAuditor(self.model_orig, self.model_unl, self.device)
        cka_score = auditor.evaluate(self.df_loader)
        self.results['cka'] = {'CKA_Score': cka_score}
        print(f"CKA Score: {cka_score:.4f}")

    def run_inversion(self):
        """Executes Threat Model 3: Trained Feature-Space Model Inversion Audit."""
        print("\n[4/6] Running Feature-Space Model Inversion Audit...")
        img_shape = (3, self.args.img_size, self.args.img_size)
        audit = FeatureSpaceInversionAudit(
            unlearned_model=self.model_unl, 
            feature_dim=self.model_unl.encoder.feature_dim, 
            img_shape=img_shape, 
            device=self.device
        )
        # Use D_r as proxy to train the decoder, D_f as target
        results = audit.evaluate(self.dr_loader, self.df_loader, epochs=self.args.inversion_epochs)
        self.results['inversion'] = results
        print(f"Inversion Results: MSE={results['MSE']:.4f}, Success Rate={results['Inversion_Success_Rate']:.2f}%")

    def run_dpd(self):
        """Evaluates Demographic Parity Difference (DPD) for FairFace and UTKFace."""
        if self.fairness_loader is None:
            print("\n[5/6] Dataset is not FairFace or UTKFace. Skipping DPD evaluation.")
            return
            
        print("\n[5/6] Running Demographic Fairness Evaluation (DPD)...")
        evaluator = DemographicFairnessEvaluator(self.model_unl, self.device)
        results = evaluator.evaluate(self.fairness_loader)
        self.results['fairness'] = results
        for attr, res in results.items():
            print(f"DPD for {attr}: {res['overall_dpd']:.4f}")

    def run_mce(self):
        """Evaluates Out-of-Distribution (OOD) robustness using Mean Corruption Error (mCE) on ImageNet-C."""
        baseline_model = self.model_retrain if self.model_retrain else self.model_orig
        
        print("\n[6/6] Running OOD Robustness Evaluation (mCE) on ImageNet-C...")
        try:
            evaluator = OODRobustnessEvaluator(
                model=self.model_unl, 
                baseline_model=baseline_model, 
                device=self.device,
                corruption_root='/home/phd/datasets/imagenet-c',
                transform=self.eval_transform,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers
            )
            results = evaluator.evaluate_mce()
            self.results['ood'] = {'mCE': results['mCE']}
            print(f"mCE Score: {results['mCE']:.2f}%")
        except FileNotFoundError as e:
            print(f"Skipping mCE evaluation: {e}")
            self.results['ood'] = {'mCE': -1.0, 'error': str(e)}

    def run_all(self):
        """Main orchestration loop for all auditing protocols."""
        print("="*60)
        print("Starting Comprehensive Auditing Protocols")
        print("="*60)
        
        self.run_utility()
        self.run_mia()
        self.run_cka()
        self.run_inversion()
        self.run_dpd()
        self.run_mce()
        
        # Save results to JSON
        os.makedirs(self.args.results_dir, exist_ok=True)
        results_path = os.path.join(self.args.results_dir, 'audit_results.json')
        
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=4, cls=NumpyEncoder)
            
        print(f"\nAudit results saved to {results_path}")
        print("="*60)
        print("Comprehensive Auditing Complete")
        print("="*60)


def parse_args():
    parser = argparse.ArgumentParser(description='Comprehensive Auditing Protocols')
    
    # Dataset and Model Arguments
    parser.add_argument('--dataset', type=str, default='cifar10', 
                        choices=['cifar10', 'cifar100', 'tinyimagenet', 'celeba', 'lfw', 'fairface', 'utkface'])
    parser.add_argument('--encoder', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'vit_s'])
    
    # Unlearning Targets
    parser.add_argument('--forget_classes', type=str, default=None, help='Comma-separated list of class indices to forget')
    parser.add_argument('--forget_instances', type=str, default=None, help='Comma-separated list of instance indices to forget')
    
    # Checkpoint Paths
    parser.add_argument('--original_ckpt', type=str, default='./checkpoints/final_original_model.pth')
    parser.add_argument('--unlearned_ckpt', type=str, default='./checkpoints/final_unlearned_model.pth')
    parser.add_argument('--retrain_ckpt', type=str, default='./checkpoints/final_retrain_baseline_model.pth')
    
    # Evaluation Arguments
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--inversion_epochs', type=int, default=20, help='Epochs to train the adversarial decoder for Inversion Audit')
    parser.add_argument('--results_dir', type=str, default='./results')
    
    args = parser.parse_args()
    
    if not args.forget_classes and not args.forget_instances:
        parser.error("Must specify either --forget_classes or --forget_instances")
        
    # Automatically set num_classes and img_size based on the dataset
    if args.dataset == 'cifar10':
        args.num_classes = 10; args.img_size = 32
    elif args.dataset == 'cifar100':
        args.num_classes = 100; args.img_size = 32
    elif args.dataset == 'tinyimagenet':
        args.num_classes = 200; args.img_size = 64
    elif args.dataset in ['celeba', 'lfw']:
        args.num_classes = 1000; args.img_size = 224
    elif args.dataset == 'fairface':
        args.num_classes = 7; args.img_size = 224
    elif args.dataset == 'utkface':
        args.num_classes = 5; args.img_size = 224
            
    return args


if __name__ == '__main__':
    args = parse_args()
    auditor = ComprehensiveAuditor(args)
    auditor.run_all()