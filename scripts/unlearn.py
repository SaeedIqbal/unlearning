"""
Main entry point for the Oracle-Free Evidential Unlearning pipeline (Eq. 24').
Designed to support the Oracle-Free Evidential Unlearning framework.

Key Methodology Mappings:
- Eq. 24': Total training objective L_total = L_EUP + lambda_1 L_forget + lambda_2 L_FSBR + lambda_3 L_trust_region.
- Orchestrates the full unlearning pipeline: Data Partitioning (EUP, Eq. 9), 
  Forgetting Manifold Statistics (Eq. 10), Adversarial Centroid Synthesis (Eqs. 11-14), 
  and the unified optimization of the unlearned model.
- Executes the comprehensive Threat Model evaluation (MIA, CKA, Inversion) and PFC validation.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

# Add project root to path to allow imports from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets import get_dataset
from data.partition import ClassLevelPartitioner, InstanceLevelPartitioner
from models.full_model import get_full_model, OracleFreeEvidentialModel
from methods.eup import EvidentialUncertaintyPartitioner
from methods.fsbr import ForgettingManifoldStatistics, AdversarialCentroidSynthesizer
from methods.liav import LocalizedInfluenceAwareVacuity
from methods.pfc import EncoderDriftTracker, ProbabilisticForgettingCertification
from losses.evidential import EvidentialPartitioningLoss
from losses.forgetting import EvidentialForgettingLoss
from losses.repulsion import FSBRMarginLoss
from losses.trust_region import TrustRegionPenalty
from evaluation.mia import AdaptiveMembershipInferenceAttack
from evaluation.cka import LinearCKAAuditor
from evaluation.inversion import FeatureSpaceInversionAudit
from evaluation.metrics import UnlearningMetricsEvaluator
from utils.logging import UnlearningLogger
from utils.checkpointing import CheckpointManager


def get_infinite_loader(loader):
    """Helper to create an infinite iterator over a DataLoader."""
    while True:
        for batch in loader:
            yield batch


class OracleFreeUnlearner:
    """
    Orchestrates the complete Oracle-Free Evidential Unlearning pipeline.
    """
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Data Transformations
        self.train_transform = self._get_train_transforms()
        self.eval_transform = self._get_eval_transforms()
        
        # 2. Datasets and Partitioning (Eq. 1)
        full_dataset = get_dataset(args.dataset, root_dir='/home/phd/datasets', split='train', transform=self.train_transform)
        
        if args.forget_classes:
            forget_targets = [int(x) for x in args.forget_classes.split(',')]
            partitioner = ClassLevelPartitioner(full_dataset, forget_targets)
            print(f"Class-level unlearning: Forgetting classes {forget_targets}")
        elif args.forget_instances:
            forget_targets = [int(x) for x in args.forget_instances.split(',')]
            partitioner = InstanceLevelPartitioner(full_dataset, forget_targets)
            print(f"Instance-level unlearning: Forgetting {len(forget_targets)} instances")
        else:
            raise ValueError("Must specify either --forget_classes or --forget_instances")
            
        self.dr_subset = partitioner.get_dr_subset()
        self.df_subset = partitioner.get_df_subset()
        
        # D_out for MIA evaluation (using test/val split as non-members)
        eval_split = 'val' if args.dataset not in ['celeba', 'lfw'] else 'test'
        self.dout_dataset = get_dataset(args.dataset, root_dir='/home/phd/datasets', split=eval_split, transform=self.eval_transform)
        
        self.df_loader = DataLoader(self.df_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        self.dout_loader = DataLoader(self.dout_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        
        # 3. Model Initialization
        # Load the original trained model
        self.model = get_full_model(encoder_name=args.encoder, num_classes=args.num_classes, pretrained=False, img_size=args.img_size).to(self.device)
        self._load_original_model()
        
        # Clone original model for CKA and Metrics evaluation
        self.model_orig = get_full_model(encoder_name=args.encoder, num_classes=args.num_classes, pretrained=False, img_size=args.img_size).to(self.device)
        self._load_original_model(model_to_load=self.model_orig)
        self.model_orig.eval()
        for param in self.model_orig.parameters():
            param.requires_grad = False
            
        # 4. Core Methodologies Initialization
        # EUP (Eq. 9)
        self.eup_partitioner = EvidentialUncertaintyPartitioner(thresholds={'tau_p': 0.5, 'tau_u': 0.5, 'tau_v': 0.05})
        
        # FSBR (Eqs. 10-16)
        self.manifold_stats = ForgettingManifoldStatistics(feature_dim=self.model.encoder.feature_dim, momentum=0.9, lambda_reg=1e-4)
        self.centroid_synthesizer = AdversarialCentroidSynthesizer(
            manifold_stats=self.manifold_stats, edl_head=self.model.edl_head, 
            num_classes=args.num_classes, num_steps=10, step_size=0.1, rho=1.5
        )
        
        # LIAV (Eqs. 17-23) - Only for instance-level
        self.liav_module = None
        if args.forget_instances:
            # LIAV requires a dataloader for the retained set to compute the empirical Hessian
            temp_dr_loader = DataLoader(self.dr_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            self.liav_module = LocalizedInfluenceAwareVacuity(
                model=self.model, retained_dataloader=temp_dr_loader, device=self.device, 
                damping=1e-3, krylov_dim=args.krylov_dim, loo_threshold=0.85
            )
            
        # PFC (Eqs. 27-28)
        self.drift_tracker = EncoderDriftTracker(self.model_orig.encoder)
        self.pfc_module = ProbabilisticForgettingCertification(num_classes=args.num_classes, drift_tracker=self.drift_tracker, drift_threshold=100.0)
        
        # 5. Losses Initialization (Eq. 24')
        self.loss_eup = EvidentialPartitioningLoss(num_classes=args.num_classes, lambda_kl=args.lambda_kl)
        self.loss_forget = EvidentialForgettingLoss(manifold_stats=self.manifold_stats, num_classes=args.num_classes, feature_dim=self.model.encoder.feature_dim, sigma=0.1, num_mc_samples=4, device=str(self.device))
        self.loss_fsbr = FSBRMarginLoss(manifold_stats=self.manifold_stats, margin=1.0)
        self.loss_trust = TrustRegionPenalty(original_encoder=self.model_orig.encoder, lambda_4=args.lambda_trust)
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.unlearn_lr, weight_decay=args.weight_decay)
        
        # 6. Logging and Checkpointing
        forget_str = args.forget_classes if args.forget_classes else f"inst_{len(forget_targets)}"
        run_name = f"unlearn_{args.dataset}_{forget_str}"
        self.logger = UnlearningLogger(log_dir=os.path.join(args.log_dir, args.dataset, args.encoder), project_name='oracle_free_unlearning', run_name=run_name)
        self.checkpoint_manager = CheckpointManager(save_dir=args.save_dir)
        
        # DataLoaders for EUP partitions (initialized after first EUP pass)
        self.dre_loader = None
        self.drc_loader = None

    def _get_train_transforms(self):
        if self.args.img_size <= 32:
            return transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        else:
            return transforms.Compose([transforms.Resize((self.args.img_size, self.args.img_size)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def _get_eval_transforms(self):
        if self.args.img_size <= 32:
            return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        else:
            return transforms.Compose([transforms.Resize((self.args.img_size, self.args.img_size)), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def _load_original_model(self, model_to_load=None):
        """Loads the pre-trained original model weights."""
        target_model = model_to_load if model_to_load else self.model
        ckpt_path = os.path.join(self.args.save_dir, 'final_original_model.pth')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Original model not found at {ckpt_path}. Please run train_original.py first.")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        target_model.load_state_dict(checkpoint['model_state_dict'])

    def _initialize_eup_partitions(self):
        """Executes EUP (Eq. 9) to partition D_r into D_r^e and D_r^c."""
        print("Executing Evidential Uncertainty Partitioning (EUP, Eq. 9)...")
        dre_indices, drc_indices = self.eup_partitioner.partition_retained_set(self.model, self.dr_subset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, device=str(self.device))
        
        self.dre_loader = DataLoader(Subset(self.dr_subset, dre_indices), batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, pin_memory=True)
        self.drc_loader = DataLoader(Subset(self.dr_subset, drc_indices), batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, pin_memory=True)
        print(f"EUP Complete: D_r^e size = {len(dre_indices)}, D_r^c size = {len(drc_indices)}")

    def _update_manifold_stats(self):
        """Updates the forgetting manifold statistics μ_f and Σ_f (Eq. 10) using D_f."""
        self.model.eval()
        with torch.no_grad():
            for batch in self.df_loader:
                imgs = batch[0].to(self.device)
                h_f = self.model.get_latent_features(imgs)
                self.manifold_stats.update(h_f)
        self.model.train()

    def unlearn_epoch(self, epoch):
        """Executes one epoch of the unlearning optimization (Eq. 24')."""
        self.model.train()
        
        if self.dre_loader is None or self.drc_loader is None:
            self._initialize_eup_partitions()
            
        # Update forgetting manifold statistics at the start of the epoch (Eq. 10')
        self._update_manifold_stats()
        
        dre_iter = get_infinite_loader(self.dre_loader)
        drc_iter = get_infinite_loader(self.drc_loader)
        
        running_loss = 0.0
        num_steps = len(self.drc_loader) # Base number of steps on the confident set size
        
        for step in range(num_steps):
            self.optimizer.zero_grad()
            
            # 1. Sample batches
            batch_drc = next(drc_iter)
            imgs_drc = batch_drc[0].to(self.device)
            targets_drc = batch_drc[1].to(self.device)
            
            batch_dre = next(dre_iter)
            imgs_dre = batch_dre[0].to(self.device)
            
            # 2. Forward pass for D_r^c to compute L_EUP (Eq. 8)
            outputs_drc = self.model(imgs_drc)
            alpha_drc = outputs_drc[0] + 1.0
            loss_eup = self.loss_eup(alpha_drc, targets_drc)
            
            # 3. Synthesize adversarial centroid z_adv* (Eqs. 11-14)
            with torch.no_grad():
                h_f_mean = self.manifold_stats.mu_f.to(self.device)
                z_adv = self.centroid_synthesizer.synthesize(h_f_mean)
                
            # 4. Forward pass for D_r^e to compute L_FSBR (Eq. 15)
            h_dre = self.model.get_latent_features(imgs_dre)
            loss_fsbr = self.loss_fsbr(h_dre, z_adv)
            
            # 5. Compute L_forget (Eq. 25)
            loss_forget = self.loss_forget(self.model.edl_head)
            
            # 6. Compute L_trust_region (Eq. 24')
            loss_trust = self.loss_trust(self.model.encoder)
            
            # 7. Total Loss (Eq. 24)
            total_loss = (loss_eup + 
                          self.args.lambda_forget * loss_forget + 
                          self.args.lambda_fsbr * loss_fsbr + 
                          loss_trust)
                          
            total_loss.backward()
            self.optimizer.step()
            
            running_loss += total_loss.item()
            
            if step % self.args.log_interval == 0:
                print(f'Epoch [{epoch+1}/{self.args.unlearn_epochs}] Step [{step}/{num_steps}] '
                      f'Total: {total_loss.item():.4f} | EUP: {loss_eup.item():.4f} | '
                      f'Forget: {loss_forget.item():.4f} | FSBR: {loss_fsbr.item():.4f} | Trust: {loss_trust.item():.4f}')
                      
        # Log epoch losses
        self.logger.log_training_losses(
            loss_eup=loss_eup.item(), loss_forget=loss_forget.item(), 
            loss_fsbr=loss_fsbr.item(), loss_trust_region=loss_trust.item(), 
            total_loss=running_loss / num_steps, step=epoch
        )
        
        # Log Evidential metrics
        stats = self.eup_partitioner.get_partition_statistics()
        if stats:
            self.logger.log_evidential_metrics(
                vacuity_mean=stats['dre_avg_u_x'], variance_mean=stats['dre_avg_var_y'], 
                evidence_mean=0.0, step=epoch # Evidence mean not directly tracked in stats dict
            )

    def evaluate(self):
        """Executes the comprehensive Threat Model evaluation and PFC validation."""
        print("Starting comprehensive evaluation...")
        self.model.eval()
        
        # 1. Utility Metrics (Accuracy, Avg. Gap)
        # Note: Avg. Gap requires the exact retrained baseline. If not available, we skip it.
        retrain_path = os.path.join(self.args.save_dir, 'final_retrain_baseline_model.pth')
        if os.path.exists(retrain_path):
            model_retrain = get_full_model(encoder_name=self.args.encoder, num_classes=self.args.num_classes, pretrained=False, img_size=self.args.img_size).to(self.device)
            ckpt = torch.load(retrain_path, map_location=self.device)
            model_retrain.load_state_dict(ckpt['model_state_dict'])
            model_retrain.eval()
        else:
            print("Warning: Retrained baseline not found. Skipping Avg. Gap computation.")
            model_retrain = None
            
        # We need a dataloader for D_r to evaluate utility
        dr_loader = DataLoader(self.dr_subset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)
        
        if model_retrain:
            metrics_eval = UnlearningMetricsEvaluator(self.model, model_retrain, self.device)
            utility_metrics = metrics_eval.evaluate_utility(dr_loader)
            self.logger.log_utility_metrics(utility_metrics['Acc_Retrain'], utility_metrics['Acc_Unl'], utility_metrics['Avg_Gap'])
            print(f"Utility: Acc_Unl={utility_metrics['Acc_Unl']:.2f}%, Avg_Gap={utility_metrics['Avg_Gap']:.2f}%")
        
        # 2. Privacy Audits (Threat Model)
        # MIA (Threat Model 1)
        mia_attack = AdaptiveMembershipInferenceAttack(self.model, self.device)
        mia_results = mia_attack.evaluate(self.df_loader, self.dout_loader)
        self.logger.log_privacy_metrics(cka_score=0.0, mia_asr=mia_results['ASR'], mia_advantage=mia_results['MIA_Advantage'], inversion_success_rate=0.0)
        print(f"MIA: ASR={mia_results['ASR']:.2f}%, Advantage={mia_results['MIA_Advantage']:.4f}")
        
        # CKA (Threat Model 2)
        cka_auditor = LinearCKAAuditor(self.model_orig, self.model, self.device)
        cka_score = cka_auditor.evaluate(self.df_loader)
        print(f"CKA (Deep Representation Extraction): {cka_score:.4f}")
        
        # Inversion (Threat Model 3)
        img_shape = (3, self.args.img_size, self.args.img_size)
        inversion_audit = FeatureSpaceInversionAudit(self.model, self.model.encoder.feature_dim, img_shape, self.device)
        # Use D_r as proxy to train the decoder, and D_f as target
        dr_proxy_loader = DataLoader(self.dr_subset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
        inv_results = inversion_audit.evaluate(dr_proxy_loader, self.df_loader, epochs=5)
        print(f"Inversion: MSE={inv_results['MSE']:.4f}, Success Rate={inv_results['Inversion_Success_Rate']:.2f}%")
        
        # Update privacy logs with CKA and Inversion
        self.logger.log_privacy_metrics(cka_score=cka_score, mia_asr=mia_results['ASR'], mia_advantage=mia_results['MIA_Advantage'], inversion_success_rate=inv_results['Inversion_Success_Rate'])
        
        # 3. PFC Validation (Eqs. 27-28)
        pfc_results = self.pfc_module.validate_bounds(self.model, self.df_loader, self.device)
        mia_corr = self.pfc_module.correlate_with_mia(mia_results['ASR'])
        print(f"PFC: Certified={pfc_results['certified']}, Epsilon={pfc_results['epsilon']:.4f}, Drift={pfc_results['encoder_drift_sq']:.2f}")
        
        return {
            'utility': utility_metrics if model_retrain else {},
            'privacy': {'MIA_ASR': mia_results['ASR'], 'CKA': cka_score, 'Inversion_SR': inv_results['Inversion_Success_Rate']},
            'pfc': pfc_results
        }

    def unlearn(self):
        """Main orchestration loop for the unlearning pipeline."""
        print(f"Starting Oracle-Free Evidential Unlearning on {self.args.dataset}...")
        
        for epoch in range(self.args.unlearn_epochs):
            self.unlearn_epoch(epoch)
            
        print("Unlearning complete. Running final evaluation...")
        final_metrics = self.evaluate()
        
        # Save final unlearned model
        self.save_model(filename='final_unlearned_model.pth', metrics=final_metrics)
        self.logger.close()

    def save_model(self, filename='final_unlearned_model.pth', metrics=None):
        """Saves the unlearned model and its evaluation metrics."""
        # Create a dummy EMA updater for the checkpoint manager to store the manifold stats
        from utils.checkpointing import ForgettingManifoldEMA
        ema_updater = ForgettingManifoldEMA(feature_dim=self.model.encoder.feature_dim)
        ema_updater.mu_f = self.manifold_stats.mu_f.cpu()
        ema_updater.Sigma_f = self.manifold_stats.Sigma_f.cpu()
        ema_updater.is_initialized = True
        
        self.checkpoint_manager.save_checkpoint(
            epoch=self.args.unlearn_epochs,
            model=self.model,
            optimizer=self.optimizer,
            ema_updater=ema_updater,
            metrics=metrics if metrics else {},
            filename=filename
        )
        print(f"Unlearned model saved to {os.path.join(self.args.save_dir, filename)}")


def parse_args():
    parser = argparse.ArgumentParser(description='Oracle-Free Evidential Unlearning Pipeline (Eq. 24')')
    
    # Dataset and Model Arguments
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'tinyimagenet', 'celeba', 'lfw', 'fairface', 'utkface'])
    parser.add_argument('--encoder', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'vit_s'])
    
    # Unlearning Targets
    parser.add_argument('--forget_classes', type=str, default=None, help='Comma-separated list of class indices to forget')
    parser.add_argument('--forget_instances', type=str, default=None, help='Comma-separated list of instance indices to forget')
    parser.add_argument('--krylov_dim', type=int, default=10, help='Dimension m of the Krylov subspace for LIAV (Eq. 20)')
    
    # Unlearning Hyperparameters (Eq. 24)
    parser.add_argument('--unlearn_epochs', type=int, default=20)
    parser.add_argument('--unlearn_lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lambda_kl', type=float, default=0.1, help='Weight for KL divergence (Eq. 8)')
    parser.add_argument('--lambda_forget', type=float, default=1.0, help='Weight for L_forget (Eq. 25)')
    parser.add_argument('--lambda_fsbr', type=float, default=1.0, help='Weight for L_FSBR (Eq. 15)')
    parser.add_argument('--lambda_trust', type=float, default=0.01, help='Weight for L_trust_region (Eq. 24')')
    
    # System Arguments
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    
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
    unlearner = OracleFreeUnlearner(args)
    unlearner.unlearn()