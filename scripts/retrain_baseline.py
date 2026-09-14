"""
Exact unlearning upper bound (training from scratch on D_r).
Designed to support the Oracle-Free Evidential Unlearning framework.

Key Methodology Mappings:
- Eq. 1: theta_retrain = argmin_theta L(theta; D_r)
- Trains the Oracle-Free Evidential Model from scratch strictly on the retained set D_r.
- The resulting model theta_retrain serves as the exact unlearning upper bound (gold standard).
- Uses the Expected Evidential Cross-Entropy (Eq. 6) and KL regularization (Eq. 7) 
  to ensure the retrained model is properly calibrated and evidential, providing a fair 
  comparison baseline for the approximate unlearning methods.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

# Add project root to path to allow imports from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets import get_dataset
from data.partition import ClassLevelPartitioner, InstanceLevelPartitioner
from models.full_model import get_full_model
from losses.evidential import EvidentialPartitioningLoss
from utils.logging import UnlearningLogger
from utils.checkpointing import CheckpointManager, ForgettingManifoldEMA


class RetrainBaselineTrainer:
    """
    Orchestrates the exact retraining of the Oracle-Free Evidential Model 
    from scratch on the retained set D_r (Eq. 1).
    """
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_acc = 0.0
        
        # 1. Data Transformations
        self.train_transform = self._get_train_transforms()
        self.eval_transform = self._get_eval_transforms()
        
        # 2. Load Full Dataset and Partition into D_r and D_f
        full_dataset = get_dataset(
            args.dataset, 
            root_dir='/home/phd/datasets', 
            split='train', 
            transform=self.train_transform
        )
        
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
            
        # Extract the retained subset D_r (Eq. 1)
        self.dr_subset = partitioner.get_dr_subset()
        print(f"Retained set D_r size: {len(self.dr_subset)} | Forgotten set D_f size: {len(full_dataset) - len(self.dr_subset)}")
        
        self.dr_loader = DataLoader(
            self.dr_subset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers, 
            pin_memory=True
        )
        
        # Evaluation dataset (standard validation/test split)
        eval_split = 'val' if args.dataset not in ['celeba', 'lfw'] else 'test'
        self.eval_dataset = get_dataset(
            args.dataset, 
            root_dir='/home/phd/datasets', 
            split=eval_split, 
            transform=self.eval_transform
        )
        self.eval_loader = DataLoader(
            self.eval_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers, 
            pin_memory=True
        )
        
        # 3. Model Initialization (Fresh weights for EDL head, optional pre-trained encoder)
        self.model = get_full_model(
            encoder_name=args.encoder,
            num_classes=args.num_classes,
            pretrained=args.pretrained,
            img_size=args.img_size
        ).to(self.device)
        
        # 4. Loss and Optimizer
        # Using Evidential Partitioning Loss (Eq. 8) to ensure proper evidential calibration
        self.criterion = EvidentialPartitioningLoss(num_classes=args.num_classes, lambda_kl=args.lambda_kl)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
        
        # 5. Logging and Checkpointing
        forget_str = args.forget_classes if args.forget_classes else f"inst_{len(forget_targets)}"
        run_name = f"retrain_{args.dataset}_{forget_str}"
        
        self.logger = UnlearningLogger(
            log_dir=os.path.join(args.log_dir, args.dataset, args.encoder),
            project_name='oracle_free_unlearning',
            run_name=run_name
        )
        self.checkpoint_manager = CheckpointManager(save_dir=args.save_dir)
        
    def _get_train_transforms(self):
        if self.args.img_size <= 32: # CIFAR-10/100
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else: # TinyImageNet, CelebA, LFW, FairFace, UTKFace
            return transforms.Compose([
                transforms.Resize((self.args.img_size, self.args.img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

    def _get_eval_transforms(self):
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

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(self.dr_loader):
            # Handle datasets returning (img, target, idx) or (img, target, demo_attrs, idx)
            imgs = batch[0].to(self.device)
            targets = batch[1].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(imgs)
            evidence = outputs[0]
            p_mean = outputs[2]
            
            # Compute Dirichlet parameters alpha = evidence + 1 (Eq. 2)
            alpha = evidence + 1.0
            
            # Compute loss (Eq. 8: L_EUP = L_ECE + lambda_kl * L_KL)
            loss = self.criterion(alpha, targets)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(p_mean.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            if batch_idx % self.args.log_interval == 0:
                print(f'Epoch [{epoch+1}/{self.args.epochs}] Step [{batch_idx}/{len(self.dr_loader)}] '
                      f'Loss: {loss.item():.4f} Acc: {100. * correct / total:.2f}%')
                      
        epoch_loss = running_loss / len(self.dr_loader)
        epoch_acc = 100. * correct / total
        self.logger.log_scalar('Retrain_Train/Loss', epoch_loss, epoch)
        self.logger.log_scalar('Retrain_Train/Accuracy', epoch_acc, epoch)
        return epoch_loss, epoch_acc

    def evaluate(self, epoch):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.eval_loader:
                imgs = batch[0].to(self.device)
                targets = batch[1].to(self.device)
                
                outputs = self.model(imgs)
                evidence = outputs[0]
                p_mean = outputs[2]
                alpha = evidence + 1.0
                
                loss = self.criterion(alpha, targets)
                
                running_loss += loss.item()
                _, predicted = torch.max(p_mean.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
        epoch_loss = running_loss / len(self.eval_loader)
        epoch_acc = 100. * correct / total
        
        self.logger.log_scalar('Retrain_Val/Loss', epoch_loss, epoch)
        self.logger.log_scalar('Retrain_Val/Accuracy', epoch_acc, epoch)
        return epoch_loss, epoch_acc

    def train(self):
        print(f"Starting exact retraining on D_r for {self.args.dataset} using {self.args.encoder}...")
        
        for epoch in range(self.args.epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.evaluate(epoch)
            
            print(f'Epoch [{epoch+1}/{self.args.epochs}] '
                  f'Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% | '
                  f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%')
                  
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.save_model(filename='best_retrain_baseline_model.pth')
                
            self.scheduler.step()
            
        self.save_model(filename='final_retrain_baseline_model.pth')
        print(f"Exact retraining complete. Best Validation Accuracy: {self.best_acc:.2f}%")
        self.logger.close()

    def save_model(self, filename='final_retrain_baseline_model.pth'):
        # Dummy EMA updater since exact retraining doesn't involve the forgetting manifold EMA
        dummy_ema = ForgettingManifoldEMA(feature_dim=self.model.encoder.feature_dim)
        
        metrics = {'best_val_acc': self.best_acc}
        self.checkpoint_manager.save_checkpoint(
            epoch=self.args.epochs,
            model=self.model,
            optimizer=self.optimizer,
            ema_updater=dummy_ema,
            metrics=metrics,
            filename=filename
        )
        print(f"Retrain baseline model saved to {os.path.join(self.args.save_dir, filename)}")


def parse_args():
    parser = argparse.ArgumentParser(description='Exact Retraining Baseline on D_r (Eq. 1)')
    
    # Dataset and Model Arguments
    parser.add_argument('--dataset', type=str, default='cifar10', 
                        choices=['cifar10', 'cifar100', 'tinyimagenet', 'celeba', 'lfw', 'fairface', 'utkface'])
    parser.add_argument('--encoder', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'vit_s'])
    parser.add_argument('--pretrained', action='store_true', default=True, help='Use ImageNet pre-trained weights for encoder')
    
    # Unlearning Targets (Must specify one)
    parser.add_argument('--forget_classes', type=str, default=None, help='Comma-separated list of class indices to forget (e.g., "0,1,2")')
    parser.add_argument('--forget_instances', type=str, default=None, help='Comma-separated list of instance indices to forget (e.g., "10,20,30")')
    
    # Training Arguments
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lr_step', type=int, default=50)
    parser.add_argument('--lr_gamma', type=float, default=0.1)
    parser.add_argument('--lambda_kl', type=float, default=0.1, help='Weight for KL divergence regularization (Eq. 7)')
    
    # System Arguments
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    
    args = parser.parse_args()
    
    if not args.forget_classes and not args.forget_instances:
        parser.error("Must specify either --forget_classes or --forget_instances")
        
    # Automatically set num_classes and img_size based on the dataset
    if args.dataset == 'cifar10':
        args.num_classes = 10
        args.img_size = 32
    elif args.dataset == 'cifar100':
        args.num_classes = 100
        args.img_size = 32
    elif args.dataset == 'tinyimagenet':
        args.num_classes = 200
        args.img_size = 64
    elif args.dataset in ['celeba', 'lfw']:
        args.num_classes = 1000 # CelebA identity (or adjust based on specific task setup)
        args.img_size = 224
    elif args.dataset == 'fairface':
        args.num_classes = 7 # Race classification
        args.img_size = 224
    elif args.dataset == 'utkface':
        args.num_classes = 5 # Race classification (filtered to 5 classes)
        args.img_size = 224
            
    return args


if __name__ == '__main__':
    args = parse_args()
    trainer = RetrainBaselineTrainer(args)
    trainer.train()