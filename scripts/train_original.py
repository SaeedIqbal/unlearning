"""
Initial empirical risk minimization on D = D_r U D_f (Eq. 1).
Designed to support the Oracle-Free Evidential Unlearning framework.

Key Methodology Mappings:
- Eq. 1: theta^* = argmin_theta L(theta; D)
- Trains the Oracle-Free Evidential Model from scratch on the full dataset D.
- The resulting model theta_orig serves as the starting point for the unlearning process.
- Uses the Expected Evidential Cross-Entropy (Eq. 6) and KL regularization (Eq. 7) 
  to ensure the model is properly calibrated and evidential from the very first epoch.
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
from models.full_model import get_full_model
from losses.evidential import EvidentialPartitioningLoss
from utils.logging import UnlearningLogger
from utils.checkpointing import CheckpointManager, ForgettingManifoldEMA


class OriginalModelTrainer:
    """
    Orchestrates the initial training of the Oracle-Free Evidential Model on the full dataset D.
    """
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_acc = 0.0
        
        # 1. Data Transformations
        self.train_transform = self._get_train_transforms()
        self.eval_transform = self._get_eval_transforms()
        
        # 2. Datasets and DataLoaders (Strictly using original datasets from /home/phd/datasets/)
        self.train_dataset = get_dataset(
            args.dataset, 
            root_dir='/home/phd/datasets', 
            split='train', 
            transform=self.train_transform
        )
        # Use 'val' for evaluation; falls back to 'test' if 'val' is not explicitly defined in the dataset wrapper
        eval_split = 'val' if args.dataset not in ['celeba', 'lfw'] else 'test'
        self.eval_dataset = get_dataset(
            args.dataset, 
            root_dir='/home/phd/datasets', 
            split=eval_split, 
            transform=self.eval_transform
        )
        
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers, 
            pin_memory=True
        )
        self.eval_loader = DataLoader(
            self.eval_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers, 
            pin_memory=True
        )
        
        # 3. Model Initialization
        self.model = get_full_model(
            encoder_name=args.encoder,
            num_classes=args.num_classes,
            pretrained=args.pretrained,
            img_size=args.img_size
        ).to(self.device)
        
        # 4. Loss and Optimizer
        # Using Evidential Partitioning Loss (Eq. 8) to train the original model with proper evidential calibration
        self.criterion = EvidentialPartitioningLoss(num_classes=args.num_classes, lambda_kl=args.lambda_kl)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
        
        # 5. Logging and Checkpointing
        self.logger = UnlearningLogger(
            log_dir=os.path.join(args.log_dir, args.dataset, args.encoder),
            project_name='oracle_free_unlearning',
            run_name=f'original_{args.dataset}_{args.encoder}'
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
        
        for batch_idx, batch in enumerate(self.train_loader):
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
                print(f'Epoch [{epoch+1}/{self.args.epochs}] Step [{batch_idx}/{len(self.train_loader)}] '
                      f'Loss: {loss.item():.4f} Acc: {100. * correct / total:.2f}%')
                      
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        self.logger.log_scalar('Original_Train/Loss', epoch_loss, epoch)
        self.logger.log_scalar('Original_Train/Accuracy', epoch_acc, epoch)
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
        
        self.logger.log_scalar('Original_Val/Loss', epoch_loss, epoch)
        self.logger.log_scalar('Original_Val/Accuracy', epoch_acc, epoch)
        return epoch_loss, epoch_acc

    def train(self):
        print(f"Starting original model training on {self.args.dataset} using {self.args.encoder}...")
        
        for epoch in range(self.args.epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.evaluate(epoch)
            
            print(f'Epoch [{epoch+1}/{self.args.epochs}] '
                  f'Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% | '
                  f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%')
                  
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.save_model(filename='best_original_model.pth')
                
            self.scheduler.step()
            
        self.save_model(filename='final_original_model.pth')
        print(f"Training complete. Best Validation Accuracy: {self.best_acc:.2f}%")
        self.logger.close()

    def save_model(self, filename='final_original_model.pth'):
        # We use a dummy EMA updater for the checkpoint manager since this is the original training phase
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
        print(f"Model saved to {os.path.join(self.args.save_dir, filename)}")


def parse_args():
    parser = argparse.ArgumentParser(description='Train Original Evidential Model (Eq. 1)')
    
    # Dataset and Model Arguments
    parser.add_argument('--dataset', type=str, default='cifar10', 
                        choices=['cifar10', 'cifar100', 'tinyimagenet', 'celeba', 'lfw', 'fairface', 'utkface'])
    parser.add_argument('--encoder', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'vit_s'])
    parser.add_argument('--pretrained', action='store_true', default=True, help='Use ImageNet pre-trained weights for encoder')
    
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
    trainer = OriginalModelTrainer(args)
    trainer.train()