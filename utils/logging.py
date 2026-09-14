"""
TensorBoard/WandB integration for tracking vacuity, CKA, and MIA over training steps.
Designed to support the Oracle-Free Evidential Unlearning framework.

Key Methodology Mappings:
- Tracks epistemic vacuity u(x) (Eq. 3) and predictive variance Var[p] (Eq. 5) to monitor EUP (Eq. 9).
- Tracks Linear CKA to monitor deep representation extraction (Threat Model 2).
- Tracks MIA Advantage/ASR to monitor membership inference vulnerability (Threat Model 1).
- Tracks all loss components: L_EUP (Eq. 8), L_forget (Eq. 25), L_FSBR (Eq. 15), L_trust_region (Eq. 24').
"""

import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional, Union

class UnlearningLogger:
    """
    Unified logging utility for the Oracle-Free Evidential Unlearning framework.
    Integrates TensorBoard and Weights & Biases (WandB) to track training dynamics,
    evidential metrics, and privacy/robustness audits.
    """
    
    def __init__(self, log_dir: str, project_name: str, run_name: str, 
                 use_wandb: bool = False, wandb_config: Optional[Dict[str, Any]] = None):
        """
        Initializes the logger, setting up TensorBoard and optionally WandB.
        
        Args:
            log_dir (str): Directory to save TensorBoard logs.
            project_name (str): WandB project name.
            run_name (str): WandB run name.
            use_wandb (bool): Whether to enable Weights & Biases logging.
            wandb_config (Optional[Dict]): Configuration dictionary to log to WandB.
        """
        self.log_dir = log_dir
        self.use_wandb = use_wandb
        
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize TensorBoard
        self.tb_writer = SummaryWriter(log_dir=log_dir)
        
        # Initialize WandB if requested
        self.wandb = None
        if self.use_wandb:
            try:
                import wandb
                self.wandb = wandb
                self.wandb.init(
                    project=project_name,
                    name=run_name,
                    config=wandb_config if wandb_config else {},
                    dir=log_dir
                )
            except ImportError:
                print("Warning: wandb is not installed. Falling back to TensorBoard only.")
                self.use_wandb = False
                
        self.step = 0

    def log_scalar(self, tag: str, value: Union[float, int], step: Optional[int] = None):
        """
        Logs a single scalar value to both TensorBoard and WandB.
        
        Args:
            tag (str): Metric name (e.g., 'Loss/L_EUP', 'Privacy/CKA').
            value (Union[float, int]): The metric value.
            step (Optional[int]): Global step. If None, uses internal step counter.
        """
        current_step = step if step is not None else self.step
        
        # TensorBoard
        self.tb_writer.add_scalar(tag, value, current_step)
        
        # WandB
        if self.use_wandb and self.wandb is not None:
            self.wandb.log({tag: value}, step=current_step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: Optional[int] = None):
        """
        Logs multiple scalar values under a main tag to TensorBoard, and individually to WandB.
        
        Args:
            main_tag (str): Main category (e.g., 'Loss', 'Evidential', 'Privacy').
            tag_scalar_dict (Dict[str, float]): Dictionary of {metric_name: value}.
            step (Optional[int]): Global step.
        """
        current_step = step if step is not None else self.step
        
        # TensorBoard
        self.tb_writer.add_scalars(main_tag, tag_scalar_dict, current_step)
        
        # WandB
        if self.use_wandb and self.wandb is not None:
            log_dict = {f"{main_tag}/{k}": v for k, v in tag_scalar_dict.items()}
            self.wandb.log(log_dict, step=current_step)

    def log_evidential_metrics(self, vacuity_mean: float, variance_mean: float, 
                               evidence_mean: float, step: Optional[int] = None):
        """
        Logs Evidential Deep Learning (EDL) metrics to monitor EUP (Eq. 9) and PFC (Eqs. 27-28).
        
        Args:
            vacuity_mean (float): Mean epistemic vacuity u(x) (Eq. 3).
            variance_mean (float): Mean predictive variance Var[p] (Eq. 5).
            evidence_mean (float): Mean evidence magnitude.
            step (Optional[int]): Global step.
        """
        metrics = {
            'vacuity_mean': vacuity_mean,
            'variance_mean': variance_mean,
            'evidence_mean': evidence_mean
        }
        self.log_scalars('Evidential', metrics, step)

    def log_privacy_metrics(self, cka_score: float, mia_asr: float, mia_advantage: float, 
                            inversion_success_rate: float, step: Optional[int] = None):
        """
        Logs privacy and structural audit metrics for the Threat Model (Section 3.3).
        
        Args:
            cka_score (float): Linear Centered Kernel Alignment (Deep Representation Extraction).
            mia_asr (float): Membership Inference Attack Success Rate (%).
            mia_advantage (float): MIA Advantage.
            inversion_success_rate (float): Feature-Space Inversion Success Rate (%).
            step (Optional[int]): Global step.
        """
        metrics = {
            'CKA': cka_score,
            'MIA_ASR': mia_asr,
            'MIA_Advantage': mia_advantage,
            'Inversion_Success_Rate': inversion_success_rate
        }
        self.log_scalars('Privacy_Audit', metrics, step)

    def log_training_losses(self, loss_eup: float, loss_forget: float, loss_fsbr: float, 
                            loss_trust_region: float, total_loss: float, step: Optional[int] = None):
        """
        Logs the components of the total training objective (Eq. 24).
        
        Args:
            loss_eup (float): Evidential Partitioning Loss (Eq. 8).
            loss_forget (float): Evidential Forgetting Loss (Eq. 25).
            loss_fsbr (float): Feature-Space Boundary Repulsion Loss (Eq. 15).
            loss_trust_region (float): L2 Trust-Region Penalty (Eq. 24').
            total_loss (float): Total combined loss.
            step (Optional[int]): Global step.
        """
        metrics = {
            'L_EUP': loss_eup,
            'L_forget': loss_forget,
            'L_FSBR': loss_fsbr,
            'L_trust_region': loss_trust_region,
            'Total': total_loss
        }
        self.log_scalars('Loss', metrics, step)

    def log_utility_metrics(self, acc_retrain: float, acc_unl: float, avg_gap: float, 
                            step: Optional[int] = None):
        """
        Logs standard utility metrics on the retained set D_r.
        
        Args:
            acc_retrain (float): Accuracy of the exact retrained baseline (%).
            acc_unl (float): Accuracy of the unlearned model (%).
            avg_gap (float): Average Performance Gap (percentage points).
            step (Optional[int]): Global step.
        """
        metrics = {
            'Acc_Retrain': acc_retrain,
            'Acc_Unl': acc_unl,
            'Avg_Gap': avg_gap
        }
        self.log_scalars('Utility', metrics, step)

    def log_hyperparameters(self, hparams: Dict[str, Any], metric_dict: Dict[str, float]):
        """
        Logs hyperparameters and final metrics for TensorBoard's HPARAMS plugin.
        
        Args:
            hparams (Dict[str, Any]): Dictionary of hyperparameters.
            metric_dict (Dict[str, float]): Dictionary of final metrics (e.g., final CKA, MIA, Avg Gap).
        """
        self.tb_writer.add_hparams(hparams, metric_dict)
        
        if self.use_wandb and self.wandb is not None:
            self.wandb.log(metric_dict)

    def increment_step(self):
        """Increments the internal step counter."""
        self.step += 1

    def close(self):
        """Closes the TensorBoard writer and finishes the WandB run."""
        self.tb_writer.close()
        if self.use_wandb and self.wandb is not None:
            self.wandb.finish()