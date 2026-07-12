from src.config import ExperimentConfig

import torch
import torch.nn as nn
import timm
from typing import Dict


class ModelFactory:
    """Create and manage vision models."""
    
    @staticmethod
    def create_model(config) -> nn.Module:
        """
        Create model from config.
        
        Args:
            config: ExperimentConfig object
            
        Returns:
            PyTorch model
        """
        model_name = config.model.name
        num_classes = config.model.num_classes
        pretrained = config.model.pretrained
        
        # Create from timm
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )
        
        return model
    
    @staticmethod
    def get_available_models() -> list:
        """List available vision models."""
        return [
            "swin_tiny_patch4_window7_224",
            "swin_small_patch4_window7_224",
            "swin_base_patch4_window7_224",
            "resnet50",
            "densenet121",
            "efficientnet_b0",
        ]


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_backbone(model: nn.Module, freeze_pct: float = 0.8):
    """
    Freeze percentage of model parameters.
    
    Args:
        model: PyTorch model
        freeze_pct: Percentage of layers to freeze (0.0 to 1.0)
    """
    params = list(model.parameters())
    num_to_freeze = int(len(params) * freeze_pct)
    
    for param in params[:num_to_freeze]:
        param.requires_grad = False
    
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    total = len(params)
    print(f"Froze {num_to_freeze}/{total} parameter groups. Trainable: {trainable}")


def create_optimizer(model: nn.Module, config) -> torch.optim.Optimizer:
    """Create optimizer from config."""
    optimizer_name = config.training.optimizer.lower()
    
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
    elif optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
    elif optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            momentum=0.9,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_scheduler(optimizer: torch.optim.Optimizer, config) -> torch.optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler from config."""
    scheduler_name = config.training.scheduler.lower()
    total_epochs = config.training.epochs
    
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
        )
    elif scheduler_name == "linear":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=total_epochs,
        )
    elif scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=total_epochs // 3,
            gamma=0.1,
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss mathematically forces the model to focus on hard, misclassified examples 
    (like Gliomas) while penalizing overconfidence, resulting in pristine calibration.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def create_loss(config, class_weights: torch.Tensor = None) -> nn.Module:
    """Create loss function using Focal Loss for maximum calibration."""
    return FocalLoss(alpha=class_weights, gamma=2.0)

"""Training loop and helpers."""