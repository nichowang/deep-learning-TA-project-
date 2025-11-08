"""
Model setup and fine-tuning utilities for ResNet-50 and Swin-T.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import swin_transformer
import warnings


def setup_resnet50(num_classes, pretrained=True):
    """
    Setup ResNet-50 model for fine-tuning.
    
    STUDENT TASK (TODO): Replace the ImageNet classification head with a
    task-specific head for Oxford Pets (num_classes outputs).
    Steps:
    1) Load torchvision.models.resnet50 with pretrained weights if requested
    2) Read feature dimension of the current final layer via model.fc.in_features
    3) Replace model.fc with nn.Linear(num_features, num_classes)

    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
    
    Returns:
        torch.nn.Module: ResNet-50 model
    """
    # TODO[STUDENT]: 1) Load pretrained ResNet-50
    model = models.resnet50(pretrained=pretrained)
    
    # TODO[STUDENT]: 2) Inspect current final layer input features
    num_features = model.fc.in_features

    # TODO[STUDENT]: 3) Replace final classification layer with correct output dim
    model.fc = nn.Linear(num_features, num_classes)
    
    return model


def setup_swin_t(num_classes, pretrained=True):
    """
    Setup Swin Transformer (Swin-T) model for fine-tuning.
    
    STUDENT TASK (TODO): Replace the ImageNet classification head with a
    task-specific head for Oxford Pets (num_classes outputs).
    Steps:
    1) Load swin_transformer.swin_t with ImageNet weights when pretrained=True
       - weights='IMAGENET1K_V1' (or None when not pretrained)
    2) Read feature dimension of the current final layer via model.head.in_features
    3) Replace model.head with nn.Linear(num_features, num_classes)

    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
    
    Returns:
        torch.nn.Module: Swin-T model
    """
    # TODO[STUDENT]: 1) Load pretrained Swin-T
    model = swin_transformer.swin_t(weights='IMAGENET1K_V1' if pretrained else None)
    
    # TODO[STUDENT]: 2) Inspect current final layer input features
    num_features = model.head.in_features

    # TODO[STUDENT]: 3) Replace final classification layer with correct output dim
    model.head = nn.Linear(num_features, num_classes)
    
    return model


def freeze_backbone(model, model_name):
    """
    Freeze all parameters except the final classification layer.
    
    STUDENT TASK (TODO): Implement feature-extraction mode by freezing all
    parameters EXCEPT the classification head, based on model_name.
    Steps:
    1) Set requires_grad=False for all parameters
    2) If model_name == 'resnet50': set requires_grad=True for model.fc params
       If model_name == 'swin_t': set requires_grad=True for model.head params
    3) Raise ValueError for unknown model names

    Args:
        model (torch.nn.Module): The model to freeze
        model_name (str): Name of the model ('resnet50' or 'swin_t')
    
    Returns:
        torch.nn.Module: Model with frozen backbone
    """
    # TODO[STUDENT]: 1) Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # TODO[STUDENT]: 2) Unfreeze only the final classification layer
    if model_name == 'resnet50':
        for param in model.fc.parameters():
            param.requires_grad = True
    elif model_name == 'swin_t':
        for param in model.head.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model


def get_trainable_parameters(model):
    """
    Get the number of trainable parameters in the model.
    
    Args:
        model (torch.nn.Module): The model
    
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_total_parameters(model):
    """
    Get the total number of parameters in the model.
    
    Args:
        model (torch.nn.Module): The model
    
    Returns:
        int: Total number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def print_model_info(model, model_name):
    """
    Print model information including parameter counts.
    
    Args:
        model (torch.nn.Module): The model
        model_name (str): Name of the model
    """
    total_params = get_total_parameters(model)
    trainable_params = get_trainable_parameters(model)
    
    print(f"\n{model_name.upper()} Model Information:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")


def setup_optimizer(model, learning_rate, weight_decay=1e-4):
    """
    Setup optimizer for the model.
    
    STUDENT TASK (TODO): Ensure only trainable parameters are passed to the optimizer.
    - Use filter(lambda p: p.requires_grad, model.parameters()) so that in
      feature-extraction mode, only the head's parameters are updated.

    Args:
        model (torch.nn.Module): The model
        learning_rate (float): Learning rate
        weight_decay (float): Weight decay for regularization
    
    Returns:
        torch.optim.Optimizer: Optimizer
    """
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )


def setup_scheduler(optimizer, num_epochs, warmup_epochs=5):
    """
    Setup learning rate scheduler.
    
    STUDENT TASK (TODO): Configure a scheduler suitable for fine-tuning.
    - Here we use CosineAnnealingLR with a small eta_min.
    - Warmup can be emulated externally; keep T_max consistent with total epochs.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer
        num_epochs (int): Total number of epochs
        warmup_epochs (int): Number of warmup epochs
    
    Returns:
        torch.optim.lr_scheduler._LRScheduler: Learning rate scheduler
    """
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs - warmup_epochs,
        eta_min=1e-6
    )
