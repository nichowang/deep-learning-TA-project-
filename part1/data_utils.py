"""
Data loading and preprocessing utilities for Oxford Pet Dataset classification.
"""

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np


def get_transforms(image_size=224, is_training=True):
    """
    Get image transforms for training and validation.
    
    STUDENT TASK (TODO): Implement/verify the transformation pipelines.
    - Training pipeline MUST include:
      1) Resize to (224, 224)
      2) RandomCrop to (image_size, image_size)
      3) RandomHorizontalFlip with p=0.5
      4) ToTensor
      5) Normalize with ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    - Validation pipeline MUST include:
      1) Resize to (image_size, image_size)
      2) ToTensor
      3) Normalize with the SAME ImageNet stats
    - Rationale: Pretrained backbones expect ImageNet-normalized inputs; train-time
      augmentations should not be applied at validation time.

    Args:
        image_size (int): Size to resize images to
        is_training (bool): Whether to apply training augmentations
    
    Returns:
        torchvision.transforms.Compose: Composed transforms
    """
    # TODO[STUDENT]: Ensure the train/val transforms match the specifications above.
    if is_training:
        # TODO[STUDENT]: Training augmentations + normalization
        transform = transforms.Compose([
            # 1) Resize larger for subsequent random crop
            transforms.Resize((224, 224)),  # Resize to larger size
            # 2) Random crop to target size
            transforms.RandomCrop(image_size),  # Random crop to target size
            # 3) Horizontal flip for augmentation
            transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
            # 4) Convert to tensor (scales pixels to [0,1])
            transforms.ToTensor(),
            # 5) Normalize using ImageNet stats expected by pretrained models
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
    else:
        # TODO[STUDENT]: Validation transforms (no randomness) + normalization
        transform = transforms.Compose([
            # 1) Deterministic resize to evaluation size
            transforms.Resize((image_size, image_size)),
            # 2) Convert to tensor
            transforms.ToTensor(),
            # 3) Normalize using the SAME ImageNet stats
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def load_datasets(data_dir='./data', image_size=224, batch_size=32, num_workers=4):
    """
    Load Oxford Pet Dataset with appropriate transforms.
    
    STUDENT TASK (TODO): Implement/verify dataset loading and DataLoader creation.
    Required steps:
    1) Create train and validation transforms via get_transforms(...)
    2) Instantiate torchvision.datasets.OxfordIIITPet for 'trainval' and 'test'
       - Use the correct transform for each split
       - Set download=True so first run fetches the data
    3) Wrap each dataset in a DataLoader
       - Train: shuffle=True
       - Val: shuffle=False
       - Use pin_memory=True when training on GPU for faster host->device transfer
    4) Compute num_classes via len(train_dataset.classes)

    Args:
        data_dir (str): Directory to store/load dataset
        image_size (int): Size to resize images to
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of worker processes for data loading
    
    Returns:
        tuple: (train_loader, val_loader, num_classes)
    """
    # TODO[STUDENT]: 1) Define transforms
    train_transform = get_transforms(image_size, is_training=True)
    val_transform = get_transforms(image_size, is_training=False)
    
    # TODO[STUDENT]: 2) Load datasets for train/val with correct splits and transforms
    train_dataset = torchvision.datasets.OxfordIIITPet(
        root=data_dir,
        split='trainval',
        download=True,
        transform=train_transform
    )
    
    val_dataset = torchvision.datasets.OxfordIIITPet(
        root=data_dir,
        split='test',
        download=True,
        transform=val_transform
    )
    
    # TODO[STUDENT]: 3) Create DataLoaders with correct shuffle/pin_memory settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # TODO[STUDENT]: 4) Determine number of classes
    num_classes = len(train_dataset.classes)
    
    print(f"Dataset loaded successfully!")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {train_dataset.classes}")
    
    return train_loader, val_loader, num_classes


def get_class_names(data_dir='./data'):
    """
    Get class names from the dataset.
    
    Args:
        data_dir (str): Directory where dataset is stored
    
    Returns:
        list: List of class names
    """
    dataset = torchvision.datasets.OxfordIIITPet(
        root=data_dir,
        split='trainval',
        download=True
    )
    return dataset.classes
