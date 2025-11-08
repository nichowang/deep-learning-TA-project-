"""
Training utilities for model training and evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def train_epoch(model, train_loader, optimizer, criterion, device, scheduler=None):
    """
    Train the model for one epoch.
    
    STUDENT TASK (TODO): Implement the standard supervised learning training loop.
    Steps per batch:
      1) Move inputs/targets to the correct device
      2) Zero optimizer gradients
      3) Forward pass to get logits
      4) Compute loss with criterion
      5) Backward pass to compute gradients
      6) Optimizer step to update parameters
    After the epoch:
      - Step the scheduler if provided
      - Return average loss and accuracy (in %)

    Args:
        model (torch.nn.Module): The model to train
        train_loader (DataLoader): Training data loader
        optimizer (torch.optim.Optimizer): Optimizer
        criterion (torch.nn.Module): Loss function
        device (torch.device): Device to run on
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler
    
    Returns:
        tuple: (average_loss, accuracy)
    """
    # TODO[STUDENT]: Put the model in training mode
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # TODO[STUDENT]: 1) Move batch to device
        data, target = data.to(device), target.to(device)
        
        # TODO[STUDENT]: 2) Zero gradients
        optimizer.zero_grad()
        
        # TODO[STUDENT]: 3) Forward pass
        output = model(data)
        
        # TODO[STUDENT]: 4) Compute loss
        loss = criterion(output, target)
        
        # TODO[STUDENT]: 5) Backward pass
        loss.backward()
        
        # TODO[STUDENT]: 6) Update parameters
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        if batch_idx % 50 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    # TODO[STUDENT]: Step the scheduler once per epoch (if provided)
    if scheduler is not None:
        scheduler.step()
    
    # TODO[STUDENT]: Compute average loss and accuracy (%)
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def validate_epoch(model, val_loader, criterion, device):
    """
    Validate the model for one epoch.
    
    STUDENT TASK (TODO): Implement the evaluation loop.
    Requirements:
      - Use model.eval() and torch.no_grad()
      - No optimizer steps or backward passes
      - Return average loss, accuracy (%), predictions, and targets

    Args:
        model (torch.nn.Module): The model to validate
        val_loader (DataLoader): Validation data loader
        criterion (torch.nn.Module): Loss function
        device (torch.device): Device to run on
    
    Returns:
        tuple: (average_loss, accuracy, all_predictions, all_targets)
    """
    # TODO[STUDENT]: Put the model in evaluation mode
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    # TODO[STUDENT]: Disable gradient computation during evaluation
    with torch.no_grad():
        for data, target in val_loader:
            # Move to device
            data, target = data.to(device), target.to(device)
            # Forward pass only
            output = model(data)
            # Compute loss
            loss = criterion(output, target)
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Store for later analysis
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # TODO[STUDENT]: Compute average loss and accuracy (%)
    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy, all_predictions, all_targets


def train_model(model, train_loader, val_loader, num_epochs, learning_rate, 
                device, model_name, training_strategy, use_scheduler=True):
    """
    Train a model with the specified strategy.
    
    Args:
        model (torch.nn.Module): The model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        num_epochs (int): Number of epochs to train
        learning_rate (float): Learning rate
        device (torch.device): Device to run on
        model_name (str): Name of the model
        training_strategy (str): 'feature_extraction' or 'full_finetuning'
        use_scheduler (bool): Whether to use learning rate scheduler
    
    Returns:
        dict: Training history and best model info
    """
    from model_utils import setup_optimizer, setup_scheduler
    
    # Setup optimizer and scheduler
    optimizer = setup_optimizer(model, learning_rate)
    scheduler = setup_scheduler(optimizer, num_epochs) if use_scheduler else None
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    best_val_acc = 0.0
    best_model_state = None
    
    print(f"\nStarting training: {model_name} - {training_strategy}")
    print(f"Learning rate: {learning_rate}")
    print(f"Number of epochs: {num_epochs}")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, scheduler
        )
        
        # Validate
        val_loss, val_acc, _, _ = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Best Val Acc: {best_val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    return {
        'history': history,
        'best_val_acc': best_val_acc,
        'best_model_state': best_model_state
    }


def evaluate_model(model, val_loader, device, class_names=None):
    """
    Comprehensive evaluation of the model.
    
    Args:
        model (torch.nn.Module): The model to evaluate
        val_loader (DataLoader): Validation data loader
        device (torch.device): Device to run on
        class_names (list, optional): List of class names
    
    Returns:
        dict: Evaluation results
    """
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probabilities = F.softmax(output, dim=1)
            
            _, predicted = torch.max(output, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    conf_matrix = confusion_matrix(all_targets, all_predictions)
    
    results = {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'predictions': all_predictions,
        'targets': all_targets,
        'probabilities': all_probabilities
    }
    
    if class_names is not None:
        results['classification_report'] = classification_report(
            all_targets, all_predictions, target_names=class_names, output_dict=True
        )
    
    return results


def plot_confusion_matrix(conf_matrix, class_names, title, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        conf_matrix (np.ndarray): Confusion matrix
        class_names (list): List of class names
        title (str): Title for the plot
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_training_history(history, title, save_path=None):
    """
    Plot training history.
    
    Args:
        history (dict): Training history
        title (str): Title for the plot
        save_path (str, optional): Path to save the plot
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_title('Accuracy Curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    # Learning rate
    ax3.plot(history['learning_rates'])
    ax3.set_title('Learning Rate Schedule')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.grid(True)
    
    # Combined loss comparison
    ax4.plot(history['val_loss'], label='Validation Loss')
    ax4.set_title('Validation Loss')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
