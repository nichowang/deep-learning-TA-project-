# Part 1: Detailed Implementation Guide

## 🎯 Overview

This guide provides step-by-step instructions for implementing the Oxford Pet Dataset classification system. You'll work with two pre-trained models (ResNet-50 and Swin-T) and two training strategies (feature extraction and full fine-tuning).

## 📚 Learning Objectives

By completing this implementation, you will understand:
- **Transfer Learning**: How to adapt pre-trained models for new tasks
- **Data Preprocessing**: Image augmentation and normalization techniques
- **Model Architecture**: Differences between CNN and Transformer architectures
- **Training Strategies**: Feature extraction vs. full fine-tuning
- **Evaluation**: Comprehensive model analysis and error patterns

## 🏗️ Implementation Structure

The project is organized into several modules, each with specific functions you need to implement:

1. **`data_utils.py`**: Data loading and preprocessing
2. **`model_utils.py`**: Model setup and configuration
3. **`training_utils.py`**: Training and evaluation loops
4. **`experiment_tracker.py`**: Experiment logging and visualization

---

## 📁 Module 1: Data Utilities (`data_utils.py`)

### Function 1: `get_transforms(image_size=224, is_training=True)`

**Purpose**: Create image transformation pipelines for training and validation.

**What you need to implement**:
```python
def get_transforms(image_size=224, is_training=True):
    """
    Get image transforms for training and validation.
    
    This function creates different transformation pipelines:
    - Training: Includes data augmentation (random crops, flips)
    - Validation: Only basic preprocessing (resize, normalize)
    
    Key concepts:
    - Data Augmentation: Random transformations to increase dataset diversity
    - Normalization: Standardize pixel values using ImageNet statistics
    - ImageNet Stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    
    Args:
        image_size (int): Target image size (224x224 is standard)
        is_training (bool): Whether to apply training augmentations
    
    Returns:
        torchvision.transforms.Compose: Composed transforms
    """
    if is_training:
        # TODO: Implement training transforms
        # Hint: You need these transforms in order:
        # 1. Resize to larger size (image_size + 32) for random cropping
        # 2. RandomCrop to target size (creates variety)
        # 3. RandomHorizontalFlip with 50% probability
        # 4. Convert to tensor (0-1 range)
        # 5. Normalize with ImageNet statistics
        
        transform = transforms.Compose([
            # Step 1: Resize to larger size for random cropping
            transforms.Resize((image_size + 32, image_size + 32)),
            
            # Step 2: Random crop to target size
            # This creates data augmentation by showing different parts of the image
            transforms.RandomCrop(image_size),
            
            # Step 3: Random horizontal flip (50% probability)
            # Helps model learn that objects can face either direction
            transforms.RandomHorizontalFlip(p=0.5),
            
            # Step 4: Convert PIL image to tensor (0-1 range)
            transforms.ToTensor(),
            
            # Step 5: Normalize using ImageNet statistics
            # This standardizes the input distribution
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean for RGB
                std=[0.229, 0.224, 0.225]   # ImageNet std for RGB
            )
        ])
    else:
        # TODO: Implement validation transforms
        # Hint: For validation, we don't want randomness
        # Just resize, convert to tensor, and normalize
        
        transform = transforms.Compose([
            # Step 1: Resize to exact target size (no random cropping)
            transforms.Resize((image_size, image_size)),
            
            # Step 2: Convert to tensor
            transforms.ToTensor(),
            
            # Step 3: Normalize with same ImageNet statistics
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    return transform
```

**Key Learning Points**:
- **Why resize to larger size first?** Random cropping needs extra pixels to work with
- **Why ImageNet normalization?** Pre-trained models expect this specific distribution
- **Why no augmentation for validation?** We want consistent evaluation

### Function 2: `load_datasets(data_dir='./data', image_size=224, batch_size=32, num_workers=4)`

**Purpose**: Load the Oxford Pet Dataset with proper train/validation splits.

**What you need to implement**:
```python
def load_datasets(data_dir='./data', image_size=224, batch_size=32, num_workers=4):
    """
    Load Oxford Pet Dataset with appropriate transforms.
    
    This function:
    1. Creates train and validation transforms
    2. Loads the Oxford Pet Dataset with proper splits
    3. Creates DataLoaders for efficient batch processing
    
    Dataset Information:
    - 37 different pet breeds
    - ~200 images per breed
    - Total: ~7,400 images
    - Split: trainval (training) and test (validation)
    
    Args:
        data_dir (str): Directory to store/load dataset
        image_size (int): Size to resize images to
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of worker processes for data loading
    
    Returns:
        tuple: (train_loader, val_loader, num_classes)
    """
    # TODO: Step 1 - Create transforms
    # Hint: Use the get_transforms function you implemented above
    train_transform = get_transforms(image_size, is_training=True)
    val_transform = get_transforms(image_size, is_training=False)
    
    # TODO: Step 2 - Load training dataset
    # Hint: Use torchvision.datasets.OxfordIIITPet
    # Parameters: root=data_dir, split='trainval', download=True, transform=train_transform
    train_dataset = torchvision.datasets.OxfordIIITPet(
        root=data_dir,
        split='trainval',  # Training and validation combined
        download=True,     # Download if not present
        transform=train_transform
    )
    
    # TODO: Step 3 - Load validation dataset
    # Hint: Same as training but with split='test' and val_transform
    val_dataset = torchvision.datasets.OxfordIIITPet(
        root=data_dir,
        split='test',      # Test split for validation
        download=True,
        transform=val_transform
    )
    
    # TODO: Step 4 - Create data loaders
    # Hint: Use torch.utils.data.DataLoader
    # Training: shuffle=True, validation: shuffle=False
    
    # Training DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,      # Shuffle for training
        num_workers=num_workers,
        pin_memory=True    # Faster GPU transfer
    )
    
    # Validation DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,     # No shuffle for validation
        num_workers=num_workers,
        pin_memory=True
    )
    
    # TODO: Step 5 - Get number of classes
    # Hint: Use len(train_dataset.classes)
    num_classes = len(train_dataset.classes)
    
    # Print dataset information
    print(f"Dataset loaded successfully!")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {train_dataset.classes}")
    
    return train_loader, val_loader, num_classes
```

**Key Learning Points**:
- **Why different splits?** trainval for training, test for validation
- **Why shuffle training but not validation?** Training benefits from randomness
- **Why pin_memory?** Faster data transfer to GPU

---

## 🏗️ Module 2: Model Utilities (`model_utils.py`)

### Function 1: `setup_resnet50(num_classes, pretrained=True)`

**Purpose**: Configure ResNet-50 for pet classification.

**What you need to implement**:
```python
def setup_resnet50(num_classes, pretrained=True):
    """
    Setup ResNet-50 model for fine-tuning.
    
    ResNet-50 Architecture:
    - 50-layer deep residual network
    - Pre-trained on ImageNet (1.2M images, 1000 classes)
    - Final layer: 1000 classes -> needs to be changed to 37 pet classes
    
    Key Steps:
    1. Load pre-trained ResNet-50
    2. Replace final classification layer
    3. Keep all other layers frozen initially
    
    Args:
        num_classes (int): Number of output classes (37 for pets)
        pretrained (bool): Whether to use pretrained weights
    
    Returns:
        torch.nn.Module: ResNet-50 model ready for fine-tuning
    """
    # TODO: Step 1 - Load pretrained ResNet-50
    # Hint: Use torchvision.models.resnet50(pretrained=pretrained)
    model = models.resnet50(pretrained=pretrained)
    
    # TODO: Step 2 - Get number of input features for final layer
    # Hint: Use model.fc.in_features
    # The final layer (fc) currently outputs 1000 classes
    # We need to know how many input features it has
    num_features = model.fc.in_features
    
    # TODO: Step 3 - Replace final classification layer
    # Hint: Create new nn.Linear layer with num_features input and num_classes output
    # This changes from 1000 ImageNet classes to 37 pet classes
    model.fc = nn.Linear(num_features, num_classes)
    
    return model
```

**Key Learning Points**:
- **Why replace final layer?** Pre-trained model was trained for 1000 ImageNet classes
- **Why keep other layers?** Lower layers learn general features (edges, textures)
- **What is fc?** "Fully Connected" - the final classification layer

### Function 2: `setup_swin_t(num_classes, pretrained=True)`

**Purpose**: Configure Swin Transformer for pet classification.

**What you need to implement**:
```python
def setup_swin_t(num_classes, pretrained=True):
    """
    Setup Swin Transformer (Swin-T) model for fine-tuning.
    
    Swin-T Architecture:
    - Transformer-based model (not CNN like ResNet)
    - Uses "shifted windows" for efficient attention computation
    - Pre-trained on ImageNet
    - Final layer: 1000 classes -> needs to be changed to 37 pet classes
    
    Key Differences from ResNet:
    - Uses attention mechanisms instead of convolutions
    - Better at capturing long-range dependencies
    - More parameters but often better performance
    
    Args:
        num_classes (int): Number of output classes (37 for pets)
        pretrained (bool): Whether to use pretrained weights
    
    Returns:
        torch.nn.Module: Swin-T model ready for fine-tuning
    """
    # TODO: Step 1 - Load pretrained Swin-T
    # Hint: Use torchvision.models.swin_transformer.swin_t
    # For pretrained weights, use weights='IMAGENET1K_V1'
    # For no pretraining, use weights=None
    if pretrained:
        model = swin_transformer.swin_t(weights='IMAGENET1K_V1')
    else:
        model = swin_transformer.swin_t(weights=None)
    
    # TODO: Step 2 - Get number of input features for final layer
    # Hint: Use model.head.in_features
    # Swin-T uses 'head' instead of 'fc' for the final layer
    num_features = model.head.in_features
    
    # TODO: Step 3 - Replace final classification layer
    # Hint: Create new nn.Linear layer with num_features input and num_classes output
    model.head = nn.Linear(num_features, num_classes)
    
    return model
```

**Key Learning Points**:
- **Why different layer name?** Swin-T uses 'head' instead of 'fc'
- **What are transformers?** Attention-based models that process sequences
- **Why Swin-T?** More efficient than full transformers, better than CNNs

### Function 3: `freeze_backbone(model, model_name)`

**Purpose**: Freeze all layers except the final classification layer for feature extraction.

**What you need to implement**:
```python
def freeze_backbone(model, model_name):
    """
    Freeze all parameters except the final classification layer.
    
    This implements "Feature Extraction" strategy:
    - Keep pre-trained features frozen
    - Only train the final classification layer
    - Faster training, less prone to overfitting
    - Good for small datasets
    
    Freezing Process:
    1. Set requires_grad=False for all parameters
    2. Set requires_grad=True only for final layer
    
    Args:
        model (torch.nn.Module): The model to freeze
        model_name (str): Name of the model ('resnet50' or 'swin_t')
    
    Returns:
        torch.nn.Module: Model with frozen backbone
    """
    # TODO: Step 1 - Freeze all parameters first
    # Hint: Loop through model.parameters() and set requires_grad=False
    for param in model.parameters():
        param.requires_grad = False
    
    # TODO: Step 2 - Unfreeze only the final classification layer
    # Hint: Check model_name and unfreeze the appropriate layer
    # ResNet-50: model.fc
    # Swin-T: model.head
    
    if model_name == 'resnet50':
        # Unfreeze the final layer (fc) for ResNet-50
        for param in model.fc.parameters():
            param.requires_grad = True
    elif model_name == 'swin_t':
        # Unfreeze the final layer (head) for Swin-T
        for param in model.head.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model
```

**Key Learning Points**:
- **Why freeze backbone?** Pre-trained features are already good
- **What is requires_grad?** Controls whether parameters are updated during training
- **Why different layer names?** Different architectures use different naming

---

## 🏋️ Module 3: Training Utilities (`training_utils.py`)

### Function 1: `train_epoch(model, train_loader, optimizer, criterion, device, scheduler=None)`

**Purpose**: Train the model for one complete epoch.

**What you need to implement**:
```python
def train_epoch(model, train_loader, optimizer, criterion, device, scheduler=None):
    """
    Train the model for one epoch.
    
    Training Loop Steps:
    1. Set model to training mode
    2. For each batch:
       a. Move data to device (GPU/CPU)
       b. Zero gradients
       c. Forward pass (compute predictions)
       d. Compute loss
       e. Backward pass (compute gradients)
       f. Update parameters
    3. Calculate average loss and accuracy
    4. Update learning rate scheduler
    
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
    # TODO: Step 1 - Set model to training mode
    # Hint: Use model.train()
    # This enables dropout, batch norm updates, etc.
    model.train()
    
    # Initialize tracking variables
    total_loss = 0.0
    correct = 0
    total = 0
    
    # TODO: Step 2 - Training loop over all batches
    for batch_idx, (data, target) in enumerate(train_loader):
        # TODO: Step 2a - Move data to device
        # Hint: Use .to(device) for both data and target
        data, target = data.to(device), target.to(device)
        
        # TODO: Step 2b - Zero gradients
        # Hint: Use optimizer.zero_grad()
        # This clears gradients from previous batch
        optimizer.zero_grad()
        
        # TODO: Step 2c - Forward pass
        # Hint: Use model(data) to get predictions
        output = model(data)
        
        # TODO: Step 2d - Compute loss
        # Hint: Use criterion(output, target)
        # CrossEntropyLoss compares predictions with true labels
        loss = criterion(output, target)
        
        # TODO: Step 2e - Backward pass
        # Hint: Use loss.backward()
        # This computes gradients for all parameters
        loss.backward()
        
        # TODO: Step 2f - Update parameters
        # Hint: Use optimizer.step()
        # This updates model parameters using computed gradients
        optimizer.step()
        
        # Track statistics
        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # Print progress every 50 batches
        if batch_idx % 50 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    # TODO: Step 3 - Update learning rate scheduler
    # Hint: Use scheduler.step() if scheduler is not None
    if scheduler is not None:
        scheduler.step()
    
    # TODO: Step 4 - Calculate average metrics
    # Hint: 
    # - Average loss = total_loss / number_of_batches
    # - Accuracy = correct_predictions / total_predictions * 100
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy
```

**Key Learning Points**:
- **Why model.train()?** Enables training-specific behaviors (dropout, batch norm)
- **Why zero_grad()?** PyTorch accumulates gradients, must clear them
- **Why .to(device)?** Data must be on same device as model (GPU/CPU)

### Function 2: `validate_epoch(model, val_loader, criterion, device)`

**Purpose**: Validate the model for one epoch.

**What you need to implement**:
```python
def validate_epoch(model, val_loader, criterion, device):
    """
    Validate the model for one epoch.
    
    Validation Loop Steps:
    1. Set model to evaluation mode
    2. Disable gradient computation (torch.no_grad())
    3. For each batch:
       a. Move data to device
       b. Forward pass (no backward pass needed)
       c. Compute loss and predictions
       d. Track statistics
    4. Calculate average metrics
    
    Key Differences from Training:
    - No gradient computation (faster)
    - No parameter updates
    - Model in evaluation mode
    
    Args:
        model (torch.nn.Module): The model to validate
        val_loader (DataLoader): Validation data loader
        criterion (torch.nn.Module): Loss function
        device (torch.device): Device to run on
    
    Returns:
        tuple: (average_loss, accuracy, all_predictions, all_targets)
    """
    # TODO: Step 1 - Set model to evaluation mode
    # Hint: Use model.eval()
    # This disables dropout, fixes batch norm, etc.
    model.eval()
    
    # Initialize tracking variables
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    # TODO: Step 2 - Disable gradient computation
    # Hint: Use torch.no_grad() context manager
    # This saves memory and speeds up inference
    with torch.no_grad():
        for data, target in val_loader:
            # TODO: Step 2a - Move data to device
            data, target = data.to(device), target.to(device)
            
            # TODO: Step 2b - Forward pass only
            # Hint: Use model(data) to get predictions
            output = model(data)
            
            # TODO: Step 2c - Compute loss
            # Hint: Use criterion(output, target)
            loss = criterion(output, target)
            
            # Track statistics
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Store predictions and targets for analysis
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # TODO: Step 3 - Calculate average metrics
    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy, all_predictions, all_targets
```

**Key Learning Points**:
- **Why model.eval()?** Disables training-specific behaviors
- **Why torch.no_grad()?** Saves memory, no gradients needed for validation
- **Why store predictions?** Needed for detailed analysis (confusion matrix, etc.)

---

## 📊 Module 4: Analysis Functions

### Function 1: `plot_loss_curves(history, title, save_path=None)`

**Purpose**: Create comprehensive training history plots.

**What you need to implement**:
```python
def plot_training_history(history, title, save_path=None):
    """
    Plot training history with multiple subplots.
    
    Creates 4 subplots:
    1. Loss curves (train vs validation)
    2. Accuracy curves (train vs validation)
    3. Learning rate schedule
    4. Validation loss only (for comparison)
    
    Args:
        history (dict): Training history with keys:
            - 'train_loss': List of training losses
            - 'val_loss': List of validation losses
            - 'train_acc': List of training accuracies
            - 'val_acc': List of validation accuracies
            - 'learning_rates': List of learning rates
        title (str): Title for the plot
        save_path (str, optional): Path to save the plot
    """
    # TODO: Step 1 - Create subplots
    # Hint: Use plt.subplots(2, 2, figsize=(15, 10))
    # This creates a 2x2 grid of subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # TODO: Step 2 - Plot loss curves
    # Hint: Plot both train_loss and val_loss on ax1
    # Use different colors and labels
    ax1.plot(history['train_loss'], label='Train Loss', color='blue')
    ax1.plot(history['val_loss'], label='Val Loss', color='red')
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # TODO: Step 3 - Plot accuracy curves
    # Hint: Plot both train_acc and val_acc on ax2
    ax2.plot(history['train_acc'], label='Train Acc', color='blue')
    ax2.plot(history['val_acc'], label='Val Acc', color='red')
    ax2.set_title('Accuracy Curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    # TODO: Step 4 - Plot learning rate schedule
    # Hint: Plot learning_rates on ax3
    ax3.plot(history['learning_rates'], color='green')
    ax3.set_title('Learning Rate Schedule')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.grid(True)
    
    # TODO: Step 5 - Plot validation loss only
    # Hint: Plot val_loss on ax4 for comparison
    ax4.plot(history['val_loss'], label='Validation Loss', color='red')
    ax4.set_title('Validation Loss')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True)
    
    # TODO: Step 6 - Set overall title and layout
    # Hint: Use plt.suptitle() and plt.tight_layout()
    plt.suptitle(title)
    plt.tight_layout()
    
    # TODO: Step 7 - Save plot if path provided
    # Hint: Use plt.savefig() with high DPI
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
```

### Function 2: `create_results_table(all_results)`

**Purpose**: Create a comprehensive results comparison table.

**What you need to implement**:
```python
def create_results_table(all_results):
    """
    Create a results comparison table.
    
    Args:
        all_results (dict): Results from all experiments
            Structure: {model_name: {strategy: {best_val_acc: float, ...}}}
    
    Returns:
        pandas.DataFrame: Formatted results table
    """
    import pandas as pd
    
    # TODO: Step 1 - Prepare data for table
    # Hint: Create lists for model names, strategies, and accuracies
    models = []
    strategies = []
    accuracies = []
    
    for model_name, model_results in all_results.items():
        for strategy, results in model_results.items():
            models.append(model_name.upper())
            strategies.append(strategy.replace('_', ' ').title())
            accuracies.append(results['best_val_acc'])
    
    # TODO: Step 2 - Create DataFrame
    # Hint: Use pd.DataFrame with columns ['Model', 'Strategy', 'Best Val Acc']
    df = pd.DataFrame({
        'Model': models,
        'Strategy': strategies,
        'Best Val Acc': accuracies
    })
    
    # TODO: Step 3 - Format the table
    # Hint: Round accuracy to 2 decimal places
    df['Best Val Acc'] = df['Best Val Acc'].round(2)
    
    return df
```

### Function 3: `analyze_errors(predictions, targets, class_names)`

**Purpose**: Perform detailed error analysis.

**What you need to implement**:
```python
def analyze_errors(predictions, targets, class_names):
    """
    Analyze prediction errors and identify challenging categories.
    
    Args:
        predictions (list): Model predictions
        targets (list): True labels
        class_names (list): List of class names
    
    Returns:
        dict: Error analysis results
    """
    from sklearn.metrics import classification_report, confusion_matrix
    import numpy as np
    
    # TODO: Step 1 - Calculate confusion matrix
    # Hint: Use confusion_matrix(targets, predictions)
    cm = confusion_matrix(targets, predictions)
    
    # TODO: Step 2 - Calculate per-class accuracy
    # Hint: Diagonal elements / row sums
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    # TODO: Step 3 - Find most challenging classes
    # Hint: Sort by accuracy (ascending)
    class_accuracies = list(zip(class_names, per_class_acc))
    class_accuracies.sort(key=lambda x: x[1])
    
    # TODO: Step 4 - Find most confused pairs
    # Hint: Find off-diagonal elements with highest values
    most_confused = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                most_confused.append((class_names[i], class_names[j], cm[i, j]))
    
    # Sort by confusion count
    most_confused.sort(key=lambda x: x[2], reverse=True)
    
    return {
        'confusion_matrix': cm,
        'per_class_accuracy': dict(class_accuracies),
        'most_challenging': class_accuracies[:5],  # Top 5 most challenging
        'most_confused': most_confused[:10]        # Top 10 most confused pairs
    }
```

---

## 🚀 Running the Complete Experiment

### Step 1: Test Individual Components

```python
# Test data loading
from data_utils import load_datasets, get_transforms
train_loader, val_loader, num_classes = load_datasets()
print(f"Loaded {num_classes} classes")

# Test model setup
from model_utils import setup_resnet50, setup_swin_t
model = setup_resnet50(num_classes)
print(f"ResNet-50 created with {sum(p.numel() for p in model.parameters()):,} parameters")
```

### Step 2: Run Training

```python
# Run the main experiment
python main.py --num_epochs 5 --batch_size 32 --use_wandb
```

### Step 3: Analyze Results

The experiment will generate:
- Training history plots
- Confusion matrices
- Comparison plots
- Results table
- Error analysis

---

## 🎯 Expected Results

After successful implementation, you should see:

1. **Training Progress**: Loss decreasing, accuracy increasing
2. **Model Comparison**: Different performance between ResNet-50 and Swin-T
3. **Strategy Comparison**: Feature extraction vs. full fine-tuning
4. **Error Patterns**: Which pet breeds are most challenging

## 🐛 Common Issues and Solutions

### Issue 1: CUDA Out of Memory
**Solution**: Reduce batch size
```python
python main.py --batch_size 16
```

### Issue 2: Slow Training
**Solution**: Use GPU and increase num_workers
```python
python main.py --device cuda --num_workers 8
```

### Issue 3: Poor Performance
**Solution**: Check data preprocessing and learning rates
- Ensure ImageNet normalization is correct
- Try different learning rates (1e-2 for feature extraction, 1e-3 for fine-tuning)

## 📚 Key Concepts Summary

1. **Transfer Learning**: Using pre-trained models for new tasks
2. **Feature Extraction**: Freezing backbone, training only final layer
3. **Fine-tuning**: Training entire network with lower learning rates
4. **Data Augmentation**: Random transformations to increase dataset diversity
5. **Model Evaluation**: Comprehensive analysis of performance and errors

This implementation will give you hands-on experience with modern deep learning techniques and prepare you for advanced computer vision projects!
