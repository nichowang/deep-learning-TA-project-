# Part 1: Image Classification with Oxford Pet Dataset

## Problem Description

This assignment focuses on image classification using the Oxford Pet Dataset with PyTorch. We will fine-tune pre-trained deep learning models to classify pet images into 37 different categories.

## Dataset

- **Dataset**: Oxford Pet Dataset
- **Categories**: 37 different pet breeds
- **Images per category**: ~200 images
- **Total images**: ~7,400 images
- **Dataset source**: Built into PyTorch (`torchvision.datasets.OxfordIIITPet`)
- **Documentation**: [PyTorch OxfordIIITPet](https://pytorch.org/vision/stable/generated/torchvision.datasets.OxfordIIITPet.html)
- **Original dataset info**: [Oxford Pet Dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/)

## Models

We will fine-tune two pre-trained architectures:

1. **ResNet-50** with ImageNet weights
2. **Swin Transformer (Swin-T)** with ImageNet weights

For each model, we will replace the final classification layer to output predictions for the 37 pet categories.

## Training Strategy

Each architecture will be trained using two different approaches:

### 1. Feature Extraction (Frozen Backbone)
- **Trainable parameters**: Only the final classification layer
- **Frozen parameters**: All pre-trained layers remain frozen
- **Purpose**: Evaluate how well the pre-trained features transfer to the pet classification task

### 2. Fine-tuning (Full Network)
- **Trainable parameters**: All layers in the network
- **Learning rate**: Typically lower for pre-trained layers
- **Purpose**: Adapt the entire network to the specific pet classification task

## Data Preprocessing

### Image Transformations
- **Resizing**: Resize images to dimensions compatible with the chosen network
- **Normalization**: 
  - Subtract mean pixel values
  - Divide by standard deviation
- **Data Augmentation**:
  - Random horizontal flips
  - Random crops

### Implementation Requirements
- Use appropriate PyTorch transforms
- Ensure transformations are compatible with both ResNet-50 and Swin-T architectures
- Apply consistent preprocessing for training and validation sets

## Expected Deliverables

### Code Implementation
1. **Model Implementation**: Code for both ResNet-50 and Swin-T fine-tuning
2. **Training Scripts**: Separate training loops for feature extraction and full fine-tuning
3. **Data Loading**: Proper dataset loading with appropriate transforms
4. **Logging Setup**: Integration with Weights and Biases or TensorBoard for experiment tracking

### Analysis and Results
5. **Loss Curves Visualization**: 
   - Plot train and test loss curves for both ResNet-50 and Swin-T as a function of epochs on the same plot
   - Ensure all lines are properly labeled
   - Use Weights and Biases or TensorBoard for visualization
   - Include both feature extraction and full fine-tuning results

6. **Performance Analysis**:
   - **Observations**: Analyze the loss curves and discuss what you observe regarding fine-tuning just the output layer compared to the entire network
   - **Accuracy Table**: Create a table reporting the best accuracy achieved by each approach:
     - ResNet-50 (Feature Extraction)
     - ResNet-50 (Full Fine-tuning)
     - Swin-T (Feature Extraction)
     - Swin-T (Full Fine-tuning)

7. **Error Analysis**:
   - Analyze the errors that each model makes
   - Compare error patterns between ResNet-50 and Swin-T
   - Discuss whether one architecture performs significantly better at some categories than others
   - Identify which pet categories are most challenging for each model

## Technical Considerations

- **Hardware**: Ensure compatibility with available computational resources
- **Memory Management**: Consider batch size limitations for fine-tuning
- **Learning Rate Scheduling**: Implement appropriate learning rate strategies
- **Early Stopping**: Prevent overfitting during training
- **Model Checkpointing**: Save best models for evaluation

## Required Analysis Components

### 1. Loss Curves Visualization
- **Plot Requirements**: 
  - Train and test loss curves for both ResNet-50 and Swin-T
  - All curves on the same plot for easy comparison
  - X-axis: epochs, Y-axis: loss values
  - Proper labeling of all lines (e.g., "ResNet-50 Train", "ResNet-50 Test", "Swin-T Train", "Swin-T Test")
- **Recommended Tools**: Weights and Biases (wandb) or TensorBoard
- **Include Both Training Strategies**: Feature extraction and full fine-tuning results

### 2. Performance Observations
- **Analysis Questions to Address**:
  - What do the loss curves tell you about feature extraction vs full fine-tuning?
  - Which approach converges faster?
  - Which approach achieves lower final loss?
  - Are there signs of overfitting in either approach?

### 3. Accuracy Results Table
Create a comprehensive table with the following structure:

| Model | Training Strategy | Best Test Accuracy |
|-------|------------------|-------------------|
| ResNet-50 | Feature Extraction | X.XX% |
| ResNet-50 | Full Fine-tuning | X.XX% |
| Swin-T | Feature Extraction | X.XX% |
| Swin-T | Full Fine-tuning | X.XX% |

### 4. Error Analysis
- **Confusion Matrix**: Generate confusion matrices for each model/strategy combination
- **Category-wise Performance**: Identify which pet breeds are most/least accurately classified
- **Architecture Comparison**: 
  - Does ResNet-50 excel at certain types of pets?
  - Does Swin-T perform better on specific categories?
  - Are there consistent patterns in misclassifications?
- **Discussion Points**:
  - Which categories are most challenging across all models?
  - Are there visual similarities between commonly confused breeds?
  - How do the architectures differ in their error patterns?

## Success Metrics

- **Accuracy**: Classification accuracy on test set
- **Training Efficiency**: Time and computational cost comparison
- **Convergence**: Training and validation loss curves
- **Generalization**: Performance gap between training and validation sets
- **Analysis Depth**: Quality of insights from error analysis and performance comparison
