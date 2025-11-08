"""
Experiment tracking utilities using Weights and Biases.
"""

import wandb
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns


class ExperimentTracker:
    """
    Experiment tracker using Weights and Biases.
    """
    
    def __init__(self, project_name="oxford-pets-classification", entity=None):
        """
        Initialize the experiment tracker.
        
        Args:
            project_name (str): Name of the W&B project
            entity (str, optional): W&B entity name
        """
        self.project_name = project_name
        self.entity = entity
        self.run = None
    
    def start_run(self, config, name=None, tags=None):
        """
        Start a new experiment run.
        
        Args:
            config (dict): Configuration dictionary
            name (str, optional): Name for the run
            tags (list, optional): Tags for the run
        """
        self.run = wandb.init(
            project=self.project_name,
            entity=self.entity,
            config=config,
            name=name,
            tags=tags
        )
    
    def log_metrics(self, metrics, step=None):
        """
        Log metrics to W&B.
        
        Args:
            metrics (dict): Dictionary of metrics to log
            step (int, optional): Step number
        """
        if self.run is not None:
            wandb.log(metrics, step=step)
    
    def log_model_info(self, model, model_name):
        """
        Log model information.
        
        Args:
            model (torch.nn.Module): The model
            model_name (str): Name of the model
        """
        if self.run is not None:
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            wandb.log({
                f"{model_name}_total_parameters": total_params,
                f"{model_name}_trainable_parameters": trainable_params,
                f"{model_name}_trainable_percentage": 100 * trainable_params / total_params
            })
    
    def log_training_history(self, history, model_name, training_strategy):
        """
        Log training history.
        
        Args:
            history (dict): Training history
            model_name (str): Name of the model
            training_strategy (str): Training strategy used
        """
        if self.run is not None:
            # Log final metrics
            final_train_loss = history['train_loss'][-1]
            final_val_loss = history['val_loss'][-1]
            final_train_acc = history['train_acc'][-1]
            final_val_acc = history['val_acc'][-1]
            best_val_acc = max(history['val_acc'])
            
            wandb.log({
                f"{model_name}_{training_strategy}_final_train_loss": final_train_loss,
                f"{model_name}_{training_strategy}_final_val_loss": final_val_loss,
                f"{model_name}_{training_strategy}_final_train_acc": final_train_acc,
                f"{model_name}_{training_strategy}_final_val_acc": final_val_acc,
                f"{model_name}_{training_strategy}_best_val_acc": best_val_acc
            })
            
            # Log learning rate schedule
            for epoch, lr in enumerate(history['learning_rates']):
                wandb.log({f"{model_name}_{training_strategy}_learning_rate": lr}, step=epoch)
    
    def log_confusion_matrix(self, conf_matrix, class_names, model_name, training_strategy):
        """
        Log confusion matrix.
        
        Args:
            conf_matrix (np.ndarray): Confusion matrix
            class_names (list): List of class names
            model_name (str): Name of the model
            training_strategy (str): Training strategy used
        """
        if self.run is not None:
            # Create confusion matrix plot
            plt.figure(figsize=(12, 10))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
            plt.title(f'Confusion Matrix - {model_name} ({training_strategy})')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Log to W&B
            wandb.log({f"{model_name}_{training_strategy}_confusion_matrix": wandb.Image(plt)})
            plt.close()
    
    def log_classification_report(self, report, model_name, training_strategy):
        """
        Log classification report.
        
        Args:
            report (dict): Classification report
            model_name (str): Name of the model
            training_strategy (str): Training strategy used
        """
        if self.run is not None:
            # Log overall metrics
            wandb.log({
                f"{model_name}_{training_strategy}_precision": report['macro avg']['precision'],
                f"{model_name}_{training_strategy}_recall": report['macro avg']['recall'],
                f"{model_name}_{training_strategy}_f1_score": report['macro avg']['f1-score']
            })
            
            # Log per-class metrics (sample a few classes to avoid clutter)
            for class_name, metrics in list(report.items())[:10]:  # First 10 classes
                if isinstance(metrics, dict) and 'precision' in metrics:
                    wandb.log({
                        f"{model_name}_{training_strategy}_{class_name}_precision": metrics['precision'],
                        f"{model_name}_{training_strategy}_{class_name}_recall": metrics['recall'],
                        f"{model_name}_{training_strategy}_{class_name}_f1": metrics['f1-score']
                    })
    
    def log_error_analysis(self, predictions, targets, class_names, model_name, training_strategy):
        """
        Log error analysis.
        
        Args:
            predictions (list): Model predictions
            targets (list): True labels
            class_names (list): List of class names
            model_name (str): Name of the model
            training_strategy (str): Training strategy used
        """
        if self.run is not None:
            # Calculate per-class accuracy
            per_class_acc = []
            for class_idx in range(len(class_names)):
                class_mask = np.array(targets) == class_idx
                if np.sum(class_mask) > 0:
                    class_acc = np.mean(np.array(predictions)[class_mask] == class_idx)
                    per_class_acc.append(class_acc)
                else:
                    per_class_acc.append(0.0)
            
            # Create per-class accuracy plot
            plt.figure(figsize=(15, 8))
            plt.bar(range(len(class_names)), per_class_acc)
            plt.title(f'Per-Class Accuracy - {model_name} ({training_strategy})')
            plt.xlabel('Class')
            plt.ylabel('Accuracy')
            plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
            plt.tight_layout()
            
            wandb.log({f"{model_name}_{training_strategy}_per_class_accuracy": wandb.Image(plt)})
            plt.close()
            
            # Log worst performing classes
            worst_classes = np.argsort(per_class_acc)[:5]  # 5 worst classes
            for i, class_idx in enumerate(worst_classes):
                wandb.log({
                    f"{model_name}_{training_strategy}_worst_class_{i+1}": class_names[class_idx],
                    f"{model_name}_{training_strategy}_worst_class_{i+1}_accuracy": per_class_acc[class_idx]
                })
    
    def finish_run(self):
        """
        Finish the current run.
        """
        if self.run is not None:
            wandb.finish()
            self.run = None


def create_comparison_plots(all_results, save_path=None):
    """
    Create comparison plots for all models and strategies.
    
    Args:
        all_results (dict): Results from all experiments
        save_path (str, optional): Path to save the plots
    """
    # Extract data for plotting
    models = []
    strategies = []
    accuracies = []
    train_losses = []
    val_losses = []
    
    for model_name, model_results in all_results.items():
        for strategy, results in model_results.items():
            models.append(model_name)
            strategies.append(strategy)
            accuracies.append(results['best_val_acc'])
            train_losses.append(results['history']['train_loss'][-1])
            val_losses.append(results['history']['val_loss'][-1])
    
    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy comparison
    x_pos = np.arange(len(models))
    colors = ['blue', 'red', 'green', 'orange']
    bars = ax1.bar(x_pos, accuracies, color=colors)
    ax1.set_title('Best Validation Accuracy Comparison')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"{m}\n{s}" for m, s in zip(models, strategies)], rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom')
    
    # Final training loss comparison
    bars2 = ax2.bar(x_pos, train_losses, color=colors)
    ax2.set_title('Final Training Loss Comparison')
    ax2.set_ylabel('Loss')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"{m}\n{s}" for m, s in zip(models, strategies)], rotation=45, ha='right')
    
    # Final validation loss comparison
    bars3 = ax3.bar(x_pos, val_losses, color=colors)
    ax3.set_title('Final Validation Loss Comparison')
    ax3.set_ylabel('Loss')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f"{m}\n{s}" for m, s in zip(models, strategies)], rotation=45, ha='right')
    
    # Loss curves comparison
    for model_name, model_results in all_results.items():
        for strategy, results in model_results.items():
            label = f"{model_name} - {strategy}"
            ax4.plot(results['history']['val_loss'], label=label, linewidth=2)
    
    ax4.set_title('Validation Loss Curves Comparison')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Log to W&B if available
    try:
        wandb.log({"comparison_plots": wandb.Image(plt)})
    except:
        pass
