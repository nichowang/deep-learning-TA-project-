"""
Main script for Oxford Pet Dataset classification experiment.
"""

import torch
import torch.nn as nn
import argparse
import os
from datetime import datetime

# Import our custom modules
from data_utils import load_datasets, get_class_names
from model_utils import (
    setup_resnet50, setup_swin_t, freeze_backbone, 
    print_model_info, setup_optimizer, setup_scheduler
)
from training_utils import train_model, evaluate_model, plot_training_history, plot_confusion_matrix
from experiment_tracker import ExperimentTracker, create_comparison_plots


def main():
    """
    Main function to run the complete experiment.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description='Oxford Pet Dataset Classification')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--image_size', type=int, default=224, help='Image size')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases')
    parser.add_argument('--save_models', action='store_true', help='Save trained models')
    parser.add_argument('--results_dir', type=str, default='./results', help='Results directory')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    train_loader, val_loader, num_classes = load_datasets(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size
    )
    
    # Get class names
    class_names = get_class_names(args.data_dir)
    
    # Initialize experiment tracker
    tracker = None
    if args.use_wandb:
        tracker = ExperimentTracker(project_name="oxford-pets-classification")
    
    # Define models and strategies to test
    models_config = {
        'resnet50': setup_resnet50,
        'swin_t': setup_swin_t
    }
    
    strategies = ['feature_extraction', 'full_finetuning']
    
    # Store all results
    all_results = {}
    
    # Run experiments
    for model_name, model_setup_func in models_config.items():
        print(f"\n{'='*60}")
        print(f"Starting experiments with {model_name.upper()}")
        print(f"{'='*60}")
        
        all_results[model_name] = {}
        
        for strategy in strategies:
            print(f"\n{'-'*40}")
            print(f"Training Strategy: {strategy}")
            print(f"{'-'*40}")
            
            # Setup model
            model = model_setup_func(num_classes, pretrained=True)
            
            # Apply training strategy
            if strategy == 'feature_extraction':
                model = freeze_backbone(model, model_name)
                lr = args.learning_rate * 10  # Higher LR for feature extraction
            else:  # full_finetuning
                lr = args.learning_rate
            
            model = model.to(device)
            
            # Print model info
            print_model_info(model, model_name)
            
            # Start experiment tracking
            if tracker:
                config = {
                    'model': model_name,
                    'strategy': strategy,
                    'num_classes': num_classes,
                    'batch_size': args.batch_size,
                    'num_epochs': args.num_epochs,
                    'learning_rate': lr,
                    'image_size': args.image_size
                }
                
                run_name = f"{model_name}_{strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                tracker.start_run(config, name=run_name, tags=[model_name, strategy])
                tracker.log_model_info(model, model_name)
            
            # Train model
            results = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=args.num_epochs,
                learning_rate=lr,
                device=device,
                model_name=model_name,
                training_strategy=strategy
            )
            
            # Evaluate model
            print(f"\nEvaluating {model_name} - {strategy}...")
            eval_results = evaluate_model(model, val_loader, device, class_names)
            
            # Store results
            all_results[model_name][strategy] = {
                'history': results['history'],
                'best_val_acc': results['best_val_acc'],
                'eval_results': eval_results
            }
            
            # Log results to tracker
            if tracker:
                tracker.log_training_history(results['history'], model_name, strategy)
                tracker.log_confusion_matrix(
                    eval_results['confusion_matrix'], 
                    class_names, 
                    model_name, 
                    strategy
                )
                tracker.log_classification_report(
                    eval_results['classification_report'], 
                    model_name, 
                    strategy
                )
                tracker.log_error_analysis(
                    eval_results['predictions'],
                    eval_results['targets'],
                    class_names,
                    model_name,
                    strategy
                )
                tracker.finish_run()
            
            # Save model if requested
            if args.save_models:
                model_path = os.path.join(args.results_dir, f"{model_name}_{strategy}.pth")
                torch.save(model.state_dict(), model_path)
                print(f"Model saved to: {model_path}")
            
            # Plot training history
            plot_path = os.path.join(args.results_dir, f"{model_name}_{strategy}_history.png")
            plot_training_history(
                results['history'], 
                f"{model_name.upper()} - {strategy.replace('_', ' ').title()}",
                plot_path
            )
            
            # Plot confusion matrix
            cm_path = os.path.join(args.results_dir, f"{model_name}_{strategy}_confusion_matrix.png")
            plot_confusion_matrix(
                eval_results['confusion_matrix'],
                class_names,
                f"{model_name.upper()} - {strategy.replace('_', ' ').title()}",
                cm_path
            )
            
            print(f"Results saved to: {args.results_dir}")
    
    # Create comparison plots
    print(f"\n{'='*60}")
    print("Creating comparison plots...")
    print(f"{'='*60}")
    
    comparison_path = os.path.join(args.results_dir, "comparison_plots.png")
    create_comparison_plots(all_results, comparison_path)
    
    # Print final results table
    print(f"\n{'='*60}")
    print("FINAL RESULTS TABLE")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'Strategy':<20} {'Best Val Acc':<15}")
    print("-" * 50)
    
    for model_name, model_results in all_results.items():
        for strategy, results in model_results.items():
            strategy_display = strategy.replace('_', ' ').title()
            print(f"{model_name:<15} {strategy_display:<20} {results['best_val_acc']:<15.2f}%")
    
    # Error analysis summary
    print(f"\n{'='*60}")
    print("ERROR ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    for model_name, model_results in all_results.items():
        print(f"\n{model_name.upper()} Results:")
        for strategy, results in model_results.items():
            eval_results = results['eval_results']
            report = eval_results['classification_report']
            
            print(f"  {strategy.replace('_', ' ').title()}:")
            print(f"    Overall Accuracy: {eval_results['accuracy']:.3f}")
            print(f"    Macro Avg Precision: {report['macro avg']['precision']:.3f}")
            print(f"    Macro Avg Recall: {report['macro avg']['recall']:.3f}")
            print(f"    Macro Avg F1-Score: {report['macro avg']['f1-score']:.3f}")
    
    print(f"\nExperiment completed! Results saved to: {args.results_dir}")


if __name__ == "__main__":
    main()
