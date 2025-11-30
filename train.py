#!/usr/bin/env python3
"""
Train image classification models using Optuna hyperparameter optimization.
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import optuna
from optuna.trial import TrialState
from pathlib import Path
import json
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

# Suppress Optuna experimental warnings
warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)
# Suppress matplotlib warnings about identical axis limits
warnings.filterwarnings('ignore', message='Attempting to set identical low and high xlims')

# Suppress Optuna logging warnings about unique value length
import logging
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger('optuna').setLevel(logging.ERROR)


class DynamicCNN(nn.Module):
    """Dynamically configured CNN based on Optuna trial suggestions."""
    
    def __init__(self, trial, config, input_channels=1, num_classes=10, input_size=28):
        super(DynamicCNN, self).__init__()
        
        n_conv_layers = trial.suggest_int('n_conv_layers', 
                                          config['n_conv_layers']['low'], 
                                          config['n_conv_layers']['high'])
        
        activation_fn = trial.suggest_categorical('activation', 
                                                  config['activation_functions']['type'])
        
        dropout_rate = trial.suggest_float('dropout_rate',
                                          float(config['dropout_rate']['low']),
                                          float(config['dropout_rate']['high']))
        
        batchnorm_enabled = trial.suggest_categorical('batchnorm_enabled',
                                                      [True, False]) if config['batchNorm']['enabled']['default'] else False
        
        batchnorm_type = None
        if batchnorm_enabled:
            batchnorm_type = trial.suggest_categorical('batchnorm_type',
                                                       config['batchNorm']['type'])
        
        pooling_enabled = trial.suggest_categorical('pooling_enabled',
                                                    [True, False]) if config['pooling']['enabled']['default'] else False
        
        pooling_type = None
        pooling_kernel = None
        pooling_stride = None
        if pooling_enabled:
            pooling_type = trial.suggest_categorical('pooling_type',
                                                     config['pooling']['type'])
            pooling_kernel = trial.suggest_int('pooling_kernel',
                                              config['pooling']['kernel_size']['low'],
                                              config['pooling']['kernel_size']['high'])
            pooling_stride = trial.suggest_int('pooling_stride',
                                              config['pooling']['stride']['low'],
                                              config['pooling']['stride']['high'])
        
        # Build convolutional layers
        self.features = nn.ModuleList()
        current_channels = input_channels
        current_size = input_size
        
        for i in range(n_conv_layers):
            n_filters = trial.suggest_int(f'n_filters_layer_{i}',
                                         config['n_filters']['low'],
                                         config['n_filters']['high'])
            kernel_size = trial.suggest_int(f'kernel_size_layer_{i}',
                                           config['kernel_size']['low'],
                                           config['kernel_size']['high'])
            
            # Make kernel size odd for proper padding
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            padding = kernel_size // 2
            
            # Conv layer
            conv = nn.Conv2d(current_channels, n_filters, kernel_size, padding=padding)
            self.features.append(conv)
            
            # Batch normalization
            if batchnorm_enabled:
                if batchnorm_type == 'BatchNorm2d':
                    self.features.append(nn.BatchNorm2d(n_filters))
                elif batchnorm_type == 'LayerNorm':
                    self.features.append(nn.LayerNorm([n_filters, current_size, current_size]))
                elif batchnorm_type == 'InstanceNorm2d':
                    self.features.append(nn.InstanceNorm2d(n_filters))
            
            # Activation
            self.features.append(self._get_activation(activation_fn))
            
            # Pooling
            if pooling_enabled and i < n_conv_layers - 1:  # Don't pool on last layer
                new_size = (current_size - pooling_kernel) // pooling_stride + 1
                if new_size >= config['min_feature_size']:
                    if pooling_type == 'MaxPool2d':
                        self.features.append(nn.MaxPool2d(pooling_kernel, pooling_stride))
                    elif pooling_type == 'AvgPool2d':
                        self.features.append(nn.AvgPool2d(pooling_kernel, pooling_stride))
                    current_size = new_size
            
            # Dropout
            if dropout_rate > 0:
                self.features.append(nn.Dropout2d(dropout_rate))
            
            current_channels = n_filters
        
        # Calculate final feature size
        self.final_size = current_size * current_size * current_channels
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.final_size, 128),
            self._get_activation(activation_fn),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def _get_activation(self, name):
        activations = {
            'ReLU': nn.ReLU(),
            'LeakyReLU': nn.LeakyReLU(),
            'ELU': nn.ELU(),
            'SELU': nn.SELU(),
            'GELU': nn.GELU(),
            'TANH': nn.Tanh()
        }
        return activations[name]
    
    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        x = self.classifier(x)
        return x


def load_config(config_path='optuna_hpo.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['optuna']


def balance_dataset(dataset, max_imbalance_ratio):
    """Balance dataset according to max_imbalance_ratio."""
    # Get all labels
    if isinstance(dataset, torch.utils.data.Subset):
        labels = [dataset.dataset.targets[i] if hasattr(dataset.dataset, 'targets') 
                 else dataset.dataset[i][1] for i in dataset.indices]
    else:
        labels = [label for _, label in dataset]
    
    # Count samples per class
    from collections import Counter
    class_counts = Counter(labels)
    
    if len(class_counts) == 0:
        return dataset
    
    min_count = min(class_counts.values())
    max_allowed_count = int(min_count * max_imbalance_ratio)
    
    print(f"Class distribution before balancing: {dict(class_counts)}")
    print(f"Min class count: {min_count}, Max allowed: {max_allowed_count}")
    
    # Select indices to keep
    class_indices = {cls: [] for cls in class_counts.keys()}
    
    if isinstance(dataset, torch.utils.data.Subset):
        for idx in dataset.indices:
            label = dataset.dataset.targets[idx] if hasattr(dataset.dataset, 'targets') else dataset.dataset[idx][1]
            class_indices[label].append(idx)
    else:
        for idx, (_, label) in enumerate(dataset):
            class_indices[label].append(idx)
    
    # Limit each class to max_allowed_count
    balanced_indices = []
    for cls, indices in class_indices.items():
        if len(indices) > max_allowed_count:
            # Randomly sample
            perm = torch.randperm(len(indices))[:max_allowed_count]
            selected = [indices[i] for i in perm]
            balanced_indices.extend(selected)
        else:
            balanced_indices.extend(indices)
    
    # Create new subset
    if isinstance(dataset, torch.utils.data.Subset):
        base_dataset = dataset.dataset
    else:
        base_dataset = dataset
    
    balanced_dataset = torch.utils.data.Subset(base_dataset, balanced_indices)
    
    # Print new distribution
    new_labels = [base_dataset.targets[i] if hasattr(base_dataset, 'targets') 
                 else base_dataset[i][1] for i in balanced_indices]
    new_counts = Counter(new_labels)
    print(f"Class distribution after balancing: {dict(new_counts)}")
    
    return balanced_dataset


def prepare_datasets(config):
    """Prepare train, validation, and test datasets."""
    # Get image configuration
    input_channels = config.get('input_image_channels', 1)
    input_size = config.get('input_image_size', [28, 28])
    if isinstance(input_size, list):
        input_size = tuple(input_size)
    
    # Define transforms
    transform_list = []
    if input_channels == 1:
        transform_list.append(transforms.Grayscale(num_output_channels=1))
    
    transform_list.extend([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,) * input_channels, (0.3081,) * input_channels)
    ])
    
    transform = transforms.Compose(transform_list)
    
    # Load full dataset
    full_dataset = datasets.ImageFolder(root=config['image_database_path'], transform=transform)
    
    # Limit dataset size if max_samples is specified
    max_samples = config.get('max_samples')
    if max_samples is not None and max_samples < len(full_dataset):
        indices = torch.randperm(len(full_dataset))[:max_samples]
        full_dataset = torch.utils.data.Subset(full_dataset, indices)
        print(f"Limited dataset to {max_samples} samples")
    
    # Balance dataset if max_imbalance_ratio is specified
    max_imbalance_ratio = config.get('max_inbalance_ratio')  # Note: using 'inbalance' as in YAML
    if max_imbalance_ratio is not None:
        print(f"\nBalancing dataset with max imbalance ratio: {max_imbalance_ratio}")
        full_dataset = balance_dataset(full_dataset, max_imbalance_ratio)
    
    # Calculate splits
    total_size = len(full_dataset)
    train_size = int(config['train_split'] * total_size)
    remaining_size = total_size - train_size
    val_size = int(config['validation_split'] * remaining_size)
    test_size = remaining_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Get number of classes
    if isinstance(full_dataset, torch.utils.data.Subset):
        num_classes = len(full_dataset.dataset.classes)
    else:
        num_classes = len(full_dataset.classes)
    
    return train_dataset, val_dataset, test_dataset, num_classes


def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics."""
    mcc = matthews_corrcoef(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    return {
        'MCC': mcc,
        'ACC': acc,
        'F1': f1,
        'PREC': prec,
        'RECALL': rec
    }


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    metrics = calculate_metrics(all_labels, all_preds)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    metrics = calculate_metrics(all_labels, all_preds)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics, all_labels, all_preds


def plot_confusion_matrix(y_true, y_pred, save_path, num_classes=10):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(num_classes), 
                yticklabels=range(num_classes))
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def objective(trial, config, train_dataset, val_dataset, num_classes, device):
    """Optuna objective function."""
    
    # Suggest hyperparameters
    batch_size = trial.suggest_int('batch_size',
                                   config['batch_size']['low'],
                                   config['batch_size']['high'])
    
    lr = trial.suggest_float('learning_rate',
                            float(config['learning_rate']['low']),
                            float(config['learning_rate']['high']),
                            log=config['learning_rate'].get('log', False))
    
    weight_decay = trial.suggest_float('weight_decay',
                                      float(config['weight_decay']['low']),
                                      float(config['weight_decay']['high']),
                                      log=config['weight_decay'].get('log', False))
    
    momentum = trial.suggest_float('momentum',
                                  config['momentum']['low'],
                                  config['momentum']['high'])
    
    lr_schedule = trial.suggest_categorical('learning_rate_schedule',
                                           config['learning_rate_schedule']['type'])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Get image configuration
    input_channels = config.get('input_image_channels', 1)
    input_size = config.get('input_image_size', [28, 28])
    if isinstance(input_size, list):
        input_size = input_size[0]  # Assume square images
    
    # Create model
    model = DynamicCNN(trial, config, input_channels=input_channels, 
                      num_classes=num_classes, input_size=input_size).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = None
    if lr_schedule == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif lr_schedule == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    elif lr_schedule == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs_per_trial'])
    
    # Training loop
    best_metric = -float('inf')
    patience_counter = 0
    patience = config['early_stopping']['patience']
    
    metric_name = config['objective_metric']
    
    # Initialize history for tracking
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_MCC': [],
        'val_MCC': [],
        'train_ACC': [],
        'val_ACC': [],
        'train_F1': [],
        'val_F1': [],
        'train_PREC': [],
        'val_PREC': [],
        'train_RECALL': [],
        'val_RECALL': []
    }
    
    # Use checkpoint config for data frequency if save_metric_data is enabled
    if config['checkpoint'].get('save_metric_data', False):
        epoch_data_frequency = config['checkpoint'].get('epoch_data_frequency', 1)
    else:
        # Fall back to matplotlib config for plotting only
        epoch_data_frequency = config['matplotlib']['training_curves'].get('epoch_data_frequency', 1)
    
    for epoch in range(config['epochs_per_trial']):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, _, _ = validate(model, val_loader, criterion, device)
        
        if scheduler:
            scheduler.step()
        
        # Record metrics at specified frequency
        if epoch % epoch_data_frequency == 0 or epoch == config['epochs_per_trial'] - 1:
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['train_MCC'].append(train_metrics['MCC'])
            history['val_MCC'].append(val_metrics['MCC'])
            history['train_ACC'].append(train_metrics['ACC'])
            history['val_ACC'].append(val_metrics['ACC'])
            history['train_F1'].append(train_metrics['F1'])
            history['val_F1'].append(val_metrics['F1'])
            history['train_PREC'].append(train_metrics['PREC'])
            history['val_PREC'].append(val_metrics['PREC'])
            history['train_RECALL'].append(train_metrics['RECALL'])
            history['val_RECALL'].append(val_metrics['RECALL'])
        
        current_metric = val_metrics[metric_name]
        
        # Report intermediate value
        trial.report(current_metric, epoch)
        
        # Handle pruning
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        # Early stopping
        if current_metric > best_metric:
            best_metric = current_metric
            patience_counter = 0
            
            # Save best model
            checkpoint_dir = Path(config['base_output_dir']) / Path(config['checkpoint']['log_dir']).name
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_data = {
                'trial_number': trial.number,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'params': trial.params
            }
            
            # Add history if save_metric_data is enabled
            if config['checkpoint'].get('save_metric_data', False):
                checkpoint_data['history'] = history
            
            torch.save(checkpoint_data, checkpoint_dir / f'trial_{trial.number}_best.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Save last checkpoint if enabled
    if config['checkpoint'].get('save_last', False):
        checkpoint_dir = Path(config['base_output_dir']) / Path(config['checkpoint']['log_dir']).name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_data = {
            'trial_number': trial.number,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': val_metrics,
            'params': trial.params
        }
        
        # Add history if save_metric_data is enabled
        if config['checkpoint'].get('save_metric_data', False):
            checkpoint_data['history'] = history
        
        torch.save(checkpoint_data, checkpoint_dir / f'trial_{trial.number}_last.pt')
    
    # Plot training curves after trial completes
    plot_training_curves(trial.number, history, config)
    
    return best_metric


def create_plots(study, config):
    """Create Optuna visualization plots."""
    if not config['matplotlib']['optuna_results']['enabled']:
        return
    
    plot_dir = Path(config['base_output_dir']) / Path(config['matplotlib']['optuna_results']['plot_dir']).name
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    plot_types = config['matplotlib']['optuna_results']['plot_types']
    
    try:
        if 'optimization_history' in plot_types:
            fig = optuna.visualization.matplotlib.plot_optimization_history(study)
            plt.tight_layout()
            plt.savefig(plot_dir / 'optimization_history.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        if 'param_importances' in plot_types and len(study.trials) > 1:
            fig = optuna.visualization.matplotlib.plot_param_importances(study)
            # Adjust figure size and layout to prevent label cutoff
            # Get the current figure if fig is an Axes object
            if hasattr(fig, 'figure'):
                fig = fig.figure
            fig.set_size_inches(10, max(6, len(study.best_params) * 0.4))
            plt.tight_layout()
            plt.savefig(plot_dir / 'param_importances.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"Optuna plots saved to {plot_dir}")
    except Exception as e:
        print(f"Error creating plots: {e}")


def plot_training_curves(trial_number, history, config):
    """Plot training curves for a trial."""
    if not config['matplotlib']['training_curves']['enabled']:
        return
    
    plot_dir = Path(config['base_output_dir']) / Path(config['matplotlib']['training_curves']['plot_dir']).name
    
    # Create trial subfolder if enabled
    if config['matplotlib']['training_curves'].get('plot_each_trial', True):
        plot_dir = plot_dir / f'trial_{trial_number}'
    
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    plot_types = config['matplotlib']['training_curves']['plot_types']
    plot_stacked = config['matplotlib']['training_curves'].get('plot_stacked_metric_with_loss', True)
    
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    plt.plot(epochs, history['val_loss'], label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Trial {trial_number} - Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / 'loss.png', dpi=150)
    plt.close()
    
    # Plot each metric
    for metric in plot_types:
        train_key = f'train_{metric}'
        val_key = f'val_{metric}'
        
        if val_key in history and train_key in history:
            if plot_stacked:
                # Stacked plot with loss and metric
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
                
                # Loss subplot
                ax1.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
                ax1.plot(epochs, history['val_loss'], label='Validation Loss', marker='s')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.set_title(f'Trial {trial_number} - Training Loss')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Metric subplot
                ax2.plot(epochs, history[train_key], label=f'Train {metric}', marker='o', color='blue')
                ax2.plot(epochs, history[val_key], label=f'Validation {metric}', marker='s', color='green')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel(metric)
                ax2.set_title(f'Trial {trial_number} - {metric}')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(plot_dir / f'{metric}_with_loss.png', dpi=150)
                plt.close()
            else:
                # Individual metric plot
                plt.figure(figsize=(10, 6))
                plt.plot(epochs, history[train_key], label=f'Train {metric}', marker='o', color='blue')
                plt.plot(epochs, history[val_key], label=f'Validation {metric}', marker='s', color='green')
                plt.xlabel('Epoch')
                plt.ylabel(metric)
                plt.title(f'Trial {trial_number} - {metric}')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(plot_dir / f'{metric}.png', dpi=150)
                plt.close()
    
    print(f"Training curves saved to {plot_dir}")


def main():
    """Main training function."""
    # Load configuration
    config = load_config()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare datasets
    print("Loading datasets...")
    train_dataset, val_dataset, test_dataset, num_classes = prepare_datasets(config)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}, Classes: {num_classes}")
    
    # Create output directory
    output_dir = Path(config['base_output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create Optuna study
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner()
    
    study = optuna.create_study(
        study_name=config['study_name'],
        direction=config['direction'],
        sampler=sampler,
        pruner=pruner
    )
    
    # Optimize (no callbacks - Optuna plots only at the end)
    print(f"Starting optimization with {config['n_trials']} trials...")
    study.optimize(
        lambda trial: objective(trial, config, train_dataset, val_dataset, num_classes, device),
        n_trials=config['n_trials'],
        timeout=config.get('timeout'),
        show_progress_bar=True
    )
    
    # Print results
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best {config['objective_metric']}: {study.best_trial.value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Save study results
    results_path = output_dir / 'study_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'best_trial': study.best_trial.number,
            'best_value': study.best_trial.value,
            'best_params': study.best_trial.params,
            'n_trials': len(study.trials),
            'objective_metric': config['objective_metric']
        }, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    # Create plots
    print("\nGenerating visualization plots...")
    create_plots(study, config)
    
    print("\nTraining complete! Use test.py to evaluate the best model on the test set.")


if __name__ == "__main__":
    main()
