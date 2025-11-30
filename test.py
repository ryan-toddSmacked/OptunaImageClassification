#!/usr/bin/env python3
"""
Test the best model from Optuna hyperparameter optimization on the test set.
"""

import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (matthews_corrcoef, accuracy_score, f1_score, 
                             precision_score, recall_score, confusion_matrix,
                             classification_report)
from tqdm import tqdm


class DynamicCNN(nn.Module):
    """Dynamically configured CNN - must match train.py architecture."""
    
    def __init__(self, params, config, input_channels=1, num_classes=10, input_size=28):
        super(DynamicCNN, self).__init__()
        
        n_conv_layers = params['n_conv_layers']
        activation_fn = params['activation']
        dropout_rate = params['dropout_rate']
        batchnorm_enabled = params.get('batchnorm_enabled', False)
        batchnorm_type = params.get('batchnorm_type', None)
        pooling_enabled = params.get('pooling_enabled', False)
        pooling_type = params.get('pooling_type', None)
        pooling_kernel = params.get('pooling_kernel', None)
        pooling_stride = params.get('pooling_stride', None)
        
        # Build convolutional layers
        self.features = nn.ModuleList()
        current_channels = input_channels
        current_size = input_size
        
        for i in range(n_conv_layers):
            n_filters = params[f'n_filters_layer_{i}']
            kernel_size = params[f'kernel_size_layer_{i}']
            
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
            if pooling_enabled and i < n_conv_layers - 1:
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


def prepare_test_dataset(config):
    """Prepare test dataset."""
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
    
    # Limit dataset size if max_samples is specified (must match train.py)
    max_samples = config.get('max_samples')
    if max_samples is not None and max_samples < len(full_dataset):
        indices = torch.randperm(len(full_dataset))[:max_samples]
        full_dataset = torch.utils.data.Subset(full_dataset, indices)
        print(f"Limited dataset to {max_samples} samples")
    
    # Balance dataset if max_imbalance_ratio is specified (must match train.py)
    max_imbalance_ratio = config.get('max_inbalance_ratio')  # Note: using 'inbalance' as in YAML
    if max_imbalance_ratio is not None:
        print(f"\nBalancing dataset with max imbalance ratio: {max_imbalance_ratio}")
        full_dataset = balance_dataset(full_dataset, max_imbalance_ratio)
    
    # Calculate splits (must match train.py)
    total_size = len(full_dataset)
    train_size = int(config['train_split'] * total_size)
    remaining_size = total_size - train_size
    val_size = int(config['validation_split'] * remaining_size)
    test_size = remaining_size - val_size
    
    # Split dataset with same seed
    _, _, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Get classes info
    if isinstance(full_dataset, torch.utils.data.Subset):
        num_classes = len(full_dataset.dataset.classes)
        class_names = full_dataset.dataset.classes
    else:
        num_classes = len(full_dataset.classes)
        class_names = full_dataset.classes
    
    return test_dataset, num_classes, class_names


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


def test_model(model, dataloader, device):
    """Test the model and return predictions."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("Testing model...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Test Set', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_per_class_metrics(y_true, y_pred, class_names, save_path):
    """Plot per-class precision, recall, and F1 scores."""
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Metrics - Test Set', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Per-class metrics plot saved to {save_path}")


def load_best_model(config, device):
    """Load the best model from training."""
    # Load study results
    results_path = Path(config['base_output_dir']) / 'study_results.json'
    if not results_path.exists():
        raise FileNotFoundError(f"Study results not found at {results_path}. Run train.py first.")
    
    with open(results_path, 'r') as f:
        study_results = json.load(f)
    
    best_trial = study_results['best_trial']
    
    # Find best checkpoint
    checkpoint_dir = Path(config['base_output_dir']) / Path(config['checkpoint']['log_dir']).name
    checkpoint_path = checkpoint_dir / f'trial_{best_trial}_best.pt'
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    return checkpoint, study_results


def main():
    """Main testing function."""
    # Load configuration
    config = load_config()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare test dataset
    print("Loading test dataset...")
    test_dataset, num_classes, class_names = prepare_test_dataset(config)
    print(f"Test set size: {len(test_dataset)}")
    print(f"Number of classes: {num_classes}")
    
    # Load best model
    print("\nLoading best model from training...")
    checkpoint, study_results = load_best_model(config, device)
    
    print(f"Best trial: {study_results['best_trial']}")
    print(f"Best validation {study_results['objective_metric']}: {study_results['best_value']:.4f}")
    
    # Get image configuration
    input_channels = config.get('input_image_channels', 1)
    input_size = config.get('input_image_size', [28, 28])
    if isinstance(input_size, list):
        input_size = input_size[0]  # Assume square images
    
    # Create model with best hyperparameters
    model = DynamicCNN(checkpoint['params'], config, 
                       input_channels=input_channels, num_classes=num_classes, 
                       input_size=input_size).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create test dataloader
    batch_size = checkpoint['params']['batch_size']
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Test model
    y_true, y_pred, y_probs = test_model(model, test_loader, device)
    
    # Calculate metrics
    print("\n" + "="*80)
    print("TEST SET RESULTS")
    print("="*80)
    
    test_metrics = calculate_metrics(y_true, y_pred)
    
    print(f"\nOverall Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Print classification report
    print("\nPer-Class Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    
    # Create output directory for test results
    test_output_dir = Path(config['base_output_dir']) / 'test_results'
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save test results
    results = {
        'test_metrics': test_metrics,
        'best_trial': study_results['best_trial'],
        'best_validation_metric': study_results['best_value'],
        'best_params': checkpoint['params'],
        'num_test_samples': len(test_dataset)
    }
    
    results_path = test_output_dir / 'test_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nTest results saved to {results_path}")
    
    # Generate plots
    print("\nGenerating visualizations...")
    
    # Confusion matrix
    cm_path = test_output_dir / 'confusion_matrix.png'
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path)
    
    # Per-class metrics
    pcm_path = test_output_dir / 'per_class_metrics.png'
    plot_per_class_metrics(y_true, y_pred, class_names, pcm_path)
    
    print("\n" + "="*80)
    print("Testing complete!")
    print(f"Results saved to {test_output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
