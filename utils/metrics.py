"""Metrics and evaluation utilities."""

import numpy as np
from scipy import stats


def compute_accuracy(predictions, labels):
    """
    Compute classification accuracy.
    
    Args:
        predictions: Predicted class indices
        labels: True class indices
    
    Returns:
        Accuracy as float
    """
    correct = (predictions == labels).sum()
    total = len(labels)
    return correct / total


def compute_top_k_accuracy(outputs, labels, k=5):
    """
    Compute top-k accuracy.
    
    Args:
        outputs: Model outputs (logits or probabilities)
        labels: True class indices
        k: Top k predictions to consider
    
    Returns:
        Top-k accuracy as float
    """
    _, top_k_preds = outputs.topk(k, dim=1)
    labels_expanded = labels.view(-1, 1).expand_as(top_k_preds)
    correct = (top_k_preds == labels_expanded).sum().item()
    total = len(labels)
    return correct / total


def compute_distance_statistics(distances):
    """
    Compute statistics for genetic distances of misclassifications.
    
    Args:
        distances: Array of genetic distances
    
    Returns:
        Dictionary of statistics
    """
    if len(distances) == 0:
        return {
            'n': 0,
            'mean': 0,
            'std': 0,
            'q1': 0,
            'median': 0,
            'q3': 0,
            'min': 0,
            'max': 0
        }
    
    return {
        'n': len(distances),
        'mean': np.mean(distances),
        'std': np.std(distances),
        'q1': np.percentile(distances, 25),
        'median': np.percentile(distances, 50),
        'q3': np.percentile(distances, 75),
        'min': np.min(distances),
        'max': np.max(distances)
    }


def perform_ttest(distances1, distances2, name1='Model 1', name2='Model 2'):
    """
    Perform t-test between two sets of error distances.
    
    Args:
        distances1: First set of distances
        distances2: Second set of distances
        name1: Name of first model
        name2: Name of second model
    
    Returns:
        Dictionary with test results
    """
    statistic, pvalue = stats.ttest_ind(distances1, distances2)
    
    return {
        'comparison': f"{name1} vs {name2}",
        'statistic': statistic,
        'pvalue': pvalue,
        'significant': pvalue < 0.05,
        'mean1': np.mean(distances1),
        'mean2': np.mean(distances2)
    }


def save_metrics(metrics, filepath):
    """
    Save metrics to text file.
    
    Args:
        metrics: Dictionary of metrics
        filepath: Path to save file
    """
    with open(filepath, 'w') as f:
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{key}\t{value:.6f}\n")
            else:
                f.write(f"{key}\t{value}\n")
    
    print(f"Metrics saved to {filepath}")


class MetricsTracker:
    """Track metrics during training."""
    
    def __init__(self):
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        self.best_val_acc = 0.0
        self.best_epoch = 0
    
    def update(self, epoch, train_loss, train_acc, val_loss, val_acc):
        """Update metrics for an epoch."""
        self.metrics['train_loss'].append(train_loss)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['val_acc'].append(val_acc)
        
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = epoch
    
    def get_best(self):
        """Get best validation accuracy and epoch."""
        return {
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch
        }
    
    def get_all(self):
        """Get all tracked metrics."""
        return self.metrics
