"""Utilities for loading and saving models."""

import torch
from pathlib import Path
import json
from models.resnet_classifier import get_model


def save_model(model, config, metrics, save_dir):
    """
    Save model weights and metadata.
    
    Args:
        model: PyTorch model
        config: Configuration dictionary
        metrics: Dictionary of metrics (accuracy, loss, etc.)
        save_dir: Directory to save model
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    experiment_name = config['experiment']['name']
    
    # Save model weights
    model_path = save_dir / f'{experiment_name}_model.pth'
    torch.save(model.state_dict(), model_path)
    
    # Save metadata
    metadata = {
        'experiment_name': experiment_name,
        'architecture': config['model']['architecture'],
        'num_classes': metrics.get('num_classes'),
        'best_val_accuracy': metrics.get('best_val_acc'),
        'final_train_accuracy': metrics.get('final_train_acc'),
        'test_accuracy': metrics.get('test_acc'),
        'loss_type': config['training']['loss_type']
    }
    
    metadata_path = save_dir / f'{experiment_name}_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved to {model_path}")
    print(f"Metadata saved to {metadata_path}")


def load_model(model_path, config, num_classes, device='cpu'):
    """
    Load a saved model.
    
    Args:
        model_path: Path to saved model weights (.pth file)
        config: Configuration dictionary
        num_classes: Number of classes
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    model = get_model(config, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {model_path}")
    return model


def get_model_info(model):
    """
    Get information about model architecture.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with model info
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_type': type(model).__name__
    }
