"""Main training script."""

import torch
import yaml
from pathlib import Path
import sys

from data.datasets import get_dataloaders
from data.genetic_distance import GeneticDistanceMatrix
from models.resnet_classifier import get_model
from models.model_loader import save_model
from training.trainer import Trainer
from utils.visualization import plot_training_curves


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_device(config):
    """Setup computing device."""
    device_config = config['training']['device']
    
    if device_config == 'auto':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_config)
    
    print(f"Using device: {device}")
    return device


def main(config_path='config/train_config.yaml'):
    """
    Main training function.
    
    Args:
        config_path: Path to training configuration file
    """
    # Load configuration
    print("Loading configuration...")
    config = load_config(config_path)
    
    # Setup device
    device = setup_device(config)
    
    # Load data
    print("\nLoading datasets...")
    dataloaders, class_names = get_dataloaders(config)
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(dataloaders['train'].dataset)}")
    print(f"Validation samples: {len(dataloaders['val'].dataset)}")
    
    # Setup genetic distance matrix if using dist_loss
    soft_label_matrix = None
    if config['training']['loss_type'] == 'dist_loss':
        print("\nSetting up genetic distance matrix...")
        genetic_data_path = config['paths']['genetic_data']
        gdm = GeneticDistanceMatrix(genetic_data_path)
        
        _, dist_mat = gdm.get_dist_mat(class_names)
        threshold = config['training']['label_threshold']
        soft_label_matrix = gdm.dist_to_soft_labels(dist_mat, threshold)
        print(f"Soft label matrix shape: {soft_label_matrix.shape}")
    
    # Create model
    print("\nCreating model...")
    model = get_model(config, num_classes)
    print(f"Model architecture: {config['model']['architecture']}")
    print(f"Pretrained: {config['model']['pretrained']}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        dataloaders={'train': dataloaders['train'], 'val': dataloaders['val']},
        config=config,
        device=device,
        soft_label_matrix=soft_label_matrix
    )
    
    # Train
    trained_model = trainer.train()
    
    # Get metrics
    metrics = trainer.get_metrics()
    best_metrics = trainer.get_best_metrics()
    
    # Save model
    print("\nSaving model...")
    save_dir = Path(config['paths']['model_save_dir'])
    save_metrics = {
        'num_classes': num_classes,
        'best_val_acc': best_metrics['best_val_acc'],
        'final_train_acc': metrics['train_acc'][-1],
    }
    save_model(trained_model, config, save_metrics, save_dir)
    
    # Save training curves
    output_dir = Path(config['paths']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    experiment_name = config['experiment']['name']
    curves_path = output_dir / f'{experiment_name}_training_curves.png'
    
    losses = {'train': metrics['train_loss'], 'val': metrics['val_loss']}
    accuracies = {'train': metrics['train_acc'], 'val': metrics['val_acc']}
    
    plot_training_curves(losses, accuracies, curves_path)
    
    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETE")
    print("="*60)


if __name__ == '__main__':
    # Allow passing config path as command line argument
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = 'config/train_config.yaml'
    
    main(config_path)
