"""Model evaluation and error analysis."""

import torch
import yaml
import numpy as np
from pathlib import Path
from PIL import Image
import shutil

from data.datasets import get_dataloaders, get_single_dataloader
from data.genetic_distance import GeneticDistanceMatrix
from data.transforms import get_transforms
from models.model_loader import load_model
from utils.visualization import (
    plot_error_comparison, 
    plot_histogram, 
    normalize_image
)
from utils.metrics import compute_distance_statistics, save_metrics


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_model(model, dataloader, device):
    """
    Evaluate model on a dataset.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader
        device: Device to run on
    
    Returns:
        Tuple of (predictions, labels, outputs)
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_outputs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_outputs.extend(outputs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_outputs)


def analyze_errors(model, dataloader, class_names, genetic_names, 
                   dist_mat, config, device):
    """
    Analyze classification errors with genetic distance.
    
    Args:
        model: PyTorch model
        dataloader: Test dataloader
        class_names: List of class names
        genetic_names: List of genetic breed names
        dist_mat: Genetic distance matrix
        config: Configuration dictionary
        device: Device to run on
    
    Returns:
        Tuple of (accuracy, error_distances)
    """
    model.eval()
    
    # Setup output directory
    experiment_name = config['experiment']['name']
    output_dir = Path(config['paths']['output_dir']) / f'{experiment_name}_errors'
    
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    
    # Load exemplar images for predictions
    exemplar_path = Path(config['paths']['exemplars'])
    transforms = get_transforms(config)['val']
    
    running_corrects = 0
    total_samples = 0
    error_distances = []
    error_count = 0
    
    print("Analyzing errors...")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels_np = labels.numpy()
            
            # Get predictions
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            preds_np = preds.cpu().numpy()
            
            # Process each sample
            for i in range(len(labels_np)):
                total_samples += 1
                true_idx = labels_np[i]
                pred_idx = preds_np[i]
                
                if true_idx == pred_idx:
                    running_corrects += 1
                else:
                    # This is an error - analyze it
                    error_count += 1
                    genetic_dist = dist_mat[true_idx, pred_idx]
                    error_distances.append(genetic_dist)
                    
                    # Save error visualization
                    if config['evaluation']['save_error_visualizations']:
                        # Get input image
                        input_img = images[i].cpu().numpy().transpose(1, 2, 0)
                        input_img = normalize_image(input_img)
                        
                        # Get exemplar for predicted class
                        pred_class = class_names[pred_idx]
                        exemplar_dir = exemplar_path / pred_class
                        exemplar_file = list(exemplar_dir.glob('*'))[0]
                        
                        pred_img = Image.open(exemplar_file)
                        pred_img = transforms(pred_img).numpy().transpose(1, 2, 0)
                        pred_img = normalize_image(pred_img)
                        
                        # Save comparison
                        true_label = genetic_names[true_idx]
                        pred_label = genetic_names[pred_idx]
                        
                        filename = (f"{genetic_dist:.4f}_{error_count}_"
                                  f"{true_label}_{pred_label}.png")
                        save_path = output_dir / filename
                        
                        plot_error_comparison(
                            input_img, true_label,
                            pred_img, pred_label,
                            genetic_dist, save_path,
                            dpi=config['evaluation']['error_vis_dpi']
                        )
            
            # Progress
            if (batch_idx + 1) % 100 == 0:
                print(f"Processed {batch_idx + 1}/{len(dataloader)} batches")
    
    accuracy = running_corrects / total_samples
    error_distances = np.array(error_distances)
    
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
    print(f"Total errors: {len(error_distances)}")
    
    return accuracy, error_distances


def main(config_path='config/train_config.yaml', model_path=None):
    """
    Main evaluation function.
    
    Args:
        config_path: Path to configuration file
        model_path: Path to saved model (if None, uses config to construct path)
    """
    # Load configuration
    print("Loading configuration...")
    config = load_config(config_path)
    
    # Setup device
    device_config = config['training']['device']
    if device_config == 'auto':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_config)
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading datasets...")
    dataloaders, class_names = get_dataloaders(config)
    num_classes = len(class_names)
    print(f"Test samples: {len(dataloaders['test'].dataset)}")
    
    # Load genetic distance matrix
    print("\nLoading genetic distance matrix...")
    genetic_data_path = config['paths']['genetic_data']
    gdm = GeneticDistanceMatrix(genetic_data_path)
    genetic_names, dist_mat = gdm.get_dist_mat(class_names)
    
    # Load model
    print("\nLoading model...")
    if model_path is None:
        experiment_name = config['experiment']['name']
        model_dir = Path(config['paths']['model_save_dir'])
        model_path = model_dir / f'{experiment_name}_model.pth'
    
    model = load_model(model_path, config, num_classes, device)
    
    # Evaluate
    print("\nEvaluating model on test set...")
    accuracy, error_distances = analyze_errors(
        model, dataloaders['test'], class_names, genetic_names,
        dist_mat, config, device
    )
    
    # Compute statistics
    print("\nComputing error statistics...")
    stats = compute_distance_statistics(error_distances)
    stats['test_accuracy'] = accuracy
    
    # Save statistics
    output_dir = Path(config['paths']['output_dir'])
    experiment_name = config['experiment']['name']
    stats_path = output_dir / f'{experiment_name}_test_stats.txt'
    save_metrics(stats, stats_path)
    
    # Generate histogram
    if config['evaluation']['generate_histograms'] and len(error_distances) > 0:
        print("\nGenerating error distance histogram...")
        hist_min = dist_mat.min()
        hist_max = dist_mat.max()
        hist_path = output_dir / f'{experiment_name}_error_histogram.png'
        
        plot_histogram(
            error_distances,
            bins=20,
            title='Error Genetic Distance Distribution',
            xlabel='Genetic Distance',
            ylabel='Density',
            save_path=hist_path,
            range_limits=(hist_min, hist_max)
        )
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        model_path = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        config_path = 'config/train_config.yaml'
        model_path = None
    
    main(config_path, model_path)
