"""Extract and analyze embeddings from trained models."""

import torch
import yaml
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

from data.datasets import get_single_dataloader
from models.model_loader import load_model
from models.resnet_classifier import get_embedder


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def extract_embeddings(model, dataloader, device):
    """
    Extract embeddings for all images in dataloader.
    
    Args:
        model: Model with embedding extraction capability
        dataloader: DataLoader
        device: Device to run on
    
    Returns:
        Dictionary mapping class names to dictionaries of {filepath: embedding}
    """
    embedder = get_embedder(model).to(device)
    embedder.eval()
    
    embeddings = {}
    
    print("Extracting embeddings...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(dataloader)):
            images = images.to(device)
            label = labels[0].item()
            
            # Get class name
            class_name = dataloader.dataset.classes[label]
            
            # Get file path
            filepath = dataloader.dataset.samples[i][0]
            
            # Extract embedding
            embedding = embedder(images).cpu().squeeze().numpy()
            
            # Store
            if class_name not in embeddings:
                embeddings[class_name] = {}
            embeddings[class_name][filepath] = embedding
    
    return embeddings


def find_nearest_neighbors(query_embedding, embeddings_dict, k=5, 
                          exclude_self=True, query_path=None):
    """
    Find k nearest neighbors to a query embedding.
    
    Args:
        query_embedding: Query embedding vector
        embeddings_dict: Dictionary of {filepath: embedding}
        k: Number of neighbors to return
        exclude_self: Whether to exclude exact matches
        query_path: Path of query image (for self-exclusion)
    
    Returns:
        List of (filepath, distance) tuples
    """
    distances = []
    
    for filepath, embedding in embeddings_dict.items():
        # Skip if this is the query itself
        if exclude_self and query_path and filepath == query_path:
            continue
        
        # Check if embeddings are identical (same image)
        if exclude_self and np.allclose(query_embedding, embedding):
            continue
        
        # Compute L2 distance
        dist = np.linalg.norm(query_embedding - embedding)
        distances.append((filepath, dist))
    
    # Sort by distance and return top k
    distances.sort(key=lambda x: x[1])
    return distances[:k]


def compute_class_centroids(embeddings):
    """
    Compute centroid embedding for each class.
    
    Args:
        embeddings: Dictionary mapping classes to {filepath: embedding}
    
    Returns:
        Dictionary mapping class names to centroid embeddings
    """
    centroids = {}
    
    for class_name, class_embeddings in embeddings.items():
        embedding_array = np.array(list(class_embeddings.values()))
        centroid = np.mean(embedding_array, axis=0)
        centroids[class_name] = centroid
    
    return centroids


def compute_intra_class_variance(embeddings):
    """
    Compute variance within each class.
    
    Args:
        embeddings: Dictionary mapping classes to {filepath: embedding}
    
    Returns:
        Dictionary mapping class names to variance values
    """
    variances = {}
    
    for class_name, class_embeddings in embeddings.items():
        embedding_array = np.array(list(class_embeddings.values()))
        variance = np.var(embedding_array)
        variances[class_name] = variance
    
    return variances


def save_embeddings(embeddings, save_path):
    """Save embeddings to pickle file."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(embeddings, f)
    
    print(f"Embeddings saved to {save_path}")


def load_embeddings(load_path):
    """Load embeddings from pickle file."""
    with open(load_path, 'rb') as f:
        embeddings = pickle.load(f)
    
    print(f"Embeddings loaded from {load_path}")
    return embeddings


def main(config_path='config/train_config.yaml', 
         model_path=None, 
         data_path=None,
         force_recompute=False):
    """
    Main function for embedding extraction and analysis.
    
    Args:
        config_path: Path to configuration file
        model_path: Path to saved model
        data_path: Path to data directory (defaults to full dataset)
        force_recompute: Force recomputation even if embeddings exist
    """
    # Load configuration
    config = load_config(config_path)
    experiment_name = config['experiment']['name']
    
    # Setup paths
    output_dir = Path(config['paths']['output_dir'])
    embeddings_path = output_dir / f'{experiment_name}_embeddings.pickle'
    
    # Check if embeddings already exist
    if embeddings_path.exists() and not force_recompute:
        print(f"Embeddings already exist at {embeddings_path}")
        print("Loading existing embeddings...")
        embeddings = load_embeddings(embeddings_path)
        return embeddings
    
    # Setup device
    device_config = config['training']['device']
    if device_config == 'auto':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_config)
    print(f"Using device: {device}")
    
    # Load data
    if data_path is None:
        # Use full dataset by default
        data_path = Path(config['paths']['dataset']).parent / 'images_all'
        if not data_path.exists():
            # Fallback to train set
            data_path = Path(config['paths']['dataset']) / 'train'
    
    print(f"\nLoading data from {data_path}...")
    dataloader, class_names = get_single_dataloader(data_path, config, shuffle=False)
    print(f"Total samples: {len(dataloader.dataset)}")
    
    # Load model
    if model_path is None:
        model_dir = Path(config['paths']['model_save_dir'])
        model_path = model_dir / f'{experiment_name}_model.pth'
    
    print(f"\nLoading model from {model_path}...")
    num_classes = len(class_names)
    model = load_model(model_path, config, num_classes, device)
    
    # Extract embeddings
    embeddings = extract_embeddings(model, dataloader, device)
    
    # Compute statistics
    print("\nComputing embedding statistics...")
    centroids = compute_class_centroids(embeddings)
    variances = compute_intra_class_variance(embeddings)
    
    print(f"Number of classes: {len(embeddings)}")
    print(f"Average variance: {np.mean(list(variances.values())):.6f}")
    
    # Save embeddings
    save_embeddings(embeddings, embeddings_path)
    
    # Save centroids separately
    centroids_path = output_dir / f'{experiment_name}_centroids.pickle'
    with open(centroids_path, 'wb') as f:
        pickle.dump(centroids, f)
    print(f"Centroids saved to {centroids_path}")
    
    return embeddings


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        model_path = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        config_path = 'config/train_config.yaml'
        model_path = None
    
    main(config_path, model_path)
