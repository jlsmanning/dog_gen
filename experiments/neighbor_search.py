"""Nearest neighbor search and visualization."""

import torch
import yaml
import numpy as np
from pathlib import Path
from PIL import Image

from data.datasets import get_single_dataloader
from data.genetic_distance import GeneticDistanceMatrix
from data.transforms import get_transforms
from models.model_loader import load_model
from models.resnet_classifier import get_embedder
from utils.visualization import plot_neighbors, normalize_image
from experiments.embedding_analysis import load_embeddings, extract_embeddings


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_top_k_predictions(model, image, k=5):
    """
    Get top-k predictions for an image.
    
    Args:
        model: PyTorch model
        image: Image tensor
        k: Number of top predictions
    
    Returns:
        List of (class_index, score) tuples
    """
    model.eval()
    
    with torch.no_grad():
        outputs = model(image)
        scores, indices = torch.topk(outputs[0], k)
    
    return list(zip(indices.cpu().numpy(), scores.cpu().numpy()))


def find_class_nearest_neighbor(query_embedding, class_embeddings, exclude_path=None):
    """
    Find nearest neighbor within a specific class.
    
    Args:
        query_embedding: Query embedding
        class_embeddings: Dictionary of {filepath: embedding} for one class
        exclude_path: Optional path to exclude (the query itself)
    
    Returns:
        Tuple of (filepath, distance) for nearest neighbor
    """
    min_dist = float('inf')
    nearest = None
    
    for filepath, embedding in class_embeddings.items():
        # Skip if this is the query itself
        if exclude_path and filepath == exclude_path:
            continue
        
        # Skip if embeddings are identical
        if np.allclose(query_embedding, embedding):
            continue
        
        dist = np.linalg.norm(query_embedding - embedding)
        if dist < min_dist:
            min_dist = dist
            nearest = filepath
    
    return nearest, min_dist


def visualize_neighbors(model, embedder, dataloader, embeddings, 
                       class_names, genetic_names, dist_mat,
                       config, device, num_queries=10, num_neighbors=5):
    """
    Visualize query images with their nearest neighbors.
    
    Args:
        model: Classification model
        embedder: Embedding extractor
        dataloader: DataLoader for queries
        embeddings: Pre-computed embeddings
        class_names: List of class names
        genetic_names: List of genetic breed names
        dist_mat: Genetic distance matrix
        config: Configuration dictionary
        device: Device to run on
        num_queries: Number of query images to process
        num_neighbors: Number of neighbors to show
    """
    output_dir = Path(config['paths']['output_dir']) / f"{config['experiment']['name']}_neighbors"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    transforms = get_transforms(config)['val']
    
    print(f"Generating neighbor visualizations...")
    
    count = 0
    for images, labels in dataloader:
        if count >= num_queries:
            break
        
        count += 1
        
        images = images.to(device)
        label = labels[0].item()
        class_name = class_names[label]
        genetic_name = genetic_names[label]
        
        # Get query image
        query_img = images[0].cpu().numpy().transpose(1, 2, 0)
        query_img = normalize_image(query_img)
        
        # Get query embedding
        with torch.no_grad():
            query_embedding = embedder(images).cpu().squeeze().numpy()
        
        # Get top-k predictions
        top_preds = get_top_k_predictions(model, images, num_neighbors)
        
        # Find nearest neighbor for each predicted class
        neighbor_imgs = []
        neighbor_labels = []
        genetic_distances = []
        
        for pred_idx, score in top_preds:
            pred_class = class_names[pred_idx]
            
            # Find nearest neighbor in predicted class
            nn_path, nn_dist = find_class_nearest_neighbor(
                query_embedding, 
                embeddings[pred_class]
            )
            
            # Load neighbor image
            nn_img = Image.open(nn_path)
            nn_img = transforms(nn_img).numpy().transpose(1, 2, 0)
            nn_img = normalize_image(nn_img)
            
            neighbor_imgs.append(nn_img)
            neighbor_labels.append(genetic_names[pred_idx])
            genetic_distances.append(dist_mat[label, pred_idx])
        
        # Save visualization
        save_path = output_dir / f'{count}.png'
        plot_neighbors(
            query_img, genetic_name,
            neighbor_imgs, neighbor_labels,
            genetic_distances, save_path,
            dpi=300
        )
        
        if count % 10 == 0:
            print(f"Processed {count}/{num_queries} queries")
    
    print(f"\nNeighbor visualizations saved to {output_dir}")


def main(config_path='config/train_config.yaml', 
         model_path=None,
         num_queries=102,
         num_neighbors=5):
    """
    Main function for neighbor search visualization.
    
    Args:
        config_path: Path to configuration file
        model_path: Path to saved model
        num_queries: Number of query images to process
        num_neighbors: Number of neighbors per query
    """
    # Load configuration
    config = load_config(config_path)
    experiment_name = config['experiment']['name']
    
    # Setup device
    device_config = config['training']['device']
    if device_config == 'auto':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_config)
    print(f"Using device: {device}")
    
    # Load embeddings or extract if needed
    output_dir = Path(config['paths']['output_dir'])
    embeddings_path = output_dir / f'{experiment_name}_embeddings.pickle'
    
    if embeddings_path.exists():
        print("Loading pre-computed embeddings...")
        embeddings = load_embeddings(embeddings_path)
    else:
        print("No pre-computed embeddings found. Run embedding_analysis.py first.")
        return
    
    # Load query data
    exemplar_path = Path(config['paths']['exemplars'])
    print(f"\nLoading query data from {exemplar_path}...")
    dataloader, class_names = get_single_dataloader(exemplar_path, config, shuffle=True)
    
    # Load genetic distance matrix
    genetic_data_path = config['paths']['genetic_data']
    gdm = GeneticDistanceMatrix(genetic_data_path)
    genetic_names, dist_mat = gdm.get_dist_mat(class_names)
    
    # Load model
    if model_path is None:
        model_dir = Path(config['paths']['model_save_dir'])
        model_path = model_dir / f'{experiment_name}_model.pth'
    
    print(f"\nLoading model from {model_path}...")
    num_classes = len(class_names)
    model = load_model(model_path, config, num_classes, device)
    embedder = get_embedder(model).to(device)
    
    # Visualize neighbors
    visualize_neighbors(
        model, embedder, dataloader, embeddings,
        class_names, genetic_names, dist_mat,
        config, device, num_queries, num_neighbors
    )
    
    print("\n" + "="*60)
    print("NEIGHBOR SEARCH COMPLETE")
    print("="*60)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        model_path = sys.argv[2] if len(sys.argv) > 2 else None
        num_queries = int(sys.argv[3]) if len(sys.argv) > 3 else 102
        num_neighbors = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    else:
        config_path = 'config/train_config.yaml'
        model_path = None
        num_queries = 102
        num_neighbors = 5
    
    main(config_path, model_path, num_queries, num_neighbors)
