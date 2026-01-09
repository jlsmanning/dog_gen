"""Visualization utilities for training and evaluation."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def normalize_image(img):
    """
    Normalize image for visualization.
    
    Args:
        img: Image array
    
    Returns:
        Normalized image in range [0, 1]
    """
    abs_max = max(abs(np.min(img)), abs(np.max(img)))
    if abs_max > 0:
        img = img / abs_max
    img = img * 0.5 + 0.5
    return img


def plot_training_curves(losses, accuracies, save_path=None):
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        losses: Dictionary with 'train' and 'val' loss lists
        accuracies: Dictionary with 'train' and 'val' accuracy lists
        save_path: Optional path to save figure
    """
    epochs = range(1, len(losses['train']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(epochs, losses['train'], 'b-', label='Train')
    ax1.plot(epochs, losses['val'], 'r-', label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, accuracies['train'], 'b-', label='Train')
    ax2.plot(epochs, accuracies['val'], 'r-', label='Validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_error_comparison(input_img, input_label, pred_img, pred_label, 
                          genetic_distance, save_path, dpi=300):
    """
    Create side-by-side comparison of input and predicted breed.
    
    Args:
        input_img: Input image (numpy array)
        input_label: True label name
        pred_img: Representative image of predicted class
        pred_label: Predicted label name
        genetic_distance: Genetic distance between breeds
        save_path: Path to save figure
        dpi: Image resolution
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    
    # Input image
    ax1.imshow(input_img, cmap='viridis')
    ax1.axis('off')
    ax1.set_title(f"Input: {input_label}", fontsize=10)
    
    # Predicted image
    ax2.imshow(pred_img, cmap='viridis')
    ax2.axis('off')
    ax2.set_title(f"Predicted: {pred_label}\nDistance: {genetic_distance:.4f}", 
                  fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_histogram(data, bins=20, title='Distribution', xlabel='Value', 
                   ylabel='Frequency', save_path=None, range_limits=None):
    """
    Plot histogram of data.
    
    Args:
        data: Data to plot
        bins: Number of histogram bins
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Optional path to save figure
        range_limits: Optional tuple of (min, max) for histogram range
    """
    plt.figure(figsize=(8, 6))
    
    if range_limits:
        plt.hist(data, bins=bins, range=range_limits, edgecolor='black', 
                 density=True, alpha=0.7)
    else:
        plt.hist(data, bins=bins, edgecolor='black', density=True, alpha=0.7)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Histogram saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix (numpy array)
        class_names: List of class names
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Show all ticks
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=6)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_neighbors(query_img, query_label, neighbor_imgs, neighbor_labels, 
                   genetic_distances, save_path, dpi=300):
    """
    Plot query image with its nearest neighbors.
    
    Args:
        query_img: Query image
        query_label: Query label name
        neighbor_imgs: List of neighbor images
        neighbor_labels: List of neighbor label names
        genetic_distances: List of genetic distances to neighbors
        save_path: Path to save figure
        dpi: Image resolution
    """
    n_neighbors = len(neighbor_imgs)
    fig = plt.figure(figsize=(3 * (n_neighbors + 1), 3))
    
    # Query image
    ax = fig.add_subplot(1, n_neighbors + 1, 1)
    ax.imshow(query_img, cmap='viridis')
    ax.axis('off')
    ax.set_title(f"Query:\n{query_label}", fontsize=8)
    
    # Neighbor images
    for i, (img, label, dist) in enumerate(zip(neighbor_imgs, neighbor_labels, 
                                                 genetic_distances)):
        ax = fig.add_subplot(1, n_neighbors + 1, i + 2)
        ax.imshow(img, cmap='viridis')
        ax.axis('off')
        prefix = "Prediction:" if i == 0 else f"Neighbor {i}:"
        ax.set_title(f"{prefix}\n{label}\nDist: {dist:.4f}", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def create_image_grid(images, labels, grid_size=(5, 5), save_path=None):
    """
    Create a grid of images.
    
    Args:
        images: List of images
        labels: List of labels (optional)
        grid_size: Tuple of (rows, cols)
        save_path: Optional path to save figure
    """
    rows, cols = grid_size
    fig = plt.figure(figsize=(cols * 2, rows * 2))
    
    for i in range(min(len(images), rows * cols)):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(images[i], cmap='viridis')
        ax.axis('off')
        if labels and i < len(labels):
            ax.set_title(labels[i], fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Image grid saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
