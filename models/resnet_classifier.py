"""ResNet-based dog breed classifier model."""

import torch
import torch.nn as nn
import torchvision.models as models


def get_model(config, num_classes):
    """
    Create a ResNet classifier model.
    
    Args:
        config: Configuration dictionary
        num_classes: Number of output classes
    
    Returns:
        PyTorch model
    """
    architecture = config['model']['architecture']
    pretrained = config['model']['pretrained']
    
    if architecture == 'resnet18':
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            model = models.resnet18(weights=weights)
        else:
            model = models.resnet18(weights=None)
    
    elif architecture == 'resnet50':
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            model = models.resnet50(weights=weights)
        else:
            model = models.resnet50(weights=None)
    
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    
    # Replace final fully connected layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model


def get_embedder(model):
    """
    Create an embedding extractor from a trained classifier.
    Removes the final classification layer.
    
    Args:
        model: Trained classifier model
    
    Returns:
        Embedding model (outputs from penultimate layer)
    """
    # Remove the final FC layer
    modules = list(model.children())[:-1]
    embedder = nn.Sequential(*modules)
    
    # Freeze parameters
    for param in embedder.parameters():
        param.requires_grad = False
    
    embedder.eval()
    return embedder


class BreedClassifier(nn.Module):
    """
    Wrapper class for breed classification with optional embedding extraction.
    """
    
    def __init__(self, config, num_classes):
        super().__init__()
        self.model = get_model(config, num_classes)
        self.num_classes = num_classes
    
    def forward(self, x):
        return self.model(x)
    
    def get_embeddings(self, x):
        """Extract embeddings (before final FC layer)."""
        modules = list(self.model.children())[:-1]
        embedder = nn.Sequential(*modules)
        with torch.no_grad():
            embeddings = embedder(x)
            # Flatten spatial dimensions, preserving batch
            if len(embeddings.shape) > 2:
                embeddings = embeddings.flatten(1)
        return embeddings
