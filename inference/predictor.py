"""Inference predictor for single image classification."""

import torch
import numpy as np
from PIL import Image
from pathlib import Path

from data.transforms import get_transforms
from models.model_loader import load_model


class BreedPredictor:
    """Dog breed classifier predictor."""
    
    def __init__(self, model_path, config, class_names, genetic_names=None, device='cpu'):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to saved model weights
            config: Configuration dictionary
            class_names: List of class names
            genetic_names: Optional list of genetic breed names
            device: Device to run inference on
        """
        self.config = config
        self.class_names = class_names
        self.genetic_names = genetic_names if genetic_names else class_names
        self.device = device
        
        # Load model
        num_classes = len(class_names)
        self.model = load_model(model_path, config, num_classes, device)
        self.model.eval()
        
        # Get transforms
        transforms = get_transforms(config)
        self.transform = transforms['val']
        
        print(f"Predictor initialized with {num_classes} classes")
    
    def preprocess_image(self, image_input):
        """
        Preprocess image for inference.
        
        Args:
            image_input: Either PIL Image, numpy array, or file path
        
        Returns:
            Preprocessed tensor ready for model
        """
        # Convert to PIL Image if needed
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            image = image.convert('RGB')
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
        
        # Apply transforms
        tensor = self.transform(image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def predict(self, image_input, top_k=5):
        """
        Predict breed for an image.
        
        Args:
            image_input: Image (PIL Image, numpy array, or file path)
            top_k: Number of top predictions to return
        
        Returns:
            Dictionary with predictions
        """
        # Preprocess
        image_tensor = self.preprocess_image(image_input)
        image_tensor = image_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities[0], top_k)
        
        # Format results
        predictions = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            predictions.append({
                'class_name': self.class_names[idx],
                'breed_name': self.genetic_names[idx],
                'probability': float(prob),
                'confidence': float(prob * 100)
            })
        
        return {
            'top_prediction': predictions[0],
            'top_k_predictions': predictions
        }
    
    def predict_batch(self, image_list, top_k=5):
        """
        Predict breeds for a batch of images.
        
        Args:
            image_list: List of images
            top_k: Number of top predictions per image
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        for image in image_list:
            result = self.predict(image, top_k)
            results.append(result)
        return results


def load_predictor(model_path, config_path, class_names_path=None, device='auto'):
    """
    Convenience function to load a predictor.
    
    Args:
        model_path: Path to saved model
        config_path: Path to config file
        class_names_path: Optional path to class names file
        device: Device to use ('auto', 'cpu', or 'cuda')
    
    Returns:
        Initialized BreedPredictor
    """
    import yaml
    import pickle
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    if device == 'auto':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    # Load class names
    if class_names_path:
        with open(class_names_path, 'rb') as f:
            data = pickle.load(f)
            class_names = data['class_names']
            genetic_names = data.get('genetic_names', class_names)
    else:
        # Try to infer from dataset
        from data.datasets import get_dataloaders
        _, class_names = get_dataloaders(config)
        
        # Try to get genetic names
        try:
            from data.genetic_distance import GeneticDistanceMatrix
            gdm = GeneticDistanceMatrix(config['paths']['genetic_data'])
            genetic_names, _ = gdm.get_dist_mat(class_names)
        except Exception:
            genetic_names = class_names
    
    return BreedPredictor(model_path, config, class_names, genetic_names, device)
