"""Dataset loading and dataloader creation."""

import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def get_dataloaders(config):
    """
    Create dataloaders for train, validation, and test sets.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Tuple of (dataloaders_dict, class_names)
        - dataloaders_dict: Dictionary with 'train', 'val', 'test' dataloaders
        - class_names: List of class names in order
    """
    from data.transforms import get_transforms
    
    dataset_path = config['paths']['dataset']
    batch_size = config['data']['batch_size']
    num_workers = config['data']['num_workers']
    
    transforms = get_transforms(config)
    subsets = ['train', 'val', 'test']
    dataloaders = {}
    class_names = None
    
    for subset in subsets:
        subset_path = os.path.join(dataset_path, subset)
        
        # Select transform
        transform = transforms['train'] if subset == 'train' else transforms['val']
        
        # Create dataset
        dataset = ImageFolder(subset_path, transform=transform)
        
        # Get class names from training set
        if subset == 'train':
            class_names = [dataset.classes[i] for i in range(len(dataset.classes))]
        
        # Create dataloader
        dataloaders[subset] = DataLoader(
            dataset,
            batch_size=batch_size if subset == 'train' else 1,
            shuffle=(subset == 'train'),
            num_workers=num_workers,
            drop_last=False
        )
    
    return dataloaders, class_names


def get_single_dataloader(path, config, shuffle=False):
    """
    Create a dataloader for a single directory of images.
    
    Args:
        path: Path to image directory
        config: Configuration dictionary
        shuffle: Whether to shuffle the data
    
    Returns:
        Tuple of (dataloader, class_names)
    """
    from data.transforms import get_transforms
    
    transforms = get_transforms(config)
    
    dataset = ImageFolder(path, transform=transforms['val'])
    class_names = [dataset.classes[i] for i in range(len(dataset.classes))]
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=config['data']['num_workers'],
        drop_last=False
    )
    
    return dataloader, class_names
