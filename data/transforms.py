"""Image transforms for training and inference."""

import torchvision.transforms as transforms


def get_transforms(config):
    """
    Get training and validation/test transforms based on config.
    
    Args:
        config: Configuration dictionary with augmentation settings
    
    Returns:
        Dictionary with 'train' and 'val' transform pipelines
    """
    img_size = config['data']['image_size']
    mean = config['data']['augmentation']['normalize']['mean']
    std = config['data']['augmentation']['normalize']['std']
    
    train_transforms = []
    
    # Training augmentations
    if config['data']['augmentation']['train']['horizontal_flip']:
        train_transforms.append(transforms.RandomHorizontalFlip())
    
    if config['data']['augmentation']['train']['vertical_flip']:
        train_transforms.append(transforms.RandomVerticalFlip())
    
    if config['data']['augmentation']['train']['rotation_degrees']:
        degrees = config['data']['augmentation']['train']['rotation_degrees']
        train_transforms.append(transforms.RandomRotation(degrees))
    
    # Resize and crop
    train_transforms.extend([
        transforms.Resize(256),
        transforms.RandomCrop(img_size) if config['data']['augmentation']['train']['random_crop'] 
        else transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Validation/test transforms (no augmentation)
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    return {
        'train': transforms.Compose(train_transforms),
        'val': val_transforms
    }
