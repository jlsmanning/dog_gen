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

    # Validation/test transforms (no augmentation)
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Training augmentations (optional - may not exist in inference config)
    train_config = config['data']['augmentation'].get('train')
    if train_config:
        train_transforms = []

        if train_config.get('horizontal_flip'):
            train_transforms.append(transforms.RandomHorizontalFlip())

        if train_config.get('vertical_flip'):
            train_transforms.append(transforms.RandomVerticalFlip())

        if train_config.get('rotation_degrees'):
            degrees = train_config['rotation_degrees']
            train_transforms.append(transforms.RandomRotation(degrees))

        # Resize and crop
        train_transforms.extend([
            transforms.Resize(256),
            transforms.RandomCrop(img_size) if train_config.get('random_crop')
            else transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_transform = transforms.Compose(train_transforms)
    else:
        # Use val transforms for inference-only configs
        train_transform = val_transforms

    return {
        'train': train_transform,
        'val': val_transforms
    }
