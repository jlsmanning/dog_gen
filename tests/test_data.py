"""Tests for data loading and preprocessing."""

import pytest
import yaml
from pathlib import Path

from data.genetic_distance import GeneticDistanceMatrix


def test_genetic_distance_matrix():
    """Test genetic distance matrix loading."""
    # This is a placeholder - would need actual test data
    # In production, you'd use a small test dataset
    pass


def test_config_loading():
    """Test configuration file loading."""
    config_path = Path("config/train_config.yaml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    assert 'paths' in config
    assert 'data' in config
    assert 'model' in config
    assert 'training' in config


def test_transforms_config():
    """Test that transforms are properly configured."""
    config_path = Path("config/train_config.yaml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    assert config['data']['image_size'] == 224
    assert len(config['data']['augmentation']['normalize']['mean']) == 3
    assert len(config['data']['augmentation']['normalize']['std']) == 3
