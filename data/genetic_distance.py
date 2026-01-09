"""Genetic distance matrix computation and soft label generation."""

import csv
import os
import numpy as np
import pickle
from pathlib import Path

from data.breed_mapping import DATA2GEN


class GeneticDistanceMatrix:
    """Handles loading and processing of genetic distance data."""
    
    def __init__(self, genetic_data_path):
        """
        Initialize genetic distance matrix handler.
        
        Args:
            genetic_data_path: Path to directory containing Breeds.txt and distances_all.txt
        """
        self.genetic_data_path = Path(genetic_data_path)
        self.breeds_file = self.genetic_data_path / 'Breeds.txt'
        self.distances_file = self.genetic_data_path / 'distances_all.txt'
        self.cache_dir = self.genetic_data_path / 'cache'
        self.cache_dir.mkdir(exist_ok=True)
        
        # Cache files
        self.samples_by_class_cache = self.cache_dir / 'samples_by_class.pickle'
        self.dist_all_cache = self.cache_dir / 'dist_all.pickle'
        self.dist_means_cache = self.cache_dir / 'dist_means.pickle'
        self.dist_mat_cache = self.cache_dir / 'dist_mat.pickle'
    
    def get_samples_by_class(self):
        """
        Load breed assignments for each sample.
        
        Returns:
            Dictionary mapping breed names to lists of sample codes
        """
        if self.samples_by_class_cache.exists():
            with open(self.samples_by_class_cache, 'rb') as f:
                return pickle.load(f)
        
        # Build the initial dict
        with open(self.breeds_file) as f:
            tsv_file = csv.reader(f, delimiter='\t')
            samples_by_class = {}
            for line in tsv_file:
                breed = line[1]
                sample = line[0]
                if breed not in samples_by_class:
                    samples_by_class[breed] = [sample]
                else:
                    samples_by_class[breed].append(sample)
        
        # Filter to only samples with genetic data
        sample_names, _ = self.get_dist_all()
        for breed in list(samples_by_class.keys()):
            valid_samples = [s for s in samples_by_class[breed] if s in sample_names]
            if valid_samples:
                samples_by_class[breed] = valid_samples
            else:
                del samples_by_class[breed]
        
        with open(self.samples_by_class_cache, 'wb') as f:
            pickle.dump(samples_by_class, f)
        
        return samples_by_class
    
    def get_dist_all(self):
        """
        Load the full distance matrix for all individual samples.
        
        Returns:
            Tuple of (sample_names, distance_matrix)
        """
        if self.dist_all_cache.exists():
            with open(self.dist_all_cache, 'rb') as f:
                data = pickle.load(f)
                return data[0], data[1]
        
        sample_names = []
        dist_matrix = []
        
        with open(self.distances_file) as f:
            tsv_file = csv.reader(f, delimiter='\t')
            for line in tsv_file:
                sample_names.append(line[0])
                row = [float(x) for x in line[1].split(' ')[:-1]]
                dist_matrix.append(row)
        
        dist_matrix = np.array(dist_matrix)
        
        with open(self.dist_all_cache, 'wb') as f:
            pickle.dump([sample_names, dist_matrix], f)
        
        return sample_names, dist_matrix
    
    def get_dist_means(self):
        """
        Compute mean genetic distances between breed pairs.
        
        Returns:
            Tuple of (breed_names, mean_distance_matrix)
        """
        if self.dist_means_cache.exists():
            with open(self.dist_means_cache, 'rb') as f:
                data = pickle.load(f)
                return data[0], data[1]
        
        samples_by_class = self.get_samples_by_class()
        sample_names, dist_all = self.get_dist_all()
        
        breeds = list(samples_by_class.keys())
        n = len(breeds)
        mean_dist_matrix = np.zeros((n, n))
        
        for i in range(n):
            breed1 = breeds[i]
            samples1 = samples_by_class[breed1]
            
            for j in range(n):
                breed2 = breeds[j]
                samples2 = samples_by_class[breed2]
                
                total_dist = 0
                count = 0
                
                if i == j:
                    # Within-breed distances
                    for x in range(len(samples1)):
                        idx_a = sample_names.index(samples1[x])
                        for y in range(x + 1, len(samples1)):
                            idx_b = sample_names.index(samples1[y])
                            total_dist += dist_all[idx_a, idx_b]
                            count += 1
                else:
                    # Between-breed distances
                    for s1 in samples1:
                        idx_a = sample_names.index(s1)
                        for s2 in samples2:
                            idx_b = sample_names.index(s2)
                            total_dist += dist_all[idx_a, idx_b]
                            count += 1
                
                mean_dist_matrix[i, j] = total_dist / count if count > 0 else 0
        
        with open(self.dist_means_cache, 'wb') as f:
            pickle.dump([breeds, mean_dist_matrix], f)
        
        return breeds, mean_dist_matrix
    
    def get_dist_mat(self, class_names):
        """
        Get distance matrix aligned with ImageNet class ordering.
        
        Args:
            class_names: List of ImageNet class names in dataset order
        
        Returns:
            Tuple of (genetic_breed_names, distance_matrix)
        """
        cache_file = self.cache_dir / f'dist_mat_{len(class_names)}.pickle'
        
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                return data[0], data[1]
        
        genetic_breeds, breed_dist_matrix = self.get_dist_means()
        
        n = len(class_names)
        dist_mat = np.zeros((n, n))
        genetic_names = []
        
        for i in range(n):
            imagenet_name = class_names[i]
            genetic_name = DATA2GEN[imagenet_name]
            genetic_names.append(genetic_name)
            idx_i = genetic_breeds.index(genetic_name)
            
            for j in range(n):
                imagenet_name_j = class_names[j]
                genetic_name_j = DATA2GEN[imagenet_name_j]
                idx_j = genetic_breeds.index(genetic_name_j)
                
                dist_mat[i, j] = breed_dist_matrix[idx_i, idx_j]
        
        with open(cache_file, 'wb') as f:
            pickle.dump([genetic_names, dist_mat], f)
        
        return genetic_names, dist_mat
    
    def dist_to_soft_labels(self, dist_mat, threshold):
        """
        Convert distance matrix to soft label matrix for training.
        
        Args:
            dist_mat: Genetic distance matrix
            threshold: Distance threshold for soft labels
        
        Returns:
            Soft label matrix (probabilities sum to 1 for each row)
        """
        soft_labels = threshold - dist_mat
        soft_labels[soft_labels < 0] = 0
        
        # Normalize each row to sum to 1
        row_sums = soft_labels.sum(axis=1, keepdims=True)
        soft_labels = soft_labels / row_sums
        
        return soft_labels


def labels_to_soft_labels(labels, soft_label_matrix):
    """
    Convert batch of hard labels to soft labels.
    
    Args:
        labels: Tensor of shape (batch_size,) with class indices
        soft_label_matrix: Soft label matrix from dist_to_soft_labels
    
    Returns:
        Numpy array of shape (batch_size, num_classes) with soft labels
    """
    batch_size = labels.shape[0]
    soft_labels = np.zeros((batch_size, soft_label_matrix.shape[1]))
    
    for i in range(batch_size):
        class_idx = labels[i].item()
        soft_labels[i] = soft_label_matrix[class_idx]
    
    return soft_labels
