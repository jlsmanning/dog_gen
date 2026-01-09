"""Statistical comparison of different models."""

import pickle
import numpy as np
from pathlib import Path
from scipy import stats
import yaml


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_error_distances(experiment_name, output_dir):
    """
    Load error distances for an experiment.
    
    Args:
        experiment_name: Name of experiment
        output_dir: Output directory path
    
    Returns:
        Array of error distances
    """
    distances_path = Path(output_dir) / f'{experiment_name}_dists.pickle'
    
    if not distances_path.exists():
        raise FileNotFoundError(f"Error distances not found: {distances_path}")
    
    with open(distances_path, 'rb') as f:
        distances = pickle.load(f)
    
    return distances


def compare_models(distances1, distances2, name1, name2):
    """
    Perform t-test to compare error distances between two models.
    
    Args:
        distances1: Error distances from model 1
        distances2: Error distances from model 2
        name1: Name of model 1
        name2: Name of model 2
    
    Returns:
        Dictionary with comparison results
    """
    # Perform independent samples t-test
    statistic, pvalue = stats.ttest_ind(distances1, distances2)
    
    # Compute effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(distances1)**2 + np.std(distances2)**2) / 2)
    cohens_d = (np.mean(distances1) - np.mean(distances2)) / pooled_std
    
    results = {
        'comparison': f"{name1} vs {name2}",
        'model1_mean': np.mean(distances1),
        'model1_std': np.std(distances1),
        'model1_n': len(distances1),
        'model2_mean': np.mean(distances2),
        'model2_std': np.std(distances2),
        'model2_n': len(distances2),
        't_statistic': statistic,
        'p_value': pvalue,
        'significant_at_0.05': pvalue < 0.05,
        'significant_at_0.01': pvalue < 0.01,
        'cohens_d': cohens_d
    }
    
    return results


def print_comparison_results(results):
    """Print comparison results in readable format."""
    print(f"\n{'='*60}")
    print(f"Comparison: {results['comparison']}")
    print(f"{'='*60}")
    print(f"Model 1: mean={results['model1_mean']:.4f}, std={results['model1_std']:.4f}, n={results['model1_n']}")
    print(f"Model 2: mean={results['model2_mean']:.4f}, std={results['model2_std']:.4f}, n={results['model2_n']}")
    print(f"\nT-statistic: {results['t_statistic']:.4f}")
    print(f"P-value: {results['p_value']:.6f}")
    print(f"Significant at α=0.05: {results['significant_at_0.05']}")
    print(f"Significant at α=0.01: {results['significant_at_0.01']}")
    print(f"Cohen's d (effect size): {results['cohens_d']:.4f}")
    
    # Interpret effect size
    abs_d = abs(results['cohens_d'])
    if abs_d < 0.2:
        effect = "negligible"
    elif abs_d < 0.5:
        effect = "small"
    elif abs_d < 0.8:
        effect = "medium"
    else:
        effect = "large"
    print(f"Effect size interpretation: {effect}")


def save_comparison_results(all_results, save_path):
    """Save comparison results to file."""
    with open(save_path, 'w') as f:
        for results in all_results:
            f.write(f"\n{results['comparison']}\n")
            f.write(f"Model 1: {results['model1_mean']:.4f} ± {results['model1_std']:.4f} (n={results['model1_n']})\n")
            f.write(f"Model 2: {results['model2_mean']:.4f} ± {results['model2_std']:.4f} (n={results['model2_n']})\n")
            f.write(f"T-statistic: {results['t_statistic']:.4f}\n")
            f.write(f"P-value: {results['p_value']:.6f}\n")
            f.write(f"Cohen's d: {results['cohens_d']:.4f}\n")
            f.write("-" * 60 + "\n")
    
    print(f"\nResults saved to {save_path}")


def main():
    """
    Main function for statistical comparison of models.
    
    Compares:
    - ResNet18 score_loss vs ResNet50 score_loss
    - ResNet18 dist_loss vs ResNet50 dist_loss
    - ResNet18 score_loss vs ResNet18 dist_loss
    - ResNet50 score_loss vs ResNet50 dist_loss
    """
    # Load config to get output directory
    config = load_config('config/train_config.yaml')
    output_dir = Path(config['paths']['output_dir'])
    
    # Define experiment names
    experiments = {
        'resnet18_score': 'resnet18_score_loss',
        'resnet18_dist': 'resnet18_dist_loss',
        'resnet50_score': 'resnet50_score_loss',
        'resnet50_dist': 'resnet50_dist_loss'
    }
    
    # Load distances
    print("Loading error distances...")
    distances = {}
    for key, exp_name in experiments.items():
        try:
            distances[key] = load_error_distances(exp_name, output_dir)
            print(f"  {exp_name}: {len(distances[key])} errors")
        except FileNotFoundError as e:
            print(f"  Warning: {e}")
            distances[key] = None
    
    # Perform comparisons
    comparisons = []
    all_results = []
    
    # Architecture comparison (same loss)
    if distances['resnet18_score'] is not None and distances['resnet50_score'] is not None:
        comparisons.append(('resnet18_score', 'resnet50_score', 
                          'ResNet18 (score_loss)', 'ResNet50 (score_loss)'))
    
    if distances['resnet18_dist'] is not None and distances['resnet50_dist'] is not None:
        comparisons.append(('resnet18_dist', 'resnet50_dist',
                          'ResNet18 (dist_loss)', 'ResNet50 (dist_loss)'))
    
    # Loss comparison (same architecture)
    if distances['resnet18_score'] is not None and distances['resnet18_dist'] is not None:
        comparisons.append(('resnet18_score', 'resnet18_dist',
                          'ResNet18 (score_loss)', 'ResNet18 (dist_loss)'))
    
    if distances['resnet50_score'] is not None and distances['resnet50_dist'] is not None:
        comparisons.append(('resnet50_score', 'resnet50_dist',
                          'ResNet50 (score_loss)', 'ResNet50 (dist_loss)'))
    
    # Run comparisons
    for key1, key2, name1, name2 in comparisons:
        results = compare_models(distances[key1], distances[key2], name1, name2)
        print_comparison_results(results)
        all_results.append(results)
    
    # Save results
    if all_results:
        save_path = output_dir / 'statistical_comparisons.txt'
        save_comparison_results(all_results, save_path)
    
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
