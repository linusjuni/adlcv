"""
Main script to run Vision Transformer hyperparameter experiments.

This script:
1. Loads experiment configurations from configs.py
2. Runs each experiment sequentially
3. Saves results (config, metrics, model) for each experiment
4. Generates a summary CSV of all experiments
"""

import os
import sys
import json
import time
from datetime import datetime
import pandas as pd

# Add parent directory to path to import adlcv_ex_2 package
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from adlcv_ex_2.imageclassification import main as train_main, set_seed
from adlcv_ex_2.experiments.configs import get_experiment_configs, get_quick_test_configs


def run_single_experiment(config, experiment_id, results_dir='results'):
    """
    Run a single experiment with the given configuration.

    Args:
        config (dict): Configuration dictionary containing all parameters
        experiment_id (int): Unique identifier for this experiment
        results_dir (str): Base directory for storing results

    Returns:
        dict: Results including metrics and metadata
    """
    # Create experiment directory
    exp_dir = os.path.join(results_dir, f'experiment_{experiment_id:03d}')
    os.makedirs(exp_dir, exist_ok=True)

    # Extract experiment name and description
    exp_name = config.get('name', f'experiment_{experiment_id}')
    exp_description = config.get('description', 'No description')

    print(f"\n{'='*80}")
    print(f"Running Experiment {experiment_id}: {exp_name}")
    print(f"Description: {exp_description}")
    print(f"{'='*80}\n")

    # Save configuration
    config_path = os.path.join(exp_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Prepare config for main() function (remove non-parameter keys)
    train_config = {k: v for k, v in config.items()
                   if k not in ['name', 'description']}

    # Set model save path
    model_path = os.path.join(exp_dir, 'model.pth')
    train_config['model_save_path'] = model_path

    # Set seed for reproducibility
    set_seed(seed=1)

    # Run training
    start_time = time.time()
    try:
        metrics = train_main(**train_config)
        elapsed_time = time.time() - start_time

        # Add metadata to metrics
        metrics['experiment_id'] = experiment_id
        metrics['experiment_name'] = exp_name
        metrics['description'] = exp_description
        metrics['elapsed_time'] = elapsed_time
        metrics['timestamp'] = datetime.now().isoformat()

        # Save metrics
        metrics_path = os.path.join(exp_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"\n{'='*80}")
        print(f"Experiment {experiment_id} completed successfully!")
        print(f"Final validation accuracy: {metrics['final_val_accuracy']:.4f}")
        print(f"Best validation loss: {metrics['best_val_loss']:.4f}")
        print(f"Time elapsed: {elapsed_time:.1f}s")
        print(f"{'='*80}\n")

        return metrics

    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR: Experiment {experiment_id} failed!")
        print(f"Error message: {str(e)}")
        print(f"{'='*80}\n")

        # Save error information
        error_info = {
            'experiment_id': experiment_id,
            'experiment_name': exp_name,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
        error_path = os.path.join(exp_dir, 'error.json')
        with open(error_path, 'w') as f:
            json.dump(error_info, f, indent=2)

        return None


def save_summary_csv(all_results, results_dir='results'):
    """
    Create and save a summary CSV of all experiments.

    Args:
        all_results (list): List of result dictionaries from all experiments
        results_dir (str): Directory to save the summary
    """
    # Filter out None results (failed experiments)
    successful_results = [r for r in all_results if r is not None]

    if not successful_results:
        print("No successful experiments to summarize!")
        return

    # Create summary dataframe
    summary_data = []
    for result in successful_results:
        summary_data.append({
            'experiment_id': result['experiment_id'],
            'name': result['experiment_name'],
            'description': result['description'],
            'final_val_accuracy': result['final_val_accuracy'],
            'final_val_loss': result['final_val_loss'],
            'best_val_loss': result['best_val_loss'],
            'final_train_loss': result['final_train_loss'],
            'elapsed_time': result['elapsed_time']
        })

    df = pd.DataFrame(summary_data)

    # Sort by validation accuracy (descending)
    df = df.sort_values('final_val_accuracy', ascending=False)

    # Save to CSV
    csv_path = os.path.join(results_dir, 'summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSummary saved to: {csv_path}")

    # Print summary table
    print("\n" + "="*100)
    print("EXPERIMENT SUMMARY")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100 + "\n")

    # Print top 3 experiments
    print("\nTop 3 Experiments by Validation Accuracy:")
    print("-" * 60)
    for idx, row in df.head(3).iterrows():
        print(f"{row['name']}: {row['final_val_accuracy']:.4f} "
              f"(val_loss: {row['final_val_loss']:.4f})")
    print()


def main(quick_test=False):
    """
    Main function to run all experiments.

    Args:
        quick_test (bool): If True, run only a small subset of experiments
    """
    # Get experiment configurations
    if quick_test:
        print("\nRunning QUICK TEST mode (3 experiments with 2 epochs each)\n")
        configs = get_quick_test_configs()
    else:
        print("\nRunning FULL experiment suite\n")
        configs = get_experiment_configs()

    print(f"Total experiments to run: {len(configs)}\n")

    # Create results directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Run all experiments
    all_results = []
    for idx, config in enumerate(configs, start=1):
        result = run_single_experiment(config, idx, results_dir)
        all_results.append(result)

    # Save summary
    save_summary_csv(all_results, results_dir)

    # Print final statistics
    successful = sum(1 for r in all_results if r is not None)
    failed = len(all_results) - successful

    print(f"\nExperiment suite completed!")
    print(f"Successful: {successful}/{len(all_results)}")
    if failed > 0:
        print(f"Failed: {failed}/{len(all_results)}")
    print(f"\nResults saved to: {results_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Run Vision Transformer hyperparameter experiments'
    )
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run a quick test with 3 experiments (2 epochs each)'
    )
    args = parser.parse_args()

    main(quick_test=args.quick_test)
