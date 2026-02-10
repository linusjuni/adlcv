"""
Visualization and analysis script for experiment results.

This script:
1. Loads results from all experiments
2. Creates comparison plots
3. Generates hyperparameter analysis
4. Saves visualizations and summary tables
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_all_results(results_dir='results'):
    """
    Load all experiment results from the results directory.

    Args:
        results_dir (str): Directory containing experiment results

    Returns:
        list: List of dicts with combined config and metrics data
    """
    results = []

    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return results

    # Iterate through experiment directories
    for exp_dir in sorted(os.listdir(results_dir)):
        exp_path = os.path.join(results_dir, exp_dir)

        # Skip if not a directory or doesn't start with 'experiment_'
        if not os.path.isdir(exp_path) or not exp_dir.startswith('experiment_'):
            continue

        config_path = os.path.join(exp_path, 'config.json')
        metrics_path = os.path.join(exp_path, 'metrics.json')

        # Check if experiment completed successfully
        if not os.path.exists(metrics_path):
            print(f"Skipping {exp_dir} - no metrics found (possibly failed)")
            continue

        # Load config and metrics
        with open(config_path, 'r') as f:
            config = json.load(f)
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        # Combine config and metrics
        result = {**config, **metrics}
        results.append(result)

    print(f"Loaded {len(results)} experiment results\n")
    return results


def create_accuracy_comparison(results, output_dir='plots'):
    """
    Create a bar chart comparing final validation accuracy across experiments.

    Args:
        results (list): List of experiment result dicts
        output_dir (str): Directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract data
    names = [r['name'] for r in results]
    accuracies = [r['final_val_accuracy'] for r in results]

    # Sort by accuracy
    sorted_indices = sorted(range(len(accuracies)), key=lambda i: accuracies[i], reverse=True)
    names = [names[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))

    colors = ['green' if i == 0 else 'skyblue' for i in range(len(names))]
    bars = ax.bar(range(len(names)), accuracies, color=colors)

    # Highlight best result
    ax.axhline(y=max(accuracies), color='green', linestyle='--', alpha=0.3, label='Best accuracy')

    ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
    ax.set_ylabel('Final Validation Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Validation Accuracy Comparison Across Experiments', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylim([min(accuracies) * 0.95, max(accuracies) * 1.02])
    ax.legend()

    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.4f}',
                ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'accuracy_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_loss_curves(results, output_dir='plots'):
    """
    Create line plots showing training and validation loss curves over epochs.

    Args:
        results (list): List of experiment result dicts
        output_dir (str): Directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot training and validation loss for each experiment
    for result in results:
        name = result['name']
        epoch_metrics = result['epoch_metrics']

        epochs = [m['epoch'] for m in epoch_metrics]
        train_losses = [m['train_loss'] for m in epoch_metrics]
        val_losses = [m['val_loss'] for m in epoch_metrics]

        ax1.plot(epochs, train_losses, label=name, alpha=0.7)
        ax2.plot(epochs, val_losses, label=name, alpha=0.7)

    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training Loss Curves', fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Validation Loss Curves', fontsize=14, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'loss_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_hyperparameter_analysis(results, output_dir='plots'):
    """
    Create grouped bar charts showing impact of each hyperparameter.

    Args:
        results (list): List of experiment result dicts
        output_dir (str): Directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)

    # Group experiments by hyperparameter type
    param_groups = defaultdict(list)

    for result in results:
        name = result['name']

        # Categorize by parameter type
        if 'pos_enc' in name:
            param_groups['Positional Encoding'].append(result)
        elif 'pool' in name:
            param_groups['Pooling Strategy'].append(result)
        elif 'embed_dim' in name:
            param_groups['Embedding Dimension'].append(result)
        elif 'layers' in name:
            param_groups['Number of Layers'].append(result)
        elif 'heads' in name:
            param_groups['Number of Heads'].append(result)
        elif 'dropout' in name:
            param_groups['Dropout Rate'].append(result)
        elif 'patch_size' in name:
            param_groups['Patch Size'].append(result)
        elif 'baseline' in name:
            # Add baseline to all groups for comparison
            for group in ['Positional Encoding', 'Pooling Strategy', 'Embedding Dimension',
                         'Number of Layers', 'Number of Heads', 'Dropout Rate', 'Patch Size']:
                param_groups[group].append(result)

    # Create subplots
    n_groups = len(param_groups)
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.flatten()

    for idx, (param_name, group_results) in enumerate(sorted(param_groups.items())):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # Extract data
        names = [r['name'] for r in group_results]
        accuracies = [r['final_val_accuracy'] for r in group_results]

        # Sort by accuracy
        sorted_indices = sorted(range(len(accuracies)), key=lambda i: accuracies[i], reverse=True)
        names = [names[i] for i in sorted_indices]
        accuracies = [accuracies[i] for i in sorted_indices]

        # Create bars with color coding
        colors = ['gold' if 'baseline' in name else 'skyblue' for name in names]
        bars = ax.bar(range(len(names)), accuracies, color=colors)

        ax.set_title(param_name, fontsize=12, fontweight='bold')
        ax.set_ylabel('Validation Accuracy', fontsize=10)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)

        # Add value labels
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.3f}',
                    ha='center', va='bottom', fontsize=7)

    # Remove unused subplots
    for idx in range(len(param_groups), len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle('Hyperparameter Impact Analysis', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'hyperparameter_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def save_summary_table(results, output_dir='results'):
    """
    Create and save a detailed summary table in multiple formats.

    Args:
        results (list): List of experiment result dicts
        output_dir (str): Directory to save the tables
    """
    # Create DataFrame
    summary_data = []
    for result in results:
        # Determine changed parameter
        name = result['name']
        if name == 'baseline':
            changed_param = 'none'
            param_value = 'baseline'
        else:
            # Extract parameter info from name
            if 'pos_enc' in name:
                changed_param = 'pos_enc'
                param_value = result['pos_enc']
            elif 'pool' in name:
                changed_param = 'pool'
                param_value = result['pool']
            elif 'embed_dim' in name:
                changed_param = 'embed_dim'
                param_value = result['embed_dim']
            elif 'layers' in name:
                changed_param = 'num_layers'
                param_value = result['num_layers']
            elif 'heads' in name:
                changed_param = 'num_heads'
                param_value = result['num_heads']
            elif 'dropout' in name:
                changed_param = 'dropout'
                param_value = result['dropout']
            elif 'patch_size' in name:
                changed_param = 'patch_size'
                param_value = str(result['patch_size'])
            else:
                changed_param = 'unknown'
                param_value = 'unknown'

        summary_data.append({
            'experiment_name': name,
            'changed_param': changed_param,
            'param_value': param_value,
            'val_accuracy': result['final_val_accuracy'],
            'val_loss': result['final_val_loss'],
            'best_val_loss': result['best_val_loss'],
            'train_loss': result['final_train_loss'],
            'elapsed_time': result.get('elapsed_time', 0)
        })

    df = pd.DataFrame(summary_data)
    df = df.sort_values('val_accuracy', ascending=False)

    # Save as CSV
    csv_path = os.path.join(output_dir, 'detailed_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Save as Markdown table for report
    md_path = os.path.join(output_dir, 'summary_table.md')
    with open(md_path, 'w') as f:
        f.write("# Experiment Results Summary\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Top 5 Configurations\n\n")
        f.write(df.head(5).to_markdown(index=False))
    print(f"Saved: {md_path}")

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Best validation accuracy: {df['val_accuracy'].max():.4f} ({df.iloc[0]['experiment_name']})")
    print(f"Worst validation accuracy: {df['val_accuracy'].min():.4f} ({df.iloc[-1]['experiment_name']})")
    print(f"Mean validation accuracy: {df['val_accuracy'].mean():.4f}")
    print(f"Std validation accuracy: {df['val_accuracy'].std():.4f}")
    print("="*80 + "\n")


def main():
    """
    Main function to load results and generate all visualizations.
    """
    # Determine paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    plots_dir = os.path.join(script_dir, 'plots')

    print("="*80)
    print("VISION TRANSFORMER EXPERIMENT VISUALIZATION")
    print("="*80 + "\n")

    # Load all results
    print("Loading experiment results...")
    results = load_all_results(results_dir)

    if not results:
        print("No results found! Please run experiments first.")
        return

    print(f"Found {len(results)} completed experiments\n")

    # Generate visualizations
    print("Generating visualizations...\n")

    print("1. Creating accuracy comparison chart...")
    create_accuracy_comparison(results, plots_dir)

    print("2. Creating loss curves...")
    create_loss_curves(results, plots_dir)

    print("3. Creating hyperparameter analysis...")
    create_hyperparameter_analysis(results, plots_dir)

    print("4. Saving summary tables...")
    save_summary_table(results, results_dir)

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"Plots saved to: {plots_dir}")
    print(f"Tables saved to: {results_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
