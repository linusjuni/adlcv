# Vision Transformer Hyperparameter Experiments

This directory contains a complete framework for running systematic hyperparameter experiments on Vision Transformers for CIFAR-10 classification.

## Overview

The framework supports:
- **Automated experiment execution** with configuration management
- **Systematic hyperparameter ablation studies** (changing one parameter at a time)
- **Results tracking** (metrics, models, and configurations)
- **Visualization generation** for reports and analysis

## Quick Start

### 1. Quick Test (3 experiments, 2 epochs each)

```bash
cd /zhome/77/2/187952/projects/adlcv/adlcv_ex_2
uv run -m experiments.run_experiments --quick-test
```

This runs a quick test with 3 experiments to verify everything works.

### 2. Run Full Experiment Suite (16 experiments, 20 epochs each)

```bash
uv run -m experiments.run_experiments
```

This will take significant time (~20 minutes per experiment = ~5-6 hours total).

### 3. Generate Visualizations

After experiments complete:

```bash
uv run -m experiments.visualize_results
```

This creates:
- `plots/accuracy_comparison.png` - Bar chart comparing all experiments
- `plots/loss_curves.png` - Training/validation loss over epochs
- `plots/hyperparameter_analysis.png` - Grouped analysis by parameter type
- `results/detailed_summary.csv` - Complete results table
- `results/summary_table.md` - Markdown table for reports

## Directory Structure

```
experiments/
├── __init__.py                  # Package initialization
├── configs.py                   # Configuration definitions
├── run_experiments.py           # Main experiment runner
├── visualize_results.py         # Analysis and visualization
├── test_setup.py               # Test suite
├── README.md                    # This file
├── results/                     # Auto-created by runner
│   ├── experiment_001/          # One directory per experiment
│   │   ├── config.json          # Full configuration
│   │   ├── metrics.json         # Training metrics
│   │   └── model.pth            # Saved model weights
│   ├── experiment_002/
│   ├── ...
│   ├── summary.csv              # Summary of all experiments
│   └── detailed_summary.csv     # Detailed summary with parameters
└── plots/                       # Auto-created by visualizer
    ├── accuracy_comparison.png
    ├── loss_curves.png
    └── hyperparameter_analysis.png
```

## Experiment Configurations

The framework tests 16 different configurations:

1. **Baseline**: Default configuration
2. **Positional Encoding**: learnable, fixed, none
3. **Pooling Strategy**: cls, mean, max
4. **Embedding Dimension**: 64, 128, 256
5. **Number of Layers**: 2, 4, 6
6. **Number of Heads**: 2, 4, 8
7. **Dropout**: 0.0, 0.1, 0.3, 0.5
8. **Patch Size**: (2,2), (4,4), (8,8)

Each experiment varies **one parameter** from the baseline to enable clear ablation analysis.

## Customizing Experiments

### Add New Experiments

Edit `configs.py` and add to `get_experiment_configs()`:

```python
experiments.append({
    'name': 'custom_config',
    'description': 'Description of what this tests',
    **{**BASE_CONFIG, 'embed_dim': 512}  # Change parameters here
})
```

### Modify Base Configuration

Edit `BASE_CONFIG` in `configs.py`:

```python
BASE_CONFIG = {
    'num_epochs': 20,  # Change number of epochs
    'batch_size': 16,  # Change batch size
    # ... other parameters
}
```

## Verifying the Framework

Run the test suite:

```bash
uv run -m adlcv_ex_2.experiments.test_setup
```

This verifies:
- Configuration generation works
- Modified `main()` function returns metrics
- Experiment runner saves results correctly

## Results Analysis

After running experiments:

1. **View Summary Table**: `cat results/summary.csv`
2. **Best Configuration**: Check first row of `summary.csv` (sorted by accuracy)
3. **Compare Visualizations**: Open PNG files in `plots/` directory
4. **Detailed Analysis**: Load `results/detailed_summary.csv` in Python/pandas

Example analysis:

```python
import pandas as pd

# Load results
df = pd.read_csv('results/detailed_summary.csv')

# Find best configuration
best = df.iloc[0]
print(f"Best: {best['experiment_name']} - Accuracy: {best['val_accuracy']:.4f}")

# Analyze by parameter type
pos_enc_results = df[df['changed_param'] == 'pos_enc']
print(pos_enc_results[['param_value', 'val_accuracy']])
```

## Tips

### Running Experiments in Background

For long-running experiments:

```bash
nohup uv run -m experiments.run_experiments > experiments.log 2>&1 &
tail -f experiments.log  # Monitor progress
```

### Interrupting Experiments

If you need to stop experiments, just Ctrl+C. Completed experiments are saved and can be analyzed.

### Re-running Specific Experiments

To re-run a specific configuration:

```python
from adlcv_ex_2.imageclassification import main, set_seed
from adlcv_ex_2.experiments.configs import get_experiment_configs

configs = get_experiment_configs()
config = configs[0]  # Run first experiment

# Remove 'name' and 'description' before passing to main()
train_config = {k: v for k, v in config.items() if k not in ['name', 'description']}
train_config['model_save_path'] = 'my_model.pth'

set_seed(seed=1)
results = main(**train_config)
```

## Troubleshooting

### "No module named 'adlcv_ex_2'"

Make sure you're running from the project root:

```bash
cd /zhome/77/2/187952/projects/adlcv/adlcv_ex_2
```

### "CUDA out of memory"

Reduce batch size in `BASE_CONFIG`:

```python
BASE_CONFIG = {
    ...
    'batch_size': 8,  # Reduce from 16
    ...
}
```

### Experiments Failing

Check `results/experiment_XXX/error.json` for error details.

## For the Report (Exercise 1.2, Task 3)

After running experiments and generating visualizations, include:

1. **Accuracy comparison chart**: `plots/accuracy_comparison.png`
2. **Hyperparameter analysis**: `plots/hyperparameter_analysis.png`
3. **Summary table**: Content from `results/summary_table.md`
4. **Discussion**: Which parameters had the biggest impact? Best configuration?

The visualizations directly address Task 3: "Illustrate how different choices for your model affect the performance of your model at your given task."
