# Experiment Framework Implementation Summary

## What Was Implemented

A complete hyperparameter experiment framework for Vision Transformer training on CIFAR-10, addressing Exercise 1.2, Task 3: "Illustrate how different choices for your model affect the performance."

## Files Modified

### `/zhome/77/2/187952/projects/adlcv/adlcv_ex_2/imageclassification.py`

**Changes:**
1. Added `model_save_path='model.pth'` parameter to `main()` function
2. Added `epoch_metrics = []` to collect per-epoch training data
3. Modified model save to use `model_save_path` parameter
4. Added metric collection at end of each epoch
5. Added return statement with final metrics and epoch history

**Purpose:** Enable programmatic execution with custom save paths and metric extraction.

### `/zhome/77/2/187952/projects/adlcv/pyproject.toml`

**Changes:**
- Added `pandas>=2.0.0` to dependencies

**Purpose:** Required for CSV generation and data analysis.

### `/zhome/77/2/187952/projects/adlcv/adlcv_ex_2/__init__.py`

**Created:** Empty file to make `adlcv_ex_2` a proper Python package.

## Files Created

### `/zhome/77/2/187952/projects/adlcv/adlcv_ex_2/experiments/__init__.py`

Empty package initialization file.

### `/zhome/77/2/187952/projects/adlcv/adlcv_ex_2/experiments/configs.py`

**Contains:**
- `BASE_CONFIG`: Default hyperparameters matching original `main()` function
- `get_experiment_configs()`: Generates 16 experiment configurations
  - 1 baseline
  - 15 ablation studies (each changing one parameter)
- `get_quick_test_configs()`: 3 quick test configurations (2 epochs each)

**Experiments:**
- Positional encoding: learnable, fixed, none
- Pooling strategy: cls, mean, max
- Embedding dimension: 64, 128, 256
- Number of layers: 2, 4, 6
- Number of heads: 2, 4, 8
- Dropout: 0.0, 0.1, 0.3, 0.5
- Patch size: (2,2), (4,4), (8,8)

### `/zhome/77/2/187952/projects/adlcv/adlcv_ex_2/experiments/run_experiments.py`

**Main experiment runner script.**

**Functions:**
- `run_single_experiment()`: Runs one experiment, saves config/metrics/model
- `save_summary_csv()`: Creates summary CSV and prints results table
- `main()`: Orchestrates all experiments with error handling

**Features:**
- Sequential execution of all experiments
- Automatic directory creation (`results/experiment_XXX/`)
- JSON storage of configs and metrics
- Error handling with graceful continuation
- Summary CSV generation sorted by accuracy
- Command-line argument for quick test mode

### `/zhome/77/2/187952/projects/adlcv/adlcv_ex_2/experiments/visualize_results.py`

**Results analysis and visualization script.**

**Functions:**
- `load_all_results()`: Loads all experiment results from disk
- `create_accuracy_comparison()`: Bar chart of final validation accuracies
- `create_loss_curves()`: Line plots of training/validation loss
- `create_hyperparameter_analysis()`: Grouped analysis by parameter type
- `save_summary_table()`: Creates CSV and Markdown summary tables
- `main()`: Generates all visualizations and tables

**Outputs:**
- `plots/accuracy_comparison.png`
- `plots/loss_curves.png`
- `plots/hyperparameter_analysis.png`
- `results/detailed_summary.csv`
- `results/summary_table.md`

### `/zhome/77/2/187952/projects/adlcv/adlcv_ex_2/experiments/test_setup.py`

**Test suite to verify framework functionality.**

**Tests:**
- Configuration generation (16 experiments)
- Modified `main()` function (returns metrics correctly)
- Experiment runner (creates files, saves results)

**Usage:** `uv run -m adlcv_ex_2.experiments.test_setup`

### `/zhome/77/2/187952/projects/adlcv/adlcv_ex_2/experiments/README.md`

Complete user documentation with:
- Quick start guide
- Directory structure explanation
- Customization instructions
- Troubleshooting tips
- Usage examples

## How to Use

### 1. Verify Installation (5 seconds)

```bash
cd /zhome/77/2/187952/projects/adlcv/adlcv_ex_2
uv run -m adlcv_ex_2.experiments.test_setup
```

Expected output: All tests pass (✓ PASSED)

### 2. Quick Test (2-3 minutes)

```bash
uv run -m experiments.run_experiments --quick-test
```

Runs 3 experiments with 2 epochs each to verify everything works.

### 3. Full Experiment Suite (5-6 hours)

```bash
uv run -m experiments.run_experiments
```

Runs all 16 experiments with 20 epochs each.

**Recommendation:** Run in background:
```bash
nohup uv run -m experiments.run_experiments > experiments.log 2>&1 &
tail -f experiments.log  # Monitor progress
```

### 4. Generate Visualizations (10 seconds)

```bash
uv run -m experiments.visualize_results
```

Creates plots and summary tables in `experiments/plots/` and `experiments/results/`.

## Expected Outputs

### After Running Experiments

```
experiments/results/
├── experiment_001/
│   ├── config.json         # Full hyperparameter configuration
│   ├── metrics.json        # Final metrics + per-epoch history
│   └── model.pth           # Trained model weights
├── experiment_002/
│   └── ...
├── ...
├── experiment_016/
│   └── ...
└── summary.csv             # All experiments ranked by accuracy
```

### After Generating Visualizations

```
experiments/plots/
├── accuracy_comparison.png          # Bar chart of all experiments
├── loss_curves.png                  # Training/val loss over epochs
└── hyperparameter_analysis.png     # Grouped by parameter type

experiments/results/
├── detailed_summary.csv             # Full results with parameters
└── summary_table.md                 # Markdown table for reports
```

## For the Report

Include these in your Exercise 1.2, Task 3 writeup:

1. **Figure 1**: `accuracy_comparison.png` - Shows which configurations perform best
2. **Figure 2**: `hyperparameter_analysis.png` - Shows impact of each parameter type
3. **Table 1**: Content from `summary_table.md` - Top 5 configurations
4. **Discussion**:
   - Which hyperparameters had the biggest impact?
   - What is the best configuration found?
   - How do positional encodings compare?
   - How does model capacity (layers, embed_dim) affect performance?

## Architecture Decisions

### Why Sequential Execution?

- Training is fast (~20 epochs in ~1 minute per experiment)
- Sequential is simpler and more robust
- Easier to monitor and debug
- Total time (~6 hours) is acceptable for overnight run

### Why Minimal Modifications?

- Only modified `main()` to add return value and save path parameter
- Preserved all existing functionality
- Can still run `imageclassification.py` directly as before
- No breaking changes to existing code

### Why JSON for Results?

- Human-readable for debugging
- Easy to load in Python/pandas for analysis
- Preserves nested structures (epoch metrics)
- Standard format for scientific computing

## Verification

All tests pass:
```
✓ Config Generation: PASSED (16 experiments generated)
✓ Modified main(): PASSED (returns correct metrics)
✓ Experiment Runner: PASSED (saves all files correctly)
```

## Troubleshooting

### Import Errors

If you see `ImportError: attempted relative import with no known parent package`:
- Make sure you're using `uv run -m experiments.run_experiments`
- Don't use `python experiments/run_experiments.py`

### CUDA Out of Memory

Edit `configs.py` and reduce `batch_size` in `BASE_CONFIG`:
```python
BASE_CONFIG = {
    ...
    'batch_size': 8,  # Reduce from 16
    ...
}
```

### Interrupted Experiments

- Completed experiments are saved automatically
- You can run visualization on partial results
- To resume, manually edit `run_experiments.py` to skip completed experiments

## Next Steps

1. **Run quick test** to verify everything works
2. **Run full experiments** (overnight recommended)
3. **Generate visualizations** after experiments complete
4. **Analyze results** for report
5. **Optional:** Add more experiment configurations to test additional hypotheses

## Technical Notes

- **Reproducibility**: Uses `set_seed(seed=1)` for all experiments
- **Error Handling**: Failed experiments are logged but don't stop the suite
- **Progress Tracking**: Console output shows experiment progress in real-time
- **File Organization**: Each experiment in separate directory for easy inspection
- **Sorting**: Summary CSV automatically sorted by validation accuracy (best first)

## Contact for Issues

If you encounter issues:
1. Check `experiments/README.md` for detailed usage instructions
2. Run `test_setup.py` to verify framework integrity
3. Check `results/experiment_XXX/error.json` for error details
4. Verify all dependencies are installed: `uv sync`
