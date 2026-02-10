"""
Configuration management for Vision Transformer hyperparameter experiments.

This module defines the base configuration and generates experiment configurations
that systematically vary individual hyperparameters for ablation studies.
"""

# Base configuration matching the default parameters from main()
BASE_CONFIG = {
    'image_size': (32, 32),
    'patch_size': (4, 4),
    'channels': 3,
    'embed_dim': 128,
    'num_heads': 4,
    'num_layers': 4,
    'num_classes': 2,
    'pos_enc': 'learnable',
    'pool': 'cls',
    'dropout': 0.3,
    'fc_dim': None,
    'num_epochs': 20,
    'batch_size': 16,
    'lr': 1e-4,
    'warmup_steps': 625,
    'weight_decay': 1e-3,
    'gradient_clipping': 1
}


def get_experiment_configs():
    """
    Generate a list of experiment configurations for hyperparameter study.

    Each configuration varies ONE parameter from the baseline to enable
    clear analysis of individual hyperparameter effects.

    Returns:
        list: List of dicts, each containing 'name', 'description', and config params
    """
    experiments = []

    # 1. Baseline experiment
    experiments.append({
        'name': 'baseline',
        'description': 'Default configuration with learnable pos_enc and cls pooling',
        **BASE_CONFIG
    })

    # 2. Positional encoding variants
    experiments.append({
        'name': 'pos_enc_fixed',
        'description': 'Fixed sinusoidal positional encoding',
        **{**BASE_CONFIG, 'pos_enc': 'fixed'}
    })

    experiments.append({
        'name': 'pos_enc_none',
        'description': 'No positional encoding',
        **{**BASE_CONFIG, 'pos_enc': 'none'}
    })

    # 3. Pooling strategy variants
    experiments.append({
        'name': 'pool_mean',
        'description': 'Mean pooling instead of CLS token',
        **{**BASE_CONFIG, 'pool': 'mean'}
    })

    experiments.append({
        'name': 'pool_max',
        'description': 'Max pooling instead of CLS token',
        **{**BASE_CONFIG, 'pool': 'max'}
    })

    # 4. Embedding dimension variants
    experiments.append({
        'name': 'embed_dim_64',
        'description': 'Smaller embedding dimension (64)',
        **{**BASE_CONFIG, 'embed_dim': 64}
    })

    experiments.append({
        'name': 'embed_dim_256',
        'description': 'Larger embedding dimension (256)',
        **{**BASE_CONFIG, 'embed_dim': 256}
    })

    # 5. Number of layers variants
    experiments.append({
        'name': 'layers_2',
        'description': 'Shallow model with 2 transformer layers',
        **{**BASE_CONFIG, 'num_layers': 2}
    })

    experiments.append({
        'name': 'layers_6',
        'description': 'Deeper model with 6 transformer layers',
        **{**BASE_CONFIG, 'num_layers': 6}
    })

    # 6. Number of heads variants
    experiments.append({
        'name': 'heads_2',
        'description': 'Fewer attention heads (2)',
        **{**BASE_CONFIG, 'num_heads': 2}
    })

    experiments.append({
        'name': 'heads_8',
        'description': 'More attention heads (8)',
        **{**BASE_CONFIG, 'num_heads': 8}
    })

    # 7. Dropout variants
    experiments.append({
        'name': 'dropout_0.0',
        'description': 'No dropout',
        **{**BASE_CONFIG, 'dropout': 0.0}
    })

    experiments.append({
        'name': 'dropout_0.1',
        'description': 'Low dropout (0.1)',
        **{**BASE_CONFIG, 'dropout': 0.1}
    })

    experiments.append({
        'name': 'dropout_0.5',
        'description': 'High dropout (0.5)',
        **{**BASE_CONFIG, 'dropout': 0.5}
    })

    # 8. Patch size variants
    experiments.append({
        'name': 'patch_size_2x2',
        'description': 'Smaller patches (2x2) - more fine-grained',
        **{**BASE_CONFIG, 'patch_size': (2, 2)}
    })

    experiments.append({
        'name': 'patch_size_8x8',
        'description': 'Larger patches (8x8) - more coarse-grained',
        **{**BASE_CONFIG, 'patch_size': (8, 8)}
    })

    return experiments


def get_quick_test_configs():
    """
    Get a small subset of configurations for quick testing.

    Returns:
        list: List of 3 experiment configurations
    """
    return [
        {
            'name': 'test_baseline',
            'description': 'Quick baseline test',
            **{**BASE_CONFIG, 'num_epochs': 2}
        },
        {
            'name': 'test_pos_enc',
            'description': 'Quick positional encoding test',
            **{**BASE_CONFIG, 'num_epochs': 2, 'pos_enc': 'fixed'}
        },
        {
            'name': 'test_pool',
            'description': 'Quick pooling test',
            **{**BASE_CONFIG, 'num_epochs': 2, 'pool': 'mean'}
        }
    ]
