"""
Quick test script to verify the experiment framework is working correctly.
"""

import os
import sys
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from adlcv_ex_2.imageclassification import main as train_main, set_seed
from adlcv_ex_2.experiments.configs import BASE_CONFIG

def test_modified_main():
    """Test that main() returns metrics correctly."""
    print("Testing modified main() function...")
    print("-" * 60)

    # Create a minimal test config with very few epochs
    test_config = {
        **BASE_CONFIG,
        'num_epochs': 2,
        'model_save_path': '/tmp/test_model.pth'
    }

    set_seed(seed=1)

    try:
        result = train_main(**test_config)

        # Check that result is returned
        assert result is not None, "main() should return a result dict"

        # Check required keys
        required_keys = ['final_train_loss', 'final_val_loss', 'final_val_accuracy',
                        'best_val_loss', 'epoch_metrics']
        for key in required_keys:
            assert key in result, f"Result should contain '{key}'"

        # Check epoch_metrics structure
        assert len(result['epoch_metrics']) == 2, "Should have 2 epoch metrics"
        assert 'epoch' in result['epoch_metrics'][0], "Epoch metrics should contain 'epoch'"
        assert 'train_loss' in result['epoch_metrics'][0], "Epoch metrics should contain 'train_loss'"
        assert 'val_loss' in result['epoch_metrics'][0], "Epoch metrics should contain 'val_loss'"
        assert 'val_accuracy' in result['epoch_metrics'][0], "Epoch metrics should contain 'val_accuracy'"

        print("✓ main() returns correct structure")
        print(f"✓ Final validation accuracy: {result['final_val_accuracy']:.4f}")
        print(f"✓ Best validation loss: {result['best_val_loss']:.4f}")
        print(f"✓ Number of epoch metrics: {len(result['epoch_metrics'])}")

        return True

    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_experiment_runner():
    """Test that run_single_experiment works correctly."""
    print("\n" + "=" * 60)
    print("Testing experiment runner...")
    print("-" * 60)

    from adlcv_ex_2.experiments.run_experiments import run_single_experiment

    # Create test config
    test_config = {
        'name': 'test_experiment',
        'description': 'Quick test',
        **BASE_CONFIG,
        'num_epochs': 2
    }

    # Create temporary results directory
    test_results_dir = '/tmp/test_experiment_results'
    os.makedirs(test_results_dir, exist_ok=True)

    try:
        result = run_single_experiment(test_config, 999, test_results_dir)

        # Check result
        assert result is not None, "Experiment should return results"

        # Check files were created
        exp_dir = os.path.join(test_results_dir, 'experiment_999')
        assert os.path.exists(exp_dir), "Experiment directory should exist"
        assert os.path.exists(os.path.join(exp_dir, 'config.json')), "config.json should exist"
        assert os.path.exists(os.path.join(exp_dir, 'metrics.json')), "metrics.json should exist"
        assert os.path.exists(os.path.join(exp_dir, 'model.pth')), "model.pth should exist"

        # Check config.json content
        with open(os.path.join(exp_dir, 'config.json'), 'r') as f:
            saved_config = json.load(f)
            assert saved_config['name'] == 'test_experiment', "Config should be saved correctly"

        # Check metrics.json content
        with open(os.path.join(exp_dir, 'metrics.json'), 'r') as f:
            saved_metrics = json.load(f)
            assert 'final_val_accuracy' in saved_metrics, "Metrics should be saved correctly"
            assert 'experiment_name' in saved_metrics, "Metrics should include experiment name"

        print("✓ Experiment runner works correctly")
        print(f"✓ Created directory: {exp_dir}")
        print(f"✓ Saved config, metrics, and model")
        print(f"✓ Final accuracy: {saved_metrics['final_val_accuracy']:.4f}")

        return True

    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_configs():
    """Test that configs are generated correctly."""
    print("\n" + "=" * 60)
    print("Testing configuration generation...")
    print("-" * 60)

    from adlcv_ex_2.experiments.configs import get_experiment_configs, get_quick_test_configs

    try:
        # Test full configs
        configs = get_experiment_configs()
        assert len(configs) > 0, "Should generate at least one config"
        assert all('name' in c for c in configs), "All configs should have 'name'"
        assert all('description' in c for c in configs), "All configs should have 'description'"

        print(f"✓ Generated {len(configs)} experiment configs")

        # Test quick configs
        quick_configs = get_quick_test_configs()
        assert len(quick_configs) == 3, "Should generate 3 quick test configs"
        assert all(c['num_epochs'] == 2 for c in quick_configs), "Quick configs should have 2 epochs"

        print(f"✓ Generated {len(quick_configs)} quick test configs")

        # Print summary
        print("\nExperiment names:")
        for config in configs:
            print(f"  - {config['name']}")

        return True

    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("EXPERIMENT FRAMEWORK TEST SUITE")
    print("=" * 60 + "\n")

    results = []

    # Test 1: Configuration generation
    results.append(("Config Generation", test_configs()))

    # Test 2: Modified main() function
    results.append(("Modified main()", test_modified_main()))

    # Test 3: Experiment runner
    results.append(("Experiment Runner", test_experiment_runner()))

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(passed for _, passed in results)
    print("=" * 60)

    if all_passed:
        print("\n✓ All tests passed! The experiment framework is ready to use.")
        print("\nNext steps:")
        print("  1. Run quick test: python -m experiments.run_experiments --quick-test")
        print("  2. Run full experiments: python -m experiments.run_experiments")
        print("  3. Generate visualizations: python -m experiments.visualize_results")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
