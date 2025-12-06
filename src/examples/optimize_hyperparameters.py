#!/usr/bin/env python3
"""
Example: Hyperparameter Optimization using Optuna
Demonstrates how to optimize CognitiveConfig parameters for better learning performance
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.hyperparameter_optimizer import HyperparameterOptimizer
from core.cognitive_models import CognitiveConfig


def main():
    """Run hyperparameter optimization example"""
    print(" Hyperparameter Optimization Example")
    print("=" * 60)
    
    # Create optimizer
    optimizer = HyperparameterOptimizer()
    
    # Create synthetic training dataset
    print("\n Creating synthetic training dataset...")
    training_data = optimizer.create_synthetic_dataset(
        num_concepts=15,
        samples_per_concept=4,
    )
    
    # Create validation dataset (different samples of same concepts)
    validation_data = optimizer.create_synthetic_dataset(
        num_concepts=15,
        samples_per_concept=2,
    )
    
    print(f"   Training samples: {len(training_data)}")
    print(f"   Validation samples: {len(validation_data)}")
    
    # Run optimization (small number of trials for demo)
    print("\n Starting optimization...")
    print("   This will take a few minutes...")
    
    study, best_config = optimizer.optimize(
        training_data=training_data,
        validation_data=validation_data,
        n_trials=20,  # Small for demo, use 100+ for real optimization
        study_name="demo_optimization",
        direction="maximize",
        seed=42,
    )
    
    # Display results
    print("\n Optimization Results:")
    print("=" * 60)
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Score: {study.best_value:.4f}")
    print(f"\nBest Parameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value:.4f}" if isinstance(value, float) else f"  {param}: {value}")
    
    # Compare with default config
    print("\n Comparison with Default Config:")
    default_config = CognitiveConfig()
    
    key_params = [
        "hebbian_learning_rate",
        "max_association_strength",
        "semantic_consolidation_threshold",
        "consolidation_threshold",
    ]
    
    print(f"{'Parameter':<40} {'Default':<15} {'Optimized':<15} {'Change':<15}")
    print("-" * 85)
    for param in key_params:
        default_val = getattr(default_config, param)
        optimized_val = getattr(best_config, param)
        change = optimized_val - default_val
        change_pct = (change / default_val * 100) if default_val != 0 else 0
        
        print(f"{param:<40} {default_val:<15.4f} {optimized_val:<15.4f} {change_pct:+.1f}%")
    
    print("\n Optimization complete!")
    print(f"   Best config saved to: optimization_studies/demo_optimization_best_config.json")
    print(f"   Study database: optimization_studies/demo_optimization.db")
    
    # Show how to use optimized config
    print("\n Usage:")
    print("   from training.hyperparameter_optimizer import HyperparameterOptimizer")
    print("   optimizer = HyperparameterOptimizer()")
    print("   best_config = optimizer.load_config('demo_optimization')")
    print("   architecture = CognitiveArchitecture(config=best_config)")


if __name__ == "__main__":
    main()


