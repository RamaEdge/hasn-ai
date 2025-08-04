#!/usr/bin/env python3
"""
Examples of using the configurable SimpleBrainNetwork
Demonstrates how to customize network parameters for different use cases
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.simplified_brain_network import SimpleBrainNetwork, NetworkConfig, NetworkPresets
import numpy as np


def example_1_basic_configuration():
    """Example 1: Basic custom configuration"""
    print("🎯 Example 1: Basic Custom Configuration")
    print("-" * 40)
    
    # Create custom configuration
    config = NetworkConfig(
        learning_rate=0.02,        # 2x faster learning
        learning_probability=0.2,  # 2x more frequent learning
        weight_range=(0.15, 1.0),  # Stronger initial connections
        max_weight=2.5             # Higher weight limits
    )
    
    network = SimpleBrainNetwork(num_neurons=50, config=config)
    
    print(f"✅ Network created with custom parameters:")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Learning probability: {config.learning_probability}")
    print(f"   Weight range: {config.weight_range}")
    print(f"   Max weight: {config.max_weight}")
    print()


def example_2_attention_configuration():
    """Example 2: Configurable attention system"""
    print("🎯 Example 2: Attention System Configuration")
    print("-" * 45)
    
    # Network with more attention modules
    config = NetworkConfig(
        num_attention_modules=8,    # Double the attention modules
        attention_init="random"     # Random attention distribution
    )
    
    network = SimpleBrainNetwork(num_neurons=100, config=config)
    
    print(f"✅ Network with advanced attention:")
    print(f"   Attention modules: {len(network.attention_weights)}")
    print(f"   Attention weights: {[f'{w:.3f}' for w in network.attention_weights]}")
    print(f"   Initialization: {config.attention_init}")
    print()


def example_3_preset_configurations():
    """Example 3: Using preset configurations"""
    print("🎯 Example 3: Preset Configurations")
    print("-" * 35)
    
    # Fast learning for rapid prototyping
    fast_config = NetworkPresets.fast_learning()
    fast_network = SimpleBrainNetwork(num_neurons=30, config=fast_config)
    
    # Stable learning for production
    stable_config = NetworkPresets.stable_learning()
    stable_network = SimpleBrainNetwork(num_neurons=30, config=stable_config)
    
    # Attention-focused for complex tasks
    attention_config = NetworkPresets.attention_focused()
    attention_network = SimpleBrainNetwork(num_neurons=30, config=attention_config)
    
    print("✅ Three networks with different presets:")
    print(f"   Fast learning rate: {fast_config.learning_rate}")
    print(f"   Stable learning rate: {stable_config.learning_rate}")
    print(f"   Attention modules: {attention_config.num_attention_modules}")
    print()


def example_4_research_configuration():
    """Example 4: Research-oriented configuration"""
    print("🎯 Example 4: Research Configuration")
    print("-" * 37)
    
    # Configuration for neuroscience research
    research_config = NetworkConfig(
        dt=0.1,                       # Higher temporal resolution
        learning_rate=0.001,          # Very slow, biologically realistic
        learning_probability=0.05,    # Sparse learning events
        weight_range=(0.05, 0.3),     # Weak synaptic connections
        max_weight=1.0,               # Physiological limits
        num_attention_modules=6,      # Moderate attention complexity
        max_spike_history=2000        # Longer memory for analysis
    )
    
    network = SimpleBrainNetwork(num_neurons=200, config=research_config)
    
    print("✅ Biologically realistic network:")
    print(f"   Time resolution: {research_config.dt}ms")
    print(f"   Synaptic strength: {research_config.weight_range}")
    print(f"   Learning sparsity: {research_config.learning_probability}")
    print(f"   Spike memory: {research_config.max_spike_history}")
    print()


def example_5_performance_tuning():
    """Example 5: Performance-oriented configuration"""
    print("🎯 Example 5: Performance Tuning")
    print("-" * 33)
    
    # High-performance configuration
    performance_config = NetworkConfig(
        dt=2.0,                       # Larger timesteps for speed
        learning_probability=0.02,    # Less frequent learning for efficiency
        max_spike_history=500,        # Smaller memory footprint
        weight_range=(0.3, 1.5),      # Strong connections for fast propagation
        num_attention_modules=2       # Minimal attention overhead
    )
    
    network = SimpleBrainNetwork(num_neurons=1000, config=performance_config)
    
    print("✅ High-performance network:")
    print(f"   Large timestep: {performance_config.dt}ms")
    print(f"   Efficient learning: {performance_config.learning_probability}")
    print(f"   Memory efficient: {performance_config.max_spike_history} spike history")
    print(f"   Minimal attention: {performance_config.num_attention_modules} modules")
    print()


def example_6_comparison():
    """Example 6: Comparing different configurations"""
    print("🎯 Example 6: Configuration Comparison")
    print("-" * 38)
    
    configs = {
        "Default": NetworkConfig(),
        "Fast": NetworkPresets.fast_learning(),
        "Stable": NetworkPresets.stable_learning(),
        "Attention": NetworkPresets.attention_focused()
    }
    
    print("✅ Configuration comparison:")
    print(f"{'Config':<10} {'Learning':<8} {'Probability':<11} {'Weights':<12} {'Attention':<9}")
    print("-" * 60)
    
    for name, config in configs.items():
        print(f"{name:<10} {config.learning_rate:<8.3f} {config.learning_probability:<11.2f} "
              f"{str(config.weight_range):<12} {config.num_attention_modules:<9}")
    print()


def example_7_custom_attention():
    """Example 7: Custom attention weights"""
    print("🎯 Example 7: Custom Attention Weights")
    print("-" * 38)
    
    # Create network with custom attention setup
    config = NetworkConfig(
        num_attention_modules=6,
        attention_init="equal"  # Start with equal, then customize
    )
    
    network = SimpleBrainNetwork(num_neurons=60, config=config)
    
    # Customize attention weights after creation
    # Example: Focus more on first and last modules
    custom_weights = np.array([0.3, 0.1, 0.1, 0.1, 0.1, 0.3])
    network.attention_weights = custom_weights
    
    print("✅ Custom attention distribution:")
    print(f"   Default: Equal weights (1/{config.num_attention_modules:.3f} each)")
    print(f"   Custom:  {[f'{w:.2f}' for w in custom_weights]}")
    print(f"   Focus on modules 0 and 5 for specialized processing")
    print()


def main():
    """Run all configuration examples"""
    print("🧠 SimpleBrainNetwork Configuration Examples")
    print("=" * 50)
    print()
    
    # Run all examples
    example_1_basic_configuration()
    example_2_attention_configuration()
    example_3_preset_configurations()
    example_4_research_configuration()
    example_5_performance_tuning()
    example_6_comparison()
    example_7_custom_attention()
    
    print("🎉 All examples completed!")
    print("\n💡 Key Takeaways:")
    print("   • No more hardcoded values - everything is configurable")
    print("   • Use presets for common scenarios")
    print("   • Create custom configs for specific research needs")
    print("   • Attention system is fully flexible")
    print("   • Performance and memory usage are tunable")
    print("   • Perfect for scientific experimentation!")


if __name__ == "__main__":
    main()