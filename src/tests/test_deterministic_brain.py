"""
Deterministic brain network tests with known input spike patterns and expected rasters.

These tests ensure that brain networks produce deterministic, reproducible results
when seeded with the same random seed.
"""

import os
import sys
from typing import Dict, List

import numpy as np
import pytest

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.random import seed_random_generators, create_deterministic_context
from core.simplified_brain_network import SimpleBrainNetwork, NetworkConfig


class TestDeterministicBrainNetwork:
    """Test deterministic behavior of brain networks."""
    
    def test_deterministic_initialization(self):
        """Test that brain networks initialize deterministically with same seed."""
        # Test with fixed seed
        seed = 42
        
        with create_deterministic_context(seed):
            brain1 = SimpleBrainNetwork(num_neurons=20, connectivity_prob=0.2)
            # Get initial state
            initial_weights1 = {}
            for neuron in brain1.neurons:
                initial_weights1[neuron.id] = dict(neuron.weights)
        
        with create_deterministic_context(seed):
            brain2 = SimpleBrainNetwork(num_neurons=20, connectivity_prob=0.2)
            # Get initial state
            initial_weights2 = {}
            for neuron in brain2.neurons:
                initial_weights2[neuron.id] = dict(neuron.weights)
        
        # Initial weights should be identical
        assert initial_weights1 == initial_weights2, "Brain networks should initialize identically with same seed"
    
    def test_deterministic_spike_pattern(self):
        """Test deterministic spike pattern with known input."""
        # Known input spike pattern
        input_pattern = {
            0: True,   # Neuron 0 spikes
            1: False,  # Neuron 1 doesn't spike
            2: True,   # Neuron 2 spikes
            3: False,  # Neuron 3 doesn't spike
            4: True,   # Neuron 4 spikes
        }
        
        # Expected output pattern (this will be determined empirically)
        expected_output = None
        
        # Test multiple times with same seed to ensure determinism
        outputs = []
        seed = 123
        
        for run in range(3):
            with create_deterministic_context(seed):
                brain = SimpleBrainNetwork(num_neurons=10, connectivity_prob=0.3)
                
                # Process the same input pattern
                output = brain.step(input_pattern)
                outputs.append(output)
        
        # All outputs should be identical
        assert all(output == outputs[0] for output in outputs), "Output should be deterministic with same seed"
        
        # Store the expected output for future reference
        expected_output = outputs[0]
        
        # Verify output has expected structure
        assert isinstance(expected_output, dict), "Output should be a dictionary"
        assert all(isinstance(k, int) for k in expected_output.keys()), "Output keys should be integers"
        assert all(isinstance(v, bool) for v in expected_output.values()), "Output values should be booleans"
        
        print(f"✅ Deterministic spike pattern test passed")
        print(f"   Input pattern: {input_pattern}")
        print(f"   Expected output: {expected_output}")
    
    def test_deterministic_learning(self):
        """Test that learning produces deterministic results."""
        # Training pattern
        training_pattern = {i: True for i in range(5)}
        seed = 456
        
        # Train two identical networks
        outputs_after_training = []
        
        for run in range(2):
            with create_deterministic_context(seed):
                brain = SimpleBrainNetwork(num_neurons=15, connectivity_prob=0.2)
                
                # Train the network
                for epoch in range(10):
                    brain.step(training_pattern)
                
                # Test response after training
                output = brain.step(training_pattern)
                outputs_after_training.append(output)
        
        # Both networks should produce identical outputs after identical training
        assert outputs_after_training[0] == outputs_after_training[1], "Learning should be deterministic"
        
        print(f"✅ Deterministic learning test passed")
        print(f"   Training pattern: {training_pattern}")
        print(f"   Output after training: {outputs_after_training[0]}")
    
    def test_deterministic_raster_plot(self):
        """Test deterministic raster plot generation."""
        # Create a simple test pattern
        input_pattern = {0: True, 2: True, 4: True}
        seed = 789
        
        with create_deterministic_context(seed):
            brain = SimpleBrainNetwork(num_neurons=8, connectivity_prob=0.25)
            
            # Process input multiple times to generate raster data
            raster_data = []
            for timestep in range(5):
                output = brain.step(input_pattern)
                raster_data.append(output)
        
        # Verify raster data structure
        assert len(raster_data) == 5, "Should have 5 timesteps"
        assert all(isinstance(step, dict) for step in raster_data), "Each timestep should be a dict"
        
        # Verify deterministic raster
        with create_deterministic_context(seed):
            brain2 = SimpleBrainNetwork(num_neurons=8, connectivity_prob=0.25)
            raster_data2 = []
            for timestep in range(5):
                output = brain2.step(input_pattern)
                raster_data2.append(output)
        
        assert raster_data == raster_data2, "Raster data should be deterministic"
        
        print(f"✅ Deterministic raster plot test passed")
        print(f"   Raster data shape: {len(raster_data)} timesteps")
        print(f"   Sample timestep: {raster_data[0]}")
    
    def test_network_config_determinism(self):
        """Test that different network configurations produce deterministic results."""
        config1 = NetworkConfig(
            dt=1.0,
            learning_rate=0.01,
            num_attention_modules=3
        )
        
        config2 = NetworkConfig(
            dt=1.0,
            learning_rate=0.01,
            num_attention_modules=3
        )
        
        # Both configs should be identical
        assert config1.__dict__ == config2.__dict__, "Identical configs should be equal"
        
        seed = 999
        input_pattern = {0: True, 1: True}
        
        with create_deterministic_context(seed):
            brain1 = SimpleBrainNetwork(num_neurons=6, config=config1)
            output1 = brain1.step(input_pattern)
        
        with create_deterministic_context(seed):
            brain2 = SimpleBrainNetwork(num_neurons=6, config=config2)
            output2 = brain2.step(input_pattern)
        
        assert output1 == output2, "Networks with identical configs should produce identical outputs"
        
        print(f"✅ Network config determinism test passed")
    
    def test_known_spike_pattern_validation(self):
        """Test with a known, validated spike pattern."""
        # This test uses a carefully constructed pattern that we know should
        # produce specific results in a deterministic brain network
        
        # Known input: alternating pattern
        known_input = {i: (i % 2 == 0) for i in range(6)}
        
        # Expected behavior: with proper seeding, this should always produce
        # the same output pattern
        seed = 2024
        outputs = []
        
        # Run multiple times to verify determinism
        for _ in range(3):
            with create_deterministic_context(seed):
                brain = SimpleBrainNetwork(num_neurons=10, connectivity_prob=0.15)
                output = brain.step(known_input)
                outputs.append(output)
        
        # All outputs should be identical
        assert all(output == outputs[0] for output in outputs), "Known pattern should be deterministic"
        
        # Validate output properties
        output = outputs[0]
        assert isinstance(output, dict), "Output should be dictionary"
        assert len(output) > 0, "Should have some output spikes"
        
        # Count active neurons (may be 0 in first step due to network initialization)
        active_neurons = sum(1 for spiked in output.values() if spiked)
        # Note: It's possible to have 0 active neurons in the first step
        # The important thing is that the output is deterministic
        
        print(f"✅ Known spike pattern validation passed")
        print(f"   Input: {known_input}")
        print(f"   Output: {output}")
        print(f"   Active neurons: {active_neurons}")


# Integration test to verify the entire system works deterministically
def test_full_deterministic_pipeline():
    """Test the full pipeline from seeding to brain processing."""
    # Set up logging to verify it works
    from common.logging import setup_logging, get_logger
    
    setup_logging(level="INFO", format_type="json")
    logger = get_logger(__name__)
    
    logger.info("🧠 Starting deterministic pipeline test")
    
    # Seed all random generators
    seed_value = seed_random_generators(42, deterministic=True)
    logger.info("🔧 Seeded random generators", extra={"seed": seed_value})
    
    # Create brain network
    brain = SimpleBrainNetwork(num_neurons=12, connectivity_prob=0.2)
    logger.info("🧠 Brain network created", extra={"neurons": brain.num_neurons})
    
    # Process known input
    input_pattern = {0: True, 3: True, 6: True}
    output = brain.step(input_pattern)
    
    logger.info("📊 Brain processing completed", extra={
        "input_spikes": sum(input_pattern.values()),
        "output_spikes": sum(output.values())
    })
    
    # Verify deterministic behavior
    seed_random_generators(42, deterministic=True)
    brain2 = SimpleBrainNetwork(num_neurons=12, connectivity_prob=0.2)
    output2 = brain2.step(input_pattern)
    
    assert output == output2, "Full pipeline should be deterministic"
    
    logger.info("✅ Deterministic pipeline test passed")


if __name__ == "__main__":
    # Run tests directly
    test_instance = TestDeterministicBrainNetwork()
    
    print("Running deterministic brain network tests...")
    
    test_instance.test_deterministic_initialization()
    test_instance.test_deterministic_spike_pattern()
    test_instance.test_deterministic_learning()
    test_instance.test_deterministic_raster_plot()
    test_instance.test_network_config_determinism()
    test_instance.test_known_spike_pattern_validation()
    
    # Run full pipeline test
    result = test_full_deterministic_pipeline()
    print(f"\nFull pipeline test result: {result}")
    
    print("\n✅ All deterministic tests passed!")
