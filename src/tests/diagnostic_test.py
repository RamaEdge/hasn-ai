"""
Quick diagnostic test to understand why networks aren't spiking
and ensure basic functionality works
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.brain_inspired_network import HierarchicalAdaptiveSpikingNetwork
from core.simplified_brain_network import SimpleBrainNetwork


def test_original_network():
    """Test original network with strong input"""
    print("Testing Original Network:")
    
    # Create small network
    modules_config = [(0, 10), (1, 10)]
    network = HierarchicalAdaptiveSpikingNetwork(modules_config)
    
    # Create strong input pattern
    def strong_input(t):
        return {0: {i: True for i in range(5)}}  # Strong constant input to first 5 neurons
    
    # Run short simulation
    results = network.run_simulation(duration=100.0, external_inputs_func=strong_input)
    
    total_spikes = len(results['spike_history'])
    print(f"  Total neurons: {sum(m.num_neurons for m in network.modules.values())}")
    print(f"  Total spikes: {total_spikes}")
    print(f"  Average activity: {np.mean(results['activity_history']):.4f}")
    
    # Check individual neuron states
    sample_neuron = network.modules[0].neurons[0]
    print(f"  Sample neuron voltage: {sample_neuron.v_membrane:.4f}")
    print(f"  Sample neuron threshold: {sample_neuron.threshold:.4f}")
    print(f"  Sample neuron connections: {len(sample_neuron.synapses)}")
    
    return results


def test_simplified_network():
    """Test simplified network with strong input"""
    print("\nTesting Simplified Network:")
    
    # Create equivalent network
    network = SimpleBrainNetwork(num_neurons=20, connectivity_prob=0.2)
    
    # Create strong input pattern
    def strong_input(t):
        return {i: True for i in range(5)}  # Strong constant input to first 5 neurons
    
    # Run short simulation
    results = network.run_simulation(duration=100.0, input_pattern_func=strong_input)
    
    total_spikes = len(results['spike_record'])
    print(f"  Total neurons: {network.num_neurons}")
    print(f"  Total spikes: {total_spikes}")
    print(f"  Average activity: {np.mean(results['activity_history']):.4f}")
    
    # Check individual neuron states
    sample_neuron = network.neurons[0]
    print(f"  Sample neuron voltage: {sample_neuron.voltage:.4f}")
    print(f"  Sample neuron threshold: {sample_neuron.params.threshold:.4f}")
    print(f"  Sample neuron connections: {len(sample_neuron.weights)}")
    
    return results


def analyze_connectivity():
    """Analyze network connectivity patterns"""
    print("\n" + "="*50)
    print("CONNECTIVITY ANALYSIS")
    print("="*50)
    
    # Original network connectivity
    modules_config = [(0, 5), (1, 5)]
    orig_network = HierarchicalAdaptiveSpikingNetwork(modules_config)
    
    total_connections = 0
    for module in orig_network.modules.values():
        for neuron in module.neurons:
            total_connections += len(neuron.synapses)
    
    print(f"Original network:")
    print(f"  Total connections: {total_connections}")
    print(f"  Connections per neuron: {total_connections / 10:.2f}")
    
    # Simplified network connectivity
    simp_network = SimpleBrainNetwork(num_neurons=10, connectivity_prob=0.1)
    
    total_connections = sum(len(neuron.weights) for neuron in simp_network.neurons)
    print(f"Simplified network:")
    print(f"  Total connections: {total_connections}")
    print(f"  Connections per neuron: {total_connections / 10:.2f}")


def test_single_neuron_dynamics():
    """Test individual neuron behavior"""
    print("\n" + "="*50)
    print("SINGLE NEURON DYNAMICS TEST")
    print("="*50)
    
    # Test original neuron
    from core.brain_inspired_network import AdaptiveSpikingNeuron, SpikingNeuronParams
    orig_params = SpikingNeuronParams()
    orig_neuron = AdaptiveSpikingNeuron(0, orig_params)
    
    print("Original neuron test:")
    for i in range(100):
        # Apply strong input
        input_spikes = {999: True}  # Fake input neuron
        orig_neuron.add_synapse(999, 50.0)  # Strong weight
        
        spiked = orig_neuron.update(0.1, i * 0.1, input_spikes)
        if spiked:
            print(f"  Spiked at time {i * 0.1:.1f}ms")
            break
        if i % 20 == 0:
            print(f"  t={i*0.1:.1f}ms: v={orig_neuron.v_membrane:.2f}mV, thresh={orig_neuron.threshold:.2f}mV")
    
    # Test simplified neuron
    from core.simplified_brain_network import SimpleSpikingNeuron, NeuronParams
    simp_params = NeuronParams()
    simp_neuron = SimpleSpikingNeuron(0, simp_params)
    
    print("\nSimplified neuron test:")
    for i in range(100):
        # Apply strong input
        input_spikes = {999: True}  # Fake input neuron
        simp_neuron.add_connection(999, 2.0)  # Strong weight
        
        spiked = simp_neuron.update(1.0, i * 1.0, input_spikes)
        if spiked:
            print(f"  Spiked at time {i * 1.0:.1f}ms")
            break
        if i % 20 == 0:
            print(f"  t={i*1.0:.1f}ms: v={simp_neuron.voltage:.2f}V, thresh={simp_neuron.params.threshold:.2f}V")


def main():
    print("=== BRAIN NETWORK DIAGNOSTIC TEST ===")
    
    # Test single neuron dynamics first
    test_single_neuron_dynamics()
    
    # Test network connectivity
    analyze_connectivity()
    
    # Test full networks
    print("\n" + "="*50)
    print("FULL NETWORK TESTS")
    print("="*50)
    
    orig_results = test_original_network()
    simp_results = test_simplified_network()
    
    # Summary
    print("\n" + "="*50)
    print("DIAGNOSTIC SUMMARY")
    print("="*50)
    
    orig_spikes = len(orig_results['spike_history'])
    simp_spikes = len(simp_results['spike_record'])
    
    print(f"Original network generated {orig_spikes} spikes")
    print(f"Simplified network generated {simp_spikes} spikes")
    
    if orig_spikes == 0 and simp_spikes == 0:
        print("❌ ISSUE: Neither network is generating spikes")
        print("   This suggests parameter or input issues")
    elif orig_spikes > 0 and simp_spikes > 0:
        print("✅ Both networks are functional")
    else:
        print("⚠️  Only one network is generating spikes")


if __name__ == "__main__":
    main()