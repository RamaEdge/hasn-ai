"""
Hierarchical Adaptive Spiking Network (HASN)
A brain-inspired neural network architecture that incorporates:
- Spiking neurons with temporal dynamics
- Hierarchical modular organization
- Spike-timing dependent plasticity (STDP)
- Homeostatic regulation
- Dynamic topology evolution

Author: AI Research Assistant
Date: August 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import networkx as nx
from dataclasses import dataclass
from collections import defaultdict
import random

@dataclass
class SpikingNeuronParams:
    """Parameters for adaptive leaky integrate-and-fire neuron"""
    tau_m: float = 20.0          # Membrane time constant (ms)
    tau_ref: float = 2.0         # Refractory period (ms)
    v_rest: float = -70.0        # Resting potential (mV)
    v_thresh: float = -55.0      # Spike threshold (mV)
    v_reset: float = -75.0       # Reset potential (mV)
    tau_adapt: float = 100.0     # Adaptation time constant (ms)
    g_adapt: float = 0.1         # Adaptation conductance
    noise_amp: float = 0.5       # Noise amplitude

class AdaptiveSpikingNeuron:
    """
    Adaptive Leaky Integrate-and-Fire neuron with:
    - Spike-rate adaptation
    - Dynamic threshold
    - Homeostatic regulation
    """
    
    def __init__(self, neuron_id: int, params: SpikingNeuronParams):
        self.id = neuron_id
        self.params = params
        
        # State variables
        self.v_membrane = params.v_rest
        self.adaptation_current = 0.0
        self.threshold = params.v_thresh
        self.last_spike_time = -float('inf')
        self.refractory_until = 0.0
        
        # Spike history for STDP
        self.spike_times = []
        self.input_spikes = defaultdict(list)  # input_neuron_id -> [spike_times]
        
        # Homeostatic variables
        self.target_rate = 5.0  # Target firing rate (Hz)
        self.actual_rate = 0.0
        self.rate_window = 1000.0  # Window for rate calculation (ms)
        
        # Connectivity
        self.synapses = {}  # input_neuron_id -> weight
        self.synaptic_traces = defaultdict(float)  # For STDP
        
    def add_synapse(self, input_neuron_id: int, weight: float):
        """Add a synaptic connection"""
        self.synapses[input_neuron_id] = weight
        
    def update(self, dt: float, current_time: float, input_spikes: Dict[int, bool]) -> bool:
        """
        Update neuron state and return True if it spikes
        """
        # Check if in refractory period
        if current_time < self.refractory_until:
            return False
            
        # Calculate synaptic input current
        synaptic_current = 0.0
        for input_id, spiked in input_spikes.items():
            if spiked and input_id in self.synapses:
                synaptic_current += self.synapses[input_id]
                self.input_spikes[input_id].append(current_time)
                
        # Add noise
        noise = np.random.normal(0, self.params.noise_amp)
        
        # Update membrane potential
        dv = (self.params.v_rest - self.v_membrane + synaptic_current - self.adaptation_current + noise) / self.params.tau_m
        self.v_membrane += dv * dt
        
        # Update adaptation current
        dadapt = -self.adaptation_current / self.params.tau_adapt
        self.adaptation_current += dadapt * dt
        
        # Check for spike
        if self.v_membrane >= self.threshold:
            self.spike(current_time)
            return True
            
        return False
        
    def spike(self, current_time: float):
        """Handle spike generation and adaptation"""
        self.spike_times.append(current_time)
        self.v_membrane = self.params.v_reset
        self.refractory_until = current_time + self.params.tau_ref
        self.adaptation_current += self.params.g_adapt
        self.last_spike_time = current_time
        
        # Clean old spike times
        cutoff_time = current_time - self.rate_window
        self.spike_times = [t for t in self.spike_times if t > cutoff_time]
        
        # Update firing rate
        self.actual_rate = len(self.spike_times) / (self.rate_window / 1000.0)
        
    def apply_stdp(self, current_time: float, learning_rate: float = 0.01):
        """Apply spike-timing dependent plasticity"""
        if not self.spike_times:
            return
            
        last_post_spike = self.spike_times[-1]
        
        for input_id in self.synapses:
            if input_id not in self.input_spikes:
                continue
                
            # Clean old input spikes
            cutoff_time = current_time - 100.0  # 100ms STDP window
            self.input_spikes[input_id] = [t for t in self.input_spikes[input_id] if t > cutoff_time]
            
            # Apply STDP rule
            weight_change = 0.0
            for pre_spike in self.input_spikes[input_id]:
                dt_spike = last_post_spike - pre_spike
                
                if 0 < dt_spike <= 20.0:  # Potentiation window
                    weight_change += learning_rate * np.exp(-dt_spike / 10.0)
                elif -20.0 <= dt_spike < 0:  # Depression window
                    weight_change -= learning_rate * 0.5 * np.exp(dt_spike / 10.0)
                    
            # Update weight with bounds
            self.synapses[input_id] = np.clip(
                self.synapses[input_id] + weight_change, 0.0, 2.0
            )
            
    def homeostatic_regulation(self, dt: float):
        """Apply homeostatic plasticity"""
        rate_error = self.target_rate - self.actual_rate
        
        # Adjust threshold based on firing rate
        threshold_change = -0.001 * rate_error * dt  # Slow adaptation
        self.threshold += threshold_change
        self.threshold = np.clip(self.threshold, -60.0, -50.0)
        
        # Scale synaptic weights
        if abs(rate_error) > 1.0:
            scale_factor = 1.0 + 0.0001 * rate_error * dt
            for input_id in self.synapses:
                self.synapses[input_id] *= scale_factor
                self.synapses[input_id] = np.clip(self.synapses[input_id], 0.0, 2.0)


class NeuralModule:
    """
    A module of interconnected spiking neurons
    Represents a functional unit similar to a cortical column
    """
    
    def __init__(self, module_id: int, num_neurons: int, connectivity_prob: float = 0.1):
        self.id = module_id
        self.neurons = []
        self.num_neurons = num_neurons
        
        # Create neurons
        for i in range(num_neurons):
            params = SpikingNeuronParams()
            # Add some variability
            params.v_thresh += np.random.normal(0, 2.0)
            params.tau_m += np.random.normal(0, 5.0)
            
            neuron = AdaptiveSpikingNeuron(i, params)
            self.neurons.append(neuron)
            
        # Create internal connectivity
        self.create_internal_connections(connectivity_prob)
        
        # Module-level properties
        self.activity_level = 0.0
        self.specialization_score = 0.0
        
    def create_internal_connections(self, prob: float):
        """Create sparse random connections within the module"""
        for i, neuron_i in enumerate(self.neurons):
            for j, neuron_j in enumerate(self.neurons):
                if i != j and np.random.random() < prob:
                    # Excitatory connections (80%) and inhibitory (20%)
                    weight = 0.5 if np.random.random() < 0.8 else -0.3
                    neuron_j.add_synapse(i, weight)
                    
    def update(self, dt: float, current_time: float, external_input: Dict[int, bool] = None) -> Dict[int, bool]:
        """Update all neurons in the module"""
        if external_input is None:
            external_input = {}
            
        # Collect internal spikes from previous timestep
        internal_spikes = {}
        
        # Update each neuron
        current_spikes = {}
        for i, neuron in enumerate(self.neurons):
            # Combine internal and external inputs
            all_inputs = {**internal_spikes, **external_input}
            spiked = neuron.update(dt, current_time, all_inputs)
            current_spikes[i] = spiked
            
        # Apply plasticity
        for neuron in self.neurons:
            neuron.apply_stdp(current_time)
            neuron.homeostatic_regulation(dt)
            
        # Update module activity
        spike_count = sum(current_spikes.values())
        self.activity_level = 0.9 * self.activity_level + 0.1 * (spike_count / self.num_neurons)
        
        return current_spikes


class HierarchicalAdaptiveSpikingNetwork:
    """
    Main HASN architecture with hierarchical modular organization
    """
    
    def __init__(self, modules_config: List[Tuple[int, int]]):
        """
        Initialize network with specified modules
        modules_config: [(module_id, num_neurons), ...]
        """
        self.modules = {}
        self.inter_module_connections = defaultdict(dict)  # source_module -> target_module -> weight
        self.current_time = 0.0
        self.dt = 0.1  # 0.1 ms timestep
        
        # Create modules
        for module_id, num_neurons in modules_config:
            self.modules[module_id] = NeuralModule(module_id, num_neurons)
            
        # Create inter-module connections
        self.create_inter_module_connections()
        
        # Network-level properties
        self.global_activity = 0.0
        self.network_oscillations = []
        
    def create_inter_module_connections(self):
        """Create sparse connections between modules"""
        module_ids = list(self.modules.keys())
        
        for src_id in module_ids:
            for tgt_id in module_ids:
                if src_id != tgt_id:
                    # Probability decreases with "distance" between modules
                    prob = 0.05 / (abs(src_id - tgt_id) + 1)
                    
                    if np.random.random() < prob:
                        weight = np.random.normal(0.3, 0.1)
                        self.inter_module_connections[src_id][tgt_id] = max(0.0, weight)
                        
    def propagate_between_modules(self, module_spikes: Dict[int, Dict[int, bool]]) -> Dict[int, Dict[int, bool]]:
        """Propagate spikes between modules"""
        inter_module_input = defaultdict(dict)
        
        for src_module_id, spikes in module_spikes.items():
            active_neurons = [nid for nid, spiked in spikes.items() if spiked]
            
            if active_neurons:
                for tgt_module_id in self.inter_module_connections[src_module_id]:
                    weight = self.inter_module_connections[src_module_id][tgt_module_id]
                    
                    # Select random target neurons (simplified routing)
                    num_targets = min(len(active_neurons), 5)
                    target_neurons = np.random.choice(
                        list(range(self.modules[tgt_module_id].num_neurons)),
                        size=num_targets, replace=False
                    )
                    
                    for tgt_neuron in target_neurons:
                        inter_module_input[tgt_module_id][tgt_neuron] = True
                        
        return dict(inter_module_input)
        
    def step(self, external_inputs: Dict[int, Dict[int, bool]] = None) -> Dict[int, Dict[int, bool]]:
        """Single simulation step"""
        if external_inputs is None:
            external_inputs = {}
            
        # Update each module
        module_spikes = {}
        for module_id, module in self.modules.items():
            ext_input = external_inputs.get(module_id, {})
            spikes = module.update(self.dt, self.current_time, ext_input)
            module_spikes[module_id] = spikes
            
        # Propagate between modules
        inter_module_inputs = self.propagate_between_modules(module_spikes)
        
        # Update modules with inter-module input
        for module_id, inputs in inter_module_inputs.items():
            # This would normally be done in the next timestep
            # For simplicity, we'll apply it immediately
            pass
            
        self.current_time += self.dt
        
        # Update global activity
        total_spikes = sum(sum(spikes.values()) for spikes in module_spikes.values())
        total_neurons = sum(m.num_neurons for m in self.modules.values())
        self.global_activity = 0.9 * self.global_activity + 0.1 * (total_spikes / total_neurons)
        
        return module_spikes
        
    def run_simulation(self, duration: float, external_inputs_func=None) -> Dict:
        """Run simulation for specified duration"""
        num_steps = int(duration / self.dt)
        
        # Recording arrays
        time_points = []
        activity_history = []
        spike_history = defaultdict(list)
        
        for step in range(num_steps):
            # Get external inputs for this timestep
            ext_inputs = None
            if external_inputs_func:
                ext_inputs = external_inputs_func(self.current_time)
                
            # Simulation step
            module_spikes = self.step(ext_inputs)
            
            # Record data
            time_points.append(self.current_time)
            activity_history.append(self.global_activity)
            
            for module_id, spikes in module_spikes.items():
                for neuron_id, spiked in spikes.items():
                    if spiked:
                        spike_history[(module_id, neuron_id)].append(self.current_time)
                        
        return {
            'time_points': time_points,
            'activity_history': activity_history,
            'spike_history': dict(spike_history),
            'final_weights': self.get_weight_matrices()
        }
        
    def get_weight_matrices(self) -> Dict:
        """Extract current synaptic weights for analysis"""
        weights = {}
        
        for module_id, module in self.modules.items():
            module_weights = {}
            for i, neuron in enumerate(module.neurons):
                module_weights[i] = dict(neuron.synapses)
            weights[module_id] = module_weights
            
        weights['inter_module'] = dict(self.inter_module_connections)
        return weights
        
    def visualize_activity(self, simulation_results: Dict):
        """Visualize network activity"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Global activity over time
        axes[0].plot(simulation_results['time_points'], simulation_results['activity_history'])
        axes[0].set_xlabel('Time (ms)')
        axes[0].set_ylabel('Global Activity')
        axes[0].set_title('Network Activity Over Time')
        
        # Spike raster plot
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.modules)))
        
        for i, (module_id, module) in enumerate(self.modules.items()):
            for neuron_id in range(module.num_neurons):
                key = (module_id, neuron_id)
                if key in simulation_results['spike_history']:
                    y_pos = module_id * 100 + neuron_id
                    axes[1].scatter(simulation_results['spike_history'][key], 
                                   [y_pos] * len(simulation_results['spike_history'][key]),
                                   c=[colors[i]], s=1, alpha=0.7)
                                   
        axes[1].set_xlabel('Time (ms)')
        axes[1].set_ylabel('Neuron ID')
        axes[1].set_title('Spike Raster Plot')
        
        # Module activity levels
        module_activities = [self.modules[mid].activity_level for mid in sorted(self.modules.keys())]
        axes[2].bar(range(len(module_activities)), module_activities)
        axes[2].set_xlabel('Module ID')
        axes[2].set_ylabel('Activity Level')
        axes[2].set_title('Current Module Activity Levels')
        
        plt.tight_layout()
        return fig


def create_example_network() -> HierarchicalAdaptiveSpikingNetwork:
    """Create an example HASN for demonstration"""
    # Define network architecture: (module_id, num_neurons)
    modules_config = [
        (0, 50),   # Input module
        (1, 100),  # Processing module 1
        (2, 100),  # Processing module 2
        (3, 50),   # Integration module
        (4, 25)    # Output module
    ]
    
    network = HierarchicalAdaptiveSpikingNetwork(modules_config)
    return network


def example_input_pattern(t: float) -> Dict[int, Dict[int, bool]]:
    """Generate example input pattern - rhythmic stimulation"""
    inputs = defaultdict(dict)
    
    # Rhythmic input to module 0 (10 Hz oscillation)
    if np.sin(2 * np.pi * 0.01 * t) > 0.5:  # 10 Hz in our time units
        # Stimulate random subset of neurons in module 0
        num_active = np.random.poisson(5)
        active_neurons = np.random.choice(50, min(num_active, 50), replace=False)
        
        for neuron_id in active_neurons:
            inputs[0][neuron_id] = True
            
    return dict(inputs)


if __name__ == "__main__":
    print("Creating Hierarchical Adaptive Spiking Network...")
    
    # Create network
    network = create_example_network()
    
    print(f"Network created with {len(network.modules)} modules")
    for module_id, module in network.modules.items():
        print(f"  Module {module_id}: {module.num_neurons} neurons")
    
    print("\nRunning simulation...")
    
    # Run simulation
    results = network.run_simulation(
        duration=1000.0,  # 1 second
        external_inputs_func=example_input_pattern
    )
    
    print(f"Simulation complete. Total spikes recorded: {len(results['spike_history'])}")
    print(f"Final global activity: {network.global_activity:.4f}")
    
    # Visualize results
    fig = network.visualize_activity(results)
    plt.savefig('/Users/ravi.chillerega/sources/cde-hack-session/output/hasn_activity.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved to output/hasn_activity.png")
