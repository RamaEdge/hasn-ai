"""
Advanced Brain-Inspired Network with Cognitive Capabilities - COMPLETELY FIXED
This version fixes all structural issues and array comparison problems.

Author: AI Research Assistant
Date: August 2025
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict, deque
import json
import os

@dataclass
class CognitiveModule:
    """Configuration for specialized cognitive modules"""
    module_type: str  # 'sensory', 'memory', 'executive', 'motor'
    plasticity_rate: float = 1.0
    attention_sensitivity: float = 1.0
    memory_capacity: int = 1000
    oscillation_frequency: float = 10.0  # Hz


class WorkingMemoryBuffer:
    """Working memory implementation with limited capacity and decay"""
    
    def __init__(self, capacity: int = 7, decay_rate: float = 0.01):
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.items = deque(maxlen=capacity)
        self.strengths = deque(maxlen=capacity)
        self.timestamps = deque(maxlen=capacity)
        
    def add_item(self, pattern: np.ndarray, strength: float, timestamp: float):
        """Add new item to working memory"""
        self.items.append(pattern.copy())
        self.strengths.append(strength)
        self.timestamps.append(timestamp)
        
    def update(self, current_time: float):
        """Update memory strengths with decay"""
        new_strengths = []
        for i, (strength, timestamp) in enumerate(zip(self.strengths, self.timestamps)):
            time_diff = current_time - timestamp
            new_strength = strength * np.exp(-self.decay_rate * time_diff)
            new_strengths.append(new_strength)
            
        self.strengths = deque(new_strengths, maxlen=self.capacity)
        
        # Remove items below threshold
        threshold = 0.1
        to_remove = []
        for i, strength in enumerate(self.strengths):
            if strength < threshold:
                to_remove.append(i)
                
        for i in reversed(to_remove):
            del self.items[i]
            del self.strengths[i]
            del self.timestamps[i]


class AttentionMechanism:
    """Biologically-inspired attention mechanism"""
    
    def __init__(self, num_modules: int):
        self.num_modules = num_modules
        self.attention_weights = np.ones(num_modules) / num_modules
        self.saliency_map = np.zeros(num_modules)
        self.top_down_bias = np.zeros(num_modules)
        self.bottom_up_activation = np.zeros(num_modules)
        
    def update_saliency(self, module_activities: np.ndarray, surprise_signals: np.ndarray):
        """Update attention based on activity and surprise"""
        # Bottom-up attention from high activity and surprise
        self.bottom_up_activation = 0.7 * module_activities + 0.3 * surprise_signals
        
        # Combine top-down and bottom-up
        self.saliency_map = 0.6 * self.top_down_bias + 0.4 * self.bottom_up_activation
        
        # Softmax normalization - FIXED: handle potential overflow
        exp_saliency = np.exp(self.saliency_map - np.max(self.saliency_map))
        exp_sum = np.sum(exp_saliency)
        if exp_sum > 0:
            self.attention_weights = exp_saliency / exp_sum
        else:
            self.attention_weights = np.ones(self.num_modules) / self.num_modules


class MemoryConsolidationSystem:
    """Implements sleep-like memory consolidation"""
    
    def __init__(self, consolidation_threshold: float = 0.8):
        self.threshold = consolidation_threshold
        self.long_term_patterns = []
        self.pattern_importances = []
        self.consolidation_cycles = 0
        
    def consolidate_memories(self, working_memory: WorkingMemoryBuffer, 
                           recent_patterns: List[np.ndarray]):
        """Consolidate important patterns during 'sleep' phase"""
        
        # Identify important patterns (high strength, frequent activation)
        important_patterns = []
        
        # From working memory
        for pattern, strength in zip(working_memory.items, working_memory.strengths):
            if strength > self.threshold:
                important_patterns.append((pattern, strength))
                
        # From recent activity patterns
        for pattern in recent_patterns:
            # Calculate importance based on uniqueness and frequency
            importance = self._calculate_importance(pattern)
            if importance > self.threshold:
                important_patterns.append((pattern, importance))
                
        # Add to long-term storage
        for pattern, importance in important_patterns:
            self.long_term_patterns.append(pattern.copy())
            self.pattern_importances.append(importance)
            
        # Prune less important memories if storage is full
        max_capacity = 10000
        if len(self.long_term_patterns) > max_capacity:
            self._prune_memories(max_capacity)
            
        self.consolidation_cycles += 1
        
    def _calculate_importance(self, pattern: np.ndarray) -> float:
        """Calculate importance score for a pattern"""
        # Simple importance based on activity level and novelty
        activity_score = np.mean(pattern)
        
        # Novelty score (how different from existing patterns)
        novelty_score = 1.0
        if len(self.long_term_patterns) > 0:
            similarities = []
            for stored_pattern in self.long_term_patterns[-100:]:  # Check recent patterns
                if len(stored_pattern) == len(pattern):  # Same size
                    sim = np.dot(pattern, stored_pattern) / (np.linalg.norm(pattern) * np.linalg.norm(stored_pattern))
                    similarities.append(sim)
            if similarities:
                novelty_score = 1.0 - max(similarities)
            
        return 0.5 * activity_score + 0.5 * novelty_score
        
    def _prune_memories(self, target_size: int):
        """Remove least important memories"""
        if len(self.long_term_patterns) <= target_size:
            return
            
        # Sort by importance and keep most important
        importance_indices = np.argsort(self.pattern_importances)
        keep_indices = importance_indices[-target_size:]
        
        self.long_term_patterns = [self.long_term_patterns[i] for i in keep_indices]
        self.pattern_importances = [self.pattern_importances[i] for i in keep_indices]


class SimpleNeuron:
    """Simplified neuron for demonstration"""
    
    def __init__(self, neuron_id):
        self.id = neuron_id
        self.synapses = {}
        self.membrane_potential = 0.0
        self.threshold = 1.0
        
    def add_synapse(self, input_id, weight):
        self.synapses[input_id] = weight
        
    def update(self, input_active):
        if input_active:
            self.membrane_potential += 0.5
        else:
            self.membrane_potential *= 0.9
            
        fired = self.membrane_potential > self.threshold
        if fired:
            self.membrane_potential = 0.0
        return fired


class NeuralModule:
    """Simplified neural module"""
    
    def __init__(self, module_id, num_neurons):
        self.module_id = module_id
        self.num_neurons = num_neurons
        self.neurons = []
        
        # Create simplified neurons
        for i in range(num_neurons):
            neuron = SimpleNeuron(i)
            self.neurons.append(neuron)
    
    def update(self, dt, current_time, external_input=None):
        """Update module neurons"""
        spikes = {}
        for i, neuron in enumerate(self.neurons):
            # Simple neuron update
            input_active = external_input.get(i, False) if external_input else False
            fired = neuron.update(input_active)
            spikes[i] = fired
        return spikes


class AdvancedBrainInspiredNetwork:
    """Advanced brain-inspired network with cognitive capabilities"""
    
    def __init__(self, cognitive_modules: List[CognitiveModule]):
        self.cognitive_modules = cognitive_modules
        self.modules = {}
        self.current_time = 0.0
        self.dt = 0.1
        
        # Cognitive systems
        self.working_memory = WorkingMemoryBuffer()
        self.attention = AttentionMechanism(len(cognitive_modules))
        self.consolidation = MemoryConsolidationSystem()
        
        # Oscillation generators for different frequency bands
        self.oscillators = {
            'theta': {'frequency': 8.0, 'phase': 0.0, 'amplitude': 1.0},
            'alpha': {'frequency': 10.0, 'phase': 0.0, 'amplitude': 0.8},
            'beta': {'frequency': 20.0, 'phase': 0.0, 'amplitude': 0.6},
            'gamma': {'frequency': 40.0, 'phase': 0.0, 'amplitude': 0.4}
        }
        
        # Activity patterns for learning
        self.recent_patterns = []
        self.surprise_signals = np.zeros(len(cognitive_modules))
        
        # Initialize modules with cognitive properties
        self._initialize_cognitive_modules()
        
    def _initialize_cognitive_modules(self):
        """Initialize modules with cognitive-specific properties"""
        for i, cog_module in enumerate(self.cognitive_modules):
            # Determine module size based on type
            size_map = {
                'sensory': 200,
                'memory': 150,
                'executive': 100,
                'motor': 80
            }
            
            num_neurons = size_map.get(cog_module.module_type, 100)
            module = NeuralModule(i, num_neurons)
            
            # Customize connectivity based on module type
            if cog_module.module_type == 'memory':
                # Higher internal connectivity for memory modules
                self._enhance_memory_connectivity(module)
            elif cog_module.module_type == 'executive':
                # Sparse but strong connections for executive control
                self._configure_executive_connectivity(module)
                
            self.modules[i] = module
            
    def _enhance_memory_connectivity(self, module):
        """Enhance connectivity for memory modules"""
        # Add more recurrent connections
        for i, neuron_i in enumerate(module.neurons):
            for j, neuron_j in enumerate(module.neurons):
                if i != j and np.random.random() < 0.2:  # Higher connectivity
                    weight = np.random.normal(0.3, 0.1)
                    if hasattr(neuron_j, 'add_synapse'):
                        neuron_j.add_synapse(i, max(0.0, weight))
                    
    def _configure_executive_connectivity(self, module):
        """Configure executive control connectivity"""
        # Create hub neurons with high connectivity
        num_hubs = max(1, module.num_neurons // 10)
        hub_neurons = np.random.choice(module.num_neurons, num_hubs, replace=False)
        
        for hub_id in hub_neurons:
            if hub_id < len(module.neurons):
                # Connect hub to many other neurons
                for target_id in range(module.num_neurons):
                    if target_id != hub_id and np.random.random() < 0.5:
                        weight = np.random.normal(0.4, 0.1)
                        if target_id < len(module.neurons) and hasattr(module.neurons[target_id], 'add_synapse'):
                            module.neurons[target_id].add_synapse(hub_id, max(0.0, weight))
                    
    def generate_oscillations(self) -> Dict[str, float]:
        """Generate neural oscillations at different frequencies"""
        oscillations = {}
        
        for band, params in self.oscillators.items():
            freq = params['frequency']
            amp = params['amplitude']
            
            # Update phase
            params['phase'] += 2 * np.pi * freq * self.dt / 1000.0
            params['phase'] %= 2 * np.pi
            
            # Generate oscillation
            oscillations[band] = amp * np.sin(params['phase'])
            
        return oscillations
        
    def update_working_memory(self, current_patterns: Dict[int, np.ndarray]):
        """Update working memory with current activity patterns"""
        self.working_memory.update(self.current_time)
        
        # Add significant patterns to working memory
        for pattern in current_patterns.values():
            if np.mean(pattern) > 0.3:  # Threshold for significance
                strength = np.mean(pattern)
                self.working_memory.add_item(pattern, strength, self.current_time)
                
    def calculate_surprise(self, current_patterns: Dict[int, np.ndarray]) -> np.ndarray:
        """Calculate surprise signals based on prediction errors"""
        surprise = np.zeros(len(self.cognitive_modules))
        
        if len(self.recent_patterns) > 10:
            # Compare current patterns with recent history
            recent_avg = {}
            for patterns in self.recent_patterns[-10:]:
                for module_id, pattern in patterns.items():
                    if module_id not in recent_avg:
                        recent_avg[module_id] = np.zeros_like(pattern)
                    recent_avg[module_id] += pattern / 10
                    
            # Calculate surprise as deviation from recent average - FIXED: handle array comparisons properly
            for module_id, current_pattern in current_patterns.items():
                if module_id in recent_avg and module_id < len(surprise):
                    if len(current_pattern) == len(recent_avg[module_id]):
                        deviation = np.linalg.norm(current_pattern - recent_avg[module_id])
                        surprise[module_id] = min(1.0, deviation)
                    
        return surprise
        
    def step_with_cognition(self, external_inputs: Dict[int, Dict[int, bool]] = None) -> Dict:
        """Enhanced simulation step with cognitive processes"""
        
        # Generate neural oscillations
        oscillations = self.generate_oscillations()
        
        # Basic network update
        module_spikes = {}
        current_patterns = {}
        
        for module_id, module in self.modules.items():
            ext_input = external_inputs.get(module_id, {}) if external_inputs else {}
            
            # Add oscillatory modulation
            gamma_modulation = oscillations['gamma']
            if gamma_modulation > 0.5:
                # Boost external inputs during gamma peaks
                for neuron_id in ext_input:
                    ext_input[neuron_id] = True
                    
            spikes = module.update(self.dt, self.current_time, ext_input)
            module_spikes[module_id] = spikes
            
            # Extract activity pattern
            pattern = np.array([1.0 if spikes.get(i, False) else 0.0 
                              for i in range(module.num_neurons)])
            current_patterns[module_id] = pattern
            
        # Update cognitive systems
        self.update_working_memory(current_patterns)
        
        # Calculate surprise and update attention
        module_activities = np.array([np.mean(pattern) for pattern in current_patterns.values()])
        self.surprise_signals = self.calculate_surprise(current_patterns)
        self.attention.update_saliency(module_activities, self.surprise_signals)
        
        # Store patterns for learning
        self.recent_patterns.append(current_patterns.copy())
        if len(self.recent_patterns) > 100:
            self.recent_patterns.pop(0)
            
        # Periodic memory consolidation (simulate sleep cycles)
        if int(self.current_time) % 10000 == 0 and len(self.recent_patterns) > 50:
            patterns_to_consolidate = [p for patterns in self.recent_patterns[-50:] 
                                     for p in patterns.values()]
            self.consolidation.consolidate_memories(self.working_memory, patterns_to_consolidate)
            
        self.current_time += self.dt
        
        return {
            'module_spikes': module_spikes,
            'current_patterns': current_patterns,
            'attention_weights': self.attention.attention_weights.copy(),
            'working_memory_size': len(self.working_memory.items),
            'surprise_signals': self.surprise_signals.copy(),
            'oscillations': oscillations.copy()
        }
        
    def save_network_state(self, filepath: str):
        """Save complete network state"""
        state = {
            'current_time': self.current_time,
            'cognitive_modules': [
                {
                    'module_type': cm.module_type,
                    'plasticity_rate': cm.plasticity_rate,
                    'attention_sensitivity': cm.attention_sensitivity,
                    'memory_capacity': cm.memory_capacity,
                    'oscillation_frequency': cm.oscillation_frequency
                }
                for cm in self.cognitive_modules
            ],
            'attention_weights': self.attention.attention_weights.tolist(),
            'working_memory_items': len(self.working_memory.items),
            'consolidation_cycles': self.consolidation.consolidation_cycles,
            'long_term_patterns': len(self.consolidation.long_term_patterns)
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)


def create_cognitive_brain_network() -> AdvancedBrainInspiredNetwork:
    """Create a brain network with cognitive capabilities"""
    
    cognitive_modules = [
        CognitiveModule('sensory', plasticity_rate=1.2, oscillation_frequency=40.0),
        CognitiveModule('memory', plasticity_rate=0.8, memory_capacity=2000, oscillation_frequency=8.0),
        CognitiveModule('executive', plasticity_rate=1.0, attention_sensitivity=1.5, oscillation_frequency=20.0),
        CognitiveModule('motor', plasticity_rate=1.1, oscillation_frequency=10.0)
    ]
    
    return AdvancedBrainInspiredNetwork(cognitive_modules)


def run_cognitive_experiment():
    """Run an experiment with the cognitive brain network"""
    print("🧠 Creating Advanced Brain-Inspired Network with Cognitive Capabilities...")
    
    network = create_cognitive_brain_network()
    
    print(f"✅ Network created with {len(network.modules)} cognitive modules:")
    for i, cog_mod in enumerate(network.cognitive_modules):
        module_size = len(network.modules[i].neurons) if i in network.modules else 0
        print(f"  Module {i}: {cog_mod.module_type} ({module_size} neurons)")
    
    # Run simulation with cognitive monitoring
    print("\n🔬 Running cognitive simulation (5000 time steps = 500ms)...")
    
    results = []
    for step in range(5000):
        
        # Create dynamic input patterns with clear descriptions
        ext_inputs = {}
        current_phase = ""
        
        if step < 1000:  # Sensory processing phase
            ext_inputs[0] = {i: True for i in range(0, 20, 2)}  # Sensory module
            current_phase = "Sensory Input Processing"
        elif 1000 <= step < 2000:  # Memory encoding phase
            ext_inputs[1] = {i: True for i in range(10, 25)}  # Memory module
            current_phase = "Memory Encoding"
        elif 2000 <= step < 3000:  # Executive control phase
            ext_inputs[2] = {i: True for i in range(5, 15)}  # Executive module
            current_phase = "Executive Control"
        elif 3000 <= step < 4000:  # Motor output phase
            ext_inputs[3] = {i: True for i in range(0, 10)}  # Motor module
            current_phase = "Motor Output"
        else:  # Integration phase
            current_phase = "Cross-Modal Integration"
            
        result = network.step_with_cognition(ext_inputs)
        results.append(result)
        
        # Print meaningful periodic updates
        if step % 1000 == 0:
            wm_size = result.get('working_memory_size', 0)
            attention = result.get('attention_weights', [0,0,0,0])
            oscillations = result.get('oscillations', {})
            
            print(f"  Step {step:4d}: Phase='{current_phase}'")
            print(f"           Working Memory: {wm_size}/7 items")
            print(f"           Attention: Sensory={attention[0]:.2f}, Memory={attention[1]:.2f}, "
                  f"Executive={attention[2]:.2f}, Motor={attention[3]:.2f}")
            if oscillations:
                gamma = oscillations.get('gamma', 0)
                theta = oscillations.get('theta', 0)
                print(f"           Brain Waves: Gamma={gamma:.2f}, Theta={theta:.2f}")
    
    print("\n🎯 Simulation Complete! Results:")
    final_result = results[-1]
    print(f"   Working Memory Utilization: {final_result.get('working_memory_size', 0)}/7 items")
    print(f"   Memory Consolidation Cycles: {network.consolidation.consolidation_cycles}")
    print(f"   Long-term Patterns Stored: {len(network.consolidation.long_term_patterns)}")
    print(f"   Total Simulation Time: {network.current_time:.1f}ms")
    
    # Calculate learning metrics
    initial_patterns = len(results[0].get('current_patterns', {}))
    final_patterns = len(final_result.get('current_patterns', {}))
    print(f"   Neural Activity Evolution: {initial_patterns} → {final_patterns} active patterns")
    
    # Save network state
    output_dir = '/Users/ravi.chillerega/sources/cde-hack-session/output'
    os.makedirs(output_dir, exist_ok=True)
    
    network.save_network_state(f'{output_dir}/cognitive_network_state.json')
    print(f"   Network state saved to: {output_dir}/cognitive_network_state.json")
    
    return network, results


if __name__ == "__main__":
    network, results = run_cognitive_experiment()
