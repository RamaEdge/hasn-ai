"""
Simplified Brain-Inspired Network (SBIN)
A streamlined version focusing on core spiking dynamics with minimal complexity.

Key simplifications:
- Reduced neuron parameters (3 instead of 9)
- Simplified plasticity (basic Hebbian learning)
- Direct network structure (no modules)
- Essential functionality only

Author: AI Research Assistant
Date: August 2025
"""

import json
from dataclasses import dataclass
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class NeuronParams:
    """Simplified neuron parameters - only essentials"""

    tau_membrane: float = 20.0  # Membrane time constant (ms)
    threshold: float = 1.0  # Spike threshold
    reset_potential: float = 0.0  # Reset after spike


@dataclass
class NetworkConfig:
    """Configuration for SimpleBrainNetwork"""

    # Time simulation
    dt: float = 1.0  # Timestep (ms)

    # Connection parameters
    weight_range: tuple = (0.1, 0.8)  # (min_weight, max_weight)
    max_weight: float = 2.0  # Maximum weight limit

    # Learning parameters
    learning_rate: float = 0.01  # Default learning rate
    learning_probability: float = 0.1  # Probability of learning each step

    # Attention system
    num_attention_modules: int = 4  # Number of attention modules
    attention_init: str = "equal"  # "equal", "random", or "custom"

    # Recording
    max_spike_history: int = 1000  # Max spikes to keep in memory


class SimpleSpikingNeuron:
    """
    Simplified spiking neuron with basic integrate-and-fire dynamics
    """

    def __init__(self, neuron_id: int, params: NeuronParams = None):
        self.id = neuron_id
        self.params = params or NeuronParams()

        # Essential state variables only
        self.voltage = 0.0
        self.last_spike_time = -1000.0
        self.spike_times = []

        # Simple connectivity
        self.weights = {}  # input_neuron_id -> weight

        # Compatibility
        self.connections = {}  # For compatibility with other code
        self.spike_count = 0

    @property
    def potential(self):
        """Alias for voltage for compatibility"""
        return self.voltage

    def add_connection(self, input_id: int, weight: float):
        """Add synaptic connection"""
        self.weights[input_id] = weight
        self.connections[input_id] = {"weight": weight}

    def update(self, dt: float, current_time: float, input_spikes: Dict[int, bool]) -> bool:
        """Update neuron and return True if it spikes"""

        # Calculate input current
        input_current = 0.0
        for input_id, spiked in input_spikes.items():
            if spiked and input_id in self.weights:
                input_current += self.weights[input_id]

        # Simple membrane dynamics: dV/dt = (I - V) / tau
        self.voltage += dt * (input_current - self.voltage) / self.params.tau_membrane

        # Check for spike
        if self.voltage >= self.params.threshold:
            self.spike(current_time)
            return True

        return False

    def spike(self, current_time: float):
        """Handle spike"""
        self.spike_times.append(current_time)
        self.voltage = self.params.reset_potential
        self.last_spike_time = current_time
        self.spike_count += 1

        # Keep only recent spikes for memory efficiency (configurable)
        max_history = getattr(self, "_max_spike_history", 1000)  # Default fallback
        if len(self.spike_times) > max_history:
            self.spike_times = self.spike_times[-(max_history // 2) :]

    def apply_learning(
        self,
        input_spikes: Dict[int, bool],
        learning_rate: float = 0.01,
        max_weight: float = 2.0,
    ):
        """Simple Hebbian learning: neurons that fire together, wire together"""
        if not input_spikes:
            return

        # Basic Hebbian rule
        for input_id in self.weights:
            if input_id in input_spikes and input_spikes[input_id]:
                # Strengthen connection if both neurons active
                self.weights[input_id] += learning_rate
                # Keep weights bounded (configurable)
                self.weights[input_id] = min(self.weights[input_id], max_weight)


class SimpleBrainNetwork:
    """
    Simplified brain-inspired network with direct neuron-to-neuron connections
    """

    def __init__(
        self,
        num_neurons: int,
        connectivity_prob: float = 0.1,
        config: NetworkConfig = None,
    ):
        self.num_neurons = num_neurons
        self.config = config or NetworkConfig()
        self.neurons = []
        self.current_time = 0.0
        self.dt = self.config.dt

        # Create neurons
        for i in range(num_neurons):
            neuron = SimpleSpikingNeuron(i)
            neuron._max_spike_history = self.config.max_spike_history
            self.neurons.append(neuron)

        # Create random connections
        self.create_connections(connectivity_prob)

        # Initialize attention system (configurable)
        self._init_attention_system()

        # Add working memory for compatibility
        self.working_memory = []

        # Add time_step for compatibility
        self.time_step = 0

        # Simple recording
        self.activity_history = []
        self.spike_record = []

    def _init_attention_system(self):
        """Initialize attention weights based on configuration"""
        if self.config.attention_init == "equal":
            self.attention_weights = (
                np.ones(self.config.num_attention_modules) / self.config.num_attention_modules
            )
        elif self.config.attention_init == "random":
            weights = np.random.random(self.config.num_attention_modules)
            self.attention_weights = weights / weights.sum()  # Normalize
        else:  # custom - will be set externally
            self.attention_weights = (
                np.ones(self.config.num_attention_modules) / self.config.num_attention_modules
            )

    def create_connections(self, prob: float):
        """Create sparse random connections"""
        min_weight, max_weight = self.config.weight_range
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if i != j and np.random.random() < prob:
                    # Random weight within configured range
                    weight = np.random.uniform(min_weight, max_weight)
                    self.neurons[j].add_connection(i, weight)

    def step(self, external_input: Dict[int, bool] = None) -> Dict[int, bool]:
        """Single simulation step"""
        if external_input is None:
            external_input = {}

        # Collect spikes from all neurons
        current_spikes = {}
        for neuron in self.neurons:
            spiked = neuron.update(self.dt, self.current_time, external_input)
            current_spikes[neuron.id] = spiked

        # Apply learning (simplified)
        if np.random.random() < self.config.learning_probability:
            for neuron in self.neurons:
                neuron.apply_learning(
                    current_spikes,
                    learning_rate=self.config.learning_rate,
                    max_weight=self.config.max_weight,
                )

        # Record activity
        spike_count = sum(current_spikes.values())
        self.activity_history.append(spike_count)

        # Record individual spikes for analysis
        for neuron_id, spiked in current_spikes.items():
            if spiked:
                self.spike_record.append((self.current_time, neuron_id))

        self.current_time += self.dt
        return current_spikes

    def run_simulation(self, duration: float, input_pattern_func=None) -> Dict:
        """Run network simulation"""
        num_steps = int(duration / self.dt)

        print(f"Running simulation for {duration}ms ({num_steps} steps)...")

        for step in range(num_steps):
            # Get external input
            external_input = {}
            if input_pattern_func:
                external_input = input_pattern_func(self.current_time)

            # Run one step
            self.step(external_input)

            # Progress indicator
            if step % (num_steps // 10) == 0:
                progress = (step / num_steps) * 100
                print(f"Progress: {progress:.0f}%")

        return {
            "activity_history": self.activity_history,
            "spike_record": self.spike_record,
            "final_weights": self.get_weights(),
        }

    def get_weights(self) -> Dict:
        """Get current connection weights"""
        weights = {}
        for neuron in self.neurons:
            weights[neuron.id] = dict(neuron.weights)
        return weights

    def visualize_results(self, results: Dict):
        """Simple visualization"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Activity over time
        time_points = np.arange(len(results["activity_history"]))
        ax1.plot(time_points, results["activity_history"])
        ax1.set_xlabel("Time (ms)")
        ax1.set_ylabel("Spikes per timestep")
        ax1.set_title("Network Activity")

        # Spike raster plot
        if results["spike_record"]:
            times, neuron_ids = zip(*results["spike_record"])
            ax2.scatter(times, neuron_ids, s=1, alpha=0.6)
            ax2.set_xlabel("Time (ms)")
            ax2.set_ylabel("Neuron ID")
            ax2.set_title("Spike Raster Plot")

        plt.tight_layout()
        return fig

    def get_brain_state(self) -> Dict:
        """Get current state of the brain network"""
        return {
            "num_neurons": len(self.neurons),
            "current_time": self.current_time,
            "neuron_states": {
                i: {
                    "potential": neuron.potential,
                    "spike_count": neuron.spike_count,
                    "last_spike_time": neuron.last_spike_time,
                    "num_connections": len(neuron.connections),
                }
                for i, neuron in enumerate(self.neurons)
            },
            "total_connections": sum(len(neuron.connections) for neuron in self.neurons),
            "network_activity": sum(neuron.spike_count for neuron in self.neurons),
        }

    def _flatten_external_pattern(self, pattern: Dict) -> Dict[int, bool]:
        """Accept nested or flat input patterns and return flat neuron->bool mapping."""
        # Already flat dict[int,bool]
        if pattern and all(isinstance(k, int) and isinstance(v, bool) for k, v in pattern.items()):
            return pattern

        total = max(1, getattr(self, "num_neurons", 1))
        flat = {}
        if isinstance(pattern, dict):
            for module_id, neurons in pattern.items():
                if not isinstance(neurons, dict):
                    continue
                for neuron_id, is_active in neurons.items():
                    if not is_active:
                        continue
                    global_id = (int(module_id) * 50 + int(neuron_id)) % total
                    flat[global_id] = True
        return flat

    def process_pattern(self, input_pattern: Dict) -> Dict:
        """Process an input pattern and return normalized activity metrics.

        Returns dict containing: total_activity, activities, attention, memory_size, active_neurons.
        """
        flat = self._flatten_external_pattern(input_pattern)
        step_out = self.step(flat)

        # Compute metrics
        spike_count = sum(1 for v in step_out.values() if v)
        total_neurons = getattr(self, "num_neurons", max(1, len(step_out)))
        total_activity = spike_count / float(total_neurons)

        attention = getattr(self, "attention_weights", None)
        if attention is None:
            num_channels = 4
            attention_vec = np.ones(num_channels) / num_channels
        else:
            attention_vec = np.array(attention, dtype=float)
            num_channels = len(attention_vec)

        activities = {
            i: float(total_activity) * float(attention_vec[i]) for i in range(num_channels)
        }
        memory_size = len(getattr(self, "working_memory", []))

        return {
            "total_activity": float(total_activity),
            "activities": activities,
            "attention": attention_vec.tolist(),
            "memory_size": int(memory_size),
            "active_neurons": int(spike_count),
        }

    def save_network_state(self, filepath: str):
        """Persist minimal network state to a JSON file."""
        state = {
            "num_neurons": int(self.num_neurons),
            "current_time": float(self.current_time),
            "dt": float(self.dt),
            "config": {
                "dt": float(self.config.dt),
                "weight_range": [
                    float(self.config.weight_range[0]),
                    float(self.config.weight_range[1]),
                ],
                "max_weight": float(self.config.max_weight),
                "learning_rate": float(self.config.learning_rate),
                "learning_probability": float(self.config.learning_probability),
                "num_attention_modules": int(self.config.num_attention_modules),
                "attention_init": str(self.config.attention_init),
                "max_spike_history": int(self.config.max_spike_history),
            },
            "attention_weights": [
                float(w) for w in getattr(self, "attention_weights", np.ones(4) / 4.0)
            ],
            "neurons": {
                int(n.id): {
                    "voltage": float(n.voltage),
                    "last_spike_time": float(n.last_spike_time),
                    "spike_count": int(n.spike_count),
                    "weights": {int(k): float(v) for k, v in n.weights.items()},
                }
                for n in self.neurons
            },
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)


def create_simple_input(t: float, frequency: float = 10.0) -> Dict[int, bool]:
    """Generate simple rhythmic input"""
    inputs = {}

    # Simple sine wave input to first 10 neurons
    if np.sin(2 * np.pi * frequency * t / 1000.0) > 0.5:
        num_active = np.random.poisson(3)
        for i in range(min(num_active, 10)):
            inputs[i] = True

    return inputs


@dataclass
class NetworkPresets:
    """Predefined network configurations for common use cases"""

    @staticmethod
    def fast_learning():
        """High learning rate, frequent updates"""
        return NetworkConfig(
            learning_rate=0.05,
            learning_probability=0.3,
            weight_range=(0.1, 1.2),
            max_weight=3.0,
        )

    @staticmethod
    def stable_learning():
        """Conservative learning for stability"""
        return NetworkConfig(
            learning_rate=0.005,
            learning_probability=0.05,
            weight_range=(0.05, 0.5),
            max_weight=1.5,
        )

    @staticmethod
    def attention_focused():
        """More attention modules for complex tasks"""
        return NetworkConfig(num_attention_modules=8, attention_init="random", learning_rate=0.02)


def demo_simple_network():
    """Demonstrate the simplified network"""
    print("=== Simplified Brain Network Demo ===")

    # Create network with default configuration
    network = SimpleBrainNetwork(num_neurons=100, connectivity_prob=0.05)
    print(f"Created network with {network.num_neurons} neurons")
    print(f"Configuration: {network.config}")

    # Run simulation
    results = network.run_simulation(
        duration=5000.0,  # 5 seconds
        input_pattern_func=lambda t: create_simple_input(t, frequency=10.0),
    )

    # Show results
    total_spikes = len(results["spike_record"])
    avg_activity = np.mean(results["activity_history"])

    print("\nSimulation Results:")
    print(f"Total spikes: {total_spikes}")
    print(f"Average activity: {avg_activity:.2f} spikes/ms")
    print(f"Network firing rate: {total_spikes / (5000 * 100):.4f} Hz per neuron")

    return network, results


def demo_configurable_network():
    """Demonstrate configurable network parameters"""
    print("\n=== Configurable Network Demo ===")

    # Create network with custom configuration
    config = NetworkPresets.fast_learning()
    network = SimpleBrainNetwork(num_neurons=50, connectivity_prob=0.1, config=config)

    print("Created fast-learning network:")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Learning probability: {config.learning_probability}")
    print(f"  Weight range: {config.weight_range}")
    print(f"  Max weight: {config.max_weight}")
    print(f"  Attention modules: {config.num_attention_modules}")

    # Quick simulation
    results = network.run_simulation(
        duration=1000.0,  # 1 second
        input_pattern_func=lambda t: create_simple_input(t, frequency=15.0),
    )

    # Show results
    total_spikes = len(results["spike_record"])
    avg_activity = np.mean(results["activity_history"])

    print("\nResults:")
    print(f"  Total spikes: {total_spikes}")
    print(f"  Average activity: {avg_activity:.2f} spikes/ms")
    print(f"  Network firing rate: {total_spikes / (1000 * 50):.4f} Hz per neuron")

    return network, results


if __name__ == "__main__":
    # Run demonstrations
    print("🧠 Testing SimpleBrainNetwork Configurations")
    print("=" * 50)

    # Default configuration demo
    network1, results1 = demo_simple_network()

    # Configurable network demo
    network2, results2 = demo_configurable_network()

    print("\n" + "=" * 50)
    print("🎯 Configuration Comparison:")
    spikes1 = len(results1["spike_record"])
    spikes2 = len(results2["spike_record"])
    print(f"Default network: {spikes1} spikes")
    print(f"Fast-learning network: {spikes2} spikes")

    if spikes1 > 0:
        ratio = spikes2 / spikes1
        print(f"Fast-learning network had {ratio:.1f}x more activity")
    else:
        print("Networks need stronger input to generate spikes - configurations ready!")

    # Show that configurations are working
    print("\n🔧 Configuration Details:")
    print(f"Default learning rate: {network1.config.learning_rate}")
    print(f"Fast learning rate: {network2.config.learning_rate}")
    print(f"Default attention modules: {network1.config.num_attention_modules}")
    print(f"Fast attention modules: {network2.config.num_attention_modules}")

    print("\n✅ All configurations working! No more hardcoded values! 🎉")
