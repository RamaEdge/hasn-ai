"""
NumPy-based backend for brain network computations.

This backend uses NumPy arrays for efficient computation of spiking neural networks.
It provides a minimal but functional SNN engine using integrate-and-fire dynamics.
"""

from typing import Any, Dict, Optional

import numpy as np

from .interface import BrainBackend


class NumpyBackend(BrainBackend):
    """
    NumPy-based backend for spiking neural network computations.

    Uses NumPy arrays for efficient vectorized operations on:
    - Membrane potentials
    - Synaptic weights
    - Spike states
    - Input currents
    """

    def __init__(self, num_neurons: int, **kwargs):
        """
        Initialize NumPy backend.

        Args:
            num_neurons: Number of neurons in the network
            **kwargs: Additional configuration parameters
        """
        super().__init__(num_neurons, **kwargs)

        # Neuron parameters
        self.tau_membrane = kwargs.get("tau_membrane", 20.0)  # Membrane time constant (ms)
        self.threshold = kwargs.get("threshold", 1.0)  # Spike threshold
        self.reset_potential = kwargs.get("reset_potential", 0.0)  # Reset potential

        # Learning parameters (configurable)
        self.background_noise_level = kwargs.get("background_noise_level", 0.01)
        self.min_weight_bound = kwargs.get("min_weight_bound", 0.0)
        self.max_weight_bound = kwargs.get("max_weight_bound", 2.0)

        # Initialize neuron states
        self.membrane_potentials = np.zeros(num_neurons, dtype=np.float32)
        self.last_spike_times = np.full(num_neurons, -1000.0, dtype=np.float32)
        self.spike_states = np.zeros(num_neurons, dtype=bool)

        # Initialize synaptic weights (sparse connectivity)
        connectivity_prob = kwargs.get("connectivity_prob", 0.1)
        self.weights = self._initialize_weights(connectivity_prob, kwargs)

        # Input currents
        self.input_currents = np.zeros(num_neurons, dtype=np.float32)

        # Recording
        self.spike_history = []
        self.membrane_history = []

    def _initialize_weights(self, connectivity_prob: float, kwargs: dict) -> np.ndarray:
        """
        Initialize synaptic weight matrix.

        Args:
            connectivity_prob: Probability of connection between neurons
            **kwargs: Additional parameters

        Returns:
            Weight matrix (num_neurons x num_neurons)
        """
        # Create sparse connectivity matrix
        weights = np.zeros((self.num_neurons, self.num_neurons), dtype=np.float32)

        # Random connections
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if i != j and np.random.random() < connectivity_prob:
                    # Random weight between min and max
                    min_weight = kwargs.get("min_weight", 0.1)
                    max_weight = kwargs.get("max_weight", 0.8)
                    weights[i, j] = np.random.uniform(min_weight, max_weight)

        return weights

    def step(self, inputs: Optional[Dict[int, bool]] = None) -> Dict[int, bool]:
        """
        Perform one simulation step.

        Args:
            inputs: Optional external input spikes as {neuron_id: spike_boolean}

        Returns:
            Dictionary of output spikes {neuron_id: spike_boolean}
        """
        # Reset spike states
        self.spike_states.fill(False)

        # Process external inputs
        if inputs:
            for neuron_id, spike in inputs.items():
                if 0 <= neuron_id < self.num_neurons and spike:
                    self.input_currents[neuron_id] += 1.0  # Add input current

        # Calculate synaptic currents from previous spikes
        # Note: spike_states are from current step, we need previous spikes
        # For now, use a simple approach with random background activity
        synaptic_currents = np.dot(self.weights, self.spike_states.astype(np.float32))

        # Add small background noise to prevent complete stillness (configurable)
        background_noise = np.random.normal(0, self.background_noise_level, self.num_neurons)

        # Total input current
        total_currents = self.input_currents + synaptic_currents + background_noise

        # Update membrane potentials using Euler integration
        # dV/dt = (I - V) / tau
        self.membrane_potentials += (
            self.dt * (total_currents - self.membrane_potentials) / self.tau_membrane
        )

        # Check for spikes
        spike_mask = self.membrane_potentials >= self.threshold
        self.spike_states = spike_mask

        # Reset spiked neurons
        self.membrane_potentials[spike_mask] = self.reset_potential
        self.last_spike_times[spike_mask] = self.current_time

        # Clear input currents
        self.input_currents.fill(0.0)

        # Record activity
        self.spike_history.append(self.current_time)
        self.membrane_history.append(self.membrane_potentials.copy())

        # Advance time
        self.advance_time()

        # Return output spikes as dictionary
        return {i: bool(self.spike_states[i]) for i in range(self.num_neurons)}

    def get_membrane_potentials(self) -> np.ndarray:
        """Get current membrane potentials for all neurons."""
        return self.membrane_potentials.copy()

    def get_synaptic_weights(self) -> np.ndarray:
        """Get current synaptic weight matrix."""
        return self.weights.copy()

    def set_synaptic_weights(self, weights: np.ndarray) -> None:
        """
        Set synaptic weight matrix.

        Args:
            weights: Weight matrix (num_neurons x num_neurons)
        """
        if weights.shape != (self.num_neurons, self.num_neurons):
            raise ValueError(
                f"Weight matrix shape {weights.shape} doesn't match network size {self.num_neurons}"
            )
        self.weights = weights.astype(np.float32)

    def reset(self) -> None:
        """Reset the network to initial state."""
        self.membrane_potentials.fill(0.0)
        self.last_spike_times.fill(-1000.0)
        self.spike_states.fill(False)
        self.input_currents.fill(0.0)
        self.current_time = 0.0
        self.spike_history.clear()
        self.membrane_history.clear()

    def get_state(self) -> Dict[str, Any]:
        """
        Get current network state for serialization.

        Returns:
            Dictionary containing network state
        """
        return {
            "membrane_potentials": self.membrane_potentials.tolist(),
            "last_spike_times": self.last_spike_times.tolist(),
            "weights": self.weights.tolist(),
            "current_time": self.current_time,
            "dt": self.dt,
            "tau_membrane": self.tau_membrane,
            "threshold": self.threshold,
            "reset_potential": self.reset_potential,
            "num_neurons": self.num_neurons,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Set network state from serialized data.

        Args:
            state: Dictionary containing network state
        """
        self.membrane_potentials = np.array(state["membrane_potentials"], dtype=np.float32)
        self.last_spike_times = np.array(state["last_spike_times"], dtype=np.float32)
        self.weights = np.array(state["weights"], dtype=np.float32)
        self.current_time = state["current_time"]
        self.dt = state["dt"]
        self.tau_membrane = state["tau_membrane"]
        self.threshold = state["threshold"]
        self.reset_potential = state["reset_potential"]

        # Validate dimensions
        if len(self.membrane_potentials) != self.num_neurons:
            raise ValueError("State membrane potentials don't match network size")
        if self.weights.shape != (self.num_neurons, self.num_neurons):
            raise ValueError("State weight matrix doesn't match network size")

    @property
    def backend_name(self) -> str:
        """Get the name of this backend."""
        return "numpy"

    @property
    def is_available(self) -> bool:
        """Check if this backend is available (NumPy is always available)."""
        return True

    def get_activity_summary(self) -> Dict[str, Any]:
        """
        Get summary of network activity.

        Returns:
            Dictionary with activity metrics
        """
        if not self.membrane_history:
            return {"total_spikes": 0, "avg_membrane": 0.0, "active_neurons": 0}

        recent_membranes = (
            self.membrane_history[-10:]
            if len(self.membrane_history) >= 10
            else self.membrane_history
        )
        avg_membrane = np.mean([np.mean(m) for m in recent_membranes])

        return {
            "total_spikes": len(self.spike_history),
            "avg_membrane": float(avg_membrane),
            "active_neurons": int(np.sum(self.spike_states)),
            "current_time": self.current_time,
        }

    def apply_learning(self, learning_rate: float = 0.01) -> None:
        """
        Apply simple Hebbian learning to weights.

        Args:
            learning_rate: Learning rate for weight updates
        """
        # Simple Hebbian learning: strengthen connections between co-active neurons
        if len(self.membrane_history) >= 2:
            # Get recent activity patterns
            recent_activity = np.array(self.membrane_history[-5:])  # Last 5 timesteps

            # Calculate correlation matrix, handling potential NaN values
            try:
                activity_corr = np.corrcoef(recent_activity.T)
                # Replace NaN values with 0
                activity_corr = np.nan_to_num(activity_corr, nan=0.0, posinf=0.0, neginf=0.0)

                # Update weights based on correlations
                weight_update = learning_rate * activity_corr
                self.weights += weight_update

                # Keep weights bounded (configurable)
                self.weights = np.clip(self.weights, self.min_weight_bound, self.max_weight_bound)
            except Exception:
                # If correlation calculation fails, skip learning
                pass
