"""
BrainBackend interface for abstracting different computational backends.

This interface allows HASN-AI to work with different backends like NumPy, Norse,
or other spiking neural network frameworks while maintaining a consistent API.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class BrainBackend(ABC):
    """
    Abstract base class for brain network computational backends.

    This interface defines the contract that all backends must implement
    to work with the HASN-AI system.
    """

    def __init__(self, num_neurons: int, **kwargs):
        """
        Initialize the backend with a specific number of neurons.

        Args:
            num_neurons: Number of neurons in the network
            **kwargs: Backend-specific configuration parameters
        """
        self.num_neurons = num_neurons
        self.current_time = 0.0
        self.dt = kwargs.get("dt", 1.0)  # Default timestep of 1ms

    @abstractmethod
    def step(self, inputs: Optional[Dict[int, bool]] = None) -> Dict[int, bool]:
        """
        Perform one simulation step.

        Args:
            inputs: Optional external input spikes as {neuron_id: spike_boolean}

        Returns:
            Dictionary of output spikes {neuron_id: spike_boolean}
        """
        pass

    @abstractmethod
    def get_membrane_potentials(self) -> np.ndarray:
        """
        Get current membrane potentials for all neurons.

        Returns:
            Array of membrane potentials
        """
        pass

    @abstractmethod
    def get_synaptic_weights(self) -> np.ndarray:
        """
        Get current synaptic weight matrix.

        Returns:
            Weight matrix (num_neurons x num_neurons)
        """
        pass

    @abstractmethod
    def set_synaptic_weights(self, weights: np.ndarray) -> None:
        """
        Set synaptic weight matrix.

        Args:
            weights: Weight matrix (num_neurons x num_neurons)
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the network to initial state.
        """
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Get current network state for serialization.

        Returns:
            Dictionary containing network state
        """
        pass

    @abstractmethod
    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Set network state from serialized data.

        Args:
            state: Dictionary containing network state
        """
        pass

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """
        Get the name of this backend.

        Returns:
            Backend name string
        """
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this backend is available (dependencies installed).

        Returns:
            True if backend is available, False otherwise
        """
        pass

    def get_network_info(self) -> Dict[str, Any]:
        """
        Get information about the network configuration.

        Returns:
            Dictionary with network information
        """
        return {
            "backend": self.backend_name,
            "num_neurons": self.num_neurons,
            "current_time": self.current_time,
            "dt": self.dt,
            "available": self.is_available,
        }

    def advance_time(self, dt: Optional[float] = None) -> None:
        """
        Advance simulation time.

        Args:
            dt: Time step (uses self.dt if None)
        """
        if dt is None:
            dt = self.dt
        self.current_time += dt

    def __str__(self) -> str:
        """String representation of the backend."""
        return f"{self.backend_name}Backend(num_neurons={self.num_neurons})"

    def __repr__(self) -> str:
        """Detailed string representation of the backend."""
        return (
            f"{self.__class__.__name__}("
            f"num_neurons={self.num_neurons}, "
            f"current_time={self.current_time}, "
            f"dt={self.dt}, "
            f"available={self.is_available})"
        )
