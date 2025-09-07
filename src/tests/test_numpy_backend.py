"""
Unit tests for the NumPy backend implementation.

Tests the NumpyBackend class to ensure it correctly implements
the BrainBackend interface and provides proper SNN functionality.
"""

import os
import sys
import tempfile
import numpy as np
import pytest
from typing import Dict

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.backend.numpy_backend import NumpyBackend
from core.backend.factory import get_backend, list_available_backends, get_backend_info


class TestNumpyBackend:
    """Test suite for NumpyBackend implementation."""
    
    def test_backend_initialization(self):
        """Test that NumpyBackend initializes correctly."""
        backend = NumpyBackend(num_neurons=10)
        
        assert backend.num_neurons == 10
        assert backend.backend_name == "numpy"
        assert backend.is_available is True
        assert backend.current_time == 0.0
        assert backend.dt == 1.0
        
        # Check internal arrays
        assert backend.membrane_potentials.shape == (10,)
        assert backend.weights.shape == (10, 10)
        assert backend.spike_states.shape == (10,)
    
    def test_backend_initialization_with_params(self):
        """Test NumpyBackend initialization with custom parameters."""
        backend = NumpyBackend(
            num_neurons=5,
            dt=0.5,
            tau_membrane=15.0,
            threshold=1.5,
            reset_potential=0.1,
            connectivity_prob=0.2,
            min_weight=0.2,
            max_weight=0.9
        )
        
        assert backend.num_neurons == 5
        assert backend.dt == 0.5
        assert backend.tau_membrane == 15.0
        assert backend.threshold == 1.5
        assert backend.reset_potential == 0.1
        
        # Check weight matrix has some connections
        assert np.any(backend.weights > 0)
    
    def test_step_without_inputs(self):
        """Test step() method without external inputs."""
        backend = NumpyBackend(num_neurons=5)
        
        # Initial state
        initial_potentials = backend.get_membrane_potentials().copy()
        
        # Run one step
        output = backend.step()
        
        # Check output format
        assert isinstance(output, dict)
        assert len(output) == 5
        assert all(isinstance(k, int) for k in output.keys())
        assert all(isinstance(v, bool) for v in output.values())
        
        # Check time advanced
        assert backend.current_time == 1.0
        
        # Check membrane potentials changed (due to decay)
        new_potentials = backend.get_membrane_potentials()
        assert not np.array_equal(initial_potentials, new_potentials)
    
    def test_step_with_inputs(self):
        """Test step() method with external inputs."""
        backend = NumpyBackend(num_neurons=5)
        
        # Provide external input
        inputs = {0: True, 2: True, 4: True}
        output = backend.step(inputs)
        
        # Check that inputs were processed
        assert isinstance(output, dict)
        assert len(output) == 5
        
        # Check that time advanced
        assert backend.current_time == 1.0
    
    def test_multiple_steps(self):
        """Test running multiple simulation steps."""
        backend = NumpyBackend(num_neurons=3)
        
        # Run multiple steps
        outputs = []
        for i in range(5):
            output = backend.step()
            outputs.append(output)
        
        # Check all steps completed
        assert len(outputs) == 5
        assert backend.current_time == 5.0
        
        # Check each output is valid
        for output in outputs:
            assert isinstance(output, dict)
            assert len(output) == 3
    
    def test_membrane_potential_dynamics(self):
        """Test membrane potential dynamics."""
        backend = NumpyBackend(num_neurons=3, tau_membrane=10.0, threshold=2.0)
        
        # Provide strong input to neuron 0
        inputs = {0: True}
        
        # Run several steps
        for _ in range(10):
            backend.step(inputs)
        
        # Check that neuron 0 has higher potential
        potentials = backend.get_membrane_potentials()
        assert potentials[0] > potentials[1]
        assert potentials[0] > potentials[2]
    
    def test_spike_generation(self):
        """Test spike generation when threshold is reached."""
        backend = NumpyBackend(num_neurons=3, threshold=0.5, tau_membrane=1.0)
        
        # Provide strong input to neuron 0
        inputs = {0: True}
        
        # Run steps until spike occurs
        spiked = False
        for _ in range(20):
            output = backend.step(inputs)
            if output[0]:  # Neuron 0 spiked
                spiked = True
                break
        
        assert spiked, "Neuron should have spiked with strong input"
    
    def test_spike_reset(self):
        """Test that neurons reset after spiking."""
        backend = NumpyBackend(num_neurons=3, threshold=0.5, reset_potential=0.0)
        
        # Provide input to make neuron 0 spike
        inputs = {0: True}
        
        # Find when neuron 0 spikes
        spike_time = None
        for _ in range(20):
            output = backend.step(inputs)
            if output[0] and spike_time is None:
                spike_time = backend.current_time
                break
        
        assert spike_time is not None, "Neuron should have spiked"
        
        # Check that potential was reset
        potentials = backend.get_membrane_potentials()
        assert potentials[0] == 0.0, "Membrane potential should be reset after spike"
    
    def test_weight_matrix_operations(self):
        """Test synaptic weight matrix operations."""
        backend = NumpyBackend(num_neurons=4)
        
        # Get initial weights
        initial_weights = backend.get_synaptic_weights()
        
        # Create new weight matrix
        new_weights = np.random.random((4, 4)) * 0.5
        backend.set_synaptic_weights(new_weights)
        
        # Check weights were set (allow for small floating point differences)
        current_weights = backend.get_synaptic_weights()
        assert np.allclose(current_weights, new_weights, rtol=1e-6)
        
        # Test invalid weight matrix
        with pytest.raises(ValueError):
            backend.set_synaptic_weights(np.random.random((3, 3)))  # Wrong size
    
    def test_reset_functionality(self):
        """Test network reset functionality."""
        backend = NumpyBackend(num_neurons=5)
        
        # Run some steps to change state
        for _ in range(10):
            backend.step({0: True})
        
        # Check state changed
        assert backend.current_time > 0
        assert len(backend.spike_history) > 0
        
        # Reset network
        backend.reset()
        
        # Check state is reset
        assert backend.current_time == 0.0
        assert len(backend.spike_history) == 0
        assert len(backend.membrane_history) == 0
        assert np.all(backend.membrane_potentials == 0.0)
        assert np.all(backend.last_spike_times == -1000.0)
    
    def test_state_serialization(self):
        """Test network state save/load functionality."""
        backend = NumpyBackend(num_neurons=3)
        
        # Run some steps to create state
        for _ in range(5):
            backend.step({0: True})
        
        # Get state
        state = backend.get_state()
        
        # Check state structure
        assert "membrane_potentials" in state
        assert "weights" in state
        assert "current_time" in state
        assert "num_neurons" in state
        
        # Create new backend and load state
        new_backend = NumpyBackend(num_neurons=3)
        new_backend.set_state(state)
        
        # Check states match
        assert np.array_equal(backend.get_membrane_potentials(), new_backend.get_membrane_potentials())
        assert np.array_equal(backend.get_synaptic_weights(), new_backend.get_synaptic_weights())
        assert backend.current_time == new_backend.current_time
    
    def test_activity_summary(self):
        """Test activity summary functionality."""
        backend = NumpyBackend(num_neurons=3)
        
        # Initial summary
        summary = backend.get_activity_summary()
        assert summary["total_spikes"] == 0
        assert summary["avg_membrane"] == 0.0
        assert summary["active_neurons"] == 0
        
        # Run some steps
        for _ in range(5):
            backend.step({0: True})
        
        # Check updated summary
        summary = backend.get_activity_summary()
        assert summary["total_spikes"] >= 0
        assert summary["current_time"] == 5.0
    
    def test_learning_functionality(self):
        """Test simple learning functionality."""
        backend = NumpyBackend(num_neurons=3)
        
        # Get initial weights
        initial_weights = backend.get_synaptic_weights().copy()
        
        # Run some steps to create activity
        for _ in range(10):
            backend.step({0: True, 1: True})
        
        # Apply learning
        backend.apply_learning(learning_rate=0.1)
        
        # Check weights changed
        new_weights = backend.get_synaptic_weights()
        assert not np.array_equal(initial_weights, new_weights)
        
        # Check weights are bounded
        assert np.all(new_weights >= 0.0)
        assert np.all(new_weights <= 2.0)


class TestBackendFactory:
    """Test suite for backend factory functionality."""
    
    def test_get_backend_default(self):
        """Test getting default backend."""
        backend = get_backend(num_neurons=5)
        assert isinstance(backend, NumpyBackend)
        assert backend.num_neurons == 5
    
    def test_get_backend_numpy(self):
        """Test getting NumPy backend explicitly."""
        backend = get_backend("numpy", num_neurons=3)
        assert isinstance(backend, NumpyBackend)
        assert backend.num_neurons == 3
    
    def test_get_backend_norse_placeholder(self):
        """Test getting Norse backend (placeholder)."""
        backend = get_backend("norse", num_neurons=3)
        assert backend.backend_name == "norse"
        # Should not be available unless Norse is installed
        assert not backend.is_available
    
    def test_get_backend_invalid(self):
        """Test getting invalid backend."""
        with pytest.raises(ValueError):
            get_backend("invalid_backend", num_neurons=3)
    
    def test_list_available_backends(self):
        """Test listing available backends."""
        available = list_available_backends()
        assert "numpy" in available
        # Norse may or may not be available depending on installation
    
    def test_get_backend_info(self):
        """Test getting backend information."""
        info = get_backend_info("numpy")
        assert info["name"] == "numpy"
        assert info["available"] is True
        assert "NumpyBackend" in info["class"]
    
    def test_environment_variable_backend_selection(self):
        """Test backend selection via environment variable."""
        # Save original value
        original_backend = os.environ.get("HASN_BACKEND")
        
        try:
            # Set environment variable
            os.environ["HASN_BACKEND"] = "numpy"
            
            # Get backend (should use environment variable)
            backend = get_backend(num_neurons=3)
            assert isinstance(backend, NumpyBackend)
            
        finally:
            # Restore original value
            if original_backend is not None:
                os.environ["HASN_BACKEND"] = original_backend
            elif "HASN_BACKEND" in os.environ:
                del os.environ["HASN_BACKEND"]


class TestToyExample:
    """Test the 10-neuron toy example as specified in deliverables."""
    
    def test_10_neuron_toy_example(self):
        """Test NumpyBackend with 10-neuron toy example."""
        # Create 10-neuron network
        backend = NumpyBackend(num_neurons=10, connectivity_prob=0.2)
        
        # Verify initialization
        assert backend.num_neurons == 10
        assert backend.is_available is True
        
        # Test basic functionality
        output = backend.step()
        assert len(output) == 10
        assert all(isinstance(k, int) for k in output.keys())
        assert all(isinstance(v, bool) for v in output.values())
        
        # Test with inputs
        inputs = {0: True, 5: True, 9: True}
        output = backend.step(inputs)
        assert len(output) == 10
        
        # Test multiple steps
        for _ in range(5):
            output = backend.step()
            assert len(output) == 10
        
        # Test state operations
        state = backend.get_state()
        assert state["num_neurons"] == 10
        
        # Test activity summary
        summary = backend.get_activity_summary()
        assert summary["current_time"] == 7.0  # 7 steps total
        
        print("✅ 10-neuron toy example test passed")
        print(f"   Network info: {backend.get_network_info()}")
        print(f"   Activity summary: {summary}")


# Integration test
def test_backend_switching():
    """Test switching between backends via environment variable."""
    # Save original value
    original_backend = os.environ.get("HASN_BACKEND")
    
    try:
        # Test NumPy backend
        os.environ["HASN_BACKEND"] = "numpy"
        backend1 = get_backend(num_neurons=5)
        assert isinstance(backend1, NumpyBackend)
        
        # Test that it works
        output1 = backend1.step({0: True})
        assert len(output1) == 5
        
        # Test Norse backend (should be placeholder)
        os.environ["HASN_BACKEND"] = "norse"
        backend2 = get_backend(num_neurons=5)
        assert backend2.backend_name == "norse"
        
        # Should not be available unless Norse is installed
        assert not backend2.is_available
        
    finally:
        # Restore original value
        if original_backend is not None:
            os.environ["HASN_BACKEND"] = original_backend
        elif "HASN_BACKEND" in os.environ:
            del os.environ["HASN_BACKEND"]


if __name__ == "__main__":
    # Run tests directly
    print("Running NumPy backend tests...")
    
    # Test basic functionality
    test_instance = TestNumpyBackend()
    test_instance.test_backend_initialization()
    test_instance.test_step_without_inputs()
    test_instance.test_step_with_inputs()
    test_instance.test_10_neuron_toy_example()
    
    # Test factory
    factory_test = TestBackendFactory()
    factory_test.test_get_backend_default()
    factory_test.test_list_available_backends()
    
    # Test backend switching
    test_backend_switching()
    
    print("\n✅ All NumPy backend tests passed!")
