"""
Backend factory for creating and managing different computational backends.

This module provides functionality to switch between different backends
using environment variables and provides a unified interface.
"""

import os
from typing import Dict, Any, Optional, List
from .interface import BrainBackend
from .numpy_backend import NumpyBackend
from .norse_backend import NorseBackend


# Registry of available backends
BACKEND_REGISTRY = {
    "numpy": NumpyBackend,
    "norse": NorseBackend,
}


def get_backend(backend_name: Optional[str] = None, num_neurons: int = 100, **kwargs) -> BrainBackend:
    """
    Get a backend instance based on the specified name or environment variable.
    
    Args:
        backend_name: Name of the backend to use (overrides environment variable)
        num_neurons: Number of neurons in the network
        **kwargs: Additional backend-specific parameters
        
    Returns:
        BrainBackend instance
        
    Raises:
        ValueError: If backend is not available or not found
        NotImplementedError: If backend is not implemented yet
    """
    # Determine backend name
    if backend_name is None:
        backend_name = os.getenv("HASN_BACKEND", "numpy").lower()
    
    # Validate backend name
    if backend_name not in BACKEND_REGISTRY:
        available = list(BACKEND_REGISTRY.keys())
        raise ValueError(f"Unknown backend '{backend_name}'. Available backends: {available}")
    
    # Get backend class
    backend_class = BACKEND_REGISTRY[backend_name]
    
    # Create backend instance
    try:
        backend = backend_class(num_neurons=num_neurons, **kwargs)
        
        # Check if backend is available
        if not backend.is_available:
            # For placeholder backends like Norse, we allow creation but mark as unavailable
            if backend_name == "norse":
                # Norse is a placeholder, so we allow it but it won't be functional
                pass
            else:
                raise NotImplementedError(
                    f"Backend '{backend_name}' is not available. "
                    f"Check dependencies or use a different backend."
                )
        
        return backend
        
    except Exception as e:
        raise ValueError(f"Failed to create backend '{backend_name}': {e}")


def list_available_backends() -> List[str]:
    """
    List all available backends.
    
    Returns:
        List of backend names that are available
    """
    available = []
    
    for name, backend_class in BACKEND_REGISTRY.items():
        try:
            # Try to create a minimal instance to check availability
            backend = backend_class(num_neurons=1)
            if backend.is_available:
                available.append(name)
        except Exception:
            # Backend is not available
            pass
    
    return available


def get_backend_info(backend_name: str) -> Dict[str, Any]:
    """
    Get information about a specific backend.
    
    Args:
        backend_name: Name of the backend
        
    Returns:
        Dictionary with backend information
    """
    if backend_name not in BACKEND_REGISTRY:
        return {
            "name": backend_name,
            "available": False,
            "error": f"Backend '{backend_name}' not found in registry"
        }
    
    backend_class = BACKEND_REGISTRY[backend_name]
    
    try:
        # Create a minimal instance to get info
        backend = backend_class(num_neurons=1)
        
        return {
            "name": backend_name,
            "class": backend_class.__name__,
            "available": backend.is_available,
            "description": backend_class.__doc__ or "No description available"
        }
        
    except Exception as e:
        return {
            "name": backend_name,
            "available": False,
            "error": str(e)
        }


def register_backend(name: str, backend_class: type) -> None:
    """
    Register a new backend class.
    
    Args:
        name: Name of the backend
        backend_class: Backend class that implements BrainBackend interface
    """
    if not issubclass(backend_class, BrainBackend):
        raise ValueError(f"Backend class must inherit from BrainBackend")
    
    BACKEND_REGISTRY[name] = backend_class


def get_default_backend() -> str:
    """
    Get the default backend name.
    
    Returns:
        Default backend name
    """
    return os.getenv("HASN_BACKEND", "numpy")


def set_default_backend(backend_name: str) -> None:
    """
    Set the default backend via environment variable.
    
    Args:
        backend_name: Name of the backend to set as default
    """
    if backend_name not in BACKEND_REGISTRY:
        available = list(BACKEND_REGISTRY.keys())
        raise ValueError(f"Unknown backend '{backend_name}'. Available backends: {available}")
    
    os.environ["HASN_BACKEND"] = backend_name


# Convenience function for quick backend creation
def create_backend(num_neurons: int = 100, **kwargs) -> BrainBackend:
    """
    Create a backend using the default configuration.
    
    Args:
        num_neurons: Number of neurons in the network
        **kwargs: Additional backend-specific parameters
        
    Returns:
        BrainBackend instance
    """
    return get_backend(num_neurons=num_neurons, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    print("Available backends:", list_available_backends())
    
    # Test default backend
    try:
        backend = get_backend(num_neurons=10)
        print(f"Created backend: {backend}")
        print(f"Backend info: {backend.get_network_info()}")
        
        # Test a simple step
        output = backend.step()
        print(f"Step output: {output}")
        
    except Exception as e:
        print(f"Error creating backend: {e}")
    
    # Test backend info
    for backend_name in BACKEND_REGISTRY.keys():
        info = get_backend_info(backend_name)
        print(f"Backend {backend_name}: {info}")
