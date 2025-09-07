# Backend abstraction for HASN-AI
"""
Backend abstraction layer for brain-inspired neural networks.
Allows swapping between different computational backends (NumPy, Norse, etc.).
"""

from .interface import BrainBackend
from .numpy_backend import NumpyBackend
from .norse_backend import NorseBackend
from .factory import get_backend, list_available_backends

__all__ = [
    "BrainBackend",
    "NumpyBackend", 
    "NorseBackend",
    "get_backend",
    "list_available_backends"
]
