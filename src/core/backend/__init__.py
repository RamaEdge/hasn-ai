# Backend abstraction for HASN-AI
"""
Backend abstraction layer for brain-inspired neural networks.
Allows swapping between different computational backends (NumPy, Norse, etc.).
"""

from .factory import get_backend, list_available_backends
from .interface import BrainBackend
from .norse_backend import NorseBackend
from .numpy_backend import NumpyBackend

__all__ = ["BrainBackend", "NumpyBackend", "NorseBackend", "get_backend", "list_available_backends"]
