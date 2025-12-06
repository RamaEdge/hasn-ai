"""
Random number generator seeding utility for HASN-AI.

Provides deterministic random number generation by seeding Python's random,
numpy, and torch (if available) with the same seed value.
"""

import os
import random
from typing import Optional, Union

import numpy as np

# Try to import torch, but don't fail if it's not available
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


def seed_random_generators(
    seed: Optional[Union[int, str]] = None, deterministic: bool = True
) -> int:
    """
    Seed all random number generators for deterministic behavior.

    This function seeds:
    - Python's built-in random module
    - NumPy's random number generator
    - PyTorch's random number generator (if available)

    Args:
        seed: Random seed value. If None, will try to get from RANDOM_SEED
              environment variable, or generate a random one
        deterministic: Whether to use deterministic algorithms (for PyTorch)

    Returns:
        The seed value that was used

    Example:
        >>> seed_value = seed_random_generators(42)
        >>> print(f"Seeded with: {seed_value}")
        >>>
        >>> # Now all random operations will be deterministic
        >>> import random
        >>> import numpy as np
        >>> print(random.random())  # Will be the same every time
        >>> print(np.random.random())  # Will be the same every time
    """
    # Determine seed value
    if seed is None:
        seed = os.getenv("RANDOM_SEED")
        if seed is not None:
            try:
                seed = int(seed)
            except ValueError:
                # If RANDOM_SEED is not a valid integer, generate a random one
                seed = random.randint(0, 2**32 - 1)
        else:
            # Generate a random seed if none provided
            seed = random.randint(0, 2**32 - 1)

    # Convert string seed to int if needed
    if isinstance(seed, str):
        try:
            seed = int(seed)
        except ValueError:
            # Use hash of string as seed
            seed = hash(seed) % (2**32)

    # Ensure seed is within valid range
    seed = int(seed) % (2**32)

    # Seed Python's random module
    random.seed(seed)

    # Seed NumPy's random number generator
    np.random.seed(seed)

    # Seed PyTorch if available
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Set deterministic behavior if requested
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    return seed


def get_random_state() -> dict:
    """
    Get the current random state of all generators.

    Returns:
        Dictionary containing the state of all random number generators

    Example:
        >>> state = get_random_state()
        >>> print(f"Python random state: {state['python']}")
        >>> print(f"NumPy random state: {state['numpy']}")
    """
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }

    if TORCH_AVAILABLE:
        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state()

    return state


def set_random_state(state: dict) -> None:
    """
    Restore the random state of all generators.

    Args:
        state: Dictionary containing the state of random number generators
               (as returned by get_random_state())

    Example:
        >>> # Save current state
        >>> state = get_random_state()
        >>>
        >>> # Do some random operations
        >>> values = [random.random() for _ in range(5)]
        >>>
        >>> # Restore state
        >>> set_random_state(state)
        >>>
        >>> # Same random operations will produce same results
        >>> values2 = [random.random() for _ in range(5)]
        >>> assert values == values2
    """
    if "python" in state:
        random.setstate(state["python"])

    if "numpy" in state:
        np.random.set_state(state["numpy"])

    if TORCH_AVAILABLE and "torch" in state:
        torch.set_rng_state(state["torch"])

        if torch.cuda.is_available() and "torch_cuda" in state:
            torch.cuda.set_rng_state(state["torch_cuda"])


def create_deterministic_context(seed: int = 42):
    """
    Context manager for deterministic random number generation.

    This context manager saves the current random state, seeds all generators
    with the specified seed, and restores the original state when exiting.

    Args:
        seed: Random seed to use within the context

    Example:
        >>> with create_deterministic_context(42):
        ...     # All random operations here will be deterministic
        ...     values = [random.random() for _ in range(3)]
        ...     print(values)  # Will always be the same
        >>>
        >>> # Random operations outside the context are not affected
        >>> print(random.random())  # Will be different each time
    """

    class DeterministicContext:
        def __init__(self, seed_value: int):
            self.seed_value = seed_value
            self.original_state = None

        def __enter__(self):
            # Save current state
            self.original_state = get_random_state()
            # Set deterministic seed
            seed_random_generators(self.seed_value, deterministic=True)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            # Restore original state
            if self.original_state:
                set_random_state(self.original_state)

    return DeterministicContext(seed)


# Convenience function for quick seeding
def seed_for_reproducibility(seed: int = 42) -> int:
    """
    Quick setup for reproducible results.

    This is a convenience function that seeds all random number generators
    with a fixed seed for reproducible results in development and testing.

    Args:
        seed: Random seed value (default: 42)

    Returns:
        The seed value that was used
    """
    return seed_random_generators(seed, deterministic=True)


# Example usage and testing
if __name__ == "__main__":
    # Test basic seeding
    print("Testing random number generator seeding...")

    # Seed with a fixed value
    seed_value = seed_random_generators(42)
    print(f"Seeded with: {seed_value}")

    # Test Python random
    python_values = [random.random() for _ in range(3)]
    print(f"Python random values: {python_values}")

    # Test NumPy random
    numpy_values = np.random.random(3).tolist()
    print(f"NumPy random values: {numpy_values}")

    # Test PyTorch random (if available)
    if TORCH_AVAILABLE:
        torch_values = torch.rand(3).tolist()
        print(f"PyTorch random values: {torch_values}")
    else:
        print("PyTorch not available")

    # Test deterministic context
    print("\nTesting deterministic context...")
    with create_deterministic_context(123):
        context_values = [random.random() for _ in range(3)]
        print(f"Context values: {context_values}")

    # Values outside context should be different
    outside_values = [random.random() for _ in range(3)]
    print(f"Outside context values: {outside_values}")

    print("\nRandom seeding test completed!")
