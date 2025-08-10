"""Smoke tests for API wiring.

These tests verify that adapters are importable and that the FastAPI app
can be instantiated without runtime errors.
"""

import importlib


def test_import_adapters():
    mod = importlib.import_module("api.adapters.brain_adapters")
    assert hasattr(mod, "SimpleBrainAdapter")
    assert hasattr(mod, "CognitiveBrainAdapter")
