"""Brain integration smoke test."""

import importlib


def test_import_brain_modules():
    importlib.import_module("core.simplified_brain_network")
    importlib.import_module("core.cognitive_brain_network")
