"""Smoke test for automated training module importability."""

import importlib


def test_import_automated_trainer():
    m = importlib.import_module("training.automated_internet_trainer")
    assert hasattr(m, "AutomatedInternetTrainer")
