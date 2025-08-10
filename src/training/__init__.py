"""Training modules for HASN-AI.

Supported training entrypoints:
- automated_internet_trainer: Continuous web-sourced training for SimpleBrainNetwork
- train_cli: CLI wrapper to start/monitor automated training
- training_monitor: Utilities to analyze training outputs

Note:
- Removed legacy interactive/comprehensive/quick demo trainers to reduce duplication.
  Use API interactive training (POST /training/interactive) for CognitiveBrainNetwork episodic training,
  and use examples under `examples/` for cognitive inference demos.
"""
