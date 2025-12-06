#!/usr/bin/env python3
"""
Tests for Hyperparameter Optimizer
"""

import shutil
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.cognitive_models import CognitiveConfig
from training.hyperparameter_optimizer import HyperparameterOptimizer


@pytest.fixture
def temp_optimization_dir():
    """Create temporary directory for optimization studies"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def optimizer(temp_optimization_dir):
    """Create optimizer with temporary storage"""
    return HyperparameterOptimizer(storage_path=temp_optimization_dir)


@pytest.fixture
def sample_training_data():
    """Create sample training dataset"""
    return [
        {"input": "cat", "concept": "animal", "context": {}},
        {"input": "dog", "concept": "animal", "context": {}},
        {"input": "cat", "concept": "animal", "context": {}},  # Repeat for consolidation
        {"input": "car", "concept": "vehicle", "context": {}},
        {"input": "bike", "concept": "vehicle", "context": {}},
    ]


@pytest.mark.skipif(
    not pytest.importorskip("optuna", reason="Optuna not available"), reason="Optuna not available"
)
class TestHyperparameterOptimizer:
    """Test hyperparameter optimizer"""

    def test_optimizer_initialization(self, optimizer):
        """Test optimizer can be initialized"""
        assert optimizer is not None
        assert optimizer.storage_path.exists()

    def test_suggest_parameters(self, optimizer):
        """Test parameter suggestion"""
        import optuna

        study = optuna.create_study()
        trial = study.ask()

        config = optimizer.suggest_parameters(trial)

        assert isinstance(config, CognitiveConfig)
        assert config.hebbian_learning_rate > 0
        assert config.max_association_strength > 0
        assert config.min_weight_bound >= 0
        assert config.max_weight_bound > config.min_weight_bound

    def test_evaluate_config(self, optimizer, sample_training_data):
        """Test config evaluation"""
        config = CognitiveConfig()

        metrics = optimizer.evaluate_config(
            config=config,
            training_data=sample_training_data,
            seed=42,
        )

        assert "composite_score" in metrics
        assert "learning_efficiency" in metrics
        assert "consolidation_quality" in metrics
        assert "concepts_learned" in metrics
        assert metrics["concepts_learned"] >= 0

    def test_evaluate_with_validation(self, optimizer, sample_training_data):
        """Test evaluation with validation data"""
        config = CognitiveConfig()
        validation_data = [
            {"input": "cat", "concept": "animal"},
            {"input": "car", "concept": "vehicle"},
        ]

        metrics = optimizer.evaluate_config(
            config=config,
            training_data=sample_training_data,
            validation_data=validation_data,
            seed=42,
        )

        assert "recall_accuracy" in metrics
        assert 0.0 <= metrics["recall_accuracy"] <= 1.0

    def test_create_synthetic_dataset(self, optimizer):
        """Test synthetic dataset creation"""
        dataset = optimizer.create_synthetic_dataset(
            num_concepts=5,
            samples_per_concept=3,
        )

        assert len(dataset) == 5 * 3
        assert all("input" in item for item in dataset)
        assert all("concept" in item for item in dataset)
        assert all("context" in item for item in dataset)

    def test_save_and_load_config(self, optimizer):
        """Test saving and loading optimized config"""
        config = CognitiveConfig(
            hebbian_learning_rate=0.05,
            max_association_strength=1.5,
        )

        optimizer.save_config(config, "test_config")
        loaded_config = optimizer.load_config("test_config")

        assert loaded_config.hebbian_learning_rate == config.hebbian_learning_rate
        assert loaded_config.max_association_strength == config.max_association_strength

    @pytest.mark.slow
    def test_optimize_small_study(self, optimizer, sample_training_data):
        """Test running a small optimization study"""
        study, best_config = optimizer.optimize(
            training_data=sample_training_data,
            n_trials=5,  # Small for testing
            study_name="test_study",
            seed=42,
        )

        assert study is not None
        assert best_config is not None
        assert isinstance(best_config, CognitiveConfig)
        assert study.best_value is not None

        # Verify best config file exists
        config_file = optimizer.storage_path / "test_study_best_config.json"
        assert config_file.exists()
