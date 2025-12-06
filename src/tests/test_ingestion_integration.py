#!/usr/bin/env python3
"""
Integration tests for ingestion and training pipeline
Tests end-to-end flow: submit -> quarantine -> consolidate -> train
"""

import shutil
import tempfile

import pytest

from core.simplified_brain_network import SimpleBrainNetwork
from ingestion.models import IngestStatus, TrainingJobStatus
from ingestion.replay_trainer import ReplayTrainer
from ingestion.service import IngestionService, QuarantineBuffer


class TestIngestionPipeline:
    """Integration tests for complete ingestion pipeline"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.buffer = QuarantineBuffer(storage_type="local", storage_path=self.temp_dir)
        self.service = IngestionService(self.buffer)
        self.brain_network = SimpleBrainNetwork(num_neurons=100, connectivity_prob=0.1)
        self.trainer = ReplayTrainer(self.buffer, brain_network=self.brain_network)

    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)

    def test_submit_to_quarantine_flow(self):
        """Test complete flow: submit -> appears in quarantine"""
        # Submit item
        item = self.service.submit_item(
            content="Test content for ingestion",
            source_url="https://example.com/article",
            license_type="MIT",
        )

        # Verify item is in quarantine
        assert item.status == IngestStatus.QUARANTINED

        # Verify item can be retrieved
        retrieved = self.buffer.get_item(item.item_id)
        assert retrieved is not None
        assert retrieved.content == "Test content for ingestion"

        # Verify item appears in list
        quarantined_items = self.buffer.list_items(status=IngestStatus.QUARANTINED)
        assert len(quarantined_items) >= 1
        assert any(i.item_id == item.item_id for i in quarantined_items)

    def test_deduplication_flow(self):
        """Test deduplication: duplicate skipped"""
        # Submit first item
        item1 = self.service.submit_item(content="Duplicate content")

        # Try to submit duplicate
        with pytest.raises(ValueError, match="Duplicate"):
            self.service.submit_item(content="Duplicate content")

        # Verify only one item exists
        all_items = self.buffer.list_items()
        assert len([i for i in all_items if i.content_hash == item1.content_hash]) == 1

    def test_consolidation_and_training_flow(self):
        """Test consolidation: replay updates network"""
        # Submit multiple items
        item1 = self.service.submit_item(content="Training content 1")
        self.service.submit_item(content="Training content 2")
        self.service.submit_item(content="Training content 3")

        # Consolidate and train
        result = self.trainer.consolidate(max_items=10)

        # Verify consolidation results
        assert result["items_found"] == 3
        assert result["job_id"] is not None
        assert result["items_processed"] == 3
        assert result["items_failed"] == 0

        # Verify items are processed
        processed_item1 = self.buffer.get_item(item1.item_id)
        assert processed_item1.status == IngestStatus.COMPLETED

        # Verify job exists
        job = self.trainer.jobs[result["job_id"]]
        assert job.status == TrainingJobStatus.COMPLETED
        assert job.items_processed == 3

    def test_metrics_endpoint_flow(self):
        """Test metrics endpoint: reports novelty and drift"""
        # Submit various items
        self.service.submit_item(content="Unique content 1")
        self.service.submit_item(content="Unique content 2")
        self.service.submit_item(content="Unique content 3")

        # Get metrics
        metrics = self.trainer.get_metrics()

        # Verify metrics structure
        assert "total_items" in metrics
        assert "novelty_score" in metrics
        assert "drift_score" in metrics
        assert "status_counts" in metrics

        # Verify novelty (should be high for unique content)
        assert metrics["novelty_score"] > 0
        assert metrics["novelty_score"] <= 1.0

        # Verify drift (should be low if no failures)
        assert metrics["drift_score"] >= 0
        assert metrics["drift_score"] <= 1.0

        # Verify status counts
        assert metrics["status_counts"]["quarantined"] >= 0
        assert metrics["status_counts"]["completed"] >= 0

    def test_complete_pipeline(self):
        """Test complete pipeline: submit -> quarantine -> consolidate -> metrics"""
        # Step 1: Submit items
        items = []
        for i in range(5):
            item = self.service.submit_item(content=f"Content item {i}", license_type="MIT")
            items.append(item)

        # Step 2: Verify all in quarantine
        quarantined = self.buffer.list_items(status=IngestStatus.QUARANTINED)
        assert len(quarantined) == 5

        # Step 3: Consolidate and train
        result = self.trainer.consolidate(max_items=10)
        assert result["items_found"] == 5
        assert result["items_processed"] == 5

        # Step 4: Verify all processed
        completed = self.buffer.list_items(status=IngestStatus.COMPLETED)
        assert len(completed) == 5

        # Step 5: Check metrics
        metrics = self.trainer.get_metrics()
        assert metrics["total_items"] == 5
        assert metrics["status_counts"]["completed"] == 5
        assert metrics["jobs_completed"] >= 1
