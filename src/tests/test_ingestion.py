#!/usr/bin/env python3
"""
Unit tests for ingestion and training pipeline
Tests IngestItem, TrainingJob models, IngestionService, QuarantineBuffer, and ReplayTrainer
"""

import shutil
import tempfile

import pytest

from core.simplified_brain_network import SimpleBrainNetwork
from ingestion.models import IngestItem, IngestStatus, TrainingJob, TrainingJobStatus
from ingestion.replay_trainer import ReplayTrainer
from ingestion.service import IngestionService, QuarantineBuffer


class TestIngestItem:
    """Test IngestItem model"""

    def test_create_ingest_item(self):
        """Test creating an IngestItem"""
        item = IngestItem(item_id="test_1", content="Test content", content_hash="abc123")
        assert item.item_id == "test_1"
        assert item.content == "Test content"
        assert item.content_hash == "abc123"
        assert item.status == IngestStatus.PENDING

    def test_ingest_item_metadata(self):
        """Test IngestItem with metadata"""
        item = IngestItem(
            item_id="test_2",
            content="Test content",
            content_hash="abc123",
            metadata={"source": "test", "category": "example"},
        )
        assert item.metadata["source"] == "test"
        assert item.metadata["category"] == "example"


class TestTrainingJob:
    """Test TrainingJob model"""

    def test_create_training_job(self):
        """Test creating a TrainingJob"""
        job = TrainingJob(job_id="job_1", ingest_item_ids=["item_1", "item_2"])
        assert job.job_id == "job_1"
        assert len(job.ingest_item_ids) == 2
        assert job.status == TrainingJobStatus.PENDING


class TestQuarantineBuffer:
    """Test QuarantineBuffer"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.buffer = QuarantineBuffer(storage_type="local", storage_path=self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)

    def test_add_item(self):
        """Test adding item to quarantine buffer"""
        item = IngestItem(item_id="test_1", content="Test content", content_hash="abc123")
        result = self.buffer.add_item(item)
        assert result is True

    def test_duplicate_detection(self):
        """Test duplicate detection"""
        item1 = IngestItem(item_id="test_1", content="Test content", content_hash="abc123")
        item2 = IngestItem(
            item_id="test_2", content="Different content", content_hash="abc123"  # Same hash
        )

        assert self.buffer.add_item(item1) is True
        assert self.buffer.add_item(item2) is False  # Duplicate

    def test_get_item(self):
        """Test retrieving item from buffer"""
        item = IngestItem(item_id="test_1", content="Test content", content_hash="abc123")
        self.buffer.add_item(item)

        retrieved = self.buffer.get_item("test_1")
        assert retrieved is not None
        assert retrieved.item_id == "test_1"
        assert retrieved.content == "Test content"

    def test_list_items(self):
        """Test listing items"""
        item1 = IngestItem(
            item_id="test_1",
            content="Test content 1",
            content_hash="abc123",
            status=IngestStatus.QUARANTINED,
        )
        item2 = IngestItem(
            item_id="test_2",
            content="Test content 2",
            content_hash="def456",
            status=IngestStatus.COMPLETED,
        )

        self.buffer.add_item(item1)
        self.buffer.add_item(item2)

        quarantined = self.buffer.list_items(status=IngestStatus.QUARANTINED)
        assert len(quarantined) == 1
        assert quarantined[0].item_id == "test_1"

        all_items = self.buffer.list_items()
        assert len(all_items) >= 2


class TestIngestionService:
    """Test IngestionService"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        buffer = QuarantineBuffer(storage_type="local", storage_path=self.temp_dir)
        self.service = IngestionService(buffer)

    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)

    def test_hash_content(self):
        """Test content hashing"""
        hash1 = self.service.hash_content("Test content")
        hash2 = self.service.hash_content("Test content")
        hash3 = self.service.hash_content("Different content")

        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 64  # SHA-256 hex length

    def test_validate_license(self):
        """Test license validation"""
        assert self.service.validate_license("MIT") is True
        assert self.service.validate_license("Apache-2.0") is True
        assert self.service.validate_license("CC-BY") is True
        assert self.service.validate_license("Invalid") is False
        assert self.service.validate_license(None) is True

    def test_submit_item(self):
        """Test submitting an item"""
        item = self.service.submit_item(content="Test content", license_type="MIT")
        assert item.status == IngestStatus.QUARANTINED
        assert item.license_type == "MIT"
        assert len(item.content_hash) == 64

    def test_submit_duplicate(self):
        """Test submitting duplicate content"""
        self.service.submit_item(content="Test content")

        with pytest.raises(ValueError, match="Duplicate"):
            self.service.submit_item(content="Test content")

    def test_submit_invalid_license(self):
        """Test submitting with invalid license"""
        with pytest.raises(ValueError, match="Invalid"):
            self.service.submit_item(content="Test content", license_type="InvalidLicense")


class TestReplayTrainer:
    """Test ReplayTrainer"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        buffer = QuarantineBuffer(storage_type="local", storage_path=self.temp_dir)
        brain_network = SimpleBrainNetwork(num_neurons=100, connectivity_prob=0.1)
        self.trainer = ReplayTrainer(buffer, brain_network=brain_network)

    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)

    def test_content_to_pattern(self):
        """Test content to pattern conversion"""
        pattern = self.trainer._content_to_pattern("hello world test")
        assert isinstance(pattern, dict)
        assert len(pattern) > 0

    def test_process_item(self):
        """Test processing an item"""
        from ingestion.service import IngestionService

        service = IngestionService(self.trainer.quarantine_buffer)
        item = service.submit_item(content="Test content for training")

        metrics = self.trainer.process_item(item)
        assert "spike_count" in metrics
        assert item.status == IngestStatus.COMPLETED

    def test_create_training_job(self):
        """Test creating a training job"""
        job = self.trainer.create_training_job(["item_1", "item_2"])
        assert job.job_id is not None
        assert len(job.ingest_item_ids) == 2
        assert job.status == TrainingJobStatus.PENDING

    def test_consolidate(self):
        """Test consolidation"""
        from ingestion.service import IngestionService

        service = IngestionService(self.trainer.quarantine_buffer)
        service.submit_item(content="Content 1")
        service.submit_item(content="Content 2")

        result = self.trainer.consolidate(max_items=10)
        assert result["items_found"] == 2
        assert result["job_id"] is not None

    def test_get_metrics(self):
        """Test getting metrics"""
        from ingestion.service import IngestionService

        service = IngestionService(self.trainer.quarantine_buffer)
        service.submit_item(content="Test content")

        metrics = self.trainer.get_metrics()
        assert "total_items" in metrics
        assert "novelty_score" in metrics
        assert "drift_score" in metrics
        assert metrics["total_items"] >= 1
