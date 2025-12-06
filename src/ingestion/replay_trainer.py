#!/usr/bin/env python3
"""
Replay Trainer - Processes quarantined items and applies Hebbian learning updates
Implements safe continual learning with governance
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List

from .models import IngestItem, IngestStatus, TrainingJob, TrainingJobStatus
from .service import QuarantineBuffer

logger = logging.getLogger(__name__)


class ReplayTrainer:
    """Trains network on quarantined items using Hebbian learning"""

    def __init__(
        self,
        quarantine_buffer: QuarantineBuffer,
        brain_network=None,  # Will be injected via dependency
    ):
        self.quarantine_buffer = quarantine_buffer
        self.brain_network = brain_network
        self.jobs: Dict[str, TrainingJob] = {}

    def set_brain_network(self, brain_network):
        """Set the brain network to train"""
        self.brain_network = brain_network

    def process_item(self, item: IngestItem) -> Dict[str, Any]:
        """
        Process a single item and apply Hebbian learning updates

        Returns:
            Dict with processing results and metrics
        """
        if self.brain_network is None:
            raise ValueError("Brain network not set")

        try:
            # Update item status
            item.status = IngestStatus.PROCESSING
            item.processed_at = datetime.now()

            # Convert content to neural pattern
            # For SimpleBrainNetwork, we need to convert text to spike pattern
            pattern = self._content_to_pattern(item.content)

            # Apply Hebbian learning update
            # This simulates one training step with the pattern
            if hasattr(self.brain_network, "step"):
                spikes = self.brain_network.step(pattern)
                spike_count = sum(1 for v in spikes.values() if v)
            elif hasattr(self.brain_network, "process_pattern"):
                result = self.brain_network.process_pattern(pattern)
                spike_count = result.get("active_neurons", 0)
            else:
                # Fallback: just mark as processed
                spike_count = 0

            # Update item status
            item.status = IngestStatus.COMPLETED

            metrics = {
                "spike_count": spike_count,
                "content_length": len(item.content),
                "processed_at": item.processed_at.isoformat(),
            }

            logger.info(f"Processed item {item.item_id}: {spike_count} spikes")
            return metrics

        except Exception as e:
            item.status = IngestStatus.FAILED
            logger.error(f"Failed to process item {item.item_id}: {e}")
            raise

    def _content_to_pattern(self, content: str) -> Dict[int, bool]:
        """
        Convert content text to neural spike pattern

        Simple implementation: hash words to neuron IDs
        """
        words = content.lower().split()
        pattern = {}

        for word in words:
            # Simple hash-based mapping to neuron IDs
            neuron_id = hash(word) % 1000  # Assuming 1000 neurons
            if neuron_id < 0:
                neuron_id = -neuron_id
            pattern[neuron_id] = True

        return pattern

    def create_training_job(self, item_ids: List[str]) -> TrainingJob:
        """Create a new training job"""
        job_id = f"job_{uuid.uuid4().hex[:16]}"
        job = TrainingJob(job_id=job_id, ingest_item_ids=item_ids, status=TrainingJobStatus.PENDING)
        self.jobs[job_id] = job
        return job

    def run_training_job(self, job_id: str) -> TrainingJob:
        """
        Run a training job to process multiple items

        Returns:
            Updated TrainingJob with results
        """
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")

        job = self.jobs[job_id]
        job.status = TrainingJobStatus.RUNNING
        job.started_at = datetime.now()

        items_processed = 0
        items_failed = 0
        all_metrics = []

        try:
            for item_id in job.ingest_item_ids:
                item = self.quarantine_buffer.get_item(item_id)
                if item is None:
                    logger.warning(f"Item {item_id} not found in quarantine")
                    items_failed += 1
                    continue

                try:
                    metrics = self.process_item(item)
                    items_processed += 1
                    all_metrics.append(metrics)
                except Exception as e:
                    logger.error(f"Failed to process item {item_id}: {e}")
                    items_failed += 1

            job.status = TrainingJobStatus.COMPLETED
            job.completed_at = datetime.now()
            job.items_processed = items_processed
            job.items_failed = items_failed
            job.metrics = {
                "total_items": len(job.ingest_item_ids),
                "items_processed": items_processed,
                "items_failed": items_failed,
                "average_spike_count": sum(m.get("spike_count", 0) for m in all_metrics)
                / max(len(all_metrics), 1),
            }

            logger.info(
                f"Job {job_id} completed: {items_processed} processed, {items_failed} failed"
            )

        except Exception as e:
            job.status = TrainingJobStatus.FAILED
            job.error_message = str(e)
            logger.error(f"Job {job_id} failed: {e}")

        return job

    def consolidate(self, max_items: int = 100) -> Dict[str, Any]:
        """
        Consolidate quarantined items into a training job

        Returns:
            Dict with consolidation results
        """
        # Get pending/quarantined items
        items = self.quarantine_buffer.list_items(status=IngestStatus.QUARANTINED, limit=max_items)

        if not items:
            return {"items_found": 0, "job_id": None, "message": "No items to consolidate"}

        # Create training job
        item_ids = [item.item_id for item in items]
        job = self.create_training_job(item_ids)

        # Run the job
        job = self.run_training_job(job.job_id)

        return {
            "items_found": len(items),
            "job_id": job.job_id,
            "items_processed": job.items_processed,
            "items_failed": job.items_failed,
            "status": job.status.value,
        }

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get training metrics including novelty and drift

        Returns:
            Dict with metrics
        """
        all_items = self.quarantine_buffer.list_items(limit=1000)

        status_counts = {}
        for status in IngestStatus:
            status_counts[status.value] = sum(1 for item in all_items if item.status == status)

        # Calculate novelty (items with unique content hashes)
        unique_hashes = set(item.content_hash for item in all_items)
        novelty_score = len(unique_hashes) / max(len(all_items), 1)

        # Simple drift metric: ratio of failed items
        drift_score = status_counts.get(IngestStatus.FAILED.value, 0) / max(len(all_items), 1)

        return {
            "total_items": len(all_items),
            "status_counts": status_counts,
            "novelty_score": novelty_score,
            "drift_score": drift_score,
            "unique_content_hashes": len(unique_hashes),
            "jobs_total": len(self.jobs),
            "jobs_completed": sum(
                1 for j in self.jobs.values() if j.status == TrainingJobStatus.COMPLETED
            ),
            "jobs_failed": sum(
                1 for j in self.jobs.values() if j.status == TrainingJobStatus.FAILED
            ),
        }
