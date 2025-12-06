#!/usr/bin/env python3
"""
Ingestion Models - Data structures for ingestion and training pipeline
Defines IngestItem and TrainingJob models for safe continual learning
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class IngestStatus(str, Enum):
    """Status of an ingested item"""

    PENDING = "pending"
    QUARANTINED = "quarantined"
    PROCESSING = "processing"
    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"


class TrainingJobStatus(str, Enum):
    """Status of a training job"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class IngestItem(BaseModel):
    """Model for ingested content items"""

    item_id: str = Field(..., description="Unique identifier for the item")
    content: str = Field(..., description="Content text to ingest")
    source_url: Optional[str] = Field(None, description="Source URL if from web")
    content_hash: str = Field(..., description="SHA-256 hash of content for deduplication")
    license_type: Optional[str] = Field(None, description="License type (e.g., 'MIT', 'CC-BY')")
    robots_txt_allowed: bool = Field(
        default=True, description="Whether robots.txt allows ingestion"
    )
    status: IngestStatus = Field(default=IngestStatus.PENDING, description="Current status")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    processed_at: Optional[datetime] = Field(None, description="Processing timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class TrainingJob(BaseModel):
    """Model for training jobs that process ingested items"""

    job_id: str = Field(..., description="Unique identifier for the job")
    ingest_item_ids: list[str] = Field(..., description="List of ingested item IDs to process")
    status: TrainingJobStatus = Field(
        default=TrainingJobStatus.PENDING, description="Current status"
    )
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    items_processed: int = Field(default=0, description="Number of items processed")
    items_failed: int = Field(default=0, description="Number of items that failed")
    error_message: Optional[str] = Field(None, description="Error message if job failed")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Training metrics")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
