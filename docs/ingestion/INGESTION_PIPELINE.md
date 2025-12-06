# Ingestion & Continuous Training Pipeline

## Overview

The ingestion and continuous training pipeline provides safe continual learning with governance. It enables the HASN-AI system to ingest content from various sources, validate it for compliance, and train the brain network using Hebbian learning updates.

## Architecture

The pipeline consists of three main components:

1. **Ingestion Service**: Validates and submits content for ingestion
2. **Quarantine Buffer**: Stores ingested items before processing
3. **Replay Trainer**: Processes quarantined items and applies learning updates

```
Content Submission
       ↓
License & Robots.txt Validation
       ↓
Deduplication Check
       ↓
Quarantine Buffer (Local/Redis)
       ↓
Replay Trainer
       ↓
Hebbian Learning Updates
       ↓
Network Training
```

## Components

### IngestItem Model

Represents a single ingested content item with validation status.

**Fields:**
- `item_id`: Unique identifier
- `content`: Content text
- `source_url`: Optional source URL
- `content_hash`: SHA-256 hash for deduplication
- `license_type`: License type (MIT, Apache-2.0, CC-BY, etc.)
- `robots_txt_allowed`: Whether robots.txt allows ingestion
- `status`: Current status (pending, quarantined, processing, completed, rejected, failed)
- `metadata`: Additional metadata dictionary

### TrainingJob Model

Represents a training job that processes multiple ingested items.

**Fields:**
- `job_id`: Unique identifier
- `ingest_item_ids`: List of item IDs to process
- `status`: Current status (pending, running, completed, failed, cancelled)
- `items_processed`: Number of items successfully processed
- `items_failed`: Number of items that failed
- `metrics`: Training metrics dictionary

### QuarantineBuffer

Manages storage of ingested items before processing. Supports two storage backends:

- **Local Storage**: Filesystem-based storage (default)
- **Redis Storage**: Redis-based storage for distributed systems

**Key Methods:**
- `add_item(item)`: Add item to quarantine (returns False if duplicate)
- `get_item(item_id)`: Retrieve item by ID
- `list_items(status, limit)`: List items with optional status filter
- `is_duplicate(content_hash)`: Check if content hash already exists

### IngestionService

Handles content ingestion with validation.

**Key Methods:**
- `submit_item(content, source_url, license_type, metadata)`: Submit content for ingestion
- `hash_content(content)`: Generate SHA-256 hash
- `validate_license(license_type)`: Validate license type
- `check_robots_txt(url)`: Check robots.txt compliance

**Supported Licenses:**
- MIT
- Apache-2.0
- BSD-3-Clause
- BSD-2-Clause
- CC-BY
- CC-BY-SA
- CC0
- Public Domain

### ReplayTrainer

Processes quarantined items and applies Hebbian learning updates to the brain network.

**Key Methods:**
- `process_item(item)`: Process single item and apply learning
- `create_training_job(item_ids)`: Create new training job
- `run_training_job(job_id)`: Run training job to process multiple items
- `consolidate(max_items)`: Consolidate quarantined items into training job
- `get_metrics()`: Get training metrics including novelty and drift

## API Endpoints

### POST /ingest/submit

Submit content for ingestion.

**Request:**
```json
{
  "content": "Content text to ingest",
  "source_url": "https://example.com/article",
  "license_type": "MIT",
  "metadata": {
    "category": "science",
    "author": "John Doe"
  }
}
```

**Response:**
```json
{
  "success": true,
  "item_id": "ingest_1234567890_abc12345",
  "status": "quarantined",
  "message": "Item submitted and quarantined"
}
```

### GET /ingest/items/{item_id}

Get an ingested item by ID.

**Response:**
```json
{
  "item_id": "ingest_1234567890_abc12345",
  "content": "Content text",
  "status": "quarantined",
  "content_hash": "abc123...",
  "created_at": "2025-01-01T12:00:00"
}
```

### GET /ingest/items

List ingested items with optional status filter.

**Query Parameters:**
- `status`: Optional status filter (pending, quarantined, processing, completed, rejected, failed)
- `limit`: Maximum number of items to return (default: 100)

**Response:**
```json
{
  "items": [
    {
      "item_id": "ingest_1234567890_abc12345",
      "content": "Content text",
      "status": "quarantined"
    }
  ],
  "count": 1
}
```

### POST /train/consolidate

Consolidate quarantined items into a training job and process them.

**Request:**
```json
{
  "max_items": 100
}
```

**Response:**
```json
{
  "success": true,
  "items_found": 5,
  "job_id": "job_abc123def456",
  "items_processed": 5,
  "items_failed": 0,
  "status": "completed",
  "message": "Consolidation completed"
}
```

### GET /train/metrics

Get training metrics including novelty and drift.

**Response:**
```json
{
  "success": true,
  "metrics": {
    "total_items": 100,
    "status_counts": {
      "quarantined": 10,
      "completed": 85,
      "failed": 5
    },
    "novelty_score": 0.95,
    "drift_score": 0.05,
    "unique_content_hashes": 95,
    "jobs_total": 10,
    "jobs_completed": 9,
    "jobs_failed": 1
  }
}
```

### GET /train/jobs/{job_id}

Get a training job by ID.

**Response:**
```json
{
  "success": true,
  "job": {
    "job_id": "job_abc123def456",
    "status": "completed",
    "items_processed": 5,
    "items_failed": 0,
    "metrics": {
      "average_spike_count": 42.5
    }
  }
}
```

## Usage Examples

### Basic Ingestion Flow

```python
from ingestion.service import IngestionService, QuarantineBuffer
from ingestion.replay_trainer import ReplayTrainer
from core.simplified_brain_network import SimpleBrainNetwork

# Initialize components
buffer = QuarantineBuffer(storage_type="local", storage_path="./quarantine")
service = IngestionService(buffer)
brain_network = SimpleBrainNetwork(num_neurons=100, connectivity_prob=0.1)
trainer = ReplayTrainer(buffer, brain_network=brain_network)

# Submit content
item = service.submit_item(
    content="Example content for training",
    source_url="https://example.com/article",
    license_type="MIT"
)

# Consolidate and train
result = trainer.consolidate(max_items=10)
print(f"Processed {result['items_processed']} items")

# Get metrics
metrics = trainer.get_metrics()
print(f"Novelty score: {metrics['novelty_score']}")
print(f"Drift score: {metrics['drift_score']}")
```

### Using Redis Storage

```python
buffer = QuarantineBuffer(
    storage_type="redis",
    redis_url="redis://localhost:6379"
)
```

### Configuration

The ingestion system can be configured via environment variables:

- `QUARANTINE_STORAGE`: Storage type ("local" or "redis", default: "local")
- `QUARANTINE_PATH`: Path for local storage (default: "./quarantine")
- `REDIS_URL`: Redis connection URL (default: "redis://localhost:6379")

## Testing

Unit and integration tests are available in:

- `src/tests/test_ingestion.py`: Unit tests for individual components
- `src/tests/test_ingestion_integration.py`: Integration tests for complete pipeline

Run tests with:

```bash
pytest src/tests/test_ingestion.py -v
pytest src/tests/test_ingestion_integration.py -v
```

## Metrics Explained

### Novelty Score

Measures the uniqueness of ingested content. Calculated as the ratio of unique content hashes to total items. Higher values indicate more diverse content.

### Drift Score

Measures the ratio of failed items to total items. Lower values indicate more stable training. Higher values may indicate content quality issues or network problems.

### Status Counts

Breakdown of items by status:
- `pending`: Items awaiting processing
- `quarantined`: Items in quarantine buffer
- `processing`: Items currently being processed
- `completed`: Successfully processed items
- `rejected`: Items rejected during validation
- `failed`: Items that failed during processing

## Implementation Details

### Deduplication

Content deduplication uses SHA-256 hashing. Identical content will generate the same hash and be rejected as duplicates.

### Content to Pattern Conversion

The replay trainer converts text content to neural spike patterns using a simple hash-based mapping. Words are hashed to neuron IDs, creating activation patterns for training.

### Hebbian Learning

The replay trainer applies Hebbian learning updates by processing content patterns through the brain network's `step()` or `process_pattern()` methods, which automatically update synaptic weights based on co-activity.

## Future Enhancements

Potential improvements for future versions:

- Full robots.txt parsing and compliance checking
- More sophisticated content-to-pattern conversion
- Human-in-the-loop approval workflow
- Advanced drift detection algorithms
- Content quality scoring
- Automatic license detection from source

