#!/usr/bin/env python3
"""
Ingestion Service - Core logic for ingestion and quarantine management
Handles validation, deduplication, and quarantine buffer management
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import redis

from .models import IngestItem, IngestStatus

logger = logging.getLogger(__name__)


class QuarantineBuffer:
    """Manages quarantine buffer for ingested items"""

    def __init__(
        self,
        storage_type: str = "local",
        storage_path: Optional[str] = None,
        redis_url: Optional[str] = None,
    ):
        """
        Initialize quarantine buffer

        Args:
            storage_type: "local" or "redis"
            storage_path: Path for local storage (default: ./quarantine)
            redis_url: Redis connection URL (default: redis://localhost:6379)
        """
        self.storage_type = storage_type

        if storage_type == "local":
            self.storage_path = Path(storage_path or "./quarantine")
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self._seen_hashes: Set[str] = set()
            self._load_seen_hashes()
        elif storage_type == "redis":
            self.redis_client = redis.from_url(
                redis_url or "redis://localhost:6379", decode_responses=True
            )
            self.redis_key_prefix = "quarantine:"
            self.redis_hash_set_key = "quarantine:hashes"
        else:
            raise ValueError(f"Unknown storage type: {storage_type}")

    def _load_seen_hashes(self):
        """Load seen hashes from disk"""
        hash_file = self.storage_path / "seen_hashes.json"
        if hash_file.exists():
            try:
                with open(hash_file, "r") as f:
                    self._seen_hashes = set(json.load(f))
            except Exception as e:
                logger.warning(f"Failed to load seen hashes: {e}")

    def _save_seen_hashes(self):
        """Save seen hashes to disk"""
        if self.storage_type == "local":
            hash_file = self.storage_path / "seen_hashes.json"
            try:
                with open(hash_file, "w") as f:
                    json.dump(list(self._seen_hashes), f)
            except Exception as e:
                logger.warning(f"Failed to save seen hashes: {e}")

    def add_item(self, item: IngestItem) -> bool:
        """
        Add item to quarantine buffer

        Returns:
            True if added, False if duplicate
        """
        if self.storage_type == "local":
            if item.content_hash in self._seen_hashes:
                return False
            self._seen_hashes.add(item.content_hash)
            self._save_seen_hashes()

            item_file = self.storage_path / f"{item.item_id}.json"
            with open(item_file, "w") as f:
                json.dump(item.model_dump(), f, default=str)
            return True
        else:  # redis
            if self.redis_client.sismember(self.redis_hash_set_key, item.content_hash):
                return False
            self.redis_client.sadd(self.redis_hash_set_key, item.content_hash)

            item_key = f"{self.redis_key_prefix}item:{item.item_id}"
            self.redis_client.set(item_key, item.model_dump_json(), ex=86400 * 7)  # 7 days TTL
            return True

    def get_item(self, item_id: str) -> Optional[IngestItem]:
        """Get item from quarantine buffer"""
        if self.storage_type == "local":
            item_file = self.storage_path / f"{item_id}.json"
            if not item_file.exists():
                return None
            try:
                with open(item_file, "r") as f:
                    data = json.load(f)
                return IngestItem(**data)
            except Exception as e:
                logger.error(f"Failed to load item {item_id}: {e}")
                return None
        else:  # redis
            item_key = f"{self.redis_key_prefix}item:{item_id}"
            data = self.redis_client.get(item_key)
            if not data:
                return None
            try:
                return IngestItem.model_validate_json(data)
            except Exception as e:
                logger.error(f"Failed to parse item {item_id}: {e}")
                return None

    def list_items(
        self, status: Optional[IngestStatus] = None, limit: int = 100
    ) -> List[IngestItem]:
        """List items in quarantine buffer"""
        items = []

        if self.storage_type == "local":
            for item_file in self.storage_path.glob("*.json"):
                if item_file.name == "seen_hashes.json":
                    continue
                try:
                    with open(item_file, "r") as f:
                        data = json.load(f)
                    item = IngestItem(**data)
                    if status is None or item.status == status:
                        items.append(item)
                        if len(items) >= limit:
                            break
                except Exception as e:
                    logger.warning(f"Failed to load item from {item_file}: {e}")
        else:  # redis
            pattern = f"{self.redis_key_prefix}item:*"
            for key in self.redis_client.scan_iter(match=pattern, count=limit):
                data = self.redis_client.get(key)
                if data:
                    try:
                        item = IngestItem.model_validate_json(data)
                        if status is None or item.status == status:
                            items.append(item)
                            if len(items) >= limit:
                                break
                    except Exception as e:
                        logger.warning(f"Failed to parse item from {key}: {e}")

        return items

    def is_duplicate(self, content_hash: str) -> bool:
        """Check if content hash already exists"""
        if self.storage_type == "local":
            return content_hash in self._seen_hashes
        else:  # redis
            return self.redis_client.sismember(self.redis_hash_set_key, content_hash) == 1


class IngestionService:
    """Service for handling content ingestion with validation"""

    def __init__(self, quarantine_buffer: QuarantineBuffer):
        self.quarantine_buffer = quarantine_buffer

    def hash_content(self, content: str) -> str:
        """Generate SHA-256 hash of content"""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def validate_license(self, license_type: Optional[str]) -> bool:
        """
        Validate license type

        For now, accepts common open licenses. Can be extended with more validation.
        """
        if license_type is None:
            return True  # Allow if no license specified

        allowed_licenses = {
            "MIT",
            "Apache-2.0",
            "BSD-3-Clause",
            "BSD-2-Clause",
            "CC-BY",
            "CC-BY-SA",
            "CC0",
            "Public Domain",
        }
        return license_type in allowed_licenses

    def check_robots_txt(self, url: Optional[str]) -> bool:
        """
        Check robots.txt compliance

        For now, returns True. In production, would fetch and parse robots.txt.
        """
        if url is None:
            return True

        # Stub implementation - in production would:
        # 1. Parse URL to get domain
        # 2. Fetch robots.txt
        # 3. Check if path is allowed
        # 4. Return result

        return True

    def submit_item(
        self,
        content: str,
        source_url: Optional[str] = None,
        license_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IngestItem:
        """
        Submit item for ingestion

        Returns:
            IngestItem with status set based on validation results
        """
        # Generate content hash
        content_hash = self.hash_content(content)

        # Check for duplicates
        if self.quarantine_buffer.is_duplicate(content_hash):
            raise ValueError(f"Duplicate content detected (hash: {content_hash[:16]}...)")

        # Validate license
        if not self.validate_license(license_type):
            raise ValueError(f"Invalid or unsupported license: {license_type}")

        # Check robots.txt
        robots_allowed = self.check_robots_txt(source_url)

        # Create ingest item
        item_id = f"ingest_{datetime.now().timestamp()}_{content_hash[:8]}"
        item = IngestItem(
            item_id=item_id,
            content=content,
            source_url=source_url,
            content_hash=content_hash,
            license_type=license_type,
            robots_txt_allowed=robots_allowed,
            status=IngestStatus.QUARANTINED,
            metadata=metadata or {},
        )

        # Add to quarantine buffer
        if not self.quarantine_buffer.add_item(item):
            raise ValueError("Failed to add item to quarantine buffer")

        logger.info(f"Item {item_id} submitted and quarantined")
        return item
