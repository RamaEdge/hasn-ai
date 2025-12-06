#!/usr/bin/env python3
"""
Ingestion routes - API endpoints for content ingestion
Implements /ingest/submit endpoint with validation
"""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ingestion.models import IngestItem
from ingestion.service import IngestionService, QuarantineBuffer

router = APIRouter()
logger = logging.getLogger(__name__)

# Global instances (will be initialized in main.py)
_ingestion_service: Optional[IngestionService] = None
_quarantine_buffer: Optional[QuarantineBuffer] = None


def get_ingestion_service() -> IngestionService:
    """Dependency injection for ingestion service"""
    global _ingestion_service
    if _ingestion_service is None:
        raise HTTPException(status_code=503, detail="Ingestion service not initialized")
    return _ingestion_service


class IngestSubmitRequest(BaseModel):
    """Request model for /ingest/submit"""

    content: str
    source_url: Optional[str] = None
    license_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class IngestSubmitResponse(BaseModel):
    """Response model for /ingest/submit"""

    success: bool
    item_id: str
    status: str
    message: str


@router.post("/submit", response_model=IngestSubmitResponse)
async def submit_item(
    request: IngestSubmitRequest, service: IngestionService = Depends(get_ingestion_service)
) -> IngestSubmitResponse:
    """
    Submit content for ingestion

    Validates license and robots.txt compliance, then adds to quarantine buffer.
    """
    try:
        item = service.submit_item(
            content=request.content,
            source_url=request.source_url,
            license_type=request.license_type,
            metadata=request.metadata,
        )

        return IngestSubmitResponse(
            success=True,
            item_id=item.item_id,
            status=item.status.value,
            message="Item submitted and quarantined",
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to submit item: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/items/{item_id}")
async def get_item(
    item_id: str, service: IngestionService = Depends(get_ingestion_service)
) -> IngestItem:
    """Get an ingested item by ID"""
    item = service.quarantine_buffer.get_item(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
    return item


@router.get("/items")
async def list_items(
    status: Optional[str] = None,
    limit: int = 100,
    service: IngestionService = Depends(get_ingestion_service),
) -> Dict[str, Any]:
    """List ingested items"""
    from ingestion.models import IngestStatus

    status_filter = None
    if status:
        try:
            status_filter = IngestStatus(status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

    items = service.quarantine_buffer.list_items(status=status_filter, limit=limit)

    return {"items": [item.model_dump() for item in items], "count": len(items)}
