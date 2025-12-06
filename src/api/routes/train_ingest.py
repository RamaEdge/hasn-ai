#!/usr/bin/env python3
"""
Training routes for ingestion pipeline
Implements /train/consolidate and /train/metrics endpoints
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from ingestion.replay_trainer import ReplayTrainer
from ingestion.service import QuarantineBuffer

router = APIRouter()
logger = logging.getLogger(__name__)

# Global instance (will be initialized in main.py)
_replay_trainer: Optional[ReplayTrainer] = None


def get_replay_trainer() -> ReplayTrainer:
    """Dependency injection for replay trainer"""
    global _replay_trainer
    if _replay_trainer is None:
        raise HTTPException(status_code=503, detail="Replay trainer not initialized")
    return _replay_trainer


class ConsolidateRequest(BaseModel):
    """Request model for /train/consolidate"""
    max_items: int = 100


class ConsolidateResponse(BaseModel):
    """Response model for /train/consolidate"""
    success: bool
    items_found: int
    job_id: Optional[str] = None
    items_processed: int = 0
    items_failed: int = 0
    status: str
    message: str


@router.post("/consolidate", response_model=ConsolidateResponse)
async def consolidate(
    request: ConsolidateRequest,
    trainer: ReplayTrainer = Depends(get_replay_trainer)
) -> ConsolidateResponse:
    """
    Consolidate quarantined items into a training job and process them
    
    Processes items from quarantine buffer and applies Hebbian learning updates.
    """
    try:
        result = trainer.consolidate(max_items=request.max_items)
        
        return ConsolidateResponse(
            success=True,
            items_found=result.get("items_found", 0),
            job_id=result.get("job_id"),
            items_processed=result.get("items_processed", 0),
            items_failed=result.get("items_failed", 0),
            status=result.get("status", "unknown"),
            message=result.get("message", "Consolidation completed")
        )
    
    except Exception as e:
        logger.error(f"Failed to consolidate: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/metrics")
async def get_metrics(
    trainer: ReplayTrainer = Depends(get_replay_trainer)
) -> Dict[str, Any]:
    """
    Get training metrics including novelty and drift
    
    Returns metrics about ingested items, training jobs, and system health.
    """
    try:
        metrics = trainer.get_metrics()
        return {
            "success": True,
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/jobs/{job_id}")
async def get_job(
    job_id: str,
    trainer: ReplayTrainer = Depends(get_replay_trainer)
) -> Dict[str, Any]:
    """Get a training job by ID"""
    if job_id not in trainer.jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job = trainer.jobs[job_id]
    return {
        "success": True,
        "job": job.model_dump()
    }

