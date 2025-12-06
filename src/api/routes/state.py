"""
State management routes - Save/load brain snapshots
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from ...storage.cognitive_serializer import CognitiveArchitectureSerializer
from ...core.cognitive_architecture import CognitiveArchitecture

router = APIRouter()
logger = logging.getLogger(__name__)

# Global serializer instance
_serializer: Optional[CognitiveArchitectureSerializer] = None


def get_serializer() -> CognitiveArchitectureSerializer:
    """Get or create serializer instance"""
    global _serializer
    if _serializer is None:
        _serializer = CognitiveArchitectureSerializer()
    return _serializer


# Request/Response models
class SaveStateRequest(BaseModel):
    snapshot_id: Optional[str] = Field(None, description="Optional snapshot ID (generated if not provided)")
    description: Optional[str] = Field(None, description="Optional description for the snapshot")


class SaveStateResponse(BaseModel):
    success: bool
    snapshot_id: str
    message: str


class LoadStateResponse(BaseModel):
    success: bool
    message: str
    snapshot_info: dict


class ListSnapshotsResponse(BaseModel):
    success: bool
    snapshots: list
    count: int


# Dependency injection placeholder (will be overridden in main.py)
def get_cognitive_architecture():
    """Dependency injection for cognitive architecture"""
    pass


@router.post("/save", response_model=SaveStateResponse)
async def save_state(
    request: SaveStateRequest,
    architecture: CognitiveArchitecture = Depends(get_cognitive_architecture),
):
    """
    Save current cognitive architecture state to snapshot.
    
    Returns snapshot ID for later retrieval.
    """
    try:
        serializer = get_serializer()
        
        snapshot_id = serializer.save(
            architecture=architecture,
            snapshot_id=request.snapshot_id,
            description=request.description,
        )
        
        return SaveStateResponse(
            success=True,
            snapshot_id=snapshot_id,
            message=f"State saved successfully as snapshot '{snapshot_id}'",
        )
    
    except Exception as e:
        logger.error(f"Error saving state: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save state: {str(e)}")


@router.get("/load/{snapshot_id}", response_model=LoadStateResponse)
async def load_state(
    snapshot_id: str,
    architecture: CognitiveArchitecture = Depends(get_cognitive_architecture),
):
    """
    Load cognitive architecture state from snapshot.
    
    Restores the architecture to the saved state.
    """
    try:
        serializer = get_serializer()
        
        # Load snapshot into existing architecture
        restored_architecture = serializer.load(snapshot_id, architecture)
        
        # Get snapshot info
        snapshots = serializer.list_snapshots()
        snapshot_info = next((s for s in snapshots if s["snapshot_id"] == snapshot_id), None)
        
        if not snapshot_info:
            snapshot_info = {"snapshot_id": snapshot_id, "status": "loaded"}
        
        return LoadStateResponse(
            success=True,
            message=f"State loaded successfully from snapshot '{snapshot_id}'",
            snapshot_info=snapshot_info,
        )
    
    except FileNotFoundError as e:
        logger.error(f"Snapshot not found: {e}")
        raise HTTPException(status_code=404, detail=f"Snapshot '{snapshot_id}' not found")
    
    except Exception as e:
        logger.error(f"Error loading state: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load state: {str(e)}")


@router.get("/list", response_model=ListSnapshotsResponse)
async def list_snapshots():
    """
    List all available snapshots.
    """
    try:
        serializer = get_serializer()
        snapshots = serializer.list_snapshots()
        
        return ListSnapshotsResponse(
            success=True,
            snapshots=snapshots,
            count=len(snapshots),
        )
    
    except Exception as e:
        logger.error(f"Error listing snapshots: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list snapshots: {str(e)}")


