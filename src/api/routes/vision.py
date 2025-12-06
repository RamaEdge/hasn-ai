"""
Vision interaction routes - Store and query vision embeddings in episodic memory
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ...core.cognitive_architecture import CognitiveArchitecture
from ...core.cognitive_models import EpisodeTrace

router = APIRouter()
logger = logging.getLogger(__name__)


# Request/Response models
class VisionRequest(BaseModel):
    """Request model for vision operations"""

    embedding: List[float] = Field(..., description="CLIP embedding vector (client-side)")
    context: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Optional context metadata"
    )


class VisionIngestResponse(BaseModel):
    """Response model for vision ingest endpoint"""

    success: bool
    trace_id: str
    message: str


class VisionTrace(BaseModel):
    """Represents a retrieved vision trace"""

    trace_id: str
    similarity: float
    timestamp: str
    context: Dict[str, Any]
    embedding: Optional[List[float]] = None


class VisionQueryResponse(BaseModel):
    """Response model for vision query endpoint"""

    success: bool
    results: List[VisionTrace]
    count: int


# Dependency injection placeholder (will be overridden in main.py)
def get_cognitive_architecture():
    """Dependency injection for cognitive architecture"""
    pass


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    try:
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        # Check dimension compatibility
        if v1.shape != v2.shape:
            logger.warning(
                f"Dimension mismatch: {v1.shape} vs {v2.shape}. Returning 0.0 similarity."
            )
            return 0.0
        
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    except Exception as e:
        logger.warning(f"Error calculating cosine similarity: {e}")
        return 0.0


@router.post("/ingest", response_model=VisionIngestResponse)
async def ingest_vision_embedding(
    request: VisionRequest,
    architecture: CognitiveArchitecture = Depends(get_cognitive_architecture),
):
    """
    Store a vision embedding in episodic memory.

    The embedding is converted to spikes and stored as an episode trace.
    """
    try:
        # Validate embedding
        if not request.embedding:
            raise HTTPException(status_code=400, detail="Embedding cannot be empty")
        
        # Validate embedding is a list of numbers
        if not isinstance(request.embedding, list) or len(request.embedding) == 0:
            raise HTTPException(status_code=400, detail="Embedding must be a non-empty list")
        
        # Validate all elements are numbers
        if not all(isinstance(x, (int, float)) for x in request.embedding):
            raise HTTPException(status_code=400, detail="Embedding must contain only numbers")

        # Convert embedding to spikes
        spike_pattern = architecture.sensory_layer.encode_embedding_to_spikes(request.embedding)

        # Create episode trace with embedding
        trace = EpisodeTrace(
            trace_id=str(uuid.uuid4()),
            sensory_input="vision_embedding",
            sensory_encoding=spike_pattern,
            embedding=request.embedding,
            context=request.context,
        )

        # Store in episodic memory
        trace_id = architecture.episodic_memory.store_trace(trace)

        logger.info(f"Stored vision embedding: {trace_id}")

        return VisionIngestResponse(
            success=True,
            trace_id=trace_id,
            message=f"Vision embedding stored successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting vision embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Vision ingest failed: {str(e)}")


@router.post("/query", response_model=VisionQueryResponse)
async def query_vision_embeddings(
    request: VisionRequest,
    architecture: CognitiveArchitecture = Depends(get_cognitive_architecture),
    limit: int = Query(default=10, ge=1, le=100),
    similarity_threshold: float = Query(default=0.5, ge=0.0, le=1.0),
):
    """
    Query episodic memory for traces similar to the provided embedding.

    Uses cosine similarity to find related vision traces.
    """
    try:
        # Validate embedding
        if not request.embedding:
            raise HTTPException(status_code=400, detail="Query embedding cannot be empty")
        
        # Validate embedding is a list of numbers
        if not isinstance(request.embedding, list) or len(request.embedding) == 0:
            raise HTTPException(status_code=400, detail="Query embedding must be a non-empty list")
        
        # Validate all elements are numbers
        if not all(isinstance(x, (int, float)) for x in request.embedding):
            raise HTTPException(status_code=400, detail="Query embedding must contain only numbers")

        # Find similar traces using embedding similarity
        similar_traces = []
        for trace_id, trace in architecture.episodic_memory.traces.items():
            # Only consider traces with embeddings
            if trace.embedding:
                similarity = _cosine_similarity(request.embedding, trace.embedding)
                if similarity >= similarity_threshold:
                    similar_traces.append((trace_id, trace, similarity))

        # Sort by similarity (descending)
        similar_traces.sort(key=lambda x: x[2], reverse=True)

        # Limit results
        similar_traces = similar_traces[:limit]

        # Convert to response format
        results = [
            VisionTrace(
                trace_id=trace_id,
                similarity=similarity,
                timestamp=trace.timestamp.isoformat(),
                context=trace.context,
                embedding=trace.embedding,
            )
            for trace_id, trace, similarity in similar_traces
        ]

        logger.info(f"Found {len(results)} similar vision traces")

        return VisionQueryResponse(
            success=True,
            results=results,
            count=len(results),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying vision embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Vision query failed: {str(e)}")

