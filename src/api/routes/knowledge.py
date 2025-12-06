"""
Knowledge search routes - Hybrid retrieval combining spike similarity and vector search
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ...core.cognitive_architecture import CognitiveArchitecture
from ...storage.qdrant_store import QdrantStore

router = APIRouter()
logger = logging.getLogger(__name__)

# Global Qdrant store instance
_qdrant_store: Optional[QdrantStore] = None


def get_qdrant_store() -> Optional[QdrantStore]:
    """Get or create Qdrant store instance"""
    global _qdrant_store
    if _qdrant_store is None:
        try:
            _qdrant_store = QdrantStore()
        except ImportError:
            logger.warning("Qdrant not available - install qdrant-client for vector search")
            return None
        except Exception as e:
            logger.warning(f"Failed to initialize Qdrant: {e}")
            return None
    return _qdrant_store


# Request/Response models
class KnowledgeSearchRequest(BaseModel):
    query_vector: Optional[List[float]] = Field(None, description="Query vector (128-dim)")
    query_spike_pattern: Optional[Dict[int, bool]] = Field(None, description="Query spike pattern")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of results")
    vector_weight: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Weight for vector similarity"
    )
    spike_weight: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Weight for spike similarity"
    )


class KnowledgeSearchResult(BaseModel):
    semantic_id: str
    concept: str
    combined_score: float
    vector_score: Optional[float] = None
    spike_score: Optional[float] = None
    activation_pattern: Dict[int, bool]
    metadata: Dict[str, Any] = {}


class KnowledgeSearchResponse(BaseModel):
    success: bool
    results: List[KnowledgeSearchResult]
    count: int
    query_info: Dict[str, Any]


# Dependency injection placeholder (will be overridden in main.py)
def get_cognitive_architecture():
    """Dependency injection for cognitive architecture"""
    pass


@router.post("/search", response_model=KnowledgeSearchResponse)
async def search_knowledge(
    request: KnowledgeSearchRequest,
    architecture: CognitiveArchitecture = Depends(get_cognitive_architecture),
):
    """
    Hybrid knowledge search combining spike similarity and vector search.

    Requires at least one of query_vector or query_spike_pattern.
    """
    try:
        # Validate request
        if not request.query_vector and not request.query_spike_pattern:
            raise HTTPException(
                status_code=400,
                detail="At least one of query_vector or query_spike_pattern must be provided",
            )

        # Get Qdrant store
        qdrant_store = get_qdrant_store()
        if not qdrant_store:
            raise HTTPException(
                status_code=503,
                detail="Qdrant vector store not available. Install qdrant-client and ensure Qdrant is running.",
            )

        # Perform hybrid search
        results = qdrant_store.hybrid_search(
            query_vector=request.query_vector,
            query_spike_pattern=request.query_spike_pattern,
            limit=request.limit,
            vector_weight=request.vector_weight,
            spike_weight=request.spike_weight,
        )

        # Convert to response format
        search_results = [
            KnowledgeSearchResult(
                semantic_id=result["semantic_id"],
                concept=result["concept"],
                combined_score=result["combined_score"],
                vector_score=result.get("vector_score"),
                spike_score=result.get("spike_score"),
                activation_pattern=result.get("activation_pattern", {}),
                metadata=result.get("metadata", {}),
            )
            for result in results
        ]

        query_info = {
            "has_vector_query": request.query_vector is not None,
            "has_spike_query": request.query_spike_pattern is not None,
            "vector_weight": request.vector_weight,
            "spike_weight": request.spike_weight,
            "limit": request.limit,
        }

        return KnowledgeSearchResponse(
            success=True,
            results=search_results,
            count=len(search_results),
            query_info=query_info,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching knowledge: {e}")
        raise HTTPException(status_code=500, detail=f"Knowledge search failed: {str(e)}")


@router.post("/index-semantic")
async def index_semantic_memory(
    architecture: CognitiveArchitecture = Depends(get_cognitive_architecture),
):
    """
    Index all semantic memories from the architecture into Qdrant.

    Useful for syncing semantic memory with vector store.
    """
    try:
        qdrant_store = get_qdrant_store()
        if not qdrant_store:
            raise HTTPException(
                status_code=503,
                detail="Qdrant vector store not available",
            )

        indexed_count = 0

        # Index all semantic memories
        for semantic_id, semantic in architecture.semantic_memory.semantic_memories.items():
            qdrant_store.upsert_semantic_memory(
                semantic_id=semantic.semantic_id,
                vector=semantic.semantic_vector,
                concept=semantic.concept,
                activation_pattern=semantic.activation_pattern,
                metadata={
                    "consolidated_count": semantic.consolidated_count,
                    "confidence": semantic.confidence,
                    "consolidated_traces": list(semantic.consolidated_traces),
                },
            )
            indexed_count += 1

        return {
            "success": True,
            "message": f"Indexed {indexed_count} semantic memories",
            "indexed_count": indexed_count,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error indexing semantic memory: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")
