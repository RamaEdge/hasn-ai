#!/usr/bin/env python3
"""
Qdrant Vector Store - Integration for semantic memory vector search
Provides hybrid retrieval combining spike similarity and vector search
"""

import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = None
    Distance = None
    VectorParams = None
    PointStruct = None


class QdrantStore:
    """
    Qdrant vector store for semantic memory.
    
    Stores semantic memories as vectors with payload for hybrid retrieval.
    Collection: hasn_semantic_v1
    """
    
    COLLECTION_NAME = "hasn_semantic_v1"
    VECTOR_SIZE = 128  # Default vector dimension
    
    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize Qdrant client.
        
        Args:
            url: Qdrant server URL (default: localhost:6333)
            api_key: Optional API key for cloud Qdrant
        """
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "Qdrant client not available. Install with: pip install qdrant-client"
            )
        
        self.url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        
        # Initialize client
        if self.api_key:
            self.client = QdrantClient(url=self.url, api_key=self.api_key)
        else:
            self.client = QdrantClient(url=self.url)
        
        # Ensure collection exists
        self._ensure_collection()
        
        print(f"🔍 Qdrant store initialized: {self.url}")
        print(f"   Collection: {self.COLLECTION_NAME}")
    
    def _ensure_collection(self):
        """Ensure the collection exists, create if not"""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.COLLECTION_NAME not in collection_names:
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )
            print(f"✅ Created collection: {self.COLLECTION_NAME}")
        else:
            print(f"✅ Collection exists: {self.COLLECTION_NAME}")
    
    def upsert_semantic_memory(
        self,
        semantic_id: str,
        vector: List[float],
        concept: str,
        activation_pattern: Dict[int, bool],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Upsert a semantic memory into Qdrant.
        
        Args:
            semantic_id: Unique semantic memory ID
            concept: Concept name
            vector: Semantic vector (128-dim)
            activation_pattern: Spike activation pattern
            metadata: Optional additional metadata
            
        Returns:
            True if successful
        """
        if len(vector) != self.VECTOR_SIZE:
            raise ValueError(f"Vector size {len(vector)} doesn't match {self.VECTOR_SIZE}")
        
        payload = {
            "semantic_id": semantic_id,
            "concept": concept,
            "activation_pattern": activation_pattern,
            "created_at": datetime.now().isoformat(),
        }
        
        if metadata:
            payload.update(metadata)
        
        point = PointStruct(
            id=self._hash_id(semantic_id),
            vector=vector,
            payload=payload,
        )
        
        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=[point],
        )
        
        return True
    
    def search_by_vector(
        self,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Search semantic memories by vector similarity.
        
        Args:
            query_vector: Query vector (128-dim)
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            
        Returns:
            List of search results with scores
        """
        if len(query_vector) != self.VECTOR_SIZE:
            raise ValueError(f"Vector size {len(query_vector)} doesn't match {self.VECTOR_SIZE}")
        
        results = self.client.search(
            collection_name=self.COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
        )
        
        return [
            {
                "semantic_id": point.payload.get("semantic_id"),
                "concept": point.payload.get("concept"),
                "score": point.score,
                "activation_pattern": point.payload.get("activation_pattern"),
                "metadata": {k: v for k, v in point.payload.items() if k not in ["semantic_id", "concept", "activation_pattern"]},
            }
            for point in results
        ]
    
    def search_by_concept(self, concept: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search semantic memories by concept name.
        
        Args:
            concept: Concept name to search for
            limit: Maximum number of results
            
        Returns:
            List of matching semantic memories
        """
        results = self.client.scroll(
            collection_name=self.COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="concept",
                        match=MatchValue(value=concept),
                    )
                ]
            ),
            limit=limit,
        )
        
        return [
            {
                "semantic_id": point.payload.get("semantic_id"),
                "concept": point.payload.get("concept"),
                "activation_pattern": point.payload.get("activation_pattern"),
                "metadata": {k: v for k, v in point.payload.items() if k not in ["semantic_id", "concept", "activation_pattern"]},
            }
            for point in results[0]  # scroll returns (points, next_page_offset)
        ]
    
    def delete_semantic_memory(self, semantic_id: str) -> bool:
        """
        Delete a semantic memory from Qdrant.
        
        Args:
            semantic_id: Semantic memory ID to delete
            
        Returns:
            True if successful
        """
        self.client.delete(
            collection_name=self.COLLECTION_NAME,
            points_selector=[self._hash_id(semantic_id)],
        )
        
        return True
    
    def get_all_memories(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all semantic memories (for debugging/testing).
        
        Args:
            limit: Maximum number of memories to return
            
        Returns:
            List of all semantic memories
        """
        results = self.client.scroll(
            collection_name=self.COLLECTION_NAME,
            limit=limit,
        )
        
        return [
            {
                "semantic_id": point.payload.get("semantic_id"),
                "concept": point.payload.get("concept"),
                "activation_pattern": point.payload.get("activation_pattern"),
                "metadata": {k: v for k, v in point.payload.items() if k not in ["semantic_id", "concept", "activation_pattern"]},
            }
            for point in results[0]
        ]
    
    @staticmethod
    def _hash_id(semantic_id: str) -> int:
        """Convert semantic ID to integer for Qdrant point ID"""
        # Simple hash function - in production, use proper hash
        return hash(semantic_id) % (2**63)  # Qdrant uses int64
    
    def calculate_spike_similarity(
        self,
        pattern1: Dict[int, bool],
        pattern2: Dict[int, bool],
    ) -> float:
        """
        Calculate Jaccard similarity between two spike patterns.
        
        Args:
            pattern1: First spike pattern
            pattern2: Second spike pattern
            
        Returns:
            Similarity score between 0 and 1
        """
        set1 = set(i for i, spike in pattern1.items() if spike)
        set2 = set(i for i, spike in pattern2.items() if spike)
        
        if not set1 and not set2:
            return 1.0
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def hybrid_search(
        self,
        query_vector: Optional[List[float]],
        query_spike_pattern: Optional[Dict[int, bool]],
        limit: int = 10,
        vector_weight: float = 0.5,
        spike_weight: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining vector similarity and spike pattern similarity.
        
        Args:
            query_vector: Query vector (128-dim) - optional
            query_spike_pattern: Query spike pattern - optional
            limit: Maximum number of results
            vector_weight: Weight for vector similarity (0-1)
            spike_weight: Weight for spike similarity (0-1)
            
        Returns:
            List of search results with combined scores
        """
        results = []
        
        # 1. Vector search (if query_vector provided)
        vector_results = []
        if query_vector:
            vector_results = self.search_by_vector(query_vector, limit=limit * 2)
        
        # 2. Spike pattern search (if query_spike_pattern provided)
        spike_results = []
        if query_spike_pattern:
            # Get all memories and calculate spike similarity
            all_memories = self.get_all_memories(limit=limit * 2)
            for memory in all_memories:
                activation_pattern = memory.get("activation_pattern", {})
                similarity = self.calculate_spike_similarity(
                    query_spike_pattern,
                    activation_pattern,
                )
                if similarity > 0:
                    spike_results.append({
                        **memory,
                        "spike_score": similarity,
                    })
            # Sort by spike similarity
            spike_results.sort(key=lambda x: x["spike_score"], reverse=True)
            spike_results = spike_results[:limit * 2]
        
        # 3. Combine results
        # Create a map of semantic_id -> result
        combined_map = {}
        
        # Add vector results
        for result in vector_results:
            semantic_id = result["semantic_id"]
            if semantic_id not in combined_map:
                combined_map[semantic_id] = result
                combined_map[semantic_id]["vector_score"] = result.get("score", 0.0)
                combined_map[semantic_id]["spike_score"] = 0.0
            else:
                combined_map[semantic_id]["vector_score"] = result.get("score", 0.0)
        
        # Add spike results
        for result in spike_results:
            semantic_id = result["semantic_id"]
            if semantic_id not in combined_map:
                combined_map[semantic_id] = result
                combined_map[semantic_id]["vector_score"] = 0.0
                combined_map[semantic_id]["spike_score"] = result.get("spike_score", 0.0)
            else:
                combined_map[semantic_id]["spike_score"] = result.get("spike_score", 0.0)
        
        # 4. Calculate combined scores
        for semantic_id, result in combined_map.items():
            vector_score = result.get("vector_score", 0.0)
            spike_score = result.get("spike_score", 0.0)
            
            # Normalize weights if only one type of search
            if query_vector and query_spike_pattern:
                combined_score = (vector_weight * vector_score) + (spike_weight * spike_score)
            elif query_vector:
                combined_score = vector_score
            elif query_spike_pattern:
                combined_score = spike_score
            else:
                combined_score = 0.0
            
            result["combined_score"] = combined_score
            results.append(result)
        
        # 5. Sort by combined score and return top results
        results.sort(key=lambda x: x["combined_score"], reverse=True)
        return results[:limit]


