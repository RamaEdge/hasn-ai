#!/usr/bin/env python3
"""
Cognitive Layer Models - Pydantic models for cognitive architecture
Defines data structures for sensory encoding, memory systems, and cognitive layers
"""

from datetime import datetime
from typing import Dict, List, Optional, Set, Any, Union
from pydantic import BaseModel, Field
import numpy as np


class CognitiveConfig(BaseModel):
    """Configuration for cognitive layer architecture"""
    
    # Sensory encoding parameters
    sensory_encoding_rate: float = Field(default=10.0, description="Poisson spike rate for sensory encoding")
    embedding_dimension: int = Field(default=128, description="Dimension of input embeddings")
    spike_threshold: float = Field(default=0.5, description="Threshold for spike generation")
    
    # Memory system parameters
    max_episodic_memories: int = Field(default=1000, description="Maximum number of episodic memories")
    max_working_memory_items: int = Field(default=50, description="Maximum items in working memory")
    working_memory_ttl: float = Field(default=30.0, description="TTL for working memory items (seconds)")
    
    # Associative memory parameters
    hebbian_learning_rate: float = Field(default=0.01, description="Learning rate for Hebbian updates")
    association_threshold: float = Field(default=0.3, description="Threshold for forming associations")
    co_activity_window: float = Field(default=0.1, description="Time window for co-activity detection (seconds)")
    
    # Episodic memory parameters
    consolidation_threshold: float = Field(default=0.7, description="Threshold for memory consolidation")
    consolidation_rate: float = Field(default=0.1, description="Rate of consolidation process")
    memory_decay_rate: float = Field(default=0.001, description="Rate of memory decay")
    
    # Semantic memory parameters
    semantic_consolidation_threshold: int = Field(default=5, description="Number of exposures needed for semantic consolidation")
    semantic_merge_threshold: float = Field(default=0.8, description="Similarity threshold for merging semantic memories")
    
    # Executive layer parameters
    arbitration_threshold: float = Field(default=0.6, description="Threshold for executive arbitration")
    recall_probability: float = Field(default=0.7, description="Probability of recall vs consolidation")
    
    # Layer connectivity
    layer_connectivity: Dict[str, List[str]] = Field(
        default={
            "sensory": ["associative", "working"],
            "associative": ["working", "episodic"],
            "working": ["episodic", "executive"],
            "episodic": ["semantic", "executive"],
            "semantic": ["executive"],
            "executive": ["working", "episodic", "semantic"]
        },
        description="Connectivity between cognitive layers"
    )


class EpisodeTrace(BaseModel):
    """Represents a single episode trace with temporal and contextual information"""
    
    trace_id: str = Field(..., description="Unique identifier for the trace")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the episode occurred")
    duration: float = Field(default=1.0, description="Duration of the episode in seconds")
    
    # Sensory information
    sensory_input: str = Field(..., description="Original sensory input (e.g., text)")
    sensory_encoding: Dict[int, bool] = Field(..., description="Spike pattern from sensory encoding")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding of the input")
    
    # Contextual information
    context: Dict[str, Any] = Field(default_factory=dict, description="Contextual information")
    attention_weights: Optional[Dict[int, float]] = Field(None, description="Attention weights for different features")
    
    # Memory associations
    associations: Set[str] = Field(default_factory=set, description="IDs of associated traces")
    activation_strength: float = Field(default=1.0, description="Current activation strength")
    access_count: int = Field(default=1, description="Number of times this trace has been accessed")
    
    # Consolidation status
    consolidation_level: float = Field(default=0.0, description="Level of consolidation (0-1)")
    last_consolidation: Optional[datetime] = Field(None, description="Last consolidation timestamp")
    
    class Config:
        arbitrary_types_allowed = True


class WorkingMemoryItem(BaseModel):
    """Represents an item in working memory with TTL"""
    
    item_id: str = Field(..., description="Unique identifier for the item")
    content: Any = Field(..., description="Content of the working memory item")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the item was created")
    ttl: float = Field(default=30.0, description="Time to live in seconds")
    access_count: int = Field(default=1, description="Number of times accessed")
    last_access: datetime = Field(default_factory=datetime.now, description="Last access time")
    
    def is_expired(self) -> bool:
        """Check if the item has expired based on TTL"""
        from datetime import datetime, timedelta
        return datetime.now() > self.timestamp + timedelta(seconds=self.ttl)
    
    def update_access(self):
        """Update access information"""
        self.access_count += 1
        self.last_access = datetime.now()


class SemanticMemory(BaseModel):
    """Represents consolidated semantic knowledge"""
    
    semantic_id: str = Field(..., description="Unique identifier for semantic memory")
    concept: str = Field(..., description="The semantic concept")
    consolidated_traces: Set[str] = Field(default_factory=set, description="Source episode traces")
    consolidated_count: int = Field(default=1, description="Number of consolidations")
    
    # Semantic representation
    semantic_vector: List[float] = Field(..., description="Consolidated semantic vector")
    activation_pattern: Dict[int, bool] = Field(..., description="Associated spike pattern")
    
    # Metadata
    creation_time: datetime = Field(default_factory=datetime.now, description="When created")
    last_update: datetime = Field(default_factory=datetime.now, description="Last update time")
    confidence: float = Field(default=0.5, description="Confidence in the semantic memory")
    
    class Config:
        arbitrary_types_allowed = True


class LayerState(BaseModel):
    """Represents the state of a cognitive layer"""
    
    layer_name: str = Field(..., description="Name of the cognitive layer")
    active_neurons: Set[int] = Field(default_factory=set, description="Currently active neurons")
    membrane_potentials: Dict[int, float] = Field(default_factory=dict, description="Membrane potentials")
    last_update: datetime = Field(default_factory=datetime.now, description="Last update time")
    
    # Layer-specific data
    layer_data: Dict[str, Any] = Field(default_factory=dict, description="Layer-specific information")
    
    class Config:
        arbitrary_types_allowed = True


class CognitiveState(BaseModel):
    """Overall state of the cognitive system"""
    
    timestamp: datetime = Field(default_factory=datetime.now, description="Current timestamp")
    
    # Layer states
    layer_states: Dict[str, LayerState] = Field(default_factory=dict, description="States of all layers")
    
    # Memory systems
    working_memory: Dict[str, WorkingMemoryItem] = Field(default_factory=dict, description="Working memory items")
    episodic_memories: Dict[str, EpisodeTrace] = Field(default_factory=dict, description="Episodic memories")
    semantic_memories: Dict[str, SemanticMemory] = Field(default_factory=dict, description="Semantic memories")
    
    # System metrics
    total_activations: int = Field(default=0, description="Total number of activations")
    consolidation_events: int = Field(default=0, description="Number of consolidation events")
    recall_events: int = Field(default=0, description="Number of recall events")
    
    class Config:
        arbitrary_types_allowed = True
