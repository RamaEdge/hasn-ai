#!/usr/bin/env python3
"""
Snapshot Schema v1 - JSON schema for brain state serialization
Defines the structure for saving and loading complete cognitive architecture state
"""

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class SnapshotMetadata(BaseModel):
    """Metadata for a brain snapshot"""

    snapshot_id: str = Field(..., description="Unique snapshot identifier")
    version: str = Field(default="1.0", description="Snapshot schema version")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    backend_name: str = Field(..., description="Backend used (numpy, norse, etc.)")
    num_neurons: int = Field(..., description="Number of neurons in the network")
    description: Optional[str] = Field(None, description="Optional description")


class BackendState(BaseModel):
    """Serialized backend state"""

    backend_name: str = Field(..., description="Backend identifier")
    state: Dict[str, Any] = Field(..., description="Backend-specific state data")
    config: Dict[str, Any] = Field(default_factory=dict, description="Backend configuration")


class CognitiveLayerState(BaseModel):
    """State of a cognitive layer"""

    layer_name: str = Field(..., description="Layer name")
    state_data: Dict[str, Any] = Field(default_factory=dict, description="Layer-specific state")


class SnapshotV1(BaseModel):
    """
    Snapshot Schema v1 - Complete brain state serialization

    This schema captures:
    - Backend state (neuron states, weights, etc.)
    - Cognitive layer states (sensory, associative, working memory, episodic, semantic, executive)
    - System metrics and configuration
    """

    metadata: SnapshotMetadata = Field(..., description="Snapshot metadata")
    backend_state: BackendState = Field(..., description="Backend state")
    cognitive_layers: Dict[str, CognitiveLayerState] = Field(
        default_factory=dict, description="States of all cognitive layers"
    )
    system_state: Dict[str, Any] = Field(
        default_factory=dict, description="Overall system state (metrics, config, etc.)"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
