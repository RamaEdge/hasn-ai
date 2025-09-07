"""
Request models for HASN-AI API endpoints.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class NeuralPatternRequest(BaseModel):
    """Request model for neural pattern processing."""
    
    pattern: Dict[int, bool] = Field(
        ..., 
        description="Neural spike pattern as a dictionary of neuron_id -> spike_boolean"
    )
    context: Optional[Dict[str, Any]] = Field(
        None, 
        description="Optional context information for the pattern"
    )


class TextToPatternRequest(BaseModel):
    """Request model for converting text to neural patterns."""
    
    text: str = Field(..., description="Text to convert to neural pattern")
    encoding_method: str = Field(
        "simple", 
        description="Method for encoding text to neural patterns"
    )
    context: Optional[Dict[str, Any]] = Field(
        None, 
        description="Optional context information"
    )


class BrainSimulationRequest(BaseModel):
    """Request model for brain network simulation."""
    
    duration: float = Field(
        100.0, 
        description="Simulation duration in milliseconds"
    )
    input_pattern: Optional[Dict[int, bool]] = Field(
        None, 
        description="Input spike pattern for the simulation"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        None, 
        description="Optional simulation parameters"
    )


class BatchProcessingRequest(BaseModel):
    """Request model for batch processing multiple patterns."""
    
    patterns: List[Dict[int, bool]] = Field(
        ..., 
        description="List of neural patterns to process"
    )
    context: Optional[Dict[str, Any]] = Field(
        None, 
        description="Optional context information for all patterns"
    )
    parallel: bool = Field(
        False, 
        description="Whether to process patterns in parallel"
    )


class TrainingRequest(BaseModel):
    """Request model for brain network training."""
    
    input_data: List[Dict[str, Any]] = Field(
        ..., 
        description="Training data with input patterns and labels"
    )
    epochs: int = Field(
        1, 
        description="Number of training epochs"
    )
    learning_rate: float = Field(
        0.01, 
        description="Learning rate for training"
    )
    batch_size: Optional[int] = Field(
        None, 
        description="Batch size for training (optional)"
    )


class InteractiveTrainingRequest(BaseModel):
    """Request model for interactive training."""
    
    input_data: List[Dict[str, Any]] = Field(
        ..., 
        description="Training data with input patterns and optional labels"
    )
    epochs: int = Field(
        1, 
        description="Number of training epochs"
    )
    learning_rate: float = Field(
        0.01, 
        description="Learning rate for training"
    )
    continuous: bool = Field(
        False, 
        description="Whether to run continuous training"
    )
    context: Optional[Dict[str, Any]] = Field(
        None, 
        description="Optional context information for training"
    )


class ChatRequest(BaseModel):
    """Request model for chat interactions with the brain network."""
    
    message: str = Field(..., description="Chat message to process")
    context: Optional[Dict[str, Any]] = Field(
        None, 
        description="Optional context information"
    )
    max_tokens: Optional[int] = Field(
        None, 
        description="Maximum number of tokens in response"
    )
    temperature: float = Field(
        0.7, 
        description="Temperature for response generation"
    )


class ModelConfigRequest(BaseModel):
    """Request model for brain network configuration."""
    
    num_neurons: int = Field(
        100, 
        description="Number of neurons in the network"
    )
    connectivity_prob: float = Field(
        0.1, 
        description="Probability of connection between neurons"
    )
    learning_rate: float = Field(
        0.01, 
        description="Learning rate for the network"
    )
    max_episodic_memories: Optional[int] = Field(
        None, 
        description="Maximum number of episodic memories (for cognitive networks)"
    )
    config_overrides: Optional[Dict[str, Any]] = Field(
        None, 
        description="Additional configuration overrides"
    )
