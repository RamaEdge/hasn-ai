"""
Response models for HASN-AI API endpoints.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class APIResponse(BaseModel):
    """Standard API response model."""
    
    success: bool = Field(..., description="Whether the request was successful")
    message: str = Field("", description="Response message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Response timestamp"
    )


class ErrorResponse(BaseModel):
    """Error response model."""
    
    success: bool = Field(False, description="Always false for error responses")
    error: str = Field(..., description="Error type or category")
    detail: str = Field(..., description="Detailed error message")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Error timestamp"
    )


class BrainProcessResponse(BaseModel):
    """Response model for brain processing operations."""
    
    success: bool = Field(..., description="Whether processing was successful")
    output_pattern: Dict[int, bool] = Field(
        ..., 
        description="Output neural spike pattern"
    )
    processing_time: float = Field(
        ..., 
        description="Processing time in milliseconds"
    )
    activity_metrics: Optional[Dict[str, float]] = Field(
        None, 
        description="Brain activity metrics"
    )
    context: Optional[Dict[str, Any]] = Field(
        None, 
        description="Optional context information"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Response timestamp"
    )


class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str = Field(..., description="Health status")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Health check timestamp"
    )
    brain_networks: Optional[Dict[str, bool]] = Field(
        None, 
        description="Status of brain network components"
    )
    system_info: Optional[Dict[str, Any]] = Field(
        None, 
        description="System information"
    )


class TrainingResponse(BaseModel):
    """Response model for training operations."""
    
    success: bool = Field(..., description="Whether training was successful")
    epochs_completed: int = Field(
        ..., 
        description="Number of epochs completed"
    )
    training_metrics: Optional[Dict[str, float]] = Field(
        None, 
        description="Training performance metrics"
    )
    brain_state: Optional[Dict[str, Any]] = Field(
        None, 
        description="Current brain network state"
    )
    processing_time: float = Field(
        ..., 
        description="Total training time in seconds"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Response timestamp"
    )


class SimulationResponse(BaseModel):
    """Response model for brain simulation operations."""
    
    success: bool = Field(..., description="Whether simulation was successful")
    duration: float = Field(..., description="Simulation duration in milliseconds")
    spike_raster: List[Dict[str, Any]] = Field(
        ..., 
        description="Spike raster data from simulation"
    )
    activity_summary: Optional[Dict[str, float]] = Field(
        None, 
        description="Summary of neural activity"
    )
    processing_time: float = Field(
        ..., 
        description="Simulation processing time in seconds"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Response timestamp"
    )


class ChatResponse(BaseModel):
    """Response model for chat interactions."""
    
    success: bool = Field(..., description="Whether chat processing was successful")
    response: str = Field(..., description="Chat response from the brain network")
    confidence: Optional[float] = Field(
        None, 
        description="Confidence score for the response"
    )
    processing_time: float = Field(
        ..., 
        description="Processing time in milliseconds"
    )
    context: Optional[Dict[str, Any]] = Field(
        None, 
        description="Optional context information"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Response timestamp"
    )
