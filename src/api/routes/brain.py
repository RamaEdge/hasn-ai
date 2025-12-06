"""
Brain processing routes for neural network operations
"""

import asyncio
import logging
import time
from typing import Any, Dict, List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)


# Simple request/response models
class NeuralPatternRequest(BaseModel):
    pattern: Dict[str, Dict[str, bool]]


class TextToPatternRequest(BaseModel):
    text: str
    encoding_method: str = "character_frequency"


class BrainSimulationRequest(BaseModel):
    input_data: Dict[str, Any]
    simulation_steps: int = 10


class BatchProcessingRequest(BaseModel):
    inputs: List[Dict[str, Any]]


class APIResponse(BaseModel):
    success: bool
    message: str = ""
    data: Dict[str, Any] = {}
    processing_time: float = 0.0


class BrainProcessResponse(BaseModel):
    success: bool
    output_pattern: Dict[str, bool]
    processing_time: float
    metadata: Dict[str, Any] = {}


# This will be injected from main.py
def get_brain_network():
    """Dependency injection for brain network"""
    pass


def get_advanced_brain():
    """Dependency injection for advanced brain"""
    pass


@router.post("/process", response_model=BrainProcessResponse)
async def process_neural_pattern(request: NeuralPatternRequest, brain=Depends(get_brain_network)):
    """
    Process a neural activation pattern through the brain network
    """
    start_time = time.time()

    try:
        logger.info(f"Processing neural pattern with {len(request.pattern)} modules")

        # Convert request pattern to brain-compatible format
        brain_pattern = {}
        for module_id, neurons in request.pattern.items():
            brain_pattern[int(module_id)] = {int(k): v for k, v in neurons.items()}

        # Process through brain network
        result = brain.process_pattern(brain_pattern)

        # Get current brain state
        brain_state = brain.get_brain_state() if hasattr(brain, "get_brain_state") else {}

        processing_time = (time.time() - start_time) * 1000  # ms

        return BrainProcessResponse(
            success=True,
            processing_result=result,
            brain_state=brain_state,
            neural_activity={
                "total_active_neurons": sum(len(neurons) for neurons in brain_pattern.values()),
                "active_modules": len(brain_pattern),
                "processing_steps": getattr(result, "steps", 100),
            },
            performance_metrics={
                "processing_time_ms": processing_time,
                "neurons_per_second": (
                    sum(len(neurons) for neurons in brain_pattern.values())
                    / (processing_time / 1000)
                    if processing_time > 0
                    else 0
                ),
            },
        )

    except Exception as e:
        logger.error(f"Error processing neural pattern: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.post("/text-to-pattern")
async def convert_text_to_pattern(request: TextToPatternRequest, brain=Depends(get_brain_network)):
    """
    Convert text input to neural activation pattern
    """
    try:
        logger.info(f"Converting text to pattern: '{request.text[:50]}...'")

        # Use brain's text processing if available
        if hasattr(brain, "text_to_pattern"):
            pattern = brain.text_to_pattern(request.text)
        else:
            # Fallback pattern generation
            pattern = _generate_basic_pattern(request.text)

        return APIResponse(
            success=True,
            message="Text converted to neural pattern",
            data={
                "text": request.text,
                "pattern": pattern,
                "processing_mode": request.processing_mode,
                "pattern_stats": {
                    "active_modules": len(pattern),
                    "total_neurons": sum(len(neurons) for neurons in pattern.values()),
                    "text_length": len(request.text),
                },
            },
        )

    except Exception as e:
        logger.error(f"Error converting text to pattern: {e}")
        raise HTTPException(status_code=500, detail=f"Text conversion failed: {str(e)}")


@router.post("/simulate")
async def simulate_brain_activity(
    request: BrainSimulationRequest, brain=Depends(get_advanced_brain)
):
    """
    Run brain simulation with specified parameters
    """
    start_time = time.time()

    try:
        logger.info(f"Running brain simulation for {request.simulation_steps} steps")

        # Convert pattern format
        pattern = {
            int(k): {int(nk): nv for nk, nv in v.items()} for k, v in request.input_pattern.items()
        }

        # Run simulation
        simulation_results = []
        current_state = pattern

        for step in range(request.simulation_steps):
            if hasattr(brain, "simulate_step"):
                step_result = brain.simulate_step(current_state)
                if request.return_intermediate_states:
                    simulation_results.append(
                        {"step": step, "state": step_result, "timestamp": time.time()}
                    )
                current_state = step_result
            else:
                # Fallback simulation
                break

        processing_time = (time.time() - start_time) * 1000

        return APIResponse(
            success=True,
            message="Brain simulation completed",
            data={
                "simulation_steps": request.simulation_steps,
                "final_state": current_state,
                "intermediate_states": (
                    simulation_results if request.return_intermediate_states else []
                ),
                "metrics": (
                    {
                        "total_time_ms": processing_time,
                        "steps_per_second": (
                            request.simulation_steps / (processing_time / 1000)
                            if processing_time > 0
                            else 0
                        ),
                    }
                    if request.track_metrics
                    else {}
                ),
            },
        )

    except Exception as e:
        logger.error(f"Error in brain simulation: {e}")
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")


@router.post("/batch")
async def batch_process(
    request: BatchProcessingRequest,
    background_tasks: BackgroundTasks,
    brain=Depends(get_brain_network),
):
    """
    Process multiple inputs in batch (async if requested)
    """
    try:
        logger.info(f"Batch processing {len(request.inputs)} inputs")

        if request.parallel_processing:
            # Process in parallel
            tasks = []
            for i, input_data in enumerate(request.inputs):
                task = _process_single_input(brain, input_data, i)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Process sequentially
            results = []
            for i, input_data in enumerate(request.inputs):
                result = await _process_single_input(brain, input_data, i)
                results.append(result)

        # Filter out exceptions
        successful_results = [r for r in results if not isinstance(r, Exception)]
        failed_results = [str(r) for r in results if isinstance(r, Exception)]

        return APIResponse(
            success=len(failed_results) == 0,
            message=f"Batch processing completed: {len(successful_results)} successful, {len(failed_results)} failed",
            data={
                "successful_count": len(successful_results),
                "failed_count": len(failed_results),
                "results": successful_results,
                "errors": failed_results,
                "processing_mode": request.processing_mode,
            },
        )

    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")


@router.get("/state")
async def get_brain_state(brain=Depends(get_advanced_brain)):
    """
    Get current brain state and activity
    """
    try:
        if hasattr(brain, "get_brain_state"):
            state = brain.get_brain_state()
        else:
            state = {"status": "active", "note": "State monitoring not available"}

        return APIResponse(success=True, message="Brain state retrieved", data=state)

    except Exception as e:
        logger.error(f"Error getting brain state: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get brain state: {str(e)}")


# Helper functions
async def _process_single_input(brain, input_data: Dict[str, Any], index: int):
    """Process a single input for batch processing"""
    try:
        if "pattern" in input_data:
            pattern = {
                int(k): {int(nk): nv for nk, nv in v.items()}
                for k, v in input_data["pattern"].items()
            }
            result = brain.process_pattern(pattern)
        elif "text" in input_data:
            if hasattr(brain, "text_to_pattern"):
                pattern = brain.text_to_pattern(input_data["text"])
                result = brain.process_pattern(pattern)
            else:
                result = {"error": "Text processing not available"}
        else:
            result = {"error": "Invalid input format"}

        return {"index": index, "result": result, "success": True}

    except Exception as e:
        return {"index": index, "error": str(e), "success": False}


def _generate_basic_pattern(text: str) -> Dict[str, Dict[str, bool]]:
    """Generate a basic neural pattern from text"""
    pattern = {}
    text_len = len(text)

    # Simple pattern generation based on text characteristics
    if text_len > 0:
        # Module 0: Length-based activation
        pattern["0"] = {str(i): True for i in range(min(text_len // 5, 10))}

        # Module 1: Character-based activation
        char_sum = sum(ord(c) for c in text[:10])
        pattern["1"] = {str(i): True for i in range(char_sum % 5, (char_sum % 5) + 3)}

    return pattern
