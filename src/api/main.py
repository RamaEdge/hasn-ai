"""
FastAPI Brain-Inspired Neural Network API
Production-ready API for HASN architecture
"""

import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Core imports
from api.adapters.brain_adapters import CognitiveBrainAdapter, SimpleBrainAdapter
from api.middleware.rate_limit import RateLimitMiddleware
from api.routes import brain, health, training
from core.cognitive_brain_network import CognitiveBrainNetwork, CognitiveConfig
from core.simplified_brain_network import SimpleBrainNetwork


# Optional route imports
def _import_optional_route(module_name: str):
    """Helper to import optional route modules"""
    try:
        return __import__(f"api.routes.{module_name}", fromlist=[module_name])
    except ImportError:
        return None


automated_training = _import_optional_route("automated_training")
state = _import_optional_route("state")
knowledge = _import_optional_route("knowledge")
ingest = _import_optional_route("ingest")
train_ingest = _import_optional_route("train_ingest")
chat = _import_optional_route("chat")

# Optional ingestion imports
try:
    from ingestion.replay_trainer import ReplayTrainer
    from ingestion.service import IngestionService, QuarantineBuffer

    INGESTION_AVAILABLE = True
except ImportError:
    INGESTION_AVAILABLE = False
    IngestionService = None
    QuarantineBuffer = None
    ReplayTrainer = None


# Response models
class APIResponse(BaseModel):
    success: bool
    message: str = ""
    data: Dict[str, Any] = {}


class ErrorResponse(BaseModel):
    success: bool = False
    error: str = ""
    detail: str = ""


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Brain-Inspired Neural Network API",
    description="Production API for Hierarchical Adaptive Spiking Network (HASN)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production: ["https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware, calls=100, period=60)

# Global instances
basic_brain: Optional[SimpleBrainAdapter] = None
advanced_brain: Optional[CognitiveBrainAdapter] = None
cognitive_architecture = None


@app.on_event("startup")
async def startup_event():
    """Initialize brain networks and services on startup"""
    global basic_brain, advanced_brain, cognitive_architecture

    try:
        logger.info("Initializing Brain Networks...")

        # Initialize basic brain network
        simple = SimpleBrainNetwork(num_neurons=100, connectivity_prob=0.05)
        basic_brain = SimpleBrainAdapter(simple)
        logger.info("SimpleBrainNetwork initialized (100 neurons)")

        # Initialize advanced cognitive brain network
        cognitive_cfg = CognitiveConfig(max_episodic_memories=200)
        cognitive = CognitiveBrainNetwork(
            num_neurons=150, connectivity_prob=0.05, config=cognitive_cfg
        )
        advanced_brain = CognitiveBrainAdapter(cognitive)
        logger.info("CognitiveBrainNetwork initialized (150 neurons)")

        # Initialize CognitiveArchitecture for state/knowledge/chat routes
        if state is not None or chat is not None:
            try:
                from core.cognitive_architecture import CognitiveArchitecture
                from core.cognitive_models import CognitiveConfig as CAConfig

                ca_config = CAConfig(max_episodic_memories=200)
                cognitive_architecture = CognitiveArchitecture(
                    config=ca_config, backend_name="numpy"
                )
                logger.info("CognitiveArchitecture initialized for state/knowledge/chat routes")

                # Set global instance for chat routes
                if chat is not None:
                    chat._cognitive_architecture = cognitive_architecture
            except Exception as e:
                logger.warning(f"Failed to initialize CognitiveArchitecture: {e}")
                cognitive_architecture = None

        # Initialize ingestion service and replay trainer
        if INGESTION_AVAILABLE and ingest is not None and train_ingest is not None:
            try:
                storage_type = os.getenv("QUARANTINE_STORAGE", "local")
                storage_path = os.getenv("QUARANTINE_PATH", "./quarantine")
                redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

                quarantine_buffer = QuarantineBuffer(
                    storage_type=storage_type, storage_path=storage_path, redis_url=redis_url
                )

                ingestion_service = IngestionService(quarantine_buffer)
                replay_trainer = ReplayTrainer(quarantine_buffer, brain_network=simple)

                # Set global instances for dependency injection
                ingest._ingestion_service = ingestion_service
                ingest._quarantine_buffer = quarantine_buffer
                train_ingest._replay_trainer = replay_trainer

                logger.info("Ingestion service and replay trainer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize ingestion service: {e}")

        logger.info("Brain API startup complete!")

    except Exception as e:
        logger.error(f"Failed to initialize brain networks: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Brain API...")


# Dependency injection functions
def get_brain_network():
    """Get basic brain network instance"""
    if basic_brain is None:
        raise HTTPException(status_code=503, detail="Brain network not initialized")
    return basic_brain


def get_advanced_brain():
    """Get advanced brain network instance"""
    if advanced_brain is None:
        raise HTTPException(status_code=503, detail="Advanced brain not initialized")
    return advanced_brain


def get_cognitive_architecture():
    """Get cognitive architecture instance"""
    if cognitive_architecture is None:
        raise HTTPException(status_code=503, detail="Cognitive architecture not initialized")
    return cognitive_architecture


# Register routes
app.include_router(health.router, prefix="/health", tags=["Health"])

# Wire dependencies for brain routes
app.dependency_overrides[brain.get_brain_network] = get_brain_network
app.dependency_overrides[brain.get_advanced_brain] = get_advanced_brain
app.dependency_overrides[training.get_brain_network] = get_brain_network
app.dependency_overrides[training.get_advanced_brain] = get_advanced_brain

app.include_router(brain.router, prefix="/brain", tags=["Brain Processing"])
app.include_router(training.router, prefix="/training", tags=["Training"])

# Optional routes
if automated_training is not None:
    app.include_router(
        automated_training.router, prefix="/automated-training", tags=["Automated Training"]
    )

if state is not None and knowledge is not None:
    app.dependency_overrides[state.get_cognitive_architecture] = get_cognitive_architecture
    app.dependency_overrides[knowledge.get_cognitive_architecture] = get_cognitive_architecture
    app.include_router(state.router, prefix="/state", tags=["State Management"])
    app.include_router(knowledge.router, prefix="/knowledge", tags=["Knowledge Search"])

if ingest is not None and train_ingest is not None:
    app.include_router(ingest.router, prefix="/ingest", tags=["Ingestion"])
    app.include_router(train_ingest.router, prefix="/train", tags=["Training"])

if chat is not None:
    app.dependency_overrides[chat.get_cognitive_architecture] = lambda: chat._cognitive_architecture
    app.include_router(chat.router, prefix="/chat", tags=["Chat"])


# Root endpoint
@app.get("/", response_model=APIResponse)
async def root():
    """Root endpoint with API information"""
    return APIResponse(
        success=True,
        message="Brain-Inspired Neural Network API",
        data={
            "version": "1.0.0",
            "description": "Production API for HASN architecture",
            "endpoints": {
                "health": "/health",
                "docs": "/docs",
                "brain_processing": "/brain",
                "training": "/training",
                "automated_training": (
                    "/automated-training" if automated_training else "not_available"
                ),
                "state_management": "/state" if state else "not_available",
                "knowledge_search": "/knowledge" if knowledge else "not_available",
                "ingestion": "/ingest" if ingest else "not_available",
                "training_pipeline": "/train" if train_ingest else "not_available",
            },
            "timestamp": datetime.now().isoformat(),
        },
    )


# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            success=False,
            error="Internal server error",
            detail=str(exc) if app.debug else "An unexpected error occurred",
        ).model_dump(),
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
