"""
FastAPI Brain-Inspired Neural Network API
Production-ready API for HASN architecture
"""

import logging
import os
import sys
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""Production FastAPI that wires both Simple and Cognitive networks via DI."""

# Import brain components
try:
    from core.cognitive_brain_network import CognitiveBrainNetwork, CognitiveConfig
    from core.simplified_brain_network import SimpleBrainNetwork
except ImportError:
    # Fallback imports with adjusted paths
    brain_core_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "core")
    sys.path.append(brain_core_path)
    try:
        from cognitive_brain_network import CognitiveBrainNetwork, CognitiveConfig
        from simplified_brain_network import SimpleBrainNetwork
    except ImportError:
        raise

from api.routes import brain, health, training

try:
    from api.routes import automated_training

    AUTOMATED_TRAINING_AVAILABLE = True
except ImportError:
    AUTOMATED_TRAINING_AVAILABLE = False
    print("⚠️  Automated training routes not available")

from api.adapters.brain_adapters import CognitiveBrainAdapter, SimpleBrainAdapter
from api.middleware.rate_limit import RateLimitMiddleware
from pydantic import BaseModel
from typing import Dict, Any

# Simple response models
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

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production: ["https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Rate limiting middleware
app.add_middleware(RateLimitMiddleware, calls=100, period=60)  # 100 calls per minute

# Adapters moved to api.adapters.brain_adapters for clarity


# Global brain instances
basic_brain: SimpleBrainAdapter | None = None
advanced_brain: CognitiveBrainAdapter | None = None


@app.on_event("startup")
async def startup_event():
    """Initialize brain networks on startup"""
    global basic_brain, advanced_brain

    try:
        logger.info("🧠 Initializing Brain Networks...")

        # Initialize basic (simplified) brain network
        simple = SimpleBrainNetwork(num_neurons=100, connectivity_prob=0.05)
        basic_brain = SimpleBrainAdapter(simple)
        logger.info("✅ SimpleBrainNetwork initialized (100 neurons)")

        # Initialize advanced cognitive brain network
        cognitive_cfg = CognitiveConfig(max_episodic_memories=200)
        cognitive = CognitiveBrainNetwork(
            num_neurons=150, connectivity_prob=0.05, config=cognitive_cfg
        )
        advanced_brain = CognitiveBrainAdapter(cognitive)
        logger.info("✅ CognitiveBrainNetwork initialized (150 neurons)")

        logger.info("🚀 Brain API startup complete!")

    except Exception as e:
        logger.error(f"❌ Failed to initialize brain networks: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down Brain API...")
    # Add cleanup logic here if needed


# Dependency to get brain instances
def get_brain_network():
    if basic_brain is None:
        raise HTTPException(status_code=503, detail="Brain network not initialized")
    return basic_brain


def get_advanced_brain():
    if advanced_brain is None:
        raise HTTPException(status_code=503, detail="Advanced brain not initialized")
    return advanced_brain


# Include routers
app.include_router(health.router, prefix="/health", tags=["Health"])

# Wire dependencies for route modules via FastAPI overrides
app.dependency_overrides[brain.get_brain_network] = get_brain_network
app.dependency_overrides[brain.get_advanced_brain] = get_advanced_brain
app.dependency_overrides[training.get_brain_network] = get_brain_network
app.dependency_overrides[training.get_advanced_brain] = get_advanced_brain

app.include_router(brain.router, prefix="/brain", tags=["Brain Processing"])
app.include_router(training.router, prefix="/training", tags=["Training"])

# Include automated training router if available
if AUTOMATED_TRAINING_AVAILABLE:
    app.include_router(
        automated_training.router,
        prefix="/automated-training",
        tags=["Automated Training"],
    )


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
                    "/automated-training" if AUTOMATED_TRAINING_AVAILABLE else "not_available"
                ),
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
        ).dict(),
    )


if __name__ == "__main__":
    # Development server
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
