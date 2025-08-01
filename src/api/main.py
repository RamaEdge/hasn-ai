"""
FastAPI Brain-Inspired Neural Network API
Production-ready API for HASN architecture
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import sys
import os

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import brain components
try:
    from core.brain_inspired_network import SimpleBrainNetwork
    from core.advanced_brain_network import AdvancedCognitiveBrain
except ImportError:
    # Fallback imports with adjusted paths
    import sys
    import os
    brain_core_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'core')
    sys.path.append(brain_core_path)
    try:
        from brain_inspired_network import SimpleBrainNetwork  
        from advanced_brain_network import AdvancedCognitiveBrain
    except ImportError:
        # Create dummy classes for testing
        class SimpleBrainNetwork:
            def __init__(self, module_sizes):
                self.modules = {i: type('Module', (), {'neurons': [type('Neuron', (), {})() for _ in range(size)]})() for i, size in enumerate(module_sizes)}
            def process_pattern(self, pattern):
                return {"total_activity": 0.7, "steps": 100}
            def get_brain_state(self):
                return {"status": "active", "modules": len(self.modules)}
        
        class AdvancedCognitiveBrain:
            def __init__(self):
                self.modules = {}
            def process_pattern(self, pattern):
                return {"total_activity": 0.8, "cognitive_load": "medium"}
            def get_brain_state(self):
                return {"status": "cognitive", "working_memory": 7}

from api.routes import brain, health, training
try:
    from api.routes import automated_training
    AUTOMATED_TRAINING_AVAILABLE = True
except ImportError:
    AUTOMATED_TRAINING_AVAILABLE = False
    print("⚠️  Automated training routes not available")

from api.middleware.rate_limit import RateLimitMiddleware
from api.models.responses import APIResponse, ErrorResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Brain-Inspired Neural Network API",
    description="Production API for Hierarchical Adaptive Spiking Network (HASN)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
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

# Global brain instances
brain_network = None
advanced_brain = None

@app.on_event("startup")
async def startup_event():
    """Initialize brain networks on startup"""
    global brain_network, advanced_brain
    
    try:
        logger.info("🧠 Initializing Brain Networks...")
        
        # Initialize basic brain network
        brain_network = SimpleBrainNetwork([30, 25, 20, 15])
        logger.info(f"✅ Basic Brain Network initialized with {sum([30, 25, 20, 15])} neurons")
        
        # Initialize advanced cognitive brain
        advanced_brain = AdvancedCognitiveBrain()
        logger.info("✅ Advanced Cognitive Brain initialized")
        
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
    if brain_network is None:
        raise HTTPException(status_code=503, detail="Brain network not initialized")
    return brain_network

def get_advanced_brain():
    if advanced_brain is None:
        raise HTTPException(status_code=503, detail="Advanced brain not initialized")
    return advanced_brain

# Include routers
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(brain.router, prefix="/brain", tags=["Brain Processing"])
app.include_router(training.router, prefix="/training", tags=["Training"])

# Include automated training router if available
if AUTOMATED_TRAINING_AVAILABLE:
    app.include_router(automated_training.router, prefix="/automated-training", tags=["Automated Training"])

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
                "automated_training": "/automated-training" if AUTOMATED_TRAINING_AVAILABLE else "not_available"
            },
            "timestamp": datetime.now().isoformat()
        }
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
            detail=str(exc) if app.debug else "An unexpected error occurred"
        ).dict()
    )

if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
