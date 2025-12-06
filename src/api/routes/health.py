"""
Health check routes for API monitoring
"""

import time
from datetime import datetime, timedelta
from typing import Any, Dict

import psutil
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


# Simple response model
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    uptime_seconds: float
    brain_networks: Dict[str, bool]
    system_info: Dict[str, Any]


# Track startup time for uptime calculation
startup_time = time.time()


@router.get("/", response_model=HealthResponse)
async def health_check():
    """
    Basic health check endpoint
    Returns system status and brain network health
    """
    current_time = time.time()
    uptime_seconds = current_time - startup_time
    uptime_str = str(timedelta(seconds=int(uptime_seconds)))

    # Get system information
    system_info = {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage("/").percent,
        "python_version": f"{psutil.PYTHON}{psutil.version_info}",
        "platform": psutil.uname().system,
    }

    return HealthResponse(
        status="healthy",
        brain_networks={"basic_brain": "active", "advanced_brain": "active"},
        system_info=system_info,
        uptime=uptime_str,
    )


@router.get("/detailed")
async def detailed_health():
    """
    Detailed health check with performance metrics
    """
    # System metrics
    cpu_times = psutil.cpu_times()
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    detailed_info = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": time.time() - startup_time,
        "system": {
            "cpu": {
                "percent": psutil.cpu_percent(interval=1),
                "count": psutil.cpu_count(),
                "user_time": cpu_times.user,
                "system_time": cpu_times.system,
                "idle_time": cpu_times.idle,
            },
            "memory": {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "percent_used": memory.percent,
                "free_gb": round(memory.free / (1024**3), 2),
            },
            "disk": {
                "total_gb": round(disk.total / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "percent_used": round((disk.used / disk.total) * 100, 2),
            },
        },
        "brain_networks": {
            "basic_brain": {
                "status": "active",
                "total_neurons": 90,  # 30+25+20+15
                "modules": 4,
            },
            "advanced_brain": {
                "status": "active",
                "cognitive_modules": ["sensory", "memory", "executive", "motor"],
                "working_memory_capacity": 7,
            },
        },
    }

    return detailed_info


@router.get("/readiness")
async def readiness_check():
    """
    Kubernetes readiness probe endpoint
    """
    try:
        # Quick brain network availability check
        # This would be replaced with actual brain network ping
        brain_ready = True

        if brain_ready:
            return {"status": "ready", "timestamp": datetime.now().isoformat()}
        else:
            return {"status": "not_ready", "reason": "brain_networks_initializing"}

    except Exception as e:
        return {"status": "not_ready", "reason": str(e)}


@router.get("/liveness")
async def liveness_check():
    """
    Kubernetes liveness probe endpoint
    """
    return {
        "status": "alive",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": time.time() - startup_time,
    }
