"""
FastAPI routes for automated training management
Provides REST API endpoints to control and monitor automated training
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, validator

# Add path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from training.automated_internet_trainer import (
        AutomatedInternetTrainer,
        TrainingConfig,
    )
    from training.training_monitor import TrainingMonitor
except ImportError:
    # Fallback for development
    print("⚠️  Training modules not available in API mode")
    AutomatedInternetTrainer = None
    TrainingMonitor = None

router = APIRouter()

# Global trainer instance
active_trainer = None  # Will be AutomatedInternetTrainer when available
training_task = None  # Will be asyncio.Task when available


class TrainingConfigRequest(BaseModel):
    """Request model for training configuration"""

    profile: str = "development"
    max_articles_per_session: Optional[int] = None
    collection_interval: Optional[int] = None
    min_article_quality_score: Optional[float] = None
    continuous: bool = False

    @validator("profile")
    def validate_profile(cls, v):
        valid_profiles = ["development", "production", "research"]
        if v not in valid_profiles:
            raise ValueError(f"Profile must be one of: {valid_profiles}")
        return v

    @validator("min_article_quality_score")
    def validate_quality_score(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError("Quality score must be between 0 and 1")
        return v


class TrainingResponse(BaseModel):
    """Response model for training operations"""

    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: str


@router.post("/start", response_model=TrainingResponse)
async def start_training(config: TrainingConfigRequest, background_tasks: BackgroundTasks):
    """Start automated training"""
    global active_trainer, training_task

    if AutomatedInternetTrainer is None:
        raise HTTPException(status_code=503, detail="Automated training modules not available")

    if active_trainer and training_task and not training_task.done():
        raise HTTPException(status_code=400, detail="Training is already running. Stop it first.")

    try:
        # Load configuration profile
        profile_config = _load_profile_config(config.profile)

        # Override with request parameters
        if config.max_articles_per_session is not None:
            profile_config.max_articles_per_session = config.max_articles_per_session
        if config.collection_interval is not None:
            profile_config.collection_interval = config.collection_interval
        if config.min_article_quality_score is not None:
            profile_config.min_article_quality_score = config.min_article_quality_score

        # Create trainer
        active_trainer = AutomatedInternetTrainer(profile_config)

        # Start training in background
        async def training_wrapper():
            try:
                await active_trainer.start_training(continuous=config.continuous)
            except Exception as e:
                print(f"Training error: {e}")

        training_task = asyncio.create_task(training_wrapper())

        return TrainingResponse(
            success=True,
            message=f"Automated training started with {config.profile} profile",
            data={
                "profile": config.profile,
                "continuous": config.continuous,
                "max_articles": profile_config.max_articles_per_session,
                "collection_interval": profile_config.collection_interval,
                "quality_threshold": profile_config.min_article_quality_score,
            },
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")


@router.post("/stop", response_model=TrainingResponse)
async def stop_training():
    """Stop automated training"""
    global active_trainer, training_task

    if not training_task or training_task.done():
        raise HTTPException(status_code=400, detail="No active training to stop")

    try:
        # Cancel the training task
        training_task.cancel()

        # Wait for graceful shutdown
        try:
            await asyncio.wait_for(training_task, timeout=10.0)
        except asyncio.TimeoutError:
            pass

        # Get final summary if trainer is available
        summary = {}
        if active_trainer:
            try:
                summary = active_trainer.get_knowledge_summary()
            except:
                pass

        return TrainingResponse(
            success=True,
            message="Automated training stopped",
            data={"final_summary": summary, "training_was_active": True},
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop training: {str(e)}")


@router.get("/status", response_model=TrainingResponse)
async def get_training_status():
    """Get current training status"""
    global active_trainer, training_task

    if TrainingMonitor is None:
        raise HTTPException(status_code=503, detail="Training monitor not available")

    try:
        monitor = TrainingMonitor()
        metrics_data = monitor.load_training_metrics()

        # Determine current status
        is_running = training_task and not training_task.done()

        status_data = {
            "is_running": is_running,
            "has_historical_data": len(metrics_data) > 0,
        }

        if metrics_data:
            latest = metrics_data[-1]
            latest_metrics = latest.get("training_metrics", {})

            status_data.update(
                {
                    "total_sessions": len(metrics_data),
                    "articles_collected": latest_metrics.get("articles_collected", 0),
                    "patterns_learned": latest_metrics.get("patterns_learned", 0),
                    "concepts_discovered": latest_metrics.get("concepts_discovered", 0),
                    "last_update": latest.get("timestamp", ""),
                    "recent_concepts": latest.get("learned_concepts", [])[-10:],
                }
            )

            # Add quality metrics if available
            if latest_metrics.get("quality_scores"):
                import numpy as np

                status_data["average_quality"] = float(np.mean(latest_metrics["quality_scores"]))

        # Add current trainer info if available
        if active_trainer:
            try:
                summary = active_trainer.get_knowledge_summary()
                status_data["current_session"] = summary
            except:
                pass

        return TrainingResponse(
            success=True,
            message="Training status retrieved",
            data=status_data,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.get("/report", response_model=TrainingResponse)
async def get_training_report():
    """Generate comprehensive training report"""
    if TrainingMonitor is None:
        raise HTTPException(status_code=503, detail="Training monitor not available")

    try:
        monitor = TrainingMonitor()
        report = monitor.generate_training_report()

        # Also get analysis data
        metrics_data = monitor.load_training_metrics()
        analysis = monitor.analyze_learning_progress(metrics_data)

        return TrainingResponse(
            success=True,
            message="Training report generated",
            data={
                "report_text": report,
                "analysis": analysis,
                "generated_at": datetime.now().isoformat(),
            },
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")


@router.get("/metrics", response_model=TrainingResponse)
async def get_training_metrics():
    """Get detailed training metrics"""
    if TrainingMonitor is None:
        raise HTTPException(status_code=503, detail="Training monitor not available")

    try:
        monitor = TrainingMonitor()
        metrics_data = monitor.load_training_metrics()

        if not metrics_data:
            return TrainingResponse(
                success=True,
                message="No training metrics available",
                data={"metrics": [], "analysis": {}},
                timestamp=datetime.now().isoformat(),
            )

        analysis = monitor.analyze_learning_progress(metrics_data)

        return TrainingResponse(
            success=True,
            message=f"Retrieved {len(metrics_data)} training sessions",
            data={
                "metrics": metrics_data[-10:],  # Last 10 sessions
                "analysis": analysis,
                "total_sessions": len(metrics_data),
            },
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.get("/profiles", response_model=TrainingResponse)
async def get_training_profiles():
    """Get available training profiles"""
    try:
        config_file = os.path.join(os.path.dirname(__file__), "../../training/training_config.json")

        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                config_data = json.load(f)
                profiles = config_data.get("training_profiles", {})
        else:
            # Default profiles
            profiles = {
                "development": {
                    "description": "Lightweight profile for testing",
                    "max_articles_per_session": 10,
                    "collection_interval": 300,
                    "min_article_quality_score": 0.4,
                },
                "production": {
                    "description": "Standard profile for regular training",
                    "max_articles_per_session": 50,
                    "collection_interval": 3600,
                    "min_article_quality_score": 0.6,
                },
                "research": {
                    "description": "Intensive profile for research and learning",
                    "max_articles_per_session": 100,
                    "collection_interval": 1800,
                    "min_article_quality_score": 0.7,
                },
            }

        return TrainingResponse(
            success=True,
            message=f"Retrieved {len(profiles)} training profiles",
            data={"profiles": profiles},
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get profiles: {str(e)}")


def _load_profile_config(profile_name: str) -> TrainingConfig:
    """Load configuration for a specific profile"""
    config_file = os.path.join(os.path.dirname(__file__), "../../training/training_config.json")

    try:
        with open(config_file, "r") as f:
            config_data = json.load(f)

        profile_config = config_data["training_profiles"].get(profile_name)
        if not profile_config:
            profile_config = config_data["training_profiles"]["development"]

        return TrainingConfig(
            max_articles_per_session=profile_config["max_articles_per_session"],
            collection_interval=profile_config["collection_interval"],
            min_article_quality_score=profile_config["min_article_quality_score"],
            learning_rate=profile_config["learning_rate"],
            save_interval=profile_config["save_interval"],
            max_concurrent_requests=profile_config["max_concurrent_requests"],
            request_delay=profile_config["request_delay"],
            sources=profile_config["sources"],
            content_filters=profile_config["content_filters"],
        )

    except Exception:
        # Fallback to default config
        return TrainingConfig()


# Health check for training system
@router.get("/health", response_model=TrainingResponse)
async def training_health_check():
    """Check if automated training system is healthy"""
    health_data = {
        "training_modules_available": AutomatedInternetTrainer is not None,
        "monitor_available": TrainingMonitor is not None,
        "output_directory_exists": os.path.exists("output"),
        "config_file_exists": os.path.exists("src/training/training_config.json"),
    }

    all_healthy = all(health_data.values())

    return TrainingResponse(
        success=all_healthy,
        message="Training system health check",
        data=health_data,
        timestamp=datetime.now().isoformat(),
    )
