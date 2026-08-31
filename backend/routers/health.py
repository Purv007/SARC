"""
Health and History Routers — System status and experiment history.
"""

from pathlib import Path
from fastapi import APIRouter, HTTPException

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DIR
from database.database import get_all_experiments, get_experiment, delete_experiment

health_router = APIRouter(tags=["health"])
history_router = APIRouter(tags=["history"])


def _abs_to_url(abs_path: str) -> str:
    """Convert an absolute file path to a URL-friendly /static/outputs/... path."""
    if not abs_path:
        return ""
    try:
        p = Path(abs_path)
        rel = p.relative_to(OUTPUT_DIR)
        return f"/static/outputs/{str(rel).replace(os.sep, '/')}"
    except (ValueError, TypeError):
        return abs_path


def _convert_experiment_paths(experiment: dict) -> dict:
    """Walk an experiment dict and convert all stored file paths to URLs."""
    # Convert compression_results paths
    for cr in experiment.get("compression_results", []):
        for key in ("output_path", "heatmap_path", "overlay_path"):
            if cr.get(key):
                cr[key] = _abs_to_url(cr[key])
    return experiment


@health_router.get("/api/health")
async def health_check():
    """Check system health and model availability."""
    models = {}

    try:
        from ultralytics import YOLO
        models["yolo"] = True
    except ImportError:
        models["yolo"] = False

    try:
        import easyocr
        models["easyocr"] = True
    except ImportError:
        models["easyocr"] = False

    try:
        import lpips
        models["lpips"] = True
    except ImportError:
        models["lpips"] = False

    try:
        import torch
        models["torch"] = True
        models["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            models["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        models["torch"] = False
        models["cuda_available"] = False

    return {
        "status": "healthy",
        "version": "1.0.0",
        "algorithm": "SARC (Semantically-Aware Regional Compression)",
        "models_loaded": models,
    }


@history_router.get("/api/experiments")
async def list_experiments(limit: int = 50, offset: int = 0):
    """List all past experiments with summary data."""
    experiments = await get_all_experiments(limit=limit, offset=offset)
    return {"experiments": experiments, "count": len(experiments)}


@history_router.get("/api/experiments/{experiment_id}")
async def get_experiment_detail(experiment_id: int):
    """Get detailed results for a specific experiment."""
    experiment = await get_experiment(experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    # Convert absolute paths to URL paths so frontend can load them
    experiment = _convert_experiment_paths(experiment)
    return experiment


@history_router.delete("/api/experiments/{experiment_id}")
async def remove_experiment(experiment_id: int):
    """Delete an experiment and its data."""
    await delete_experiment(experiment_id)
    return {"status": "deleted", "experiment_id": experiment_id}

