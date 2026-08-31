"""
Compression Router — Main API endpoint for SARC processing.

POST /api/compress — Upload an image and get SARC + uniform compression results.
"""

import json
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, UploadFile, Form, HTTPException

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import UPLOAD_DIR, OUTPUT_DIR, SARCConfig
from sarc.engine import SARCEngine
from database.database import (
    create_experiment, save_detections, save_compression_result, save_metrics
)

router = APIRouter(prefix="/api", tags=["compression"])

# Shared engine instance (models loaded once)
_engine: Optional[SARCEngine] = None


def get_engine() -> SARCEngine:
    """Get or create the shared SARC engine instance."""
    global _engine
    if _engine is None:
        _engine = SARCEngine()
    return _engine


@router.post("/compress")
async def compress_image(
    file: UploadFile = File(...),
    w_obj: Optional[float] = Form(None),
    w_text: Optional[float] = Form(None),
    w_edge: Optional[float] = Form(None),
    w_spatial: Optional[float] = Form(None),
    w_context: Optional[float] = Form(None),
    context_factor: Optional[float] = Form(None),
    q_min: Optional[int] = Form(None),
    q_max: Optional[int] = Form(None),
    uniform_quality: Optional[int] = Form(None),
    boundary_sigma: Optional[float] = Form(None),
):
    """
    Upload an image and process it through the SARC pipeline.
    
    Returns comprehensive results including:
    - Object and text detections
    - Importance and quality maps
    - SARC and uniform compressed images
    - All evaluation metrics (PSNR, SSIM, LPIPS, etc.)
    - Comparison results
    """
    # Validate file type
    is_video = False
    if file.content_type:
        if file.content_type.startswith("video/"):
            is_video = True
        elif not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"File must be an image or video. Got: {file.content_type}"
            )

    # Generate unique ID for this experiment
    experiment_id = str(uuid.uuid4())[:8]

    # Save uploaded file
    upload_path = UPLOAD_DIR / f"{experiment_id}_{file.filename}"
    try:
        with open(upload_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    # Build config overrides
    config_override = {}
    if w_obj is not None: config_override["w_obj"] = w_obj
    if w_text is not None: config_override["w_text"] = w_text
    if w_edge is not None: config_override["w_edge"] = w_edge
    if w_spatial is not None: config_override["w_spatial"] = w_spatial
    if w_context is not None: config_override["w_context"] = w_context
    if context_factor is not None: config_override["context_factor"] = context_factor
    if q_min is not None: config_override["q_min"] = q_min
    if q_max is not None: config_override["q_max"] = q_max
    if uniform_quality is not None: config_override["uniform_quality"] = uniform_quality
    if boundary_sigma is not None: config_override["boundary_sigma"] = boundary_sigma

    # Run SARC pipeline
    try:
        engine = get_engine()
        if is_video:
            result = engine.process_video(
                video_path=str(upload_path),
                output_prefix=experiment_id,
                config_override=config_override if config_override else None
            )
        else:
            result = engine.process(
                image_path=str(upload_path),
                output_prefix=experiment_id,
                config_override=config_override if config_override else None
            )
    except Exception as e:
        # Clean up on failure
        upload_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

    # Persist to SQLite
    try:
        db_experiment_id = await create_experiment(
            filename=file.filename,
            original_path=str(upload_path),
            width=result["image_info"]["width"],
            height=result["image_info"]["height"],
            original_size_bytes=result["image_info"]["original_size_bytes"],
            config_json=json.dumps(result.get("config", {}))
        )

        # Save detections
        db_detections = []
        for obj in result["detections"]["objects"]:
            db_detections.append({
                "detector_type": "yolo",
                "class_name": obj["class_name"],
                "confidence": obj["confidence"],
                "bbox_x1": obj["bbox"][0],
                "bbox_y1": obj["bbox"][1],
                "bbox_x2": obj["bbox"][2],
                "bbox_y2": obj["bbox"][3],
                "importance_score": obj["importance"],
            })
        for text_r in result["detections"]["text_regions"]:
            db_detections.append({
                "detector_type": "ocr",
                "class_name": text_r["text"][:100],
                "confidence": text_r["confidence"],
                "bbox_x1": text_r["bbox"][0],
                "bbox_y1": text_r["bbox"][1],
                "bbox_x2": text_r["bbox"][2],
                "bbox_y2": text_r["bbox"][3],
                "importance_score": 1.0,
            })
        if db_detections:
            await save_detections(db_experiment_id, db_detections)

        # Save compression results
        for method in ["sarc", "uniform"]:
            path_key = f"{method}_compressed"
            await save_compression_result(
                experiment_id=db_experiment_id,
                method=method,
                output_path=result["output_paths"].get(path_key, ""),
                compressed_size_bytes=result["metrics"][method]["file_size_bytes"],
                heatmap_path=result["output_paths"].get("importance_heatmap"),
                overlay_path=result["output_paths"].get("detection_overlay"),
            )

        # Save metrics
        for method in ["sarc", "uniform"]:
            m = result["metrics"][method]
            await save_metrics(
                experiment_id=db_experiment_id,
                method=method,
                psnr=m.get("psnr"),
                ssim=m.get("ssim"),
                lpips_val=m.get("lpips"),
                file_size_bytes=m.get("file_size_bytes"),
                compression_ratio=m.get("compression_ratio"),
                bandwidth_savings_percent=m.get("bandwidth_savings_percent"),
                processing_time_seconds=m.get("processing_time_seconds"),
            )

        result["db_experiment_id"] = db_experiment_id

    except Exception as e:
        # DB errors are non-fatal — still return results
        print(f"Warning: Database save failed: {e}")

    # Convert output paths to URL-friendly paths
    for key, path in result["output_paths"].items():
        # Make paths relative to output dir for serving
        rel_path = Path(path).relative_to(OUTPUT_DIR.parent) if Path(path).is_absolute() else path
        result["output_paths"][key] = f"/static/{str(rel_path).replace(os.sep, '/')}"

    return result
