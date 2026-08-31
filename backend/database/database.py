"""
SQLite database layer for SARC experiment persistence.

Stores all experiment metadata, detection results, metrics, and compression outputs.
Uses aiosqlite for async compatibility with FastAPI.
"""

import aiosqlite
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

from config import DB_PATH


# SQL schema
_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    original_path TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    original_size_bytes INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    config_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    detector_type TEXT NOT NULL,  -- 'yolo' or 'ocr'
    class_name TEXT,
    confidence REAL,
    bbox_x1 REAL NOT NULL,
    bbox_y1 REAL NOT NULL,
    bbox_x2 REAL NOT NULL,
    bbox_y2 REAL NOT NULL,
    importance_score REAL NOT NULL,
    extra_json TEXT DEFAULT '{}',
    FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS compression_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    method TEXT NOT NULL,  -- 'uniform' or 'sarc'
    output_path TEXT NOT NULL,
    compressed_size_bytes INTEGER NOT NULL,
    quality_config TEXT DEFAULT '{}',
    heatmap_path TEXT,
    overlay_path TEXT,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    method TEXT NOT NULL,  -- 'uniform' or 'sarc'
    psnr REAL,
    ssim REAL,
    lpips REAL,
    file_size_bytes INTEGER,
    compression_ratio REAL,
    bandwidth_savings_percent REAL,
    processing_time_seconds REAL,
    extra_json TEXT DEFAULT '{}',
    FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
);
"""


async def init_db():
    """Initialize the SQLite database and create tables if they don't exist."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.executescript(_SCHEMA)
        await db.commit()


async def create_experiment(
    filename: str,
    original_path: str,
    width: int,
    height: int,
    original_size_bytes: int,
    config_json: str = "{}"
) -> int:
    """Create a new experiment record and return its ID."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        cursor = await db.execute(
            """INSERT INTO experiments 
               (filename, original_path, width, height, original_size_bytes, config_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (filename, original_path, width, height, original_size_bytes,
             config_json, datetime.now(timezone.utc).isoformat())
        )
        await db.commit()
        return cursor.lastrowid


async def save_detections(experiment_id: int, detections: List[Dict[str, Any]]):
    """Save detection results (YOLO or OCR) for an experiment."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        for det in detections:
            await db.execute(
                """INSERT INTO detections 
                   (experiment_id, detector_type, class_name, confidence,
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2, importance_score, extra_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (experiment_id, det["detector_type"], det.get("class_name"),
                 det.get("confidence"), det["bbox_x1"], det["bbox_y1"],
                 det["bbox_x2"], det["bbox_y2"], det["importance_score"],
                 json.dumps(det.get("extra", {})))
            )
        await db.commit()


async def save_compression_result(
    experiment_id: int,
    method: str,
    output_path: str,
    compressed_size_bytes: int,
    quality_config: str = "{}",
    heatmap_path: Optional[str] = None,
    overlay_path: Optional[str] = None
) -> int:
    """Save a compression result for an experiment."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        cursor = await db.execute(
            """INSERT INTO compression_results 
               (experiment_id, method, output_path, compressed_size_bytes,
                quality_config, heatmap_path, overlay_path)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (experiment_id, method, output_path, compressed_size_bytes,
             quality_config, heatmap_path, overlay_path)
        )
        await db.commit()
        return cursor.lastrowid


async def save_metrics(
    experiment_id: int,
    method: str,
    psnr: Optional[float] = None,
    ssim: Optional[float] = None,
    lpips_val: Optional[float] = None,
    file_size_bytes: Optional[int] = None,
    compression_ratio: Optional[float] = None,
    bandwidth_savings_percent: Optional[float] = None,
    processing_time_seconds: Optional[float] = None,
    extra: Optional[Dict] = None
) -> int:
    """Save evaluation metrics for an experiment."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        cursor = await db.execute(
            """INSERT INTO metrics 
               (experiment_id, method, psnr, ssim, lpips, file_size_bytes,
                compression_ratio, bandwidth_savings_percent, processing_time_seconds, extra_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (experiment_id, method, psnr, ssim, lpips_val, file_size_bytes,
             compression_ratio, bandwidth_savings_percent, processing_time_seconds,
             json.dumps(extra or {}))
        )
        await db.commit()
        return cursor.lastrowid


async def get_experiment(experiment_id: int) -> Optional[Dict]:
    """Retrieve a single experiment with all related data."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row

        # Get experiment
        cursor = await db.execute(
            "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return None
        experiment = dict(row)

        # Get detections
        cursor = await db.execute(
            "SELECT * FROM detections WHERE experiment_id = ?", (experiment_id,)
        )
        experiment["detections"] = [dict(r) for r in await cursor.fetchall()]

        # Get compression results
        cursor = await db.execute(
            "SELECT * FROM compression_results WHERE experiment_id = ?", (experiment_id,)
        )
        experiment["compression_results"] = [dict(r) for r in await cursor.fetchall()]

        # Get metrics
        cursor = await db.execute(
            "SELECT * FROM metrics WHERE experiment_id = ?", (experiment_id,)
        )
        experiment["metrics"] = [dict(r) for r in await cursor.fetchall()]

        return experiment


async def get_all_experiments(limit: int = 50, offset: int = 0) -> List[Dict]:
    """Retrieve all experiments with summary data (no full detections)."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT e.*, 
                      (SELECT COUNT(*) FROM detections WHERE experiment_id = e.id) as detection_count
               FROM experiments e
               ORDER BY e.created_at DESC
               LIMIT ? OFFSET ?""",
            (limit, offset)
        )
        rows = await cursor.fetchall()
        experiments = []
        for row in rows:
            exp = dict(row)
            # Get metrics summary
            mcursor = await db.execute(
                "SELECT * FROM metrics WHERE experiment_id = ?", (exp["id"],)
            )
            exp["metrics"] = [dict(r) for r in await mcursor.fetchall()]
            experiments.append(exp)
        return experiments


async def delete_experiment(experiment_id: int):
    """Delete an experiment and all related data."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))
        await db.commit()
