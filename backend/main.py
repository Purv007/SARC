"""
SARC Backend — FastAPI Application Entry Point

Semantically-Aware Regional Compression (SARC) API server.
Serves the SARC compression pipeline and React frontend static files.
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Ensure backend directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import UPLOAD_DIR, OUTPUT_DIR
from database.database import init_db
from routers.compress import router as compress_router
from routers.health import health_router, history_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — initialize DB and preload models."""
    # Startup
    print("=" * 60)
    print("  SARC — Semantically-Aware Regional Compression")
    print("  Starting up...")
    print("=" * 60)

    # Initialize database
    await init_db()
    print("[OK] SQLite database initialized")

    # Ensure directories exist
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[OK] Upload and output directories ready")

    print("=" * 60)
    print("  Server ready! Open http://localhost:8000")
    print("=" * 60)

    yield

    # Shutdown
    print("SARC server shutting down...")


# Create FastAPI app
app = FastAPI(
    title="SARC — Semantically-Aware Regional Compression",
    description="AI-powered context-aware multimedia compression using dynamic importance scoring",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving processed images
app.mount("/static/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
app.mount("/static/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# Register routers
app.include_router(compress_router)
app.include_router(health_router)
app.include_router(history_router)


@app.get("/")
async def root():
    """Root endpoint — API info."""
    return {
        "name": "SARC — Semantically-Aware Regional Compression",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
