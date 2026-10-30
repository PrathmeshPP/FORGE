"""
F.O.R.G.E. Website Backend — Configuration Module
Reads configuration from environment variables with safe loopback defaults.
"""

import os
from pathlib import Path
from typing import List

# Base directory for the website workspace
BASE_DIR = Path(__file__).resolve().parent.parent

# Server Binding — Strictly loopback for private Tailscale Serve
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))

# Dedicated storage directory outside Git repository and outside synced cloud folders
default_data_dir = r"C:\forge_data" if os.name == "nt" else str(BASE_DIR.parent / "forge_data")
FORGE_DATA_DIR = Path(os.environ.get("FORGE_DATA_DIR", default_data_dir)).resolve()

# Jobs directory under data root
JOBS_DIR = FORGE_DATA_DIR / "jobs"
DB_PATH = FORGE_DATA_DIR / "forge_jobs.db"

# Upload and Preflight Limits
FORGE_MAX_UPLOAD_BYTES = int(os.environ.get("FORGE_MAX_UPLOAD_BYTES", 2 * 1024 * 1024 * 1024)) # 2 GiB
FORGE_MAX_VIDEO_SECONDS = float(os.environ.get("FORGE_MAX_VIDEO_SECONDS", 900.0)) # 15 minutes
FORGE_ACCEPTED_EXTENSIONS: List[str] = [".mp4", ".mov"]
FORGE_MIN_FREE_DISK_PERCENT = float(os.environ.get("FORGE_MIN_FREE_DISK_PERCENT", 10.0))
FORGE_RAW_RETENTION_HOURS = int(os.environ.get("FORGE_RAW_RETENTION_HOURS", 24))

# Pipeline Integration Root (read-only original FORGE checkout)
FORGE_PIPELINE_ROOT = Path(
    os.environ.get("FORGE_PIPELINE_ROOT", r"C:\Users\25beevdt025\Desktop\FORGE")
).resolve()

# Worker Mode ('mock' for Phase 2/3, 'real' only after handoff approval)
FORGE_WORKER_MODE = os.environ.get("FORGE_WORKER_MODE", "mock").lower()

# Frontend distribution directory for unified single-origin serving
FRONTEND_DIST_DIR = (BASE_DIR / "viewer" / "dist").resolve()

