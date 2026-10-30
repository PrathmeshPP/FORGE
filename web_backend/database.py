"""
F.O.R.G.E. Website Backend — SQLite Database Module
Manages job persistence outside the Git repository.
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from web_backend.config import DB_PATH, FORGE_DATA_DIR


def get_db_connection() -> sqlite3.Connection:
    """Creates a thread-safe connection to the SQLite database."""
    FORGE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Initializes the database schema if not present."""
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                status TEXT NOT NULL,
                progress_stage TEXT NOT NULL DEFAULT 'idle',
                progress_stage_label TEXT NOT NULL DEFAULT '',
                worker_mode TEXT NOT NULL DEFAULT 'mock',
                is_sample_job INTEGER NOT NULL DEFAULT 0,
                keep_raw_input INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                duration_seconds INTEGER,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                last_error_code TEXT,
                last_error_message TEXT,
                input_path TEXT,
                input_bytes INTEGER DEFAULT 0,
                preflight_json TEXT,
                scene_json TEXT,
                quality_summary_json TEXT,
                logs_json TEXT
            )
        """)
        conn.commit()


def row_to_job_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Converts a database row into a structured job dictionary."""
    d = dict(row)
    # Parse JSON fields safely
    for json_field in ["preflight_json", "scene_json", "quality_summary_json", "logs_json"]:
        raw_val = d.pop(json_field, None)
        target_key = json_field.replace("_json", "")
        if raw_val:
            try:
                d[target_key] = json.loads(raw_val)
            except json.JSONDecodeError:
                d[target_key] = None
        else:
            d[target_key] = [] if target_key == "logs" else None

    # Filter out internal filesystem paths from API response
    d.pop("input_path", None)
    d["is_sample_job"] = bool(d.get("is_sample_job", 0))
    d["keep_raw_input"] = bool(d.get("keep_raw_input", 0))
    return d


def save_job(job_data: Dict[str, Any]):
    """Inserts or updates a job record in the database."""
    fields = [
        "id", "display_name", "original_filename", "status", "progress_stage",
        "progress_stage_label", "worker_mode", "is_sample_job", "keep_raw_input",
        "created_at", "started_at", "finished_at", "duration_seconds",
        "attempt_count", "last_error_code", "last_error_message", "input_path",
        "input_bytes", "preflight_json", "scene_json", "quality_summary_json", "logs_json"
    ]

    # Serialize JSON fields
    serialized = dict(job_data)
    for field in ["preflight", "scene", "quality_summary", "logs"]:
        if field in serialized and serialized[field] is not None:
            serialized[f"{field}_json"] = json.dumps(serialized[field])
        else:
            serialized[f"{field}_json"] = None

    serialized["is_sample_job"] = 1 if serialized.get("is_sample_job") else 0
    serialized["keep_raw_input"] = 1 if serialized.get("keep_raw_input") else 0

    # Ensure all columns exist in the dict with default None
    for f in fields:
        if f not in serialized:
            serialized[f] = None

    col_names = ", ".join(fields)
    placeholders = ", ".join([f":{f}" for f in fields])
    updates = ", ".join([
        f"{f} = COALESCE(excluded.{f}, jobs.{f})" if f in ["input_path", "input_bytes"] else f"{f} = excluded.{f}"
        for f in fields if f != "id"
    ])

    sql = f"""
        INSERT INTO jobs ({col_names})
        VALUES ({placeholders})
        ON CONFLICT(id) DO UPDATE SET {updates}
    """

    with get_db_connection() as conn:
        conn.execute(sql, serialized)
        conn.commit()


def get_job_by_id(job_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a single job by UUID."""
    with get_db_connection() as conn:
        cursor = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = cursor.fetchone()
        if row:
            return row_to_job_dict(row)
    return None


def get_job_internal(job_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves complete internal job record including internal storage paths."""
    with get_db_connection() as conn:
        cursor = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = cursor.fetchone()
        if row:
            return dict(row)
    return None


def list_jobs(limit: int = 50) -> List[Dict[str, Any]]:
    """Lists recent jobs ordered by creation timestamp."""
    with get_db_connection() as conn:
        cursor = conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        return [row_to_job_dict(r) for r in cursor.fetchall()]


def delete_job_record(job_id: str) -> bool:
    """Removes a job record from the database."""
    with get_db_connection() as conn:
        cursor = conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        conn.commit()
        return cursor.rowcount > 0
