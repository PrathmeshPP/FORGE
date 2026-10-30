"""
F.O.R.G.E. Website Backend — Job Management Service
Handles streaming uploads, path security containment, directory isolation, SQLite state persistence,
and mock worker job scheduling.
"""

import os
import shutil
import uuid
import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import UploadFile

from web_backend.config import (
    FORGE_DATA_DIR,
    JOBS_DIR,
    FORGE_MAX_UPLOAD_BYTES,
    FORGE_MIN_FREE_DISK_PERCENT,
    FORGE_WORKER_MODE,
)
from web_backend.database import (
    save_job,
    get_job_by_id,
    get_job_internal,
    list_jobs,
    delete_job_record,
)
from web_backend.models import JobStatus, PipelineStage
from web_backend.services.video_validator import validate_and_probe_video, VideoValidationError
from web_backend.worker.mock_worker import mock_worker, WorkerQueueFullError


class JobServiceError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def check_free_disk_space(target_dir: Path) -> float:
    """Returns free disk space percentage for the drive hosting target_dir."""
    try:
        total, used, free = shutil.disk_usage(str(target_dir))
        percent_free = (free / total) * 100.0
        return percent_free
    except Exception:
        return 100.0


def verify_path_containment(target_path: Path, allowed_root: Path) -> bool:
    """Ensures target_path resolves strictly inside allowed_root (prevents traversal attacks)."""
    try:
        resolved_target = target_path.resolve()
        resolved_root = allowed_root.resolve()
        return resolved_root in resolved_target.parents or resolved_target == resolved_root
    except Exception:
        return False


async def handle_streaming_upload(
    file: UploadFile,
    keep_raw_input: bool = False
) -> Dict[str, Any]:
    """
    Streams an uploaded video to an isolated per-job directory outside Git.
    Enforces strict byte-size limits during streaming, prevents directory traversal,
    and runs substantive OpenCV video validation.
    """
    # 1. Disk Space Safety Check
    free_percent = check_free_disk_space(FORGE_DATA_DIR)
    if free_percent < FORGE_MIN_FREE_DISK_PERCENT:
        raise JobServiceError(
            "DISK_SPACE_LOW",
            f"Workstation host disk free space is below safety reserve ({free_percent:.1f}% free, minimum {FORGE_MIN_FREE_DISK_PERCENT}%).",
            status_code=503
        )

    # 2. Server-side UUID and Isolated Job Directory
    job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    # Server-generated safe filename (never use user-supplied path)
    original_filename = Path(file.filename or "video.mp4").name
    ext = Path(original_filename).suffix.lower() or ".mp4"
    safe_target_path = input_dir / f"source_video{ext}"

    # Verify directory containment
    if not verify_path_containment(safe_target_path, JOBS_DIR):
        shutil.rmtree(job_dir, ignore_errors=True)
        raise JobServiceError("ACCESS_DENIED", "Illegal target path traversal detected.", status_code=403)

    # 3. Stream Upload with Chunk-by-Chunk Byte Limit Enforcement
    total_bytes = 0
    chunk_size = 64 * 1024  # 64 KiB chunks

    try:
        with open(safe_target_path, "wb") as f_out:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                total_bytes += len(chunk)

                if total_bytes > FORGE_MAX_UPLOAD_BYTES:
                    raise JobServiceError(
                        "UPLOAD_TOO_LARGE",
                        f"Uploaded file exceeds maximum configured upload limit of {FORGE_MAX_UPLOAD_BYTES / (1024**3):.1f} GiB.",
                        status_code=413
                    )
                f_out.write(chunk)
    except Exception as e:
        # Cleanup incomplete file on failure
        shutil.rmtree(job_dir, ignore_errors=True)
        if isinstance(e, JobServiceError):
            raise e
        raise JobServiceError("UPLOAD_FAILED", f"Stream write failed: {str(e)}", status_code=500)

    if total_bytes == 0:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise JobServiceError("INVALID_VIDEO", "Uploaded file is zero bytes / empty.", status_code=400)

    # 4. OpenCV Substantive Video Content Validation
    try:
        preflight_meta = validate_and_probe_video(
            file_path=safe_target_path,
            original_filename=original_filename,
            file_size_bytes=total_bytes
        )
    except VideoValidationError as ve:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise JobServiceError(ve.code, ve.message, status_code=422)

    # 5. Persist Job Record to SQLite
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    clean_display_name = Path(original_filename).stem.replace("_", " ").strip() or "Indoor Space"

    job_record = {
        "id": job_id,
        "display_name": clean_display_name,
        "original_filename": original_filename,
        "status": JobStatus.PREFLIGHT.value,
        "progress_stage": PipelineStage.IDLE.value,
        "progress_stage_label": "Preflight Validation Passed",
        "worker_mode": FORGE_WORKER_MODE,
        "is_sample_job": False,
        "keep_raw_input": keep_raw_input,
        "created_at": now_iso,
        "started_at": None,
        "finished_at": None,
        "duration_seconds": None,
        "attempt_count": 0,
        "last_error_code": None,
        "last_error_message": None,
        "input_path": str(safe_target_path),
        "input_bytes": total_bytes,
        "preflight": preflight_meta.model_dump(),
        "scene": None,
        "quality_summary": None,
        "logs": [
            {
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                "stage": "ingestion",
                "message": (
                    f"Video received ({preflight_meta.width}x{preflight_meta.height}, "
                    f"{preflight_meta.duration_seconds}s, {preflight_meta.file_size_mb} MB). "
                    f"Container & frame decoding verified by OpenCV."
                ),
            }
        ],
    }

    save_job(job_record)
    return get_job_by_id(job_id) or job_record


async def start_job(job_id: str) -> Dict[str, Any]:
    """
    Transitions a preflighted job into the workstation worker queue and enqueues it.
    Prevents duplicate runs and validates queue capacity.
    """
    job = get_job_by_id(job_id)
    if not job:
        raise JobServiceError("JOB_NOT_FOUND", f"Job '{job_id}' does not exist.", status_code=404)

    # Idempotent: If already running, validating, or ready, return existing state
    if job["status"] in [
        JobStatus.RUNNING.value,
        JobStatus.VALIDATING_OUTPUT.value,
        JobStatus.READY.value,
    ]:
        return job

    # If job already queued, ensure worker queue has it and return
    if job["status"] == JobStatus.QUEUED.value:
        try:
            await mock_worker.enqueue_job(job_id)
        except WorkerQueueFullError:
            pass
        return job

    # Only preflight jobs (or retried failed jobs) can be queued
    if job["status"] not in [JobStatus.PREFLIGHT.value, JobStatus.FAILED.value]:
        raise JobServiceError(
            "INVALID_JOB_STATE",
            f"Cannot start job in state '{job['status']}'. Job must be in 'preflight' state.",
            status_code=400,
        )

    # Enqueue in mock worker
    try:
        await mock_worker.enqueue_job(job_id)
    except WorkerQueueFullError as qe:
        raise JobServiceError("QUEUE_FULL", str(qe), status_code=429)

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    job["status"] = JobStatus.QUEUED.value
    job["started_at"] = now_iso
    job["progress_stage"] = PipelineStage.INGESTION.value
    job["progress_stage_label"] = "Queued in local workstation task queue"
    job["attempt_count"] = job.get("attempt_count", 0) + 1
    job["last_error_code"] = None
    job["last_error_message"] = None
    job["logs"].append({
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        "stage": "ingestion",
        "message": "Job enqueued for deterministic local workstation processing.",
    })

    save_job(job)
    return job


async def cancel_job(job_id: str) -> Dict[str, Any]:
    """
    Transitions an active or queued job to cancelled, terminating worker simulation.
    """
    job = get_job_by_id(job_id)
    if not job:
        raise JobServiceError("JOB_NOT_FOUND", f"Job '{job_id}' does not exist.", status_code=404)

    if job["status"] in [JobStatus.READY.value, JobStatus.CANCELLED.value, JobStatus.FAILED.value]:
        return job

    await mock_worker.cancel_job(job_id)
    return get_job_by_id(job_id) or job


async def delete_job(job_id: str) -> bool:
    """
    Cancels any running tasks, deletes job files on disk, and removes SQLite record.
    Guarantees strict path containment within JOBS_DIR.
    """
    internal_record = get_job_internal(job_id)
    if not internal_record:
        raise JobServiceError("JOB_NOT_FOUND", f"Job '{job_id}' does not exist.", status_code=404)

    # Cancel if active or queued
    await mock_worker.cancel_job(job_id)

    # Delete files on disk
    job_dir = JOBS_DIR / job_id
    if verify_path_containment(job_dir, JOBS_DIR) and job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)

    # Remove from DB
    return delete_job_record(job_id)
