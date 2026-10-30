"""
F.O.R.G.E. Website Backend — API Router
Implements the endpoints specified in API_CONTRACT.md.
"""

import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Response
from fastapi.responses import JSONResponse, FileResponse

from web_backend.config import (
    FORGE_MAX_UPLOAD_BYTES,
    FORGE_MAX_VIDEO_SECONDS,
    FORGE_ACCEPTED_EXTENSIONS,
    FORGE_WORKER_MODE,
    FORGE_DATA_DIR,
    JOBS_DIR,
)
from web_backend.models import (
    HealthResponse,
    HealthLimits,
    HostInfo,
    JobResponse,
    ArtifactItem,
    ErrorResponse,
)
from web_backend.services.job_service import (
    handle_streaming_upload,
    start_job,
    cancel_job,
    delete_job,
    check_free_disk_space,
    verify_path_containment,
    JobServiceError,
)
from web_backend.database import get_job_by_id, list_jobs

router = APIRouter(prefix="/api/v1")


def validate_uuid_string(val: str) -> str:
    """Validates that a string is a well-formed UUID."""
    try:
        return str(uuid.UUID(val))
    except (ValueError, AttributeError):
        raise HTTPException(
            status_code=400,
            detail={"code": "INVALID_JOB_ID", "message": f"Identifier '{val}' is not a valid UUID."}
        )


@router.get("/health", response_model=HealthResponse)
def get_health():
    """
    Returns lightweight service readiness status without exposing sensitive environment values.
    """
    free_disk_gb = 0.0
    try:
        import shutil
        total, used, free = shutil.disk_usage(str(FORGE_DATA_DIR))
        free_disk_gb = round(free / (1024 ** 3), 1)
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        worker="idle",
        accepting_jobs=True,
        worker_mode=FORGE_WORKER_MODE,
        limits=HealthLimits(
            max_upload_bytes=FORGE_MAX_UPLOAD_BYTES,
            max_video_duration_seconds=FORGE_MAX_VIDEO_SECONDS,
            accepted_extensions=FORGE_ACCEPTED_EXTENSIONS,
            max_active_jobs=1
        ),
        host_info=HostInfo(
            platform="Windows 11 Pro",
            gpu_model="NVIDIA GeForce RTX 2070 SUPER",
            vram_total_mb=8192,
            vram_free_mb=8100,
            disk_free_gb=free_disk_gb
        )
    )


@router.post("/jobs", response_model=JobResponse, status_code=201)
async def create_job(
    file: UploadFile = File(...),
    keep_raw_input: bool = Form(False)
):
    """
    Streams an uploaded indoor video, executes OpenCV container & decodability checks,
    and returns the preflighted job record.
    """
    try:
        job = await handle_streaming_upload(file=file, keep_raw_input=keep_raw_input)
        return job
    except JobServiceError as e:
        raise HTTPException(
            status_code=e.status_code,
            detail={"code": e.code, "message": e.message}
        )


@router.post("/jobs/{job_id}/start", response_model=JobResponse)
def start_reconstruction_job(job_id: str):
    """Transitions a validated preflight job into the active queue."""
    clean_id = validate_uuid_string(job_id)
    try:
        job = start_job(clean_id)
        return job
    except JobServiceError as e:
        raise HTTPException(
            status_code=e.status_code,
            detail={"code": e.code, "message": e.message, "job_id": clean_id}
        )


@router.get("/jobs", response_model=List[JobResponse])
def get_jobs_list(limit: int = 50):
    """Returns bounded recent job list."""
    return list_jobs(limit=min(limit, 100))


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str):
    """Returns safe metadata for a specific job."""
    clean_id = validate_uuid_string(job_id)
    job = get_job_by_id(clean_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail={"code": "JOB_NOT_FOUND", "message": f"Job '{clean_id}' was not found.", "job_id": clean_id}
        )
    return job


@router.post("/jobs/{job_id}/cancel", response_model=JobResponse)
def cancel_reconstruction_job(job_id: str):
    """Cancels a queued or running job."""
    clean_id = validate_uuid_string(job_id)
    try:
        return cancel_job(clean_id)
    except JobServiceError as e:
        raise HTTPException(
            status_code=e.status_code,
            detail={"code": e.code, "message": e.message, "job_id": clean_id}
        )


@router.delete("/jobs/{job_id}", status_code=204)
def delete_reconstruction_job(job_id: str):
    """Deletes a job and permanently removes its associated local files."""
    clean_id = validate_uuid_string(job_id)
    try:
        delete_job(clean_id)
        return Response(status_code=204)
    except JobServiceError as e:
        raise HTTPException(
            status_code=e.status_code,
            detail={"code": e.code, "message": e.message, "job_id": clean_id}
        )


@router.get("/jobs/{job_id}/artifacts", response_model=List[ArtifactItem])
def get_job_artifacts(job_id: str):
    """
    Returns allowlisted generated 3D artifacts for a completed job.
    Only available after output validation passes.
    """
    clean_id = validate_uuid_string(job_id)
    job = get_job_by_id(clean_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail={"code": "JOB_NOT_FOUND", "message": f"Job '{clean_id}' was not found.", "job_id": clean_id}
        )

    if job.get("status") != "ready":
        return []

    job_dir = JOBS_DIR / clean_id / "outputs"
    if not job_dir.exists():
        return []

    artifacts: List[ArtifactItem] = []
    allowlisted_files = {
        "primary_model": ("textured_mesh.obj", "OBJ"),
        "texture_atlas": ("texture_atlas.png", "PNG"),
        "material_def": ("textured_mesh.mtl", "MTL"),
        "point_cloud": ("surface_mesh_photographic.ply", "PLY"),
    }

    for key, (filename, fmt) in allowlisted_files.items():
        file_path = job_dir / filename
        if file_path.exists() and verify_path_containment(file_path, job_dir):
            artifacts.append(
                ArtifactItem(
                    key=key,
                    filename=filename,
                    format=fmt,
                    file_size_bytes=file_path.stat().st_size,
                    download_url=f"/api/v1/jobs/{clean_id}/artifacts/{key}"
                )
            )

    return artifacts


@router.get("/jobs/{job_id}/artifacts/{artifact_key}")
def download_job_artifact(job_id: str, artifact_key: str):
    """
    Serves a validated artifact file.
    Enforces logical key allowlist and strict path containment.
    """
    clean_id = validate_uuid_string(job_id)
    job = get_job_by_id(clean_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail={"code": "JOB_NOT_FOUND", "message": f"Job '{clean_id}' was not found."}
        )

    allowlist_map = {
        "primary_model": ("textured_mesh.obj", "text/plain"),
        "texture_atlas": ("texture_atlas.png", "image/png"),
        "material_def": ("textured_mesh.mtl", "text/plain"),
        "point_cloud": ("surface_mesh_photographic.ply", "application/octet-stream"),
    }

    if artifact_key not in allowlist_map:
        raise HTTPException(
            status_code=404,
            detail={"code": "ARTIFACT_NOT_FOUND", "message": f"Unknown artifact key '{artifact_key}'."}
        )

    filename, media_type = allowlist_map[artifact_key]
    job_dir = (JOBS_DIR / clean_id / "outputs").resolve()
    file_path = (job_dir / filename).resolve()

    if not verify_path_containment(file_path, job_dir) or not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail={"code": "ARTIFACT_NOT_FOUND", "message": f"Artifact file '{filename}' does not exist on host."}
        )

    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )

