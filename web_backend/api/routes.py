"""
F.O.R.G.E. Website Backend — API Router
Implements the endpoints specified in API_CONTRACT.md and TRD.md.
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
from web_backend.worker.mock_worker import mock_worker
from web_backend.services.model_discovery_service import (
    get_recent_qualifying_models,
    get_model_file_path,
)
from web_backend.config import FORGE_PIPELINE_ROOT

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

    worker_status = "busy" if mock_worker.is_busy() else "idle"
    accepting = not mock_worker.is_queue_full()
    active_id = mock_worker.get_active_job_id()

    return HealthResponse(
        status="ok",
        worker=worker_status,
        accepting_jobs=accepting,
        worker_mode=FORGE_WORKER_MODE,
        active_job_id=active_id,
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
async def start_reconstruction_job(job_id: str):
    """Transitions a validated preflight job into the active queue."""
    clean_id = validate_uuid_string(job_id)
    try:
        job = await start_job(clean_id)
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
async def cancel_reconstruction_job(job_id: str):
    """Cancels a queued or running job."""
    clean_id = validate_uuid_string(job_id)
    try:
        return await cancel_job(clean_id)
    except JobServiceError as e:
        raise HTTPException(
            status_code=e.status_code,
            detail={"code": e.code, "message": e.message, "job_id": clean_id}
        )


@router.delete("/jobs/{job_id}", status_code=204)
async def delete_reconstruction_job(job_id: str):
    """Deletes a job and permanently removes its associated local files."""
    clean_id = validate_uuid_string(job_id)
    try:
        await delete_job(clean_id)
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
        "material_def": ("textured_mesh.mtl", "MTL"),
        "point_cloud": ("surface_mesh_photographic.ply", "PLY"),
        "manifest": ("artifact_manifest.json", "JSON"),
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
    Enforces logical key and filename allowlist and strict path containment.
    """
    clean_id = validate_uuid_string(job_id)
    job = get_job_by_id(clean_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail={"code": "JOB_NOT_FOUND", "message": f"Job '{clean_id}' was not found."}
        )

    if job.get("status") != "ready":
        raise HTTPException(
            status_code=403,
            detail={"code": "JOB_NOT_READY", "message": f"Job '{clean_id}' is not in ready state."}
        )

    allowlist_map = {
        "primary_model": ("textured_mesh.obj", "text/plain"),
        "textured_mesh.obj": ("textured_mesh.obj", "text/plain"),
        "material_def": ("textured_mesh.mtl", "text/plain"),
        "textured_mesh.mtl": ("textured_mesh.mtl", "text/plain"),
        "point_cloud": ("surface_mesh_photographic.ply", "application/octet-stream"),
        "surface_mesh_photographic.ply": ("surface_mesh_photographic.ply", "application/octet-stream"),
        "texture_atlas": ("texture_atlas.png", "image/png"),
        "texture_atlas.png": ("texture_atlas.png", "image/png"),
        "manifest": ("artifact_manifest.json", "application/json"),
        "artifact_manifest.json": ("artifact_manifest.json", "application/json"),
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
        headers={"Content-Disposition": f'inline; filename="{filename}"'}
    )


# ------------------------------------------------------------------------------
# Recent Qualifying Models & 3D Model Asset Endpoints
# ------------------------------------------------------------------------------

@router.get("/recent-models")
def get_recent_models(limit: int = 5):
    """
    Returns the top qualifying genuinely completed reconstructions (up to 5),
    sorted by completion timestamp (newest first).
    """
    return get_recent_qualifying_models(limit=min(limit, 10))


@router.get("/models/{model_id}/glb")
def download_model_glb(model_id: str):
    """
    Streams the verified genuine GLB for the requested model.
    """
    file_path = get_model_file_path(model_id)
    if not file_path or not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail={"code": "MODEL_NOT_FOUND", "message": f"Model GLB for '{model_id}' was not found on host."}
        )

    return FileResponse(
        path=str(file_path),
        filename=f"{model_id}.glb",
        media_type="model/gltf-binary",
        headers={
            "Content-Disposition": f'inline; filename="{model_id}.glb"',
            "Cache-Control": "public, max-age=3600"
        }
    )


@router.get("/models/{model_id}/thumbnail")
def get_model_thumbnail(model_id: str):
    """
    Serves the preview thumbnail or contact sheet for the requested model.
    """
    scene_dir = FORGE_PIPELINE_ROOT / "outputs" / "indoor" / model_id
    if not scene_dir.exists():
        job_dir = JOBS_DIR / model_id / "outputs"
        if job_dir.exists():
            scene_dir = job_dir

    candidate_thumbs = [
        scene_dir / "preview_thumbnail.png",
        scene_dir / "video_audit_contact_sheet.png",
        scene_dir / "renders" / "baseline_textured_59.6s.png",
        scene_dir / "renders" / "candidate_textured_59.6s.png",
    ]

    for thumb_path in candidate_thumbs:
        if thumb_path.exists():
            return FileResponse(
                path=str(thumb_path),
                media_type="image/png",
                headers={"Cache-Control": "public, max-age=86400"}
            )

    # Fallback placeholder SVG
    svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" width="400" height="240" viewBox="0 0 400 240">
        <rect width="400" height="240" fill="#0f172a"/>
        <rect x="20" y="20" width="360" height="200" rx="8" fill="#1e293b" stroke="#38bdf8" stroke-width="1" stroke-dasharray="4"/>
        <text x="200" y="115" fill="#38bdf8" font-family="system-ui, sans-serif" font-size="16" font-weight="600" text-anchor="middle">F.O.R.G.E. 3D Digital Twin</text>
        <text x="200" y="145" fill="#94a3b8" font-family="system-ui, sans-serif" font-size="12" text-anchor="middle">{model_id}</text>
    </svg>'''
    return Response(content=svg_content, media_type="image/svg+xml")


@router.get("/models/{model_id}/metadata")
def get_model_metadata(model_id: str):
    """
    Returns full scene navigation and quality metadata for the model.
    """
    scene_dir = FORGE_PIPELINE_ROOT / "outputs" / "indoor" / model_id
    nav_path = scene_dir / "scene_navigation_metadata.json"
    meta_path = scene_dir / "reconstruction_metadata.json"

    result = {}
    if nav_path.exists():
        try:
            with open(nav_path, "r", encoding="utf-8") as f:
                result["navigation"] = json.load(f)
        except Exception:
            pass
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                result["reconstruction"] = json.load(f)
        except Exception:
            pass

    if not result:
        raise HTTPException(
            status_code=404,
            detail={"code": "METADATA_NOT_FOUND", "message": f"Metadata for '{model_id}' was not found."}
        )

    return result
