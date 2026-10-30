"""
F.O.R.G.E. Website Backend — Data Models & Schemas
Matches API_CONTRACT.md and TRD.md specifications.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    RECEIVING = "receiving"
    VALIDATING = "validating"
    PREFLIGHT = "preflight"
    QUEUED = "queued"
    RUNNING = "running"
    VALIDATING_OUTPUT = "validating_output"
    READY = "ready"
    FAILED = "failed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    DELETING = "deleting"


class PipelineStage(str, Enum):
    IDLE = "idle"
    INGESTION = "ingestion"
    KEYFRAME_EXTRACTION = "keyframe_extraction"
    SPATIAL_INFERENCE = "spatial_inference"
    DENSE_MESHING = "dense_meshing"
    TEXTURING = "texturing"
    QA_VERIFICATION = "qa_verification"


class HealthLimits(BaseModel):
    max_upload_bytes: int
    max_video_duration_seconds: float
    accepted_extensions: List[str]
    max_active_jobs: int = 1


class HostInfo(BaseModel):
    platform: str
    gpu_model: str
    vram_total_mb: int
    vram_free_mb: int
    disk_free_gb: float


class HealthResponse(BaseModel):
    status: str = "ok"
    worker: str = "idle"
    accepting_jobs: bool = True
    worker_mode: str = "mock"
    active_job_id: Optional[str] = None
    limits: HealthLimits
    host_info: Optional[HostInfo] = None


class PreflightMetadata(BaseModel):
    filename: str
    file_size_bytes: int
    file_size_mb: float
    mime_type: str = "video/mp4"
    container_valid: bool = True
    duration_seconds: float
    width: int
    height: int
    aspect_ratio: str
    framerate_fps: Optional[float] = None
    estimated_frames: Optional[int] = None
    status: str = "passed"
    rejection_reasons: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class JobLogEntry(BaseModel):
    timestamp: str
    stage: str
    message: str


class ArtifactItem(BaseModel):
    key: str
    filename: str
    format: str
    file_size_bytes: int
    download_url: str


class JobResponse(BaseModel):
    id: str
    display_name: str
    original_filename: str
    status: JobStatus
    progress_stage: PipelineStage = PipelineStage.IDLE
    progress_stage_label: str = ""
    worker_mode: str = "mock"
    is_sample_job: bool = False
    keep_raw_input: bool = False
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_seconds: Optional[int] = None
    attempt_count: int = 0
    last_error_code: Optional[str] = None
    last_error_message: Optional[str] = None
    preflight: Optional[PreflightMetadata] = None
    scene: Optional[Dict[str, Any]] = None
    quality_summary: Optional[Dict[str, Any]] = None
    logs: List[JobLogEntry] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    code: str
    message: str
    job_id: Optional[str] = None

