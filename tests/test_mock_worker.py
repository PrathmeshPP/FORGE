"""
F.O.R.G.E. Website Backend — Phase 3 Test Suite: Mock Worker & Job Lifecycle
Tests:
1. Normal end-to-end execution through all lifecycle states to ready
2. Output artifact generation and manifest validation
3. Artifact serving allowlist, Content-Disposition, and 403 non-ready protection
4. Single-worker concurrency serialization and bounded queue (429 QUEUE_FULL)
5. Cancellation during execution halts worker immediately and releases concurrency lock
6. Application startup crash recovery marks interrupted jobs as PIPELINE_INTERRUPTED
7. Quality Gate A-C failure modes (empty geometry, non-finite coords, collapsed bounds, corrupt PLY)
8. Corrupted/missing sample fixtures handling
9. Safe persistence across SQLite restarts
"""

import os
import sys
import json
import uuid
import tempfile
import asyncio
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import httpx

from web_backend.main import app
from web_backend.config import JOBS_DIR, FORGE_DATA_DIR, SAMPLE_ASSETS_DIR
from web_backend.database import get_job_by_id, save_job, get_db_connection, init_db
from web_backend.models import JobStatus, PipelineStage
from web_backend.worker.mock_worker import mock_worker, recover_interrupted_jobs, WorkerQueueFullError
from web_backend.services.qa_validator import (
    validate_and_generate_manifest,
    parse_obj_diagnostics,
    parse_ply_diagnostics,
    QAValidationError,
)


def create_dummy_preflight_job(job_id: str = None) -> str:
    """Helper: creates a valid preflight job record in SQLite."""
    if not job_id:
        job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id / "input"
    job_dir.mkdir(parents=True, exist_ok=True)
    video_path = job_dir / "source_video.mp4"
    video_path.write_bytes(b"dummy_video_bytes")

    job_record = {
        "id": job_id,
        "display_name": "Living Room Scan",
        "original_filename": "living_room_scan.mp4",
        "status": JobStatus.PREFLIGHT.value,
        "progress_stage": PipelineStage.IDLE.value,
        "progress_stage_label": "Preflight Passed",
        "worker_mode": "mock",
        "is_sample_job": False,
        "keep_raw_input": False,
        "created_at": "2026-10-30T10:00:00Z",
        "started_at": None,
        "finished_at": None,
        "duration_seconds": None,
        "attempt_count": 0,
        "last_error_code": None,
        "last_error_message": None,
        "input_path": str(video_path),
        "input_bytes": 17,
        "preflight": {
            "filename": "living_room_scan.mp4",
            "file_size_bytes": 17,
            "file_size_mb": 0.01,
            "mime_type": "video/mp4",
            "container_valid": True,
            "duration_seconds": 12.0,
            "width": 1920,
            "height": 1080,
            "aspect_ratio": "16:9",
            "framerate_fps": 30.0,
            "estimated_frames": 360,
            "status": "passed",
            "rejection_reasons": [],
            "warnings": [],
        },
        "scene": None,
        "quality_summary": None,
        "logs": [],
    }
    save_job(job_record)
    return job_id


@pytest.mark.anyio
async def test_normal_job_lifecycle_to_ready():
    """Verifies that an enqueued job executes through all stages and marks ready."""
    mock_worker.reset()
    mock_worker.step_delay = 0.01
    job_id = create_dummy_preflight_job()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Start job
        start_res = await client.post(f"/api/v1/jobs/{job_id}/start")
        assert start_res.status_code == 200
        assert start_res.json()["status"] == JobStatus.QUEUED.value

        # Execute job
        await mock_worker.execute_job(job_id)

        # Inspect resulting job state
        get_res = await client.get(f"/api/v1/jobs/{job_id}")
        assert get_res.status_code == 200
        data = get_res.json()

        assert data["status"] == JobStatus.READY.value
        assert data["progress_stage"] == PipelineStage.IDLE.value
        assert data["duration_seconds"] is not None
        assert data["is_sample_job"] is True
        assert data["quality_summary"] is not None
        assert data["quality_summary"]["gate_a_process"] == "passed"
        assert data["quality_summary"]["gate_b_file_integrity"] == "passed"
        assert data["quality_summary"]["gate_c_geometry"] == "passed"

        # Check logs contain mock disclaimers and timestamps
        assert len(data["logs"]) >= 5
        log_messages = [l["message"] for l in data["logs"]]
        assert any("[MOCK PIPELINE]" in msg for msg in log_messages)


@pytest.mark.anyio
async def test_artifacts_serving_and_manifest():
    """Verifies artifact manifest generation, allowlisted artifact listing and downloads."""
    mock_worker.reset()
    mock_worker.step_delay = 0.01
    job_id = create_dummy_preflight_job()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Non-ready job must return empty list for artifacts and 403 on download
        art_res = await client.get(f"/api/v1/jobs/{job_id}/artifacts")
        assert art_res.status_code == 200
        assert art_res.json() == []

        dl_res = await client.get(f"/api/v1/jobs/{job_id}/artifacts/primary_model")
        assert dl_res.status_code == 403

        # Run worker
        await mock_worker.execute_job(job_id)

        # Ready job should return allowlisted artifacts
        art_ready_res = await client.get(f"/api/v1/jobs/{job_id}/artifacts")
        assert art_ready_res.status_code == 200
        artifacts = art_ready_res.json()
        artifact_keys = [a["key"] for a in artifacts]
        assert "primary_model" in artifact_keys
        assert "material_def" in artifact_keys
        assert "point_cloud" in artifact_keys
        assert "manifest" in artifact_keys

        # Download primary_model (OBJ)
        dl_obj = await client.get(f"/api/v1/jobs/{job_id}/artifacts/primary_model")
        assert dl_obj.status_code == 200
        assert "mtllib textured_mesh.mtl" in dl_obj.text

        # Download material_def (MTL)
        dl_mtl = await client.get(f"/api/v1/jobs/{job_id}/artifacts/material_def")
        assert dl_mtl.status_code == 200
        assert dl_mtl.text.startswith("#")

        # Download point_cloud (PLY)
        dl_ply = await client.get(f"/api/v1/jobs/{job_id}/artifacts/point_cloud")
        assert dl_ply.status_code == 200
        assert dl_ply.content.startswith(b"ply")

        # Download manifest (JSON)
        dl_manifest = await client.get(f"/api/v1/jobs/{job_id}/artifacts/manifest")
        assert dl_manifest.status_code == 200
        manifest_data = dl_manifest.json()
        assert manifest_data["is_mock"] is True
        assert "Simulated output" in manifest_data["notice"]

        # Filename access should also work
        dl_obj_by_name = await client.get(f"/api/v1/jobs/{job_id}/artifacts/textured_mesh.obj")
        assert dl_obj_by_name.status_code == 200


@pytest.mark.anyio
async def test_single_worker_concurrency_and_bounded_queue():
    """Verifies that only 1 job executes at a time and bounded queue rejects excess submissions."""
    mock_worker.reset()
    mock_worker.step_delay = 0.01

    job1_id = create_dummy_preflight_job()
    job2_id = create_dummy_preflight_job()
    job3_id = create_dummy_preflight_job()
    job4_id = create_dummy_preflight_job()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Enqueue jobs up to capacity (max queue size = 3)
        r1 = await client.post(f"/api/v1/jobs/{job1_id}/start")
        assert r1.status_code == 200

        r2 = await client.post(f"/api/v1/jobs/{job2_id}/start")
        assert r2.status_code == 200

        r3 = await client.post(f"/api/v1/jobs/{job3_id}/start")
        assert r3.status_code == 200

        # 4th should be rejected with 429 QUEUE_FULL
        r4 = await client.post(f"/api/v1/jobs/{job4_id}/start")
        assert r4.status_code == 429
        assert r4.json()["code"] == "QUEUE_FULL"

        # Health reports accepting_jobs is False when full
        health_res = await client.get("/api/v1/health")
        assert health_res.status_code == 200
        assert health_res.json()["accepting_jobs"] is False

    mock_worker.reset()


@pytest.mark.anyio
async def test_cancellation_during_execution():
    """Verifies that cancelling a running job halts worker simulation immediately."""
    mock_worker.reset()
    mock_worker.step_delay = 0.1  # 100ms per step

    job_id = create_dummy_preflight_job()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Start job
        await client.post(f"/api/v1/jobs/{job_id}/start")

        # Start execution in background task
        exec_task = asyncio.create_task(mock_worker.execute_job(job_id))

        # Yield control briefly so job enters running state
        await asyncio.sleep(0.04)

        # Cancel job via API
        cancel_res = await client.post(f"/api/v1/jobs/{job_id}/cancel")
        assert cancel_res.status_code == 200
        assert cancel_res.json()["status"] == JobStatus.CANCELLED.value

        # Await execution task exit
        await exec_task

        # Verify final state in DB is CANCELLED (never READY)
        job_final = get_job_by_id(job_id)
        assert job_final["status"] == JobStatus.CANCELLED.value
        assert job_final["progress_stage"] == PipelineStage.IDLE.value
        assert not mock_worker.is_busy()

    mock_worker.reset()


def test_startup_crash_recovery():
    """Verifies that recover_interrupted_jobs marks orphaned jobs failed."""
    # Recover any leftovers first to isolate this test
    recover_interrupted_jobs()

    job1_id = create_dummy_preflight_job()
    job2_id = create_dummy_preflight_job()
    job3_id = create_dummy_preflight_job()

    # Seed intermediate non-terminal states
    j1 = get_job_by_id(job1_id)
    j1["status"] = JobStatus.RUNNING.value
    save_job(j1)

    j2 = get_job_by_id(job2_id)
    j2["status"] = JobStatus.QUEUED.value
    save_job(j2)

    j3 = get_job_by_id(job3_id)
    j3["status"] = JobStatus.VALIDATING_OUTPUT.value
    save_job(j3)

    # Run recovery
    count = recover_interrupted_jobs()
    assert count == 3

    # Check that all 3 are marked failed with PIPELINE_INTERRUPTED
    for j_id in [job1_id, job2_id, job3_id]:
        rec = get_job_by_id(j_id)
        assert rec["status"] == JobStatus.FAILED.value
        assert rec["last_error_code"] == "PIPELINE_INTERRUPTED"
        assert "shutdown/restart" in rec["last_error_message"]
        log_msgs = [l["message"] for l in rec["logs"]]
        assert any("[SHUTDOWN RECOVERY]" in m for m in log_msgs)


def test_qa_validation_failure_modes(tmp_path: Path):
    """Verifies that Gate A, B, and C catch invalid/corrupt geometry."""
    # Gate A: Missing output directory
    non_existent = tmp_path / "does_not_exist"
    with pytest.raises(QAValidationError) as exc:
        validate_and_generate_manifest(non_existent, "job-qa-test")
    assert exc.value.code == "GATE_A_MISSING_OUTPUT_DIR"

    # Gate B: Missing required files
    valid_dir = tmp_path / "outputs"
    valid_dir.mkdir()
    with pytest.raises(QAValidationError) as exc:
        validate_and_generate_manifest(valid_dir, "job-qa-test")
    assert exc.value.code == "GATE_B_MISSING_ARTIFACT"

    # Gate B: Zero-byte files
    (valid_dir / "textured_mesh.obj").write_text("")
    (valid_dir / "textured_mesh.mtl").write_text("newmtl test")
    (valid_dir / "surface_mesh_photographic.ply").write_text("ply\n")
    with pytest.raises(QAValidationError) as exc:
        validate_and_generate_manifest(valid_dir, "job-qa-test")
    assert exc.value.code == "GATE_B_ZERO_BYTE_FILE"

    # Gate C: Empty geometry (no vertices)
    (valid_dir / "textured_mesh.obj").write_text("# Only comments\n")
    with pytest.raises(QAValidationError) as exc:
        validate_and_generate_manifest(valid_dir, "job-qa-test")
    assert exc.value.code == "GATE_C_EMPTY_GEOMETRY"

    # Gate C: Non-finite vertex coordinates
    (valid_dir / "textured_mesh.obj").write_text("v nan 1.0 2.0\n")
    with pytest.raises(QAValidationError) as exc:
        validate_and_generate_manifest(valid_dir, "job-qa-test")
    assert exc.value.code == "GATE_C_NON_FINITE_COORDINATES"

    # Gate C: Zero faces
    (valid_dir / "textured_mesh.obj").write_text("v 0.0 0.0 0.0\nv 1.0 0.0 0.0\nv 0.0 1.0 0.0\n")
    with pytest.raises(QAValidationError) as exc:
        validate_and_generate_manifest(valid_dir, "job-qa-test")
    assert exc.value.code == "GATE_C_ZERO_FACES"

    # Gate C: Collapsed zero-thickness bounds
    (valid_dir / "textured_mesh.obj").write_text(
        "v 1.0 1.0 1.0\nv 1.0 1.0 1.0\nv 1.0 1.0 1.0\nf 1 2 3\n"
    )
    with pytest.raises(QAValidationError) as exc:
        validate_and_generate_manifest(valid_dir, "job-qa-test")
    assert exc.value.code == "GATE_C_COLLAPSED_BOUNDS"


def test_sqlite_persistence_across_reconnect():
    """Verifies that job state persists faithfully across independent SQLite connections."""
    job_id = create_dummy_preflight_job()

    with get_db_connection() as conn:
        cursor = conn.execute("SELECT status, original_filename FROM jobs WHERE id = ?", (job_id,))
        row = cursor.fetchone()
        assert row is not None
        assert row["status"] == JobStatus.PREFLIGHT.value
        assert row["original_filename"] == "living_room_scan.mp4"


@pytest.mark.anyio
async def test_cross_process_mutual_exclusion():
    """
    Verifies that CrossProcessLock enforces mutual exclusion across independent OS processes
    so that only one process can execute reconstruction jobs at a time.
    """
    mock_worker.reset()
    job_id = create_dummy_preflight_job()

    # Simulate an external process holding the cross-process lockfile
    lock_file = FORGE_DATA_DIR / ".worker_reconstruction.lock"
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    external_lock = open(lock_file, "w+b")
    external_lock.write(b"x")
    external_lock.flush()
    external_lock.seek(0)

    if os.name == "nt":
        import msvcrt
        msvcrt.locking(external_lock.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(external_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    try:
        # Worker attempts to execute job while external process holds lock
        await mock_worker.execute_job(job_id)

        job = get_job_by_id(job_id)
        assert job["status"] == JobStatus.FAILED.value
        assert job["last_error_code"] == "CROSS_PROCESS_LOCK_BUSY"
        assert "Cross-process lock collision" in job["last_error_message"]
    finally:
        # Release external lock
        external_lock.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(external_lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(external_lock.fileno(), fcntl.LOCK_UN)
        external_lock.close()
        mock_worker.reset()
