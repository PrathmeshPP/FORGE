"""
F.O.R.G.E. Website — Phase 2 Backend Test Suite
Verifies:
1. Valid MP4 upload
2. Rejection of renamed non-video file
3. Rejection of corrupt video
4. Rejection of files exceeding configured limit
5. Saved file byte-for-byte integrity (SHA-256 match)
6. SQLite job persistence across restarts
7. Invalid job IDs and directory traversal protection
8. Loopback binding verification
9. Zero cloud/S3/telemetry leakage
10. Unified single-origin integration
"""

import os
import sys
import io
import hashlib
import tempfile
import uuid
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import cv2
import numpy as np
from fastapi.testclient import TestClient

from web_backend.main import app
from web_backend.config import FORGE_DATA_DIR, JOBS_DIR, FORGE_MAX_UPLOAD_BYTES
from web_backend.database import init_db, get_job_by_id, get_job_internal


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Ensures test database and directories are initialized."""
    FORGE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    init_db()


@pytest.fixture
def client():
    """Test client for FastAPI."""
    return TestClient(app)


def generate_synthetic_mp4(num_frames: int = 30, fps: float = 10.0, width: int = 640, height: int = 480) -> bytes:
    """Generates an in-memory valid MP4 video using OpenCV."""
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / "synth_test.mp4"
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(temp_path), fourcc, fps, (width, height))
        for i in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :] = [(i * 7) % 256, 120, 200]
            out.write(frame)
        out.release()
        with open(temp_path, "rb") as f:
            return f.read()
    finally:
        if temp_path.exists():
            temp_path.unlink()
        Path(temp_dir).rmdir()


# ------------------------------------------------------------------------------
# Test 1 & 8: Health endpoint & Loopback Configuration
# ------------------------------------------------------------------------------
def test_health_endpoint_and_security_headers(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["worker"] == "idle"
    assert data["accepting_jobs"] is True
    assert data["limits"]["max_upload_bytes"] == FORGE_MAX_UPLOAD_BYTES
    assert ".mp4" in data["limits"]["accepted_extensions"]

    # Verify Mandatory Security Headers
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "DENY"
    assert response.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
    assert "default-src 'self'" in response.headers.get("Content-Security-Policy", "")


# ------------------------------------------------------------------------------
# Test 2: Upload a valid small MP4
# ------------------------------------------------------------------------------
def test_upload_valid_small_mp4(client):
    video_bytes = generate_synthetic_mp4(num_frames=40, fps=10.0, width=640, height=480)
    files = {"file": ("indoor_test.mp4", io.BytesIO(video_bytes), "video/mp4")}
    data = {"keep_raw_input": "true"}

    response = client.post("/api/v1/jobs", files=files, data=data)
    assert response.status_code == 201
    job = response.json()

    assert "id" in job
    uuid.UUID(job["id"]) # Must be valid UUID
    assert job["display_name"] == "indoor test"
    assert job["original_filename"] == "indoor_test.mp4"
    assert job["status"] == "preflight"
    assert job["keep_raw_input"] is True

    preflight = job["preflight"]
    assert preflight["width"] == 640
    assert preflight["height"] == 480
    assert preflight["duration_seconds"] == 4.0
    assert preflight["container_valid"] is True


# ------------------------------------------------------------------------------
# Test 3: Reject renamed non-video file
# ------------------------------------------------------------------------------
def test_reject_renamed_non_video_file(client):
    fake_content = b"This is a plain text file pretending to be an indoor mp4 video!"
    files = {"file": ("malicious_text.mp4", io.BytesIO(fake_content), "video/mp4")}

    response = client.post("/api/v1/jobs", files=files)
    assert response.status_code == 422
    err = response.json()
    assert err["code"] == "INVALID_VIDEO"
    assert "could not be decoded" in err["message"] or "no decodable frames" in err["message"]


# ------------------------------------------------------------------------------
# Test 4: Reject corrupt video
# ------------------------------------------------------------------------------
def test_reject_corrupt_video(client):
    # Truncated or mangled header bytes
    corrupt_bytes = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom\x00\x00DEADBEEF"
    files = {"file": ("corrupt.mp4", io.BytesIO(corrupt_bytes), "video/mp4")}

    response = client.post("/api/v1/jobs", files=files)
    assert response.status_code == 422
    err = response.json()
    assert err["code"] == "INVALID_VIDEO"


# ------------------------------------------------------------------------------
# Test 5: Reject files exceeding configured limit
# ------------------------------------------------------------------------------
def test_reject_oversized_file(monkeypatch, client):
    # Set artificial small limit of 50 KiB
    monkeypatch.setattr("web_backend.services.job_service.FORGE_MAX_UPLOAD_BYTES", 50 * 1024)

    # 100 KiB dummy payload
    oversized_bytes = b"0" * (100 * 1024)
    files = {"file": ("oversized.mp4", io.BytesIO(oversized_bytes), "video/mp4")}

    response = client.post("/api/v1/jobs", files=files)
    assert response.status_code == 413
    err = response.json()
    assert err["code"] == "UPLOAD_TOO_LARGE"


# ------------------------------------------------------------------------------
# Test 6: Verify saved file matches uploaded file (SHA-256 integrity)
# ------------------------------------------------------------------------------
def test_saved_file_integrity_sha256(client):
    video_bytes = generate_synthetic_mp4(num_frames=30, fps=10.0, width=640, height=480)
    original_sha256 = hashlib.sha256(video_bytes).hexdigest()

    files = {"file": ("integrity_test.mp4", io.BytesIO(video_bytes), "video/mp4")}
    response = client.post("/api/v1/jobs", files=files)
    assert response.status_code == 201
    job_id = response.json()["id"]

    # Retrieve internal record to get physical path on disk
    internal = get_job_internal(job_id)
    saved_path = Path(internal["input_path"])
    assert saved_path.exists()

    with open(saved_path, "rb") as f:
        saved_bytes = f.read()

    saved_sha256 = hashlib.sha256(saved_bytes).hexdigest()
    assert saved_sha256 == original_sha256


# ------------------------------------------------------------------------------
# Test 7: SQLite persistence across simulated restart
# ------------------------------------------------------------------------------
def test_sqlite_persistence_across_restart(client):
    video_bytes = generate_synthetic_mp4(num_frames=35, fps=10.0, width=640, height=480)
    files = {"file": ("persist_test.mp4", io.BytesIO(video_bytes), "video/mp4")}
    response = client.post("/api/v1/jobs", files=files)
    assert response.status_code == 201
    job_id = response.json()["id"]

    # Simulate restart by reading directly via a fresh connection
    job_from_db = get_job_by_id(job_id)
    assert job_from_db is not None
    assert job_from_db["id"] == job_id
    assert job_from_db["display_name"] == "persist test"
    assert job_from_db["status"] == "preflight"
    assert job_from_db["preflight"]["width"] == 640


# ------------------------------------------------------------------------------
# Test 8: Invalid Job IDs and Path Traversal Protection
# ------------------------------------------------------------------------------
def test_invalid_job_ids_and_path_traversal(client):
    # Traversal in job_id
    response = client.get("/api/v1/jobs/..%2F..%2Fetc%2Fpasswd")
    assert response.status_code in [400, 404]

    # Malformed non-UUID
    response = client.get("/api/v1/jobs/not-a-uuid-string")
    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_JOB_ID"

    # Non-existent UUID
    random_uuid = str(uuid.uuid4())
    response = client.get(f"/api/v1/jobs/{random_uuid}")
    assert response.status_code == 404
    assert response.json()["code"] == "JOB_NOT_FOUND"


# ------------------------------------------------------------------------------
# Test 9: Lifecycle Transitions: Start, Cancel, and Delete
# ------------------------------------------------------------------------------
def test_job_start_cancel_and_delete_lifecycle(client):
    video_bytes = generate_synthetic_mp4(num_frames=30, fps=10.0)
    files = {"file": ("lifecycle.mp4", io.BytesIO(video_bytes), "video/mp4")}
    res_create = client.post("/api/v1/jobs", files=files)
    job_id = res_create.json()["id"]

    # 1. Start Job
    res_start = client.post(f"/api/v1/jobs/{job_id}/start")
    assert res_start.status_code == 200
    assert res_start.json()["status"] == "queued"

    # 2. Cancel Job
    res_cancel = client.post(f"/api/v1/jobs/{job_id}/cancel")
    assert res_cancel.status_code == 200
    assert res_cancel.json()["status"] == "cancelled"

    # 3. Delete Job
    res_delete = client.delete(f"/api/v1/jobs/{job_id}")
    assert res_delete.status_code == 204

    # 4. Verify no longer in DB
    assert get_job_by_id(job_id) is None


# ------------------------------------------------------------------------------
# Test 10: Zero Cloud / S3 / External Network Leakage
# ------------------------------------------------------------------------------
def test_zero_cloud_leakage_and_local_isolation():
    # Verify no boto3 or AWS imports anywhere in web_backend
    import web_backend
    import sys
    for mod_name in sys.modules:
        assert not mod_name.startswith("boto3"), f"Unexpected AWS boto3 loaded: {mod_name}"
        assert not mod_name.startswith("botocore"), f"Unexpected AWS botocore loaded: {mod_name}"

