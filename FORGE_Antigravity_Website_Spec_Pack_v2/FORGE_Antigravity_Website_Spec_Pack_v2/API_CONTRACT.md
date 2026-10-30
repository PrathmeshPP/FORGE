# Proposed API Contract

**Status:** Interface proposal. Antigravity must implement consistently and adjust only after inspecting the current repo; document any needed change before coding.

Base path: `/api/v1` (preferred so the interface can evolve). The frontend and backend should share one origin in production. All routes are private to the authorized Tailscale group in v1.

## 1. Common rules

- JSON response for metadata. Upload is `multipart/form-data`.
- Use stable machine-readable `code` values and clear, non-sensitive `message` text.
- Never return absolute filesystem paths, raw command strings, full tracebacks, environment variables or secrets.
- Job IDs are random UUIDs. They are identifiers, not an authorization mechanism.
- IDs/paths are validated server-side. Do not accept `output_path`, `script_path`, `model_path`, `command`, arbitrary URLs or user-selected executable.
- Appropriate status codes: `400` malformed request; `401/403` access denial if application auth is later implemented; `404` unknown job/artifact; `413` request too large; `415` unsupported media; `422` invalid/corrupt video; `429` queue/rate limit; `500` generic unexpected failure; `503` worker unavailable or disk/resource safety gate triggered.

## 2. `GET /api/v1/health`

Purpose: a lightweight readiness signal. Must not expose sensitive host details.

Example success:
```json
{
  "status": "ok",
  "worker": "idle",
  "accepting_jobs": true,
  "limits": {
    "max_upload_bytes": 2147483648,
    "max_video_duration_seconds": 900,
    "accepted_extensions": [".mp4", ".mov"]
  }
}
```

Only return values that represent the actual current configuration. Do not leak GPU names, VRAM, absolute paths or environment values unless the user explicitly adds an operational diagnostics view with access control.

## 3. `POST /api/v1/jobs`

Purpose: create an upload/job record. Accept `file` field only; other metadata is optional and allowlisted. Streaming write to a server-created temporary file under a UUID directory.

Success: `201 Created` with a job object and `status: "validating"` or `"preflight"`.

Failure codes:
- `UPLOAD_MISSING`
- `UPLOAD_TOO_LARGE`
- `UNSUPPORTED_VIDEO`
- `INVALID_VIDEO`
- `DISK_SPACE_LOW`
- `QUEUE_FULL`
- `WORKER_UNAVAILABLE`

If upload is separated from “Start reconstruction,” upload and preflight return a draft job, and `POST /jobs/{id}/start` explicitly starts it. Use only one of these designs; don't make ambiguous duplicate route behaviours.

## 4. `POST /api/v1/jobs/{job_id}/start` (if two-stage flow is used)

Purpose: start an already validated/preflighted job. Must verify job state and queue capacity. Repeated start requests must not create duplicate worker processes; return current job state if already running.

## 5. `GET /api/v1/jobs`

Return a bounded, paginated/list-limited set of jobs with safe fields only: ID, display name, upload size, status, safe progress stage, timestamps and result availability. Do not return logs, local paths or arbitrary stored JSON.

## 6. `GET /api/v1/jobs/{job_id}`

Returns safe job metadata and a limited quality summary. Never return the raw exception or private log file.

## 7. `POST /api/v1/jobs/{job_id}/cancel`

Only queued/running jobs may be cancelled. Make cancellation idempotent and reflect intermediate state (`cancelling`) if process termination has not completed. Ensure cancellation kills child process(es) safely.

## 8. `GET /api/v1/jobs/{job_id}/artifacts`

Returns the allowlisted generated artifacts, each with display name, type, byte size and a server-generated download/stream route. Only return artifacts after job state is `ready` and output validation passed.

## 9. `GET /api/v1/jobs/{job_id}/artifacts/{artifact_key}`

- `artifact_key` is an allowlisted logical key like `primary_model` or `texture_atlas`, not a filename/path.
- Look it up in the server-generated artifact manifest.
- Resolve the path and verify it remains under the job root; reject symlink escape.
- Use a correct content type and safe `Content-Disposition` for downloads.
- Never serve arbitrary job files.

## 10. `DELETE /api/v1/jobs/{job_id}`

Requires an explicit user confirmation in the UI. Verify state, stop any process first, validate resolved paths, delete only that job's data, and keep an audit event with job ID/time/outcome. Do not accept a path in the request. Return `204 No Content` on success.

## 11. Status enum

`receiving`, `validating`, `preflight`, `queued`, `running`, `validating_output`, `ready`, `failed`, `cancelling`, `cancelled`, `deleting`.

Terminal states: `ready`, `failed`, `cancelled`. An interrupted job can use `failed` plus `error_code: PIPELINE_INTERRUPTED`.

## 12. Progress update approach

Use short polling (e.g. 1–2 seconds while active, back off when idle) or Server-Sent Events only if the current architecture supports it cleanly. The progress stage must come from a real worker event/log parser; never invent a percentage. Display stage + elapsed time when the exact percent cannot be computed.
