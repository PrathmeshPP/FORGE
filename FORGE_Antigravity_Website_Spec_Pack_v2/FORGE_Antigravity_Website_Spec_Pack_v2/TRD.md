# Technical Requirements Document (TRD)

**Status:** Proposed design. Antigravity must inspect the real code and revise this document if the repository differs; it may not silently change the key security decisions.

## 1. Repository separation and current source context

Website implementation target:
- Local folder: `C:\Users\25beevdt025\Desktop\FORGE_website`
- GitHub: <https://github.com/PrathmeshPP/FORGE_website> (appeared empty when checked on 30 October 2026; verify again).

Existing reconstruction source (read-only for this workstream): <https://github.com/PrathmeshPP/FORGE>. The public README documents:
- `run_pipeline.py` at repository root;
- reconstruction modules in `src/` and scripts in `scripts/`;
- a React 19 / TypeScript / Vite / Three.js viewer under `viewer/`;
- AWS Lambda/S3 video preflight under `aws/preflight/`;
- model assets and reports under `outputs/` and `docs/`.

Repository: <https://github.com/PrathmeshPP/FORGE>

This is a documentation-level snapshot. The checked-out branch may differ. Antigravity must inspect the source and real CLI arguments before writing an adapter. Do not edit the original source repo. Website code lives only in `FORGE_website`; configure the local pipeline checkout through a host-only `FORGE_PIPELINE_ROOT`. Until the separate model workstream provides a tested indoor contract in `PIPELINE_INTEGRATION_HANDOFF.md`, use a deterministic mock worker.

## 2. Architectural decision

### V1 topology: one-origin, self-hosted on Windows, private

The owner has a Windows workstation with an NVIDIA GPU with 8 GB VRAM. Exact GPU model, driver, CUDA/PyTorch support, available VRAM, RAM and free disk must be detected at runtime; do not assume compatibility. Use PowerShell and Windows-compatible process handling by default. One active reconstruction job is the safe default.

- Frontend: reuse and extend the existing Vite/React/Three.js viewer.
- Backend: Python API, preferably FastAPI if compatible with the current dependency environment; otherwise select the smallest maintained framework that meets the same controls.
- Worker: one bounded local worker on the owner's Windows workstation, invoking the separately maintained pipeline checkout through a controlled adapter after a validated handoff. No website task may edit the original pipeline repo.
- State: SQLite for job metadata and state transitions; files are stored in a dedicated, local, non-synchronized data root outside the static frontend/Git directory. Do not store video blobs in SQLite. Check available drives and confirm that the chosen folder is not OneDrive/Dropbox/Google Drive/iCloud synchronized.
- Access: the web process binds to `127.0.0.1` by default. Tailscale Serve proxies it to the authorized tailnet. Configure explicit Tailscale grants/ACLs; do not assume every tailnet member should have access.
- TLS: let Tailscale Serve supply HTTPS for remote clients. Local loopback HTTP is only between Serve and the local process.
- External cloud: no cloud GPU, no automatic S3 video transfer, no analytics or external API call in v1.

### Why not Vercel serverless for the reconstruction API?

The current Vercel Functions documentation lists a 4.5 MB function request/response payload limit and finite function execution windows. Large video uploads and multi-minute GPU reconstruction do not fit that model well. Vercel could host a public static UI in a later architecture, but it would not execute the local GPU pipeline and would add authentication/CORS/networking boundaries. V1 avoids this complexity by serving UI and API from one private host.

References: <https://vercel.com/docs/functions/limitations> and <https://vercel.com/docs/errors/function_payload_too_large>.

## 3. Process layout (suggested, subject to repository inspection)

All website source changes must be inside the `FORGE_website` checkout. The original `FORGE` repo may be inspected read-only. Reuse/copy viewer components only into the website repo after checking dependencies and preserving notices; do not change the original viewer as part of this task.

- `viewer/`: create/adapt the React/Three.js viewer inside the website repository. Inspect the original `FORGE/viewer/` read-only and copy/reuse only what is necessary, after checking dependencies and preserving applicable notices.
- `web_backend/` or `src/web/`: add API and service layer; select one location after inspecting the actual website repository conventions.
- `web_backend/api/`: routes and request/response schemas.
- `web_backend/services/`: upload validation, job repository, worker adapter, output validation, cleanup.
- `web_backend/worker/`: one bounded job runner.
- Runtime job input/output/logs and SQLite: place under a configured `FORGE_DATA_DIR` outside the Git repository and outside synchronized folders, for example `<approved-local-data-root>/jobs/<uuid>/...`.

Do not add this structure blindly if an existing pattern fits better. Never place uploads or private generated models in `viewer/public/`, `viewer/src/assets/`, the original repo's `outputs/` reference directories, or another publicly served directory. The website repo may include only small clearly licensed fixtures/sample models needed for development; never commit private videos or full generated job outputs.

## 4. API/runtime requirements

- Serve the production frontend and `/api/*` from one origin to avoid unnecessary CORS complexity.
- Development mode may use Vite's local proxy to the API, but development servers must stay on loopback and must not be exposed through Tailscale until specifically configured.
- Use streaming upload support (e.g. FastAPI `UploadFile`) rather than loading the entire video in memory.
- Accept `multipart/form-data`; require the expected field name; enforce `Content-Length` where available and count actual bytes while streaming because headers can be missing or misleading.
- Return JSON errors with stable error codes and short human-readable messages; never return tracebacks.
- Apply request timeouts and size limits at application boundaries.
- Use a health endpoint that exposes only online/busy status and configured upload limits—not absolute paths, hostnames, dependency secrets or detailed system information.

## 5. Job data model

Suggested SQLite fields:
- `id`: random UUID; never sequential.
- `display_name`: sanitized user-visible label.
- `original_filename`: display-only metadata; never used to form a filesystem path.
- `input_bytes`, `input_sha256` (optional), `input_duration_seconds`, `input_width`, `input_height`, `input_fps` when known.
- `status`: allowlisted state enum.
- `progress_stage`: finite stage label, not arbitrary client-supplied text.
- `created_at`, `updated_at`, `started_at`, `finished_at` (UTC ISO timestamps).
- `duration_seconds` only when measured.
- `attempt_count`, `last_error_code`, `quality_report_json`.
- relative paths derived server-side under configured data root, never from a client field.

A UUID is an identifier, not an authorization boundary. The private Tailscale policy is the access boundary in v1. This means all authorized teammates may see the shared job list; make that explicit. If private per-user jobs are required, implement identity and per-job authorization before release.

## 6. Worker execution

- One active GPU reconstruction job at a time; bounded waiting queue.
- Inspect `run_pipeline.py` and its actual CLI parser. Use its supported command; do not guess flags or infer success from file existence alone.
- Launch with `subprocess.Popen`/equivalent using a fixed trusted `sys.executable`, a constructed argument list, `shell=False`, a known `cwd`, and a minimal environment allowlist. Implement and test Windows-specific process-group cancellation/termination; do not assume POSIX signals/process groups behave identically.
- Reject any user-controlled executable, command argument, script/module path or environment override.
- Use server-generated per-job directories and ensure all resolved paths remain under the configured data root.
- Set a configurable timeout. On timeout/cancel, terminate the process group, escalate to kill after a grace period, wait for exit, and mark the job failed/cancelled accurately.
- Capture stdout/stderr to per-job log files with limits and safe redaction. The UI receives only an allowlisted summary and optional safe tail of logs after redaction.
- On process restart, mark previously running jobs `failed_interrupted` or an equivalent terminal/attention state; never auto-rerun expensive inference without an explicit policy.
- Do not start a second GPU task while one is active. Check available disk and host resource status before launching.

## 7. File and media validation

- Initial accepted extensions: `.mp4`, `.mov`; expand only after tests prove decoder support. Use an allowlist, not a denylist.
- Validate actual container/media through a pinned/known media probe (e.g. `ffprobe`) and then ensure a short safe decode/metadata pass succeeds. MIME type and extension are hints, not proof.
- Upload cap is configurable; suggested starting default `FORGE_MAX_UPLOAD_BYTES = 2 * (1024 ** 3)` (2 GiB) only if sufficient disk and cleanup policy are in place. Reject if available disk would drop below a safety reserve (suggest 10–15% free space; confirm for the host).
- Suggested initial video duration cap: 900 seconds (15 minutes), configurable and changed only after memory/time testing.
- Validate dimensions/FPS/duration against configured bounds. Reject impossible metadata, empty streams and malformed media. Preserve a clear error for codecs that are not supported in the current environment.
- Use random/server-side filenames and store the original filename only as escaped display text.
- Never extract archives. This app accepts videos, not ZIPs or arbitrary model bundles.

## 8. Output validation

- Determine expected outputs by inspecting the pipeline's actual artifacts and viewer loader. Do not assume a GLB is generated if the current pipeline only writes OBJ/PLY and dependencies.
- Check file existence, nonzero size and successful parsing; geometry must contain finite vertex coordinates, valid indices/faces and positive finite bounds. Flag empty geometry, severe degeneracy, zero-sized axes and implausible bounds.
- Treat visual inspection of the actual output in the existing viewer as a required gate; automate browser smoke testing/screenshot where feasible and record the screenshot path locally.
- Preserve failed attempts and report what failed. Automatic retries must obey `MODEL_QA_AND_RETRY_POLICY.md`.

## 9. Static model serving

- Never mount the whole `data/` directory as static content.
- Serve only artifacts from a ready job through a route that checks the job state and resolves the requested artifact against an allowlisted manifest generated by the backend.
- Prevent path traversal, symlink escapes and arbitrary file reads. Reject `..`, absolute paths and unknown filenames at the application layer even if UUID routes are used.
- Prefer self-contained web model files when available to avoid path-based texture dependencies. Otherwise serve only the exact mesh/material/texture files listed in the job manifest.
- Add appropriate download content types and safe content disposition headers.

## 10. Config and operations

Use a host-only setting such as `FORGE_PIPELINE_ROOT` to point to the original pipeline checkout. Never commit a machine-specific Windows path. Use PowerShell/Windows-compatible start and stop scripts, and verify the actual Python/Node package-manager environment before generating scripts.

Use environment variables or a local config file ignored by Git. Suggested knobs:
- `FORGE_DATA_DIR`
- `FORGE_MAX_UPLOAD_BYTES`
- `FORGE_MAX_VIDEO_SECONDS`
- `FORGE_MAX_ACTIVE_JOBS=1`
- `FORGE_MAX_QUEUE_SIZE`
- `FORGE_JOB_TIMEOUT_SECONDS`
- `FORGE_RAW_RETENTION_HOURS`
- `FORGE_MIN_FREE_DISK_PERCENT`
- `FORGE_LOG_LEVEL`

Never expose these secrets/config values through the frontend. Do not add secrets unless needed. Add a `.env.example` with placeholders only, plus ignore rules for `.env*` except safe examples, private data, SQLite DB, logs, uploads and generated job outputs.

## 11. Dependency and supply-chain practices

- Inspect existing Python/Node versions and lockfiles before changing dependencies.
- Prefer existing dependencies; justify every new runtime dependency.
- Pin direct dependencies or use project lockfiles consistently; review transitive changes.
- Run `npm audit` (or the package manager's equivalent) and a Python dependency audit such as `pip-audit` where compatible. A tool finding must be triaged rather than automatically resolved by an uncontrolled major upgrade.
- Do not run install scripts or copy in new binaries without review. Use official sources for dependencies.
- Do not claim “secure” or “vulnerability-free” merely because scans show no findings.
