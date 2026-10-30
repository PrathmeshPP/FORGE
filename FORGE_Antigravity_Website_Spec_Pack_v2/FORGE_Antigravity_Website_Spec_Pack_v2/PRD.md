# Product Requirements Document (PRD)

**Product:** F.O.R.G.E. Private Indoor Video-to-3D Web Tool  
**Version:** 1.0 draft  
**Status:** Requirements for implementation; not proof of a working product.

## 1. Product summary

A straightforward private website for the project owner and explicitly authorized teammates. A user selects a video of an indoor space, uploads it, sees transparent job status, and opens the resulting 3D reconstruction in the existing F.O.R.G.E. Explorer viewer.

The website is an interface around the existing pipeline; it is not a new reconstruction algorithm. The first release prioritizes a trustworthy upload-to-output path over broad public access or extensive product features.

## 2. Problem

The current workflow involves local scripts, pipeline outputs and a separate viewer. It is inconvenient for a teammate to initiate a reconstruction and inspect the result through a consistent browser workflow. The goal is to make the experience understandable without requiring each teammate to learn the project's Python/GPU setup. Website implementation belongs in the separate `FORGE_website` repository; the original `FORGE` reconstruction repository is read-only to this workstream.

## 3. Goals

1. Allow an authorized teammate to upload an indoor video from a browser.
2. Validate the upload before running heavy reconstruction.
3. Execute the existing F.O.R.G.E. pipeline on the designated GPU workstation using a controlled backend worker.
4. Show honest state transitions and useful progress information.
5. Run objective output checks before marking a job ready.
6. Load the actual generated output in the existing viewer.
7. Share the app privately while the owner's Windows workstation is online, without paying for a cloud GPU. The current host has an NVIDIA GPU with 8 GB VRAM; actual model, driver and available VRAM must be detected.
8. Make it clear where files are stored, how long inputs are retained and who can access this private tool.
9. Keep the UI visually restrained, human-designed and focused on the core action.

## 4. Non-goals for the first version

- A public, multi-tenant SaaS or unrestricted public upload portal.
- Running reconstruction inside the browser.
- Running the full GPU pipeline on Vercel/serverless functions.
- Editing or owning the original `FORGE` reconstruction repo from this website workstream.
- Running actual indoor inference before a tested interface is handed off by the separate model workstream; use a mock worker until then.
- Guaranteeing that every low-powered CPU/GPU can reconstruct a video.
- Drone workflow redesign. Existing drone functionality should remain intact, but the immediate upload UI is for indoor smartphone/video captures.
- User billing, subscriptions, analytics, social sharing or a chatbot.
- Automatic cloud/S3 storage or upload.
- Fully automatic AR navigation, BIM conversion, POI search, obstacle-aware pathfinding or other competition functions unless separately scoped and approved.
- Claiming global metric accuracy or “perfect” reconstruction without evidence.

## 5. Users and use conditions

**Primary users:** Project owner and a small group of teammates/known collaborators.  
**Operating model:** The host workstation is switched on and the app/worker are started when the group needs them. The host may be offline or unavailable at other times.  
**Trust assumption:** Only named, authorized people are added to the Tailscale access policy. The central Windows workstation performs the compute; remote teammates use browser clients and do not contribute GPU/CPU to the host. All authorized users of v1 should be treated as able to see the team job list unless per-user application authentication/authorization is later implemented.

## 6. User journey

1. User visits the private website.
2. User sees a short explanation of accepted video formats, current file-size/duration limits, local-processing policy and expected workflow.
3. User selects an indoor video or drags it onto the upload area.
4. UI shows filename, size and basic metadata, then requires explicit confirmation to upload.
5. Backend streams the upload to a new server-generated job workspace and validates it.
6. A preflight stage checks decodability, video duration, dimensions, frame rate and basic suitability. It does not promise that reconstruction will succeed.
7. User clicks “Start reconstruction” if preflight passes or indicates a reviewable warning.
8. A single worker executes the existing pipeline with bounded resources.
9. The app shows truthful status, elapsed time and stage updates. It must not fake a percentage when the pipeline doesn't expose measured progress.
10. On output completion, the backend runs artifact and geometry checks.
11. The existing viewer loads the generated model. The user can orbit, pan, zoom, reset framing and switch representations only where supported by the current viewer.
12. User can download supported outputs and delete the job.
13. On completion/failure/restart, the documented cleanup policy is applied.

## 7. Functional requirements

### FR-01 — Private access
- The app is private by default and accessed via the host's Tailscale Serve URL.
- Only explicitly authorized team users/devices may reach it.
- The backend is not exposed directly to the public internet.

### FR-02 — Upload
- Accept only explicitly configured video containers (initial allowlist: MP4 and MOV; confirm decoder support in the current environment).
- Enforce a configurable upload cap (initial suggestion: 2 GiB maximum, subject to actual disk capacity); expose the actual configured value in the UI.
- Do not trust the extension or browser MIME type alone. Verify actual decodability with the configured media probe.
- Generate a UUID job ID and safe server-side filenames.
- Stream data to disk; do not load the whole video into memory.
- Reject zero-byte, oversized, unsupported, malformed or incomplete uploads with actionable messages.

### FR-03 — Preflight
- Derive duration, width, height, frame rate and file size using the available media tooling.
- Show a useful validation result: accepted, warning/review required, or rejected.
- Do not begin expensive inference until the upload passes hard validation.

### FR-04 — Job lifecycle
- Use explicit states: `receiving`, `validating`, `preflight`, `queued`, `running`, `validating_output`, `ready`, `failed`, `cancelling`, `cancelled`, `deleting`.
- Persist job metadata sufficiently to survive an ordinary backend restart.
- Keep v1 to one active reconstruction job. Configure a small bounded queue; never silently launch multiple memory-heavy GPU jobs.
- A user can cancel a queued job and request cancellation of a running job. Running cancellation must terminate the managed process tree safely and mark the state honestly.
- Provide failure summaries without exposing stack traces or local secrets.

### FR-05 — Reconstruction adapter
- Inspect the actual `run_pipeline.py` and its current argument parser/output semantics before implementing a wrapper.
- Use the existing pipeline's real command line or a safe importable function; do not invent flags.
- Invoke child processes with a fixed executable and argument list, `shell=False`, explicit working directory, bounded timeout and controlled environment.
- Do not edit core reconstruction algorithms as a side effect of building the web UI.

### FR-06 — Output validation and viewing
- Only mark a job `ready` after required outputs exist and parse/geometry checks pass.
- Use the current viewer-compatible formats. Do not introduce a new conversion chain unless the current outputs cannot be consumed.
- Load the actual output, not a sample model, for successful job results.
- If output validation fails, show the failure and preserve diagnostics; do not display a false success state.

### FR-07 — Job library
- List job ID, user-facing display name, submission date/time, status, duration where known, file size and result availability.
- Do not show absolute host filesystem paths, raw server logs, environment variables or traceback data.
- In v1, the Tailscale-approved group may see the shared job list; show this privacy model to users. Per-user private jobs require a later explicit auth/authorization design.

### FR-08 — Delete and retention
- Provide explicit “Delete job” with confirmation.
- Delete only files under that job's validated workspace.
- Default retention proposal: delete raw input videos after the job reaches a terminal state, unless the user chooses “keep input” before upload; if kept, expire raw inputs after 24 hours. Keep generated outputs until the user deletes them, subject to a configurable disk-cleanup safeguard. This policy must be implemented and documented before release.
- On startup, clean orphaned temporary uploads and apply retention to stale jobs.
- Never delete source video outside the app's designated workspace.

### FR-09 — Operational status
- Display whether the local worker is online and whether the GPU/CPU worker is busy.
- If the host stops or is asleep, visitors should see an availability error from the browser rather than a fake processing state.
- Show estimated duration only if based on documented prior measurements; otherwise show elapsed time and stage status.

### FR-10 — Privacy
- Default all video processing to the local workstation.
- Never send uploaded media to AWS/S3, model APIs, analytics or a third party without explicit opt-in, visible disclosure, and user approval.
- Do not add telemetry by default.

## 8. Non-functional requirements

- **Security:** Meet all mandatory controls in `SECURITY_BASELINE.md` and release gates in `TEST_PLAN.md`.
- **Reliability:** A failed or cancelled job must not corrupt previously generated outputs or silently leave a GPU subprocess running.
- **Resource safety:** Single active job; configurable upload, duration, time, disk and queue limits; explicit error on limit reached.
- **Usability:** Clear status language, accessible labels, keyboard support, sensible mobile/tablet layout for the interface (reconstruction itself runs on the host).
- **Performance:** Frontend shall not load full raw videos into browser memory. Progress updates should be lightweight. Model assets should load only when a user opens a ready result.
- **Maintainability:** Add an isolated backend adapter/API; preserve `src/`, `scripts/`, core pipeline and current viewer conventions.
- **Transparency:** No fabricated progress, completion, quality, metric accuracy or runtime claims.

## 9. Success criteria

A v1 pilot is successful when an authorized teammate, using a separate computer, can upload a valid test indoor video, see it validated and processed by the host worker, open the actual model in the viewer, and delete the job. The app must reject test files that violate configured limits and must remain inaccessible to a non-authorized external device.

The actual indoor reconstruction pipeline must be proven with at least one indoor sample. A UI mock-up or preloaded sample asset alone does not pass the end-to-end success criterion.

## 10. Open decisions (do not assume)

- Exact current supported input containers, video-duration cap and upload-size cap after checking host disk.
- Whether the pipeline currently handles smartphone indoor footage well enough; requires testing.
- Whether the current viewer can load the generated outputs directly or needs a controlled adapter.
- Exact retention default and whether source videos must be kept for reproducibility.
- Whether teammates can see each other's jobs (v1 proposes yes only within the explicitly authorized private team) or need per-user accounts.
- Whether any cloud functionality is ever wanted. Local-only is the default until explicitly changed.
