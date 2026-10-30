# Master Prompt for Antigravity — F.O.R.G.E. Private Indoor Reconstruction Website (Windows / 8 GB VRAM)

**Version:** 2.0  
**Updated:** 30 October 2026  
**Primary workspace:** `C:\Users\25beevdt025\Desktop\FORGE_website`  
**Website repository:** <https://github.com/PrathmeshPP/FORGE_website>  
**Existing reconstruction source repository (read-only for this task):** <https://github.com/PrathmeshPP/FORGE>

You are working on the **website project only**. The owner has split work across two conversations/workstreams: this Antigravity task builds the website in `FORGE_website`; a separate workstream is testing and improving the reconstruction/model pipeline in the original `FORGE` project. Keep those workspaces separate.

## 1. Read the project instructions first

Read these documents before changing code:
- `AGENTS.md`
- `WORKSPACE_PROFILE.md`
- `PRD.md`
- `TRD.md`
- `SECURITY_BASELINE.md`
- `THREAT_MODEL.md`
- `API_CONTRACT.md`
- `UI_UX_GUIDELINES.md`
- `MODEL_QA_AND_RETRY_POLICY.md`
- `PIPELINE_INTEGRATION_HANDOFF.md`
- `TEST_PLAN.md`
- `DEPLOYMENT_RUNBOOK.md`
- `ARCHITECTURE_DECISIONS.md`

If these files are nested inside the extracted specification-pack folder, locate them and report their current location. Do not move, overwrite or reorganize them until you have reported the current layout and proposed a safe structure.

## 2. Project mission and strict scope

Create a clean, restrained, private website for the owner and a small group of explicitly authorized teammates. Its single primary purpose is:

**Upload an indoor smartphone video → run the indoor reconstruction pipeline on the owner's main Windows workstation → validate the generated 3D assets → inspect the real result in an interactive web viewer.**

The main workstation is available and has an NVIDIA GPU with **8 GB VRAM**. This is the current hardware ceiling, not a promise that 8 GB is always free. Detect the exact GPU, driver, CUDA/PyTorch availability and free VRAM; do not guess them or change drivers/CUDA/PyTorch versions without approval.

The initial website is **indoor-video only**. The existing F.O.R.G.E. drone-video workflow must be preserved in its own project, but drone input is not part of this website's v1 upload UI or Namma Space workflow. Do not merge or redesign the drone and indoor model pipelines as part of this task.

This is not a generic AI landing page, chatbot, public SaaS product or new reconstruction algorithm. Reuse/adapt the existing viewer where justified, but make all website code changes inside the `FORGE_website` repository only.

## 3. Workspace and repository boundaries — mandatory

### Website workspace (the only write target)

- Local path: `C:\Users\25beevdt025\Desktop\FORGE_website`
- GitHub remote intended for this work: `https://github.com/PrathmeshPP/FORGE_website`
- The GitHub page appeared empty when checked on 30 October 2026. Recheck current remote state and the local folder before assuming Git is initialized or a remote is configured.
- The user says the specification ZIP has been extracted into this folder. Inspect the actual directory tree; do not assume the specs are at the root.

### Existing pipeline repository (read-only from this website task)

- URL: `https://github.com/PrathmeshPP/FORGE`
- It contains the existing reconstruction pipeline and the current Explorer viewer. Inspect it as a reference and locate any local checkout if one already exists.
- Do **not** edit, reformat, delete, commit, push or change files in this repository during the website task. A separate workstream owns reconstruction/model changes.
- Do not assume the local pipeline checkout path. If useful, check the likely sibling path `C:\Users\25beevdt025\Desktop\FORGE`, but verify it before use.
- The website must refer to the pipeline through a trusted local configuration such as `FORGE_PIPELINE_ROOT`, not a hard-coded machine path committed to Git.
- Do not clone, move or modify the pipeline repository automatically. If a local checkout is missing, report that and propose the safest next step.

### Cross-workstream coordination

A separate chat/workstream is testing indoor reconstruction. Do not race it by running expensive model inference or editing the original pipeline. For website development, create a **mock/fixture pipeline adapter** first. The mock must be visibly marked as sample/test mode and must never present a sample model as the uploaded video's result.

Do not connect the real pipeline until the owner supplies or approves the integration contract recorded in `PIPELINE_INTEGRATION_HANDOFF.md`: exact tested command/arguments, input requirements, output manifest, model/runtime dependencies, commit/version, GPU memory notes and measured run result. If the other workstream has not completed this contract, continue frontend/backend development against the mock adapter and clearly mark real integration as blocked/pending.

## 4. Host and runtime assumptions

- Host OS: Windows, inferred from the supplied workspace path; detect the exact Windows version/build rather than assuming it.
- Host GPU: NVIDIA GPU with 8 GB VRAM; exact model unknown until detected.
- Use Windows-native **PowerShell** commands and scripts by default. Do not assume Bash, Linux, WSL or macOS.
- Prefer Windows-compatible `.ps1`/`.bat` launch scripts after testing them. Avoid requiring administrator privileges.
- Use `py -3`, the project's actual Python launcher/interpreter, the existing Node/package manager and existing lockfiles. Do not upgrade major toolchains casually.
- Do not change Windows execution policy globally. If script activation is blocked, use the venv interpreter directly or provide a session-scoped, user-approved alternative.
- Detect GPU and software versions with safe, non-secret diagnostics such as `nvidia-smi`, Python/PyTorch checks, `node --version`, package-manager version and `git --version`. Do not dump all environment variables or secrets.
- Keep GPU concurrency at one reconstruction job by default. Check free VRAM, system RAM and disk before starting. No automatic concurrent model runs. CPU fallback may be offered only if the actual pipeline supports it and has been tested; do not promise that any CPU can run VGGT or other model stages.
- The host workstation performs reconstruction. Teammates' computers use a browser for upload, job status and viewing; they do not automatically provide compute to the host.

## 5. First response: audit only, do not code

Before editing any file, report:

1. Confirmed current directory and actual folder tree, including where the extracted specification files are located.
2. Whether the website folder is a Git repository; current branch; `git status --short --branch`; configured remotes; whether the remote URL matches `FORGE_website`; and whether the online repository is still empty.
3. Any existing source files, ignored files, untracked files or user changes. Do not remove or overwrite them.
4. Whether the original `FORGE` repository is present locally, its path/branch/status, and whether it is safe to inspect without modifying it. Do not alter it.
5. Actual Windows version, Python version/path, Node and package-manager versions, Git version, available disk space, NVIDIA GPU model, VRAM, driver and PyTorch/CUDA status if installed. Avoid exposing credentials or unrelated environment values.
6. Existing website/frontend entry points, package manager, scripts, viewer components, supported model formats and available tests in the website workspace (if any).
7. From the read-only original repo, identify the current `run_pipeline.py` invocation contract, input assumptions, output contract and viewer reuse options—but do not run a heavy reconstruction and do not edit the source repo.
8. Current AWS/S3 and external-service paths in the original source; explain how the new website upload flow will avoid silently using them.
9. Missing tools, model weights, local source checkout, Windows compatibility concerns, 8 GB VRAM risks and any conflict between these specifications and actual code.
10. A prioritized implementation plan split into safe phases, identifying what can be built now with a mock worker and what is blocked on the separate model workstream.

Do not initialize Git, change remotes, install dependencies, edit code, run reconstruction, commit, push, deploy or change Tailscale settings during this first audit. Wait for the owner's approval of the plan.

## 6. Implementation phases — approval required between phases

### Phase 1 — Website foundation and UI shell

- Work only in `FORGE_website`.
- Build the focused indoor-video tool UI in the existing framework if one exists; otherwise propose the smallest sensible stack before introducing dependencies.
- Design upload, video preflight, queued/processing, job history, result, validation failure and system-unavailable states.
- Use a clearly labeled sample asset for visual development only.
- The actual generated model must be the primary content of the result view once available.
- Use a restrained, professional engineering-tool visual language. Do not use generic AI gradients, chatbots, fake dashboards, fake counters or made-up progress.
- Run available type checks, lint, unit tests and build. Capture screenshots where practical.

### Phase 2 — Windows-compatible local backend foundation

- Prefer FastAPI if compatible with the actual Python environment; otherwise justify a maintained alternative.
- Frontend and API should share one origin in the production-like local setup.
- Bind only to `127.0.0.1` by default.
- Implement safe streaming upload, allowlist validation, generated IDs/filenames, request/body limits, video probing, free-disk checks, per-job isolation, status persistence, cleanup and safe error messages.
- Store private uploads, intermediates, logs and generated models outside the Git repository and outside cloud-synced Desktop/OneDrive folders. Select/confirm a local data root after checking available drives and free space. Never silently choose a synced folder.
- Do not mount the jobs/data directory as a static directory.
- Do not connect AWS/S3 or any external upload service.

### Phase 3 — Mock worker and pipeline adapter boundary

- Implement a deterministic mock worker first to test status transitions, upload, cancellation, errors and result rendering without using the GPU.
- The real pipeline adapter must use a fixed trusted Python interpreter, fixed script path under the configured `FORGE_PIPELINE_ROOT`, explicit allowlisted arguments, known working directory and `shell=False`.
- Never accept paths, commands, executable names, script paths, model paths, output paths or environment overrides from the browser.
- Use Windows-safe process-group handling for timeout/cancellation; verify that child processes are actually stopped.
- Single active GPU job; bounded queue; timeout; disk and VRAM safety gate. Do not automatically retry an out-of-memory job with arbitrary settings.
- Do not edit the original pipeline repository. The integration contract comes from `PIPELINE_INTEGRATION_HANDOFF.md` after the separate model workstream validates it.

### Phase 4 — Actual model QA and viewer integration

- Follow `MODEL_QA_AND_RETRY_POLICY.md`.
- Validate the output manifest, file integrity, geometry and viewer compatibility.
- Load the actual generated model in the real viewer and inspect the visible result. Record screenshots/evidence and measured run time.
- Retry only within the default maximum of three full reconstruction attempts, and only when an evidence-based correction exists. Do not rerun expensive inference to fix a viewer-only packaging problem.
- Do not mark a job `ready` until required structural and viewer checks pass. Visual review must clearly state remaining limitations.
- Do not claim survey accuracy, metric scale, watertightness, mathematical perfection or “hours reduced to minutes” without the appropriate measurements/ground truth/benchmark.

### Phase 5 — Security hardening and tests

- Implement `SECURITY_BASELINE.md` and `THREAT_MODEL.md`.
- Test upload abuse, path traversal, malicious filenames, MIME spoofing, corrupt video, oversized input, disk exhaustion, cancellation, concurrent job attempts, artifact authorization, secret leakage and cleanup.
- Verify the application listens only on loopback. Private teammate access is through Tailscale Serve with explicit least-privilege grants/ACLs for named collaborators.
- Do not use Tailscale Funnel, public tunnel, public API, router port forwarding or a public reverse proxy.
- Verify no silent cloud/S3/telemetry transfer.
- Run compatible dependency/security checks and triage findings without uncontrolled major upgrades.

### Phase 6 — Local and private team operation

- Create tested Windows start/stop/status scripts or documented PowerShell commands.
- The owner starts the website/backend/worker and Tailscale Serve when the team needs it. The machine must stay awake, powered and online while teammates use it.
- Teammates open the private HTTPS Serve URL in a browser and use the website; they do not need the reconstruction model installed locally, but they need authorized Tailscale access.
- Verify access from one authorized teammate device and one unauthorized device/account.
- Complete release checklist. If any blocking control fails, say **NOT READY TO SHARE**.

## 7. Mandatory 3D quality behavior

Never stop just because the first run exits successfully or emits a file. For every real model:
- parse the output;
- check finite coordinates, valid faces/indices, non-empty geometry, finite bounds and configured quality diagnostics;
- load the actual output in the actual viewer;
- inspect screenshots/visual output when available;
- diagnose failures from evidence and apply targeted corrections;
- retry within the bounded policy (default max three full attempts per job), preserving attempts, parameters, logs and artifacts;
- stop when acceptance gates pass or the retry limit is reached;
- report remaining defects and uncertainty honestly.

Never fabricate a QA score, claim perfection, or label a model ready when required checks fail. A valid mesh is not proof of physical-world fidelity.

## 8. Security hard rules

- Treat uploads and model files as untrusted input.
- Use allowlists, streaming byte limits, media-parser validation, server-generated filenames, isolated job directories and cleanup.
- Never interpolate user input into a shell command. Use `shell=False` and fixed arguments.
- No arbitrary file browsing, arbitrary paths, remote URL fetching, archive extraction or browser-controlled command execution.
- Do not return absolute paths, tracebacks, secrets or environment values to the browser.
- Never expose raw uploads, logs, intermediate files or unrestricted output roots through static file hosting.
- Private access only through Tailscale Serve + narrow explicit access policy; no Funnel/public listener.
- No cloud upload, analytics or third-party model/API calls unless the owner separately approves a documented data-flow change.
- Never commit `.env`, API keys, local video, private models, logs, SQLite DB or job outputs.
- Do not install system drivers, modify Windows security settings, commit, push, deploy or change network exposure without explicit approval.

## 9. Design and communication rules

The product should look like a thoughtfully designed engineering tool made for real users—not an AI-generated landing page. Use clear hierarchy, natural concise copy, restrained colors, precise spacing, real product states, accessible controls and the reconstructed model as the focal point. Avoid gratuitous AI references and generic templates.

The F.O.R.G.E. long-term differentiator is intended rapid video-to-digital-twin reconstruction with lower computational overhead. In this website, show measured processing time and hardware only when actually measured. Never invent speed comparisons or advertise a guaranteed number of minutes.

## 10. Reporting discipline

At each phase report:
- changed files and the reason for each;
- exact commands run (PowerShell syntax where applicable);
- passed, failed, not run and blocked tests separately;
- GPU/VRAM use and measured processing time only when observed;
- remaining risks and the next smallest step.

Preserve a clean diff. Do not commit, push, deploy, change the original pipeline repository or expose the backend publicly unless the owner explicitly approves that exact action. Do not say “done” if only the UI builds and the real indoor pipeline has not been tested.
