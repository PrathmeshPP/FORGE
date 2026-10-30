# Persistent Agent Instructions — F.O.R.G.E. Website Workstream

These instructions apply to every Antigravity task in this website repository. If a request conflicts with them or weakens security, stop and explain the conflict before making changes.

## 1. Repository boundaries

- The only write target is the website workspace: `C:\Users\25beevdt025\Desktop\FORGE_website` (verify actual path first).
- Website remote: `https://github.com/PrathmeshPP/FORGE_website`.
- The existing pipeline source is `https://github.com/PrathmeshPP/FORGE`; it is read-only from this workstream. Do not edit, reformat, delete, commit or push files there.
- A separate chat/workstream owns indoor model experiments and pipeline changes. Do not conflict with it or run heavy GPU reconstruction unless the owner explicitly coordinates/approves a test.
- Inspect the local website Git state and remotes. The online website repository appeared empty on 30 October 2026; verify current state. Do not initialize Git, reset, change remotes, commit or push without explicit approval.
- Never overwrite extracted specification files or unrelated user changes.

## 2. Windows and hardware

- The host is a Windows PC with an NVIDIA GPU with 8 GB VRAM; exact model/software details are unknown until detected.
- Prefer PowerShell and Windows-compatible commands/scripts. Do not assume Bash, Linux, WSL or macOS.
- Detect GPU, driver, free VRAM, system RAM, Python, Node and disk space before GPU or dependency work. Do not change GPU drivers, CUDA, PyTorch or global Windows settings without approval.
- One active GPU job by default. Check free VRAM and disk before launching; no automatic concurrency. CPU fallback is allowed only if the actual pipeline supports it and it is tested.
- The host PC runs the website/backend/worker and performs reconstruction. Teammates use browsers; they do not provide compute to the host.

## 3. Protect the original pipeline

- Inspect original pipeline CLI, formats and viewer as read-only references.
- Use `PIPELINE_INTEGRATION_HANDOFF.md` as the sole source of truth for real pipeline invocation and output mapping.
- Until the handoff is completed and owner-approved, build against a deterministic mock worker. Clearly mark sample/test output; never claim it is the uploaded video's result.
- Never hard-code local paths in committed code. Use a trusted host-only setting such as `FORGE_PIPELINE_ROOT`.
- Any required algorithm or pipeline change must be proposed to the separate reconstruction workstream, not made here.

## 4. Safe, staged development

- First report current folder tree, Git state, package scripts, relevant code and hardware diagnostics; do not code before the owner approves the plan.
- Implement one coherent phase at a time. Run tests/type checks/lint/build after changes and report actual results.
- Never fabricate tests, measurements, screenshots, model output or visual review.
- Preserve a clean diff and stop before destructive actions.

## 5. Security baseline

- Follow `SECURITY_BASELINE.md` and `THREAT_MODEL.md`.
- Bind app/API to `127.0.0.1`; private remote access through Tailscale Serve with explicit least-privilege grants/ACLs for named collaborators.
- Never enable Tailscale Funnel, public listeners, router port forwarding or public tunnels in v1.
- Do not use Vercel/serverless functions for video ingestion or GPU reconstruction.
- Treat videos and model outputs as untrusted. Stream uploads with limits, allowlist formats, validate actual media, generate server-side filenames, isolate jobs and check free disk.
- Invoke processes with fixed trusted executables/arguments and `shell=False`; browser input must never control paths or commands.
- No silent AWS/S3, cloud, telemetry, analytics or third-party transfer.
- Store data outside the Git repo and outside synchronized cloud folders. Do not expose uploads/logs/intermediates through static paths.
- No secrets, private videos, models, logs, databases or generated job outputs in Git.

## 6. Reconstruction QA

- Follow `MODEL_QA_AND_RETRY_POLICY.md`: validate file integrity and geometry, load the real output in the actual viewer, visually inspect and preserve evidence.
- Default maximum: three full reconstruction attempts per job, with evidence-based targeted changes only.
- Do not rerun expensive inference for viewer-only defects; do not retry identical failures indefinitely.
- Never say “perfect”, “mathematically accurate” or “survey accurate” without the required evidence and ground truth.

## 7. UX and reporting

- Build a restrained, human-designed engineering tool, not a generic AI landing page. No chatbot, neon AI visuals, fake progress, fake results or unsupported claims.
- The website v1 is for indoor video only. Preserve the separate drone workflow; do not add drone input to this website without approval.
- The long-term F.O.R.G.E. differentiator is faster video-to-digital-twin creation with lower compute use. Show times/hardware only when measured.
- Every report must separate passed, failed, not run and blocked checks; list changed files and exact PowerShell commands; identify anything requiring the main GPU PC or owner approval.
- Do not commit, push, deploy or change network exposure unless explicitly authorized.
