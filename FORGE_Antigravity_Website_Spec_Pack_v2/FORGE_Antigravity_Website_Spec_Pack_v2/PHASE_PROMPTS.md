# Minimal Follow-up Prompts for Antigravity — Windows Website Workstream

Use `AGENTS.md`, `WORKSPACE_PROFILE.md` and the full specification pack as persistent context. Approve one phase at a time. All website edits stay in `FORGE_website`; the original `FORGE` repo is read-only.

## Prompt 0 — Audit only

> Read the complete specification pack. Do not edit files. Audit the actual Windows checkout, folder tree, Git state/remotes, online website repo, read-only original pipeline repo, GPU model and 8 GB VRAM/free VRAM, RAM/disk, Python/Node versions, package scripts, viewer formats, AWS/S3 paths and security posture. Identify any missing indoor pipeline handoff. Return a prioritized phase plan and exact PowerShell commands. No installs, Git initialization, code edits, reconstruction, commits, pushes or deployment.

## Prompt 1 — Frontend shell

> Implement the UI shell in `FORGE_website` only. Reuse/adapt viewer components only inside this repo after inspection. Build indoor-video upload, preflight, processing, jobs, result and error states. Use a clearly labeled sample/mock asset. Do not connect the real pipeline, modify the original `FORGE` repo or deploy. Run frontend checks and inspect screenshots.

## Prompt 2 — Secure Windows backend foundation

> Implement the local backend according to `TRD.md`, `API_CONTRACT.md`, `SECURITY_BASELINE.md` and `THREAT_MODEL.md`. Use Windows-compatible paths/process/file handling. Add streaming upload, media validation, per-job isolation, persisted state, limits, cleanup and safe errors. Bind to loopback. Store job data outside the repo and outside synchronized folders. Do not connect AWS/S3 or the real pipeline yet. Add and run upload/path/resource security tests.

## Prompt 3 — Mock worker and adapter boundary

> Implement and test a deterministic mock worker. Then prepare the adapter interface, but do not invoke the real pipeline until `PIPELINE_INTEGRATION_HANDOFF.md` is completed and owner-approved. Verify Windows timeout/cancellation actually terminates child processes. Keep one active job and a bounded queue. Do not modify the original pipeline repository.

## Prompt 4 — Real pipeline integration

> Only after the owner approves the completed indoor pipeline handoff, connect the exact trusted command and output contract using fixed arguments, `shell=False`, and `FORGE_PIPELINE_ROOT`. Do not accept commands or paths from the browser. Do not edit the original source repo. Run a real indoor sample only when the owner confirms the separate model workstream is not using the GPU. Record GPU/VRAM, runtime, outputs and errors honestly.

## Prompt 5 — Output QA and viewer

> Follow `MODEL_QA_AND_RETRY_POLICY.md`. Validate artifacts and geometry, load the actual uploaded-video result in the actual viewer, inspect visual quality and capture evidence. If output fails, diagnose and make evidence-led targeted corrections. Retry at most three full attempts per job. Preserve attempts/logs. Never claim perfection or return a failed model as ready.

## Prompt 6 — Security hardening

> Run applicable tests from `TEST_PLAN.md`. Verify loopback bind, private Tailscale Serve, explicit collaborator access, no Funnel/public exposure, no AWS/S3 transfer, safe path/process handling, upload limits, deletion/retention and no secret leakage. Run compatible dependency checks and triage findings. Do not deploy or push.

## Prompt 7 — Release readiness

> Test the full UI and local workflow on the Windows host. Verify real indoor inference only if the handoff is approved and GPU test is authorized. Test one authorized and one unauthorized teammate device. Write tested PowerShell start/stop/access instructions. If anything material remains untested, state it. If a release blocker remains, say **NOT READY TO SHARE**. Do not commit/push/deploy.

## Emergency prompt if the agent overreaches

> Stop. Do not add features or rewrite the reconstruction pipeline. Show the current diff and compare every change with `AGENTS.md` and the approved plan. Do not delete user data or revert unrelated changes. Do not modify the original `FORGE` repository. Ask for approval before destructive, external, Git or network-exposure actions.
