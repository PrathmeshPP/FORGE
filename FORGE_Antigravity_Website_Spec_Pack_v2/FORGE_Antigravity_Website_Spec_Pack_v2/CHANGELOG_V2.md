# Specification Pack v2 — Changes

Updated 30 October 2026 based on the owner's new workstation and repository setup.

## Key changes

1. Website work now targets `C:\Users\25beevdt025\Desktop\FORGE_website` and `https://github.com/PrathmeshPP/FORGE_website`.
2. The original `https://github.com/PrathmeshPP/FORGE` remains the reconstruction source but is read-only for the website workstream.
3. The host is a Windows PC with an NVIDIA GPU with 8 GB VRAM. Exact GPU model, driver, free VRAM, CUDA/PyTorch state, system RAM and disk must be detected locally.
4. Website development and reconstruction/model testing are separate workstreams. Website development must use a clearly labeled mock worker until the separate model workstream produces an owner-approved indoor pipeline handoff.
5. Added `WORKSPACE_PROFILE.md` for the supplied path, repositories, Windows diagnostics and unknowns.
6. Added `PIPELINE_INTEGRATION_HANDOFF.md` to pass the real indoor CLI/output contract from the model workstream to the website workstream.
7. Updated `AGENTS.md`, `MASTER_ANTIGRAVITY_PROMPT.md`, PRD, TRD, architecture decisions, deployment runbook, quick start and phase prompts to enforce the repository boundary and Windows-safe workflow.
8. Explicitly prohibit modifying the original pipeline repository from the website task, running overlapping GPU tests, making network exposure changes, and pushing commits without approval.

## Important: not yet verified

This pack does not verify that indoor reconstruction works on the 8 GB VRAM workstation. It does not implement the website, run the pipeline, test the GPU, configure Tailscale, initialize Git, commit or push files, or perform a penetration test. Antigravity must perform the audit first and wait for approval.
