# Antigravity Quick Start — Windows Website Workstream

1. Open Antigravity at `C:\Users\25beevdt025\Desktop\FORGE_website` (verify the folder first).
2. Locate the extracted spec pack. Do not move or overwrite existing files before reporting the tree.
3. Make `AGENTS.md` available as persistent project instructions. Merge with any existing instruction file; do not overwrite existing rules blindly.
4. Read `WORKSPACE_PROFILE.md`, `PIPELINE_INTEGRATION_HANDOFF.md` and the rest of the specification pack.
5. Paste `MASTER_ANTIGRAVITY_PROMPT.md` as the first prompt. It requires an audit only and no code edits until you approve the plan.
6. Keep all website changes in `FORGE_website`. The original `FORGE` pipeline repository is read-only for this workstream.
7. Use a mock worker until the separate model workstream provides the tested indoor pipeline handoff.
8. Use PowerShell/Windows-compatible commands. Detect the 8 GB VRAM GPU model, free VRAM, driver, Python/Node environment and free disk; do not guess or upgrade.
9. Do not commit, push, deploy, expose the backend publicly or change network settings without explicit approval.

**Shortest safe initial prompt:**

> Read `AGENTS.md` and every specification in the pack, especially `WORKSPACE_PROFILE.md` and `PIPELINE_INTEGRATION_HANDOFF.md`. Do not code yet. Audit the actual Windows folder, local Git state/remotes, online `FORGE_website` remote, original `FORGE` repo (read-only), GPU/8 GB VRAM status, available RAM/disk, Python/Node versions, frontend structure, pipeline CLI/output contract, viewer formats, AWS/S3 paths and security controls. Report what is known, unknown, blocked, and a phased plan. Do not install dependencies, edit files, initialize/reset Git, run reconstruction, commit, push, deploy or change Tailscale settings until I approve the plan.
