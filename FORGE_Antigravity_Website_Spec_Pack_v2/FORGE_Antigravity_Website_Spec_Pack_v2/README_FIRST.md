# F.O.R.G.E. Website — Antigravity Project Pack (Windows / 8 GB VRAM)

**Purpose:** A private, practical website for trusted teammates to submit an indoor video to the existing F.O.R.G.E. reconstruction pipeline and inspect the resulting 3D digital twin.

**Status:** Specification only. These files do not mean the website, upload flow, security controls, or indoor reconstruction have been implemented or verified.

## Read these files in order

1. `PROMPT_QUICK_START.md` — shortest safe way to start.
2. `AGENTS.md` — persistent rules Antigravity must obey on every task.
3. `WORKSPACE_PROFILE.md` — Windows workspace, repository separation and known hardware facts.
4. `MASTER_ANTIGRAVITY_PROMPT.md` — use this as the first project prompt.
5. `PRD.md` — product scope and user needs.
6. `TRD.md` — proposed technical design; reconcile it with the actual checkout before coding.
7. `SECURITY_BASELINE.md` and `THREAT_MODEL.md` — mandatory safety requirements.
8. `API_CONTRACT.md` — proposed browser/backend contract.
9. `UI_UX_GUIDELINES.md` — appearance and interaction requirements.
10. `MODEL_QA_AND_RETRY_POLICY.md` — objective 3D output verification and retry rules.
11. `PIPELINE_INTEGRATION_HANDOFF.md` — required interface from the separate model workstream.
12. `TEST_PLAN.md` — testing and release acceptance gates.
13. `DEPLOYMENT_RUNBOOK.md` — local Windows use and private sharing through Tailscale.
14. `ARCHITECTURE_DECISIONS.md` — accepted decisions and unresolved items.
15. `RESEARCH_REFERENCES.md` — authoritative security/engineering sources reviewed for this pack.
16. `PHASE_PROMPTS.md` — short follow-up prompts for implementing one tested phase at a time.

## The intended first release

- Private use by the project owner and explicitly authorized teammates.
- Website code lives in `C:\Users\25beevdt025\Desktop\FORGE_website` / `https://github.com/PrathmeshPP/FORGE_website`.
- The original `https://github.com/PrathmeshPP/FORGE` repository remains the reconstruction source and is read-only from the website workstream.
- One website, hosted by the main Windows workstation with an NVIDIA GPU with 8 GB VRAM that actually runs reconstruction.
- The site is reachable remotely through Tailscale Serve, not a public backend or an open router port.
- Indoor-video-to-3D is the immediate workflow. Keep F.O.R.G.E.'s drone workflow intact; do not mix drone reconstruction into the Namma Space indoor UI unless explicitly requested later.
- Existing F.O.R.G.E. reconstruction and viewer code are to be inspected and reused/adapted only inside the website repository where justified; do not edit the original pipeline repository.
- Use a clearly labeled mock worker until the separate model workstream supplies a tested indoor pipeline handoff.
- The existing AWS preflight/S3 workflow is **not** to receive uploads from this web app automatically. Local processing is the default. Any cloud transfer must be a separate, explicit opt-in design decision.

## Important limits

- A website cannot make every computer capable of running the same reconstruction. In this deployment the host workstation does the GPU/CPU work; the teammate's browser only uploads, monitors and views results.
- “Safe” is not a binary property. The pack defines baseline controls and release tests; it is not a penetration-test report or a security certification.
- A mathematically valid mesh is not automatically a faithful digital twin. Both automated geometry checks and a real rendered-view review are required. No software agent may claim “perfect” reconstruction without measurable evidence.

## How to use with Antigravity

1. Open Antigravity at `C:\Users\25beevdt025\Desktop\FORGE_website` and locate the extracted specification folder. Do not reorganize it until the actual folder tree has been inspected.
2. Keep `AGENTS.md` visible as persistent project instructions. Merge with existing instructions; do not overwrite unrelated rules.
3. Paste `MASTER_ANTIGRAVITY_PROMPT.md` as the first prompt; it requires an audit only before code changes.
4. Keep all website code changes inside `FORGE_website`. The original `FORGE` repository is read-only for this workstream.
5. Use the mock worker until `PIPELINE_INTEGRATION_HANDOFF.md` is completed by the separate model workstream.
6. Approve a phase at a time. Inspect diffs and exact PowerShell test results. Never allow an agent to push to GitHub without explicit authorization.

## Repository basis

The website repository is <https://github.com/PrathmeshPP/FORGE_website>; it appeared empty on GitHub when checked on 30 October 2026, so Antigravity must verify the current remote and local Git state. The existing source repository at <https://github.com/PrathmeshPP/FORGE> documents a root `run_pipeline.py`, `src/` and `scripts/` reconstruction code, a React/TypeScript/Vite/Three.js viewer under `viewer/`, and AWS video-preflight. It is read-only for this website task. Treat checked-out code—not README claims—as truth. See `WORKSPACE_PROFILE.md` and `PIPELINE_INTEGRATION_HANDOFF.md`.
