# Architecture Decision Records (ADR)

**Updated 30 October 2026:** Website code is developed in `FORGE_website`; the existing `FORGE` source repository is read-only to this workstream. The host is a Windows PC with 8 GB NVIDIA VRAM (exact GPU model to be detected).

## ADR-001: Host reconstruction on the owner's Windows GPU workstation

**Status:** Accepted for v1.  
**Decision:** The workstation runs the website backend and reconstruction worker. Teammates use browser UI; compute stays on the host.  
**Reason:** No cloud GPU budget is assumed; F.O.R.G.E. already has local GPU pipeline components.  
**Trade-offs:** Host must stay powered, awake and online; available GPU/CPU/RAM/disk limit throughput. The 8 GB VRAM ceiling may constrain model/video settings; detect free VRAM and keep one active job by default.

## ADR-002: Private remote access via Tailscale Serve

**Status:** Accepted for v1.  
**Decision:** Bind app to loopback and share through Tailscale Serve, with explicit least-privilege grants/ACLs for named team members.  
**Reason:** Small known group, no need for a public portal, avoids opening router ports, supports HTTPS over the tailnet.  
**Trade-offs:** Teammates need Tailscale; tailnet policy must be correct. Tailscale Serve is private to the tailnet; Funnel is not to be used.

## ADR-003: Do not host the reconstruction API in Vercel Functions

**Status:** Accepted for v1.  
**Decision:** Do not run video upload/reconstruction through a Vercel serverless function.  
**Reason:** Vercel currently documents 4.5 MB function request/response body limits and bounded execution duration. Video files/intermediate artifacts may be large and reconstruction is long-running.  
**Trade-offs:** The app is not a public always-on service. A static hosted frontend could be considered later, but adds a separate API security boundary and does not eliminate the need for the local worker.

## ADR-004: Same-origin frontend and API

**Status:** Accepted for v1.  
**Decision:** In the production-like local deployment, serve the built React app and `/api/v1` from a single origin.  
**Reason:** Less CORS complexity and fewer cross-origin security mistakes.  
**Trade-offs:** Add a small serving integration around the existing viewer; preserve the existing Vite development workflow.

## ADR-005: Keep the existing reconstruction pipeline in a separate repository

**Status:** Accepted.  
**Decision:** Website code is implemented in `FORGE_website`. The original `FORGE` repository is read-only from this workstream. The website may inspect it and later invoke a local pipeline checkout through a controlled adapter, but only after the separate model workstream supplies and the owner approves a tested indoor invocation/output contract in `PIPELINE_INTEGRATION_HANDOFF.md`. Until then, use a deterministic mock worker.  
**Reason:** Avoid cross-chat conflicts, regressions and unverified CLI assumptions.  
**Trade-offs:** Real website reconstruction integration remains pending until the handoff is validated.

## ADR-006: Local-only data flow by default

**Status:** Accepted.  
**Decision:** No automatic upload of videos/results to AWS S3, cloud GPU, model API, analytics or third party. Existing AWS preflight is not automatically wired into the new upload flow.  
**Reason:** Privacy, safety, cost and user budget.  
**Trade-offs:** Local workstation is the only compute and storage service in v1.

## ADR-007: Model QA must pass before ready state

**Status:** Accepted.  
**Decision:** Use artifact, geometry, viewer-load and visual review gates. Use bounded, evidence-led retries up to three attempts by default.  
**Reason:** Avoid returning the first failed/unviewable model while preventing infinite expensive loops.  
**Trade-offs:** Some jobs end in an honest failed/review-required state; a software agent cannot guarantee perfection from computer vision alone.

## ADR-008: Shared job list in v1

**Status:** Provisional acceptance, user should review.  
**Decision:** Every user granted access through the private Tailscale policy can see all jobs in v1.  
**Reason:** Small trusted collaborators; avoids rushing a custom user management system.  
**Risk/limitation:** The app is not suitable for confidential videos unless all authorized users are allowed to see them. If that is unacceptable, implement application user identities and per-job authorization before use.

## ADR-009: Retention policy

**Status:** Proposed, must be confirmed during implementation.  
**Decision:** Raw uploads default to delete after terminal completion; optional keep-input retention expires after 24 hours. Generated outputs remain until user deletes them, but a disk safety monitor may stop accepting jobs before storage is exhausted.  
**Reason:** Minimize retention of potentially sensitive indoor footage while preserving useful outputs.  
**Required:** Expose policy to users and test cleanup boundaries before release.
