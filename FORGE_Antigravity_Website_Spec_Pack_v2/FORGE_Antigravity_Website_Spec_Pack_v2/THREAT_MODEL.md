# Threat Model — F.O.R.G.E. Private Web App

## 1. Assets to protect

- Raw indoor videos, which may reveal building layouts, people, possessions or access controls.
- Generated 3D models, textures, point clouds and derived outputs.
- GPU workstation availability, VRAM, RAM, CPU time and disk space.
- Local model weights and reconstruction scripts.
- Tailscale/device access, repository credentials and any cloud credentials present on the host.
- Job metadata and logs.

## 2. Trust boundaries

1. Browser on a teammate's device -> Tailscale encrypted network -> Tailscale Serve -> local application.
2. Web API -> local filesystem and SQLite job database.
3. Web API/worker -> reconstruction subprocess, media parser and geometry libraries.
4. Generated files -> viewer loader and download endpoint.
5. Existing optional AWS preflight subsystem -> AWS services. This path is deliberately **out of scope for automatic use by the new web app** unless later authorized.

## 3. Threat actors / mistakes

- Someone outside the team who obtains a URL or scans the network.
- A tailnet member who should not have access because access policy is too broad.
- A trusted teammate accidentally uploads a malformed or too-large video.
- Malicious media crafted to exploit a media parser or consume CPU/RAM/disk.
- A filename or job parameter crafted to escape a directory or alter a subprocess command.
- A bug that exposes another job's files or logs.
- A dependency or model package with a known vulnerability or unsafe install behavior.
- A developer/agent accidentally commits secrets, enables public access, deletes reference outputs or uploads data to an unapproved service.

## 4. Principal abuse cases and controls

| Threat | Consequence | Required controls | Verification |
|---|---|---|---|
| Public exposure of API | Unknown people upload jobs or read models; workstation/resource abuse | Loopback bind; Tailscale Serve; no Funnel/port forwarding; explicit grants/ACL | Test from non-tailnet and unapproved device; inspect listening sockets and tunnel state |
| Overbroad tailnet policy | Unintended users access videos/models | Allow only named users/devices; document shared-job model | Check effective policy and test with an unapproved tailnet member |
| Oversize or corrupt video | Disk/memory exhaustion or parser failure | Streaming cap, actual media probing, duration cap, free-disk threshold, timeout | Oversize, malformed, zero-byte and misleading-MIME tests |
| Path traversal in filename/artifact path | Read/delete files outside job workspace | UUID job directory; generated file names; allowlisted artifact manifest; resolved-path containment | Test `../`, absolute paths, encoded traversal and unknown artifacts |
| Command injection | Arbitrary code execution | Fixed interpreter/script; argument list; `shell=False`; no user-controlled flags or paths | Static review of subprocess call; test suspicious filenames |
| Excess concurrent GPU jobs | Out-of-memory/crash/host hang | One active job; bounded queue; no arbitrary parallelism | Submit multiple jobs; verify second waits/rejects and cancellation works |
| Local file exposure | Source videos/weights/logs downloadable | No static mount of data root; route authorization; allowlisted artifact serving | Attempt direct URL guesses and path traversal |
| Secret exposure | API/cloud account compromise | No secrets in frontend or Git; redact logs; `.env` ignore; least-privilege environment | Secret scan, inspect bundles/logs and Git diff |
| Silent cloud upload | Confidential footage leaves workstation | Local-only default; no hidden external integration; explicit opt-in for any future cloud feature | Inspect code/config and test network activity with sample upload |
| Fake success / invalid model | User trusts bad geometry; misleading demo | Artifact checks, geometry checks, actual viewer load, screenshot/manual review, bounded retry | QA report with actual output and test evidence |
| Dependency compromise | Remote code execution or data leakage | Lockfiles, dependency audit, minimal packages, official sources | Audit command output and dependency-change review |
| Unsafe delete/cleanup | Reference data or other jobs are deleted | Delete only verified job root; never recursive-delete a client-supplied path | Unit tests with symlinks and path boundaries; test deleting one job doesn't affect others |

## 5. Risk ranking

### Critical before sharing
- API unexpectedly public.
- Arbitrary file read/delete or arbitrary subprocess execution.
- Upload processing that can exhaust or destabilize the host.
- Missing access restriction or public tunnel.

### High
- Unsafe video parser/dependency behaviour.
- Output file path manipulation.
- Secret leakage or unexpected cloud upload.
- No retention/cleanup policy.

### Medium
- Logs reveal local paths or project metadata.
- A teammate can delete another team's job (shared-job v1 trade-off).
- Dependency audit findings not reviewed.

### Lower, but still fix
- Decorative UI accessibility, small error copy inconsistencies, cosmetic display issues.

## 6. Accepted v1 limitations

- The host workstation must be online for processing and serving results.
- All authorized tailnet users have access to the shared job list in v1; this must be visible in the privacy notice. If individual job privacy is required, block release until per-user authentication and object-level authorization are added.
- This project pack is not a formal penetration test, code audit, or security certification.
- The model-quality gate reduces obvious failures but does not establish physical-world accuracy without suitable ground truth.

## 7. Revisit triggers

Re-review this model before any of the following: public deployment, public tunnel/Funnel, public unauthenticated uploads, multi-user private jobs, cloud GPU execution, S3/object-storage uploads, third-party APIs/analytics, new file types, AR/camera permissions, or remote URL ingestion.
