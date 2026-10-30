# Test Plan and Release Acceptance

## 1. Test layers

### Unit tests
- File-type allowlist and filename display sanitization.
- Streaming byte cap and upload-limit boundary checks.
- UUID and path-containment functions, including `..`, absolute paths and symlink attempts.
- State transition rules and duplicate start/cancel handling.
- Job storage/cleanup cannot touch files outside the job root.
- Geometry parser checks for empty, invalid, non-finite and degenerate mesh fixtures.
- Error response sanitization.

### Integration tests
- Upload a small known-good video and confirm the backend validates it.
- Upload corrupt and unsupported videos; ensure pipeline is never launched.
- Run a fixture or mocked pipeline adapter to verify state transitions, logs, timeout and cancellation without requiring the GPU.
- Run the real pipeline on the main GPU workstation with at least one indoor sample.
- Confirm the actual output manifest is served only after QA passes.
- Confirm the existing viewer opens a generated model and downloads only allowed assets.
- Verify retention cleanup deletes raw input according to policy and does not delete outputs unexpectedly.

### Frontend tests
- Upload idle/drag/drop/keyboard selection.
- Validating, queued, running, cancelling, ready, failed and worker-unavailable states.
- Upload progress is distinct from reconstruction progress.
- No fabricated percentage when backend reports only a stage.
- Responsive widths and keyboard-only navigation.
- Safe display of long filenames, Unicode and hostile strings.
- Viewer loading, reset camera, error state and empty result.

### Security tests
- Oversize upload, empty upload, wrong extension, forged MIME type and corrupt media.
- Filename path traversal and directory traversal in route parameters.
- Unknown job ID and artifact key.
- Attempt to retrieve SQLite DB, `.env`, logs, source videos, arbitrary output paths or other local files.
- Repeated uploads to check queue/resource limits.
- Process timeout, cancellation and host restart while a job is active.
- Verify backend binds only to loopback.
- Verify only approved tailnet users/devices can visit the app; verify non-tailnet clients cannot.
- Confirm Tailscale Funnel is off and no router port forwarding is required or configured for this service.
- Confirm no automatic AWS/S3/cloud upload or telemetry occurs.
- Scan source and built frontend for secrets; check dependency audits.
- Check logs and API errors contain no secrets, stack traces or full absolute server paths.

## 2. Commands to discover/use

Antigravity must first inspect existing package scripts and Python requirements. Use the repo's real scripts where possible. Candidate checks may include:
- Frontend: current package manager's lint, test, typecheck and production build scripts.
- Python: current pytest suite; static checks such as Ruff/Bandit only if installed or approved.
- Dependency audits: `npm audit`/equivalent and `pip-audit` when compatible.
- Browser smoke tests: Playwright only if already installed or a justified small dev dependency.

Do not claim commands passed if they were not run. Record dependency/tool absence separately.

## 3. Release blockers (all must pass before sharing)

- [ ] Host binds to `127.0.0.1` by default; no public backend listener.
- [ ] Tailscale Serve is configured for private access and access policy has been tested with an unauthorized device/account.
- [ ] Uploads are streamed, validated and bounded.
- [ ] File paths and subprocess arguments cannot be chosen by the user.
- [ ] One-job-at-a-time and queue/timeout/cancel controls work.
- [ ] Source uploads are stored outside the static web root and do not automatically leave the workstation.
- [ ] The app uses an allowlisted output manifest; arbitrary files cannot be downloaded.
- [ ] Delete/cleanup is restricted to the selected job directory.
- [ ] Browser error responses do not reveal stack traces/secrets/local paths.
- [ ] Existing reference outputs and project files remain unchanged by the build process.
- [ ] Existing frontend production build passes.
- [ ] At least one end-to-end indoor reconstruction has been tested on the actual workstation.
- [ ] Generated model passes structural checks and loads in the real viewer.
- [ ] Actual model has been visually reviewed; any limitations documented.
- [ ] UI has real screenshots for idle, error, processing and successful result states.
- [ ] README/runbook accurately explains how to start/stop and access the service.
- [ ] Known unresolved security findings and environment-specific limits are documented.

## 4. Acceptance decision

If any release blocker fails, Antigravity must say “NOT READY TO SHARE” and list evidence. It may offer a local-only development demo that has not passed release checks, but must label it clearly. No automated tool may mark all checks passed without evidence.
