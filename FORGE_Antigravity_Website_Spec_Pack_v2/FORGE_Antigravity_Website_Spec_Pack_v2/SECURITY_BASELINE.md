# Security Baseline and Secure Build Checklist

**Classification:** Mandatory for v1. A convenience prototype is not permission to skip safety controls.  
**Scope:** Private site used by the owner and named teammates to upload indoor videos to a GPU workstation.

## 1. Security posture

- Private-by-default. No public upload endpoint or public API in v1.
- Backend and frontend are served from the GPU workstation on one origin.
- Application binds to `127.0.0.1`; remote access is through Tailscale Serve.
- Configure explicit Tailscale grants/ACLs for named users/devices. Do not assume the default tailnet policy is restrictive: Tailscale documents that absent access-control policy can permit broad access.
- Do not enable Tailscale Funnel, router port forwarding, public reverse proxy or a public tunnel in v1.
- Do not put the reconstruction backend in a Vercel function or other public serverless endpoint.
- All uploaded videos and output artifacts stay on the workstation unless the user explicitly opts into a separately designed data transfer.

## 2. Upload protection (mandatory)

- Allowlist only tested video containers. Initial proposal: MP4 and MOV, confirmed against installed decoders.
- Enforce a configurable byte cap during streaming, not only from the client or `Content-Length` header.
- Reject empty, over-limit, malformed, unsupported and incomplete files.
- Validate actual media with a maintained probe/decoder; do not trust file extensions or Content-Type by themselves.
- Generate server-side UUIDs and filenames. Original names are display-only and must be escaped in UI.
- Save outside any directory that is served as static frontend assets.
- Do not accept scripts, archives, arbitrary model files, command parameters, remote URLs or paths.
- Check free disk before receiving and before processing; enforce retention and maximum queue size.
- Do not buffer large videos fully in RAM.

## 3. Process-execution protection (mandatory)

- Client cannot select an executable, Python module, working directory, model file or output path.
- Construct commands from fixed trusted paths and validated server-owned paths; pass an argument array with `shell=False`.
- Never interpolate filename, job ID, form data or config into a shell command.
- Use a minimal environment allowlist. Do not pass the user's entire shell environment to child processes without review.
- Enforce per-job timeout, one active GPU job, a bounded queue, cancellation and clean child-process termination.
- Keep worker permissions limited to the app data folder and model/output locations required by the pipeline.

## 4. Access control and network

- Only named project collaborators should be allowed by the Tailscale policy. Verify policy in practice using an authorized device and an unauthorized device/account.
- App binds to loopback by default; no `0.0.0.0` default.
- Check that no router port-forwarding rule exists and public exposure is off.
- Use Tailscale Serve, not Funnel, for private HTTPS sharing.
- If architecture later introduces a public frontend, remote API, login/session, browser cookies or CORS, pause and update this threat model. Do not implement broad `Access-Control-Allow-Origin: *` for the API.
- In v1, users authorized through Tailscale share the job list. Do not upload sensitive/private-person footage until the team accepts this visibility model. Per-user privacy needs explicit application authentication and object-level authorization.

## 5. Browser and HTTP protections

- Production should be same-origin; no CORS is needed for the v1 deployment.
- Return only allowlisted JSON fields. Do not reveal tracebacks, local absolute paths, command lines, secrets or environment variables to browsers.
- Set appropriate headers: `X-Content-Type-Options: nosniff`, a restrictive `Content-Security-Policy` compatible with the viewer, `Referrer-Policy: strict-origin-when-cross-origin`, and a minimal `Permissions-Policy` (camera/microphone/geolocation disabled unless later required). Use `frame-ancestors 'none'` unless embedding is a consciously approved requirement.
- Use safe text rendering; never inject filenames, error messages or metadata with `dangerouslySetInnerHTML` or equivalent.
- Avoid third-party scripts, analytics, external fonts/media and unnecessary external requests. Self-host assets where practical.
- Use anti-CSRF protections if cookie-based authentication or other ambient credentials are later added. SameSite cookies alone should not be treated as the only defense for sensitive state-changing routes.
- Keep production debug mode and detailed API docs disabled or access-controlled if they reveal operational details.

## 6. File and output protection

- Use UUID job IDs, but do not treat UUIDs as access control. V1 authorization is the private tailnet policy.
- Resolve all artifact paths; prove they stay inside the selected job directory and do not follow symlinks outside the data root.
- Serve files only through an allowlisted manifest and only for a job in `ready` state.
- Never expose `data/`, upload temp directories, logs, SQLite DB, source videos or environment files via static file serving.
- Delete only within a validated job root. Confirm the resolved path remains under `FORGE_DATA_DIR` before recursively deleting.
- Apply retention after success/failure and on startup. Use a configurable disk safety threshold to stop new jobs before the disk fills.

## 7. Privacy controls

- Show a short notice before upload: where data is processed, who can access it, retention behaviour and whether the original video will be deleted.
- Default: local workstation processing; no automatic AWS S3, cloud model, analytics or telemetry transfer.
- Do not log video content, access credentials or raw request bodies.
- Keep event logs operational only: job ID, transition, timestamp, safe error code, byte count and measured elapsed time where necessary.
- Make deletion available and document that deleting a job may remove its generated models.
- Use only indoor videos the uploader is permitted to process. Avoid footage containing bystanders or sensitive content unless consent and rights are clear.

## 8. Secrets and configuration

- No credentials in source code, frontend bundles, README screenshots, sample data, logs or Git history.
- `.env` and real config files must be ignored by Git. Provide `.env.example` with dummy placeholders only.
- Do not reuse cloud credentials for the local site. Remove unused AWS creds from the web process environment if they are not required.
- Rotate any secret that is accidentally exposed; removing it from a later commit does not invalidate a leaked credential.

## 9. Resource-exhaustion controls

- Maximum upload bytes and duration.
- Maximum single active job (one by default).
- Small bounded queue, with full-queue `429`/`503` response.
- Worker timeout and child-process cleanup.
- Free-disk reserve and output directory quota/cleanup.
- Limited log file size/rotation.
- Reject duplicate submissions while the same video is still being processed if the implementation can safely identify duplicates.
- Do not trust client-supplied progress, dimensions or frame counts.

## 10. Logging and error handling

- Log state transitions and safe error codes; avoid sensitive content.
- Keep detailed worker logs local and restrict file permissions. Redact secrets and unnecessary paths.
- Browser errors must be short and useful: `UPLOAD_TOO_LARGE`, `UNSUPPORTED_VIDEO`, `INVALID_VIDEO`, `DISK_SPACE_LOW`, `QUEUE_FULL`, `PIPELINE_FAILED`, `OUTPUT_INVALID`, `JOB_NOT_FOUND`, `ACCESS_DENIED`.
- Unexpected exceptions return a generic message and a safe correlation/job ID; stack traces remain server-side.

## 11. Minimum security verification before sharing

- Check dependency audit findings; document unresolved findings.
- Test invalid extension, forged MIME, empty file, oversize upload, corrupt video, filename traversal, malformed UUID, path traversal on artifact routes, repeated submissions, cancellation, and deleting a job.
- Verify an outsider cannot connect from a non-tailnet network/device.
- Verify an unapproved tailnet user/device cannot access the site using the configured policy.
- Check no process listens publicly on 0.0.0.0 and no public tunnel is active.
- Check the browser receives no absolute paths, tracebacks, environment variables or secrets.
- Confirm uploaded files do not go to S3/cloud by inspecting code/config and verifying network flows during a test.
- Confirm job folders are not directly browsable and old raw videos are removed according to policy.
- Do not describe the application as fully secure or penetration-tested unless those tasks were actually completed.

## 12. Standards basis

Use OWASP Top 10:2025 as broad risk awareness, OWASP ASVS 5.0 as a verification framework, and relevant OWASP cheat sheets for file upload, input validation, DoS, REST API, SSRF, CSRF, secrets and logging. These are practical baseline requirements, not a certification claim.
