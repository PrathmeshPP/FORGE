# Research References and Security Basis

**Reviewed:** 30 October 2026. These references were chosen from primary documentation and standards sources; this is not an exhaustive search of every website. The pack translates relevant guidance into requirements for this specific private upload/reconstruction app.

## F.O.R.G.E. repository

- F.O.R.G.E. public repository and README: <https://github.com/PrathmeshPP/FORGE>
  - Public README describes root `run_pipeline.py`, `src/`, `scripts/`, a React 19 / TypeScript / Vite / Three.js viewer under `viewer/`, outputs in supported formats and an AWS Lambda/S3 video-preflight path. The README also distinguishes preflight from downstream GPU inference. Confirm current source code before implementation.

## OWASP application security

- OWASP Top 10:2025 (current released awareness list): <https://top10.owasp.org/2025/>
- OWASP Application Security Verification Standard 5.0.0: <https://asvs.owasp.org/>
- OWASP API Security Top 10 (2023): <https://api-security.owasp.org/>
- OWASP File Upload Cheat Sheet: <https://cheatsheetseries.owasp.org/cheatsheets/File_Upload_Cheat_Sheet.html>
  - Supports allowlisting extensions, checking actual content rather than trusting browser MIME type, generated filenames, size limits and authorization.
- OWASP Input Validation Cheat Sheet: <https://cheatsheetseries.owasp.org/cheatsheets/Input_Validation_Cheat_Sheet.html>
  - Supports allowlist validation and validating request size before buffering/parsing.
- OWASP Denial of Service Cheat Sheet: <https://cheatsheetseries.owasp.org/cheatsheets/Denial_of_Service_Cheat_Sheet.html>
  - Supports request-size limits, timeouts, concurrency/resource caps and access controls.
- OWASP REST Security Cheat Sheet: <https://cheatsheetseries.owasp.org/cheatsheets/REST_Security_Cheat_Sheet.html>
  - Supports proper status codes and avoiding detailed internal error disclosure.
- OWASP Server Side Request Forgery Prevention Cheat Sheet: <https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html>
  - Relevant to the decision not to allow arbitrary remote URLs or user-supplied webhooks/input URLs in v1.
- OWASP Cross-Site Request Forgery Prevention Cheat Sheet: <https://cheatsheetseries.owasp.org/cheatsheets/Cross-Site_Request_Forgery_Prevention_Cheat_Sheet.html>
  - Relevant if cookies or browser-ambient credentials are added later.
- OWASP Secrets Management Cheat Sheet: <https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html>
  - Supports keeping keys out of source/logs, least privilege, rotation and documented handling.
- OWASP Logging Cheat Sheet: <https://cheatsheetseries.owasp.org/cheatsheets/Logging_Cheat_Sheet.html>
  - Supports excluding/masking credentials and sensitive content from logs.

## Private sharing and network architecture

- Tailscale Serve: <https://tailscale.com/docs/features/tailscale-serve>
  - Describes sharing local services with the tailnet and applying tailnet access rules. Serve is private to the tailnet; Funnel is for public exposure and is not selected for v1.
- Tailscale access control: <https://tailscale.com/docs/features/access-control>
- Tailscale ACL/grants reference: <https://tailscale.com/docs/features/access-control/acls>
  - Important nuance: the Tailscale docs state that access policy may allow broad access if no restrictive ACL/grants policy is configured. Explicitly scope access to named collaborators.

## Framework and hosting constraints

- FastAPI file uploads: <https://fastapi.tiangolo.com/tutorial/request-files/>
  - `UploadFile` supports multipart file uploads; the implementation still must add application-level streaming limits and validation.
- Vercel Functions limits: <https://vercel.com/docs/functions/limitations>
  - Current docs list a 4.5 MB request/response body limit for a function and bounded execution times. This is a poor fit for large video files and lengthy local GPU reconstruction jobs.
- Vercel payload-too-large guidance: <https://vercel.com/docs/errors/function_payload_too_large>
- Vercel environment variables: <https://vercel.com/docs/environment-variables>
  - Useful if a separately approved hosted deployment is ever designed. Do not put private secrets in frontend-prefixed variables or bundles.
- MDN Same-Origin Policy: <https://developer.mozilla.org/en-US/docs/Web/Security/Defenses/Same-origin_policy>
  - Supports the simpler same-origin v1 choice and explains why local file URLs have special origin semantics.

## How to interpret this pack

- The standards are guidance and verification baselines; incorporating them does not certify the software as secure.
- Security and privacy requirements must be verified through code inspection and actual tests.
- Links, features, limits and framework behaviour can change. Recheck official documentation before public deployment.


## Current workspace details (owner-provided; verify on host)

- Website working folder: `C:\Users\25beevdt025\Desktop\FORGE_website`
- Website repository: <https://github.com/PrathmeshPP/FORGE_website> (observed empty on GitHub on 30 October 2026; verify again).
- Existing pipeline repository: <https://github.com/PrathmeshPP/FORGE> (read-only to the website workstream).
- Host: Windows PC with NVIDIA GPU and 8 GB VRAM. Exact GPU, driver, free VRAM, CUDA/PyTorch compatibility, RAM and disk are unverified until probed locally.
- The website must use a mock worker until `PIPELINE_INTEGRATION_HANDOFF.md` is completed by the separate model workstream.

Windows official documentation: <https://learn.microsoft.com/en-us/powershell/> and Python on Windows: <https://docs.python.org/3/using/windows.html>. Use the actual installed toolchain and current official documentation rather than assuming a specific version.
