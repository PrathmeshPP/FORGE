# Deployment and Operations Runbook

**V1 hosting model:** The owner's Windows workstation (NVIDIA GPU, 8 GB VRAM) is the host. The web API, built frontend and local worker run on that machine. Approved teammates access it over the private Tailscale network. Exact GPU/driver/runtime versions must be detected, not assumed. Website code is in `FORGE_website`; the original `FORGE` pipeline repository is read-only to this workstream.


## 0. Windows host notes

- Use PowerShell for setup and operations. The host path supplied by the owner is `C:\Users\25beevdt025\Desktop\FORGE_website`.
- Verify the actual folder, Git state/remotes, Windows build, Python, Node/package manager, available drive space and `nvidia-smi` before setup.
- Do not assume the original `FORGE` repo is cloned beside the website. If it is present, treat it as read-only and configure its local path through `FORGE_PIPELINE_ROOT` in an ignored local configuration file. Never commit the owner's absolute path.
- Put uploads, temporary files, model outputs, SQLite and logs in a local non-synchronized directory outside the repo. Confirm free space and that the directory is not under OneDrive or another sync provider.
- The 8 GB VRAM figure is total capacity, not guaranteed free memory. Check free VRAM before each job and run one job at a time.
- The separate model workstream owns real pipeline tests. Until its indoor pipeline handoff is approved, use the deterministic mock worker and label it clearly.

Read-only diagnostic examples (PowerShell):

```powershell
Get-Location
Get-ChildItem -Force
git status --short --branch
git remote -v
py -3 --version
node --version
npm --version
nvidia-smi
Get-Volume | Select-Object DriveLetter, Size, SizeRemaining
```

Do not dump all environment variables, install global tools, change execution policy globally, update drivers/CUDA or run a heavy reconstruction without owner approval.

## 1. Pre-deployment checks

- Confirm the main workstation has compatible Python, Node/package manager, model weights and a working local reconstruction setup.
- Confirm the current F.O.R.G.E. pipeline can run manually on a test indoor video before connecting the UI.
- Confirm sufficient disk space for input, intermediate files, model outputs, and at least one preserved previous output.
- Check current requirements/lockfiles and repository branch; do not upgrade the full environment without a reason.
- Ensure `data/`, logs, SQLite DB, `.env`, uploads and new job outputs are excluded from Git.
- Back up known-good/reference models and inspect Git status before the agent modifies anything.

## 2. First local run (development)

The exact commands must be derived from the website repo's actual scripts and package manager. On Windows, add tested PowerShell `.ps1` or `.bat` launchers (or documented PowerShell commands) only after testing them. Do not assume Bash scripts can run natively.

Expected workflow:
1. Start backend on `127.0.0.1:<configured-port>`.
2. Start Vite in development mode on `127.0.0.1` if needed; use the documented proxy to the backend.
3. Open the local site and test with a small, non-sensitive indoor sample.
4. Stop both processes with the documented stop commands.

Do not expose the Vite dev server or API to the LAN by default.

## 3. Production-like private run

1. Build the frontend using the existing package scripts.
2. Serve the built frontend and API from one local origin (the backend or a local reverse proxy) to avoid unnecessary CORS configuration.
3. Start the backend on loopback only.
4. Verify the health endpoint from the host.
5. Configure Tailscale Serve to proxy its local port, following the current official docs: <https://tailscale.com/docs/features/tailscale-serve>.
6. Configure explicit grants/ACLs so only named project collaborators can reach the host/service. Review: <https://tailscale.com/docs/features/access-control>.
7. Verify no Tailscale Funnel/public exposure is active. Do not configure router port forwarding.
8. From an approved teammate's device with Tailscale enabled, open the Serve URL and test a small upload. From a non-approved device/account, verify access fails.
9. Keep the terminal/service process running while the team needs it. If the machine sleeps, reboots, loses network or the app stops, the site is unavailable until restored.

Exact `tailscale serve` command depends on the current Tailscale client and chosen local port. Use the official current CLI docs and do not invent a stale command.

## 4. Start/stop operations

Create tested scripts or platform-specific instructions for:
- start local backend/worker;
- stop gracefully;
- show service/worker health;
- inspect job status;
- run dependency/test checks.

Stop sequence:
1. Reject new uploads.
2. Allow active jobs to complete or explicitly cancel them.
3. Confirm worker child processes are not left running.
4. Stop backend.
5. Stop Tailscale Serve foreground/session or disable the configured route using current Tailscale commands.
6. Confirm remote access no longer works.

Do not delete job data just because the web service stops.

## 5. Operational safety

- Do not accept confidential/third-party videos until the privacy and access-control assumptions are understood by the team.
- Check disk space before each job and monitor storage after model outputs/intermediates accumulate.
- Keep only one reconstruction job running by default.
- Use a separate per-job workspace; preserve the known-good reference outputs.
- After an unclean shutdown, inspect stale jobs and clean temporary files using the app's controlled cleanup path.
- If a secret is leaked, stop the service if necessary, rotate the secret and inspect Git history/build output/logs.

## 6. Troubleshooting

- **Site unreachable:** check host power/network, backend health, loopback port, Tailscale status, HTTPS/Serve status and explicit access policy.
- **Teammate can see UI but upload fails:** inspect same-origin API base URL, body-size limit, streaming upload logs and disk space. Do not work around by making the backend public.
- **Upload succeeds but reconstruction fails:** inspect safe job diagnostics and local worker logs; confirm actual CLI/model weights/GPU memory and input compatibility.
- **Model generated but viewer fails:** validate artifact manifest, format/texture references, browser network tab and console. Prefer fixing output packaging/loader rather than rerunning expensive model inference unnecessarily.
- **GPU out of memory:** stop safely, preserve diagnostic evidence, do not automatically launch multiple concurrent jobs.

## 7. Not part of v1

No public Vercel API, no public tunnel, no serverless reconstruction, no cloud GPU and no automatic AWS/S3 transfer. Any move to public hosting requires a revised architecture and threat-model review.
