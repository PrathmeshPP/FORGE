# Workspace Profile — F.O.R.G.E. Website Workstream

**Last updated:** 30 October 2026

## Known workspace facts supplied by the owner

| Item | Current value |
|---|---|
| Website working folder | `C:\Users\25beevdt025\Desktop\FORGE_website` |
| New website GitHub repository | <https://github.com/PrathmeshPP/FORGE_website> |
| Existing reconstruction repository | <https://github.com/PrathmeshPP/FORGE> |
| Main development/compute PC | Windows PC, available for development and testing |
| GPU memory ceiling | NVIDIA GPU with 8 GB VRAM (exact GPU model not yet confirmed) |
| Website scope | Indoor-video upload and indoor 3D digital-twin inspection only |
| Reconstruction workstream | Separate chat/workstream; website work must not modify its pipeline source |
| Intended deployment | Run website/backend on the owner's PC; share privately with approved teammates through Tailscale Serve |

## Repository state observed online on 30 October 2026

- `FORGE_website` displayed the GitHub “This repository is empty” state when checked. Recheck before acting; the remote may have changed since then.
- `FORGE` is the existing source repository. Its public README describes a root `run_pipeline.py`, `src/` and `scripts/` reconstruction code, and a React/TypeScript/Vite/Three.js viewer under `viewer/`. Confirm all details against the actual checkout and current commit.

## Important unknowns — detect, do not assume

- Exact Windows edition/build.
- Exact GPU make/model, driver version and free VRAM.
- Installed CUDA/PyTorch versions and whether PyTorch can see the GPU.
- Python interpreter/version and Node/package-manager versions.
- System RAM and available disk space on each drive.
- Whether the local website folder is already a Git clone and whether `origin` is configured.
- Whether the original `FORGE` pipeline repository is cloned locally and its actual path.
- Whether Desktop is redirected or synchronized by OneDrive/another sync client.
- Which indoor video format, duration and dimensions the validated pipeline will support.
- Actual indoor pipeline CLI arguments, dependencies, output paths and asset formats.

## Windows diagnostic commands (read-only)

Run these in PowerShell as appropriate; do not print secrets or all environment variables:

```powershell
Get-Location
Get-ChildItem -Force
Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, BuildNumber
Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory
Get-Volume | Select-Object DriveLetter, FileSystem, Size, SizeRemaining
py -3 --version
python --version
node --version
npm --version
git --version
git status --short --branch
git remote -v
nvidia-smi
```

Some commands may not exist or may return an error; report that as “not available” rather than installing/upgrading tools without approval. `nvidia-smi` output should be summarized to the GPU model, total/used/free VRAM and driver version; do not dump unrelated host details into browser logs.

## Data location and privacy

- Do not store uploaded videos, generated models, temp files, logs or SQLite data inside the Git working tree.
- Do not silently store job data under Desktop/OneDrive/Dropbox/Google Drive/iCloud or another synchronized folder.
- Inspect available local drives and free space, then propose a local non-synced `FORGE_DATA_DIR`; get owner approval before creating the data store or retaining user videos.
- No automatic AWS/S3, cloud GPU, telemetry, analytics, external API or third-party upload in v1.

## Cross-workstream handoff

The separate reconstruction/model workstream must fill in `PIPELINE_INTEGRATION_HANDOFF.md` with a validated indoor invocation and output contract. Website work can proceed with a clearly labeled mock worker while this is pending. Do not edit the original pipeline repository from the website workstream.
