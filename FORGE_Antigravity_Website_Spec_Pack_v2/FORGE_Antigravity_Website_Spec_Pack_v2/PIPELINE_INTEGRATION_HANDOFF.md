# Pipeline Integration Handoff — Reconstruction Workstream to Website Workstream

**Purpose:** Keep the website implementation and the separate reconstruction/model workstream safely decoupled. The website integration must not guess CLI arguments, output folders or supported formats.

## Status

- [ ] Not yet validated for an indoor smartphone video.
- [ ] Do not connect the real website worker until the fields below are completed and reviewed by the owner.

## Required handoff evidence

### Source identity

- Original repository URL: `https://github.com/PrathmeshPP/FORGE`
- Tested commit SHA / branch:
- Local checkout path on Windows (do not commit this path):
- Python interpreter/version:
- Relevant environment/lockfile:

### Input contract

- Accepted file extensions/containers:
- Required codecs:
- Maximum tested duration/resolution/FPS:
- Required frame-selection or preprocessing steps:
- Known indoor-capture limitations:

### Exact invocation contract

- Exact entry script/module:
- Exact argument list (with placeholders clearly identified):
- Required working directory:
- Required environment variables (names only; never write secret values):
- Expected timeout range:
- Process exit/success criteria:

### Output contract

- Output directory pattern (relative to per-job output root):
- Primary model artifact and format:
- Related texture/material files:
- Viewer-compatible format and known loader path:
- Any additional diagnostic files:
- Files that must not be exposed to the browser:

### Hardware and performance evidence

- GPU model:
- VRAM total and observed peak usage:
- System RAM:
- Video input dimensions/duration:
- Reconstruction duration (start/end measurement method):
- Output vertex/face counts and file size:
- Known failure modes and whether CPU fallback is supported:

### Quality evidence

- Actual output loaded in the existing viewer? (yes/no)
- Screenshot/evidence path:
- Geometry validation performed:
- Visual review result and known defects:
- Scale/accuracy ground truth available? (yes/no; describe only if available)

## Integration rules

1. The website adapter invokes only the exact validated entry point and argument contract above.
2. The browser cannot select executables, scripts, command arguments, model paths or output paths.
3. Local paths are configured on the host through an ignored local config/environment variable such as `FORGE_PIPELINE_ROOT`; never commit the owner's machine-specific path.
4. The website workstream must not edit the original pipeline repo. Any change needed in that repo is returned to the separate model workstream for implementation and testing.
5. Until this handoff is complete, use a clearly labeled mock worker. Never label mock output as a reconstruction of the uploaded video.
6. Any change to this contract requires a new tested handoff and adapter tests.
