# 3D Reconstruction QA and Bounded Retry Policy

## 1. Purpose

The pipeline must not treat “the script returned zero” or “a mesh file exists” as proof that the expected digital twin was reconstructed. It must validate technical artefacts and load the actual result in the viewer before marking a job ready.

No finite checklist can prove that a reconstruction is perfect or mathematically identical to reality. The goal is to find and reject detectable failures and to report uncertainty honestly.

## 2. Required gates

### Gate A — Process execution
- Record actual command exit code, start/end times and bounded stdout/stderr logs.
- Confirm the pipeline reports success. A zero exit code is necessary but not sufficient.
- Confirm the expected output manifest for the selected scene actually exists and each required artifact is non-empty.

### Gate B — File integrity
- Parse every required 3D output with an appropriate library/loader.
- Check for truncated/corrupt files, invalid indices or unreadable textures/material references.
- Check output dimensions and total output bytes stay within configured safety limits.

### Gate C — Geometry diagnostics
Where the output format permits, compute and record:
- vertex count and face/triangle count;
- finite coordinates (no NaN or Infinity);
- index validity and finite bounds;
- positive, non-zero bounding box extents (with tolerances appropriate to the scene);
- number and proportion of degenerate/zero-area faces;
- unreferenced vertices and duplicate geometry where a reliable check exists;
- connected-component count or largest-component ratio as a diagnostic (do not fail solely on multiple components without scene-specific criteria—furniture and separate objects may form components);
- missing/invalid normals or UV/material references where applicable;
- output file sizes and texture resolution where known.

Use thresholds only after checking existing project outputs and known-good examples. A threshold should be configurable, documented and tested; don't invent a universal threshold that causes valid scenes to fail or hides bad ones.

### Gate D — Viewer integration
- Load the actual generated model in the application's actual viewer.
- Confirm the render finishes, geometry is visible and the camera can frame it.
- Check for browser console errors, failed asset requests, missing textures, black/white-only or empty output, extreme clipping, and obviously broken camera framing.
- Capture a screenshot of the result and record the local evidence path.
- If automated browser testing cannot run, open it manually and record that manual verification was performed. If neither is possible, do not mark visual verification as passed.

### Gate E — Visual review
Inspect the actual result in at least one 3D view, preferably several camera angles. Review:
- Does it resemble the target indoor space enough to be useful?
- Are floor/walls/doors/corridors recognisable and reasonably connected where captured?
- Are there obvious floating slabs, stretched geometry, repeated/duplicated surfaces, severe holes, texture smearing or camera inversion?
- Is the central area of interest visible rather than out of frame?
- Are limitations explained when source coverage is poor?

This is a visual judgment, not a certified accuracy measurement. Preserve the source video/capture conditions for comparison when authorized.

## 3. Ground truth and claims

- “Mesh parse succeeded” means only that the mesh is structurally readable.
- “Viewer load passed” means only that the viewer loaded the output without detected failure.
- “Visual QA passed” means the reviewer found no blocking visible defects against the named criteria.
- “Accurate to X cm” requires appropriate ground truth and a defined metric. Do not infer metric scale or positional accuracy from a plausible-looking render.
- “Watertight” requires a real topology/manifoldness test and should specify which mesh was tested.
- “Digital twin” may be used as product language for a 3D representation, but do not imply survey-grade fidelity unless verified.

## 4. Bounded retries

- Default maximum: 3 full reconstruction attempts per job. This may be lowered for long/expensive workloads; do not raise it silently.
- Every retry must have a reason tied to evidence (e.g., output parse failure, empty geometry, missing texture, or a known setting strongly implicated by diagnostics).
- Do not repeatedly run the same expensive pipeline with identical settings after the same failure.
- Preserve each attempt's parameters, timing, logs, metrics and artifacts under a controlled job subdirectory, with a storage cap.
- Prefer targeted repairs to rerunning VGGT when only a downstream conversion or viewer issue failed.
- Never mutate the frozen/reference output. Each attempt writes to a new isolated folder.
- After each attempt, run Gates A–D. Perform Gate E for final candidate and at least first candidate when a visual failure is suspected.
- If acceptance passes, stop and mark ready with the measured QA report.
- If the attempt limit is reached, stop, mark failed/review required, show the most relevant safe error and retain diagnostic data until cleanup. Do not return the first failed output labelled as successful.

## 5. Retry decision examples

- **Empty mesh / non-finite geometry:** inspect pipeline error and frame selection; if a documented setting can address it, change one relevant variable and rerun. Otherwise fail with diagnosis.
- **Mesh valid but viewer cannot parse format:** fix the supported output adapter/asset manifest; do not rerun geometry inference unnecessarily.
- **Missing texture files:** inspect export paths and asset manifest, fix export/copy process, rerun export only if possible.
- **Very poor geometry caused by low-quality footage:** explain capture limitation and offer capture guidance. Do not fabricate surfaces or run unlimited retries.
- **No visible output due to camera framing:** fix viewer framing if geometry bounds are valid; do not rerun reconstruction just to change the camera.
- **GPU out-of-memory:** stop process safely, record resource failure, lower workload only via approved/validated configuration, then retry within limit if the user has allowed it.

## 6. What UI can say

Allowed examples:
- “Geometry validation passed; visual review is still required.”
- “The model loaded in the viewer. Some surfaces may be missing because they were not visible in the capture.”
- “Attempt 2 of 3: the previous export was missing texture references; retrying the export stage.”
- “Reconstruction failed after 3 attempts. See the diagnostic summary.”

Not allowed:
- “Perfect reconstruction guaranteed.”
- “100% mathematically accurate” without ground-truth validation.
- “Industry-leading speed” without benchmark protocol and evidence.
- Fake percent progress inferred only from elapsed time.
