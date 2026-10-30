# UI / UX and Visual Design Guide

## 1. Product character

Design F.O.R.G.E. as a serious, deliberately designed engineering tool for indoor spatial reconstruction. It should feel like practical surveying/3D inspection software, not an “AI landing page.” This is an aesthetic direction, not a claim about who or what wrote the code.

## 2. Visual rules

- Use a restrained palette: warm white or light neutral surfaces, charcoal text, one muted technical accent (for example deep blue, slate or forest green), and clear status colors only where they carry meaning.
- Prefer a grid, consistent spacing, restrained borders, clear type hierarchy and generous whitespace.
- Use the existing F.O.R.G.E. visual language where it is coherent; inspect the current viewer before replacing anything.
- Avoid: neon/aurora gradients, excessive glow, glassmorphism everywhere, giant gradient slogans, emoji cards, AI brain icons, chat widgets, fake live counters, stock “AI robot” imagery, unnecessary floating pills, overanimated backgrounds and generic startup template language.
- Do not add illustrations merely to make a sparse screen feel busy. The 3D model itself is the visual centrepiece after reconstruction.
- Prefer real icons used consistently over emoji. Do not add decorative badges to every label.
- Use transitions sparingly; support `prefers-reduced-motion`.

## 3. Information architecture

1. **Upload / Home** — short product title, one-sentence purpose, dropzone/select file, allowed formats/limits, local-processing/privacy note, current host status.
2. **Preflight** — filename, size, duration/resolution, basic video suitability, validation warnings and explicit “Start reconstruction” action.
3. **Processing** — honest stage, elapsed time, queued/running state, cancel action where supported, no fake percentage.
4. **Result** — actual 3D viewer, reset camera, orbit/pan/zoom, view/representation controls supported by current viewer, quality report, measured runtime, download/delete.
5. **Jobs** — small recent job list and statuses, with a clear note that the private team share may make jobs visible to all authorized teammates.
6. **Settings / About** — configured file limits, storage/retention policy, build/version info and troubleshooting.

Keep the first screen focused on one primary action: upload an indoor video.

## 4. Copy style

- Plain and specific: “Upload indoor video,” “Checking video,” “Waiting for GPU,” “Reconstructing geometry,” “Checking 3D output,” “Open model,” “Delete job.”
- Avoid vague claims such as “Transform reality instantly,” “perfect reconstruction,” “AI magic” or “100% accurate.”
- Explain limitations where relevant: “The result depends on video coverage and quality. Missing/occluded surfaces may not be reconstructed.”
- Use “3D reconstruction” or “digital twin” only when a 3D output actually exists. A queued job is not a digital twin yet.
- Never show “100% complete” unless the backend reports completion of all required stages and output validation passes.

## 5. Accessibility and usability

- Every form control has a visible/accessible label; upload works by keyboard as well as drag-and-drop.
- Clear focus indicators, suitable contrast, readable font sizes and responsive layouts.
- Error messages are associated with the relevant field and explain how to recover.
- Status changes should be announced accessibly without overwhelming the user.
- The viewer must have labeled controls, including reset framing. Do not rely only on color to convey status.
- Make upload progress and processing stage distinct: bytes transferred is not model-processing progress.

## 6. Visual QA

Antigravity must run the application and capture screenshots of (at minimum) upload idle, invalid upload/error, processing, failed job and a successful actual model result. Inspect screenshots at desktop and narrow viewport widths. Do not declare visual verification complete from source code alone.

For a success screenshot, use an actual valid generated model. A seeded sample asset may be used in a clearly labeled demo/test path, but may not be presented as a new reconstruction.
