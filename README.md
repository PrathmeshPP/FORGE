# F.O.R.G.E — Fast Optical Reconstruction & Geospatial Estimation

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV 5.0.0.93](https://img.shields.io/badge/OpenCV-5.0.0.93-red.svg)](https://pypi.org/project/opencv-python/5.0.0.93/)
[![Three.js](https://img.shields.io/badge/Three.js-r186-black.svg)](https://threejs.org/)
[![Vite](https://img.shields.io/badge/Vite-8.3-646CFF.svg)](https://vitejs.dev/)
[![Git LFS](https://img.shields.io/badge/Git%20LFS-3.5%20GB-orange.svg)](https://git-lfs.github.com/)
[![License Status](https://img.shields.io/badge/License-All%20Rights%20Reserved%20(Pending%20Selection)-lightgrey.svg)](docs/THIRD_PARTY_ASSET_PROVENANCE.md)

F.O.R.G.E is a spatial intelligence and 3D scene reconstruction platform built around feed-forward vision foundation models (VGGT-Ω). It processes monocular aerial drone and corridor video sequences into metric 3D point clouds, dense depth maps, camera trajectory poses, textured triangle surface meshes, and interactive web visualization assets.

---

## Architecture & System Overview

The platform consists of three integrated subsystems:
1. **F.O.R.G.E AWS Video Preflight Service (`aws/preflight/`, `scripts/run_aws_preflight.py`)**: Serverless AWS Lambda quality and feasibility inspection worker using `opencv-python-headless==5.0.0.93` and private Amazon S3 storage (`forge-drone3d-prathmesh-2026-8xq472`). Automatically triggered on video upload to evaluate container integrity, compute Laplacian sharpness and ORB feature distributions, assemble a visual contact sheet, and enforce a 3D reconstruction feasibility gate (`APPROVED_FOR_3D_RECONSTRUCTION`, `NEEDS_REVIEW`, `REJECTED`) before downstream GPU foundation model inference.
2. **F.O.R.G.E Core Reconstruction Pipeline (`src/`, `scripts/`)**: OpenCV 5 quality/diversity frame curation (`src/video/opencv5_frame_analyzer.py`), feed-forward foundation model inference (VGGT-Ω 1B), pinhole depth backprojection, multi-view consistency filtering, Poisson/BPA surface meshing, photographic KD-tree color transfer, and Umeyama–RANSAC 7-DoF geospatial alignment.
3. **F.O.R.G.E Explorer Interactive Viewer (`viewer/`, `scripts/serve_viewer.py`)**: React 19, TypeScript, Vite, and Three.js web application for real-time 3D inspection, camera framing presets, multi-representation rendering (textured, photographic, untextured, point cloud), and split-view comparative analysis.

```
                  [ Raw Drone Video Upload ]
                              │
                              ▼
            [ Amazon S3: FORGE/inputs/videos/*.mp4 ]
                              │
                    (S3 ObjectCreated Event)
                              ▼
       [ AWS Lambda Worker: OpenCV 5 Video Preflight ]
         ├── Container Probe (FPS, Dimensions, Frames)
         ├── Bounded Corridor Sampling
         ├── OpenCV 5 Laplacian & Tenengrad Sharpness
         ├── ORB Invariant Feature Distribution
         └── Feasibility Gate & Contact Sheet Generation
                              │
                 (JSON Report & Contact Sheet)
                              ▼
        [ Amazon S3: FORGE/outputs/aws_preflight/ ]
                              │
                   [ Gate: Approved Video ]
                              │
                              ▼
            [ Local / Field Edge GPU Workstation ]
                              │
                              ▼
            [ OpenCV 5 Dynamic Viewpoint Curation ]
                              │
                              ▼
          [ VGGT-Ω 1B Feed-Forward Foundation Model ]
                              │
             ┌────────────────┴────────────────┐
             ▼                                 ▼
   [ Metric Depth & Confidence ]      [ Camera Poses (R, T, K) ]
             │                                 │
             └────────────────┬────────────────┘
                              ▼
                 [ Dense Point Cloud Fusion ]
                              │
                              ▼
                   [ Surface Mesh Generation ]
                  (Poisson / Ball-Pivoting)
                              │
                              ▼
                [ Photographic Texture Mapping ]
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                           ▼
[ F.O.R.G.E Explorer Viewer ]           [ Standalone Inspection Exports ]
 (Three.js Web Application)              (OBJ, PLY, GLB, MTL, 4K Atlases)
```

---

## Benchmark Scenes & Reconstruction Baseline

The project evaluates four real-world video sources across five benchmark entries:

| Scene Entry | Category | Source Video | Geometry | Metric & Calibration Status | Key Scene Characteristics |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Campus 0049** | Architectural Survey | `dji_fly_...0049...mp4` | 128k v, 253k f | Model-relative; uncalibrated scale. Non-georeferenced. | Institutional courtyard, buildings, reflecting pool, walkways. Hovering drone limits facade angles. |
| **2. Desert Road (Candidate A)** | Highway Corridor | `desert_road.mp4` | 53k v, 105k f | Camera-relative plane fit. Uncalibrated scale. | Static clean highway across sand dunes; moving traffic removed. |
| **3. Desert Road (Candidate B)** | Dynamic Preservation | `desert_road.mp4` | 53k v, 105k f | Camera-relative plane fit. Uncalibrated scale. | Highway corridor preserving single reference vehicle at `[0.03, 0.04, 0.45]`. |
| **4. Forest Road** | Terrain & Vegetation | `forest_road.mp4` | 462k v, 919k f | Model-relative; uncalibrated scale. Non-georeferenced. | Dense mountain tree canopy, winding asphalt road with yellow lane markings. |
| **5. River/Bridge** | Civil Infrastructure | `segment_52s_64s.mp4` | 132k v, 260k f | Metric GPS georeferenced via 7-DoF Umeyama–RANSAC alignment. | Dual-span highway bridge, concrete piers, riverbed channel and approach road. |

> [!IMPORTANT]
> **Reconstruction Baseline & Deliverable Clarification:**
> - **Coordinate Scale & Calibration:** Unless explicitly georeferenced via drone telemetry (e.g., River/Bridge), scene coordinates are model-relative. Metric scale is arbitrary. Unseen building facades, foliage voids, and reflective water specular noise reflect physical capture geometry and are honestly not synthesized.
> - **Deliverable Provenance:** Standalone deliverables in `outputs/manual_model_review/` are **format-converted inspection packages** (OBJ, PLY, GLB) generated from the frozen Sprint 9 baseline (`outputs/accuracy_experiment/sprint9_video_to_viewer_release/`), not newly run reconstructions.

---

## Canonical Asset Deliverables

The repository maintains two canonical deliverables directories:

### 1. Live Web Viewer Baseline (`outputs/accuracy_experiment/sprint9_video_to_viewer_release/`)
The authoritative frozen baseline served directly to the F.O.R.G.E Explorer web application:
- `textured_mesh.obj` & `textured_mesh.mtl`: Polygonal Wavefront mesh with explicit UV mapping.
- `texture_atlas.png`: 4096×4096 unified diffuse texture atlas.
- `surface_mesh_photographic.ply`: KD-tree vertex-colored photographic mesh.
- `surface_mesh_repaired.ply`: Watertight geometric surface mesh.
- Verified render previews and candidate metadata JSONs.

### 2. Standalone Manual Review Packages (`outputs/manual_model_review/`)
Independent deliverables for direct inspection in **Blender**, **MeshLab**, **CloudCompare**, or **Windows 3D Viewer**:
- `model_geometry.ply` & `model_geometry.obj`: Clean coordinate geometry with zero texture dependencies.
- `model_photographic.ply`: Photographic vertex-colored RGB mesh.
- `model_textured.obj`, `model_textured.mtl`, and `texture_atlas.png`: Textured model for 3D DCC tools.
- `model_preview.glb`: Self-contained glTF 2.0 binary with embedded textures for 1-click inspection.
- `forge_models_for_manual_inspection.zip` (354.2 MB): Consolidated master archive containing all 5 scenes.

Detailed opening guides and SHA-256 checksums are cataloged in [`outputs/manual_model_review/MODEL_CATALOG.md`](outputs/manual_model_review/MODEL_CATALOG.md).

### 3. Sprint 11 Dynamic Masking & Recovery V2.1 Deliverables (`outputs/dynamic_object_masking/sprint11/`)
The canonical deliverables integrating pixel-accurate instance segmentation dynamic masking:
- **Campus 0049 (Recovery V2.1):** [`outputs/dynamic_object_masking/sprint11/recovery_v2_1/campus_0049/`](outputs/dynamic_object_masking/sprint11/recovery_v2_1/campus_0049/) — Audited baseline geometry (128k v, 253k f) + 4096×4096 texture atlas with sharp ground lettering.
- **Desert Road (Recovery V2.1):** [`outputs/dynamic_object_masking/sprint11/recovery_v2_1/desert_road/`](outputs/dynamic_object_masking/sprint11/recovery_v2_1/desert_road/) — Moving vehicle & floating balloon projections completely excised; clean inpainted 4K atlas; planar road corridor (< 2.0 mm mean deviation).
- **Forest Road (Recovery V2):** [`outputs/dynamic_object_masking/sprint11/recovery_v2/forest_road/`](outputs/dynamic_object_masking/sprint11/recovery_v2/forest_road/) — High-fidelity canopy (445k v, 889k f) with verified 0 residual vehicle points.
- **River/Bridge (Recovery V2):** [`outputs/dynamic_object_masking/sprint11/recovery_v2/river_bridge/`](outputs/dynamic_object_masking/sprint11/recovery_v2/river_bridge/) — Structural concrete bridge deck, piers, and riverbanks (127k v, 251k f).

Complete catalog, SHA-256 checksums, and LFS accounting are documented in [`docs/SPRINT11_RELEASE_MANIFEST.md`](docs/SPRINT11_RELEASE_MANIFEST.md).

---

## Source Video Licensing & Provenance

The four source videos used across the five scenes have distinct redistribution terms:

| Video Asset | Path | Size | Origin / Rights Holder | License | Redistribution Status |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **Campus 0049** | `data/source_videos/dji_fly_...0049...mp4` | 154.5 MB | Attributed to Team (`PrathmeshPP` / SIH) | First-party proprietary | **UNCONFIRMED** (Plausible First-Party Inference; Pending Formal Release) |
| **River/Bridge** | `data/source_videos/segment_52s_64s.mp4` | 66.8 MB | Attributed to Team (`PrathmeshPP` / SIH) | First-party proprietary | **UNCONFIRMED** (Plausible First-Party Inference; Pending Formal Release) |
| **Desert Road** | `data/source_videos/desert_road.mp4` | 6.2 MB | Meta AI & Univ. of Oxford (`VGGT-Omega`) | [Meta FAIR Noncommercial License](docs/licenses/META_FAIR_NONCOMMERCIAL_LICENSE.txt) | **RESTRICTED** (Noncommercial Research only; AUP restrictions apply) |
| **Forest Road** | `data/source_videos/forest_road.mp4` | 10.2 MB | Meta AI & Univ. of Oxford (`VGGT-Omega`) | [Meta FAIR Noncommercial License](docs/licenses/META_FAIR_NONCOMMERCIAL_LICENSE.txt) | **RESTRICTED** (Noncommercial Research only; AUP restrictions apply) |

> [!NOTE]
> **Source Video Distribution & Local Input Guidance:**
> - **Benchmark Sequences (`desert_road.mp4`, `forest_road.mp4`):** Cleared for noncommercial research and educational distribution under the [Meta FAIR Noncommercial Research License](docs/licenses/META_FAIR_NONCOMMERCIAL_LICENSE.txt).
> - **Proprietary Drone Flights (`Campus 0049`, `River/Bridge`):** Attributed to first-party team flight captures. Private master 4K recordings are backed up securely in the team's private S3 bucket.
> - **Supplying Your Own Video:** F.O.R.G.E is fully input-agnostic. To run 3D reconstruction on your own drone footage, simply point the pipeline to your local MP4 file:
>   ```bash
>   python run_pipeline.py --video "path/to/your_drone_video.mp4" --output-dir "outputs/my_reconstruction"
>   ```
> - **Historical Git LFS Note:** Historical commits in Git retain LFS object pointers. In accordance with zero-destructive-history policies, Git history has been preserved without force-pushes or history rewrites. Detailed legal analysis is documented in [`docs/THIRD_PARTY_ASSET_PROVENANCE.md`](docs/THIRD_PARTY_ASSET_PROVENANCE.md).

---

## Repository Documentation Index

| Document | Purpose & Contents |
| :--- | :--- |
| [`docs/AWS_DEPLOYMENT.md`](docs/AWS_DEPLOYMENT.md) | **AWS Cloud Preflight Architecture & Deployment:** Serverless OpenCV 5 Lambda worker, S3 trigger, IAM least-privilege policy, SAM template, and verification guide. |
| [`docs/SPRINT11_RELEASE_MANIFEST.md`](docs/SPRINT11_RELEASE_MANIFEST.md) | **Sprint 11 Canonical Release Manifest:** Selected models, SHA-256 hashes, LFS accounting, and reproduction scripts. |
| [`docs/THIRD_PARTY_ASSET_PROVENANCE.md`](docs/THIRD_PARTY_ASSET_PROVENANCE.md) | Exhaustive legal provenance, upstream URLs, Meta FAIR terms, and project licensing options. |
| [`docs/REPOSITORY_ASSET_POLICY.md`](docs/REPOSITORY_ASSET_POLICY.md) | Durable three-tier artifact policy (Git vs Git LFS vs External untracked foundation weights). |
| [`docs/GIT_LFS_AND_STORAGE.md`](docs/GIT_LFS_AND_STORAGE.md) | Git LFS inventory, bandwidth limits, selective checkout commands, and historical purge caveats. |
| [`docs/ASSET_MANIFEST.md`](docs/ASSET_MANIFEST.md) | Human-readable verification manifest of all 484 tracked project deliverables. |
| [`docs/asset_manifest.json`](docs/asset_manifest.json) | Machine-readable SHA-256 asset manifest with source video verification statuses. |
| [`docs/SPRINT10_LIVE_DEMO_GUIDE.md`](docs/SPRINT10_LIVE_DEMO_GUIDE.md) | Complete presenter walk-through script and camera framing guide for evaluators. |
| [`docs/SPRINT11_DYNAMIC_MASKING_REPORT.md`](docs/SPRINT11_DYNAMIC_MASKING_REPORT.md) | Sprint 11 YOLO dynamic object masking, visual diagnostics, and 3D reconstruction benchmark report. |
| [`docs/OPENCV_COMPLETION_ROADMAP.md`](docs/OPENCV_COMPLETION_ROADMAP.md) | Mathematical formulation for GPS alignment, pose recovery, and texturing. |

---

## Installation & Quickstart

### 1. Prerequisites
- **Python:** 3.11 or 3.12 recommended.
- **Node.js:** v18+ (for building the viewer).
- **Git LFS:** Required for retrieving large 3D models and source videos.
- **CUDA GPU:** Strongly recommended for inference (`cuda:0`), optional for viewer inspection.

### 2. Git LFS Checkout
```bash
# Install Git LFS hooks
git lfs install

# Clone the repository
git clone https://github.com/PrathmeshPP/FORGE.git
cd FORGE

# Pull all tracked LFS assets (~3.5 GB)
git lfs pull
```

*Selective checkout options (to pull only specific scenes or viewer meshes) are documented in [`docs/GIT_LFS_AND_STORAGE.md`](docs/GIT_LFS_AND_STORAGE.md).*

### 3. Python Environment Setup & OpenCV 5 Configuration
F.O.R.G.E is fully upgraded and pinned to **OpenCV 5** (`opencv-python==5.0.0.93`):
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies (ensures single clean opencv-python distribution)
pip install -r requirements.txt
pip install -r requirements-viewer.txt

# Verify active OpenCV runtime is 5.x
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
# Expected: OpenCV version: 5.0.0
```
> [!IMPORTANT]
> **Single OpenCV Wheel Rule:** Do not install multiple OpenCV distributions concurrently (`opencv-python` and `opencv-python-headless`). Only `opencv-python==5.0.0.93` should be installed.

### 4. OpenCV 5 Smoke Test & Verification
Execute the reproducible OpenCV 5 compliance suite to verify video decoding, multi-metric sharpness, ORB feature distribution, and epipolar RANSAC verification on real drone video:
```bash
# Run standalone smoke test (auto-locates local drone footage or accepts --video)
python tests/test_opencv5_compliance.py

# Or run via unittest module
python -m unittest tests.test_opencv5_compliance
```

### 5. Running the Pipeline with OpenCV 5 Preprocessing
The primary pipeline automatically uses OpenCV 5 for deterministic quality and viewpoint curation (`--frame-selector opencv5` is the default):
```bash
# Run real video ingestion with OpenCV 5 multi-metric curation and VGGT-Omega inference
python run_pipeline.py --video "data/source_videos/desert_road.mp4" --output-dir "outputs/pipeline_run"

# Optional explicit arguments
python run_pipeline.py --video "path/to/drone_video.mp4" --frame-selector opencv5 --max-frames 6
```
Where OpenCV 5 is actively used:
- **`src/video/opencv5_frame_analyzer.py`**:
  - Multi-metric sharpness: Dual evaluation with Laplacian variance `cv2.Laplacian` and Tenengrad `cv2.Sobel`.
  - Radiometric fidelity: Mean intensity, RMS contrast, and dynamic range calculation.
  - Spatial feature entropy: ORB keypoints (`cv2.ORB_create`) across a 4×4 spatial grid with Shannon distribution entropy.
  - Pairwise visual disparity: Grayscale histogram correlation (`cv2.calcHist`, `cv2.compareHist`) and Bhattacharyya distance.
  - Epipolar covisibility: Feature correspondence (`cv2.BFMatcher`) and Fundamental matrix RANSAC estimation (`cv2.findFundamentalMat`).
- **`src/video/extract_frames.py` & `src/video/frame_sampler.py`**: Stream reading, FPS/dimension probing, and color space transformations via `cv2.VideoCapture` and `cv2.cvtColor`.
- Complete migration details are recorded in [`OPENCV5_MIGRATION_REPORT.md`](OPENCV5_MIGRATION_REPORT.md).

### 6. AWS Cloud Video Preflight Service (Lambda + S3)
F.O.R.G.E integrates an event-driven AWS serverless preflight worker ([`aws/preflight/lambda_function.py`](aws/preflight/lambda_function.py)) designed to evaluate uploaded drone footage in Amazon S3 (`forge-drone3d-prathmesh-2026-8xq472` in `ap-southeast-2`) before allocating compute for GPU reconstruction.

The preflight stage enforces strict quality gates (Laplacian blur threshold >= 80, ORB feature density >= 100):
```bash
# Run the preflight inspector (auto-detects AWS credentials or runs local evaluation engine)
python scripts/run_aws_preflight.py --video "data/source_videos/desert_road.mp4"

# Run against Campus 0049 drone footage
python scripts/run_aws_preflight.py --video "data/source_videos/dji_fly_20260727_172754_0049_1785175051818_video.mp4"

# Run tests for AWS Lambda preflight component (OpenCV 5 runtime, mock S3, metrics)
python -m unittest tests.test_aws_preflight
```
Deployment templates (AWS SAM [`aws/preflight/template.yaml`](aws/preflight/template.yaml)), IAM least-privilege policies, and step-by-step cloud deployment commands are documented in [`docs/AWS_DEPLOYMENT.md`](docs/AWS_DEPLOYMENT.md).

### 7. Launching F.O.R.G.E Explorer

#### One-Click Launch (Windows)
Double-click `start_forge_explorer.bat` or execute in command prompt:
```cmd
start_forge_explorer.bat
```

#### Manual Launch
```bash
python scripts/serve_viewer.py --port 5173
```
Open [http://localhost:5173](http://localhost:5173) in Chrome, Edge, or Firefox.

### 8. Running YOLO Dynamic-Object Masking
```bash
# Run YOLO dynamic masking on GPU across all benchmark scenes
python scripts/run_dynamic_masking.py --scene all --device cuda

# Run on Desert Road specifically
python scripts/run_dynamic_masking.py --scene desert_road --device cuda

# Execute comparative masked vs unmasked 3D reconstruction benchmark
python scripts/run_desert_road_masked_reconstruction.py
```

---

## Automated Test Suite

F.O.R.G.E includes comprehensive unit and integration tests verifying manifest accuracy, geometry extents, coordinate finiteness, texture bindings, and HTTP asset serving:

```bash
# Run AWS Lambda preflight unit and integration test suite
python -m unittest tests.test_aws_preflight

# Run OpenCV 5 compliance suite
python -m unittest tests.test_opencv5_compliance

# Run comprehensive release test suites
pytest tests/test_sprint9_video_to_viewer.py tests/test_sprint10_demo_hardening.py tests/test_sprint10_1_viewer_refinement.py tests/test_sprint10_2_standalone_review.py tests/test_sprint11_dynamic_masking.py -v
```

---

## Project License & Ownership

- **Repository Owner:** [PrathmeshPP](https://github.com/PrathmeshPP)
- **Current License Status:** **All Rights Reserved** (Pending formal license selection by owner; see [`docs/THIRD_PARTY_ASSET_PROVENANCE.md`](docs/THIRD_PARTY_ASSET_PROVENANCE.md)).
- **Third-Party Research Materials:** `desert_road.mp4`, `forest_road.mp4`, and VGGT-Ω foundation model code/weights are licensed under the **Meta FAIR Noncommercial Research License (v1, Oct 2024)**. Commercial exploitation is prohibited without separate licensing from Meta AI.
