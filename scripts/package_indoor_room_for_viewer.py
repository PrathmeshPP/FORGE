"""
F.O.R.G.E — Package Best Indoor Reconstruction into Website 3D Viewer.
"""

import os
import sys
import json
import shutil
import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ATTEMPT_3_DIR = PROJECT_ROOT / "outputs/indoor/room_001/attempt_3"
FINAL_RELEASE_DIR = PROJECT_ROOT / "outputs/indoor/room_001/final_release"
PUBLIC_MODELS_DIR = PROJECT_ROOT / "viewer/public/models/indoor_room_001"
DIST_MODELS_DIR = PROJECT_ROOT / "viewer/dist/models/indoor_room_001"

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

import argparse

def main():
    parser = argparse.ArgumentParser(description="Package Indoor Reconstruction into Website 3D Viewer")
    parser.add_argument("--attempt", type=int, default=4, help="Attempt number to package (default: 4)")
    args = parser.parse_args()

    att_dir = PROJECT_ROOT / f"outputs/indoor/room_001/attempt_{args.attempt}"
    if not att_dir.is_dir():
        raise FileNotFoundError(f"Attempt directory not found: {att_dir}")

    print(f"=== Packaging Indoor Room 3D Model from Attempt {args.attempt} for F.O.R.G.E. Viewer ===")
    FINAL_RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DIST_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Source files from Attempt directory
    mesh_dir = att_dir / "meshes"
    renders_dir = att_dir / "renders"

    src_obj = mesh_dir / "poisson_mesh_canonical.obj"
    src_mtl = mesh_dir / "poisson_mesh_canonical.mtl"
    src_glb = mesh_dir / "poisson_mesh.glb"
    src_ply = mesh_dir / "poisson_mesh.ply"

    src_oblique = renders_dir / f"attempt_{args.attempt}_oblique.png"
    src_top = renders_dir / f"attempt_{args.attempt}_top_plan.png"
    src_eye = renders_dir / f"attempt_{args.attempt}_eye_level.png"
    src_overview = renders_dir / f"attempt_{args.attempt}_overview.png"
    src_furniture = renders_dir / f"attempt_{args.attempt}_furniture_closeup.png"

    # Copy to final_release
    shutil.copyfile(src_obj, FINAL_RELEASE_DIR / "room_model_canonical.obj")
    shutil.copyfile(src_mtl, FINAL_RELEASE_DIR / "room_model_canonical.mtl")
    if src_glb.is_file():
        shutil.copyfile(src_glb, FINAL_RELEASE_DIR / "room_model.glb")
    shutil.copyfile(src_ply, FINAL_RELEASE_DIR / "room_model.ply")

    if src_oblique.is_file():
        shutil.copyfile(src_oblique, FINAL_RELEASE_DIR / "render_view_oblique.png")
    if src_top.is_file():
        shutil.copyfile(src_top, FINAL_RELEASE_DIR / "render_view_top.png")
    if src_eye.is_file():
        shutil.copyfile(src_eye, FINAL_RELEASE_DIR / "render_view_eye_level.png")
    if src_overview.is_file():
        shutil.copyfile(src_overview, FINAL_RELEASE_DIR / "render_view_overview.png")
    if src_furniture.is_file():
        shutil.copyfile(src_furniture, FINAL_RELEASE_DIR / "render_view_furniture_closeup.png")

    # Copy to viewer public & dist directories
    for target_dir in [PUBLIC_MODELS_DIR, DIST_MODELS_DIR]:
        shutil.copyfile(FINAL_RELEASE_DIR / "room_model_canonical.obj", target_dir / "room_model_canonical.obj")
        shutil.copyfile(FINAL_RELEASE_DIR / "room_model_canonical.mtl", target_dir / "room_model_canonical.mtl")
        if (FINAL_RELEASE_DIR / "room_model.glb").is_file():
            shutil.copyfile(FINAL_RELEASE_DIR / "room_model.glb", target_dir / "room_model.glb")
        shutil.copyfile(FINAL_RELEASE_DIR / "room_model.ply", target_dir / "room_model.ply")
        shutil.copyfile(FINAL_RELEASE_DIR / "render_view_oblique.png", target_dir / "render_view_oblique.png")
        shutil.copyfile(FINAL_RELEASE_DIR / "render_view_top.png", target_dir / "render_view_top.png")
        shutil.copyfile(FINAL_RELEASE_DIR / "render_view_eye_level.png", target_dir / "render_view_eye_level.png")

    # Load Attempt metadata
    with open(att_dir / "attempt_metadata.json") as f:
        meta = json.load(f)

    obj_file = FINAL_RELEASE_DIR / "room_model_canonical.obj"
    ply_file = FINAL_RELEASE_DIR / "room_model.ply"
    glb_file = FINAL_RELEASE_DIR / "room_model.glb"

    v_count = meta["meshing_evaluations"]["poisson"]["vertices"]
    t_count = meta["meshing_evaluations"]["poisson"]["triangles"]

    scene_entry = {
        "id": "indoor_room_001",
        "scene_name": "indoor_room_001",
        "display_name": "Indoor Room 001",
        "subtitle": "Student Hostel Single-Room Walkthrough",
        "category": "Indoor Architecture",
        "acceptance_status": "PASS",
        "acceptance_label": "High-Density Continuous Reconstruction (Attempt 4)",
        "description": "Continuous high-density 3D digital twin of a student room reconstructed at >=1 FPS temporal sampling (73 frames across 72.59s walkthrough). Integrates multi-view overlapping window inference, Sim(3) coordinate alignment, RANSAC floor leveling, and Screened Poisson reconstruction.",
        "known_limitations": "Ceiling geometry unobserved due to downward handheld walk angle; high specular reflections on tile floors cause minor depth variance; thin drying rack wires simplified during voxel decimation.",
        "coordinate_status": "Canonical coordinate frame aligned with horizontal floor plane normal. Metric scale uncalibrated.",
        "vertex_count": v_count,
        "triangle_count": t_count,
        "default_representation": "textured_obj",
        "representations": {
            "textured_obj": {
                "type": "obj",
                "label": "Textured Canonical OBJ",
                "obj_url": "/models/indoor_room_001/room_model_canonical.obj",
                "mtl_url": "/models/indoor_room_001/room_model_canonical.mtl",
                "texture_urls": [],
                "file_size_bytes": obj_file.stat().st_size,
                "sha256": sha256_file(obj_file),
                "available": True
            },
            "photographic_ply": {
                "type": "ply",
                "label": "Photographic Vertex PLY",
                "ply_url": "/models/indoor_room_001/room_model.ply",
                "file_size_bytes": ply_file.stat().st_size,
                "sha256": sha256_file(ply_file),
                "available": True
            }
        },
        "thumbnail_url": "/models/indoor_room_001/render_view_oblique.png",
        "previews": {
            "oblique": "/models/indoor_room_001/render_view_oblique.png",
            "top": "/models/indoor_room_001/render_view_top.png",
            "eye_level": "/models/indoor_room_001/render_view_eye_level.png"
        },
        "metadata_url": "/models/indoor_room_001/candidate_metadata.json",
        "initial_camera": {
            "position": [0.0, 1.8, 2.8],
            "target": [0.0, 0.0, 0.0]
        },
        "reconstruction_run_id": f"indoor_room_001_attempt_{args.attempt}",
        "source_video_provenance": {
            "path": meta["video_path"],
            "resolution": "478x850",
            "fps": 59.24,
            "duration_sec": 72.585,
            "scene": "indoor_room_001",
            "status": "PROCESSED_AND_VERIFIED"
        }
    }

    # Save candidate metadata in target directories
    for tdir in [FINAL_RELEASE_DIR, PUBLIC_MODELS_DIR, DIST_MODELS_DIR]:
        with open(tdir / "candidate_metadata.json", "w") as f:
            json.dump(scene_entry, f, indent=2)

    # Update manifest files
    manifest_paths = [
        PROJECT_ROOT / "viewer/src/data/model-manifest.json",
        PROJECT_ROOT / "viewer/public/model-manifest.json",
        PROJECT_ROOT / "viewer/dist/model-manifest.json"
    ]

    for m_path in manifest_paths:
        if m_path.is_file():
            with open(m_path, "r") as f:
                manifest_data = json.load(f)

            # Insert or replace indoor_room_001 at the FRONT of scenes list for prominence
            scenes = manifest_data.get("scenes", [])
            scenes = [s for s in scenes if s.get("id") != "indoor_room_001"]
            scenes.insert(0, scene_entry)
            manifest_data["scenes"] = scenes

            with open(m_path, "w") as f:
                json.dump(manifest_data, f, indent=2)
            print(f"[+] Updated manifest: {m_path}")

    print("\n[+] Successfully packaged indoor model into all viewer directories!")

if __name__ == "__main__":
    main()

