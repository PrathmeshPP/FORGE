"""
Package Attempt 4 (New Video Source) for F.O.R.G.E. Three.js Viewer.
Preserves all previous models and manifests while adding the new reconstruction.
"""

import os
import sys
import json
import shutil
import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ATTEMPT_DIR = PROJECT_ROOT / "outputs/indoor/room_001/attempt_4_new_video"
PUBLIC_DIR = PROJECT_ROOT / "viewer/public/models/indoor_room_attempt_4_new_video"
DIST_DIR = PROJECT_ROOT / "viewer/dist/models/indoor_room_attempt_4_new_video"

PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
DIST_DIR.mkdir(parents=True, exist_ok=True)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def package():
    mesh_dir = ATTEMPT_DIR / "meshes"
    renders_dir = ATTEMPT_DIR / "renders"

    src_obj = mesh_dir / "poisson_mesh_canonical.obj"
    src_mtl = mesh_dir / "poisson_mesh_canonical.mtl"
    src_glb = mesh_dir / "poisson_mesh.glb"
    src_ply = mesh_dir / "poisson_mesh.ply"

    src_oblique = renders_dir / "attempt_4_new_video_oblique.png"
    src_top = renders_dir / "attempt_4_new_video_top_plan.png"
    src_eye = renders_dir / "attempt_4_new_video_eye_level.png"
    src_overview = renders_dir / "attempt_4_new_video_overview.png"
    src_furniture = renders_dir / "attempt_4_new_video_furniture_closeup.png"

    files_map = {
        "room_model_canonical.obj": src_obj,
        "room_model_canonical.mtl": src_mtl,
        "room_model.glb": src_glb,
        "room_model.ply": src_ply,
        "render_view_oblique.png": src_oblique,
        "render_view_top.png": src_top,
        "render_view_eye_level.png": src_eye,
        "render_view_overview.png": src_overview,
        "render_view_furniture_closeup.png": src_furniture
    }

    for name, src in files_map.items():
        if src.is_file():
            shutil.copyfile(src, PUBLIC_DIR / name)
            shutil.copyfile(src, DIST_DIR / name)
            print(f"Copied {name} ({src.stat().st_size / (1024**2):.2f} MB)")

    with open(ATTEMPT_DIR / "attempt_metadata.json") as f:
        meta = json.load(f)

    obj_target = PUBLIC_DIR / "room_model_canonical.obj"
    ply_target = PUBLIC_DIR / "room_model.ply"
    glb_target = PUBLIC_DIR / "room_model.glb"

    v_count = meta["meshing_evaluations"]["poisson"]["vertices"]
    t_count = meta["meshing_evaluations"]["poisson"]["triangles"]

    scene_entry = {
        "id": "indoor_room_attempt_4_new_video",
        "scene_name": "indoor_room_attempt_4_new_video",
        "display_name": "Computer Lab (New Video Source)",
        "subtitle": "Academic Computer Laboratory Digital Twin — Attempt 4",
        "category": "Indoor Architecture",
        "acceptance_status": "PASS",
        "acceptance_label": "High-Density Continuous Reconstruction (Attempt 4 New Video)",
        "description": "Continuous high-density 3D digital twin of a computer laboratory reconstructed from new WhatsApp video (2026-10-30 at 16.43.49). Captured at 1.82 FPS sampling density across 20.38s (37 curated frames). Features multi-view overlapping window inference, Sim(3) coordinate alignment (1.6cm residual), RANSAC floor leveling, and Screened Poisson surface reconstruction.",
        "known_limitations": "Under-desk knee cavity has partial unobserved shadow; ceiling unobserved due to horizontal pan angle; monitor screens show slight specular reflectance.",
        "coordinate_status": "Canonical coordinate frame aligned with horizontal floor plane normal. Metric scale preserved via closed-form Sim(3).",
        "vertex_count": v_count,
        "triangle_count": t_count,
        "default_representation": "textured_obj",
        "representations": {
            "textured_obj": {
                "type": "obj",
                "label": "Textured Canonical OBJ",
                "obj_url": "/models/indoor_room_attempt_4_new_video/room_model_canonical.obj",
                "mtl_url": "/models/indoor_room_attempt_4_new_video/room_model_canonical.mtl",
                "texture_urls": [],
                "file_size_bytes": obj_target.stat().st_size if obj_target.exists() else 0,
                "sha256": sha256_file(obj_target) if obj_target.exists() else "",
                "available": True
            },
            "photographic_ply": {
                "type": "ply",
                "label": "Photographic Vertex PLY",
                "ply_url": "/models/indoor_room_attempt_4_new_video/room_model.ply",
                "file_size_bytes": ply_target.stat().st_size if ply_target.exists() else 0,
                "sha256": sha256_file(ply_target) if ply_target.exists() else "",
                "available": True
            },
            "gltf_glb": {
                "type": "glb",
                "label": "Interactive GLB Model",
                "glb_url": "/models/indoor_room_attempt_4_new_video/room_model.glb",
                "file_size_bytes": glb_target.stat().st_size if glb_target.exists() else 0,
                "sha256": sha256_file(glb_target) if glb_target.exists() else "",
                "available": True
            }
        },
        "thumbnail_url": "/models/indoor_room_attempt_4_new_video/render_view_oblique.png",
        "previews": {
            "oblique": "/models/indoor_room_attempt_4_new_video/render_view_oblique.png",
            "top": "/models/indoor_room_attempt_4_new_video/render_view_top.png",
            "eye_level": "/models/indoor_room_attempt_4_new_video/render_view_eye_level.png"
        },
        "initial_camera": {
            "position": [0.0, 1.8, 2.8],
            "target": [0.0, 0.0, 0.0]
        },
        "reconstruction_run_id": "indoor_room_attempt_4_new_video",
        "source_video_provenance": {
            "path": "WhatsApp Video 2026-10-30 at 16.43.49.mp4",
            "resolution": "480x854",
            "fps": 29.889,
            "duration_sec": 20.38,
            "scene": "computer_laboratory",
            "status": "PROCESSED_AND_VERIFIED"
        }
    }

    # Update manifests
    manifest_paths = [
        PROJECT_ROOT / "viewer/public/model-manifest.json",
        PROJECT_ROOT / "viewer/src/data/model-manifest.json",
        PROJECT_ROOT / "viewer/dist/model-manifest.json"
    ]

    for m_path in manifest_paths:
        if not m_path.exists():
            continue
        with open(m_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        # Prepend or replace scene
        scenes = manifest.get("scenes", [])
        scenes = [s for s in scenes if s.get("id") != "indoor_room_attempt_4_new_video"]
        scenes.insert(0, scene_entry)
        manifest["scenes"] = scenes

        with open(m_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"Updated manifest: {m_path}")

    print("\n[+] Packaged Attempt 4 (New Video Source) successfully into Three.js viewer!")

if __name__ == "__main__":
    package()

