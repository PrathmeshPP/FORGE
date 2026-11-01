"""
Register November 1 (18:05:30) Rashtriya Raksha University Lobby Scene in F.O.R.G.E. Viewer
"""

import json
import shutil
import cv2
from pathlib import Path

EXP_DIR = Path("outputs/indoor/texture_experiment_video_20261101_180530")
PUBLIC_MODEL_DIR = Path("viewer/public/models/indoor_nov1_180530")
PUBLIC_MODEL_DIR.mkdir(parents=True, exist_ok=True)

def main():
    meta_path = EXP_DIR / "reconstruction_metadata.json"
    if not meta_path.exists():
        print(f"[!] Reconstruction metadata not yet generated at {meta_path}")
        return

    meta = json.loads(meta_path.read_text())

    # Previews
    render_front = EXP_DIR / "renders/render_04_front_corridor.png"
    render_overview = EXP_DIR / "renders/render_01_full_room_overview.png"
    render_top = EXP_DIR / "renders/render_08_top_down_floorplan.png"

    if render_front.exists():
        im = cv2.imread(str(render_front))
        cv2.imwrite(str(PUBLIC_MODEL_DIR / "preview_eye_level.png"), im)
        thumb = cv2.resize(im, (320, 180), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(PUBLIC_MODEL_DIR / "preview_thumbnail.png"), thumb)
        print("[+] Saved preview_eye_level.png and preview_thumbnail.png")

    if render_top.exists():
        shutil.copyfile(str(render_top), str(PUBLIC_MODEL_DIR / "preview_top.png"))
        print("[+] Saved preview_top.png")

    shutil.copyfile(str(meta_path), str(PUBLIC_MODEL_DIR / "metadata.json"))

    rec_m = meta["reconstruction_metrics"]
    glb_del = meta["deliverables"]["glb"]
    obj_del = meta["deliverables"]["obj"]
    ply_del = meta["deliverables"]["ply"]

    # Calculate calibrated initial camera and navigation presets
    # In canonical Three.js space:
    # Floor is at Y = rec_m['floor_level_y']
    floor_y = rec_m.get("floor_level_y", -0.8)
    eye_y = round(floor_y + 1.6, 2) # Human eye level 1.6m above floor

    SCENE_ENTRY = {
        "id": "indoor_nov1_180530_photographic",
        "scene_name": "indoor_nov1_180530_photographic",
        "display_name": "RASHTRIYA RAKSHA UNIVERSITY — NSCS LOBBY & CORRIDOR",
        "subtitle": "Ray-Oriented Screened Poisson + 8K Photographic Multi-View Texture Atlas",
        "category": "Indoor Architecture",
        "acceptance_status": "PRODUCTION_PASS",
        "acceptance_label": "FRESH INDEPENDENT DIGITAL TWIN",
        "description": "Full 3D photographic digital twin reconstructed from WhatsApp Video 2026-11-01 at 18.05.30.mp4. Captures the National Security Council Secretariat lobby, 3-seater waiting chairs, brown architectural switchboard frame, corridor pillars with Fire Exit signage, multi-story courtyard window facade, and concrete staircase well.",
        "vertex_count": rec_m["mesh_vertices"],
        "triangle_count": rec_m["mesh_triangles"],
        "default_representation": "gltf_glb",
        "representations": {
            "gltf_glb": {
                "type": "glb",
                "label": "Textured GLB (Embedded 8K Photographic Atlas)",
                "glb_url": "/outputs/indoor/texture_experiment_video_20261101_180530/room_photographic_model.glb",
                "file_size_bytes": glb_del["size_bytes"],
                "sha256": glb_del["sha256"],
                "available": True
            },
            "textured_obj": {
                "type": "obj",
                "label": "Wavefront OBJ + MTL (with vt Texture Coordinates)",
                "obj_url": "/outputs/indoor/texture_experiment_video_20261101_180530/room_photographic_model_canonical.obj",
                "mtl_url": "/outputs/indoor/texture_experiment_video_20261101_180530/room_photographic_model_canonical.mtl",
                "file_size_bytes": obj_del["size_bytes"],
                "sha256": obj_del["sha256"],
                "available": True
            },
            "photographic_ply": {
                "type": "ply",
                "label": "Photographic Vertex PLY (High-Density Mesh)",
                "ply_url": "/outputs/indoor/texture_experiment_video_20261101_180530/room_photographic_model.ply",
                "file_size_bytes": ply_del["size_bytes"],
                "sha256": ply_del["sha256"],
                "available": True
            }
        },
        "thumbnail_url": "/models/indoor_nov1_180530/preview_thumbnail.png",
        "previews": {
            "eye_level": "/models/indoor_nov1_180530/preview_eye_level.png",
            "top": "/models/indoor_nov1_180530/preview_top.png"
        },
        "metadata_url": "/models/indoor_nov1_180530/metadata.json",
        "initial_camera": {
            "position": [0.0, eye_y, -2.2],
            "target": [0.0, eye_y * 0.9, 0.5]
        },
        "camera_presets": [
            {
                "id": "entrance",
                "label": "Lobby Entrance",
                "description": "Standing at lobby entrance looking along the corridor",
                "position": [0.0, eye_y, -2.2],
                "target": [0.0, eye_y * 0.9, 0.5]
            },
            {
                "id": "waiting_chairs",
                "label": "Waiting Chairs & NSCS Banner",
                "description": "Close perspective on the 3-seater metal chairs and display wall",
                "position": [-0.8, eye_y * 0.7, -0.6],
                "target": [-1.6, eye_y * 0.6, 0.0]
            },
            {
                "id": "window_facade",
                "label": "Courtyard Window Wall",
                "description": "Facing the multi-story architectural glass facade and pillars",
                "position": [0.6, eye_y, 0.5],
                "target": [1.8, eye_y, 0.5]
            },
            {
                "id": "staircase",
                "label": "Staircase Well",
                "description": "Looking down the descending staircase with handrail and wall gallery",
                "position": [0.3, eye_y, 1.4],
                "target": [0.8, eye_y * 0.3, 2.4]
            },
            {
                "id": "overview",
                "label": "High-Angle Overview",
                "description": "Overhead inspection showing complete lobby and corridor boundaries",
                "position": [0.0, 3.2, -3.8],
                "target": [0.0, 0.0, 0.0]
            }
        ],
        "reconstruction_run_id": "texture_experiment_video_20261101_180530",
        "source_video_provenance": meta["source_video"]
    }

    manifest_targets = [
        Path("viewer/public/model-manifest.json"),
        Path("viewer/src/data/model-manifest.json"),
        Path("viewer/dist/model-manifest.json")
    ]

    for mp in manifest_targets:
        if not mp.exists():
            continue
        data = json.loads(mp.read_text())
        scenes = data.get("scenes", [])
        # Remove any existing entry with this id
        scenes = [s for s in scenes if s["id"] != "indoor_nov1_180530_photographic"]
        # Insert as premier first scene
        scenes.insert(0, SCENE_ENTRY)
        data["scenes"] = scenes
        mp.write_text(json.dumps(data, indent=2))
        print(f"[+] Updated {mp} (total scenes: {len(scenes)})")

    # Also copy preview models to viewer/dist/models/indoor_nov1_180530
    dist_dir = Path("viewer/dist/models/indoor_nov1_180530")
    dist_dir.mkdir(parents=True, exist_ok=True)
    for f in PUBLIC_MODEL_DIR.glob("*"):
        shutil.copyfile(f, dist_dir / f.name)
    print("[+] Synced dist preview models.")

if __name__ == "__main__":
    main()
