"""
Register November 1 (19:10:43) V3 Quality Scene in F.O.R.G.E. Viewer
"""

import json
import shutil
import cv2
from pathlib import Path

EXP_DIR = Path("outputs/indoor/texture_experiment_video_20261101_191043_v3_quality")
PUBLIC_MODEL_DIR = Path("viewer/public/models/indoor_nov1_191043_v3")
PUBLIC_MODEL_DIR.mkdir(parents=True, exist_ok=True)

def main():
    meta_path = EXP_DIR / "reconstruction_metadata.json"
    if not meta_path.exists():
        print(f"[!] Reconstruction metadata not found at {meta_path}")
        return

    meta = json.loads(meta_path.read_text())

    # Previews
    render_front = EXP_DIR / "renders/render_01_full_room_overview.png"
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

    geom = meta["geometry"]
    deliv = meta["deliverables"]
    glb_del = deliv["glb"]
    obj_del = deliv["obj"]
    ply_del = deliv["ply"]

    floor_y = -0.8
    eye_y = round(floor_y + 1.6, 2)

    glb_file_p = EXP_DIR / "room_photographic_model.glb"
    obj_file_p = EXP_DIR / "room_photographic_model_canonical.obj"
    ply_file_p = EXP_DIR / "room_photographic_model.ply"

    SCENE_ENTRY = {
        "id": "indoor_nov1_191043_photographic_v3",
        "scene_name": "indoor_nov1_191043_photographic_v3",
        "display_name": "NOVEMBER 1 \u2014 19:10:43 INDOOR DIGITAL TWIN (V3 QUALITY)",
        "subtitle": "Hostel Room B 314 (Dense 8mm Geometry & Aspect-Preserved 8K Atlas)",
        "category": "Indoor Architecture",
        "acceptance_status": "PRODUCTION_PASS",
        "acceptance_label": "V3 COMPLETE 3D QUALITY IMPROVEMENT",
        "description": "High-fidelity V3 digital twin of Hostel Room B 314 from WhatsApp Video 2026-11-01 at 19.10.43.mp4. Reconstructed with high-density 8mm point cloud backprojection (294,087 structural points, 4.26x retention increase), ray-oriented normal consensus, depth=10 Screened Poisson meshing (599,999 triangles, boundary edges reduced to 4,105), depth-occlusion gating, 4-pass topological chart consolidation, and aspect-preserving 8K photographic atlas with zero distortion.",
        "vertex_count": geom["vertices"],
        "triangle_count": geom["triangles"],
        "default_representation": "gltf_glb",
        "representations": {
            "gltf_glb": {
                "type": "glb",
                "label": "V3 Textured GLB (Dense 8mm Mesh + 8K Atlas)",
                "glb_url": "/outputs/indoor/texture_experiment_video_20261101_191043_v3_quality/room_photographic_model.glb",
                "file_size_bytes": glb_file_p.stat().st_size,
                "sha256": glb_del["sha256"],
                "available": True
            },
            "textured_obj": {
                "type": "obj",
                "label": "V3 Wavefront OBJ + MTL (Dense Geometry + Precise UVs)",
                "obj_url": "/outputs/indoor/texture_experiment_video_20261101_191043_v3_quality/room_photographic_model_canonical.obj",
                "mtl_url": "/outputs/indoor/texture_experiment_video_20261101_191043_v3_quality/room_photographic_model_canonical.mtl",
                "file_size_bytes": obj_file_p.stat().st_size,
                "sha256": obj_del["sha256"],
                "available": True
            },
            "photographic_ply": {
                "type": "ply",
                "label": "V3 Photographic Vertex PLY (300k Vertices)",
                "ply_url": "/outputs/indoor/texture_experiment_video_20261101_191043_v3_quality/room_photographic_model.ply",
                "file_size_bytes": ply_file_p.stat().st_size,
                "sha256": ply_del["sha256"],
                "available": True
            }
        },
        "thumbnail_url": "/models/indoor_nov1_191043_v3/preview_thumbnail.png",
        "previews": {
            "eye_level": "/models/indoor_nov1_191043_v3/preview_eye_level.png",
            "top": "/models/indoor_nov1_191043_v3/preview_top.png"
        },
        "metadata_url": "/models/indoor_nov1_191043_v3/metadata.json",
        "initial_camera": {
            "position": [0.0, eye_y, -2.4],
            "target": [0.0, eye_y * 0.9, 0.5]
        },
        "camera_presets": [
            {
                "id": "entrance",
                "label": "Room Entrance",
                "description": "Standing at the entrance doorway looking into Hostel Room B 314",
                "position": [0.0, eye_y, -2.4],
                "target": [0.0, eye_y * 0.9, 0.5]
            },
            {
                "id": "bed1_wardrobe",
                "label": "Bed 1 & Dual Steel Wardrobes",
                "description": "Crisp view of Bed 1 floral mattress and twin steel wardrobes with suitcases",
                "position": [-0.8, eye_y * 0.8, -0.6],
                "target": [-1.6, eye_y * 0.7, 0.0]
            },
            {
                "id": "bed2_closeup",
                "label": "Bed 2 & Maroon Mattress",
                "description": "Perspective facing the second single bed with maroon floral mattress and wall switchboard",
                "position": [0.6, eye_y * 0.8, -0.4],
                "target": [1.6, eye_y * 0.7, 0.0]
            },
            {
                "id": "bed3_cooler",
                "label": "Bed 3 & Air Cooler",
                "description": "Perspective looking towards Bed 3, blue clothes drying rack, and evaporative cooler",
                "position": [0.3, eye_y * 0.8, 1.2],
                "target": [0.8, eye_y * 0.7, 2.2]
            },
            {
                "id": "overview",
                "label": "High-Angle Overview",
                "description": "Overhead inspection showing complete hostel room boundaries and furnishings",
                "position": [0.0, 3.4, -3.8],
                "target": [0.0, 0.0, 0.0]
            }
        ],
        "reconstruction_run_id": "texture_experiment_video_20261101_191043_v3_quality",
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
        scenes = [s for s in scenes if s["id"] != "indoor_nov1_191043_photographic_v3"]
        scenes.insert(0, SCENE_ENTRY)
        data["scenes"] = scenes
        mp.write_text(json.dumps(data, indent=2))
        print(f"[+] Registered scene in {mp} (total scenes: {len(scenes)})")

    dist_dir = Path("viewer/dist/models/indoor_nov1_191043_v3")
    dist_dir.mkdir(parents=True, exist_ok=True)
    for f in PUBLIC_MODEL_DIR.glob("*"):
        shutil.copyfile(f, dist_dir / f.name)
    print("[+] Synced dist preview models for V3.")

if __name__ == "__main__":
    main()

