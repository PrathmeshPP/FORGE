import json
import shutil
import cv2
from pathlib import Path

cand_c_dir = Path("outputs/indoor/photographic_texture_and_geometry_quality_recovery_nov1/candidate_c_combined_object_aware")
comp_dir = Path("outputs/indoor/photographic_texture_and_geometry_quality_recovery_nov1/comparisons")
public_model_dir = Path("viewer/public/models/indoor_nov1_candidate_c")
public_model_dir.mkdir(parents=True, exist_ok=True)

# 1. Prepare preview images
eye_render = comp_dir / "candidate_c_front_aisle.png"
if eye_render.exists():
    im = cv2.imread(str(eye_render))
    cv2.imwrite(str(public_model_dir / "preview_eye_level.png"), im)
    thumb = cv2.resize(im, (320, 180), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(public_model_dir / "preview_thumbnail.png"), thumb)
    print("Saved preview_eye_level.png and preview_thumbnail.png")

top_render = comp_dir / "candidate_c_top_overview.png"
if top_render.exists():
    shutil.copyfile(str(top_render), str(public_model_dir / "preview_top.png"))
    print("Saved preview_top.png")

metadata = {
    "scene_id": "indoor_nov1_candidate_c_photographic",
    "display_name": "HOSTEL ROOM B 314 — CANDIDATE C (OBJECT-AWARE 8K TWIN)",
    "vertex_count": 224998,
    "triangle_count": 450000,
    "source_video": "WhatsApp Video 2026-11-01 at 14.28.11.mp4",
    "sha256": "3b250d6e785cbbec86d78f550e62f004d70886ddddf0d1563ea1aa6700ee3454",
    "file_size_bytes": 62426468,
    "atlas_resolution": [8192, 8192],
    "atlas_sha256": "d88e6ff1eb6e5dbf069aa2fe728d5e2683d966e69e62c0437793c1e6014f7d54"
}
with open(public_model_dir / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

NEW_SCENE = {
    "id": "indoor_nov1_candidate_c_photographic",
    "scene_name": "indoor_nov1_candidate_c_photographic",
    "display_name": "HOSTEL ROOM B 314 — CANDIDATE C (OBJECT-AWARE 8K TWIN)",
    "subtitle": "Ray-Oriented Poisson (450k Tris) + Object-Aware Semantic Recognition + 8K Multi-View Blended Atlas",
    "category": "Indoor Architecture",
    "acceptance_status": "PRODUCTION_PASS",
    "acceptance_label": "RECOVERED PHOTOGRAPHIC DIGITAL TWIN",
    "description": "Advanced photographic 3D digital twin from WhatsApp Video 2026-11-01 at 14.28.11.mp4. Corrects 483,932 inverted normals via per-point optical ray tracking, doubles geometric fidelity to 450,000 triangles without punching holes in floor or furniture surfaces, and applies an 8192x8192 super-sampled photographic texture atlas with object-aware semantic weighting and multi-view exposure blending.",
    "vertex_count": 224998,
    "triangle_count": 450000,
    "default_representation": "gltf_glb",
    "representations": {
        "gltf_glb": {
            "type": "glb",
            "label": "Candidate C GLB (Embedded 8K Atlas, 450k Faces)",
            "glb_url": "/outputs/indoor/photographic_texture_and_geometry_quality_recovery_nov1/candidate_c_combined_object_aware/candidate_c_photographic_room.glb",
            "file_size_bytes": 62426468,
            "sha256": "3b250d6e785cbbec86d78f550e62f004d70886ddddf0d1563ea1aa6700ee3454",
            "available": True
        },
        "textured_obj": {
            "type": "obj",
            "label": "Candidate C Wavefront OBJ + MTL (with vt Coordinates)",
            "obj_url": "/outputs/indoor/photographic_texture_and_geometry_quality_recovery_nov1/candidate_c_combined_object_aware/candidate_c_photographic_room_canonical.obj",
            "mtl_url": "/outputs/indoor/photographic_texture_and_geometry_quality_recovery_nov1/candidate_c_combined_object_aware/candidate_c_photographic_room_canonical.mtl",
            "file_size_bytes": 56660618,
            "sha256": "9347b86f3f4a31a8b6142905f752100a26af8eff8ae4e99d42d614bbddfc40b3",
            "available": True
        },
        "photographic_ply": {
            "type": "ply",
            "label": "Candidate C Photographic Vertex PLY (450k Faces)",
            "ply_url": "/outputs/indoor/photographic_texture_and_geometry_quality_recovery_nov1/candidate_c_combined_object_aware/candidate_c_photographic_room.ply",
            "file_size_bytes": 17325224,
            "sha256": "ff785a383e29d9f5fe0ff27b21874f79d3fa3b04c223a70f0c9342249b94e594",
            "available": True
        }
    },
    "thumbnail_url": "/models/indoor_nov1_candidate_c/preview_thumbnail.png",
    "previews": {
        "eye_level": "/models/indoor_nov1_candidate_c/preview_eye_level.png",
        "top": "/models/indoor_nov1_candidate_c/preview_top.png"
    },
    "metadata_url": "/models/indoor_nov1_candidate_c/metadata.json",
    "initial_camera": {
        "position": [0.0, 0.29, -2.40],
        "target": [0.0, 0.15, 0.50]
    },
    "camera_presets": [
        {
            "id": "entrance",
            "label": "Entrance View",
            "description": "Standing at room entrance facing inward along central aisle",
            "position": [0.0, 0.29, -2.40],
            "target": [0.0, 0.15, 0.50]
        },
        {
            "id": "left_cot",
            "label": "Left Cot & Desk",
            "description": "Perspective focused on left cot and study desk",
            "position": [-0.60, 0.20, -0.20],
            "target": [-1.65, -0.10, 0.10]
        },
        {
            "id": "right_cot",
            "label": "Right Cot & Wardrobe",
            "description": "Perspective focused on right cot and metal wardrobes",
            "position": [0.60, 0.20, -0.20],
            "target": [1.65, -0.10, 0.10]
        },
        {
            "id": "overview",
            "label": "Room Overview",
            "description": "High-angle overview showing room bounds and ceiling",
            "position": [0.0, 2.40, -3.20],
            "target": [0.0, 0.10, 0.0]
        }
    ],
    "reconstruction_run_id": "photographic_texture_and_geometry_quality_recovery_nov1",
    "source_video_provenance": {
        "path": "C:\\Users\\25beevdt025\\Downloads\\WhatsApp Video 2026-11-01 at 14.28.11.mp4",
        "duration_sec": 73.13,
        "resolution": [848, 480],
        "fps": 25.0,
        "sha256": "1c02de602ff34bd63d7332f662dfb6fb73de5676a05ca07c2704f1fbe6c6ce35"
    }
}

manifest_paths = [
    Path("viewer/public/model-manifest.json"),
    Path("viewer/src/data/model-manifest.json"),
    Path("viewer/dist/model-manifest.json")
]

for p in manifest_paths:
    if not p.exists():
        continue
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["scenes"] = [s for s in data["scenes"] if s.get("id") != "indoor_nov1_candidate_c_photographic"]
    data["scenes"].insert(0, NEW_SCENE)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Updated {p} (total scenes: {len(data['scenes'])})")
