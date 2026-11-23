import json
from pathlib import Path

manifest_paths = [
    Path("viewer/public/model-manifest.json"),
    Path("viewer/src/data/model-manifest.json"),
    Path("viewer/dist/model-manifest.json")
]

new_scene = {
    "id": "clean_slate_video1_textured",
    "scene_name": "clean_slate_video1_textured",
    "display_name": "HOSTEL ROOM B 314 — CLEAN-SLATE TWIN (15mm TSDF)",
    "subtitle": "Fresh Clean-Slate Digital Twin — 54 Adaptive Keyframes + 15mm Scalable TSDF + Native 4K Atlas",
    "category": "Indoor Architecture",
    "acceptance_status": "EXPERIMENTAL_PASS",
    "acceptance_label": "CLEAN-SLATE TEXTURED DIGITAL TWIN",
    "description": "Clean-slate production digital twin reconstructed from original source video (54 curated frames, 17 windows). Eliminates door-transition residual spike via targeted adaptive curation (residual reduced from 47.5cm to 18.9cm). Features Candidate A Bounded Sim(3) alignment with mandatory translation recomputation, 15mm Scalable TSDF fusion, and native-resolution 4K photographic texture atlas. 77k vertices, 138k triangles, 0.91 cm median surface agreement, zero ghost walls.",
    "vertex_count": 77781,
    "triangle_count": 138213,
    "default_representation": "gltf_glb",
    "representations": {
        "gltf_glb": {
            "type": "glb",
            "label": "Clean-Slate Textured Twin GLB (Embedded 4K Atlas)",
            "glb_url": "/outputs/indoor/clean_reconstruction_video1_20261030/textured_tsdf_model/model_clean_textured.glb",
            "file_size_bytes": 20478648,
            "available": True
        },
        "textured_obj": {
            "type": "obj",
            "label": "Canonical Wavefront OBJ + MTL",
            "obj_url": "/outputs/indoor/clean_reconstruction_video1_20261030/textured_tsdf_model/model_clean_textured_canonical.obj",
            "mtl_url": "/outputs/indoor/clean_reconstruction_video1_20261030/textured_tsdf_model/model_clean_textured_canonical.mtl",
            "file_size_bytes": 33454710,
            "available": True
        },
        "photographic_ply": {
            "type": "ply",
            "label": "Photographic Vertex Coloured PLY",
            "ply_url": "/outputs/indoor/clean_reconstruction_video1_20261030/textured_tsdf_model/model_clean_textured.ply",
            "file_size_bytes": 3830690,
            "available": True
        }
    },
    "initial_camera": {
        "position": [0.0, 0.8, 1.8],
        "target": [-0.05, 0.17, -0.44]
    }
}

for mp in manifest_paths:
    if mp.exists():
        data = json.loads(mp.read_text(encoding="utf-8"))
        scenes = data.get("scenes", [])
        scenes = [s for s in scenes if s.get("id") != new_scene["id"]]
        scenes.insert(0, new_scene)
        data["scenes"] = scenes
        mp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"[+] Updated manifest: {mp}")

