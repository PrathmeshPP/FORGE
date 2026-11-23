import json
from pathlib import Path

manifest_paths = [
    Path("viewer/public/model-manifest.json"),
    Path("viewer/src/data/model-manifest.json"),
    Path("viewer/dist/model-manifest.json")
]

new_scene = {
    "id": "video1_vggt_tsdf_candidate_a_textured",
    "scene_name": "video1_vggt_tsdf_candidate_a_textured",
    "display_name": "HOSTEL ROOM B 314 — COMPLETE TEXTURED TWIN (40s)",
    "subtitle": "Complete Digital Twin — 15mm TSDF Mesh + Native 4K Photographic Texture Atlas",
    "category": "Indoor Architecture",
    "acceptance_status": "EXPERIMENTAL_PASS",
    "acceptance_label": "VIDEO 1 TEXTURED DIGITAL TWIN",
    "description": "Complete production digital twin across the 40-second indoor video (63 frames, 20 windows). Features Candidate A Bounded Sim(3) alignment with mandatory translation recomputation, 15mm Scalable TSDF fusion, and native-resolution 4K photographic texture atlas. 86k vertices, 154k triangles, 0.94 cm median surface agreement, zero ghost walls.",
    "vertex_count": 86048,
    "triangle_count": 154763,
    "default_representation": "gltf_glb",
    "representations": {
        "gltf_glb": {
            "type": "glb",
            "label": "Textured Digital Twin GLB (Embedded 4K Atlas)",
            "glb_url": "/outputs/indoor/vggt_tsdf_video_20261030_170234/textured_tsdf_model/model_video1_textured.glb",
            "file_size_bytes": 22072524,
            "available": True
        },
        "textured_obj": {
            "type": "obj",
            "label": "Canonical Wavefront OBJ + MTL",
            "obj_url": "/outputs/indoor/vggt_tsdf_video_20261030_170234/textured_tsdf_model/model_video1_textured_canonical.obj",
            "mtl_url": "/outputs/indoor/vggt_tsdf_video_20261030_170234/textured_tsdf_model/model_video1_textured_canonical.mtl",
            "file_size_bytes": 37444710,
            "available": True
        },
        "photographic_ply": {
            "type": "ply",
            "label": "Photographic Vertex Coloured PLY",
            "ply_url": "/outputs/indoor/vggt_tsdf_video_20261030_170234/textured_tsdf_model/model_video1_textured.ply",
            "file_size_bytes": 4330690,
            "available": True
        }
    },
    "initial_camera": {
        "position": [0.0, 0.8, 1.8],
        "target": [-0.07, 0.06, -0.61]
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
