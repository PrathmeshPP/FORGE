import json
from pathlib import Path

scene_entry = {
    "id": "indoor_vggt_tsdf_room_20261031",
    "scene_name": "indoor_vggt_tsdf_room_20261031",
    "display_name": "Indoor Room (VGGT + TSDF Fusion)",
    "subtitle": "Oct 31 Recording — 15mm TSDF Fusion & Photographic Texture Atlas",
    "category": "Indoor Architecture",
    "acceptance_status": "VERIFIED",
    "acceptance_label": "Scalable TSDF Fusion Architecture",
    "description": "Full indoor twin reconstructed from 136-second video (4,081 frames) with VGGT scale-bounded Sim(3) chaining, Open3D Scalable TSDF volume fusion, and multi-view MRF photographic texture baking (79.5% direct coverage).",
    "vertex_count": 111310,
    "triangle_count": 194731,
    "default_representation": "gltf_glb",
    "representations": {
        "gltf_glb": {
            "type": "glb",
            "label": "TSDF Textured Model (26.2 MB)",
            "glb_url": "/outputs/indoor/vggt_tsdf_video_20261031_142303/textured_tsdf_model/model_tsdf_textured.glb",
            "available": True
        },
        "textured_obj": {
            "type": "obj",
            "label": "Canonical OBJ + MTL",
            "obj_url": "/outputs/indoor/vggt_tsdf_video_20261031_142303/textured_tsdf_model/model_tsdf_textured_canonical.obj",
            "mtl_url": "/outputs/indoor/vggt_tsdf_video_20261031_142303/textured_tsdf_model/model_tsdf_textured_canonical.mtl",
            "available": True
        },
        "point_cloud_ply": {
            "type": "ply",
            "label": "Confidence-Filtered Point Cloud",
            "ply_url": "/outputs/indoor/vggt_tsdf_video_20261031_142303/point_clouds/filtered_point_cloud_leveled.ply",
            "available": True
        }
    },
    "initial_camera": {
        "position": [0.0, 0.4, 2.2],
        "target": [0.0, 0.0, 0.0]
    }
}

manifest_paths = [
    Path("viewer/public/model-manifest.json"),
    Path("viewer/src/data/model-manifest.json"),
    Path("viewer/dist/model-manifest.json")
]

for p in manifest_paths:
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        data["scenes"] = [s for s in data.get("scenes", []) if s.get("id") != scene_entry["id"]]
        data["scenes"].insert(0, scene_entry)
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Updated {p} ({len(data['scenes'])} scenes)")

