"""
Update model-manifest.json with Gate D Full Pre-TSDF Point Cloud scene
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

pub_manifest_path = PROJECT_ROOT / "viewer/public/model-manifest.json"
src_manifest_path = PROJECT_ROOT / "viewer/src/data/model-manifest.json"
dist_manifest_path = PROJECT_ROOT / "viewer/dist/model-manifest.json"

gate_d_scene = {
    "id": "gate_d_full_pre_tsdf",
    "scene_name": "gate_d_full_pre_tsdf",
    "display_name": "GATE D — FULL PRE-TSDF POINT CLOUD (136s, 59 WINS)",
    "subtitle": "Complete Full-Duration Room Pre-TSDF Validation — Candidate A Alignment (179 Frames)",
    "category": "Indoor Diagnostic & Pilot",
    "acceptance_status": "EXPERIMENTAL_PASS",
    "acceptance_label": "GATE D PRE-TSDF (CHECKPOINT)",
    "description": "Complete full-duration merged RGB point cloud across all 59 windows (179 frames, 136 seconds). Aligned using Candidate A (Bounded Sim(3) + mandatory translation recomputation + camera constraints + p20 confidence filtering). Contains 266,923 clean points spanning [2.67m x 1.56m x 3.19m] with 1.76 cm mean 3D held-out residual.",
    "vertex_count": 266923,
    "triangle_count": 0,
    "default_representation": "photographic_ply",
    "representations": {
        "photographic_ply": {
            "type": "ply",
            "label": "Full Pre-TSDF Clean Point Cloud (266k Pts, 6.87 MB)",
            "ply_url": "/outputs/indoor/distortion_investigation_20261031/gate_d_full_duration/full_pre_tsdf_point_cloud_12mm.ply",
            "file_size_bytes": 7206921,
            "available": True
        },
        "untextured_ply": {
            "type": "ply",
            "label": "Spatial Density Heatmap (256k Pts)",
            "ply_url": "/outputs/indoor/distortion_investigation_20261031/gate_d_full_duration/full_pre_tsdf_density_heatmap.ply",
            "file_size_bytes": 6934383,
            "available": True
        }
    },
    "initial_camera": {
        "position": [0.0, 1.2, 3.2],
        "target": [0.0, -0.2, 0.0]
    }
}

for manifest_p in [pub_manifest_path, src_manifest_path, dist_manifest_path]:
    if manifest_p.exists():
        data = json.loads(manifest_p.read_text())
        existing = [s for s in data["scenes"] if s["id"] != "gate_d_full_pre_tsdf"]
        data["scenes"] = [gate_d_scene] + existing
        manifest_p.write_text(json.dumps(data, indent=2))
        print(f"[+] Added Gate D scene at index 0 in {manifest_p.relative_to(PROJECT_ROOT)}")

