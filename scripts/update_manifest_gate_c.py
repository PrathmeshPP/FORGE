"""
Register Gate C models into model-manifest.json:
- BASELINE — DEFECTIVE
- CANDIDATE A — CORRECTED SIM(3)
- CANDIDATE B — RIGID SE(3)
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

pub_manifest_path = PROJECT_ROOT / "viewer/public/model-manifest.json"
src_manifest_path = PROJECT_ROOT / "viewer/src/data/model-manifest.json"

gate_c_scenes = [
    {
        "id": "gate_c_candidate_a",
        "scene_name": "gate_c_candidate_a",
        "display_name": "CANDIDATE A — CORRECTED SIM(3)",
        "subtitle": "Gate C Extended (36 Frames, 11 Windows) — Bounded Sim(3) + Mandatory Recomputation",
        "category": "Indoor Architecture",
        "acceptance_status": "EXPERIMENTAL_PASS",
        "acceptance_label": "CANDIDATE A (OPTIMAL)",
        "description": "Gate C Extended Pilot Model (frames 0..35, 11 windows). Evaluates Bounded Sim(3) with camera constraints (w_cam=10.0), p20 confidence threshold (2.5205), and mandatory translation recomputation (t_new = mu_Y - s*R*mu_X). Achieves 2.27 cm mean held-out error, 0.93 cm median TSDF surface distance, and zero parallel ghost wall artifacts.",
        "vertex_count": 23727,
        "triangle_count": 41582,
        "default_representation": "gltf_glb",
        "representations": {
            "gltf_glb": {
                "type": "glb",
                "label": "TSDF Mesh GLB (0.84 MB)",
                "glb_url": "/outputs/indoor/distortion_investigation_20261031/gate_c_extended/tsdf_mesh_candidate_A_bounded_sim3.glb",
                "file_size_bytes": 880628,
                "available": True
            },
            "photographic_ply": {
                "type": "ply",
                "label": "TSDF Point Cloud (22.9k Pts)",
                "ply_url": "/outputs/indoor/distortion_investigation_20261031/gate_c_extended/tsdf_pcd_candidate_A_bounded_sim3.ply",
                "file_size_bytes": 1173980,
                "available": True
            },
            "untextured_ply": {
                "type": "ply",
                "label": "Pre-TSDF RGB Point Cloud (72.8k Pts)",
                "ply_url": "/outputs/indoor/distortion_investigation_20261031/gate_c_extended/pre_tsdf_pcd_candidate_A_bounded_sim3.ply",
                "file_size_bytes": 1967220,
                "available": True
            }
        },
        "initial_camera": {
            "position": [0.0, 0.7, 1.6],
            "target": [0.0, -0.1, 0.0]
        }
    },
    {
        "id": "gate_c_baseline_defective",
        "scene_name": "gate_c_baseline_defective",
        "display_name": "BASELINE — DEFECTIVE",
        "subtitle": "Gate C Extended — Original Scale Clamping without Translation Recomputation",
        "category": "Indoor Architecture",
        "acceptance_status": "QA FAILED",
        "acceptance_label": "BASELINE (DEFECTIVE)",
        "description": "Gate C baseline with uncorrected translation shift upon scale clamping. Exhibits 5.06 cm mean held-out error, 7.76 cm 95th-percentile surface tail, and visible parallel ghost wall doubling due to centroid displacement.",
        "vertex_count": 26067,
        "triangle_count": 46316,
        "default_representation": "gltf_glb",
        "representations": {
            "gltf_glb": {
                "type": "glb",
                "label": "TSDF Mesh GLB (0.93 MB)",
                "glb_url": "/outputs/indoor/distortion_investigation_20261031/gate_c_extended/tsdf_mesh_baseline_defective.glb",
                "file_size_bytes": 975876,
                "available": True
            },
            "photographic_ply": {
                "type": "ply",
                "label": "TSDF Point Cloud (25.1k Pts)",
                "ply_url": "/outputs/indoor/distortion_investigation_20261031/gate_c_extended/tsdf_pcd_baseline_defective.ply",
                "file_size_bytes": 1284000,
                "available": True
            },
            "untextured_ply": {
                "type": "ply",
                "label": "Pre-TSDF RGB Point Cloud (73.6k Pts)",
                "ply_url": "/outputs/indoor/distortion_investigation_20261031/gate_c_extended/pre_tsdf_pcd_baseline_defective.ply",
                "file_size_bytes": 1987740,
                "available": True
            }
        },
        "initial_camera": {
            "position": [0.0, 0.7, 1.6],
            "target": [0.0, -0.1, 0.0]
        }
    },
    {
        "id": "gate_c_candidate_b",
        "scene_name": "gate_c_candidate_b",
        "display_name": "CANDIDATE B — RIGID SE(3)",
        "subtitle": "Gate C Extended — Rigid SE(3) s=1.000 with Mandatory Recomputation",
        "category": "Indoor Architecture",
        "acceptance_status": "DIAGNOSTIC",
        "acceptance_label": "CANDIDATE B (RIGID SE3)",
        "description": "Gate C Candidate B with forced s=1.000 rigid SE(3). Violates natural local scale variations, leading to 12.05 cm error spike in Window 6 and 4.31 cm mean held-out error.",
        "vertex_count": 26724,
        "triangle_count": 45448,
        "default_representation": "gltf_glb",
        "representations": {
            "gltf_glb": {
                "type": "glb",
                "label": "TSDF Mesh GLB (0.93 MB)",
                "glb_url": "/outputs/indoor/distortion_investigation_20261031/gate_c_extended/tsdf_mesh_candidate_B_rigid_se3.glb",
                "file_size_bytes": 975876,
                "available": True
            },
            "photographic_ply": {
                "type": "ply",
                "label": "TSDF Point Cloud (25.3k Pts)",
                "ply_url": "/outputs/indoor/distortion_investigation_20261031/gate_c_extended/tsdf_pcd_candidate_B_rigid_se3.ply",
                "file_size_bytes": 1295280,
                "available": True
            },
            "untextured_ply": {
                "type": "ply",
                "label": "Pre-TSDF RGB Point Cloud (75.8k Pts)",
                "ply_url": "/outputs/indoor/distortion_investigation_20261031/gate_c_extended/pre_tsdf_pcd_candidate_B_rigid_se3.ply",
                "file_size_bytes": 2046870,
                "available": True
            }
        },
        "initial_camera": {
            "position": [0.0, 0.7, 1.6],
            "target": [0.0, -0.1, 0.0]
        }
    }
]

for manifest_p in [pub_manifest_path, src_manifest_path]:
    data = json.loads(manifest_p.read_text())
    # Filter out any existing gate_c scenes
    existing = [s for s in data["scenes"] if not s["id"].startswith("gate_c_")]
    data["scenes"] = gate_c_scenes + existing
    manifest_p.write_text(json.dumps(data, indent=2))
    print(f"[+] Updated {manifest_p.relative_to(PROJECT_ROOT)} with 3 Gate C scenes at index 0..2")

# Also update dist manifest if dist exists
dist_manifest_p = PROJECT_ROOT / "viewer/dist/model-manifest.json"
if dist_manifest_p.exists():
    data = json.loads(dist_manifest_p.read_text())
    existing = [s for s in data["scenes"] if not s["id"].startswith("gate_c_")]
    data["scenes"] = gate_c_scenes + existing
    dist_manifest_p.write_text(json.dumps(data, indent=2))
    print(f"[+] Updated {dist_manifest_p.relative_to(PROJECT_ROOT)}")

