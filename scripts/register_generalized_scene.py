"""
F.O.R.G.E. — Registers a generalized reconstruction scene into viewer/public/model-manifest.json
Attaches scene-specific navigation metadata, floor bounds, spawn coordinates, and model paths.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

def register_scene(exp_dir: str | Path, scene_id: str, display_name: str, subtitle: str):
    exp_dir = Path(exp_dir)
    manifest_path = Path("viewer/public/model-manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    meta_path = exp_dir / "reconstruction_metadata.json"
    nav_path = exp_dir / "scene_navigation_metadata.json"

    meta = json.load(open(meta_path, "r", encoding="utf-8")) if meta_path.exists() else {}
    nav_meta = json.load(open(nav_path, "r", encoding="utf-8")) if nav_path.exists() else {}

    glb_name = "room_photographic_model.glb"
    obj_name = "room_photographic_model_canonical.obj"
    mtl_name = "room_photographic_model_canonical.mtl"
    ply_name = "room_photographic_model.ply"

    glb_file = exp_dir / glb_name
    glb_size = glb_file.stat().st_size if glb_file.exists() else 0

    exp_rel = f"/outputs/indoor/{exp_dir.name}"

    scene_entry = {
        "id": scene_id,
        "scene_name": scene_id,
        "display_name": display_name,
        "subtitle": subtitle,
        "category": "Generalized Indoor Architecture",
        "acceptance_status": "PRODUCTION_PASS" if nav_meta.get("route_validation", {}).get("eligibility_status") == "WALKTHROUGH_READY" else "PARTIAL_PASS",
        "acceptance_label": f"GENERALIZED ADAPTIVE PIPELINE ({nav_meta.get('route_validation', {}).get('eligibility_status', 'EVALUATED')})",
        "description": f"Reconstructed via generalized adaptive F.O.R.G.E. pipeline. {meta.get('geometry', {}).get('triangles', 0):,} triangles, {meta.get('texturing', {}).get('photographic_coverage_pct', 0)}% photographic coverage.",
        "vertex_count": meta.get("geometry", {}).get("vertices", 0),
        "triangle_count": meta.get("geometry", {}).get("triangles", 0),
        "default_representation": "gltf_glb",
        "representations": {
            "gltf_glb": {
                "type": "glb",
                "label": "Textured GLB (Generalized Adaptive)",
                "glb_url": f"{exp_rel}/{glb_name}",
                "file_size_bytes": glb_size,
                "available": glb_file.exists()
            },
            "textured_obj": {
                "type": "obj",
                "label": "Wavefront OBJ + MTL",
                "obj_url": f"{exp_rel}/{obj_name}",
                "mtl_url": f"{exp_rel}/{mtl_name}",
                "available": (exp_dir / obj_name).exists()
            },
            "photographic_ply": {
                "type": "ply",
                "label": "Photographic Vertex PLY",
                "ply_url": f"{exp_rel}/{ply_name}",
                "available": (exp_dir / ply_name).exists()
            }
        },
        "initial_camera": {
            "position": [0.0, nav_meta.get("eye_height", 0.5) + 0.3, -1.8],
            "target": [0.0, nav_meta.get("floor_y", 0.0) + 0.3, 0.5]
        },
        "reconstruction_run_id": exp_dir.name,
        "source_video_provenance": meta.get("source_video", {}),
        "navigation_metadata": nav_meta,
        "walkthrough_ready": (nav_meta.get("route_validation", {}).get("eligibility_status") == "WALKTHROUGH_READY")
    }

    # Replace existing or append
    scenes = manifest.get("scenes", [])
    found = False
    for i, s in enumerate(scenes):
        if s["id"] == scene_id:
            scenes[i] = scene_entry
            found = True
            break
    if not found:
        scenes.insert(0, scene_entry)  # place at top of list

    manifest["scenes"] = scenes
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[+] Scene {scene_id} registered into {manifest_path} (Total scenes: {len(scenes)})")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python register_generalized_scene.py <exp_dir> <scene_id> <display_name> <subtitle>")
        sys.exit(1)
    register_scene(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
