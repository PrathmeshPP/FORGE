"""
Register My Movie 4 Adaptive View-Dependent Scene in F.O.R.G.E. Viewer Manifests
"""

import json
import hashlib
import shutil
from pathlib import Path
import cv2

EXP_DIR = Path("outputs/indoor/dataset_my_movie_4_adaptive")
PUBLIC_MODEL_DIR = Path("viewer/public/models/indoor_my_movie_4_adaptive")
PUBLIC_MODEL_DIR.mkdir(parents=True, exist_ok=True)

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def main():
    meta_path = EXP_DIR / "reconstruction_metadata.json"
    if not meta_path.exists():
        print(f"[!] Metadata not found at {meta_path}")
        return

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    
    # Previews
    render_front = EXP_DIR / "renders/render_01_overview_lod0.png"
    render_top = EXP_DIR / "renders/render_04_top_down_lod0.png"

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

    # File paths & hashes
    p_lod0_glb = EXP_DIR / "room_photographic_model_lod0.glb"
    p_lod1_glb = EXP_DIR / "room_photographic_model_lod1.glb"
    p_lod2_glb = EXP_DIR / "room_photographic_model_lod2.glb"
    p_lod0_obj = EXP_DIR / "room_photographic_model_lod0.obj"
    p_lod0_mtl = EXP_DIR / "room_photographic_model_lod0.mtl"
    p_lod0_ply = EXP_DIR / "room_photographic_model_lod0.ply"
    p_chunks_json = EXP_DIR / "adaptive_spatial_chunks.json"

    sha_lod0_glb = compute_sha256(p_lod0_glb)
    sha_lod1_glb = compute_sha256(p_lod1_glb)
    sha_lod2_glb = compute_sha256(p_lod2_glb)
    sha_lod0_obj = compute_sha256(p_lod0_obj)
    sha_lod0_ply = compute_sha256(p_lod0_ply)
    sha_chunks = compute_sha256(p_chunks_json)

    SCENE_ENTRY = {
        "id": "indoor_my_movie_4_adaptive",
        "scene_name": "indoor_my_movie_4_adaptive",
        "display_name": "MY MOVIE 4 — ADAPTIVE 3D DIGITAL TWIN (VIEW-DEPENDENT LOD)",
        "subtitle": "Hierarchical Multi-Resolution LOD 0/1/2, Dynamic Frame Budgeting & Spatial Culling",
        "category": "Indoor Architecture",
        "acceptance_status": "PRODUCTION_PASS",
        "acceptance_label": "ADAPTIVE GPU UPGRADE",
        "description": "Next-generation adaptive view-dependent 3D reconstruction of indoor environment from My movie 4.mp4 (421s, 1080x1920) with VGGT-Omega windowed neural poses, 8mm scale-aware point cloud backprojection (337,439 structural points), floor plane leveling, ray-oriented normal consensus, and depth=10 Screened Poisson meshing. Features hierarchical multi-resolution LODs (LOD 0: 549,999 tris + 8K atlas, LOD 1: 250,000 tris + 4K atlas, LOD 2: 79,999 tris + 2K atlas), 4 spatial quadrant chunks with distance/view-angle screen-space error (SSE) evaluation, frustum culling, dynamic frame-budget throttling, and real-time performance HUD.",
        "vertex_count": 273591,
        "triangle_count": 549999,
        "default_representation": "gltf_glb",
        "representations": {
            "gltf_glb": {
                "type": "glb",
                "label": "Master Mesh LOD 0 (550k tris + 8K Atlas)",
                "glb_url": "/outputs/indoor/dataset_my_movie_4_adaptive/room_photographic_model_lod0.glb",
                "file_size_bytes": p_lod0_glb.stat().st_size,
                "sha256": sha_lod0_glb,
                "available": True
            },
            "adaptive_chunks": {
                "type": "adaptive_chunks",
                "label": "Adaptive View-Dependent Chunks (Dynamic LOD 0/1/2 + Frustum Culling)",
                "chunks_url": "/outputs/indoor/dataset_my_movie_4_adaptive/adaptive_spatial_chunks.json",
                "file_size_bytes": p_chunks_json.stat().st_size,
                "sha256": sha_chunks,
                "available": True
            },
            "lod0": {
                "type": "glb",
                "label": "LOD 0 — High Detail (550k tris + 8K Atlas)",
                "glb_url": "/outputs/indoor/dataset_my_movie_4_adaptive/room_photographic_model_lod0.glb",
                "file_size_bytes": p_lod0_glb.stat().st_size,
                "sha256": sha_lod0_glb,
                "available": True
            },
            "lod1": {
                "type": "glb",
                "label": "LOD 1 — Balanced (250k tris + 4K Atlas)",
                "glb_url": "/outputs/indoor/dataset_my_movie_4_adaptive/room_photographic_model_lod1.glb",
                "file_size_bytes": p_lod1_glb.stat().st_size,
                "sha256": sha_lod1_glb,
                "available": True
            },
            "lod2": {
                "type": "glb",
                "label": "LOD 2 — Performance (80k tris + 2K Atlas)",
                "glb_url": "/outputs/indoor/dataset_my_movie_4_adaptive/room_photographic_model_lod2.glb",
                "file_size_bytes": p_lod2_glb.stat().st_size,
                "sha256": sha_lod2_glb,
                "available": True
            },
            "textured_obj": {
                "type": "obj",
                "label": "Wavefront OBJ + MTL (LOD 0 Mesh + UV Chart Atlas)",
                "obj_url": "/outputs/indoor/dataset_my_movie_4_adaptive/room_photographic_model_lod0.obj",
                "mtl_url": "/outputs/indoor/dataset_my_movie_4_adaptive/room_photographic_model_lod0.mtl",
                "file_size_bytes": p_lod0_obj.stat().st_size,
                "sha256": sha_lod0_obj,
                "available": True
            },
            "photographic_ply": {
                "type": "ply",
                "label": "Photographic Vertex PLY (273k Vertices)",
                "ply_url": "/outputs/indoor/dataset_my_movie_4_adaptive/room_photographic_model_lod0.ply",
                "file_size_bytes": p_lod0_ply.stat().st_size,
                "sha256": sha_lod0_ply,
                "available": True
            }
        },
        "thumbnail_url": "/models/indoor_my_movie_4_adaptive/preview_thumbnail.png",
        "previews": {
            "eye_level": "/models/indoor_my_movie_4_adaptive/preview_eye_level.png",
            "top": "/models/indoor_my_movie_4_adaptive/preview_top.png"
        },
        "metadata_url": "/models/indoor_my_movie_4_adaptive/metadata.json",
        "initial_camera": {
            "position": [0.0, 0.45, -0.9],
            "target": [0.0, 0.1, 0.0]
        },
        "walkthrough": {
            "spawn_position": [0.0, 0.25, -0.6],
            "spawn_yaw": 0.0,
            "walk_speed": 1.2,
            "head_bobbing": True,
            "collision_radius": 0.25
        }
    }

    manifest_paths = [
        Path("viewer/public/model-manifest.json"),
        Path("viewer/src/data/model-manifest.json")
    ]

    for mp in manifest_paths:
        if mp.exists():
            data = json.loads(mp.read_text(encoding="utf-8"))
            scenes = data.get("scenes", [])
            # Filter out existing if any
            scenes = [s for s in scenes if s["id"] != "indoor_my_movie_4_adaptive"]
            # Insert at beginning
            scenes.insert(0, SCENE_ENTRY)
            data["scenes"] = scenes
            mp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            print(f"[+] Updated manifest at {mp}")

    print("\n[+] Successfully registered indoor_my_movie_4_adaptive!")

if __name__ == "__main__":
    main()
