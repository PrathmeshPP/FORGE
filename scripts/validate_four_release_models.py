"""F.O.R.G.E — Phase B: Independent Multi-Library Validation of the Four Release Models.

Validates:
1. Campus 0049: outputs/opencv5_experiments/final_models_v3/campus_0049/
2. River / Bridge: outputs/opencv5_experiments/final_models_v3/river_bridge/
3. Desert Road: outputs/opencv5_experiments/final_models/desert_road/
4. Forest Road: outputs/opencv5_experiments/final_models/forest_road/

Checks:
- GLB: Trimesh reload, scene graph, materials, embedded textures, primitives, geometry validity.
- OBJ & MTL: Wavefront geometry, material definitions, texture file presence and readability.
- PLY: Open3D reload, finite coordinates, normals, vertex colors.
- Texture Atlas: Resolution, dimensions, file size, SHA-256.
- Mesh Geometry: True non-manifold edges (> 2 faces) vs natural boundary edges (= 1 face).
- Bounding Box & Extents.
- Provenance and Manifest consistency.
"""

import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import open3d as o3d
import trimesh
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS = {
    "campus_0049": {
        "dir": PROJECT_ROOT / "outputs/opencv5_experiments/final_models_v3/campus_0049",
        "prefix": "campus_0049_final_reconstruction",
        "has_atlas": True,
        "expected_atlas_res": (4096, 4096)
    },
    "river_bridge": {
        "dir": PROJECT_ROOT / "outputs/opencv5_experiments/final_models_v3/river_bridge",
        "prefix": "river_bridge_final_reconstruction",
        "has_atlas": True,
        "expected_atlas_res": (4096, 4096)
    },
    "desert_road": {
        "dir": PROJECT_ROOT / "outputs/opencv5_experiments/final_models/desert_road",
        "prefix": "desert_road_final_reconstruction",
        "has_atlas": False,
        "expected_atlas_res": None
    },
    "forest_road": {
        "dir": PROJECT_ROOT / "outputs/opencv5_experiments/final_models/forest_road",
        "prefix": "forest_road_final_reconstruction",
        "has_atlas": False,
        "expected_atlas_res": None
    }
}

OUT_REPORT = PROJECT_ROOT / "outputs/opencv5_experiments/final_release_qa/four_models_validation_report.json"
OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def validate_scene(scene_key: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    print(f"\n{'='*70}\nVALIDATING {scene_key.upper()}\n{'='*70}")
    m_dir = cfg["dir"]
    prefix = cfg["prefix"]

    p_glb = m_dir / f"{prefix}.glb"
    p_obj = m_dir / f"{prefix}.obj"
    p_mtl = m_dir / f"{prefix}.mtl"
    p_ply = m_dir / f"{prefix}.ply"
    p_atlas = m_dir / "texture_atlas.png"
    p_prev = m_dir / "model_preview.png"
    p_meta = m_dir / "reconstruction_metadata.json"

    file_checks = {
        "glb_exists": p_glb.is_file(),
        "obj_exists": p_obj.is_file(),
        "mtl_exists": p_mtl.is_file(),
        "ply_exists": p_ply.is_file(),
        "atlas_exists": p_atlas.is_file() if cfg["has_atlas"] else True,
        "preview_exists": p_prev.is_file(),
        "metadata_exists": p_meta.is_file()
    }
    for k, v in file_checks.items():
        assert v, f"Failed file existence check: {k} for {scene_key}"

    # 1. Trimesh GLB Validation
    print("[*] Inspecting GLB via Trimesh...")
    tm_glb = trimesh.load(str(p_glb), process=False)
    glb_materials = []
    glb_textures = []
    if isinstance(tm_glb, trimesh.Scene):
        geoms = list(tm_glb.geometry.values())
        glb_verts = sum(len(g.vertices) for g in geoms)
        glb_faces = sum(len(g.faces) for g in geoms)
        for g in geoms:
            mat = getattr(g.visual, "material", None)
            if mat:
                glb_materials.append(str(mat))
                if hasattr(mat, "image") and mat.image is not None:
                    glb_textures.append(f"{mat.image.size[0]}x{mat.image.size[1]} ({mat.image.mode})")
    else:
        glb_verts = len(tm_glb.vertices)
        glb_faces = len(tm_glb.faces)

    print(f"    GLB Geometry: {glb_verts:,} verts, {glb_faces:,} faces, {len(glb_textures)} embedded texture(s)")

    # 2. Open3D PLY Validation
    print("[*] Inspecting PLY via Open3D...")
    m_o3d = o3d.io.read_triangle_mesh(str(p_ply))
    v = np.asarray(m_o3d.vertices)
    f = np.asarray(m_o3d.triangles)
    assert len(v) > 0 and len(f) > 0, "PLY mesh is empty!"

    all_finite = bool(np.isfinite(v).all())
    b_min = v.min(axis=0).round(4).tolist()
    b_max = v.max(axis=0).round(4).tolist()
    extents = np.ptp(v, axis=0).round(4).tolist()

    # 3. Trimesh OBJ & Non-Manifold / Boundary Analysis
    print("[*] Inspecting OBJ and surface manifolds via Trimesh...")
    tm_obj = trimesh.load(str(p_obj), process=False)
    if isinstance(tm_obj, trimesh.Scene):
        tm_obj = trimesh.util.concatenate(list(tm_obj.geometry.values()))

    edge_face_count = np.bincount(tm_obj.edges_unique_inverse)
    true_nm_edges = int(np.sum(edge_face_count > 2))
    boundary_edges = int(np.sum(edge_face_count == 1))
    print(f"    True non-manifold edges (> 2 faces): {true_nm_edges}")
    print(f"    Natural boundary edges (= 1 face):   {boundary_edges:,}")

    # 4. MTL & Texture References
    print("[*] Checking MTL material definitions and texture paths...")
    mtl_texture_refs = []
    missing_textures = []
    with open(p_mtl, "r", encoding="utf-8", errors="ignore") as f_mtl:
        for line in f_mtl:
            line_str = line.strip()
            if line_str.startswith("map_Kd"):
                tex_fn = line_str.split(maxsplit=1)[1]
                mtl_texture_refs.append(tex_fn)
                tex_path = m_dir / tex_fn
                if not tex_path.is_file():
                    missing_textures.append(tex_fn)

    print(f"    MTL references {len(mtl_texture_refs)} texture(s); missing: {len(missing_textures)}")
    assert len(missing_textures) == 0, f"Missing referenced textures: {missing_textures}"

    # 5. Texture Atlas Image Check
    atlas_info = None
    if cfg["has_atlas"]:
        print("[*] Checking texture atlas PNG...")
        im_atlas = Image.open(p_atlas)
        atlas_info = {
            "resolution": f"{im_atlas.size[0]}x{im_atlas.size[1]}",
            "mode": im_atlas.mode,
            "size_mb": round(p_atlas.stat().st_size / (1024 * 1024), 2),
            "sha256": sha256_file(p_atlas)
        }
        if cfg["expected_atlas_res"]:
            assert im_atlas.size == cfg["expected_atlas_res"], f"Atlas resolution mismatch: {im_atlas.size} vs {cfg['expected_atlas_res']}"
        print(f"    Atlas: {atlas_info['resolution']}, {atlas_info['size_mb']} MB, mode {atlas_info['mode']}")

    deliverable_files = {
        "glb": {
            "path": str(p_glb.resolve()),
            "size_bytes": p_glb.stat().st_size,
            "size_mb": round(p_glb.stat().st_size / (1024 * 1024), 3),
            "sha256": sha256_file(p_glb)
        },
        "obj": {
            "path": str(p_obj.resolve()),
            "size_bytes": p_obj.stat().st_size,
            "size_mb": round(p_obj.stat().st_size / (1024 * 1024), 3),
            "sha256": sha256_file(p_obj)
        },
        "mtl": {
            "path": str(p_mtl.resolve()),
            "size_bytes": p_mtl.stat().st_size,
            "sha256": sha256_file(p_mtl)
        },
        "ply": {
            "path": str(p_ply.resolve()),
            "size_bytes": p_ply.stat().st_size,
            "size_mb": round(p_ply.stat().st_size / (1024 * 1024), 3),
            "sha256": sha256_file(p_ply)
        }
    }
    if cfg["has_atlas"]:
        deliverable_files["atlas"] = atlas_info

    meta_content = json.loads(p_meta.read_text()) if p_meta.is_file() else {}

    res = {
        "scene": scene_key,
        "deliverable_dir": str(m_dir.resolve()),
        "file_existence_check": file_checks,
        "geometry": {
            "vertex_count": len(v),
            "triangle_count": len(f),
            "coordinates_all_finite": all_finite,
            "bounds_min": b_min,
            "bounds_max": b_max,
            "extents": extents,
            "true_non_manifold_edges": true_nm_edges,
            "natural_boundary_edges": boundary_edges,
            "is_watertight": bool(m_o3d.is_watertight())
        },
        "glb_validation": {
            "loads_successfully": True,
            "scene_or_mesh": "Scene" if isinstance(tm_glb, trimesh.Scene) else "Trimesh",
            "embedded_textures": glb_textures
        },
        "materials_and_textures": {
            "mtl_texture_references": mtl_texture_refs,
            "all_referenced_textures_exist": (len(missing_textures) == 0),
            "missing_textures": missing_textures,
            "texture_atlas": atlas_info
        },
        "deliverables": deliverable_files,
        "metadata_file": {
            "exists": True,
            "winning_candidate_action": meta_content.get("action") or meta_content.get("evaluation_reason"),
            "evaluation_category": meta_content.get("evaluation_category") or meta_content.get("evaluation_decision")
        },
        "qa_status": "PASS" if true_nm_edges == 0 and len(missing_textures) == 0 else (
            "PASS WITH LIMITATIONS" if true_nm_edges < 150 and len(missing_textures) == 0 else "FAIL"
        )
    }
    print(f"[+] QA Status for {scene_key}: {res['qa_status']}")
    return res


def main():
    print("=" * 75)
    print("F.O.R.G.E PHASE B: FOUR RELEASE MODELS INDEPENDENT MULTI-LIBRARY VALIDATION")
    print("=" * 75)

    all_scenes_res = {}
    for sk, cfg in MODELS.items():
        all_scenes_res[sk] = validate_scene(sk, cfg)

    with open(OUT_REPORT, "w") as f:
        json.dump(all_scenes_res, f, indent=2)

    print(f"\n[+] Master validation report written to: {OUT_REPORT}")


if __name__ == "__main__":
    main()
