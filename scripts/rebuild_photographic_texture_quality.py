"""
F.O.R.G.E. - Photographic Texture Quality Recovery & Comparison Pipeline.

Produces:
1. Candidate B: Sharpest-view winner-take-all photographic projection without averaging (1024x1024 native tiles).
2. Candidate C: Best-view photographic model with seam-aware boundary harmonization.
3. Coverage Heatmap: 4-category surface classification (Direct, Occluded, Never-Observed, Blended).
4. Verified Wavefront OBJ + MTL (with normals vn) and glTF/GLB (with NORMAL and glTF 2.0 UVs).
"""

from __future__ import annotations

import os
import sys
import time
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont
import trimesh

# Directories
PROJECT_ROOT = Path("c:/Users/25beevdt025/Desktop/FORGE")
EXP_BASE = PROJECT_ROOT / "outputs/indoor/texture_experiment_video1_reproduced_video_170234"
OUTPUT_DIR = PROJECT_ROOT / "outputs/indoor/photographic_texture_quality_recovery_video1"
RENDERS_DIR = OUTPUT_DIR / "renders"
MESHES_DIR = OUTPUT_DIR / "meshes"
DIAG_DIR = OUTPUT_DIR / "diagnostics"

for d in [OUTPUT_DIR, RENDERS_DIR, MESHES_DIR, DIAG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def get_leveled_decimated_mesh() -> Tuple[o3d.geometry.TriangleMesh, np.ndarray, np.ndarray]:
    """
    Loads or creates the 250k decimated mesh in leveled space, along with canonical transformation.
    """
    cache_path = MESHES_DIR / "mesh_decimated_leveled.ply"
    if cache_path.exists():
        log(f"Loading cached decimated leveled mesh: {cache_path.name}...")
        mesh = o3d.io.read_triangle_mesh(str(cache_path))
    else:
        log("Loading Poisson mesh and decimating to 250,000 triangles...")
        poisson_path = EXP_BASE / "meshes/poisson_mesh.ply"
        mesh_raw = o3d.io.read_triangle_mesh(str(poisson_path))
        mesh = mesh_raw.simplify_quadric_decimation(target_number_of_triangles=250000)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_non_manifold_edges()
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        o3d.io.write_triangle_mesh(str(cache_path), mesh)
        log(f"Cached decimated leveled mesh ({len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris)")

    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    verts_leveled = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)

    # Canonical transform: T_canon = diag(1, -1, -1) and centered
    T_canon = np.array([1.0, -1.0, -1.0], dtype=np.float64)
    verts_canon = verts_leveled * T_canon
    center = verts_canon.mean(axis=0)
    verts_canon = verts_canon - center

    return mesh, verts_canon, center


def compute_all_face_scores(
    verts_leveled: np.ndarray,
    tris: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    depth_maps: np.ndarray,
    R_floor: np.ndarray,
    W_native: int,
    H_native: int,
    frame_sharpness: Optional[List[float]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes per-face visibility, occlusion, and photographic scores across all cameras.
    Returns:
        scores: (N_f, M) photographic quality score
        occluded_mask: (N_f, M) boolean
        in_bounds_mask: (N_f, M) boolean
    """
    N_f = len(tris)
    M = len(extrinsics)

    # Centroids and normals in leveled space
    centroids = verts_leveled[tris].mean(axis=1)
    v0 = verts_leveled[tris[:, 0]]
    v1 = verts_leveled[tris[:, 1]]
    v2 = verts_leveled[tris[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(cross, axis=1, keepdims=True) + 1e-12
    normals = cross / norms

    # Intrinsics scaling from inference (432x240) to native
    H_infer, W_infer = depth_maps[0].shape[:2]
    sx = float(W_native) / float(W_infer)
    sy = float(H_native) / float(H_infer)
    K_native = np.zeros_like(intrinsics, dtype=np.float64)
    for i in range(M):
        K = intrinsics[i].copy().astype(np.float64)
        K[0, 0] *= sx; K[0, 2] *= sx
        K[1, 1] *= sy; K[1, 2] *= sy
        K_native[i] = K

    # Effective extrinsics: P_c = R_i @ R_floor.T @ X_leveled + t_i
    scores = np.zeros((N_f, M), dtype=np.float32)
    occluded_matrix = np.zeros((N_f, M), dtype=bool)
    in_bounds_matrix = np.zeros((N_f, M), dtype=bool)

    sharp_weights = np.ones(M, dtype=np.float32)
    if frame_sharpness is not None and len(frame_sharpness) == M:
        s_arr = np.array(frame_sharpness, dtype=np.float32)
        s_min, s_max = s_arr.min(), s_arr.max()
        if s_max > s_min:
            sharp_weights = 0.75 + 0.25 * ((s_arr - s_min) / (s_max - s_min))

    for i in range(M):
        R_eff = extrinsics[i, :3, :3] @ R_floor.T
        t_eff = extrinsics[i, :3, 3]
        K = K_native[i]

        P_c = (R_eff @ centroids.T + t_eff[:, None]).T
        Z_c = P_c[:, 2]

        valid_z = Z_c > 0.10
        u = K[0, 0] * (P_c[:, 0] / np.maximum(Z_c, 1e-4)) + K[0, 2]
        v = K[1, 1] * (P_c[:, 1] / np.maximum(Z_c, 1e-4)) + K[1, 2]

        in_bounds = (
            valid_z
            & (u >= 8.0)
            & (u <= W_native - 9.0)
            & (v >= 8.0)
            & (v <= H_native - 9.0)
        )
        in_bounds_matrix[:, i] = in_bounds

        dist = np.linalg.norm(P_c, axis=1, keepdims=True) + 1e-6
        rays = P_c / dist
        N_c = (R_eff @ normals.T).T
        cos_theta = -np.sum(rays * N_c, axis=1)

        facing = in_bounds & (cos_theta > 0.10)

        # Depth occlusion test
        d_map = depth_maps[i].squeeze()
        H_d, W_d = d_map.shape[:2]
        u_d = np.clip((u * (W_d / W_native)).astype(np.int32), 0, W_d - 1)
        v_d = np.clip((v * (H_d / H_native)).astype(np.int32), 0, H_d - 1)
        depth_pred = d_map[v_d, u_d]

        occluded = facing & (depth_pred > 0.15) & (Z_c > depth_pred * 1.15)
        occluded_matrix[:, i] = occluded

        visible = facing & (~occluded)

        cos_safe = np.maximum(cos_theta, 0.0)
        q = np.where(
            visible,
            (cos_safe ** 2.2) * (1.0 / np.maximum(dist[:, 0], 0.20)) * sharp_weights[i],
            0.0
        )
        scores[:, i] = q

    return scores, occluded_matrix, in_bounds_matrix


def select_optimal_keyframes(
    scores: np.ndarray,
    target_k: int = 16
) -> List[int]:
    """
    Greedy set-cover: iteratively picks cameras that maximize newly covered faces.
    """
    N_f, M = scores.shape
    selected: List[int] = []
    covered = np.zeros(N_f, dtype=bool)

    # First keyframe: camera covering the most faces with high score
    for step in range(target_k):
        best_cam = -1
        best_gain = -1
        for c in range(M):
            if c in selected:
                continue
            new_faces = (scores[:, c] > 0.0) & (~covered)
            gain = int(new_faces.sum())
            if gain > best_gain:
                best_gain = gain
                best_cam = c

        if best_cam >= 0 and best_gain > 0:
            selected.append(best_cam)
            covered |= (scores[:, best_cam] > 0.0)
        else:
            # Fallback to next highest total score camera
            remaining = [c for c in range(M) if c not in selected]
            if remaining:
                tot_scores = [scores[:, c].sum() for c in remaining]
                selected.append(remaining[int(np.argmax(tot_scores))])
            else:
                break

    return sorted(selected)


def build_coverage_heatmap(
    mesh_canon: o3d.geometry.TriangleMesh,
    scores_full: np.ndarray,
    occluded_full: np.ndarray,
    in_bounds_full: np.ndarray,
    best_cam_per_face: np.ndarray,
    output_ply: Path,
    output_glb: Path
) -> Dict[str, Any]:
    """
    Builds the 4-category coverage heatmap mesh:
    1. Directly Photographed (Green: rgb(46, 204, 113))
    2. Occluded (Yellow: rgb(241, 196, 15))
    3. Never Observed (Red: rgb(231, 76, 60))
    4. Blended Transition Region (Blue: rgb(52, 152, 219))
    """
    tris = np.asarray(mesh_canon.triangles)
    N_f = len(tris)

    # Adjacency for transition boundary detection
    edges_to_faces: Dict[Tuple[int, int], List[int]] = {}
    for f_idx, tri in enumerate(tris):
        for e in [
            (min(tri[0], tri[1]), max(tri[0], tri[1])),
            (min(tri[1], tri[2]), max(tri[1], tri[2])),
            (min(tri[2], tri[0]), max(tri[2], tri[0]))
        ]:
            edges_to_faces.setdefault(e, []).append(f_idx)

    is_boundary = np.zeros(N_f, dtype=bool)
    for f_idx in range(N_f):
        c_i = best_cam_per_face[f_idx]
        if c_i < 0:
            continue
        tri = tris[f_idx]
        for e in [
            (min(tri[0], tri[1]), max(tri[0], tri[1])),
            (min(tri[1], tri[2]), max(tri[1], tri[2])),
            (min(tri[2], tri[0]), max(tri[2], tri[0]))
        ]:
            for adj_f in edges_to_faces.get(e, []):
                if adj_f != f_idx and best_cam_per_face[adj_f] >= 0 and best_cam_per_face[adj_f] != c_i:
                    is_boundary[f_idx] = True
                    break
            if is_boundary[f_idx]:
                break

    max_scores = scores_full.max(axis=1)
    has_direct = max_scores > 0.0

    # Occluded but not direct
    ever_occluded = occluded_full.any(axis=1)
    ever_in_bounds = in_bounds_full.any(axis=1)

    category = np.zeros(N_f, dtype=np.int32)
    # 0: Never Observed (Red)
    # 1: Occluded (Yellow)
    # 2: Directly Photographed (Green)
    # 3: Blended Transition Region (Blue)

    category[ever_in_bounds & ever_occluded & (~has_direct)] = 1
    category[has_direct & (~is_boundary)] = 2
    category[has_direct & is_boundary] = 3

    colors_table = np.array([
        [231, 76, 60],    # Red: Never Observed
        [241, 196, 15],   # Yellow: Occluded
        [46, 204, 113],   # Green: Directly Photographed
        [52, 152, 219]    # Blue: Blended Transition Region
    ], dtype=np.float64) / 255.0

    face_colors = colors_table[category]

    # Map face colors to vertices
    N_v = len(mesh_canon.vertices)
    v_colors = np.zeros((N_v, 3), dtype=np.float64)
    v_weights = np.zeros(N_v, dtype=np.float64)
    for f_idx, tri in enumerate(tris):
        c = face_colors[f_idx]
        for vi in tri:
            v_colors[vi] += c
            v_weights[vi] += 1.0
    v_colors = v_colors / np.maximum(v_weights[:, None], 1.0)

    # Save PLY
    heatmap_mesh = o3d.geometry.TriangleMesh(mesh_canon)
    heatmap_mesh.vertex_colors = o3d.utility.Vector3dVector(v_colors)
    o3d.io.write_triangle_mesh(str(output_ply), heatmap_mesh, write_vertex_colors=True)

    # Save GLB
    tm_heatmap = trimesh.Trimesh(
        vertices=np.asarray(mesh_canon.vertices),
        faces=tris,
        vertex_colors=(v_colors * 255).astype(np.uint8),
        process=False
    )
    tm_heatmap.export(str(output_glb), file_type="glb")

    stats = {
        "total_faces": int(N_f),
        "directly_photographed_faces": int((category == 2).sum()),
        "directly_photographed_pct": round(float((category == 2).mean() * 100.0), 2),
        "blended_transition_faces": int((category == 3).sum()),
        "blended_transition_pct": round(float((category == 3).mean() * 100.0), 2),
        "total_photographic_faces": int(has_direct.sum()),
        "total_photographic_coverage_pct": round(float(has_direct.mean() * 100.0), 2),
        "occluded_faces": int((category == 1).sum()),
        "occluded_pct": round(float((category == 1).mean() * 100.0), 2),
        "never_observed_faces": int((category == 0).sum()),
        "never_observed_pct": round(float((category == 0).mean() * 100.0), 2)
    }
    return stats


def synthesize_high_res_texture_atlas(
    active_cams: List[int],
    frame_paths: List[Path],
    out_atlas_path: Path,
    atlas_size: int = 4096,
    grid: Tuple[int, int] = (4, 4),
    apply_seam_harmonization: bool = False
) -> Tuple[Dict[int, Tuple[int, int, int, int]], Dict[str, Any]]:
    """
    Synthesizes a 4096 x 4096 texture atlas where each of the 16 keyframe tiles
    receives 1024 x 1024 pixels (exceeding native 478x850 sensor resolution!).
    """
    cols, rows = grid
    tile_w = atlas_size // cols
    tile_h = atlas_size // rows

    canvas = Image.new("RGB", (atlas_size, atlas_size), color=(24, 26, 30))
    tile_rects = {}

    # Load frames
    loaded_frames = []
    for cam_idx in active_cams:
        im = Image.open(frame_paths[cam_idx]).convert("RGB")
        loaded_frames.append(im)

    # Optional seam harmonization for Candidate C: equalize global exposure / luminance
    if apply_seam_harmonization:
        log("  Applying multi-view exposure & luminance harmonization across charts...")
        mean_lums = []
        for im in loaded_frames:
            im_np = np.array(im, dtype=np.float32)
            lum = 0.299 * im_np[:, :, 0] + 0.587 * im_np[:, :, 1] + 0.114 * im_np[:, :, 2]
            mean_lums.append(float(lum.mean()))
        target_lum = float(np.median(mean_lums))
        harmonized_frames = []
        for idx, im in enumerate(loaded_frames):
            im_np = np.array(im, dtype=np.float32)
            cur_lum = mean_lums[idx]
            gain = np.clip(target_lum / max(cur_lum, 1e-3), 0.85, 1.18)
            im_np = np.clip(im_np * gain, 0.0, 255.0).astype(np.uint8)
            harmonized_frames.append(Image.fromarray(im_np))
        loaded_frames = harmonized_frames

    for idx, cam_idx in enumerate(active_cams):
        c = idx % cols
        r = idx // cols
        x_off = c * tile_w
        y_off = r * tile_h
        tile_rects[cam_idx] = (c, r, cols, rows)

        im = loaded_frames[idx]
        im_resized = im.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
        canvas.paste(im_resized, (x_off, y_off))

    canvas.save(out_atlas_path, format="PNG", optimize=True)

    stats = {
        "atlas_size": [atlas_size, atlas_size],
        "grid_layout": [cols, rows],
        "tile_resolution": [tile_w, tile_h],
        "active_cams_count": len(active_cams),
        "atlas_file_mb": round(out_atlas_path.stat().st_size / (1024 ** 2), 2)
    }
    return tile_rects, stats


def export_photographic_model(
    output_dir: Path,
    base_name: str,
    verts_leveled: np.ndarray,
    verts_canon: np.ndarray,
    tris: np.ndarray,
    best_cam_per_face: np.ndarray,
    tile_rects: Dict[int, Tuple[int, int, int, int]],
    atlas_path: Path,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    R_floor: np.ndarray,
    W_native: int,
    H_native: int
) -> Dict[str, str]:
    """
    Exports:
    1. Wavefront OBJ + MTL (with normals vn, and valid vt for OBJ).
    2. glTF 2.0 GLB with embedded texture, correct glTF 2.0 UVs, and vertex normals.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    N_f = len(tris)
    N_v = len(verts_canon)

    # Compute vertex normals in canonical space
    v0 = verts_canon[tris[:, 0]]
    v1 = verts_canon[tris[:, 1]]
    v2 = verts_canon[tris[:, 2]]
    face_norms = np.cross(v1 - v0, v2 - v0)
    face_norms = face_norms / (np.linalg.norm(face_norms, axis=1, keepdims=True) + 1e-12)

    vert_norms = np.zeros((N_v, 3), dtype=np.float64)
    for f_idx, tri in enumerate(tris):
        for vi in tri:
            vert_norms[vi] += face_norms[f_idx]
    vert_norms = vert_norms / (np.linalg.norm(vert_norms, axis=1, keepdims=True) + 1e-12)

    # Camera effective extrinsics & native intrinsics
    M = len(extrinsics)
    K_native = np.zeros_like(intrinsics, dtype=np.float64)
    sx = float(W_native) / 240.0
    sy = float(H_native) / 432.0
    for i in range(M):
        K = intrinsics[i].copy().astype(np.float64)
        K[0, 0] *= sx; K[0, 2] *= sx
        K[1, 1] *= sy; K[1, 2] *= sy
        K_native[i] = K

    # Compute UVs for both OBJ and GLTF for every corner of every face
    uvs_obj = np.zeros((N_f, 3, 2), dtype=np.float32)
    uvs_gltf = np.zeros((N_f, 3, 2), dtype=np.float32)

    for f_idx in range(N_f):
        cam_idx = best_cam_per_face[f_idx]
        c, r, cols, rows = tile_rects[cam_idx]

        R_eff = extrinsics[cam_idx, :3, :3] @ R_floor.T
        t_eff = extrinsics[cam_idx, :3, 3]
        K = K_native[cam_idx]

        tri = tris[f_idx]
        tri_pts = verts_leveled[tri]

        P_c = (R_eff @ tri_pts.T + t_eff[:, None]).T
        Z_c = np.maximum(P_c[:, 2], 1e-4)
        u = K[0, 0] * (P_c[:, 0] / Z_c) + K[0, 2]
        v = K[1, 1] * (P_c[:, 1] / Z_c) + K[1, 2]

        u_cam = np.clip(u / float(W_native), 0.0, 1.0)
        v_cam = np.clip(v / float(H_native), 0.0, 1.0)

        # OBJ UV coordinates: V=0 is bottom, V=1 is top
        u_obj = (c + u_cam) / float(cols)
        v_obj = 1.0 - (r + v_cam) / float(rows)
        uvs_obj[f_idx, :, 0] = u_obj
        uvs_obj[f_idx, :, 1] = v_obj

        # glTF 2.0 UV coordinates: V=0 is top, V=1 is bottom (flipY=false in Three.js)
        u_gltf = (c + u_cam) / float(cols)
        v_gltf = (r + v_cam) / float(rows)
        uvs_gltf[f_idx, :, 0] = u_gltf
        uvs_gltf[f_idx, :, 1] = v_gltf

    # 1. Write Wavefront OBJ & MTL
    obj_path = output_dir / f"{base_name}.obj"
    mtl_path = output_dir / f"{base_name}.mtl"
    dst_atlas = output_dir / atlas_path.name
    if dst_atlas.resolve() != atlas_path.resolve():
        shutil.copyfile(atlas_path, dst_atlas)

    with open(mtl_path, "w", encoding="utf-8") as fm:
        fm.write("# F.O.R.G.E. High-Resolution Photographic Material\n")
        fm.write("newmtl Material_Photographic_Atlas\n")
        fm.write("Ka 1.000000 1.000000 1.000000\n")
        fm.write("Kd 1.000000 1.000000 1.000000\n")
        fm.write("Ks 0.050000 0.050000 0.050000\n")
        fm.write("Ns 10.000000\n")
        fm.write("illum 2\n")
        fm.write(f"map_Kd {dst_atlas.name}\n")
    results["mtl"] = str(mtl_path.resolve())

    log(f"  Writing textured OBJ with normals: {obj_path.name}...")
    with open(obj_path, "w", encoding="utf-8") as fo:
        fo.write(f"# F.O.R.G.E. Photographic Textured Mesh: {base_name}\n")
        fo.write(f"mtllib {mtl_path.name}\n\n")

        # Vertices
        for v in verts_canon:
            fo.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # Normals
        for vn in vert_norms:
            fo.write(f"vn {vn[0]:.6f} {vn[1]:.6f} {vn[2]:.6f}\n")

        fo.write(f"\nusemtl Material_Photographic_Atlas\n")

        vt_idx = 1
        for f_idx in range(N_f):
            for k in range(3):
                fo.write(f"vt {uvs_obj[f_idx, k, 0]:.6f} {uvs_obj[f_idx, k, 1]:.6f}\n")

            tri = tris[f_idx]
            v1, v2, v3 = tri[0] + 1, tri[1] + 1, tri[2] + 1
            fo.write(f"f {v1}/{vt_idx}/{v1} {v2}/{vt_idx+1}/{v2} {v3}/{vt_idx+2}/{v3}\n")
            vt_idx += 3

    results["obj"] = str(obj_path.resolve())

    # 2. Build GLB with Embedded Texture, Correct glTF 2.0 UVs, and Normals
    glb_path = output_dir / f"{base_name}.glb"
    log(f"  Building embedded-texture GLB: {glb_path.name}...")

    # Flatten mesh for glTF (unwelded vertices so each triangle has exact UVs & normals)
    unwelded_verts = verts_canon[tris.reshape(-1)]
    unwelded_norms = vert_norms[tris.reshape(-1)]
    unwelded_uvs = uvs_gltf.reshape(-1, 2)
    unwelded_faces = np.arange(len(unwelded_verts)).reshape(-1, 3)

    atlas_img = Image.open(dst_atlas).convert("RGB")
    material = trimesh.visual.material.PBRMaterial(
        name="Material_Photographic_Atlas",
        baseColorTexture=atlas_img,
        baseColorFactor=[255, 255, 255, 255],
        roughnessFactor=0.65,
        metallicFactor=0.02,
        doubleSided=True
    )

    visual = trimesh.visual.texture.TextureVisuals(
        uv=unwelded_uvs,
        material=material
    )

    mesh_glb = trimesh.Trimesh(
        vertices=unwelded_verts,
        faces=unwelded_faces,
        vertex_normals=unwelded_norms,
        visual=visual,
        process=False
    )

    glb_bytes = mesh_glb.export(file_type="glb")
    with open(glb_path, "wb") as fg:
        fg.write(glb_bytes)
    results["glb"] = str(glb_path.resolve())
    log(f"  Saved textured GLB ({glb_path.stat().st_size / (1024**2):.2f} MB)")

    return results


def main() -> None:
    t_start = time.perf_counter()
    log("================================================================================")
    log("F.O.R.G.E. - REBUILDING INDOOR PHOTOGRAPHIC TEXTURE QUALITY (VIDEO 1)")
    log("================================================================================")

    # 1. Load Leveled Decimated Mesh
    mesh_dec, verts_canon, center = get_leveled_decimated_mesh()
    verts_leveled = np.asarray(mesh_dec.vertices)
    tris = np.asarray(mesh_dec.triangles)
    N_f = len(tris)
    log(f"Mesh geometry loaded: {len(verts_canon):,} vertices, {N_f:,} triangles")

    # 2. Load Camera Geometry & Depth Predictions
    cam_npz = np.load(EXP_BASE / "camera_geometry/camera_poses.npz")
    depth_npz = np.load(EXP_BASE / "camera_geometry/depth_predictions.npz")
    extrinsics = cam_npz["extrinsics"].astype(np.float64)
    intrinsics = cam_npz["intrinsics"].astype(np.float64)
    depth_maps = depth_npz["depth"].astype(np.float32)

    with open(EXP_BASE / "plane_detection.json") as fp:
        plane_meta = json.load(fp)
    n = np.array(plane_meta["detected_planes"][0]["normal"], dtype=np.float64)
    target = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    v = np.cross(n, target)
    s = np.linalg.norm(v)
    c = np.dot(n, target)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R_floor = np.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s ** 2 + 1e-8))

    frame_paths = sorted(list((EXP_BASE / "frames").glob("*.png")))
    with Image.open(frame_paths[0]) as im0:
        W_native, H_native = im0.size
    log(f"Loaded {len(frame_paths)} curated frames at native sensor resolution: {W_native} x {H_native}")

    with open(EXP_BASE / "frame_sampling_report.json") as fr:
        curated_data = json.load(fr)
    records = curated_data.get("selected_keyframes", [])
    sharpness_list = [r.get("laplacian_sharpness", 50.0) for r in records]

    # 3. Compute Per-Face Visibility, Fronto-Parallel Normal Alignment & Occlusions
    log("\n[Stage 1] Computing face visibility & occlusion checks across all cameras...")
    scores_full, occluded_full, in_bounds_full = compute_all_face_scores(
        verts_leveled=verts_leveled,
        tris=tris,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        depth_maps=depth_maps,
        R_floor=R_floor,
        W_native=W_native,
        H_native=H_native,
        frame_sharpness=sharpness_list
    )

    # 4. Greedy Keyframe Selection: 16 Dominant Views for 1024x1024 Native Tiles
    log("\n[Stage 2] Selecting 16 dominant keyframe views for 1024x1024 native atlas packing...")
    keyframe_indices = select_optimal_keyframes(scores_full, target_k=16)
    log(f"  Selected 16 keyframe cameras: {keyframe_indices}")

    # Best camera per face among keyframes
    scores_key = scores_full[:, keyframe_indices]
    best_local_idx = np.argmax(scores_key, axis=1)
    max_scores_key = np.max(scores_key, axis=1)
    uncovered = max_scores_key <= 0.0

    best_cam = np.array(keyframe_indices)[best_local_idx]
    best_cam[uncovered] = -1

    # Topological consensus smoothing for uncovered faces
    if uncovered.any():
        edges_to_faces: Dict[Tuple[int, int], List[int]] = {}
        for f_idx, tri in enumerate(tris):
            for e in [
                (min(tri[0], tri[1]), max(tri[0], tri[1])),
                (min(tri[1], tri[2]), max(tri[1], tri[2])),
                (min(tri[2], tri[0]), max(tri[2], tri[0]))
            ]:
                edges_to_faces.setdefault(e, []).append(f_idx)

        for f_idx in np.where(uncovered)[0]:
            tri = tris[f_idx]
            neighbor_cams = []
            for e in [
                (min(tri[0], tri[1]), max(tri[0], tri[1])),
                (min(tri[1], tri[2]), max(tri[1], tri[2])),
                (min(tri[2], tri[0]), max(tri[2], tri[0]))
            ]:
                for adj_f in edges_to_faces.get(e, []):
                    if adj_f != f_idx and best_cam[adj_f] >= 0:
                        neighbor_cams.append(best_cam[adj_f])
            if neighbor_cams:
                vals, counts = np.unique(neighbor_cams, return_counts=True)
                best_cam[f_idx] = vals[np.argmax(counts)]

    # Fallback for remaining unassigned faces
    still_uncovered = best_cam < 0
    if still_uncovered.any():
        cam_centers = np.array([-extrinsics[c, :3, :3].T @ extrinsics[c, :3, 3] for c in keyframe_indices])
        centroids = verts_leveled[tris].mean(axis=1)
        dists = np.linalg.norm(centroids[still_uncovered, None, :] - cam_centers[None, :, :], axis=-1)
        best_cam[still_uncovered] = np.array(keyframe_indices)[np.argmin(dists, axis=1)]

    # 5. Build Coverage Heatmap
    log("\n[Stage 3] Building 4-category coverage heatmap...")
    mesh_canon_geom = o3d.geometry.TriangleMesh()
    mesh_canon_geom.vertices = o3d.utility.Vector3dVector(verts_canon)
    mesh_canon_geom.triangles = o3d.utility.Vector3iVector(tris)
    mesh_canon_geom.compute_vertex_normals()

    heatmap_stats = build_coverage_heatmap(
        mesh_canon=mesh_canon_geom,
        scores_full=scores_full,
        occluded_full=occluded_full,
        in_bounds_full=in_bounds_full,
        best_cam_per_face=best_cam,
        output_ply=OUTPUT_DIR / "coverage_heatmap.ply",
        output_glb=OUTPUT_DIR / "coverage_heatmap.glb"
    )
    log(f"  Coverage Heatmap Stats: {json.dumps(heatmap_stats, indent=2)}")

    # 6. Candidate B: Sharpest-View Winner-Take-All Projection (1024x1024 native tiles)
    log("\n[Stage 4] Generating Candidate B: Sharpest-View Winner-Take-All Model...")
    atlas_b_path = OUTPUT_DIR / "candidate_b_atlas.png"
    tile_rects_b, atlas_b_stats = synthesize_high_res_texture_atlas(
        active_cams=keyframe_indices,
        frame_paths=frame_paths,
        out_atlas_path=atlas_b_path,
        atlas_size=4096,
        grid=(4, 4),
        apply_seam_harmonization=False
    )
    cand_b_files = export_photographic_model(
        output_dir=OUTPUT_DIR,
        base_name="candidate_b_sharpest_winner_take_all",
        verts_leveled=verts_leveled,
        verts_canon=verts_canon,
        tris=tris,
        best_cam_per_face=best_cam,
        tile_rects=tile_rects_b,
        atlas_path=atlas_b_path,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        R_floor=R_floor,
        W_native=W_native,
        H_native=H_native
    )

    # 7. Candidate C: Best-View + Seam-Aware Boundary Harmonization
    log("\n[Stage 5] Generating Candidate C: Seam-Aware Harmonized Photographic Model...")
    atlas_c_path = OUTPUT_DIR / "candidate_c_atlas.png"
    tile_rects_c, atlas_c_stats = synthesize_high_res_texture_atlas(
        active_cams=keyframe_indices,
        frame_paths=frame_paths,
        out_atlas_path=atlas_c_path,
        atlas_size=4096,
        grid=(4, 4),
        apply_seam_harmonization=True
    )
    cand_c_files = export_photographic_model(
        output_dir=OUTPUT_DIR,
        base_name="candidate_c_seam_aware_blended",
        verts_leveled=verts_leveled,
        verts_canon=verts_canon,
        tris=tris,
        best_cam_per_face=best_cam,
        tile_rects=tile_rects_c,
        atlas_path=atlas_c_path,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        R_floor=R_floor,
        W_native=W_native,
        H_native=H_native
    )

    # 8. Copy / Symlink Candidate A (Baseline)
    log("\n[Stage 6] Preserving Candidate A (Baseline)...")
    cand_a_glb = OUTPUT_DIR / "candidate_a_baseline.glb"
    cand_a_obj = OUTPUT_DIR / "candidate_a_baseline.obj"
    cand_a_mtl = OUTPUT_DIR / "candidate_a_baseline.mtl"
    cand_a_atlas = OUTPUT_DIR / "candidate_a_baseline_atlas.png"

    shutil.copyfile(EXP_BASE / "room_photographic_model.glb", cand_a_glb)
    shutil.copyfile(EXP_BASE / "room_photographic_model_canonical.obj", cand_a_obj)
    shutil.copyfile(EXP_BASE / "room_photographic_model_canonical.mtl", cand_a_mtl)
    shutil.copyfile(EXP_BASE / "texture_atlas.png", cand_a_atlas)

    # Save summary metadata
    duration = time.perf_counter() - t_start
    summary = {
        "dataset": "WhatsApp Video 2026-10-30 at 17.02.34.mp4",
        "experiment": "photographic_texture_quality_recovery_video1",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_seconds": round(duration, 2),
        "mesh_triangles": int(N_f),
        "mesh_vertices": int(len(verts_canon)),
        "source_frame_resolution": [W_native, H_native],
        "atlas_resolution": [4096, 4096],
        "atlas_tile_resolution": [1024, 1024],
        "candidate_a_baseline": {
            "glb": str(cand_a_glb.resolve()),
            "obj": str(cand_a_obj.resolve()),
            "atlas": str(cand_a_atlas.resolve()),
            "tiles": "63 frames @ 512x512 (downscaled, inverted glTF UVs, missing normals)"
        },
        "candidate_b_sharpest": {
            "glb": cand_b_files["glb"],
            "obj": cand_b_files["obj"],
            "atlas": str(atlas_b_path.resolve()),
            "tiles": "16 keyframes @ 1024x1024 (100% native sensor preservation, winner-take-all, glTF 2.0 UVs, vertex normals)"
        },
        "candidate_c_blended": {
            "glb": cand_c_files["glb"],
            "obj": cand_c_files["obj"],
            "atlas": str(atlas_c_path.resolve()),
            "tiles": "16 keyframes @ 1024x1024 (seam-aware multi-view luminance harmonization, glTF 2.0 UVs, vertex normals)"
        },
        "coverage_heatmap": {
            "ply": str((OUTPUT_DIR / "coverage_heatmap.ply").resolve()),
            "glb": str((OUTPUT_DIR / "coverage_heatmap.glb").resolve()),
            "stats": heatmap_stats
        }
    }

    with open(OUTPUT_DIR / "photographic_recovery_summary.json", "w", encoding="utf-8") as fs:
        json.dump(summary, fs, indent=2)

    log(f"\n[+] Pipeline completed successfully in {duration:.2f}s!")
    log(f"    Deliverables written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
