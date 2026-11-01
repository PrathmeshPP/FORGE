"""
F.O.R.G.E. — ADAPTIVE 3D RECONSTRUCTION & MULTI-RESOLUTION PIPELINE
Dataset: My movie 4.mp4 (421s, 1080x1920, 30fps)
Output: outputs/indoor/dataset_my_movie_4_adaptive/

Key Innovations:
1. Full VGGT-Omega Neural Pose & Geometry Estimation across curated keyframes.
2. High-Density 8mm Point Cloud Fusion & Ray-Oriented Normal Orientation.
3. Floor Plane Leveling to upright canonical orientation.
4. Screened Poisson Surface Reconstruction (depth=10).
5. Multi-Resolution LOD Mesh Generation:
   - LOD 0: 550k triangles, 8K atlas (Full Master Quality)
   - LOD 1: 250k triangles, 4K atlas (Balanced Quality)
   - LOD 2:  80k triangles, 2K atlas (Performance / Distant Overview)
6. Spatial Chunking (4 Quadrant Spatial Partitions) with per-chunk LODs.
7. Aspect-Preserving Multi-Scale Texture Atlas Synthesis.
"""

from __future__ import annotations

import os
import sys
import json
import time
import shutil
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
import trimesh
import torch

BASE_DIR = Path(r"C:\Users\25beevdt025\Desktop\FORGE")
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.reconstruction.indoor_pipeline import (
    IndoorPipelineConfig,
    IndoorReconstructionEngine,
    compute_floor_leveling_rotation
)
from src.reconstruction.point_cloud import depth_to_points, save_ply

INPUT_VIDEO = Path(r"C:\Users\25beevdt025\Downloads\My movie 4.mp4")
OUTPUT_DIR = BASE_DIR / "outputs" / "indoor" / "dataset_my_movie_4_adaptive"
ARTIFACT_DIR = Path(r"C:\Users\25beevdt025\.gemini\antigravity\brain\536a497a-fba6-496d-b4a9-b4f0cf756953")

CAMERA_DIR = OUTPUT_DIR / "camera_geometry"
MESH_DIR = OUTPUT_DIR / "meshes"
FRAMES_DIR = OUTPUT_DIR / "frames"
RENDERS_DIR = OUTPUT_DIR / "renders"
CHUNKS_DIR = OUTPUT_DIR / "spatial_chunks"

for d in [OUTPUT_DIR, CAMERA_DIR, MESH_DIR, FRAMES_DIR, RENDERS_DIR, CHUNKS_DIR, ARTIFACT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

log_f = open(OUTPUT_DIR / "reconstruction.log", "w", encoding="utf-8")

def log(msg: str):
    t_str = time.strftime("%H:%M:%S")
    line = f"[{t_str}] {msg}"
    print(line, flush=True)
    log_f.write(line + "\n")
    log_f.flush()

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024 * 8):
            h.update(chunk)
    return h.hexdigest()

def analyze_mesh_topology(mesh: o3d.geometry.TriangleMesh) -> Dict[str, Any]:
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    all_ab = mesh.get_non_manifold_edges(allow_boundary_edges=False)
    strict_nm = mesh.get_non_manifold_edges(allow_boundary_edges=True)
    b_edges = len(all_ab) - len(strict_nm)
    clusters, counts, _ = mesh.cluster_connected_triangles()
    
    bbox_min = verts.min(axis=0) if len(verts) > 0 else np.zeros(3)
    bbox_max = verts.max(axis=0) if len(verts) > 0 else np.zeros(3)
    span = bbox_max - bbox_min
    try:
        area = mesh.get_surface_area()
    except Exception:
        area = 0.0

    return {
        "vertices": int(len(verts)),
        "triangles": int(len(tris)),
        "boundary_edges": int(b_edges),
        "connected_components": int(len(counts)),
        "surface_area_m2": round(float(area), 3),
        "bounds_min": [round(float(v), 3) for v in bbox_min],
        "bounds_max": [round(float(v), 3) for v in bbox_max],
        "span_xyz": [round(float(v), 3) for v in span],
    }


# ==============================================================================
# MULTI-RESOLUTION PHOTOGRAPHIC TEXTURE ENGINE
# ==============================================================================
class AdaptiveTextureEngine:
    def __init__(
        self,
        mesh: o3d.geometry.TriangleMesh,
        extrinsics: np.ndarray,
        intrinsics: np.ndarray,
        frame_paths: List[Path],
        depth_maps: Optional[np.ndarray] = None,
        floor_rotation: Optional[np.ndarray] = None,
        min_cos_theta: float = 0.15
    ):
        self.mesh = o3d.geometry.TriangleMesh(mesh)
        self.mesh.compute_triangle_normals()
        self.mesh.compute_vertex_normals()

        self.extrinsics = extrinsics.astype(np.float64)
        self.intrinsics_infer = intrinsics.astype(np.float64)
        self.frame_paths = [Path(p) for p in frame_paths]
        self.depth_maps = depth_maps
        self.floor_rotation = floor_rotation if floor_rotation is not None else np.eye(3, dtype=np.float64)
        self.min_cos_theta = min_cos_theta

        self.num_cams = len(self.frame_paths)
        self.verts = np.asarray(self.mesh.vertices)
        self.tris = np.asarray(self.mesh.triangles)
        self.num_faces = len(self.tris)

        self.face_normals = np.asarray(self.mesh.triangle_normals)
        self.face_centroids = self.verts[self.tris].mean(axis=1)

        with Image.open(self.frame_paths[0]) as im:
            self.W_native, self.H_native = im.size

        if self.depth_maps is not None:
            d0 = self.depth_maps[0].squeeze()
            self.H_infer, self.W_infer = d0.shape[:2]
        else:
            self.H_infer, self.W_infer = 432, 240

        self.intrinsics_native = np.zeros_like(self.intrinsics_infer)
        sx = float(self.W_native) / float(self.W_infer)
        sy = float(self.H_native) / float(self.H_infer)
        for i in range(self.num_cams):
            K = self.intrinsics_infer[i].copy()
            K[0, 0] *= sx
            K[0, 2] *= sx
            K[1, 1] *= sy
            K[1, 2] *= sy
            self.intrinsics_native[i] = K

        self.eff_extrinsics = np.zeros_like(self.extrinsics)
        for i in range(self.num_cams):
            R_i = self.extrinsics[i, :3, :3]
            t_i = self.extrinsics[i, :3, 3]
            R_eff = R_i @ self.floor_rotation.T
            self.eff_extrinsics[i, :3, :3] = R_eff
            self.eff_extrinsics[i, :3, 3] = t_i

    def compute_face_visibility(
        self,
        frame_sharpness: Optional[List[float]] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        N_f = self.num_faces
        M = self.num_cams
        scores = np.zeros((N_f, M), dtype=np.float32)

        cam_centers = []
        for i in range(M):
            R_eff = self.eff_extrinsics[i, :3, :3]
            t_eff = self.eff_extrinsics[i, :3, 3]
            c_eff = -R_eff.T @ t_eff
            cam_centers.append(c_eff)
        cam_centers = np.array(cam_centers)

        sharp_weights = np.ones(M, dtype=np.float32)
        if frame_sharpness is not None and len(frame_sharpness) == M:
            s_arr = np.array(frame_sharpness, dtype=np.float32)
            s_min, s_max = s_arr.min(), s_arr.max()
            if s_max > s_min:
                sharp_weights = 0.75 + 0.25 * ((s_arr - s_min) / (s_max - s_min))

        for i in range(M):
            R_eff = self.eff_extrinsics[i, :3, :3]
            t_eff = self.eff_extrinsics[i, :3, 3]
            K = self.intrinsics_native[i]

            P_c = (R_eff @ self.face_centroids.T + t_eff[:, None]).T
            Z_c = P_c[:, 2]

            valid_z = Z_c > 0.10
            u = K[0, 0] * (P_c[:, 0] / np.maximum(Z_c, 1e-4)) + K[0, 2]
            v = K[1, 1] * (P_c[:, 1] / np.maximum(Z_c, 1e-4)) + K[1, 2]

            in_bounds = (
                valid_z
                & (u >= 8.0)
                & (u <= self.W_native - 9.0)
                & (v >= 8.0)
                & (v <= self.H_native - 9.0)
            )

            dist = np.linalg.norm(P_c, axis=1, keepdims=True) + 1e-6
            rays = P_c / dist
            N_c = (R_eff @ self.face_normals.T).T
            cos_theta = -np.sum(rays * N_c, axis=1)
            facing = in_bounds & (cos_theta > self.min_cos_theta)

            occluded = np.zeros(N_f, dtype=bool)
            if self.depth_maps is not None and i < len(self.depth_maps):
                d_map = self.depth_maps[i].squeeze()
                H_d, W_d = d_map.shape
                u_d = np.clip((u * (W_d / self.W_native)).astype(np.int32), 0, W_d - 1)
                v_d = np.clip((v * (H_d / self.H_native)).astype(np.int32), 0, H_d - 1)
                depth_pred = d_map[v_d, u_d]
                occluded = facing & (depth_pred > 0.15) & (Z_c > depth_pred * 1.15)

            visible = facing & (~occluded)
            cos_safe = np.maximum(cos_theta, 0.0)
            q = np.where(
                visible,
                (cos_safe ** 3.0) * (1.0 / np.maximum(dist[:, 0], 0.35)) * sharp_weights[i],
                0.0
            )
            scores[:, i] = q

        best_cam = np.argmax(scores, axis=1)
        max_scores = np.max(scores, axis=1)
        uncovered = max_scores <= 0.0
        best_cam[uncovered] = -1

        # Chart consolidation
        edges_to_faces: Dict[Tuple[int, int], List[int]] = {}
        for f_idx, tri in enumerate(self.tris):
            for edge in [
                (min(tri[0], tri[1]), max(tri[0], tri[1])),
                (min(tri[1], tri[2]), max(tri[1], tri[2])),
                (min(tri[2], tri[0]), max(tri[2], tri[0]))
            ]:
                edges_to_faces.setdefault(edge, []).append(f_idx)

        face_neighbors: List[List[int]] = [[] for _ in range(N_f)]
        for edge, faces in edges_to_faces.items():
            if len(faces) == 2:
                face_neighbors[faces[0]].append(faces[1])
                face_neighbors[faces[1]].append(faces[0])

        if uncovered.any():
            for f_idx in np.where(uncovered)[0]:
                n_cams = [best_cam[nb] for nb in face_neighbors[f_idx] if best_cam[nb] >= 0]
                if n_cams:
                    vals, counts = np.unique(n_cams, return_counts=True)
                    best_cam[f_idx] = vals[np.argmax(counts)]

        still_uncovered = best_cam < 0
        if still_uncovered.any():
            dists = np.linalg.norm(self.face_centroids[still_uncovered, None, :] - cam_centers[None, :, :], axis=-1)
            best_cam[still_uncovered] = np.argmin(dists, axis=1)

        # 3 consolidation passes
        for p in range(3):
            for f_idx in range(N_f):
                curr_c = best_cam[f_idx]
                nbrs = face_neighbors[f_idx]
                if len(nbrs) >= 2:
                    nbr_cams = [best_cam[nb] for nb in nbrs]
                    vals, counts = np.unique(nbr_cams, return_counts=True)
                    dom_cam = vals[np.argmax(counts)]
                    if dom_cam != curr_c and counts.max() >= 2:
                        dom_score = scores[f_idx, dom_cam]
                        curr_score = scores[f_idx, curr_c]
                        if dom_score >= curr_score * 0.65 and dom_score > 0.0:
                            best_cam[f_idx] = dom_cam

        cov = float((~uncovered).mean() * 100.0)
        stats = {
            "total_faces": N_f,
            "direct_faces": int((~uncovered).sum()),
            "coverage_pct": round(cov, 2),
            "active_cams": int(len(np.unique(best_cam)))
        }
        return best_cam, scores, stats

    def synthesize_atlas(
        self,
        best_cam_per_face: np.ndarray,
        out_atlas_path: Path,
        atlas_size: int = 8192
    ) -> Tuple[Dict[int, Tuple[int, int, int, int]], Dict[str, Any]]:
        active_cams = sorted(list(np.unique(best_cam_per_face)))
        cols, rows = 11, 6
        # Aspect-ratio preserved tile dimensions
        # 1080x1920 has aspect ratio 0.5625
        tile_w = int(atlas_size / cols * 0.95)
        tile_h = int(tile_w / (self.W_native / self.H_native))
        pad_x = (atlas_size - (cols * tile_w)) // (cols + 1)
        pad_y = (atlas_size - (rows * tile_h)) // (rows + 1)

        canvas = Image.new("RGB", (atlas_size, atlas_size), color=(26, 28, 32))
        tile_rects = {}
        for idx, cam_idx in enumerate(active_cams):
            c = idx % cols
            r = idx // cols
            x_off = pad_x + c * (tile_w + pad_x)
            y_off = pad_y + r * (tile_h + pad_y)
            tile_rects[cam_idx] = (x_off, y_off, tile_w, tile_h)

            im = Image.open(self.frame_paths[cam_idx]).convert("RGB")
            im_resized = im.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
            canvas.paste(im_resized, (x_off, y_off))

        out_atlas_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_atlas_path, format="PNG", optimize=True)
        stats = {
            "atlas_size": [atlas_size, atlas_size],
            "tile_size": [tile_w, tile_h],
            "tile_aspect": round(tile_w / tile_h, 4),
            "file_size_mb": round(out_atlas_path.stat().st_size / (1024**2), 2)
        }
        return tile_rects, stats

    def export_textured_mesh(
        self,
        output_dir: Path,
        base_name: str,
        best_cam_per_face: np.ndarray,
        tile_rects: Dict[int, Tuple[int, int, int, int]],
        atlas_path: Path,
        atlas_size: int = 8192
    ) -> Dict[str, str]:
        output_dir.mkdir(parents=True, exist_ok=True)
        results = {}

        verts = self.verts.copy()
        T_canon = np.array([1.0, -1.0, -1.0], dtype=np.float64)
        verts = verts * T_canon
        center = verts.mean(axis=0)
        verts = verts - center

        obj_path = output_dir / f"{base_name}.obj"
        mtl_path = output_dir / f"{base_name}.mtl"
        dst_atlas = output_dir / atlas_path.name
        if dst_atlas.resolve() != atlas_path.resolve():
            shutil.copyfile(atlas_path, dst_atlas)

        with open(mtl_path, "w", encoding="utf-8") as f_mtl:
            f_mtl.write(f"newmtl Material_Photographic\nKa 1 1 1\nKd 1 1 1\nKs 0.04 0.04 0.04\nillum 2\nNs 15\nmap_Kd {dst_atlas.name}\n")
        results["mtl"] = str(mtl_path.resolve())

        inv_atlas = 1.0 / float(atlas_size)
        with open(obj_path, "w", encoding="utf-8") as f_obj:
            f_obj.write(f"mtllib {mtl_path.name}\n\n")
            for v in verts:
                f_obj.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            f_obj.write(f"\nusemtl Material_Photographic\n")

            vt_idx = 1
            for f_idx in range(self.num_faces):
                cam_idx = best_cam_per_face[f_idx]
                x_off, y_off, tw, th = tile_rects[cam_idx]

                R_eff = self.eff_extrinsics[cam_idx, :3, :3]
                t_eff = self.eff_extrinsics[cam_idx, :3, 3]
                K = self.intrinsics_native[cam_idx]

                tri = self.tris[f_idx]
                tri_pts = self.verts[tri]
                P_c = (R_eff @ tri_pts.T + t_eff[:, None]).T
                Z_c = np.maximum(P_c[:, 2], 1e-4)
                u = K[0, 0] * (P_c[:, 0] / Z_c) + K[0, 2]
                v = K[1, 1] * (P_c[:, 1] / Z_c) + K[1, 2]

                u_cam = np.clip(u / float(self.W_native), 0.0, 1.0)
                v_cam = np.clip(v / float(self.H_native), 0.0, 1.0)
                u_atlas = (x_off + u_cam * tw) * inv_atlas
                v_atlas = 1.0 - (y_off + v_cam * th) * inv_atlas

                for k in range(3):
                    f_obj.write(f"vt {u_atlas[k]:.6f} {v_atlas[k]:.6f}\n")
                v1, v2, v3 = tri[0] + 1, tri[1] + 1, tri[2] + 1
                f_obj.write(f"f {v1}/{vt_idx} {v2}/{vt_idx+1} {v3}/{vt_idx+2}\n")
                vt_idx += 3

        results["obj"] = str(obj_path.resolve())

        # GLB export
        glb_path = output_dir / f"{base_name}.glb"
        try:
            tm = trimesh.load(str(obj_path), process=False)
            glb_bytes = tm.export(file_type="glb")
            with open(glb_path, "wb") as f_glb:
                f_glb.write(glb_bytes)
            results["glb"] = str(glb_path.resolve())
        except Exception as e:
            log(f"[!] GLB export error: {e}")

        # Vertex color PLY
        ply_path = output_dir / f"{base_name}.ply"
        N_v = len(self.verts)
        v_colors = np.zeros((N_v, 3), dtype=np.float64)
        vert_to_cams: Dict[int, List[int]] = {}
        for f_idx, tri in enumerate(self.tris):
            c_idx = best_cam_per_face[f_idx]
            for vi in tri:
                vert_to_cams.setdefault(vi, []).append(c_idx)

        cached_frames = {}
        for c_idx in np.unique(best_cam_per_face):
            im = cv2.imread(str(self.frame_paths[c_idx]))
            cached_frames[c_idx] = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        for vi in range(N_v):
            cams = vert_to_cams.get(vi, [])
            if not cams:
                continue
            vals, counts = np.unique(cams, return_counts=True)
            top_cam = vals[np.argmax(counts)]
            R_eff = self.eff_extrinsics[top_cam, :3, :3]
            t_eff = self.eff_extrinsics[top_cam, :3, 3]
            K = self.intrinsics_native[top_cam]
            P_c = R_eff @ self.verts[vi] + t_eff
            Z_c = P_c[2]
            if Z_c > 0.05:
                u = int(round(K[0, 0] * (P_c[0] / Z_c) + K[0, 2]))
                v = int(round(K[1, 1] * (P_c[1] / Z_c) + K[1, 2]))
                if 0 <= u < self.W_native and 0 <= v < self.H_native:
                    v_colors[vi] = cached_frames[top_cam][v, u] / 255.0

        canon_mesh = o3d.geometry.TriangleMesh()
        canon_mesh.vertices = o3d.utility.Vector3dVector(verts)
        canon_mesh.triangles = o3d.utility.Vector3iVector(self.tris)
        canon_mesh.vertex_colors = o3d.utility.Vector3dVector(v_colors)
        canon_mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(ply_path), canon_mesh, write_vertex_colors=True)
        results["ply"] = str(ply_path.resolve())

        return results


# ==============================================================================
# MAIN RECONSTRUCTION
# ==============================================================================
def main():
    log("=" * 76)
    log("  F.O.R.G.E. — ADAPTIVE 3D RECONSTRUCTION & MULTI-RESOLUTION PIPELINE")
    log("  Dataset: My movie 4.mp4 (421s, 1080x1920)")
    log(f"  Output Dir: {OUTPUT_DIR}")
    if torch.cuda.is_available():
        free_mem, total_mem = torch.cuda.mem_get_info()
        log(f"  CUDA GPU:   {torch.cuda.get_device_name(0)} (Free: {free_mem/(1024**2):.1f} MB)")
    log("=" * 76)

    t0_master = time.perf_counter()

    # Step 1: Load Curated Keyframes
    sampling_rep_p = OUTPUT_DIR / "frame_sampling_report.json"
    rep_data = json.loads(sampling_rep_p.read_text())
    frame_records = rep_data["frames"]
    image_paths = [r["file_path"] for r in frame_records]
    num_frames = len(image_paths)
    log(f"\n[Step 1] Loaded {num_frames} curated keyframes across 421s footage.")

    # Step 2: VGGT-Omega Neural Camera Geometry & Depth Estimation
    config = IndoorPipelineConfig(
        image_resolution=320,
        window_size=6,
        overlap=3,
        min_depth=0.25,
        max_depth=6.0,
        min_confidence=0.75,
        voxel_size=0.008,
        poisson_depth=10,
        level_floor=True
    )
    engine = IndoorReconstructionEngine(config)

    log("\n[Step 2] Executing VGGT-Omega Overlapping Window Inference...")
    t0_vggt = time.perf_counter()
    window_results = engine.run_overlapping_windows(
        image_paths=image_paths,
        window_size=config.window_size,
        overlap=config.overlap,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        log_callback=log
    )
    dur_vggt = time.perf_counter() - t0_vggt

    extrinsics = window_results["extrinsics"]
    intrinsics = window_results["intrinsics"]
    depth_maps = window_results["depth"]
    conf_maps = window_results["confidence"]
    vggt_perf = window_results["performance"]

    np.savez_compressed(CAMERA_DIR / "camera_poses.npz", extrinsics=extrinsics, intrinsics=intrinsics)
    np.savez_compressed(CAMERA_DIR / "depth_predictions.npz", depth=depth_maps, confidence=conf_maps)
    with open(CAMERA_DIR / "vggt_performance.json", "w") as f:
        json.dump(vggt_perf, f, indent=2)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log(f"  [+] Camera geometry estimated in {dur_vggt:.2f}s. Residual error: {vggt_perf['mean_sim3_alignment_error_m']*100:.2f} cm")

    # Step 3: High-Density 8mm Point Cloud Backprojection
    log("\n[Step 3] Point Cloud Backprojection & 8mm Scale-Aware Fusion...")
    t0_pcd = time.perf_counter()
    frames_bgr = [cv2.imread(p) for p in image_paths]
    N, H_inf, W_inf = depth_maps.shape[:3]

    cam_centers = []
    for i in range(N):
        R = extrinsics[i, :3, :3]
        t = extrinsics[i, :3, 3]
        cam_centers.append(-R.T @ t)
    cam_centers_np = np.array(cam_centers)

    all_pts, all_cols = [], []
    total_raw_points = 0
    for i in range(N):
        d = depth_maps[i].squeeze()
        c = conf_maps[i].squeeze()
        im_rgb = cv2.cvtColor(cv2.resize(frames_bgr[i], (W_inf, H_inf)), cv2.COLOR_BGR2RGB)
        valid = (d >= 0.25) & (d <= 6.0) & (c >= 0.75)
        total_raw_points += H_inf * W_inf
        if not np.any(valid):
            continue
        pts_3d = depth_to_points(d, extrinsics[i], intrinsics[i])
        all_pts.append(pts_3d[valid])
        all_cols.append(im_rgb[valid])

    merged_pts = np.vstack(all_pts)
    merged_cols = np.vstack(all_cols)
    log(f"  Raw Points: {total_raw_points:,} -> Valid Masked Points: {len(merged_pts):,}")

    raw_pcd_path = OUTPUT_DIR / "raw_point_cloud.ply"
    save_ply(merged_pts, merged_cols, str(raw_pcd_path))

    # Scale-aware 8mm voxel fusion
    pcd_8mm = o3d.geometry.PointCloud()
    pcd_8mm.points = o3d.utility.Vector3dVector(merged_pts.astype(np.float64))
    pcd_8mm.colors = o3d.utility.Vector3dVector(merged_cols.astype(np.float64) / 255.0)
    pcd_8mm = pcd_8mm.voxel_down_sample(voxel_size=0.008)

    pcd_8mm_clean, ind = pcd_8mm.remove_statistical_outlier(nb_neighbors=25, std_ratio=2.2)
    clean_pts = np.asarray(pcd_8mm_clean.points, dtype=np.float32)
    clean_cols = (np.asarray(pcd_8mm_clean.colors, dtype=np.float32) * 255.0).clip(0, 255).astype(np.uint8)
    dur_pcd = time.perf_counter() - t0_pcd

    filtered_pcd_path = OUTPUT_DIR / "filtered_point_cloud.ply"
    save_ply(clean_pts, clean_cols, str(filtered_pcd_path))
    log(f"  [+] Filtered Point Cloud: {len(clean_pts):,} structural points ({dur_pcd:.2f}s).")

    # Step 4: Floor Plane Detection & Floor Leveling
    log("\n[Step 4] Floor Plane Detection & Canonical Upright Alignment...")
    plane_model, inliers = pcd_8mm_clean.segment_plane(distance_threshold=0.025, ransac_n=3, num_iterations=1000)
    a, b, c, d_plane = plane_model
    normal = np.array([a, b, c], dtype=np.float32)
    if normal[1] < 0:
        normal = -normal
    R_floor = compute_floor_leveling_rotation(normal).astype(np.float32)

    clean_pts_rot = (R_floor @ clean_pts.T).T.astype(np.float32)
    cam_centers_rot = [(R_floor @ c) for c in cam_centers]
    cam_centers_rot_np = np.array(cam_centers_rot)

    plane_results = {
        "plane_normal": [round(float(v), 4) for v in normal],
        "rotation_matrix": R_floor.tolist()
    }
    with open(OUTPUT_DIR / "plane_detection.json", "w") as f:
        json.dump(plane_results, f, indent=2)
    log(f"  [+] Floor leveling aligned plane normal {normal.tolist()} to [0, 1, 0].")

    # Step 5: Ray-Oriented Interior Surface Normal Field
    log("\n[Step 5] Ray-Oriented Interior Normal Estimation...")
    pcd_rot = o3d.geometry.PointCloud()
    pcd_rot.points = o3d.utility.Vector3dVector(clean_pts_rot.astype(np.float64))
    pcd_rot.colors = o3d.utility.Vector3dVector(clean_cols.astype(np.float64) / 255.0)
    pcd_rot.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=30))

    pts_rot_np = np.asarray(pcd_rot.points)
    normals_np = np.asarray(pcd_rot.normals)
    dists_to_cams = np.linalg.norm(pts_rot_np[:, None, :] - cam_centers_rot_np[None, :, :], axis=-1)
    closest_cam_idx = np.argmin(dists_to_cams, axis=1)
    view_rays = cam_centers_rot_np[closest_cam_idx] - pts_rot_np
    view_rays /= (np.linalg.norm(view_rays, axis=1, keepdims=True) + 1e-6)

    dot_prod = np.sum(normals_np * view_rays, axis=1)
    flip_mask = dot_prod < 0.0
    normals_np[flip_mask] = -normals_np[flip_mask]
    pcd_rot.normals = o3d.utility.Vector3dVector(normals_np)
    log(f"  [+] Aligned {flip_mask.sum():,} ({flip_mask.mean()*100:.1f}%) normals towards camera lines of sight.")

    # Step 6: Screened Poisson Surface Meshing (depth=10)
    log("\n[Step 6] Screened Poisson Surface Meshing (depth=10)...")
    t0_mesh = time.perf_counter()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_rot, depth=10, linear_fit=True
    )
    dens_arr = np.asarray(densities)
    trim_thresh = np.percentile(dens_arr, 1.2)
    mesh.remove_vertices_by_mask(dens_arr < trim_thresh)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()

    clusters, counts, _ = mesh.cluster_connected_triangles()
    counts = np.asarray(counts)
    keep_clusters = np.where(counts >= 300)[0]
    mesh.remove_triangles_by_mask(~np.isin(np.asarray(clusters), keep_clusters))
    mesh.remove_unreferenced_vertices()

    dur_mesh = time.perf_counter() - t0_mesh
    log(f"  Raw reconstructed mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles ({dur_mesh:.2f}s).")

    # Step 7: Multi-Resolution LOD Mesh Hierarchy
    log("\n[Step 7] Generating Multi-Resolution LOD Hierarchy...")
    # LOD 0: Master High Quality (Target: 550,000 triangles)
    mesh_lod0 = mesh.simplify_quadric_decimation(target_number_of_triangles=min(550000, len(mesh.triangles)))
    mesh_lod0.remove_degenerate_triangles()
    mesh_lod0.remove_non_manifold_edges()
    mesh_lod0.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(MESH_DIR / "mesh_lod0.ply"), mesh_lod0)
    log(f"  [+] LOD 0 (High Quality):    {len(mesh_lod0.vertices):,} verts, {len(mesh_lod0.triangles):,} tris")

    # LOD 1: Balanced Quality (Target: 250,000 triangles)
    mesh_lod1 = mesh_lod0.simplify_quadric_decimation(target_number_of_triangles=250000)
    mesh_lod1.remove_degenerate_triangles()
    mesh_lod1.remove_non_manifold_edges()
    mesh_lod1.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(MESH_DIR / "mesh_lod1.ply"), mesh_lod1)
    log(f"  [+] LOD 1 (Balanced):        {len(mesh_lod1.vertices):,} verts, {len(mesh_lod1.triangles):,} tris")

    # LOD 2: Performance Overview (Target: 80,000 triangles)
    mesh_lod2 = mesh_lod1.simplify_quadric_decimation(target_number_of_triangles=80000)
    mesh_lod2.remove_degenerate_triangles()
    mesh_lod2.remove_non_manifold_edges()
    mesh_lod2.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(MESH_DIR / "mesh_lod2.ply"), mesh_lod2)
    log(f"  [+] LOD 2 (Performance):     {len(mesh_lod2.vertices):,} verts, {len(mesh_lod2.triangles):,} tris")

    # Step 8: Multi-Resolution Photographic Texturing & Deliverables Export
    log("\n[Step 8] Photographic Texturing & Multi-Resolution Deliverable Export...")
    frame_files = sorted(list(FRAMES_DIR.glob("*.png")))
    frame_sharpness = [r["sharpness"] for r in frame_records]

    # Texture LOD 0 with 8K Atlas
    log("  [*] Processing LOD 0 (8K Atlas)...")
    tex_engine0 = AdaptiveTextureEngine(
        mesh=mesh_lod0,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        frame_paths=frame_files,
        depth_maps=depth_maps,
        floor_rotation=R_floor
    )
    best_cam0, _, stats0 = tex_engine0.compute_face_visibility(frame_sharpness=frame_sharpness)
    atlas_p0 = OUTPUT_DIR / "texture_atlas_8k.png"
    rects0, atlas_stats0 = tex_engine0.synthesize_atlas(best_cam0, atlas_p0, atlas_size=8192)
    deliv_lod0 = tex_engine0.export_textured_mesh(
        output_dir=OUTPUT_DIR,
        base_name="room_photographic_model_lod0",
        best_cam_per_face=best_cam0,
        tile_rects=rects0,
        atlas_path=atlas_p0,
        atlas_size=8192
    )

    # Texture LOD 1 with 4K Atlas
    log("  [*] Processing LOD 1 (4K Atlas)...")
    tex_engine1 = AdaptiveTextureEngine(
        mesh=mesh_lod1,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        frame_paths=frame_files,
        depth_maps=depth_maps,
        floor_rotation=R_floor
    )
    best_cam1, _, stats1 = tex_engine1.compute_face_visibility(frame_sharpness=frame_sharpness)
    atlas_p1 = OUTPUT_DIR / "texture_atlas_4k.png"
    rects1, atlas_stats1 = tex_engine1.synthesize_atlas(best_cam1, atlas_p1, atlas_size=4096)
    deliv_lod1 = tex_engine1.export_textured_mesh(
        output_dir=OUTPUT_DIR,
        base_name="room_photographic_model_lod1",
        best_cam_per_face=best_cam1,
        tile_rects=rects1,
        atlas_path=atlas_p1,
        atlas_size=4096
    )

    # Texture LOD 2 with 2K Atlas
    log("  [*] Processing LOD 2 (2K Atlas)...")
    tex_engine2 = AdaptiveTextureEngine(
        mesh=mesh_lod2,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        frame_paths=frame_files,
        depth_maps=depth_maps,
        floor_rotation=R_floor
    )
    best_cam2, _, stats2 = tex_engine2.compute_face_visibility(frame_sharpness=frame_sharpness)
    atlas_p2 = OUTPUT_DIR / "texture_atlas_2k.png"
    rects2, atlas_stats2 = tex_engine2.synthesize_atlas(best_cam2, atlas_p2, atlas_size=2048)
    deliv_lod2 = tex_engine2.export_textured_mesh(
        output_dir=OUTPUT_DIR,
        base_name="room_photographic_model_lod2",
        best_cam_per_face=best_cam2,
        tile_rects=rects2,
        atlas_path=atlas_p2,
        atlas_size=2048
    )

    # Step 9: Spatial Chunk Partitioning (4 Spatial Quadrants)
    log("\n[Step 9] Partitioning Environment into 4 Spatial Chunks...")
    verts_canon = np.asarray(mesh_lod0.vertices) * np.array([1.0, -1.0, -1.0])
    center_c = verts_canon.mean(axis=0)
    verts_canon = verts_canon - center_c

    cx, cy, cz = 0.0, 0.0, 0.0
    tris0 = np.asarray(mesh_lod0.triangles)
    tri_centroids = verts_canon[tris0].mean(axis=1)

    chunk_quadrants = [
        ("chunk_0_northwest", (tri_centroids[:, 0] < cx) & (tri_centroids[:, 2] < cz)),
        ("chunk_1_northeast", (tri_centroids[:, 0] >= cx) & (tri_centroids[:, 2] < cz)),
        ("chunk_2_southwest", (tri_centroids[:, 0] < cx) & (tri_centroids[:, 2] >= cz)),
        ("chunk_3_southeast", (tri_centroids[:, 0] >= cx) & (tri_centroids[:, 2] >= cz)),
    ]

    spatial_manifest = {
        "spatial_chunks_count": 4,
        "grid_layout": "2x2 Quadrants (Northwest, Northeast, Southwest, Southeast)",
        "center": [float(cx), float(cy), float(cz)],
        "chunks": []
    }

    for chunk_id, (chunk_name, mask) in enumerate(chunk_quadrants):
        tri_indices = np.where(mask)[0]
        sub_tris = tris0[tri_indices]
        unique_v_idx, new_tris = np.unique(sub_tris, return_inverse=True)
        sub_verts = verts_canon[unique_v_idx]
        new_tris = new_tris.reshape(-1, 3)

        chunk_mesh = o3d.geometry.TriangleMesh()
        chunk_mesh.vertices = o3d.utility.Vector3dVector(sub_verts)
        chunk_mesh.triangles = o3d.utility.Vector3iVector(new_tris)
        chunk_mesh.compute_vertex_normals()

        # Decimate chunk for LOD 1 and LOD 2
        chunk_lod1 = chunk_mesh.simplify_quadric_decimation(target_number_of_triangles=max(1000, len(new_tris)//2))
        chunk_lod2 = chunk_mesh.simplify_quadric_decimation(target_number_of_triangles=max(500, len(new_tris)//5))

        c_bbox_min = sub_verts.min(axis=0).tolist()
        c_bbox_max = sub_verts.max(axis=0).tolist()
        c_center = ((sub_verts.min(axis=0) + sub_verts.max(axis=0)) / 2.0).tolist()

        c_glb_lod0 = CHUNKS_DIR / f"{chunk_name}_lod0.glb"
        c_glb_lod1 = CHUNKS_DIR / f"{chunk_name}_lod1.glb"
        c_glb_lod2 = CHUNKS_DIR / f"{chunk_name}_lod2.glb"

        # Export chunk meshes
        tm0 = trimesh.Trimesh(vertices=sub_verts, faces=new_tris, process=False)
        tm0.export(str(c_glb_lod0))
        tm1 = trimesh.Trimesh(vertices=np.asarray(chunk_lod1.vertices), faces=np.asarray(chunk_lod1.triangles), process=False)
        tm1.export(str(c_glb_lod1))
        tm2 = trimesh.Trimesh(vertices=np.asarray(chunk_lod2.vertices), faces=np.asarray(chunk_lod2.triangles), process=False)
        tm2.export(str(c_glb_lod2))

        spatial_manifest["chunks"].append({
            "chunk_id": chunk_id,
            "name": chunk_name,
            "bbox_min": [round(v, 3) for v in c_bbox_min],
            "bbox_max": [round(v, 3) for v in c_bbox_max],
            "center": [round(v, 3) for v in c_center],
            "triangles_lod0": int(len(new_tris)),
            "triangles_lod1": int(len(chunk_lod1.triangles)),
            "triangles_lod2": int(len(chunk_lod2.triangles)),
            "files": {
                "lod0": c_glb_lod0.name,
                "lod1": c_glb_lod1.name,
                "lod2": c_glb_lod2.name
            }
        })
        log(f"    Exported {chunk_name}: {len(new_tris):,} tris (LOD 0), {len(chunk_lod1.triangles):,} (LOD 1), {len(chunk_lod2.triangles):,} (LOD 2)")

    with open(OUTPUT_DIR / "adaptive_spatial_chunks.json", "w") as f:
        json.dump(spatial_manifest, f, indent=2)

    # Step 10: Multi-View Previews & LOD Comparison Panel
    log("\n[Step 10] Rendering LOD Comparison Previews...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1280, height=720)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.94, 0.95, 0.96])
    opt.light_on = True
    ctr = vis.get_view_control()

    m_ply0 = o3d.io.read_triangle_mesh(deliv_lod0["ply"])
    m_ply1 = o3d.io.read_triangle_mesh(deliv_lod1["ply"])
    m_ply2 = o3d.io.read_triangle_mesh(deliv_lod2["ply"])

    v_c = np.asarray(m_ply0.vertices)
    cx, cy, cz = v_c.mean(axis=0)
    x_min, y_min, z_min = v_c.min(axis=0)
    x_max, y_max, z_max = v_c.max(axis=0)
    span_z = z_max - z_min

    lod_views = [
        ("01_overview", [cx, cy + 2.5, cz - span_z * 0.9], [cx, cy, cz], [0, 1, 0]),
        ("02_entrance", [cx, y_min + 1.5, z_min + 0.3], [cx, y_min + 1.3, cz], [0, 1, 0]),
        ("03_furniture_closeup", [cx - 0.5, y_min + 1.1, cz - 0.3], [x_min + 0.2, y_min + 0.9, cz], [0, 1, 0]),
        ("04_top_down", [cx, cy + (y_max - y_min) * 2.2, cz], [cx, cy, cz], [0, 0, 1]),
    ]

    lod_renders = []
    for tag, eye, target, up in lod_views:
        # LOD 0
        vis.clear_geometries()
        vis.add_geometry(m_ply0)
        ctr.set_lookat(np.asarray(target, dtype=np.float64))
        ctr.set_front(np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64))
        ctr.set_up(np.asarray(up, dtype=np.float64))
        ctr.set_zoom(0.7)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.1)
        r0 = RENDERS_DIR / f"render_{tag}_lod0.png"
        vis.capture_screen_image(str(r0), do_render=True)

        # LOD 1
        vis.clear_geometries()
        vis.add_geometry(m_ply1)
        ctr.set_lookat(np.asarray(target, dtype=np.float64))
        ctr.set_front(np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64))
        ctr.set_up(np.asarray(up, dtype=np.float64))
        ctr.set_zoom(0.7)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.1)
        r1 = RENDERS_DIR / f"render_{tag}_lod1.png"
        vis.capture_screen_image(str(r1), do_render=True)

        # LOD 2
        vis.clear_geometries()
        vis.add_geometry(m_ply2)
        ctr.set_lookat(np.asarray(target, dtype=np.float64))
        ctr.set_front(np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64))
        ctr.set_up(np.asarray(up, dtype=np.float64))
        ctr.set_zoom(0.7)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.1)
        r2 = RENDERS_DIR / f"render_{tag}_lod2.png"
        vis.capture_screen_image(str(r2), do_render=True)

        lod_renders.append((tag, r0, r1, r2))

    vis.destroy_window()

    # Build LOD Comparison Panel (4 rows x 3 columns)
    tile_w, tile_h = 560, 315
    header_h = 32
    panel_w = tile_w * 3
    panel_h = len(lod_renders) * (tile_h + header_h) + 70
    lod_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

    title_text = "F.O.R.G.E. — MULTI-RESOLUTION ADAPTIVE LOD COMPARISON (MY MOVIE 4): LOD 0 vs LOD 1 vs LOD 2"
    cv2.putText(lod_panel, title_text, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    for i, (tag, p0, p1, p2) in enumerate(lod_renders):
        y0 = 70 + i * (tile_h + header_h)
        clean_tag = tag.replace("_", " ").upper()

        im0 = cv2.resize(cv2.imread(str(p0)), (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        im1 = cv2.resize(cv2.imread(str(p1)), (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        im2 = cv2.resize(cv2.imread(str(p2)), (tile_w, tile_h), interpolation=cv2.INTER_AREA)

        hdr0 = np.zeros((header_h, tile_w, 3), dtype=np.uint8)
        hdr1 = np.zeros((header_h, tile_w, 3), dtype=np.uint8)
        hdr2 = np.zeros((header_h, tile_w, 3), dtype=np.uint8)
        cv2.putText(hdr0, f"[LOD 0: MASTER HIGH QUALITY] {clean_tag} (550k tris, 8K atlas)", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        cv2.putText(hdr1, f"[LOD 1: BALANCED RUNTIME] {clean_tag} (250k tris, 4K atlas)", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 120), 1)
        cv2.putText(hdr2, f"[LOD 2: PERFORMANCE OVERVIEW] {clean_tag} (80k tris, 2K atlas)", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        lod_panel[y0 : y0 + header_h, 0 : tile_w] = hdr0
        lod_panel[y0 + header_h : y0 + header_h + tile_h, 0 : tile_w] = im0
        lod_panel[y0 : y0 + header_h, tile_w : tile_w * 2] = hdr1
        lod_panel[y0 + header_h : y0 + header_h + tile_h, tile_w : tile_w * 2] = im1
        lod_panel[y0 : y0 + header_h, tile_w * 2 : tile_w * 3] = hdr2
        lod_panel[y0 + header_h : y0 + header_h + tile_h, tile_w * 2 : tile_w * 3] = im2

    lod_panel_p = OUTPUT_DIR / "my_movie_4_lod_comparison_panel.png"
    cv2.imwrite(str(lod_panel_p), lod_panel)
    shutil.copyfile(lod_panel_p, ARTIFACT_DIR / "my_movie_4_lod_comparison_panel.png")
    log(f"[+] Saved LOD comparison panel to: {lod_panel_p.name}")

    # Step 11: Write Metadata & Deliverables
    dur_master = time.perf_counter() - t0_master
    glb0_file = Path(deliv_lod0["glb"])
    glb1_file = Path(deliv_lod1["glb"])
    glb2_file = Path(deliv_lod2["glb"])

    metadata = {
        "pipeline_version": "adaptive_multiresolution_v1",
        "dataset_name": "My movie 4 (Adaptive Quality Full Survey)",
        "source_video": {
            "path": str(INPUT_VIDEO),
            "sha256": compute_sha256(INPUT_VIDEO),
            "duration_sec": 421.47,
            "resolution": [1080, 1920],
            "fps": 30.0
        },
        "reconstruction_runtime_sec": round(dur_master, 2),
        "point_cloud": {
            "total_raw_points": total_raw_points,
            "valid_masked_points": len(merged_pts),
            "filtered_points": len(clean_pts),
            "voxel_size_m": 0.008
        },
        "lod_hierarchy": {
            "lod0": {
                "label": "Master High Quality",
                "triangles": len(mesh_lod0.triangles),
                "vertices": len(mesh_lod0.vertices),
                "atlas": "8192x8192 (8K)",
                "glb_size_mb": round(glb0_file.stat().st_size / (1024**2), 2),
                "sha256": compute_sha256(glb0_file)
            },
            "lod1": {
                "label": "Balanced Runtime Quality",
                "triangles": len(mesh_lod1.triangles),
                "vertices": len(mesh_lod1.vertices),
                "atlas": "4096x4096 (4K)",
                "glb_size_mb": round(glb1_file.stat().st_size / (1024**2), 2),
                "sha256": compute_sha256(glb1_file)
            },
            "lod2": {
                "label": "Performance / Distant Overview",
                "triangles": len(mesh_lod2.triangles),
                "vertices": len(mesh_lod2.vertices),
                "atlas": "2048x2048 (2K)",
                "glb_size_mb": round(glb2_file.stat().st_size / (1024**2), 2),
                "sha256": compute_sha256(glb2_file)
            }
        },
        "spatial_partitioning": spatial_manifest,
        "deliverables": {
            "lod0_glb": glb0_file.name,
            "lod1_glb": glb1_file.name,
            "lod2_glb": glb2_file.name,
            "canonical_obj": Path(deliv_lod0["obj"]).name,
            "canonical_mtl": Path(deliv_lod0["mtl"]).name,
            "vertex_ply": Path(deliv_lod0["ply"]).name,
            "texture_atlas_8k": atlas_p0.name,
            "texture_atlas_4k": atlas_p1.name,
            "texture_atlas_2k": atlas_p2.name,
            "filtered_point_cloud": filtered_pcd_path.name,
            "raw_point_cloud": raw_pcd_path.name
        }
    }

    with open(OUTPUT_DIR / "reconstruction_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    log(f"\n[+] Adaptive reconstruction complete in {dur_master:.2f}s!")

if __name__ == "__main__":
    main()
