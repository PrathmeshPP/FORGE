"""
F.O.R.G.E — Indoor 3D Reconstruction Pipeline.

Designed specifically for indoor single-room environments:
1. Indoor Depth & Confidence Gating:
   - Near-field bound: 0.25m (furniture in close proximity)
   - Far-field bound: 5.0m - 6.0m (room boundary walls)
   - Preserves ceiling and upper walls by disabling outdoor-specific sky/horizon culling.
2. Scale-Calibrated Voxel Grid Fusion:
   - High-density indoor grid (0.010m - 0.015m) preserving table edges, chair frames, door trims.
3. Multi-Camera Oriented Normal Fields:
   - Correctly orients surface normals toward camera observation rays.
4. Structural Plane Analysis:
   - Detects dominant horizontal (floor/ceiling) and vertical (walls) planes using RANSAC.
5. Tri-Method Meshing Engine:
   - Method 1: Screened Poisson Surface Reconstruction with KDTree Support-Distance Trimming.
   - Method 2: Open3D Scalable TSDF Volume Integration.
   - Method 3: Multi-Scale Ball-Pivoting Algorithm (BPA).
6. Multi-Format Deliverables & Three.js Canonical Alignment:
   - Exports OBJ+MTL, GLB, PLY.
   - glTF Upright Canonical Coordinate Alignment T = diag(1, -1, -1).
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import shutil
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Tuple, Optional, Union

import cv2
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw

from src.reconstruction.point_cloud import (
    depth_to_points,
    voxel_grid_fusion,
    save_ply,
    filter_by_multiview_support
)
from src.vggt.inference import VGGTInferencePipeline, DEFAULT_CHECKPOINT_PATH, load_vggt_model
import torch

for _candidate_vggt_dir in [
    r"C:\Users\25beevdt025\Desktop\SIH\third_party\vggt_omega",
    r"C:\Users\25beevdt025\Desktop\coding\third_party\vggt_omega",
]:
    if os.path.isdir(_candidate_vggt_dir) and _candidate_vggt_dir not in sys.path:
        sys.path.insert(0, _candidate_vggt_dir)

try:
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera
except ImportError:
    load_and_preprocess_images = None
    encoding_to_camera = None


@dataclass
class IndoorPipelineConfig:
    image_resolution: int = 320
    window_size: int = 6
    overlap: int = 3
    min_depth: float = 0.20
    max_depth: float = 5.5
    min_confidence: float = 0.8
    voxel_size: float = 0.012  # 1.2 cm voxel resolution
    poisson_depth: int = 9
    density_trim_quantile: float = 0.015  # Gentle trimming (1.5%)
    support_dist_multiplier: float = 8.0  # Support distance preserving continuous room boundaries
    min_cluster_faces: int = 40
    level_floor: bool = True  # Automatically level room geometry to horizontal floor plane
    tsdf_voxel_length: float = 0.015  # 1.5 cm for TSDF
    tsdf_trunc: float = 0.04
    bpa_radii: List[float] = field(default_factory=lambda: [0.008, 0.016, 0.032, 0.064])


def solve_umeyama_sim3(
    source_pts: np.ndarray,
    target_pts: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """
    Computes closed-form Sim(3) alignment: target ≈ s * (R @ source.T).T + t.
    Returns (scale s, rotation R (3x3), translation t (3,), median_residual_error_m).
    """
    X = np.asarray(source_pts, dtype=np.float64)
    Y = np.asarray(target_pts, dtype=np.float64)
    if len(X) < 3 or len(Y) < 3:
        return 1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64), 0.0

    mu_X = X.mean(axis=0)
    mu_Y = Y.mean(axis=0)
    X_c = X - mu_X
    Y_c = Y - mu_Y
    var_X = float(np.mean(np.sum(X_c ** 2, axis=1)))
    if var_X < 1e-8:
        return 1.0, np.eye(3, dtype=np.float64), (mu_Y - mu_X).astype(np.float64), 0.0

    cov = (Y_c.T @ X_c) / len(X)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3, dtype=np.float64)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1.0

    R = U @ S @ Vt
    s = float(np.sum(D * np.diag(S)) / var_X)
    if s <= 0.01 or s > 100.0 or math.isnan(s):
        s = 1.0
    t = mu_Y - s * (R @ mu_X)

    residuals = np.linalg.norm(Y - (s * (X @ R.T) + t), axis=1)
    med_err = float(np.median(residuals))
    return s, R, t, med_err


def compute_floor_leveling_rotation(plane_normal: np.ndarray) -> np.ndarray:
    """Computes 3x3 rotation matrix aligning plane normal to [0, 1, 0]."""
    n = np.asarray(plane_normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    if n[1] < 0:
        n = -n
    target = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    v = np.cross(n, target)
    c = float(np.dot(n, target))
    s = float(np.linalg.norm(v))
    if s < 1e-5:
        return np.eye(3, dtype=np.float64)
    v = v / s
    K = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ], dtype=np.float64)
    theta = math.acos(np.clip(c, -1.0, 1.0))
    R = np.eye(3, dtype=np.float64) + math.sin(theta) * K + (1.0 - c) * (K @ K)
    return R


class IndoorReconstructionEngine:
    def __init__(self, config: Optional[IndoorPipelineConfig] = None):
        self.config = config or IndoorPipelineConfig()

    def run_overlapping_windows(
        self,
        image_paths: List[str],
        window_size: int = 6,
        overlap: int = 3,
        checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
        device: str = "cuda:0",
        log_callback: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Executes bounded overlapping-window inference over full video frames.
        Enforces sustained >=1 FPS throughput with FP16 autocast and chains
        coordinate frames via closed-form Umeyama Sim(3) alignment.
        """
        def _log(msg: str):
            if log_callback:
                log_callback(msg)
            else:
                print(msg, flush=True)

        N = len(image_paths)
        if N == 0:
            raise ValueError("image_paths cannot be empty")

        _log(f"[*] Initializing VGGT-Omega overlapping window engine (Total frames: {N}, Window: {window_size}, Overlap: {overlap})...")
        t_init_start = time.perf_counter()
        model = load_vggt_model(checkpoint_path, device)
        init_time_sec = time.perf_counter() - t_init_start
        _log(f"    Model loaded into GPU in {init_time_sec:.2f}s")

        step = max(1, window_size - overlap)
        windows = []
        start = 0
        while start + window_size <= N:
            windows.append(list(range(start, start + window_size)))
            start += step
        if start < N:
            final_win = list(range(max(0, N - window_size), N))
            if not windows or final_win != windows[-1]:
                windows.append(final_win)

        _log(f"    Generated {len(windows)} overlapping windows across {N} frames.")

        global_frames = {}
        alignment_residuals = []
        total_vggt_infer_time = 0.0

        if "cuda" in device:
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        for w_idx, win_indices in enumerate(windows):
            t_w0 = time.perf_counter()
            win_paths = [image_paths[i] for i in win_indices]

            inp = load_and_preprocess_images(win_paths, image_resolution=self.config.image_resolution).to(device)

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                out = model(inp)
                pose_enc = out.get("pose_enc")
                ext, intr = encoding_to_camera(pose_enc, image_size_hw=inp.shape[-2:])

            if "cuda" in device:
                torch.cuda.synchronize()
            w_time = time.perf_counter() - t_w0
            total_vggt_infer_time += w_time

            ext_np = ext.squeeze(0).cpu().numpy()       # (W, 3, 4)
            intr_np = intr.squeeze(0).cpu().numpy()     # (W, 3, 3)
            depth_np = out["depth"].squeeze(0).cpu().numpy()  # (W, H, W, 1)
            conf_np = out["depth_conf"].squeeze(0).cpu().numpy()  # (W, H, W)

            if w_idx == 0:
                for local_pos, global_idx in enumerate(win_indices):
                    R = ext_np[local_pos, :3, :3]
                    t = ext_np[local_pos, :3, 3]
                    center = -R.T @ t
                    global_frames[global_idx] = {
                        "ext": ext_np[local_pos].copy(),
                        "intr": intr_np[local_pos].copy(),
                        "depth": depth_np[local_pos].copy(),
                        "conf": conf_np[local_pos].copy(),
                        "center": center,
                        "scale": 1.0
                    }
                _log(f"    Window [0/{len(windows)}] (frames {win_indices[0]}:{win_indices[-1]}): Processed in {w_time:.2f}s ({len(win_indices)/w_time:.2f} FPS)")
            else:
                shared_local_indices = []
                shared_global_indices = []
                for local_pos, global_idx in enumerate(win_indices):
                    if global_idx in global_frames:
                        shared_local_indices.append(local_pos)
                        shared_global_indices.append(global_idx)

                if len(shared_local_indices) >= 2:
                    pts_local_list = []
                    pts_global_list = []
                    H_d, W_d = depth_np.shape[1], depth_np.shape[2]
                    sub_y = np.linspace(int(H_d * 0.15), int(H_d * 0.85), 18, dtype=int)
                    sub_x = np.linspace(int(W_d * 0.15), int(W_d * 0.85), 18, dtype=int)

                    for l_idx, g_idx in zip(shared_local_indices, shared_global_indices):
                        g_entry = global_frames[g_idx]
                        p_glob = depth_to_points(g_entry["depth"], g_entry["ext"], g_entry["intr"])
                        p_loc = depth_to_points(depth_np[l_idx], ext_np[l_idx], intr_np[l_idx])

                        c_loc = conf_np[l_idx]
                        c_glob = g_entry["conf"]
                        c_loc_sub = c_loc[sub_y][:, sub_x]
                        c_glob_sub = c_glob[sub_y][:, sub_x]
                        valid_mask = (c_loc_sub >= 0.7) & (c_glob_sub >= 0.7)

                        p_loc_sub = p_loc[sub_y][:, sub_x][valid_mask]
                        p_glob_sub = p_glob[sub_y][:, sub_x][valid_mask]

                        if len(p_loc_sub) > 10:
                            pts_local_list.append(p_loc_sub)
                            pts_global_list.append(p_glob_sub)

                    if pts_local_list and pts_global_list:
                        X = np.vstack(pts_local_list)
                        Y = np.vstack(pts_global_list)
                        s_w, R_w, t_w, med_err = solve_umeyama_sim3(X, Y)
                    else:
                        s_w, R_w, t_w, med_err = 1.0, np.eye(3), np.zeros(3), 0.0
                else:
                    s_w, R_w, t_w, med_err = 1.0, np.eye(3), np.zeros(3), 0.0

                alignment_residuals.append(med_err)
                _log(f"    Window [{w_idx}/{len(windows)}] (frames {win_indices[0]}:{win_indices[-1]}): Processed in {w_time:.2f}s | Sim(3) scale: {s_w:.4f}, residual: {med_err*100:.1f} cm")

                for local_pos, global_idx in enumerate(win_indices):
                    if global_idx not in global_frames:
                        R_local = ext_np[local_pos, :3, :3]
                        t_local = ext_np[local_pos, :3, 3]
                        c_local = -R_local.T @ t_local

                        c_global = s_w * (R_w @ c_local) + t_w
                        R_global = R_local @ R_w.T
                        t_global = -R_global @ c_global

                        ext_global = np.hstack([R_global, t_global.reshape(3, 1)])
                        depth_global = s_w * depth_np[local_pos]

                        global_frames[global_idx] = {
                            "ext": ext_global.astype(np.float32),
                            "intr": intr_np[local_pos].astype(np.float32),
                            "depth": depth_global.astype(np.float32),
                            "conf": conf_np[local_pos].copy(),
                            "center": c_global,
                            "scale": float(s_w)
                        }

        all_ext = np.zeros((N, 3, 4), dtype=np.float32)
        all_intr = np.zeros((N, 3, 3), dtype=np.float32)
        H_d, W_d = global_frames[0]["depth"].shape[0], global_frames[0]["depth"].shape[1]
        all_depth = np.zeros((N, H_d, W_d, 1), dtype=np.float32)
        all_conf = np.zeros((N, H_d, W_d), dtype=np.float32)

        for i in range(N):
            if i in global_frames:
                all_ext[i] = global_frames[i]["ext"]
                all_intr[i] = global_frames[i]["intr"]
                all_depth[i] = global_frames[i]["depth"]
                all_conf[i] = global_frames[i]["conf"]

        peak_vram = torch.cuda.max_memory_allocated(0) / (1024**2) if "cuda" in device else 0.0
        avg_fps = N / max(0.001, total_vggt_infer_time)
        mean_res = float(np.mean(alignment_residuals)) if alignment_residuals else 0.0

        return {
            "extrinsics": all_ext,
            "intrinsics": all_intr,
            "depth": all_depth,
            "confidence": all_conf,
            "performance": {
                "num_frames": N,
                "num_windows": len(windows),
                "total_vggt_inference_sec": round(total_vggt_infer_time, 2),
                "model_init_sec": round(init_time_sec, 2),
                "average_time_per_window_sec": round(total_vggt_infer_time / len(windows), 3),
                "sustained_inference_fps": round(avg_fps, 2),
                "peak_vram_mb": round(peak_vram, 2),
                "mean_sim3_alignment_error_m": round(mean_res, 4),
                "window_alignment_residuals": [round(float(r), 4) for r in alignment_residuals]
            }
        }

    @staticmethod
    def assess_indoor_trajectory(extrinsics: np.ndarray) -> Dict[str, Any]:
        """Analyzes camera centers and trajectory bounds in world space."""
        N = len(extrinsics)
        cam_centers = np.zeros((N, 3), dtype=np.float64)
        rotations = np.zeros((N, 3, 3), dtype=np.float64)

        for i in range(N):
            R = extrinsics[i, :3, :3]
            t = extrinsics[i, :3, 3]
            cam_centers[i] = -R.T @ t
            rotations[i] = R

        steps = np.linalg.norm(cam_centers[1:] - cam_centers[:-1], axis=1) if N > 1 else np.array([0.0])
        total_path = float(np.sum(steps))
        net_dist = float(np.linalg.norm(cam_centers[-1] - cam_centers[0])) if N > 1 else 0.0
        extents = np.ptp(cam_centers, axis=0) if N > 1 else np.zeros(3)

        return {
            "num_cameras": N,
            "camera_centers": cam_centers.round(4).tolist(),
            "mean_camera_center": cam_centers.mean(axis=0).round(4).tolist(),
            "total_path_length_m": round(total_path, 4),
            "net_displacement_m": round(net_dist, 4),
            "trajectory_bounding_box": extents.round(4).tolist()
        }

    def backproject_indoor_pcd(
        self,
        frames_bgr: List[np.ndarray],
        depth_maps: np.ndarray,
        confidence_maps: np.ndarray,
        extrinsics: np.ndarray,
        intrinsics: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Backprojects depth maps into a fused, filtered indoor point cloud.
        """
        N, H_inf, W_inf = depth_maps.shape[:3]
        all_pts = []
        all_cols = []
        total_raw_points = 0

        cam_centers = []
        for i in range(N):
            R = extrinsics[i, :3, :3]
            t = extrinsics[i, :3, 3]
            cam_centers.append(-R.T @ t)

        for i in range(N):
            d = depth_maps[i]
            if d.ndim == 3:
                d = d.squeeze(-1)
            c = confidence_maps[i]
            if c.ndim == 3:
                c = c.squeeze(-1)

            # Resize frame to inference resolution for 1:1 color mapping
            im_rgb = cv2.cvtColor(cv2.resize(frames_bgr[i], (W_inf, H_inf)), cv2.COLOR_BGR2RGB)

            # Indoor validity mask
            valid = (d >= self.config.min_depth) & (d <= self.config.max_depth) & (c >= self.config.min_confidence)
            total_raw_points += H_inf * W_inf

            if not np.any(valid):
                continue

            pts_3d = depth_to_points(d, extrinsics[i], intrinsics[i])
            valid_pts = pts_3d[valid]
            valid_cols = im_rgb[valid]

            all_pts.append(valid_pts)
            all_cols.append(valid_cols)

        if not all_pts:
            raise RuntimeError("No valid 3D points recovered from depth maps.")

        merged_pts = np.vstack(all_pts)
        merged_cols = np.vstack(all_cols)

        # Scale-aware voxel grid fusion
        fused_pts, fused_cols = voxel_grid_fusion(merged_pts, merged_cols, voxel_size=self.config.voxel_size)

        # Statistical outlier removal using Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(fused_pts.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(fused_cols.astype(np.float64) / 255.0)

        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=1.8)
        clean_pcd = pcd.select_by_index(ind)

        clean_pts = np.asarray(clean_pcd.points, dtype=np.float32)
        clean_cols = (np.asarray(clean_pcd.colors, dtype=np.float32) * 255.0).clip(0, 255).astype(np.uint8)

        stats = {
            "total_raw_points": total_raw_points,
            "valid_masked_points": len(merged_pts),
            "fused_points": len(fused_pts),
            "clean_points": len(clean_pts),
            "retention_rate": round(len(clean_pts) / max(1, len(merged_pts)) * 100, 2),
            "extents": np.ptp(clean_pts, axis=0).round(4).tolist()
        }

        return clean_pts, clean_cols, (merged_pts, merged_cols), stats

    def detect_room_planes(self, points: np.ndarray, distance_threshold: float = 0.04) -> Dict[str, Any]:
        """Detects dominant floor and wall planes via RANSAC."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

        planes = []
        rest_pcd = pcd
        for p_idx in range(4):  # Look for up to 4 major planes
            if len(rest_pcd.points) < 500:
                break
            plane_model, inliers = rest_pcd.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=3,
                num_iterations=1000
            )
            [a, b, c, d] = plane_model
            normal = np.array([a, b, c])
            normal = normal / np.linalg.norm(normal)

            # Classify plane orientation
            # If normal is predominantly vertical (|Y| or |Z| depending on axis)
            is_horizontal = abs(normal[1]) > 0.8
            plane_type = "horizontal_floor_or_ceiling" if is_horizontal else "vertical_wall"

            planes.append({
                "plane_index": p_idx,
                "equation": [round(float(v), 5) for v in [a, b, c, d]],
                "normal": [round(float(v), 4) for v in normal],
                "inlier_count": len(inliers),
                "inlier_percentage": round(len(inliers) / len(points) * 100, 2),
                "type": plane_type
            })

            rest_pcd = rest_pcd.select_by_index(inliers, invert=True)

        return {"detected_planes": planes}

    def reconstruct_poisson_mesh(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        cam_centers: List[np.ndarray],
        depth: int = 9
    ) -> Tuple[o3d.geometry.TriangleMesh, o3d.geometry.TriangleMesh]:
        """Screened Poisson reconstruction returning (raw_untrimmed_mesh, postprocessed_mesh)."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

        # Estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.config.voxel_size * 4, max_nn=30)
        )

        # Orient normals towards average camera position
        mean_cam = np.mean(cam_centers, axis=0)
        pcd.orient_normals_towards_camera_location(mean_cam)

        # Poisson reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
        raw_untrimmed = o3d.geometry.TriangleMesh(mesh)
        raw_untrimmed.compute_vertex_normals()

        # Gentle density trimming
        dens = np.asarray(densities)
        trim_thresh = np.percentile(dens, self.config.density_trim_quantile * 100)
        mesh.remove_vertices_by_mask(dens < trim_thresh)

        # KDTree support distance trimming: prune vertices too far from genuine observation points
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        verts = np.asarray(mesh.vertices)
        max_dist = self.config.voxel_size * self.config.support_dist_multiplier

        unsupported_mask = np.zeros(len(verts), dtype=bool)
        for vi in range(len(verts)):
            [k, idx, d2] = kdtree.search_knn_vector_3d(verts[vi], 1)
            if k == 0 or np.sqrt(d2[0]) > max_dist:
                unsupported_mask[vi] = True

        mesh.remove_vertices_by_mask(unsupported_mask)

        # Clean topology & remove small disconnected floating components
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_non_manifold_edges()

        clusters, counts, _ = mesh.cluster_connected_triangles()
        counts = np.asarray(counts)
        keep_clusters = np.where(counts >= self.config.min_cluster_faces)[0]
        tri_mask = np.isin(np.asarray(clusters), keep_clusters)
        mesh.remove_triangles_by_mask(~tri_mask)
        mesh.remove_unreferenced_vertices()

        # Vertex color transfer from KDTree nearest neighbors
        mesh_verts = np.asarray(mesh.vertices)
        if len(mesh_verts) > 0:
            v_cols = np.zeros((len(mesh_verts), 3), dtype=np.float64)
            p_cols = np.asarray(pcd.colors)
            for vi in range(len(mesh_verts)):
                [k, idx, _] = kdtree.search_knn_vector_3d(mesh_verts[vi], 3)
                if k > 0:
                    v_cols[vi] = np.mean(p_cols[idx], axis=0)
            mesh.vertex_colors = o3d.utility.Vector3dVector(v_cols)

        mesh.compute_vertex_normals()
        return raw_untrimmed, mesh

    def reconstruct_tsdf_mesh(
        self,
        frames_bgr: List[np.ndarray],
        depth_maps: np.ndarray,
        confidence_maps: np.ndarray,
        extrinsics: np.ndarray,
        intrinsics: np.ndarray
    ) -> o3d.geometry.TriangleMesh:
        """Scalable TSDF volume integration for indoor geometry."""
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.config.tsdf_voxel_length,
            sdf_trunc=self.config.tsdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        N, H_inf, W_inf = depth_maps.shape[:3]
        for i in range(N):
            im_bgr = frames_bgr[i]
            im_rgb = cv2.cvtColor(cv2.resize(im_bgr, (W_inf, H_inf)), cv2.COLOR_BGR2RGB)
            d = depth_maps[i].squeeze()
            c = confidence_maps[i].squeeze()

            valid = (d >= self.config.min_depth) & (d <= self.config.max_depth) & (c >= self.config.min_confidence)
            d_clean = np.where(valid, d, 0.0).astype(np.float32)

            color_o3d = o3d.geometry.Image(im_rgb)
            depth_o3d = o3d.geometry.Image(d_clean)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d, depth_scale=1.0, depth_trunc=self.config.max_depth, convert_rgb_to_intensity=False
            )

            pinhole_int = o3d.camera.PinholeCameraIntrinsic(
                W_inf, H_inf,
                intrinsics[i, 0, 0], intrinsics[i, 1, 1],
                intrinsics[i, 0, 2], intrinsics[i, 1, 2]
            )
            ext4x4 = np.eye(4)
            ext4x4[:3, :] = extrinsics[i]
            volume.integrate(rgbd, pinhole_int, ext4x4)

        mesh = volume.extract_triangle_mesh()
        mesh.remove_non_manifold_edges()
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()

        clusters, counts, _ = mesh.cluster_connected_triangles()
        counts = np.asarray(counts)
        keep_clusters = np.where(counts >= self.config.min_cluster_faces)[0]
        tri_mask = np.isin(np.asarray(clusters), keep_clusters)
        mesh.remove_triangles_by_mask(~tri_mask)
        mesh.remove_unreferenced_vertices()

        mesh.compute_vertex_normals()
        return mesh

    def reconstruct_bpa_mesh(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        cam_centers: List[np.ndarray]
    ) -> o3d.geometry.TriangleMesh:
        """Multi-Scale Ball-Pivoting Algorithm (BPA)."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.config.voxel_size * 4, max_nn=30)
        )
        mean_cam = np.mean(cam_centers, axis=0)
        pcd.orient_normals_towards_camera_location(mean_cam)

        radii = o3d.utility.DoubleVector(self.config.bpa_radii)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)

        mesh.remove_non_manifold_edges()
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()

        clusters, counts, _ = mesh.cluster_connected_triangles()
        counts = np.asarray(counts)
        keep_clusters = np.where(counts >= self.config.min_cluster_faces)[0]
        tri_mask = np.isin(np.asarray(clusters), keep_clusters)
        mesh.remove_triangles_by_mask(~tri_mask)
        mesh.remove_unreferenced_vertices()

        mesh.compute_vertex_normals()
        return mesh

    @staticmethod
    def export_all_formats(
        mesh: o3d.geometry.TriangleMesh,
        out_dir: Path,
        base_name: str = "room_model"
    ) -> Dict[str, str]:
        """
        Exports PLY, OBJ/MTL, and glTF/GLB canonical upright models.
        Three.js coordinate alignment: T = diag(1, -1, -1).
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        results = {}

        # 1. Native PLY
        ply_path = out_dir / f"{base_name}.ply"
        o3d.io.write_triangle_mesh(str(ply_path), mesh, write_vertex_colors=True)
        results["ply"] = str(ply_path.resolve())

        # 2. Canonical Coordinate Alignment for Three.js WebGL
        # Three.js uses Y-up, right-handed.
        # Align mesh to canonical orientation
        canonical_mesh = o3d.geometry.TriangleMesh(mesh)
        # Flip Y and Z to convert from OpenCV camera coordinates to standard glTF/Three.js convention
        T_canonical = np.diag([1.0, -1.0, -1.0, 1.0])
        canonical_mesh.transform(T_canonical)

        # Center mesh at origin
        center = canonical_mesh.get_center()
        canonical_mesh.translate(-center)

        # 3. Canonical OBJ + MTL
        obj_path = out_dir / f"{base_name}_canonical.obj"
        o3d.io.write_triangle_mesh(str(obj_path), canonical_mesh, write_vertex_colors=True)
        results["obj"] = str(obj_path.resolve())

        # Also write a standard MTL file
        mtl_path = out_dir / f"{base_name}_canonical.mtl"
        with open(mtl_path, "w") as f:
            f.write("# F.O.R.G.E. Material Definition\n")
            f.write("newmtl material_0\n")
            f.write("Ka 0.200000 0.200000 0.200000\n")
            f.write("Kd 0.800000 0.800000 0.800000\n")
            f.write("Ks 0.100000 0.100000 0.100000\n")
            f.write("illum 2\n")
            f.write("Ns 10.000000\n")
        results["mtl"] = str(mtl_path.resolve())

        # 4. GLB Export using trimesh
        try:
            import trimesh
            verts = np.asarray(canonical_mesh.vertices)
            faces = np.asarray(canonical_mesh.triangles)
            v_colors = (np.asarray(canonical_mesh.vertex_colors) * 255.0).clip(0, 255).astype(np.uint8) if canonical_mesh.has_vertex_colors() else None

            tm = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=v_colors, process=False)
            glb_path = out_dir / f"{base_name}.glb"
            glb_bytes = tm.export(file_type="glb")
            with open(glb_path, "wb") as f:
                f.write(glb_bytes)
            results["glb"] = str(glb_path.resolve())
        except Exception as e:
            print(f"[!] GLB export notice: {e}")

        return results

    @staticmethod
    def render_visual_previews(
        mesh: o3d.geometry.TriangleMesh,
        out_dir: Path,
        prefix: str = "render"
    ) -> List[str]:
        """Renders multi-angle diagnostic previews of the mesh (Overview, Eye-Level, Oblique, Top-Down, Furniture Close-up)."""
        out_dir.mkdir(parents=True, exist_ok=True)
        saved_renders = []

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=1280, height=720)
        vis.add_geometry(mesh)

        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.94, 0.95, 0.96])
        opt.light_on = True

        ctr = vis.get_view_control()

        # Render 1: Front / Eye-level
        ctr.set_front([0.0, 0.0, -1.0])
        ctr.set_up([0.0, -1.0, 0.0])
        vis.poll_events()
        vis.update_renderer()
        img1 = out_dir / f"{prefix}_eye_level.png"
        vis.capture_screen_image(str(img1), do_render=True)
        saved_renders.append(str(img1))

        # Render 2: Oblique perspective
        ctr.set_front([0.6, -0.5, -0.6])
        ctr.set_up([0.0, -1.0, 0.0])
        vis.poll_events()
        vis.update_renderer()
        img2 = out_dir / f"{prefix}_oblique.png"
        vis.capture_screen_image(str(img2), do_render=True)
        saved_renders.append(str(img2))

        # Render 3: Top-down plan view
        ctr.set_front([0.0, -1.0, 0.0])
        ctr.set_up([0.0, 0.0, -1.0])
        vis.poll_events()
        vis.update_renderer()
        img3 = out_dir / f"{prefix}_top_plan.png"
        vis.capture_screen_image(str(img3), do_render=True)
        saved_renders.append(str(img3))

        # Render 4: Room Overview
        ctr.set_front([0.3, -0.4, -0.8])
        ctr.set_up([0.0, -1.0, 0.0])
        vis.poll_events()
        vis.update_renderer()
        img4 = out_dir / f"{prefix}_overview.png"
        vis.capture_screen_image(str(img4), do_render=True)
        saved_renders.append(str(img4))

        # Render 5: Furniture Close-up (Study desk & chair)
        ctr.set_front([0.4, -0.2, -0.5])
        ctr.set_up([0.0, -1.0, 0.0])
        ctr.set_zoom(0.55)
        vis.poll_events()
        vis.update_renderer()
        img5 = out_dir / f"{prefix}_furniture_closeup.png"
        vis.capture_screen_image(str(img5), do_render=True)
        saved_renders.append(str(img5))

        vis.destroy_window()
        return saved_renders

    @staticmethod
    def render_point_cloud_preview(
        pcd: o3d.geometry.PointCloud,
        out_dir: Path,
        save_name: str = "pcd_preview.png"
    ) -> str:
        """Renders offscreen diagnostic preview of point cloud."""
        out_dir.mkdir(parents=True, exist_ok=True)
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=1280, height=720)
        vis.add_geometry(pcd)

        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.94, 0.95, 0.96])
        opt.point_size = 2.5

        ctr = vis.get_view_control()
        ctr.set_front([0.6, -0.5, -0.6])
        ctr.set_up([0.0, -1.0, 0.0])
        vis.poll_events()
        vis.update_renderer()

        out_file = out_dir / save_name
        vis.capture_screen_image(str(out_file), do_render=True)
        vis.destroy_window()
        return str(out_file)

