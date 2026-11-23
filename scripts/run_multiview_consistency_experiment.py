"""
F.O.R.G.E. — Isolated Multi-View Geometry-Consistency Experiment.

Runs rigorous baseline-versus-experiment comparative benchmarking across all 4 release scenes:
1. Campus_0049
2. River_Bridge
3. Desert_Road
4. Forest_Road

Preserves all baseline models and outputs. Evaluates:
- Geometric consistency rate (%)
- Outlier removal (% of points pruned)
- Mesh completeness (surface area, face count, volume)
- Disconnected components and floating fragments
- Lost legitimate structures (structural bbox IoU, core elevation preservation)
- Processing time and Peak System RAM / GPU VRAM
All evaluation measures lacking ground-truth survey points are explicitly labeled as PROXIES.
"""

from __future__ import annotations

import os
import sys
import time
import json
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import open3d as o3d
import trimesh
import psutil
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.reconstruction.multiview_consistency import (
    MultiViewConsistencyConfig,
    MultiViewConsistencyResult,
    filter_point_cloud_multiview
)
from src.reconstruction.point_cloud import depth_to_points

SCENES = {
    "campus_0049": {
        "pred_dir": PROJECT_ROOT / "outputs/opencv5_experiments/final_models/campus_0049/predictions",
        "frames_dir": PROJECT_ROOT / "outputs/opencv5_experiments/final_models/campus_0049/frames",
        "description": "Dense urban campus with building facades, roof fixtures, and courtyard",
    },
    "river_bridge": {
        "pred_dir": PROJECT_ROOT / "outputs/opencv5_experiments/final_models/river_bridge/predictions",
        "frames_dir": PROJECT_ROOT / "outputs/opencv5_experiments/final_models/river_bridge/frames",
        "description": "Linear infrastructure over water with thin bridge deck and suspension pillars",
    },
    "desert_road": {
        "pred_dir": PROJECT_ROOT / "outputs/opencv5_experiments/final_models/desert_road/predictions",
        "frames_dir": PROJECT_ROOT / "outputs/opencv5_experiments/final_models/desert_road/frames",
        "description": "Arid terrain highway with moving vehicle dynamic exclusion",
    },
    "forest_road": {
        "pred_dir": PROJECT_ROOT / "outputs/opencv5_experiments/final_models/forest_road/predictions",
        "frames_dir": PROJECT_ROOT / "outputs/opencv5_experiments/final_models/forest_road/frames",
        "description": "Canopy-lined rural asphalt road with semi-transparent foliage boundaries",
    },
}

EXP_OUT_ROOT = PROJECT_ROOT / "outputs/experiments_multiview_consistency"
EXP_OUT_ROOT.mkdir(parents=True, exist_ok=True)


def get_peak_memory() -> Dict[str, float]:
    """Returns current process RAM and GPU VRAM in megabytes."""
    mem_info = psutil.Process().memory_info()
    ram_mb = mem_info.rss / (1024 * 1024)

    vram_mb = 0.0
    try:
        import torch
        if torch.cuda.is_available():
            vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        pass

    return {"ram_mb": round(ram_mb, 2), "vram_mb": round(vram_mb, 2)}


def load_scene_data(scene_key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], Optional[np.ndarray]]:
    """Loads camera poses, depth maps, frames, and confidences for a scene."""
    cfg = SCENES[scene_key]
    pred_dir = cfg["pred_dir"]
    frames_dir = cfg["frames_dir"]

    poses_path = pred_dir / "camera_poses.npz"
    depth_path = pred_dir / "depth_predictions.npz"

    if not poses_path.exists() or not depth_path.exists():
        raise FileNotFoundError(f"Missing predictions for scene {scene_key} in {pred_dir}")

    poses_data = np.load(str(poses_path))
    extrinsics = poses_data["extrinsics"] # (N, 3, 4)
    intrinsics = poses_data["intrinsics"] # (N, 3, 3)

    depth_data = np.load(str(depth_path))
    depth = depth_data["depth"] # (N, H, W, 1) or (N, H, W)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]

    confidences = None
    if "confidence" in depth_data.files:
        confidences = depth_data["confidence"]

    # Load frames
    frame_files = sorted(glob.glob(str(frames_dir / "*.png")))
    frames = []
    for f in frame_files[:len(extrinsics)]:
        img = Image.open(f).convert("RGB")
        # Resize to match depth map dimensions if necessary
        if (img.width, img.height) != (depth.shape[2], depth.shape[1]):
            img = img.resize((depth.shape[2], depth.shape[1]), Image.Resampling.BILINEAR)
        frames.append(np.array(img))

    return depth, extrinsics, intrinsics, frames, confidences


def generate_baseline_point_cloud(
    depth: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    frames: List[np.ndarray],
    confidences: Optional[np.ndarray] = None,
    min_confidence: float = 0.15
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates standard unfiltered baseline point cloud by straightforward backprojection."""
    N, H, W = depth.shape
    v_grid, u_grid = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    u_flat = u_grid.flatten()
    v_flat = v_grid.flatten()

    pts_list = []
    cols_list = []

    for i in range(N):
        d_i = depth[i].flatten()
        valid = np.isfinite(d_i) & (d_i > 0.05)
        if confidences is not None:
            valid &= (confidences[i].flatten() >= min_confidence)

        if not np.any(valid):
            continue

        u_val = u_flat[valid]
        v_val = v_flat[valid]
        z_val = d_i[valid]

        fx = intrinsics[i, 0, 0]
        fy = intrinsics[i, 1, 1]
        cx = intrinsics[i, 0, 2]
        cy = intrinsics[i, 1, 2]

        x_cam = (u_val - cx) / fx * z_val
        y_cam = (v_val - cy) / fy * z_val
        pts_cam = np.column_stack([x_cam, y_cam, z_val])

        # World = R^T * (P_cam - T)
        R = extrinsics[i, :3, :3]
        T = extrinsics[i, :3, 3]
        pts_world = np.dot(pts_cam - T, R)

        pts_list.append(pts_world)
        if i < len(frames):
            frame_norm = frames[i].astype(np.float64) / 255.0
            cols_list.append(frame_norm[v_val, u_val])
        else:
            cols_list.append(np.full((len(pts_world), 3), 0.7))

    return np.vstack(pts_list), np.vstack(cols_list)


def reconstruct_mesh_poisson(
    points: np.ndarray,
    colors: np.ndarray,
    poisson_depth: int = 8,
    density_quantile: float = 0.03
) -> Tuple[o3d.geometry.TriangleMesh, Dict[str, Any]]:
    """Runs Screened Poisson surface reconstruction and returns mesh and telemetry."""
    t0 = time.perf_counter()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Estimate and orient normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k=15)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth
    )

    if density_quantile > 0.0 and len(densities) > 0:
        dens_np = np.asarray(densities)
        thresh = np.quantile(dens_np, density_quantile)
        mesh.remove_vertices_by_mask(dens_np < thresh)

    duration = time.perf_counter() - t0

    num_verts = len(mesh.vertices)
    num_faces = len(mesh.triangles)

    area = float(mesh.get_surface_area()) if num_faces > 0 else 0.0
    bbox = mesh.get_axis_aligned_bounding_box()
    vol = float(bbox.volume())

    return mesh, {
        "vertices": num_verts,
        "faces": num_faces,
        "surface_area_m2_proxy": round(area, 2),
        "bounding_volume_m3_proxy": round(vol, 2),
        "meshing_time_s": round(duration, 3)
    }


def analyze_mesh_components(mesh: o3d.geometry.TriangleMesh) -> Dict[str, Any]:
    """Analyzes connected components and floating fragments in mesh."""
    if len(mesh.triangles) == 0:
        return {"total_components": 0, "fragment_components": 0, "largest_component_faces": 0}

    triangle_clusters, num_triangles, _ = mesh.cluster_connected_triangles()
    num_clusters = len(num_triangles)

    fragment_count = sum(1 for cnt in num_triangles if cnt < 50)
    largest_count = max(num_triangles) if num_triangles else 0

    return {
        "total_components": num_clusters,
        "fragment_components": fragment_count,
        "largest_component_faces": int(largest_count),
    }


def compute_structural_preservation_proxy(
    base_points: np.ndarray,
    exp_points: np.ndarray
) -> Dict[str, Any]:
    """
    Computes proxy structural preservation metrics comparing baseline and experimental point clouds.
    Explicitly labeled as proxy because physical ground truth coordinates are unavailable.
    """
    if len(base_points) == 0 or len(exp_points) == 0:
        return {"bbox_iou_proxy": 0.0, "core_z_range_ratio_proxy": 0.0}

    # 1. Bounding box IoU proxy
    b_min, b_max = np.min(base_points, axis=0), np.max(base_points, axis=0)
    e_min, e_max = np.min(exp_points, axis=0), np.max(exp_points, axis=0)

    i_min = np.maximum(b_min, e_min)
    i_max = np.minimum(b_max, e_max)
    inter_dims = np.maximum(0.0, i_max - i_min)
    inter_vol = np.prod(inter_dims)

    vol_base = np.prod(np.maximum(0.0, b_max - b_min))
    vol_exp = np.prod(np.maximum(0.0, e_max - e_min))
    union_vol = vol_base + vol_exp - inter_vol
    bbox_iou = float(inter_vol / max(union_vol, 1e-6))

    # 2. Core elevation range preservation (5th to 95th percentile Z)
    base_z5, base_z95 = np.percentile(base_points[:, 2], [5, 95])
    exp_z5, exp_z95 = np.percentile(exp_points[:, 2], [5, 95])

    base_range = max(base_z95 - base_z5, 0.01)
    exp_range = max(exp_z95 - exp_z5, 0.01)
    z_range_ratio = float(exp_range / base_range)

    return {
        "bbox_iou_proxy": round(bbox_iou, 4),
        "core_z_range_ratio_proxy": round(z_range_ratio, 4),
        "base_z5_to_z95_m_proxy": round(base_range, 2),
        "exp_z5_to_z95_m_proxy": round(exp_range, 2)
    }


def run_single_scene_experiment(
    scene_key: str,
    filter_config: MultiViewConsistencyConfig
) -> Dict[str, Any]:
    """Runs isolated comparative experiment for one scene."""
    scene_out = EXP_OUT_ROOT / scene_key
    scene_out.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f"Running Multi-View Consistency Experiment: {scene_key}")
    print(f"Description: {SCENES[scene_key]['description']}")
    print(f"=======================================================")

    depth, exts, intrs, frames, confs = load_scene_data(scene_key)

    # 1. Baseline Reconstruction
    t_base_0 = time.perf_counter()
    base_pts, base_cols = generate_baseline_point_cloud(
        depth, exts, intrs, frames, confidences=confs, min_confidence=filter_config.min_confidence
    )
    t_base_pcd = time.perf_counter() - t_base_0

    base_mesh, base_mesh_stats = reconstruct_mesh_poisson(base_pts, base_cols, poisson_depth=8)
    base_components = analyze_mesh_components(base_mesh)
    base_mem = get_peak_memory()

    # Save baseline artifacts in experimental folder (never touch final_models/)
    base_pcd_o3d = o3d.geometry.PointCloud()
    base_pcd_o3d.points = o3d.utility.Vector3dVector(base_pts)
    base_pcd_o3d.colors = o3d.utility.Vector3dVector(base_cols)
    o3d.io.write_point_cloud(str(scene_out / "baseline_point_cloud.ply"), base_pcd_o3d)
    o3d.io.write_triangle_mesh(str(scene_out / "baseline_mesh.ply"), base_mesh)

    # 2. Experimental Multi-View Consistency Filter
    t_exp_0 = time.perf_counter()
    exp_pts, exp_cols, filter_res = filter_point_cloud_multiview(
        depth_maps=depth,
        camera_poses_extrinsics=exts,
        camera_intrinsics=intrs,
        color_frames=frames,
        confidences=confs,
        config=filter_config
    )
    t_exp_filter = time.perf_counter() - t_exp_0

    exp_mesh, exp_mesh_stats = reconstruct_mesh_poisson(exp_pts, exp_cols, poisson_depth=8)
    exp_components = analyze_mesh_components(exp_mesh)
    exp_mem = get_peak_memory()

    # Save experimental artifacts
    exp_pcd_o3d = o3d.geometry.PointCloud()
    exp_pcd_o3d.points = o3d.utility.Vector3dVector(exp_pts)
    exp_pcd_o3d.colors = o3d.utility.Vector3dVector(exp_cols)
    o3d.io.write_point_cloud(str(scene_out / "experimental_point_cloud.ply"), exp_pcd_o3d)
    o3d.io.write_triangle_mesh(str(scene_out / "experimental_mesh.ply"), exp_mesh)

    # 3. Compute Comparative Metrics
    structural_metrics = compute_structural_preservation_proxy(base_pts, exp_pts)

    # Statistical Outlier Rejection (SOR) proxy on retained clouds
    def compute_sor_outlier_pct(pcd_o3d: o3d.geometry.PointCloud) -> float:
        if len(pcd_o3d.points) < 50:
            return 0.0
        _, inliers = pcd_o3d.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        return round(100.0 * (1.0 - len(inliers) / len(pcd_o3d.points)), 2)

    base_sor_pct = compute_sor_outlier_pct(base_pcd_o3d)
    exp_sor_pct = compute_sor_outlier_pct(exp_pcd_o3d)

    # Compile scene summary
    scene_report = {
        "scene": scene_key,
        "description": SCENES[scene_key]["description"],
        "filter_config": filter_res.config,
        "point_cloud_metrics": {
            "baseline_point_count": len(base_pts),
            "experimental_point_count": len(exp_pts),
            "pruned_points_count": filter_res.pruned_points_count,
            "pruned_points_pct": filter_res.pruned_points_pct,
            "geometric_consistency_rate_pct": round(100.0 - filter_res.pruned_points_pct, 2),
            "baseline_sor_noise_pct_proxy": base_sor_pct,
            "experimental_sor_noise_pct_proxy": exp_sor_pct,
            "sor_noise_reduction_pct_proxy": round(base_sor_pct - exp_sor_pct, 2),
            "per_frame_retained_pct": filter_res.per_frame_retained_pct
        },
        "mesh_metrics": {
            "baseline": {
                **base_mesh_stats,
                **base_components
            },
            "experimental": {
                **exp_mesh_stats,
                **exp_components
            },
            "mesh_face_retention_pct": round(100.0 * exp_mesh_stats["faces"] / max(base_mesh_stats["faces"], 1), 2),
            "fragment_components_reduction": base_components["fragment_components"] - exp_components["fragment_components"]
        },
        "structural_preservation_proxy": structural_metrics,
        "telemetry_and_resources": {
            "filter_execution_time_s": round(t_exp_filter, 3),
            "baseline_meshing_time_s": base_mesh_stats["meshing_time_s"],
            "experimental_meshing_time_s": exp_mesh_stats["meshing_time_s"],
            "peak_ram_mb": exp_mem["ram_mb"],
            "peak_vram_mb": exp_mem["vram_mb"]
        }
    }

    # Save per-scene json
    with open(scene_out / "experiment_metrics.json", "w") as f:
        json.dump(scene_report, f, indent=2)

    print(f"Results for {scene_key}:")
    print(f"  Points: Baseline={len(base_pts):,} -> Exp={len(exp_pts):,} (Pruned {filter_res.pruned_points_pct}%)")
    print(f"  Faces:  Baseline={base_mesh_stats['faces']:,} -> Exp={exp_mesh_stats['faces']:,}")
    print(f"  Floating Fragments: Baseline={base_components['fragment_components']} -> Exp={exp_components['fragment_components']}")
    print(f"  Bbox IoU (structural proxy): {structural_metrics['bbox_iou_proxy']}")
    print(f"  Filter time: {t_exp_filter:.2f}s | RAM: {exp_mem['ram_mb']} MB")

    return scene_report


def main():
    print("=" * 70)
    print("F.O.R.G.E. Multi-View Geometry-Consistency Isolated Experiment")
    print("Benchmarking across Campus_0049, River_Bridge, Desert_Road, Forest_Road")
    print("=" * 70)

    # Configurable experimental filtering parameters
    # Conservative, photogrammetrically defensible settings:
    # 4% relative depth error tolerance, 2.5 pixel reprojection tolerance, min 1 confirming overlapping view
    filter_config = MultiViewConsistencyConfig(
        depth_relative_thresh=0.04,
        reproj_error_thresh_pix=2.5,
        min_consistent_views=1,
        min_confidence=0.15,
        downsample_voxel_m=None  # preserve raw resolution
    )

    all_results = {}
    for scene_key in ["campus_0049", "river_bridge", "desert_road", "forest_road"]:
        res = run_single_scene_experiment(scene_key, filter_config)
        all_results[scene_key] = res

    # Master report
    master_json_path = EXP_OUT_ROOT / "master_experiment_report.json"
    with open(master_json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll 4 scenes evaluated successfully!")
    print(f"Master experimental report written to: {master_json_path}")


if __name__ == "__main__":
    main()
