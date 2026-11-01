"""
F.O.R.G.E. — V3 Geometry Meshing Benchmark
Dataset: November 1, 2026 — 19:10:43 (Hostel Room B 314)
Benchmarks:
1. High-Density Screened Poisson (Voxel=8mm, Depth=9)
2. Ultra-Density Screened Poisson (Voxel=8mm, Depth=10)
3. Scalable TSDF Volume Integration (Voxel=15mm, Trunc=30mm)
4. Multi-Scale Ball-Pivoting Algorithm (BPA)
"""

import os
import sys
import time
import json
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d

BASE_DIR = Path(r"C:\Users\25beevdt025\Desktop\FORGE")
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.reconstruction.point_cloud import depth_to_points, voxel_grid_fusion

OUTPUT_V1 = BASE_DIR / "outputs" / "indoor" / "texture_experiment_video_20261101_191043"
CAM_DIR = OUTPUT_V1 / "camera_geometry"
FRAMES_DIR = OUTPUT_V1 / "frames"

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def analyze_mesh(mesh: o3d.geometry.TriangleMesh, name: str, runtime_sec: float) -> dict:
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    
    # Boundary edges & manifoldness
    is_edge_manifold = mesh.is_edge_manifold(allow_boundary_edges=True)
    is_vertex_manifold = mesh.is_vertex_manifold()
    is_watertight = mesh.is_watertight()
    
    # Boundary edges count
    # In Open3D, mesh.get_non_manifold_edges(allow_boundary_edges=False) returns all non-manifold + boundary edges
    # mesh.get_non_manifold_edges(allow_boundary_edges=True) returns strictly non-manifold edges
    all_abnormal = mesh.get_non_manifold_edges(allow_boundary_edges=False)
    strict_non_manifold = mesh.get_non_manifold_edges(allow_boundary_edges=True)
    boundary_edges_count = len(all_abnormal) - len(strict_non_manifold)
    
    clusters, counts, _ = mesh.cluster_connected_triangles()
    num_components = len(counts)
    
    bbox_min = verts.min(axis=0) if len(verts) > 0 else np.zeros(3)
    bbox_max = verts.max(axis=0) if len(verts) > 0 else np.zeros(3)
    span = bbox_max - bbox_min
    
    try:
        surface_area = mesh.get_surface_area()
    except Exception:
        surface_area = 0.0

    stats = {
        "algorithm": name,
        "runtime_sec": round(runtime_sec, 2),
        "vertices": int(len(verts)),
        "triangles": int(len(tris)),
        "is_watertight": bool(is_watertight),
        "is_edge_manifold": bool(is_edge_manifold),
        "is_vertex_manifold": bool(is_vertex_manifold),
        "boundary_edges": int(boundary_edges_count),
        "strict_non_manifold_edges": int(len(strict_non_manifold)),
        "connected_components": int(num_components),
        "surface_area_m2": round(float(surface_area), 3),
        "bounds_min": [round(float(v), 3) for v in bbox_min],
        "bounds_max": [round(float(v), 3) for v in bbox_max],
        "span_xyz": [round(float(v), 3) for v in span],
    }
    return stats

def main():
    log("=" * 76)
    log("  F.O.R.G.E. V3 GEOMETRY BENCHMARK: MULTI-ALGORITHM EVALUATION")
    log("  Hostel Room B 314 — Dataset: 2026-11-01 19:10:43")
    log("=" * 76)

    # 1. Load Camera Geometries & Depth Maps
    poses_path = CAM_DIR / "camera_poses.npz"
    depths_path = CAM_DIR / "depth_predictions.npz"
    poses = np.load(poses_path)
    depths = np.load(depths_path)
    
    extrinsics = poses["extrinsics"] # (60, 3, 4)
    intrinsics = poses["intrinsics"] # (60, 3, 3)
    depth_maps = depths["depth"]     # (60, 432, 240, 1)
    conf_maps = depths["confidence"] # (60, 432, 240)
    
    frame_paths = sorted(list(FRAMES_DIR.glob("*.png")))
    frames_bgr = [cv2.imread(str(p)) for p in frame_paths]
    N, H_inf, W_inf = depth_maps.shape[:3]
    
    log(f"Loaded {N} camera poses and depth maps ({W_inf}x{H_inf}).")

    # Camera centers
    cam_centers = []
    for i in range(N):
        R = extrinsics[i, :3, :3]
        t = extrinsics[i, :3, 3]
        cam_centers.append(-R.T @ t)
    cam_centers_np = np.array(cam_centers)

    # Backproject dense points (high quality mask)
    log("Backprojecting raw depth maps with confidence >= 0.75 and depth in [0.25, 5.5]m...")
    all_pts = []
    all_cols = []
    for i in range(N):
        d = depth_maps[i].squeeze()
        c = conf_maps[i].squeeze()
        im_rgb = cv2.cvtColor(cv2.resize(frames_bgr[i], (W_inf, H_inf)), cv2.COLOR_BGR2RGB)
        valid = (d >= 0.25) & (d <= 5.5) & (c >= 0.75)
        if not np.any(valid):
            continue
        pts_3d = depth_to_points(d, extrinsics[i], intrinsics[i])
        all_pts.append(pts_3d[valid])
        all_cols.append(im_rgb[valid])

    merged_pts = np.vstack(all_pts)
    merged_cols = np.vstack(all_cols)
    log(f"Total valid un-downsampled points: {len(merged_pts):,}")

    # Test two voxel sizes: 0.008 (8mm, V3 high fidelity) and 0.015 (15mm, baseline V1)
    log("Applying scale-aware voxel fusion at voxel_size=0.008m (8mm)...")
    t0 = time.perf_counter()
    pcd_8mm = o3d.geometry.PointCloud()
    pcd_8mm.points = o3d.utility.Vector3dVector(merged_pts.astype(np.float64))
    pcd_8mm.colors = o3d.utility.Vector3dVector(merged_cols.astype(np.float64) / 255.0)
    pcd_8mm = pcd_8mm.voxel_down_sample(voxel_size=0.008)
    
    # Statistical outlier removal with milder threshold (std_ratio=2.2 instead of 1.8) to preserve fine furniture
    pcd_8mm_clean, ind = pcd_8mm.remove_statistical_outlier(nb_neighbors=25, std_ratio=2.2)
    pts_8mm = np.asarray(pcd_8mm_clean.points)
    cols_8mm = (np.asarray(pcd_8mm_clean.colors) * 255.0).clip(0, 255).astype(np.uint8)
    log(f"8mm filtered point cloud: {len(pts_8mm):,} points (retained {len(pts_8mm)/69007:.2f}x more points than V1's 69,007 points!)")

    # Estimate Ray-Oriented Normals for 8mm point cloud
    log("Estimating ray-oriented normal field...")
    pcd_8mm_clean.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=30))
    normals = np.asarray(pcd_8mm_clean.normals)
    dists_to_cams = np.linalg.norm(pts_8mm[:, None, :] - cam_centers_np[None, :, :], axis=-1)
    closest_cam_idx = np.argmin(dists_to_cams, axis=1)
    view_rays = cam_centers_np[closest_cam_idx] - pts_8mm
    view_rays /= (np.linalg.norm(view_rays, axis=1, keepdims=True) + 1e-6)
    dot_prod = np.sum(normals * view_rays, axis=1)
    flip_mask = dot_prod < 0.0
    normals[flip_mask] = -normals[flip_mask]
    pcd_8mm_clean.normals = o3d.utility.Vector3dVector(normals)
    log(f"Normal field oriented ({flip_mask.sum():,} flipped towards cameras).")

    results = []

    # -------------------------------------------------------------
    # Method 1: Screened Poisson (depth=9, voxel=8mm, density trim 0.012)
    # -------------------------------------------------------------
    log("\n[Benchmark 1/4] Screened Poisson (depth=9, voxel=8mm)...")
    t0_m1 = time.perf_counter()
    mesh1, dens1 = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_8mm_clean, depth=9, linear_fit=True)
    dens1_arr = np.asarray(dens1)
    thresh1 = np.percentile(dens1_arr, 1.2) # prune only lowest 1.2% density
    mesh1.remove_vertices_by_mask(dens1_arr < thresh1)
    mesh1.remove_degenerate_triangles()
    mesh1.remove_duplicated_vertices()
    mesh1.remove_duplicated_triangles()
    mesh1.remove_non_manifold_edges()
    clusters, counts, _ = mesh1.cluster_connected_triangles()
    counts = np.asarray(counts)
    keep_clusters = np.where(counts >= 150)[0]
    mesh1.remove_triangles_by_mask(~np.isin(np.asarray(clusters), keep_clusters))
    mesh1.remove_unreferenced_vertices()
    dur_m1 = time.perf_counter() - t0_m1
    stats1 = analyze_mesh(mesh1, "Screened Poisson (depth=9, 8mm)", dur_m1)
    log(f"  -> Vertices: {stats1['vertices']:,}, Triangles: {stats1['triangles']:,}, Boundary Edges: {stats1['boundary_edges']:,}, Components: {stats1['connected_components']}")
    results.append(stats1)

    # -------------------------------------------------------------
    # Method 2: Screened Poisson (depth=10, voxel=8mm, density trim 0.012)
    # -------------------------------------------------------------
    log("\n[Benchmark 2/4] Screened Poisson (depth=10, voxel=8mm)...")
    t0_m2 = time.perf_counter()
    mesh2, dens2 = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_8mm_clean, depth=10, linear_fit=True)
    dens2_arr = np.asarray(dens2)
    thresh2 = np.percentile(dens2_arr, 1.2)
    mesh2.remove_vertices_by_mask(dens2_arr < thresh2)
    mesh2.remove_degenerate_triangles()
    mesh2.remove_duplicated_vertices()
    mesh2.remove_duplicated_triangles()
    mesh2.remove_non_manifold_edges()
    clusters, counts, _ = mesh2.cluster_connected_triangles()
    counts = np.asarray(counts)
    keep_clusters = np.where(counts >= 150)[0]
    mesh2.remove_triangles_by_mask(~np.isin(np.asarray(clusters), keep_clusters))
    mesh2.remove_unreferenced_vertices()
    dur_m2 = time.perf_counter() - t0_m2
    stats2 = analyze_mesh(mesh2, "Screened Poisson (depth=10, 8mm)", dur_m2)
    log(f"  -> Vertices: {stats2['vertices']:,}, Triangles: {stats2['triangles']:,}, Boundary Edges: {stats2['boundary_edges']:,}, Components: {stats2['connected_components']}")
    results.append(stats2)

    # -------------------------------------------------------------
    # Method 3: Scalable TSDF Volume Integration
    # -------------------------------------------------------------
    log("\n[Benchmark 3/4] Scalable TSDF Volume Integration (voxel=15mm, trunc=35mm)...")
    t0_m3 = time.perf_counter()
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.015,
        sdf_trunc=0.035,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    for i in range(N):
        d = depth_maps[i].squeeze()
        c = conf_maps[i].squeeze()
        im_rgb = cv2.cvtColor(cv2.resize(frames_bgr[i], (W_inf, H_inf)), cv2.COLOR_BGR2RGB)
        valid = (d >= 0.25) & (d <= 5.5) & (c >= 0.75)
        d_clean = np.where(valid, d, 0.0).astype(np.float32)
        color_o3d = o3d.geometry.Image(im_rgb)
        depth_o3d = o3d.geometry.Image(d_clean)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d, depth_scale=1.0, depth_trunc=5.5, convert_rgb_to_intensity=False
        )
        pinhole_int = o3d.camera.PinholeCameraIntrinsic(
            W_inf, H_inf,
            intrinsics[i, 0, 0], intrinsics[i, 1, 1],
            intrinsics[i, 0, 2], intrinsics[i, 1, 2]
        )
        ext4x4 = np.eye(4)
        ext4x4[:3, :] = extrinsics[i]
        volume.integrate(rgbd, pinhole_int, ext4x4)
    mesh3 = volume.extract_triangle_mesh()
    mesh3.remove_non_manifold_edges()
    mesh3.remove_degenerate_triangles()
    mesh3.remove_duplicated_vertices()
    mesh3.remove_duplicated_triangles()
    clusters, counts, _ = mesh3.cluster_connected_triangles()
    counts = np.asarray(counts)
    if len(counts) > 0:
        keep_clusters = np.where(counts >= 150)[0]
        mesh3.remove_triangles_by_mask(~np.isin(np.asarray(clusters), keep_clusters))
        mesh3.remove_unreferenced_vertices()
    dur_m3 = time.perf_counter() - t0_m3
    stats3 = analyze_mesh(mesh3, "Scalable TSDF Volume (15mm)", dur_m3)
    log(f"  -> Vertices: {stats3['vertices']:,}, Triangles: {stats3['triangles']:,}, Boundary Edges: {stats3['boundary_edges']:,}, Components: {stats3['connected_components']}")
    results.append(stats3)

    # -------------------------------------------------------------
    # Method 4: Multi-Scale Ball-Pivoting Algorithm (BPA)
    # -------------------------------------------------------------
    log("\n[Benchmark 4/4] Multi-Scale Ball-Pivoting Algorithm (BPA)...")
    t0_m4 = time.perf_counter()
    radii = o3d.utility.DoubleVector([0.008, 0.016, 0.032, 0.064])
    mesh4 = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd_8mm_clean, radii)
    mesh4.remove_non_manifold_edges()
    mesh4.remove_degenerate_triangles()
    mesh4.remove_duplicated_vertices()
    mesh4.remove_duplicated_triangles()
    clusters, counts, _ = mesh4.cluster_connected_triangles()
    counts = np.asarray(counts)
    if len(counts) > 0:
        keep_clusters = np.where(counts >= 150)[0]
        mesh4.remove_triangles_by_mask(~np.isin(np.asarray(clusters), keep_clusters))
        mesh4.remove_unreferenced_vertices()
    dur_m4 = time.perf_counter() - t0_m4
    stats4 = analyze_mesh(mesh4, "Multi-Scale BPA (8-64mm)", dur_m4)
    log(f"  -> Vertices: {stats4['vertices']:,}, Triangles: {stats4['triangles']:,}, Boundary Edges: {stats4['boundary_edges']:,}, Components: {stats4['connected_components']}")
    results.append(stats4)

    # Summary
    log("\n" + "=" * 76)
    log("  BENCHMARK SUMMARY RESULTS TABLE")
    log("=" * 76)
    fmt = "{:<32} | {:>9} | {:>9} | {:>14} | {:>10} | {:>8}"
    log(fmt.format("Algorithm", "Vertices", "Triangles", "Boundary Edges", "Components", "Time (s)"))
    log("-" * 92)
    for s in results:
        log(fmt.format(
            s["algorithm"],
            f"{s['vertices']:,}",
            f"{s['triangles']:,}",
            f"{s['boundary_edges']:,}",
            s["connected_components"],
            f"{s['runtime_sec']:.1f}s"
        ))

    # Save benchmark json to scripts or outputs
    out_json = BASE_DIR / "scripts" / "geometry_algorithm_benchmark_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nSaved benchmark results to {out_json}")

if __name__ == "__main__":
    main()
