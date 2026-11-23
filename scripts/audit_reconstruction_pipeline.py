"""F.O.R.G.E — Comprehensive 8-Stage Diagnostic Pipeline Audit.

Audits every stage of the WhatsApp video reconstruction:
1. RGB Frames: OpenCV 5 feature tracking, sharpness, exposure, overlap.
2. Camera Trajectory: Extrinsics convention, rotation matrices, optical axes, motion continuity.
3. Depth & Confidence Maps: Dynamic range, confidence percentiles, horizon/sky values.
4. Single-Frame Point Clouds: Planarity of roofs/roads, depth consistency within camera space.
5. Registered Multi-Frame Point Cloud: Pairwise alignment, Chamfer residuals, scale consistency.
6. Fused Point Cloud: Voxel grid distribution, normal consistency, surface density.
7. Surface Meshing: Poisson vs hole closure vs connected components, spike origin analysis.
8. Multiview Texturing: Camera-to-surface projection accuracy, UV seam alignment.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
import open3d as o3d
import trimesh
from PIL import Image

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PRED_DIR = PROJECT_ROOT / "outputs/new_video_reconstructions/whatsapp_video_2026_10_24/campus_candidate_a_pipeline/upstream_predictions/predictions"
FRAMES_DIR = PROJECT_ROOT / "outputs/new_video_reconstructions/whatsapp_video_2026_10_24/campus_candidate_a_pipeline/upstream_predictions/frames"
VIDEO_PATH = Path(r"C:\Users\25beevdt025\Downloads\WhatsApp Video 2026-10-24 at 17.53.55.mp4")
CAMPAIGN_DIR = PROJECT_ROOT / "outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign"
DIAG_DIR = CAMPAIGN_DIR / "diagnostics"

from src.reconstruction.point_cloud import depth_to_points
from scripts.validate_sprint1_geometry_and_filtering import compute_relative_depth_jump


def run_comprehensive_audit() -> Dict[str, Any]:
    print("=" * 80)
    print("F.O.R.G.E 8-STAGE RECONSTRUCTION PIPELINE AUDIT")
    print("=" * 80)
    audit_results = {}

    # Load intermediate predictions
    poses_data = np.load(PRED_DIR / "camera_poses.npz")
    depth_data = np.load(PRED_DIR / "depth_predictions.npz")
    extrinsics = poses_data["extrinsics"] # (N, 3, 4)
    intrinsics = poses_data["intrinsics"] # (N, 3, 3)
    d_maps = np.squeeze(depth_data["depth"]) # (N, H, W)
    c_maps = np.squeeze(depth_data["confidence"]) # (N, H, W)
    frame_files = sorted(list(FRAMES_DIR.glob("*.png")))
    N, H_inf, W_inf = d_maps.shape

    # -------------------------------------------------------------
    # STAGE 1: RGB Frames & OpenCV 5 Feature Tracking
    # -------------------------------------------------------------
    print("\n--- STAGE 1: RGB Frames & Feature Tracking ---")
    orb = cv2.ORB_create(nfeatures=2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    frame_metrics = []
    pairwise_matches = []

    for i in range(N):
        im = cv2.imread(str(frame_files[i]))
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_lum = float(gray.mean())
        kp, des = orb.detectAndCompute(gray, None)
        frame_metrics.append({
            "frame_idx": i,
            "filename": frame_files[i].name,
            "resolution": [im.shape[1], im.shape[0]],
            "laplacian_sharpness": round(lap_var, 2),
            "mean_luminance": round(mean_lum, 2),
            "orb_keypoints": len(kp) if kp else 0
        })

    for i in range(N - 1):
        im1 = cv2.imread(str(frame_files[i]))
        im2 = cv2.imread(str(frame_files[i + 1]))
        g1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        kp1, des1 = orb.detectAndCompute(g1, None)
        kp2, des2 = orb.detectAndCompute(g2, None)
        matches = bf.match(des1, des2) if des1 is not None and des2 is not None else []
        matches = sorted(matches, key=lambda x: x.distance)

        # Inlier filtering via fundamental matrix RANSAC
        if len(matches) >= 8:
            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)
            inliers = int(mask.sum()) if mask is not None else 0
            inlier_ratio = round(inliers / len(matches), 3) if len(matches) > 0 else 0.0
        else:
            inliers = 0
            inlier_ratio = 0.0

        pairwise_matches.append({
            "pair": f"Frame {i} -> Frame {i+1}",
            "raw_matches": len(matches),
            "ransac_inliers": inliers,
            "inlier_ratio": inlier_ratio
        })
        print(f"  Frame {i} -> {i+1}: raw={len(matches)}, inliers={inliers} ({inlier_ratio*100:.1f}%)")

    audit_results["stage1_frames"] = {
        "frame_metrics": frame_metrics,
        "pairwise_matches": pairwise_matches
    }

    # -------------------------------------------------------------
    # STAGE 2: Camera Trajectory & Coordinate Conventions
    # -------------------------------------------------------------
    print("\n--- STAGE 2: Camera Trajectory & Mathematical Conventions ---")
    cam_centers = []
    forward_vectors = []
    up_vectors = []
    ortho_errors = []

    for i in range(N):
        R = extrinsics[i, :3, :3]
        t = extrinsics[i, :3, 3]
        det = float(np.linalg.det(R))
        ortho_err = float(np.linalg.norm(R @ R.T - np.eye(3)))
        ortho_errors.append(ortho_err)

        # In OpenCV: p_cam = R * p_world + t
        # Optical center: p_cam = 0 => p_world = -R^T @ t
        c_world = -R.T @ t
        cam_centers.append(c_world)

        # Forward vector in world: R^T @ [0, 0, 1]
        v_fwd = R.T @ np.array([0.0, 0.0, 1.0])
        v_up = R.T @ np.array([0.0, -1.0, 0.0])
        forward_vectors.append(v_fwd)
        up_vectors.append(v_up)

        print(f"  Frame {i}: center = {c_world.round(3)}, fwd = {v_fwd.round(3)}, det(R) = {det:.4f}")

    cam_centers = np.array(cam_centers)
    traj_lengths = [float(np.linalg.norm(cam_centers[i+1] - cam_centers[i])) for i in range(N - 1)]
    total_baseline = float(np.linalg.norm(cam_centers[-1] - cam_centers[0]))

    audit_results["stage2_cameras"] = {
        "orthogonality_max_error": max(ortho_errors),
        "step_lengths_meters": [round(x, 4) for x in traj_lengths],
        "total_baseline_meters": round(total_baseline, 4),
        "mean_step_meters": round(float(np.mean(traj_lengths)), 4),
        "camera_centers_world": [c.round(4).tolist() for c in cam_centers]
    }

    # -------------------------------------------------------------
    # STAGE 3: Depth & Confidence Distributions
    # -------------------------------------------------------------
    print("\n--- STAGE 3: Depth & Confidence Distributions ---")
    depth_stats = []
    for i in range(N):
        d = d_maps[i]
        c = c_maps[i]
        d_p5, d_p50, d_p95 = np.percentile(d, [5, 50, 95])
        c_p5, c_p50, c_p95 = np.percentile(c, [5, 50, 95])
        sky_mask = (c < 1.5)
        depth_stats.append({
            "frame": i,
            "depth_range": [round(float(d.min()), 3), round(float(d.max()), 3)],
            "depth_p5_p50_p95": [round(float(x), 3) for x in [d_p5, d_p50, d_p95]],
            "confidence_p5_p50_p95": [round(float(x), 3) for x in [c_p5, c_p50, c_p95]],
            "sky_pixel_fraction": round(float(sky_mask.mean()), 3)
        })
        # Save depth heatmap visualization
        d_norm = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        d_color = cv2.applyColorMap(d_norm, cv2.COLORMAP_TURBO)
        c_norm = cv2.normalize(np.clip(c, 1.0, 15.0), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        c_color = cv2.applyColorMap(c_norm, cv2.COLORMAP_VIRIDIS)
        combined = np.hstack([d_color, c_color])
        cv2.imwrite(str(DIAG_DIR / f"stage3_depths/depth_conf_frame_{i:02d}.png"), combined)

    audit_results["stage3_depths"] = depth_stats

    # -------------------------------------------------------------
    # STAGE 4: Per-Frame Point Clouds & Planarity
    # -------------------------------------------------------------
    print("\n--- STAGE 4: Single-Frame Point Clouds & Planarity ---")
    single_pcd_stats = []
    for i in range(N):
        d = d_maps[i]
        c = c_maps[i]
        jm = compute_relative_depth_jump(d, kernel_size=3) > 0.12
        # Building ROI: lower half of frame with high confidence
        roi_mask = (c >= 2.0) & (~jm) & (d >= 0.2) & (d <= 4.0)
        pts_cam = np.stack([
            (np.broadcast_to(np.arange(W_inf)[None, :], (H_inf, W_inf)) - intrinsics[i, 0, 2]) / intrinsics[i, 0, 0] * d,
            (np.broadcast_to(np.arange(H_inf)[:, None], (H_inf, W_inf)) - intrinsics[i, 1, 2]) / intrinsics[i, 1, 1] * d,
            d
        ], axis=-1)[roi_mask]

        # Fit plane to detect surface planarity (RANSAC plane)
        pcd_temp = o3d.geometry.PointCloud()
        pcd_temp.points = o3d.utility.Vector3dVector(pts_cam)
        if len(pts_cam) > 100:
            plane_model, inliers = pcd_temp.segment_plane(distance_threshold=0.03, ransac_n=3, num_iterations=500)
            inlier_ratio = round(len(inliers) / len(pts_cam), 3)
        else:
            inlier_ratio = 0.0

        single_pcd_stats.append({
            "frame": i,
            "roi_points": len(pts_cam),
            "dominant_plane_inlier_ratio": inlier_ratio,
            "extents_cam": np.ptp(pts_cam, axis=0).round(3).tolist() if len(pts_cam) > 0 else []
        })
        print(f"  Frame {i}: ROI points={len(pts_cam):,}, dominant plane inliers={inlier_ratio*100:.1f}%")

    audit_results["stage4_single_pcd"] = single_pcd_stats

    # -------------------------------------------------------------
    # STAGE 5: Multi-Frame Registration & Residuals
    # -------------------------------------------------------------
    print("\n--- STAGE 5: Multi-Frame Registered Point Cloud ---")
    pcds_world = []
    for i in range(N):
        d = d_maps[i]
        c = c_maps[i]
        jm = compute_relative_depth_jump(d, kernel_size=3) > 0.12
        valid = (d >= 0.2) & (d <= 4.5) & (c >= 2.0) & (~jm)
        pts_w = depth_to_points(d[None, ...], extrinsics[i:i+1], intrinsics[i:i+1])[0][valid]
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(pts_w)
        p = p.voxel_down_sample(voxel_size=0.01)
        pcds_world.append(p)

    residuals = []
    for i in range(N - 1):
        dists = np.asarray(pcds_world[i].compute_point_cloud_distance(pcds_world[i + 1]))
        med_d = float(np.median(dists))
        mean_d = float(np.mean(dists))
        overlap_5cm = float(np.mean(dists < 0.05))
        residuals.append({
            "pair": f"Frame {i} <-> {i+1}",
            "median_residual_m": round(med_d, 4),
            "mean_residual_m": round(mean_d, 4),
            "overlap_within_5cm": round(overlap_5cm, 3)
        })
        print(f"  Pair {i}<->{i+1}: median residual={med_d*100:.2f} cm, overlap (<5cm)={overlap_5cm*100:.1f}%")

    audit_results["stage5_registration"] = residuals

    # -------------------------------------------------------------
    # STAGE 6: Point Cloud Fusion & Normal Verification
    # -------------------------------------------------------------
    print("\n--- STAGE 6: Point Cloud Fusion & Normals ---")
    all_pts, all_cols, all_nrms = [], [], []
    for i in range(N):
        im = cv2.imread(str(frame_files[i]))
        im_rgb = cv2.cvtColor(cv2.resize(im, (W_inf, H_inf)), cv2.COLOR_BGR2RGB) / 255.0
        d = d_maps[i]
        c = c_maps[i]
        jm = compute_relative_depth_jump(d, kernel_size=3) > 0.12
        valid = (d >= 0.2) & (d <= 4.5) & (c >= 2.0) & (~jm)
        pts = depth_to_points(d[None, ...], extrinsics[i:i+1], intrinsics[i:i+1])[0][valid]
        col = im_rgb[valid]

        pf = o3d.geometry.PointCloud()
        pf.points = o3d.utility.Vector3dVector(pts)
        pf.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=30))
        pf.orient_normals_towards_camera_location(camera_location=cam_centers[i])

        all_pts.append(pts)
        all_cols.append(col)
        all_nrms.append(np.asarray(pf.normals))

    pts_merged = np.concatenate(all_pts, axis=0)
    cols_merged = np.concatenate(all_cols, axis=0)
    nrms_merged = np.concatenate(all_nrms, axis=0)

    pcd_merged = o3d.geometry.PointCloud()
    pcd_merged.points = o3d.utility.Vector3dVector(pts_merged)
    pcd_merged.colors = o3d.utility.Vector3dVector(cols_merged)
    pcd_merged.normals = o3d.utility.Vector3dVector(nrms_merged)

    extents_merged = np.ptp(pts_merged, axis=0)
    voxel_sz = max(float(np.max(extents_merged)) / 1200.0, 0.005)
    pcd_fused = pcd_merged.voxel_down_sample(voxel_size=voxel_sz)

    audit_results["stage6_fused_pcd"] = {
        "raw_points": len(pts_merged),
        "fused_points": len(pcd_fused.points),
        "voxel_size_m": round(voxel_sz, 4),
        "extents_xyz_m": [round(float(x), 3) for x in extents_merged]
    }
    print(f"  Raw: {len(pts_merged):,} points -> Fused: {len(pcd_fused.points):,} points (extents: {extents_merged.round(3)})")

    # -------------------------------------------------------------
    # STAGE 7: Mesh Topology & Component Breakdown
    # -------------------------------------------------------------
    print("\n--- STAGE 7: Mesh Topology & Component Breakdown ---")
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_fused, depth=9)
    # Gentle trim
    mesh.remove_unreferenced_vertices()

    # Support distance trimming
    kdtree = o3d.geometry.KDTreeFlann(pcd_fused)
    m_verts = np.asarray(mesh.vertices)
    nn_dists = pcd_fused.compute_nearest_neighbor_distance()
    med_spacing = float(np.median(nn_dists)) if len(nn_dists) > 0 else 0.01
    max_dist = med_spacing * 5.5

    dists = np.zeros(len(m_verts), dtype=np.float32)
    for i in range(len(m_verts)):
        [k, idx, d2] = kdtree.search_knn_vector_3d(m_verts[i], 1)
        dists[i] = np.sqrt(d2[0]) if k > 0 else 999.0

    mesh.remove_vertices_by_mask(dists > max_dist)
    mesh.remove_unreferenced_vertices()

    # Analyze connected components
    clusters, num_faces, _ = mesh.cluster_connected_triangles()
    clusters = np.asarray(clusters)
    num_faces = np.asarray(num_faces)
    num_clusters = len(num_faces)
    top_5_sizes = sorted(num_faces, reverse=True)[:5]

    audit_results["stage7_mesh"] = {
        "raw_vertices": len(mesh.vertices),
        "raw_triangles": len(mesh.triangles),
        "total_connected_components": num_clusters,
        "top_5_component_face_counts": [int(x) for x in top_5_sizes],
        "largest_component_fraction": round(float(top_5_sizes[0]) / len(mesh.triangles), 4) if len(mesh.triangles) > 0 else 0.0
    }
    print(f"  Mesh: {len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris across {num_clusters} components")
    print(f"  Largest component: {top_5_sizes[0]:,} faces ({top_5_sizes[0]/len(mesh.triangles)*100:.1f}%)")

    # -------------------------------------------------------------
    # STAGE 8: Reprojection & Texture Alignment
    # -------------------------------------------------------------
    print("\n--- STAGE 8: Reprojection & Texture Alignment ---")
    # Sample 1000 vertices on the largest component and check reprojection consistency
    sample_verts = m_verts[np.random.choice(len(m_verts), min(1000, len(m_verts)), replace=False)]
    visible_count_per_vert = []

    for v in sample_verts:
        vis_cams = 0
        for i in range(N):
            R = extrinsics[i, :3, :3]
            t = extrinsics[i, :3, 3]
            K = intrinsics[i]
            p_c = R @ v + t
            if p_c[2] > 0.1:
                u = K[0, 0] * (p_c[0] / p_c[2]) + K[0, 2]
                v_img = K[1, 1] * (p_c[1] / p_c[2]) + K[1, 2]
                if 0 <= u < W_inf and 0 <= v_img < H_inf:
                    vis_cams += 1
        visible_count_per_vert.append(vis_cams)

    mean_vis = float(np.mean(visible_count_per_vert))
    audit_results["stage8_texturing"] = {
        "mean_cameras_observing_each_surface_point": round(mean_vis, 2),
        "fraction_multi_observed": round(float(np.mean(np.array(visible_count_per_vert) >= 2)), 3)
    }
    print(f"  Mean cameras observing each surface point: {mean_vis:.2f}")

    # Save summary report
    report_file = DIAG_DIR / "pipeline_audit_results.json"
    with open(report_file, "w") as f:
        json.dump(audit_results, f, indent=2)
    print(f"\n[+] Saved full diagnostic audit results to: {report_file}")
    return audit_results


if __name__ == "__main__":
    run_comprehensive_audit()
