#!/usr/bin/env python3
"""
F.O.R.G.E V10 — Multi-View Stereo Dense Reconstruction & Depth Fusion
Performs calibrated stereo rectification and disparity estimation across all adjacent
and stride-2 image pairs from the PyCOLMAP bundle-adjusted camera poses,
followed by geometric multi-view fusion and normal estimation.
"""

import sys
import os
import time
from pathlib import Path
import cv2
import numpy as np
import pycolmap
import open3d as o3d

PROJECT_ROOT = Path(r"C:\Users\25beevdt025\Desktop\FORGE")
PILOT_DIR = PROJECT_ROOT / "outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign/final_best_canonical_v10/pilot_sfm"
REC_DIR = PILOT_DIR / "sparse/0"
IMAGES_DIR = PILOT_DIR / "images"
DENSE_DIR = PILOT_DIR / "dense_mvs"
DENSE_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 80)
    print("F.O.R.G.E V10 — MULTI-VIEW STEREO DENSE RECONSTRUCTION & FUSION")
    print("=" * 80)
    t0 = time.time()

    rec = pycolmap.Reconstruction(str(REC_DIR))
    images = sorted(list(rec.images.values()), key=lambda x: x.name)
    n_images = len(images)
    print(f"Loaded {n_images} registered cameras from PyCOLMAP SfM.")

    cam = list(rec.cameras.values())[0]
    f_sfm, cx_sfm, cy_sfm, k1 = cam.params
    K = np.array([
        [f_sfm, 0.0, cx_sfm],
        [0.0, f_sfm, cy_sfm],
        [0.0, 0.0, 1.0]
    ])
    dist_coeffs = np.array([k1, 0.0, 0.0, 0.0]) # k1, k2, p1, p2

    print(f"Calibrated Camera Intrinsics:")
    print(f"  focal = {f_sfm:.2f} px, cx = {cx_sfm:.1f}, cy = {cy_sfm:.1f}, k1 = {k1:.6f}")

    # Build stereo pairs list: adjacent (stride 1) and stride 2
    pairs = []
    for i in range(n_images - 1):
        pairs.append((i, i + 1, 1))
    for i in range(n_images - 2):
        pairs.append((i, i + 2, 2))

    print(f"Total stereo baseline pairs to process: {len(pairs)} (23 adjacent + 22 stride-2)")

    # Cache loaded images in memory (24 frames of 1920x1080 BGR is ~140 MB total, easily fits in RAM)
    print("\nCaching image frames in memory...")
    cached_bgr = {}
    for idx, img_info in enumerate(images):
        p = IMAGES_DIR / img_info.name
        cached_bgr[idx] = cv2.imread(str(p))
    h, w, _ = cached_bgr[0].shape
    print(f"Frames cached: {len(cached_bgr)} frames of size {w}x{h}")

    # Process pairs
    all_fused_points = []
    all_fused_colors = []
    pair_stats = []

    print("\nProcessing stereo pairs...")
    for p_idx, (idx1, idx2, stride) in enumerate(pairs):
        t_pair_start = time.time()
        img1 = images[idx1]
        img2 = images[idx2]

        im1_bgr = cached_bgr[idx1]
        im2_bgr = cached_bgr[idx2]

        cfw1 = img1.cam_from_world()
        cfw2 = img2.cam_from_world()
        R1, t1 = cfw1.rotation.matrix(), cfw1.translation
        R2, t2 = cfw2.rotation.matrix(), cfw2.translation

        C1 = -R1.T @ t1
        C2 = -R2.T @ t2
        base_in_1 = R1 @ (C2 - C1)

        # Ensure left/right ordering: left camera has right camera in +X direction
        if base_in_1[0] < 0:
            # img2 is Left, img1 is Right
            imL_bgr, imR_bgr = im2_bgr, im1_bgr
            RL, tL = R2, t2
            RR, tR = R1, t1
            left_name, right_name = img2.name, img1.name
        else:
            # img1 is Left, img2 is Right
            imL_bgr, imR_bgr = im1_bgr, im2_bgr
            RL, tL = R1, t1
            RR, tR = R2, t2
            left_name, right_name = img1.name, img2.name

        R_rel = RR @ RL.T
        t_rel = (tR - RR @ RL.T @ tL).reshape(3, 1)
        baseline = float(np.linalg.norm(t_rel))

        # Stereo rectification
        RL_rect, RR_rect, PL_rect, PR_rect, Q, roi1, roi2 = cv2.stereoRectify(
            K, dist_coeffs, K, dist_coeffs, (w, h), R_rel, t_rel, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )

        mapLx, mapLy = cv2.initUndistortRectifyMap(K, dist_coeffs, RL_rect, PL_rect, (w, h), cv2.CV_32FC1)
        mapRx, mapRy = cv2.initUndistortRectifyMap(K, dist_coeffs, RR_rect, PR_rect, (w, h), cv2.CV_32FC1)

        imL_rect = cv2.remap(imL_bgr, mapLx, mapLy, cv2.INTER_LINEAR)
        imR_rect = cv2.remap(imR_bgr, mapRx, mapRy, cv2.INTER_LINEAR)

        # Adaptive disparity range based on stride and baseline
        num_disp = 16 * 10 if stride == 1 else 16 * 14 # 160 or 224 disparities
        sgbm = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disp,
            blockSize=5,
            P1=8 * 3 * 5**2,
            P2=32 * 3 * 5**2,
            disp12MaxDiff=1,
            uniquenessRatio=12,
            speckleWindowSize=80,
            speckleRange=16,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        disp = sgbm.compute(imL_rect, imR_rect).astype(np.float32) / 16.0
        valid_disp = (disp > 3.0) & (disp < (num_disp - 1))

        points_3d = cv2.reprojectImageTo3D(disp, Q)
        pts_rect = points_3d[valid_disp]
        cols_rect = cv2.cvtColor(imL_rect, cv2.COLOR_BGR2RGB)[valid_disp] / 255.0

        # Physical depth filtering: drone flies at ~20-30m, scene depth is within 3.2m to 22.0m in SfM units
        mask_depth = (pts_rect[:, 2] > 3.2) & (pts_rect[:, 2] < 22.0) & np.isfinite(pts_rect[:, 2])
        pts_pair = pts_rect[mask_depth]
        cols_pair = cols_rect[mask_depth]

        # Transform to PyCOLMAP world coordinate space
        # P_cL = RL_rect^T @ P_rect; P_w = RL^T @ (P_cL - tL)
        P_cL = (RL_rect.T @ pts_pair.T).T
        P_w = (RL.T @ (P_cL - tL).T).T

        # Subsample to keep memory optimal before global fusion (take 1 every 2 points)
        step = 2
        pts_sub = P_w[::step]
        cols_sub = cols_pair[::step]

        all_fused_points.append(pts_sub)
        all_fused_colors.append(cols_sub)

        dt = time.time() - t_pair_start
        pair_stats.append({
            "pair_idx": p_idx,
            "stride": stride,
            "baseline": baseline,
            "valid_points": len(pts_sub) * step,
            "sampled_points": len(pts_sub),
            "dt_sec": dt
        })

        if (p_idx + 1) % 5 == 0 or (p_idx + 1) == len(pairs):
            print(f"  [{p_idx+1:02d}/{len(pairs)}] Stride {stride} | Baseline: {baseline:.3f}m | Sampled: {len(pts_sub):,} pts | Time: {dt:.2f}s")

    # Combine all points
    total_pts = np.vstack(all_fused_points)
    total_cols = np.vstack(all_fused_colors)
    print(f"\nRaw fused multi-view point cloud: {len(total_pts):,} points")

    # Create Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(total_pts)
    pcd.colors = o3d.utility.Vector3dVector(total_cols)

    # 1. Spatial Voxel Grid Downsampling (voxel size = 0.06m = 6cm)
    voxel_size = 0.06
    print(f"\nApplying spatial voxel grid downsampling ({voxel_size*100:.1f} cm)...")
    pcd_vox = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"Points after voxel downsampling: {len(pcd_vox.points):,}")

    # 2. Filter by distance to validated sparse feature cloud (reject points farther than 3.5 units from sparse points)
    print("Filtering points by distance to validated sparse feature cloud...")
    sparse_pcd = o3d.io.read_point_cloud(str(PILOT_DIR / "pilot_sparse_points.ply"))
    kdtree_sparse = o3d.geometry.KDTreeFlann(sparse_pcd)

    vox_pts = np.asarray(pcd_vox.points)
    vox_cols = np.asarray(pcd_vox.colors)
    supported_mask = np.zeros(len(vox_pts), dtype=bool)

    for i, pt in enumerate(vox_pts):
        [_, _, d2] = kdtree_sparse.search_knn_vector_3d(pt, 1)
        if np.sqrt(d2[0]) < 3.2: # within 3.2 units of verified multi-view feature tracks
            supported_mask[i] = True

    pcd_supported = o3d.geometry.PointCloud()
    pcd_supported.points = o3d.utility.Vector3dVector(vox_pts[supported_mask])
    pcd_supported.colors = o3d.utility.Vector3dVector(vox_cols[supported_mask])
    print(f"Points supported by sparse SfM features: {len(pcd_supported.points):,} ({supported_mask.mean()*100:.1f}%)")

    # 3. Statistical Outlier Removal to eliminate isolated noise and floating points
    print("Applying statistical outlier filter (nb_neighbors=35, std_ratio=1.2)...")
    pcd_clean, ind = pcd_supported.remove_statistical_outlier(nb_neighbors=35, std_ratio=1.2)
    print(f"Points after statistical outlier removal: {len(pcd_clean.points):,}")

    # 3. Surface Normal Estimation
    print("Estimating surface normals (search radius = 0.35m)...")
    pcd_clean.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.35, max_nn=35)
    )

    # Orient normals consistently upwards or towards average camera center
    all_cam_centers = []
    for img in images:
        cfw = img.cam_from_world()
        all_cam_centers.append(-cfw.rotation.matrix().T @ cfw.translation)
    mean_cam_center = np.mean(all_cam_centers, axis=0)
    print(f"Average camera center: {mean_cam_center}")
    pcd_clean.orient_normals_towards_camera_location(camera_location=mean_cam_center)

    # Save dense point cloud
    out_dense_ply = PILOT_DIR / "dense_mvs/fused_dense_point_cloud.ply"
    o3d.io.write_point_cloud(str(out_dense_ply), pcd_clean)
    print(f"\nSaved clean dense multi-view point cloud: {out_dense_ply.name} ({out_dense_ply.stat().st_size:,} bytes)")

    # Bounds & Geometry Analysis
    pts_clean_np = np.asarray(pcd_clean.points)
    normals_np = np.asarray(pcd_clean.normals)
    min_b = pts_clean_np.min(axis=0)
    max_b = pts_clean_np.max(axis=0)
    extent = max_b - min_b

    print(f"\nFinal Dense Point Cloud Metrics:")
    print(f"  Total Clean 3D Points: {len(pts_clean_np):,}")
    print(f"  Bounding Box Extent: X={extent[0]:.2f}m, Y={extent[1]:.2f}m, Z={extent[2]:.2f}m")
    print(f"  Coordinate Bounds:")
    print(f"    X: [{min_b[0]:.2f}, {max_b[0]:.2f}] m")
    print(f"    Y: [{min_b[1]:.2f}, {max_b[1]:.2f}] m")
    print(f"    Z: [{min_b[2]:.2f}, {max_b[2]:.2f}] m")

    # Render Orthogonal Projections on pure white background
    print("\nRendering dense point cloud orthogonal views...")
    sub_render_idx = np.random.choice(len(pts_clean_np), size=min(200000, len(pts_clean_np)), replace=False)
    pts_r = pts_clean_np[sub_render_idx]
    cols_r = (np.asarray(pcd_clean.colors)[sub_render_idx] * 255).astype(np.uint8)

    canvas_w, canvas_h = 2400, 900
    panel_w, panel_h = 750, 750
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    def render_panel(pts_2d, cols, title, x_label, y_label):
        panel = np.ones((panel_h, panel_w, 3), dtype=np.uint8) * 255
        p_min = pts_2d.min(axis=0)
        p_max = pts_2d.max(axis=0)
        span = np.maximum(p_max - p_min, 1e-4)
        margin = 40
        scale = min((panel_w - 2 * margin) / span[0], (panel_h - 2 * margin) / span[1])

        cx = panel_w / 2.0
        cy = panel_h / 2.0
        mid = (p_min + p_max) / 2.0

        u = (cx + (pts_2d[:, 0] - mid[0]) * scale).astype(int)
        v = (cy - (pts_2d[:, 1] - mid[1]) * scale).astype(int)

        valid = (u >= 0) & (u < panel_w) & (v >= 0) & (v < panel_h)
        u_v, v_v = u[valid], v[valid]
        c_v = cols[valid]

        for i in range(len(u_v)):
            color = (int(c_v[i, 2]), int(c_v[i, 1]), int(c_v[i, 0]))
            cv2.circle(panel, (u_v[i], v_v[i]), 1, color, -1)

        cv2.rectangle(panel, (0, 0), (panel_w - 1, panel_h - 1), (200, 200, 200), 1)
        cv2.putText(panel, title, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(panel, f"{x_label} vs {y_label}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 100), 1, cv2.LINE_AA)
        return panel

    panel1 = render_panel(pts_r[:, [0, 1]], cols_r, "Dense MVS — Top-Down View", "X (m)", "Y (m)")
    panel2 = render_panel(pts_r[:, [0, 2]], cols_r, "Dense MVS — Front Elevation", "X (m)", "Z (m)")
    panel3 = render_panel(pts_r[:, [1, 2]], cols_r, "Dense MVS — Side Profile", "Y (m)", "Z (m)")

    canvas[100:100+panel_h, 30:30+panel_w] = panel1
    canvas[100:100+panel_h, 825:825+panel_w] = panel2
    canvas[100:100+panel_h, 1620:1620+panel_w] = panel3

    cv2.putText(canvas, "F.O.R.G.E V10 Candidate A — Calibrated Multi-View Stereo Dense Geometry", (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (10, 10, 10), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Reconstructed from 45 calibrated stereo pairs | 24 registered PyCOLMAP cameras | Total Clean Points: {len(pts_clean_np):,}", (30, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 80, 80), 1, cv2.LINE_AA)

    out_render_path = PILOT_DIR / "dense_mvs/dense_mvs_point_cloud_orthogonal_views.png"
    cv2.imwrite(str(out_render_path), canvas)
    print(f"Saved dense MVS render: {out_render_path.name}")

    total_time = time.time() - t0
    print(f"\nDense MVS Reconstruction & Fusion completed successfully in {total_time:.2f}s ({total_time/60:.2f} min).")

if __name__ == "__main__":
    main()
