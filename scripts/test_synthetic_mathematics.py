"""
F.O.R.G.E. — Part 6: Synthetic Mathematical Tests
Tests A through E verifying:
Test A: Pinhole projection and unprojection roundtrip.
Test B: Known camera transformation & TSDF integration (single plane vs doubled surface).
Test C: Sim(3) scale consistency across extrinsics, centers, depths, and 3D points.
Test D: RGB-D dimension and depth-scale validation.
Test E: Synthetic room TSDF integration against known ground truth.
"""

import sys
import json
import math
from pathlib import Path
import numpy as np
import open3d as o3d
import cv2

OUT_DIR = Path("outputs/indoor/distortion_investigation_20261031")
OUT_DIR.mkdir(parents=True, exist_ok=True)

results = {}

# ==============================================================================
# TEST A: Pinhole Projection and Unprojection Roundtrip
# ==============================================================================
print("\n" + "=" * 76)
print("  TEST A: PINHOLE PROJECTION AND UNPROJECTION ROUNDTRIP")
print("=" * 76)

W, H = 240, 432
fx, fy = 250.0, 250.0
cx, cy = 120.0, 216.0
K = np.array([
    [fx, 0.0, cx],
    [0.0, fy, cy],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

# Generate 500 known 3D points in camera frame (Z > 0.5m)
np.random.seed(42)
N_pts = 500
X_cam_gt = np.random.uniform(-0.8, 0.8, (N_pts, 3))
X_cam_gt[:, 2] = np.random.uniform(0.6, 3.5, N_pts)  # Depth in [0.6, 3.5] m

# Project to image
u = fx * (X_cam_gt[:, 0] / X_cam_gt[:, 2]) + cx
v = fy * (X_cam_gt[:, 1] / X_cam_gt[:, 2]) + cy
depths_gt = X_cam_gt[:, 2]

# Unproject back to 3D
X_recon = np.zeros_like(X_cam_gt)
X_recon[:, 0] = (u - cx) * depths_gt / fx
X_recon[:, 1] = (v - cy) * depths_gt / fy
X_recon[:, 2] = depths_gt

err_A = np.max(np.abs(X_cam_gt - X_recon))
print(f"[*] Test A Max Coordinate Error: {err_A:.2e} m")
results["test_A"] = {
    "status": "PASS" if err_A < 1e-7 else "FAIL",
    "max_error_m": float(err_A),
    "points_tested": N_pts
}

# ==============================================================================
# TEST B: Known Camera Transformation & TSDF Integration
# ==============================================================================
print("\n" + "=" * 76)
print("  TEST B: KNOWN CAMERA TRANSFORMATION & TSDF INTEGRATION")
print("=" * 76)

# Synthetic world: Vertical wall plane at Z_world = 2.0 meters, spanning X in [-1.5, 1.5], Y in [-1.5, 1.5]
# Two cameras observing the wall:
# Camera 1 at origin (0, 0, 0) looking at +Z
# Camera 2 at (0.3, 0.1, -0.2) looking at wall with 8-deg rotation around Y

# Intrinsics
K_b = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

# Camera 1 poses:
# camera-from-world:
R1 = np.eye(3)
t1 = np.zeros(3)
ext1_w2c = np.eye(4)
ext1_w2c[:3, :3] = R1
ext1_w2c[:3, 3] = t1

# Camera 2 poses:
# Camera 2 center in world:
C2 = np.array([0.3, 0.1, -0.2])
ang = np.radians(8.0)
R2_c2w = np.array([
    [math.cos(ang), 0, math.sin(ang)],
    [0, 1, 0],
    [-math.sin(ang), 0, math.cos(ang)]
])
R2 = R2_c2w.T
t2 = -R2 @ C2
ext2_w2c = np.eye(4)
ext2_w2c[:3, :3] = R2
ext2_w2c[:3, 3] = t2

# Inverted extrinsics (c2w instead of w2c)
ext1_c2w = np.linalg.inv(ext1_w2c)
ext2_c2w = np.linalg.inv(ext2_w2c)

# Render depth maps for Cam 1 and Cam 2 for the plane Z_world = 2.0
v_grid, u_grid = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
rays_dir1 = np.stack([(u_grid - cx) / fx, (v_grid - cy) / fy, np.ones_like(u_grid)], axis=-1)

# For Cam 1 (at origin looking along +Z): ray is lambda * [rx, ry, 1]. Z_world = 2 -> lambda = 2.0
depth_map1 = np.full((H, W), 2.0, dtype=np.float32)

# For Cam 2 (at C2, ray in world is C2 + lambda * R2_c2w @ [rx, ry, 1]):
# We need Z_world = 2.0:
# (C2 + lambda * R2_c2w @ r)_z = 2.0 -> C2[2] + lambda * (R2_c2w @ r)_z = 2.0
# lambda = (2.0 - C2[2]) / (R2_c2w @ r)_z
# Depth in camera frame is lambda (since r has z=1)
rays_rot = np.einsum("ij,hwj->hwi", R2_c2w, rays_dir1)
ray_z = rays_rot[..., 2]
depth_map2 = np.where(ray_z > 0.1, (2.0 - C2[2]) / ray_z, 0.0).astype(np.float32)

# Create RGBD images
color_dummy = np.full((H, W, 3), 180, dtype=np.uint8)
rgbd1 = o3d.geometry.RGBDImage.create_from_color_and_depth(
    o3d.geometry.Image(color_dummy), o3d.geometry.Image(depth_map1),
    depth_scale=1.0, depth_trunc=4.0, convert_rgb_to_intensity=False
)
rgbd2 = o3d.geometry.RGBDImage.create_from_color_and_depth(
    o3d.geometry.Image(color_dummy), o3d.geometry.Image(depth_map2),
    depth_scale=1.0, depth_trunc=4.0, convert_rgb_to_intensity=False
)

# Configuration B1: Correct world-to-camera [R | t]
vol_b1 = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=0.015, sdf_trunc=0.045, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
vol_b1.integrate(rgbd1, K_b, ext1_w2c)
vol_b1.integrate(rgbd2, K_b, ext2_w2c)
pcd_b1 = vol_b1.extract_point_cloud()
pts_b1 = np.asarray(pcd_b1.points)

# Configuration B2: Inverted camera-to-world
vol_b2 = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=0.015, sdf_trunc=0.045, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
vol_b2.integrate(rgbd1, K_b, ext1_c2w)
vol_b2.integrate(rgbd2, K_b, ext2_c2w)
pcd_b2 = vol_b2.extract_point_cloud()
pts_b2 = np.asarray(pcd_b2.points)

# Check plane thickness for B1
z_b1_std = float(np.std(pts_b1[:, 2])) if len(pts_b1) > 0 else 999.0
z_b1_mean = float(np.mean(pts_b1[:, 2])) if len(pts_b1) > 0 else 999.0
print(f"[*] Configuration B1 (world-to-camera): {len(pts_b1):,} pts | Z Mean={z_b1_mean:.4f}m (GT=2.000m) | Z StdDev={z_b1_std*1000:.2f} mm")

# Check plane thickness for B2
z_b2_std = float(np.std(pts_b2[:, 2])) if len(pts_b2) > 0 else 999.0
z_b2_mean = float(np.mean(pts_b2[:, 2])) if len(pts_b2) > 0 else 999.0
print(f"[*] Configuration B2 (inverted / c2w):  {len(pts_b2):,} pts | Z Mean={z_b2_mean:.4f}m | Z StdDev={z_b2_std*1000:.2f} mm")

results["test_B"] = {
    "config_B1_w2c": {
        "status": "PASS" if abs(z_b1_mean - 2.0) < 0.01 and z_b1_std < 0.005 else "FAIL",
        "pts_count": len(pts_b1),
        "mean_z_m": z_b1_mean,
        "std_z_mm": z_b1_std * 1000.0,
        "plane_doubling_detected": bool(z_b1_std > 0.015)
    },
    "config_B2_inverted": {
        "pts_count": len(pts_b2),
        "mean_z_m": z_b2_mean,
        "std_z_mm": z_b2_std * 1000.0,
        "displaced_or_distorted": bool(abs(z_b2_mean - 2.0) > 0.05 or z_b2_std > 0.02)
    }
}

# ==============================================================================
# TEST C: Sim(3) Scale Consistency
# ==============================================================================
print("\n" + "=" * 76)
print("  TEST C: SIM(3) SCALE CONSISTENCY (CAMERA, DEPTH & 3D POINTS)")
print("=" * 76)

# Known ground-truth Sim(3) transformation between Local Frame (w) and Global Frame (G):
# X_G = s_gt * R_gt @ X_w + t_gt
s_gt = 1.250
ang_c = np.radians(15.0)
R_gt = np.array([
    [math.cos(ang_c), -math.sin(ang_c), 0],
    [math.sin(ang_c), math.cos(ang_c), 0],
    [0, 0, 1]
])
t_gt = np.array([0.45, -0.20, 0.35])

# In local frame w, define 100 3D scene points
X_w_gt = np.random.uniform(-1.0, 1.0, (100, 3))
X_w_gt[:, 2] += 2.0
# Global coordinates:
X_G_gt = s_gt * (X_w_gt @ R_gt.T) + t_gt

# In local frame w, define local camera extrinsics:
R_loc = np.eye(3)
t_loc = np.array([0.1, -0.05, 0.2])
c_loc = -R_loc.T @ t_loc

# Camera projection in local frame:
P_cam_loc = (R_loc @ X_w_gt.T + t_loc[:, None]).T
d_loc = P_cam_loc[:, 2]

# Transformation to global frame:
# Camera center in global frame:
c_glob = s_gt * (R_gt @ c_loc) + t_gt
# Camera rotation in global frame:
R_glob = R_loc @ R_gt.T
# Camera translation in global frame:
t_glob = -R_glob @ c_glob
# Scaled depth in global frame:
d_glob = s_gt * d_loc

# Now test: Does the global camera extrinsic with global depth backproject EXACTLY to X_G_gt?
# Camera coordinates in global frame:
# X_cam_G = R_glob @ X_G + t_glob = R_glob @ (X_G - c_glob)
P_cam_glob = (R_glob @ X_G_gt.T + t_glob[:, None]).T
z_glob_reconstructed = P_cam_glob[:, 2]

depth_scale_error = np.max(np.abs(z_glob_reconstructed - d_glob))
print(f"[*] Sim(3) Global Depth Consistency Error: {depth_scale_error:.2e} m")

# Unproject from global camera coordinates back to world:
# X_world_recon = R_glob.T @ (P_cam_glob.T) - R_glob.T @ t_glob
X_world_recon = (R_glob.T @ (P_cam_glob.T - t_glob[:, None])).T
point_recon_error = np.max(np.abs(X_world_recon - X_G_gt))
print(f"[*] Sim(3) 3D Point Global Reconstruction Error: {point_recon_error:.2e} m")

results["test_C"] = {
    "status": "PASS" if depth_scale_error < 1e-6 and point_recon_error < 1e-6 else "FAIL",
    "depth_scale_error_m": float(depth_scale_error),
    "point_reconstruction_error_m": float(point_recon_error),
    "transformation_rule_verified": "c_glob = s * R @ c_loc + t; R_glob = R_loc @ R.T; t_glob = -R_glob @ c_glob; d_glob = s * d_loc"
}

# ==============================================================================
# TEST D: RGB-D Dimension and Depth-Scale Validation
# ==============================================================================
print("\n" + "=" * 76)
print("  TEST D: RGB-D DIMENSION & DEPTH-SCALE VALIDATION")
print("=" * 76)

# Test Float32 depth (meters) with depth_scale=1.0 vs Uint16 depth (mm) with depth_scale=1000.0
d_float = np.array([[1.50, 2.00], [2.50, 3.00]], dtype=np.float32)
d_uint16 = (d_float * 1000.0).astype(np.uint16)
c_dummy = np.zeros((2, 2, 3), dtype=np.uint8)

rgbd_f = o3d.geometry.RGBDImage.create_from_color_and_depth(
    o3d.geometry.Image(c_dummy), o3d.geometry.Image(d_float),
    depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False
)
rgbd_u = o3d.geometry.RGBDImage.create_from_color_and_depth(
    o3d.geometry.Image(c_dummy), o3d.geometry.Image(d_uint16),
    depth_scale=1000.0, depth_trunc=5.0, convert_rgb_to_intensity=False
)

d_f_arr = np.asarray(rgbd_f.depth)
d_u_arr = np.asarray(rgbd_u.depth)
diff_D = np.max(np.abs(d_f_arr - d_u_arr))
print(f"[*] Float32 (scale=1.0) vs UInt16 (scale=1000.0) Diff: {diff_D:.2e} m")

results["test_D"] = {
    "status": "PASS" if diff_D < 1e-6 else "FAIL",
    "depth_scale_diff_m": float(diff_D),
    "float32_depth_scale": 1.0,
    "uint16_depth_scale": 1000.0
}

# ==============================================================================
# TEST E: Synthetic Room TSDF Integration vs CAD Ground Truth
# ==============================================================================
print("\n" + "=" * 76)
print("  TEST E: SYNTHETIC ROOM TSDF INTEGRATION VS CAD GROUND TRUTH")
print("=" * 76)

# Synthetic room:
# Dimensions: X in [-1.5, 1.5] (width 3m), Y in [0, 2.4] (height 2.4m, floor at Y=0), Z in [-1.5, 1.5] (depth 3m)
# A box table inside: X in [-0.5, 0.5], Y in [0, 0.75], Z in [-0.4, 0.4]

# Define 6 camera viewpoints inside the room looking around:
room_vol = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=0.015, sdf_trunc=0.045, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

# Room planes:
# Floor: Y=0
# Ceiling: Y=2.4
# Wall North: Z=1.5
# Wall South: Z=-1.5
# Wall East: X=1.5
# Wall West: X=-1.5
# Box: X in [-0.5, 0.5], Y in [0, 0.75], Z in [-0.4, 0.4]

cam_positions = [
    np.array([0.0, 1.2, -1.0]),   # Looking toward North (+Z)
    np.array([0.8, 1.2, 0.0]),    # Looking toward West (-X)
    np.array([0.0, 1.2, 1.0]),    # Looking toward South (-Z)
    np.array([-0.8, 1.2, 0.0]),   # Looking toward East (+X)
    np.array([0.6, 1.4, -0.8]),   # Looking toward table center
    np.array([-0.6, 1.4, 0.8]),   # Looking toward table center
]

targets = [
    np.array([0.0, 1.2, 1.5]),
    np.array([-1.5, 1.2, 0.0]),
    np.array([0.0, 1.2, -1.5]),
    np.array([1.5, 1.2, 0.0]),
    np.array([0.0, 0.4, 0.0]),
    np.array([0.0, 0.4, 0.0]),
]

def raycast_synthetic_room(C, R_c2w, W, H, fx, fy, cx, cy):
    # R_c2w maps camera rays to world
    v_g, u_g = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    r_cam = np.stack([(u_g - cx) / fx, (v_g - cy) / fy, np.ones_like(u_g)], axis=-1)
    r_cam /= np.linalg.norm(r_cam, axis=-1, keepdims=True)
    r_world = np.einsum("ij,hwj->hwi", R_c2w, r_cam)
    
    # Intersect with 6 planes
    depth_out = np.full((H, W), np.nan, dtype=np.float32)
    
    planes = [
        # (normal, d_val, min_bounds, max_bounds)
        (np.array([0, 1, 0]), 0.0, np.array([-1.5, -0.01, -1.5]), np.array([1.5, 0.01, 1.5])),    # Floor
        (np.array([0, 1, 0]), 2.4, np.array([-1.5, 2.39, -1.5]), np.array([1.5, 2.41, 1.5])),     # Ceiling
        (np.array([0, 0, 1]), 1.5, np.array([-1.5, 0.0, 1.49]), np.array([1.5, 2.4, 1.51])),      # North
        (np.array([0, 0, 1]), -1.5, np.array([-1.5, 0.0, -1.51]), np.array([1.5, 2.4, -1.49])),   # South
        (np.array([1, 0, 0]), 1.5, np.array([1.49, 0.0, -1.5]), np.array([1.51, 2.4, 1.5])),      # East
        (np.array([1, 0, 0]), -1.5, np.array([-1.51, 0.0, -1.5]), np.array([-1.49, 2.4, 1.5])),   # West
        # Table top: Y=0.75, X in [-0.5, 0.5], Z in [-0.4, 0.4]
        (np.array([0, 1, 0]), 0.75, np.array([-0.5, 0.74, -0.4]), np.array([0.5, 0.76, 0.4])),
        # Table North side: Z=0.4, X in [-0.5, 0.5], Y in [0, 0.75]
        (np.array([0, 0, 1]), 0.4, np.array([-0.5, 0.0, 0.39]), np.array([0.5, 0.75, 0.41])),
        # Table South side: Z=-0.4, X in [-0.5, 0.5], Y in [0, 0.75]
        (np.array([0, 0, 1]), -0.4, np.array([-0.5, 0.0, -0.41]), np.array([0.5, 0.75, -0.39])),
        # Table East side: X=0.5, Y in [0, 0.75], Z in [-0.4, 0.4]
        (np.array([1, 0, 0]), 0.5, np.array([0.49, 0.0, -0.4]), np.array([0.51, 0.75, 0.4])),
        # Table West side: X=-0.5, Y in [0, 0.75], Z in [-0.4, 0.4]
        (np.array([1, 0, 0]), -0.5, np.array([-0.51, 0.0, -0.4]), np.array([-0.49, 0.75, 0.4])),
    ]
    
    t_min = np.full((H, W), 1e9, dtype=np.float32)
    for n, d, bmin, bmax in planes:
        denom = np.sum(r_world * n, axis=-1)
        valid_ray = np.abs(denom) > 1e-4
        t_hit = (d - np.sum(C * n)) / np.where(valid_ray, denom, 1.0)
        # Check if t_hit > 0.05 and within bounds
        pos_hit = C[None, None, :] + t_hit[..., None] * r_world
        in_bounds = (
            (pos_hit[..., 0] >= bmin[0] - 0.02) & (pos_hit[..., 0] <= bmax[0] + 0.02) &
            (pos_hit[..., 1] >= bmin[1] - 0.02) & (pos_hit[..., 1] <= bmax[1] + 0.02) &
            (pos_hit[..., 2] >= bmin[2] - 0.02) & (pos_hit[..., 2] <= bmax[2] + 0.02)
        )
        mask = valid_ray & in_bounds & (t_hit > 0.1) & (t_hit < t_min)
        t_min[mask] = t_hit[mask]
    
    # Camera depth along principal axis (+Z):
    # P_cam = R_c2w.T @ (pos_hit - C)
    # Z_cam = (R_c2w.T @ r_world * t_min)_z = r_cam_z * t_min
    z_depth = t_min * r_cam[..., 2]
    z_depth[t_min >= 1e8] = 0.0
    return z_depth.astype(np.float32)

print("[*] Integrating synthetic room with 6 cameras...")
for ci, (C, T_look) in enumerate(zip(cam_positions, targets)):
    # Build lookat matrix
    fwd = T_look - C
    fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, fwd)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    down /= np.linalg.norm(down)
    
    # OpenCV camera coordinate: X=right, Y=down, Z=fwd
    R_c2w = np.column_stack([right, down, fwd])
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ C
    ext_w2c = np.eye(4)
    ext_w2c[:3, :3] = R_w2c
    ext_w2c[:3, 3] = t_w2c
    
    d_map = raycast_synthetic_room(C, R_c2w, W, H, fx, fy, cx, cy)
    c_img = np.full((H, W, 3), 160 + ci * 15, dtype=np.uint8)
    
    rgbd_ci = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(c_img), o3d.geometry.Image(d_map),
        depth_scale=1.0, depth_trunc=4.5, convert_rgb_to_intensity=False
    )
    room_vol.integrate(rgbd_ci, K_b, ext_w2c)

mesh_e = room_vol.extract_triangle_mesh()
mesh_e.compute_vertex_normals()
pcd_e = room_vol.extract_point_cloud()
pts_e = np.asarray(pcd_e.points)

bbox_e = mesh_e.get_axis_aligned_bounding_box()
ext_e = bbox_e.get_extent().round(4).tolist()
min_e = bbox_e.get_min_bound().round(4).tolist()
max_e = bbox_e.get_max_bound().round(4).tolist()

print(f"[*] Synthetic Room Mesh Extracted: {len(mesh_e.vertices):,} V | {len(mesh_e.triangles):,} F")
print(f"    Bounding Box Extents: {ext_e} (CAD GT: [3.0, 2.4, 3.0] m)")
print(f"    Min Bound: {min_e} (CAD GT: [-1.5, 0.0, -1.5])")
print(f"    Max Bound: {max_e} (CAD GT: [1.5, 2.4, 1.5])")

# Check table top surface detection: Y in [0.73, 0.77]
table_pts = pts_e[(pts_e[:, 0] >= -0.5) & (pts_e[:, 0] <= 0.5) & (pts_e[:, 2] >= -0.4) & (pts_e[:, 2] <= 0.4)]
table_top = table_pts[np.abs(table_pts[:, 1] - 0.75) < 0.03]
table_top_count = len(table_top)
print(f"    Table Top Inlier Points (Y=0.75m +/- 3cm): {table_top_count:,} pts")

results["test_E"] = {
    "status": "PASS" if len(mesh_e.vertices) > 1000 and abs(ext_e[0] - 3.0) < 0.1 and abs(ext_e[1] - 2.4) < 0.1 and table_top_count > 100 else "FAIL",
    "mesh_vertices": len(mesh_e.vertices),
    "mesh_triangles": len(mesh_e.triangles),
    "mesh_extents_m": ext_e,
    "min_bound_m": min_e,
    "max_bound_m": max_e,
    "table_top_pts": table_top_count
}

out_json = OUT_DIR / "synthetic_tests_results.json"
out_json.write_text(json.dumps(results, indent=2))
print(f"\n[+] Saved Synthetic Mathematical Test Results: {out_json.name}")

