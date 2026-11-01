r"""
F.O.R.G.E. — Phase 3: Point Cloud Fusion, Floor Detection & Coordinate Leveling
Video: WhatsApp Video 2026-11-01 at 14.28.11.mp4
Output: outputs/indoor/digital_twin_reconstruction_video_20261101_142811/
"""

import sys
import os
import json
import time
from pathlib import Path
import numpy as np
import cv2
import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.reconstruction.point_cloud import depth_to_points, save_ply

EXP_DIR = PROJECT_ROOT / "outputs/indoor/digital_twin_reconstruction_video_20261101_142811"
FRAMES_DIR = EXP_DIR / "frames"
GEOM_DIR = EXP_DIR / "camera_geometry"
PCD_DIR = EXP_DIR / "point_clouds"

print("=" * 76, flush=True)
print("  F.O.R.G.E. PHASE 3: FUSION, FLOOR DETECTION & COORDINATE LEVELING", flush=True)
print("=" * 76, flush=True)

# 1. Load chained global camera poses and depths
raw_data = np.load(GEOM_DIR / "global_camera_poses_raw.npz")
extrinsics = raw_data["extrinsics"]  # (40, 3, 4)
intrinsics = raw_data["intrinsics"]  # (40, 3, 3)
depths = raw_data["depths"]          # (40, H, W)
confs = raw_data["confs"]            # (40, H, W)

N_frames, H_d, W_d = depths.shape
print(f"[*] Loaded {N_frames} frames of depth ({W_d}x{H_d}) and poses.", flush=True)

# Load RGB frames
frame_files = sorted(list(FRAMES_DIR.glob("frame_*.png")))
raw_rgbs = []
for f in frame_files:
    bgr = cv2.imread(str(f))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_resized = cv2.resize(rgb, (W_d, H_d), interpolation=cv2.INTER_AREA)
    raw_rgbs.append(rgb_resized)

# 2. Backproject points from all 40 views
all_pts_list = []
all_cols_list = []

print(f"[*] Backprojecting high-confidence 3D points across all {N_frames} views...", flush=True)
for i in range(N_frames):
    d = depths[i]
    c = confs[i]
    rgb = raw_rgbs[i]

    # Confidence threshold 0.55, depth range 0.2m to 5.0m
    mask = (c >= 0.55) & (d >= 0.25) & (d <= 5.0)
    pts = depth_to_points(d, extrinsics[i], intrinsics[i])

    valid_pts = pts[mask]
    valid_cols = rgb[mask]

    all_pts_list.append(valid_pts)
    all_cols_list.append(valid_cols)

all_pts = np.concatenate(all_pts_list, axis=0)
all_cols = np.concatenate(all_cols_list, axis=0)
print(f"[+] Total raw backprojected points: {len(all_pts):,}", flush=True)

# 3. Downsample and filter outliers
pcd_raw = o3d.geometry.PointCloud()
pcd_raw.points = o3d.utility.Vector3dVector(all_pts)
pcd_raw.colors = o3d.utility.Vector3dVector(all_cols / 255.0)

# Voxel size 1.5 cm for room-scale balance
voxel_size = 0.015
pcd_down = pcd_raw.voxel_down_sample(voxel_size=voxel_size)
pcd_down, inliers = pcd_down.remove_statistical_outlier(nb_neighbors=25, std_ratio=1.8)
pts_filtered = np.asarray(pcd_down.points)
cols_filtered = np.asarray(pcd_down.colors)
print(f"[+] Cleaned point cloud: {len(pts_filtered):,} points at {voxel_size*100:.1f} cm voxel spacing.", flush=True)

# 4. RANSAC Floor Plane Fitting
print("\n[*] Detecting ground floor plane via RANSAC...", flush=True)
# Floor is near the bottom of camera trajectory
cam_centers = []
for i in range(N_frames):
    R = extrinsics[i, :3, :3]
    t = extrinsics[i, :3, 3]
    C = - R.T @ t
    cam_centers.append(C)
cam_centers = np.array(cam_centers)
cam_mean = cam_centers.mean(axis=0)

# Points below camera center
low_pts_mask = pts_filtered[:, 1] < (cam_mean[1] - 0.2)  # assuming Y is somewhat vertical or downward
# Let's run plane segment on candidate floor points
plane_model, inliers_floor = pcd_down.segment_plane(distance_threshold=0.03, ransac_n=3, num_iterations=2000)
a, b, c, d_plane = plane_model
floor_normal = np.array([a, b, c], dtype=np.float64)
floor_normal = floor_normal / np.linalg.norm(floor_normal)

# Ensure floor normal points upward (towards camera center)
# If floor_normal . (cam_mean - pt_on_plane) < 0, invert normal
pt_on_plane = - d_plane * floor_normal
if np.dot(floor_normal, cam_mean - pt_on_plane) < 0:
    floor_normal = - floor_normal
    d_plane = - d_plane

print(f"[+] Detected Floor Plane: normal = [{floor_normal[0]:.4f}, {floor_normal[1]:.4f}, {floor_normal[2]:.4f}], d = {d_plane:.4f}", flush=True)

# 5. Compute Coordinate Leveling Rotation R_level
# We want floor_normal -> [0, 1, 0] (Y up)
target_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

v = np.cross(floor_normal, target_up)
s = np.linalg.norm(v)
c_val = np.dot(floor_normal, target_up)

if s < 1e-6:
    R_level = np.eye(3) if c_val > 0 else -np.eye(3)
else:
    vx = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ])
    R_level = np.eye(3) + vx + (vx @ vx) * ((1.0 - c_val) / (s ** 2))

# Rotate points and find floor Y
pts_rotated = (R_level @ pts_filtered.T).T
cam_centers_rotated = (R_level @ cam_centers.T).T

# The floor in rotated space has normal [0, 1, 0], so its Y is constant
floor_inliers_rot = pts_rotated[inliers_floor]
floor_y = float(np.median(floor_inliers_rot[:, 1]))
print(f"[+] Rotated Floor Y elevation: {floor_y:.4f} m (Camera heights: {cam_centers_rotated[:, 1].min():.2f}m to {cam_centers_rotated[:, 1].max():.2f}m)", flush=True)

# Shift Y so Floor is EXACTLY at Y = 0.000 m
t_level = np.array([0.0, -floor_y, 0.0], dtype=np.float64)

# Also find dominant horizontal orientation (walls parallel to X and Z)
# Let's project camera trajectory or wall normals onto the XZ plane
# Frame 0 is looking at the entrance door. In frame 0, camera faces +Z or -Z.
# Let's orient so Entrance Door is at Z_front (min Z or max Z) and Far Window is at Z_back.
# In frame 0, camera faces the door. Let's inspect frame 0 view vector in rotated space:
R0 = extrinsics[0, :3, :3]
cam0_dir = R0.T @ np.array([0, 0, 1.0])  # camera optical axis
cam0_dir_rot = R_level @ cam0_dir
cam0_dir_xz = np.array([cam0_dir_rot[0], 0.0, cam0_dir_rot[2]])
cam0_dir_xz = cam0_dir_xz / (np.linalg.norm(cam0_dir_xz) + 1e-12)

# Rotate around Y so camera 0 faces -Z (toward front door at -Z)
yaw = np.arctan2(cam0_dir_xz[0], -cam0_dir_xz[2])
R_yaw = np.array([
    [np.cos(-yaw), 0.0, np.sin(-yaw)],
    [0.0, 1.0, 0.0],
    [-np.sin(-yaw), 0.0, np.cos(-yaw)]
])

# Full leveling transformation: X_leveled = R_yaw @ (R_level @ X + t_level)
# R_total = R_yaw @ R_level
# t_total = R_yaw @ t_level = [0, -floor_y, 0] (since R_yaw preserves Y)
R_total = R_yaw @ R_level
t_total = t_level.copy()

pts_leveled = (R_total @ pts_filtered.T).T + t_total

# Verify floor is strictly at Y = 0.0
floor_y_new = np.median(pts_leveled[inliers_floor, 1])
print(f"[+] Leveled Floor Y: {floor_y_new:.4f} m (strictly zeroed!)", flush=True)

# Level all camera poses:
# X_cam = R @ X_world + t
# X_leveled = R_total @ X_world + t_total  =>  X_world = R_total.T @ (X_leveled - t_total)
# X_cam = R @ R_total.T @ (X_leveled - t_total) + t
#       = (R @ R_total.T) @ X_leveled + (t - R @ R_total.T @ t_total)
# R_leveled = R @ R_total.T
# t_leveled = t - R_leveled @ t_total
leveled_extrinsics = np.zeros_like(extrinsics)
leveled_cam_centers = []

for i in range(N_frames):
    R = extrinsics[i, :3, :3]
    t = extrinsics[i, :3, 3]
    R_lev = R @ R_total.T
    t_lev = t - R_lev @ t_total
    leveled_extrinsics[i, :3, :3] = R_lev
    leveled_extrinsics[i, :3, 3] = t_lev

    C_lev = - R_lev.T @ t_lev
    leveled_cam_centers.append(C_lev)

leveled_cam_centers = np.array(leveled_cam_centers)
print(f"[+] Camera trajectory leveled: Y ranges from {leveled_cam_centers[:, 1].min():.2f}m to {leveled_cam_centers[:, 1].max():.2f}m (eye height ~1.3-1.6m)", flush=True)

# 6. Room Bounding Box & Architectural Wall Plane Coordinates
# Filter points to room enclosure
valid_box_mask = (pts_leveled[:, 1] >= -0.05) & (pts_leveled[:, 1] <= 3.2)
pts_room = pts_leveled[valid_box_mask]
cols_room = cols_filtered[valid_box_mask]

# Compute percentiles for room boundaries
x_min, x_max = float(np.percentile(pts_room[:, 0], 1.5)), float(np.percentile(pts_room[:, 0], 98.5))
y_min, y_max = 0.00, float(np.percentile(pts_room[:, 1], 99.0))
z_min, z_max = float(np.percentile(pts_room[:, 2], 1.5)), float(np.percentile(pts_room[:, 2], 98.5))

room_width = x_max - x_min
room_height = y_max - y_min
room_depth = z_max - z_min

print("\n" + "=" * 76, flush=True)
print("  RECONSTRUCTED ROOM METRIC BOUNDARIES:", flush=True)
print(f"  Width  (X): [{x_min:+.2f}m .. {x_max:+.2f}m] = {room_width:.2f} m", flush=True)
print(f"  Height (Y): [{y_min:+.2f}m .. {y_max:+.2f}m] = {room_height:.2f} m", flush=True)
print(f"  Length (Z): [{z_min:+.2f}m .. {z_max:+.2f}m] = {room_depth:.2f} m", flush=True)
print("=" * 76, flush=True)

# Save leveled camera geometry and metadata
np.savez_compressed(
    GEOM_DIR / "camera_poses_leveled.npz",
    extrinsics=leveled_extrinsics,
    intrinsics=intrinsics,
    depths=depths,
    confs=confs,
    R_total=R_total,
    t_total=t_total,
    room_bounds=np.array([x_min, x_max, y_min, y_max, z_min, z_max]),
    cam_centers=leveled_cam_centers
)

room_meta = {
    "num_frames": N_frames,
    "room_width_m": round(room_width, 3),
    "room_height_m": round(room_height, 3),
    "room_length_m": round(room_depth, 3),
    "bounds": {
        "x_min": round(x_min, 3),
        "x_max": round(x_max, 3),
        "y_min": round(y_min, 3),
        "y_max": round(y_max, 3),
        "z_min": round(z_min, 3),
        "z_max": round(z_max, 3)
    },
    "floor_y_m": 0.0,
    "camera_height_mean_m": round(float(leveled_cam_centers[:, 1].mean()), 3),
    "voxel_size_m": voxel_size,
    "num_points": len(pts_room)
}

with open(EXP_DIR / "diagnostics/room_spatial_metadata.json", "w") as f:
    json.dump(room_meta, f, indent=2)

# Save fused leveled point cloud PLY
ply_path = PCD_DIR / "fused_room_pointcloud.ply"
save_ply(pts_room, (cols_room * 255.0).astype(np.uint8), str(ply_path))
print(f"[+] Saved fused leveled point cloud: {ply_path.name} ({len(pts_room):,} points)", flush=True)
print("=" * 76, flush=True)

