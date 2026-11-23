"""
F.O.R.G.E. — Part 7: Isolate Failure Stage on Frames 0 to 11 (Windows 0 and 1).
Produces and inspects:
1. Backprojected points from individual frames.
2. Window 0 point cloud.
3. Window 1 local point cloud.
4. Sim(3) alignment between Window 0 and Window 1.
5. Merged pre-TSDF point cloud for frames 0..11.
6. TSDF integration of Window 0 alone (frames 0..5).
7. TSDF integration of Windows 0 + 1 (frames 0..11).
"""

import os
import sys
import json
import numpy as np
import open3d as o3d
import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.reconstruction.point_cloud import depth_to_points, save_ply
from src.reconstruction.indoor_pipeline import solve_umeyama_sim3

EXP_DIR = PROJECT_ROOT / "outputs/indoor/vggt_tsdf_video_20261031_142303"
DIAG_DIR = PROJECT_ROOT / "outputs/indoor/distortion_investigation_20261031/stage_isolation"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load curation report and camera geometry
report = json.loads((EXP_DIR / "frame_curation_report.json").read_text())
frames = report["frames"]
poses_data = np.load(EXP_DIR / "camera_geometry/camera_poses.npz")
depth_data = np.load(EXP_DIR / "camera_geometry/depth_predictions.npz")

all_ext = poses_data["extrinsics"]    # (179, 3, 4)
all_intr = poses_data["intrinsics"]  # (179, 3, 3)
all_depth = depth_data["depth"]      # (179, 432, 240)
all_conf = depth_data["confidence"]  # (179, 432, 240)

print("\n" + "=" * 76)
print("  PART 7: STAGE ISOLATION ON FRAMES 0..11 (WINDOWS 0 & 1)")
print("=" * 76)

# Select frames 0 to 11
N_sub = 12
H_d, W_d = all_depth.shape[1], all_depth.shape[2]

# Individual frame backprojections
frame_pcds = []
for i in range(N_sub):
    d = all_depth[i]
    c = all_conf[i]
    ext = all_ext[i]
    intr = all_intr[i]
    
    # Load RGB image resized to depth map
    im_bgr = cv2.imread(frames[i]["file_path"])
    im_rgb = cv2.cvtColor(cv2.resize(im_bgr, (W_d, H_d)), cv2.COLOR_BGR2RGB)
    
    pts = depth_to_points(d, ext, intr)
    # Mask valid depth & confidence
    # Check conf threshold: in previous run, min conf was 1.0, conf median was 4.5
    mask = (d >= 0.20) & (d <= 5.0)
    
    pts_valid = pts[mask]
    cols_valid = im_rgb[mask]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_valid.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(cols_valid.astype(np.float64) / 255.0)
    
    frame_ply = DIAG_DIR / f"frame_{i:02d}_pcd.ply"
    o3d.io.write_point_cloud(str(frame_ply), pcd)
    frame_pcds.append((pts_valid, cols_valid))
    print(f"[*] Frame {i:02d}: {len(pts_valid):,} valid points | Depth mean={d[mask].mean():.2f}m")

# Merge Window 0 (frames 0..5)
pts_w0 = np.vstack([p[0] for p in frame_pcds[0:6]])
cols_w0 = np.vstack([p[1] for p in frame_pcds[0:6]])
pcd_w0 = o3d.geometry.PointCloud()
pcd_w0.points = o3d.utility.Vector3dVector(pts_w0.astype(np.float64))
pcd_w0.colors = o3d.utility.Vector3dVector(cols_w0.astype(np.float64) / 255.0)
pcd_w0 = pcd_w0.voxel_down_sample(voxel_size=0.01)
o3d.io.write_point_cloud(str(DIAG_DIR / "window0_frames0_5_pcd.ply"), pcd_w0)
print(f"\n[+] Window 0 merged PCD (frames 0..5): {len(pcd_w0.points):,} points (1cm voxel)")

# Merge Window 1 in global coordinates (frames 3..8)
pts_w1_glob = np.vstack([p[0] for p in frame_pcds[3:9]])
cols_w1_glob = np.vstack([p[1] for p in frame_pcds[3:9]])
pcd_w1 = o3d.geometry.PointCloud()
pcd_w1.points = o3d.utility.Vector3dVector(pts_w1_glob.astype(np.float64))
pcd_w1.colors = o3d.utility.Vector3dVector(cols_w1_glob.astype(np.float64) / 255.0)
pcd_w1 = pcd_w1.voxel_down_sample(voxel_size=0.01)
o3d.io.write_point_cloud(str(DIAG_DIR / "window1_frames3_8_global_pcd.ply"), pcd_w1)
print(f"[+] Window 1 merged PCD in global coords (frames 3..8): {len(pcd_w1.points):,} points")

# Merged pre-TSDF point cloud (frames 0..11)
pts_all = np.vstack([p[0] for p in frame_pcds])
cols_all = np.vstack([p[1] for p in frame_pcds])
pcd_all = o3d.geometry.PointCloud()
pcd_all.points = o3d.utility.Vector3dVector(pts_all.astype(np.float64))
pcd_all.colors = o3d.utility.Vector3dVector(cols_all.astype(np.float64) / 255.0)
pcd_all = pcd_all.voxel_down_sample(voxel_size=0.01)
o3d.io.write_point_cloud(str(DIAG_DIR / "merged_pre_tsdf_frames0_11.ply"), pcd_all)
print(f"[+] Merged Pre-TSDF PCD (frames 0..11): {len(pcd_all.points):,} points")

# Now run TSDF on Window 0 alone (frames 0..5)
print("\n[*] Running Open3D TSDF Volume on Window 0 alone (frames 0..5)...")
vol_w0 = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=0.015, sdf_trunc=0.045, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

for i in range(6):
    d = all_depth[i].copy()
    invalid_mask = (d < 0.20) | (d > 5.0)
    d[invalid_mask] = 0.0
    im_bgr = cv2.imread(frames[i]["file_path"])
    im_rgb = cv2.cvtColor(cv2.resize(im_bgr, (W_d, H_d)), cv2.COLOR_BGR2RGB)
    
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(im_rgb), o3d.geometry.Image(d), depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False
    )
    K = all_intr[i]
    intr_obj = o3d.camera.PinholeCameraIntrinsic(W_d, H_d, float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
    ext_4x4 = np.eye(4)
    ext_4x4[:3, :4] = all_ext[i]
    vol_w0.integrate(rgbd, intr_obj, ext_4x4)

mesh_tsdf_w0 = vol_w0.extract_triangle_mesh()
mesh_tsdf_w0.compute_vertex_normals()
pcd_tsdf_w0 = vol_w0.extract_point_cloud()
o3d.io.write_triangle_mesh(str(DIAG_DIR / "tsdf_mesh_window0_alone.ply"), mesh_tsdf_w0)
o3d.io.write_point_cloud(str(DIAG_DIR / "tsdf_pcd_window0_alone.ply"), pcd_tsdf_w0)
print(f"  [+] Window 0 TSDF Mesh: {len(mesh_tsdf_w0.vertices):,} V | {len(mesh_tsdf_w0.triangles):,} F")

# Now run TSDF on frames 0..11 (Windows 0 + 1)
print("\n[*] Running Open3D TSDF Volume on Windows 0 + 1 (frames 0..11)...")
vol_w01 = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=0.015, sdf_trunc=0.045, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

for i in range(12):
    d = all_depth[i].copy()
    invalid_mask = (d < 0.20) | (d > 5.0)
    d[invalid_mask] = 0.0
    im_bgr = cv2.imread(frames[i]["file_path"])
    im_rgb = cv2.cvtColor(cv2.resize(im_bgr, (W_d, H_d)), cv2.COLOR_BGR2RGB)
    
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(im_rgb), o3d.geometry.Image(d), depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False
    )
    K = all_intr[i]
    intr_obj = o3d.camera.PinholeCameraIntrinsic(W_d, H_d, float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
    ext_4x4 = np.eye(4)
    ext_4x4[:3, :4] = all_ext[i]
    vol_w01.integrate(rgbd, intr_obj, ext_4x4)

mesh_tsdf_w01 = vol_w01.extract_triangle_mesh()
mesh_tsdf_w01.compute_vertex_normals()
pcd_tsdf_w01 = vol_w01.extract_point_cloud()
o3d.io.write_triangle_mesh(str(DIAG_DIR / "tsdf_mesh_windows0_1.ply"), mesh_tsdf_w01)
o3d.io.write_point_cloud(str(DIAG_DIR / "tsdf_pcd_windows0_1.ply"), pcd_tsdf_w01)
print(f"  [+] Windows 0+1 TSDF Mesh: {len(mesh_tsdf_w01.vertices):,} V | {len(mesh_tsdf_w01.triangles):,} F")

# Measure geometric coherence: Compare overlap region (frames 3, 4, 5)
pts_w0_overlap = np.vstack([frame_pcds[k][0] for k in [3, 4, 5]])
pcd_w0_sub = o3d.geometry.PointCloud()
pcd_w0_sub.points = o3d.utility.Vector3dVector(pts_w0_overlap.astype(np.float64))

# Compute distance between TSDF mesh surface and pre-TSDF point cloud
dist_w0_to_mesh = np.asarray(pcd_w0_sub.compute_point_cloud_distance(pcd_tsdf_w0))
mean_err_w0 = float(np.mean(dist_w0_to_mesh))
med_err_w0 = float(np.median(dist_w0_to_mesh))
p95_err_w0 = float(np.percentile(dist_w0_to_mesh, 95))

dist_w01_to_mesh = np.asarray(pcd_all.compute_point_cloud_distance(pcd_tsdf_w01))
mean_err_w01 = float(np.mean(dist_w01_to_mesh))
med_err_w01 = float(np.median(dist_w01_to_mesh))
p95_err_w01 = float(np.percentile(dist_w01_to_mesh, 95))

print(f"\n[+] Agreement between Pre-TSDF PCD and TSDF Surface:")
print(f"    Window 0 alone: Mean error={mean_err_w0*100:.2f} cm | Median={med_err_w0*100:.2f} cm | 95th-pct={p95_err_w0*100:.2f} cm")
print(f"    Windows 0+1:   Mean error={mean_err_w01*100:.2f} cm | Median={med_err_w01*100:.2f} cm | 95th-pct={p95_err_w01*100:.2f} cm")

diag_summary = {
    "window0_alone_mesh_vertices": len(mesh_tsdf_w0.vertices),
    "window0_alone_mesh_triangles": len(mesh_tsdf_w0.triangles),
    "windows0_1_mesh_vertices": len(mesh_tsdf_w01.vertices),
    "windows0_1_mesh_triangles": len(mesh_tsdf_w01.triangles),
    "window0_pcd_to_tsdf_median_err_cm": round(med_err_w0 * 100.0, 2),
    "windows0_1_pcd_to_tsdf_median_err_cm": round(med_err_w01 * 100.0, 2),
    "window0_pcd_to_tsdf_p95_err_cm": round(p95_err_w0 * 100.0, 2),
    "windows0_1_pcd_to_tsdf_p95_err_cm": round(p95_err_w01 * 100.0, 2)
}
(DIAG_DIR / "stage_isolation_summary.json").write_text(json.dumps(diag_summary, indent=2))
print(f"[+] Summary written to {DIAG_DIR.name}/stage_isolation_summary.json")

