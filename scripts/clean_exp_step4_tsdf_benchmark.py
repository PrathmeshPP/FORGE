"""
F.O.R.G.E. Clean-Slate Experiment — Step 4: Multi-Resolution Scalable TSDF Benchmark
Source: WhatsApp Video 2026-10-30 at 17.02.34.mp4
Output: outputs/indoor/clean_reconstruction_video1_20261030
"""

import sys
import os
import json
import time
from pathlib import Path
import numpy as np
import open3d as o3d
import cv2
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EXP_DIR = Path("outputs/indoor/clean_reconstruction_video1_20261030")
FRAMES_DIR = EXP_DIR / "frames"
CAM_DIR = EXP_DIR / "camera_geometry"
PCD_DIR = EXP_DIR / "point_clouds"
TSDF_DIR = EXP_DIR / "tsdf_benchmark"
DIAG_DIR = EXP_DIR / "diagnostics"

TSDF_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 76)
print("  F.O.R.G.E. — CLEAN-SLATE STEP 4: MULTI-RESOLUTION SCALABLE TSDF BENCHMARK")
print("=" * 76)

# Load curation report & camera poses
comp_p = DIAG_DIR / "frame_selection_comparison.json"
comp = json.loads(comp_p.read_text())
frame_records = comp["frames"]
N_frames = len(frame_records)
image_paths = [Path(r["file_path"]) for r in frame_records]

cam_p = CAM_DIR / "camera_poses_candidate_a.npz"
if not cam_p.exists():
    raise FileNotFoundError(f"Missing camera poses: {cam_p}")

cam_data = np.load(cam_p)
extrinsics_all = cam_data["extrinsics"]  # (N, 3, 4)
intrinsics_all = cam_data["intrinsics"]  # (N, 3, 3)
depth_all = cam_data["depth"]            # (N, H, W)
conf_all = cam_data["confidence"]        # (N, H, W)
centers_all = cam_data["camera_centers"] # (N, 3)

H_d, W_d = depth_all.shape[1], depth_all.shape[2]

# Load calibrated confidence threshold
conf_diag_p = DIAG_DIR / "confidence_distribution.json"
conf_diag = json.loads(conf_diag_p.read_text())
CONF_THRESH = conf_diag.get("recommended_p20_threshold", 2.0)
print(f"[*] Calibrated Confidence Threshold: {CONF_THRESH:.4f}")

# Load clean pre-TSDF point cloud
clean_pcd_p = PCD_DIR / "fused_point_cloud_clean.ply"
clean_pcd = o3d.io.read_point_cloud(str(clean_pcd_p))
print(f"[*] Loaded clean pre-TSDF point cloud: {len(clean_pcd.points):,} points")

tsdf_configs = [
    {"id": "voxel_10mm", "voxel_len": 0.010, "sdf_trunc": 0.030, "label": "10mm High-Resolution"},
    {"id": "voxel_15mm", "voxel_len": 0.015, "sdf_trunc": 0.045, "label": "15mm Balanced (Primary)"},
    {"id": "voxel_20mm", "voxel_len": 0.020, "sdf_trunc": 0.060, "label": "20mm Coarse Fast"}
]

benchmark_results = {}

for cfg in tsdf_configs:
    v_id = cfg["id"]
    voxel_len = cfg["voxel_len"]
    sdf_trunc = cfg["sdf_trunc"]
    v_label = cfg["label"]

    cfg_dir = TSDF_DIR / v_id
    cfg_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[*] Integrating TSDF Volume: {v_label} (voxel={voxel_len*1000:.1f}mm, trunc={sdf_trunc*1000:.1f}mm)...")
    t0_integ = time.perf_counter()

    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_len,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    for i in range(N_frames):
        d = depth_all[i].copy()
        mask_inv = (d < 0.20) | (d > 5.0) | (conf_all[i] < CONF_THRESH)
        d[mask_inv] = 0.0

        im_bgr = cv2.imread(str(image_paths[i]))
        im_rgb = cv2.cvtColor(cv2.resize(im_bgr, (W_d, H_d)), cv2.COLOR_BGR2RGB)

        o3d_color = o3d.geometry.Image(im_rgb)
        o3d_depth = o3d.geometry.Image(d.astype(np.float32))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color,
            o3d_depth,
            depth_scale=1.0,
            depth_trunc=5.0,
            convert_rgb_to_intensity=False
        )

        intr = intrinsics_all[i]
        cam_intr = o3d.camera.PinholeCameraIntrinsic(
            W_d, H_d,
            float(intr[0, 0]), float(intr[1, 1]),
            float(intr[0, 2]), float(intr[1, 2])
        )

        ext_3x4 = extrinsics_all[i]
        ext_4x4 = np.eye(4, dtype=np.float64)
        ext_4x4[:3, :] = ext_3x4

        vol.integrate(rgbd, cam_intr, ext_4x4)

    integ_time = time.perf_counter() - t0_integ
    print(f"  Integration completed in {integ_time:.2f}s")

    # Extract point cloud
    pcd_tsdf = vol.extract_point_cloud()
    pcd_path = cfg_dir / "tsdf_point_cloud.ply"
    o3d.io.write_point_cloud(str(pcd_path), pcd_tsdf)

    # Extract triangle mesh
    mesh_tsdf = vol.extract_triangle_mesh()
    mesh_tsdf.remove_degenerate_triangles()
    mesh_tsdf.remove_duplicated_vertices()
    mesh_tsdf.remove_duplicated_triangles()
    mesh_tsdf.remove_non_manifold_edges()
    mesh_tsdf.compute_vertex_normals()

    mesh_ply_path = cfg_dir / "tsdf_mesh.ply"
    o3d.io.write_triangle_mesh(str(mesh_ply_path), mesh_tsdf)

    # Convert to GLB
    glb_path = cfg_dir / f"tsdf_mesh_{v_id}.glb"
    tm_mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh_tsdf.vertices),
        faces=np.asarray(mesh_tsdf.triangles),
        vertex_colors=np.asarray(mesh_tsdf.vertex_colors) if mesh_tsdf.has_vertex_colors() else None,
        vertex_normals=np.asarray(mesh_tsdf.vertex_normals) if mesh_tsdf.has_vertex_normals() else None,
        process=False
    )
    tm_mesh.export(str(glb_path))
    tm_mesh.export(str(TSDF_DIR / f"tsdf_mesh_{v_id}.glb"))

    # Surface distance evaluation
    dists = clean_pcd.compute_point_cloud_distance(pcd_tsdf)
    dists_cm = np.asarray(dists) * 100.0
    med_dist = float(np.median(dists_cm))
    p95_dist = float(np.percentile(dists_cm, 95))
    mean_dist = float(np.mean(dists_cm))

    non_manifold = not mesh_tsdf.is_edge_manifold()
    num_verts = len(mesh_tsdf.vertices)
    num_faces = len(mesh_tsdf.triangles)
    num_pcd_pts = len(pcd_tsdf.points)

    res_entry = {
        "voxel_size_mm": voxel_len * 1000.0,
        "sdf_trunc_mm": sdf_trunc * 1000.0,
        "label": v_label,
        "integration_time_sec": round(integ_time, 2),
        "point_count": num_pcd_pts,
        "vertex_count": num_verts,
        "face_count": num_faces,
        "surface_dist_median_cm": round(med_dist, 2),
        "surface_dist_mean_cm": round(mean_dist, 2),
        "surface_dist_p95_cm": round(p95_dist, 2),
        "is_manifold": not non_manifold,
        "glb_size_mb": round(glb_path.stat().st_size / (1024**2), 2)
    }
    benchmark_results[v_id] = res_entry
    print(f"  [+] Extracted: {num_pcd_pts:,} PCD pts | {num_verts:,} V / {num_faces:,} F mesh")
    print(f"  [+] Surface Distance to Pre-TSDF: median={med_dist:.2f}cm, 95th-pct={p95_dist:.2f}cm | Manifold: {not non_manifold} | GLB: {res_entry['glb_size_mb']} MB")

summary_path = TSDF_DIR / "tsdf_benchmark_summary.json"
summary_path.write_text(json.dumps(benchmark_results, indent=2))
print(f"\n[+] TSDF Multi-Resolution Benchmark Summary saved to {summary_path.name}")

