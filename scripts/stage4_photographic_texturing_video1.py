"""
F.O.R.G.E. Stage 4 — Photographic Texturing for Video 1 via Stage2TextureEngine
Applies:
- 15mm Primary TSDF Mesh
- Multi-view visibility & occlusion checks
- Crease-aware normals (38 deg)
- MRF seam energy minimization
- Native aspect-ratio 4K texture atlas (4096 x 4096)
- Exposure compensation
- Complete deliverables export: GLB, OBJ, MTL, PNG atlas, PLY
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

from src.reconstruction.stage2_texture_engine import Stage2TextureEngine

OUTPUT_DIR = PROJECT_ROOT / "outputs/indoor/vggt_tsdf_video_20261030_170234"
FRAMES_DIR = OUTPUT_DIR / "frames"
CAM_DIR = OUTPUT_DIR / "camera_geometry"
TSDF_DIR = OUTPUT_DIR / "tsdf_benchmark"
TEX_DIR = OUTPUT_DIR / "textured_tsdf_model"
DIAG_DIR = OUTPUT_DIR / "diagnostics"

TEX_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 76)
print("  F.O.R.G.E. — STAGE 4: PHOTOGRAPHIC TEXTURING & DELIVERABLES EXPORT")
print("=" * 76)

t0_tex = time.perf_counter()

# 1. Load Curation & Camera Poses
curation_p = OUTPUT_DIR / "frame_curation_report.json"
curation = json.loads(curation_p.read_text())
frame_records = curation["frames"]
N_frames = len(frame_records)
image_paths = [Path(r["file_path"]) for r in frame_records]
sharpness_list = [r["sharpness"] for r in frame_records]

cam_p = CAM_DIR / "camera_poses_candidate_a.npz"
cam_data = np.load(cam_p)
extrinsics_all = cam_data["extrinsics"]
intrinsics_all = cam_data["intrinsics"]
depth_all = cam_data["depth"]

# 2. Load Winning TSDF Mesh (15mm)
tsdf_mesh_p = TSDF_DIR / "voxel_15mm" / "tsdf_mesh.ply"
if not tsdf_mesh_p.exists():
    raise FileNotFoundError(f"Missing TSDF mesh: {tsdf_mesh_p}")

mesh_tex_in = o3d.io.read_triangle_mesh(str(tsdf_mesh_p))
print(f"[*] Loaded 15mm TSDF mesh: {len(mesh_tex_in.vertices):,} V, {len(mesh_tex_in.triangles):,} F")

target_triangles = 180000
if len(mesh_tex_in.triangles) > target_triangles:
    print(f"[*] Decimating mesh to {target_triangles:,} triangles for optimal WebGL performance...")
    mesh_tex_in = mesh_tex_in.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
    mesh_tex_in.remove_degenerate_triangles()
    mesh_tex_in.remove_duplicated_vertices()
    mesh_tex_in.remove_duplicated_triangles()
    mesh_tex_in.remove_non_manifold_edges()

mesh_tex_in.compute_vertex_normals()
mesh_tex_in.compute_triangle_normals()

# 3. Initialize Stage2TextureEngine
print(f"[*] Initializing Stage2TextureEngine with {N_frames} native frames (478x850)...")
tex_engine = Stage2TextureEngine(
    mesh=mesh_tex_in,
    extrinsics=extrinsics_all,
    intrinsics=intrinsics_all,
    frame_paths=image_paths,
    depth_maps=depth_all,
    floor_rotation=np.eye(3),
    min_cos_theta=0.08
)

# Photographic Quality Matrix
print("[*] Computing photographic quality matrix & line-of-sight visibility...")
scores, vis_mask, occ_mask = tex_engine.compute_photographic_quality_matrix(frame_sharpness=sharpness_list)

# Master Keyframe Selection
print("[*] Selecting optimal master keyframes (16 master cameras)...")
master_indices = tex_engine.select_master_keyframes(scores, target_cameras=16)
print(f"    Selected {len(master_indices)} master keyframes: {master_indices}")

# MRF Seam Optimization
print("[*] Optimizing camera chart assignments with seam energy minimization...")
tri_cam_opt, seam_stats = tex_engine.optimize_camera_charts_mrf(scores, master_indices, seam_penalty=0.45)

# Exposure Compensation
print("[*] Computing exposure compensation gains...")
exposure_gains = tex_engine.compute_exposure_compensation_gains(
    master_cameras=master_indices,
    best_cam_per_face=tri_cam_opt,
    vis_mask=vis_mask
)

# Synthesize Aspect-Preserving Native 4K Texture Atlas
atlas_path = TEX_DIR / "texture_atlas.png"
print("[*] Synthesizing aspect-preserving native 4K texture atlas (4096 x 4096)...")
tile_rects, atlas_stats = tex_engine.synthesize_native_aspect_atlas(
    master_cameras=master_indices,
    exposure_gains=exposure_gains,
    out_atlas_path=atlas_path,
    atlas_dim=4096
)

# Surface Coverage Heatmap
cov_ply_path = TEX_DIR / "surface_coverage_heatmap.ply"
cov_stats = tex_engine.generate_four_class_coverage_mesh(
    output_ply_path=cov_ply_path,
    vis_mask=vis_mask,
    occ_mask=occ_mask,
    best_cam_per_face=tri_cam_opt
)

# Export Canonical Textured Deliverables
base_model_name = "model_video1_textured"
print(f"[*] Exporting canonical textured deliverables ({base_model_name})...")
deliverables = tex_engine.export_stage2_deliverables(
    output_dir=TEX_DIR,
    base_name=base_model_name,
    best_cam_per_face=tri_cam_opt,
    tile_rects=tile_rects,
    atlas_path=atlas_path,
    crease_angle_deg=38.0,
    cull_exterior_hull=False
)

total_tex_sec = time.perf_counter() - t0_tex
print(f"[+] Photographic Texturing Complete in {total_tex_sec:.2f}s!")

# Save metadata
tex_report = {
    "texture_baking_time_sec": round(total_tex_sec, 2),
    "target_mesh_vertices": len(mesh_tex_in.vertices),
    "target_mesh_triangles": len(mesh_tex_in.triangles),
    "master_cameras_selected": len(master_indices),
    "master_camera_indices": master_indices,
    "seam_statistics": seam_stats,
    "atlas_statistics": atlas_stats,
    "coverage_statistics": cov_stats,
    "delivered_files": {k: str(Path(v).name) for k, v in deliverables.items()}
}
(DIAG_DIR / "texturing_report.json").write_text(json.dumps(tex_report, indent=2))
print(f"[+] Saved texturing report to {DIAG_DIR / 'texturing_report.json'}")

