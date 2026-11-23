r"""
F.O.R.G.E. — Indoor Reconstruction Orchestration Script
Reusing the Proven Outdoor Pipeline (campus_0049_candidate_a) for Indoor Video

Strictly invokes canonical implementations:
- src.vggt.inference.VGGTInferencePipeline
- src.reconstruction.point_cloud.depth_to_points
- scripts.validate_sprint1_geometry_and_filtering.compute_relative_depth_jump
- src.reconstruction.multiframe_stitcher.WindowedInferenceManager
- src.reconstruction.multiframe_stitcher.SurfaceConnectivityMesher
- src.reconstruction.photographic_texture.extract_native_frames
- src.reconstruction.photographic_texture.scale_camera_intrinsics
- src.reconstruction.photographic_texture.MultiViewTextureEngine
"""

import os
import sys
import json
import time
import shutil
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
import open3d as o3d
import trimesh
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vggt.inference import VGGTInferencePipeline, DEFAULT_CHECKPOINT_PATH
from src.reconstruction.point_cloud import depth_to_points, save_ply
from src.reconstruction.multiframe_stitcher import (
    WindowedInferenceManager,
    SurfaceConnectivityMesher
)
from src.reconstruction.photographic_texture import (
    extract_native_frames,
    scale_camera_intrinsics,
    MultiViewTextureEngine
)
from scripts.validate_sprint1_geometry_and_filtering import compute_relative_depth_jump

# Paths
VIDEO_PATH = Path(r"C:\Users\25beevdt025\Downloads\WhatsApp Video 2026-10-30 at 17.02.34.mp4")
EXP_DIR = PROJECT_ROOT / "outputs/indoor/outdoor_pipeline_applied_video1"
FRAMES_DIR = EXP_DIR / "frames"
PRED_DIR = EXP_DIR / "predictions"
PCD_DIR = EXP_DIR / "point_clouds"
MESH_DIR = EXP_DIR / "meshes"
TEX_DIR = EXP_DIR / "textured"
DIAG_DIR = EXP_DIR / "diagnostics"
PREVIEWS_DIR = EXP_DIR / "previews"

for d in [EXP_DIR, FRAMES_DIR, PRED_DIR, PCD_DIR, MESH_DIR, TEX_DIR, DIAG_DIR, PREVIEWS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


# ==============================================================================
# STAGE 1: VGGT Windowed Inference (Window Size 8, Overlap 4, 6 Windows)
# ==============================================================================
def run_stage1_inference(frame_files: List[Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    print("\n" + "=" * 76)
    print("  STAGE 1: CANONICAL VGGT WINDOWED INFERENCE (WINDOW_SIZE=8, OVERLAP=4)")
    print("=" * 76)
    
    poses_path = PRED_DIR / "camera_poses.npz"
    depth_path = PRED_DIR / "depth_predictions.npz"
    align_rep_path = DIAG_DIR / "window_alignment_report.json"
    
    if poses_path.is_file() and depth_path.is_file() and align_rep_path.is_file():
        print(f"[*] Found cached inference outputs in {PRED_DIR}. Loading...")
        poses_data = np.load(poses_path)
        depth_data = np.load(depth_path)
        align_rep = json.loads(align_rep_path.read_text())
        return (
            poses_data["extrinsics"],
            poses_data["intrinsics"],
            depth_data["depth"],
            depth_data["confidence"],
            align_rep
        )

    print(f"[*] Initializing canonical VGGTInferencePipeline...")
    pipe = VGGTInferencePipeline(image_resolution=512)
    manager = WindowedInferenceManager(pipeline=pipe, window_size=8, overlap=4)
    
    str_paths = [str(p) for p in frame_files]
    t0_inf = time.perf_counter()
    inf_res = manager.run_windowed_inference(str_paths)
    inf_time = time.perf_counter() - t0_inf
    print(f"[+] Canonical windowed inference completed in {inf_time:.2f}s across {len(frame_files)} frames.")
    
    ext = inf_res["extrinsics"].astype(np.float32)
    intr = inf_res["intrinsics"].astype(np.float32)
    depth = inf_res["depth"].astype(np.float32)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)
    conf = inf_res["confidence"].astype(np.float32)
    if conf.ndim == 4 and conf.shape[-1] == 1:
        conf = conf.squeeze(-1)
        
    np.savez_compressed(poses_path, extrinsics=ext, intrinsics=intr)
    np.savez_compressed(depth_path, depth=depth, confidence=conf)
    align_rep_path.write_text(json.dumps(inf_res["alignment_reports"], indent=2))
    
    print(f"[+] Saved poses: {poses_path}")
    print(f"[+] Saved depth: {depth_path}")
    print(f"[+] Saved alignment report: {align_rep_path}")
    
    return ext, intr, depth, conf, inf_res["alignment_reports"]


# ==============================================================================
# STAGE 2: Depth & Confidence Distribution Audit & Filter Sensitivity
# ==============================================================================
def run_stage2_depth_and_filter_audit(
    depth_maps: np.ndarray,
    conf_maps: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray
) -> Dict[str, Any]:
    print("\n" + "=" * 76)
    print("  STAGE 2: DEPTH & CONFIDENCE AUDIT & FILTER SENSITIVITY BENCHMARK")
    print("=" * 76)
    
    valid_d = depth_maps[depth_maps > 0.01]
    d_stats = {
        "min": float(np.min(valid_d)),
        "p01": float(np.percentile(valid_d, 1)),
        "p05": float(np.percentile(valid_d, 5)),
        "median": float(np.median(valid_d)),
        "p95": float(np.percentile(valid_d, 95)),
        "p99": float(np.percentile(valid_d, 99)),
        "max": float(np.max(valid_d))
    }
    print(f"[*] Depth Distribution:")
    print(f"    Min:    {d_stats['min']:.3f} m")
    print(f"    p01:    {d_stats['p01']:.3f} m")
    print(f"    p05:    {d_stats['p05']:.3f} m")
    print(f"    Median: {d_stats['median']:.3f} m")
    print(f"    p95:    {d_stats['p95']:.3f} m")
    print(f"    p99:    {d_stats['p99']:.3f} m")
    print(f"    Max:    {d_stats['max']:.3f} m")
    
    # Confidence distribution
    valid_c = conf_maps.flatten()
    c_stats = {
        "min": float(np.min(valid_c)),
        "p10": float(np.percentile(valid_c, 10)),
        "p20": float(np.percentile(valid_c, 20)),
        "p30": float(np.percentile(valid_c, 30)),
        "p40": float(np.percentile(valid_c, 40)),
        "median": float(np.median(valid_c)),
        "mean": float(np.mean(valid_c)),
        "p75": float(np.percentile(valid_c, 75)),
        "max": float(np.max(valid_c))
    }
    print(f"\n[*] Confidence Distribution:")
    print(f"    Min:    {c_stats['min']:.3f}")
    print(f"    p10:    {c_stats['p10']:.3f}")
    print(f"    p20:    {c_stats['p20']:.3f}")
    print(f"    p30:    {c_stats['p30']:.3f}")
    print(f"    p40:    {c_stats['p40']:.3f}")
    print(f"    Median: {c_stats['median']:.3f}")
    print(f"    Mean:   {c_stats['mean']:.3f}")
    print(f"    Max:    {c_stats['max']:.3f}")
    
    # Evaluate Jump RTOL sensitivity: 0.10, 0.12, 0.15
    jump_sens = {}
    N_frames = len(depth_maps)
    
    # Compute jump masks
    all_jumps = []
    for i in range(N_frames):
        d_i = depth_maps[i]
        j_i = compute_relative_depth_jump(d_i, kernel_size=3)
        all_jumps.append(j_i)
    all_jumps = np.stack(all_jumps, axis=0)
    
    d_min = max(0.10, d_stats["p01"])
    d_max = min(6.50, d_stats["p99"] * 1.05)
    c_thresh = c_stats["p20"] # Using calibrated p20
    
    print(f"\n[*] Calibrated Physical Depth Range: [{d_min:.2f} m, {d_max:.2f} m]")
    print(f"[*] Calibrated Confidence Threshold (p20): {c_thresh:.3f}")
    
    total_raw_points = depth_maps.size
    base_mask = (depth_maps >= d_min) & (depth_maps <= d_max) & (conf_maps >= c_thresh)
    base_count = int(np.sum(base_mask))
    
    for rtol in [0.10, 0.12, 0.15]:
        jump_mask = all_jumps > rtol
        retained_mask = base_mask & (~jump_mask)
        ret_count = int(np.sum(retained_mask))
        removed_by_jump = base_count - ret_count
        jump_sens[f"jump_rtol_{rtol:.2f}"] = {
            "retained_points": ret_count,
            "retained_pct_of_base": round(ret_count / base_count * 100.0, 2),
            "removed_edge_points": removed_by_jump,
            "removed_pct": round(removed_by_jump / base_count * 100.0, 2)
        }
        print(f"    jump_rtol={rtol:.2f}: {ret_count:,} points ({ret_count/base_count*100:.1f}%) retained, {removed_by_jump:,} depth-edge points removed")
        
    audit_report = {
        "depth_stats": d_stats,
        "confidence_stats": c_stats,
        "calibrated_depth_range": [d_min, d_max],
        "calibrated_conf_thresh": c_thresh,
        "jump_sensitivity": jump_sens
    }
    (DIAG_DIR / "depth_and_filter_audit.json").write_text(json.dumps(audit_report, indent=2))
    return audit_report


# ==============================================================================
# STAGE 3: Canonical Point Cloud Construction & Voxel Benchmark
# ==============================================================================
def run_stage3_point_cloud_pipeline(
    frame_files: List[Path],
    depth_maps: np.ndarray,
    conf_maps: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    calibrated_depth_range: Tuple[float, float],
    calibrated_conf_thresh: float,
    jump_rtol: float = 0.12
) -> Dict[str, Any]:
    print("\n" + "=" * 76)
    print("  STAGE 3: POINT CLOUD GENERATION, PER-VIEW NORMAL ESTIMATION & VOXEL BENCHMARK")
    print("=" * 76)
    
    num_frames, H, W_img = depth_maps.shape
    d_min, d_max = calibrated_depth_range
    
    # Load frame RGBs at inference resolution
    frame_rgbs = []
    for f in frame_files:
        im = Image.open(f).convert("RGB")
        if im.size != (W_img, H):
            im = im.resize((W_img, H), Image.Resampling.BILINEAR)
        frame_rgbs.append(np.array(im, dtype=np.uint8) / 255.0)
    frame_rgbs = np.stack(frame_rgbs, axis=0) # (N, H, W, 3)
    
    # Backproject per-frame and compute per-view outward normal orientation
    all_pts = []
    all_cols = []
    all_normals = []
    
    print(f"[*] Backprojecting and orienting normals per-view...")
    for i in range(num_frames):
        d_i = depth_maps[i]
        c_i = conf_maps[i]
        ext_i = extrinsics[i]
        intr_i = intrinsics[i]
        rgb_i = frame_rgbs[i]
        
        # Camera center in world space: C = -R.T @ t
        R_c = ext_i[:3, :3]
        t_c = ext_i[:3, 3]
        cam_center = -R_c.T @ t_c
        
        # Depth-edge jump filter
        jump_i = compute_relative_depth_jump(d_i, kernel_size=3)
        valid_i = (d_i >= d_min) & (d_i <= d_max) & (c_i >= calibrated_conf_thresh) & (jump_i <= jump_rtol)
        
        # Backprojection using canonical depth_to_points
        pts_i = depth_to_points(d_i[None, ...], ext_i[None, ...], intr_i[None, ...])[0]
        
        v_pts = pts_i[valid_i]
        v_cols = rgb_i[valid_i]
        
        # Build Open3D frame cloud to compute normals
        pcd_i = o3d.geometry.PointCloud()
        pcd_i.points = o3d.utility.Vector3dVector(v_pts)
        pcd_i.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=30))
        # Orient normals towards camera center
        pcd_i.orient_normals_towards_camera_location(cam_center)
        
        all_pts.append(v_pts)
        all_cols.append(v_cols)
        all_normals.append(np.asarray(pcd_i.normals))
        
    pts_merged = np.concatenate(all_pts, axis=0)
    cols_merged = np.concatenate(all_cols, axis=0)
    normals_merged = np.concatenate(all_normals, axis=0)
    print(f"[+] Merged raw point cloud: {len(pts_merged):,} points across {num_frames} frames.")
    
    # Save raw merged cloud
    pcd_raw = o3d.geometry.PointCloud()
    pcd_raw.points = o3d.utility.Vector3dVector(pts_merged)
    pcd_raw.colors = o3d.utility.Vector3dVector(cols_merged)
    pcd_raw.normals = o3d.utility.Vector3dVector(normals_merged)
    o3d.io.write_point_cloud(str(PCD_DIR / "indoor_raw_point_cloud.ply"), pcd_raw)
    
    # Spatial Extents
    extents = np.ptp(pts_merged, axis=0)
    max_extent = float(np.max(extents))
    scale_relative_voxel = max(max_extent / 1200.0, 0.005)
    print(f"[*] Scene Extents: [{extents[0]:.2f}, {extents[1]:.2f}, {extents[2]:.2f}] m (max: {max_extent:.2f} m)")
    print(f"[*] Scale-Relative Baseline Voxel: {scale_relative_voxel*1000.0:.2f} mm")
    
    # Voxel Benchmark: 5mm, 10mm, 15mm vs Scale-Relative Baseline
    voxel_benchmark = {}
    voxel_pcds = {}
    for vox_name, vox_size in [
        ("scale_relative_baseline", scale_relative_voxel),
        ("voxel_5mm", 0.005),
        ("voxel_10mm", 0.010),
        ("voxel_15mm", 0.015)
    ]:
        pcd_down = pcd_raw.voxel_down_sample(voxel_size=vox_size)
        voxel_benchmark[vox_name] = {
            "voxel_size_m": vox_size,
            "voxel_size_mm": round(vox_size * 1000.0, 2),
            "point_count": len(pcd_down.points)
        }
        voxel_pcds[vox_name] = pcd_down
        print(f"    {vox_name} ({vox_size*1000.0:.1f} mm): {len(pcd_down.points):,} points")
        
    # Statistical Outlier Removal Comparison:
    # Baseline: nb=20, std=2.0
    # Gentle: nb=20, std=2.5 (preserves thin legs)
    pcd_cand = voxel_pcds["scale_relative_baseline"]
    pcd_sor_std20, ind20 = pcd_cand.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd_sor_std25, ind25 = pcd_cand.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)
    
    print(f"[*] Statistical Outlier Filtering:")
    print(f"    SOR std_ratio=2.0 (outdoor baseline): {len(pcd_sor_std20.points):,} points retained ({len(pcd_cand.points)-len(pcd_sor_std20.points):,} removed)")
    print(f"    SOR std_ratio=2.5 (gentle indoor):    {len(pcd_sor_std25.points):,} points retained ({len(pcd_cand.points)-len(pcd_sor_std25.points):,} removed)")
    
    # We select gentle std_ratio=2.5 to preserve thin furniture structures
    pcd_final_fused = pcd_sor_std25
    fused_ply_path = PCD_DIR / "indoor_fused_point_cloud.ply"
    o3d.io.write_point_cloud(str(fused_ply_path), pcd_final_fused)
    print(f"[+] Saved final fused point cloud: {fused_ply_path} ({len(pcd_final_fused.points):,} points)")
    
    pcd_report = {
        "raw_points": len(pts_merged),
        "scene_extents_m": extents.tolist(),
        "voxel_benchmark": voxel_benchmark,
        "sor_comparison": {
            "std_ratio_2.0": len(pcd_sor_std20.points),
            "std_ratio_2.5_selected": len(pcd_sor_std25.points)
        },
        "selected_pcd_points": len(pcd_final_fused.points)
    }
    (DIAG_DIR / "point_cloud_benchmark.json").write_text(json.dumps(pcd_report, indent=2))
    
    return {
        "pcd_fused": pcd_final_fused,
        "fused_ply_path": fused_ply_path,
        "report": pcd_report
    }


# ==============================================================================
# STAGE 4: Primary Meshing — Screened Poisson HQ Depth 8 vs Depth 9 Benchmark
# ==============================================================================
def run_stage4_poisson_benchmark(pcd_fused: o3d.geometry.PointCloud) -> Dict[str, Any]:
    print("\n" + "=" * 76)
    print("  STAGE 4: SCREENED POISSON HQ BENCHMARK (DEPTH 8 VS DEPTH 9) & REPAIR CANDIDATE")
    print("=" * 76)
    
    candidates = {}
    
    for depth in [8, 9]:
        print(f"[*] Running canonical SurfaceConnectivityMesher.build_poisson_hq(depth={depth}, trim=0.005, support_factor=4.0)...")
        t0_m = time.perf_counter()
        mesh_poisson, p_meta = SurfaceConnectivityMesher.build_poisson_hq(
            pcd_fused,
            depth=depth,
            gentle_trim_quantile=0.005,
            support_distance_factor=4.0
        )
        dt_m = time.perf_counter() - t0_m
        
        # Measure manifold properties
        mesh_poisson.compute_vertex_normals()
        nme = len(mesh_poisson.get_non_manifold_edges(allow_boundary_edges=True))
        nme_strict = len(mesh_poisson.get_non_manifold_edges(allow_boundary_edges=False))
        
        print(f"    Depth {depth}: {len(mesh_poisson.vertices):,} verts, {len(mesh_poisson.triangles):,} tris, {nme} boundary edges ({dt_m:.2f}s)")
        
        raw_mesh_path = MESH_DIR / f"indoor_poisson_depth_{depth}_raw.ply"
        o3d.io.write_triangle_mesh(str(raw_mesh_path), mesh_poisson)
        
        candidates[f"depth_{depth}"] = {
            "depth": depth,
            "mesh": mesh_poisson,
            "raw_path": raw_mesh_path,
            "runtime_sec": round(dt_m, 2),
            "vertices": len(mesh_poisson.vertices),
            "triangles": len(mesh_poisson.triangles),
            "non_manifold_edges": nme_strict,
            "boundary_edges": nme
        }
        
    # Evaluate Repair Candidate on Depth 9 (our primary candidate)
    best_raw_mesh = candidates["depth_9"]["mesh"]
    print(f"[*] Running canonical SurfaceConnectivityMesher.repair_surface_connectivity(max_hole_vertices=16)...")
    mesh_repaired, rep_meta = SurfaceConnectivityMesher.repair_surface_connectivity(
        best_raw_mesh,
        max_hole_vertices=16
    )
    mesh_repaired.compute_vertex_normals()
    rep_nme = len(mesh_repaired.get_non_manifold_edges(allow_boundary_edges=False))
    rep_boundary = len(mesh_repaired.get_non_manifold_edges(allow_boundary_edges=True))
    
    repaired_path = MESH_DIR / "indoor_poisson_depth_9_repaired.ply"
    o3d.io.write_triangle_mesh(str(repaired_path), mesh_repaired)
    print(f"[+] Repaired Depth 9 mesh: {len(mesh_repaired.vertices):,} verts, {len(mesh_repaired.triangles):,} tris, {rep_nme} non-manifold edges")
    
    candidates["depth_9_repaired"] = {
        "depth": 9,
        "mesh": mesh_repaired,
        "raw_path": repaired_path,
        "runtime_sec": 0.0,
        "vertices": len(mesh_repaired.vertices),
        "triangles": len(mesh_repaired.triangles),
        "non_manifold_edges": rep_nme,
        "boundary_edges": rep_boundary
    }
    
    mesh_report = {
        "depth_8": {
            "vertices": candidates["depth_8"]["vertices"],
            "triangles": candidates["depth_8"]["triangles"],
            "runtime_sec": candidates["depth_8"]["runtime_sec"],
            "non_manifold_edges": candidates["depth_8"]["non_manifold_edges"]
        },
        "depth_9": {
            "vertices": candidates["depth_9"]["vertices"],
            "triangles": candidates["depth_9"]["triangles"],
            "runtime_sec": candidates["depth_9"]["runtime_sec"],
            "non_manifold_edges": candidates["depth_9"]["non_manifold_edges"]
        },
        "depth_9_repaired": {
            "vertices": candidates["depth_9_repaired"]["vertices"],
            "triangles": candidates["depth_9_repaired"]["triangles"],
            "non_manifold_edges": candidates["depth_9_repaired"]["non_manifold_edges"]
        }
    }
    (DIAG_DIR / "poisson_meshing_benchmark.json").write_text(json.dumps(mesh_report, indent=2))
    
    return {
        "selected_mesh": mesh_repaired,
        "selected_mesh_path": repaired_path,
        "candidates": candidates,
        "report": mesh_report
    }


# ==============================================================================
# STAGE 5: Multi-View Photographic Texturing Engine & 4K Packed Atlas Export
# ==============================================================================
def run_stage5_photographic_texturing(
    mesh_o3d: o3d.geometry.TriangleMesh,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    depth_maps: np.ndarray,
    timestamps: List[float],
    atlas_size: int = 4096
) -> Dict[str, Any]:
    print("\n" + "=" * 76)
    print("  STAGE 5: MULTI-VIEW PHOTOGRAPHIC TEXTURING ENGINE & 4K PACKED ATLAS")
    print("=" * 76)
    
    # 1. Extract uncompressed native video frames
    native_frames_dir = TEX_DIR / "native_frames"
    print(f"[*] Extracting uncompressed native frames from {VIDEO_PATH.name}...")
    extracted_info = extract_native_frames(
        video_path=str(VIDEO_PATH),
        timestamps=timestamps,
        out_dir=native_frames_dir
    )
    native_frame_paths = [info[0] for info in extracted_info]
    native_w, native_h = extracted_info[0][1], extracted_info[0][2]
    infer_h, infer_w = depth_maps.shape[1], depth_maps.shape[2]
    
    print(f"[*] Native resolution: {native_w}x{native_h} | Inference resolution: {infer_w}x{infer_h}")
    
    # 2. Scale camera intrinsics to native resolution
    scaled_intrinsics = scale_camera_intrinsics(
        intrinsics=intrinsics,
        infer_res=(infer_w, infer_h),
        native_res=(native_w, native_h)
    )
    
    # 3. MultiViewTextureEngine
    print(f"[*] Initializing canonical MultiViewTextureEngine with {len(native_frame_paths)} camera views...")
    engine = MultiViewTextureEngine(
        mesh=mesh_o3d,
        extrinsics=extrinsics,
        intrinsics_native=scaled_intrinsics,
        frame_paths=native_frame_paths,
        depth_maps=depth_maps,
        min_cos_theta=0.05
    )
    
    # 4. Occlusion-Aware Face Visibility and Photographic Scoring
    print(f"[*] Computing face visibility, occlusion, and fronto-parallel photographic scoring...")
    t0_tex = time.perf_counter()
    best_cam_per_face, quality_scores, vis_stats = engine.compute_face_visibility_and_scores()
    
    # 5. Export Packed UV Atlas OBJ + MTL
    obj_path = TEX_DIR / "indoor_reconstruction.obj"
    print(f"[*] Baking {atlas_size}x{atlas_size} photographic UV texture atlas into OBJ/MTL...")
    obj_path, atlas_png_path, atlas_stats = engine.export_packed_atlas_obj(
        output_obj_path=obj_path,
        best_cam_per_face=best_cam_per_face,
        atlas_size=atlas_size
    )
    
    # Export photographic vertex ply as fallback
    ply_path = TEX_DIR / "indoor_reconstruction.ply"
    engine.export_photographic_vertex_ply(ply_path, best_cam_per_face)
    
    # 6. Self-contained GLB Export with embedded 4K texture atlas
    glb_path = TEX_DIR / "indoor_reconstruction.glb"
    print(f"[*] Exporting self-contained binary GLB with embedded texture atlas...")
    tm = trimesh.load(str(obj_path), process=False)
    glb_bytes = tm.export(file_type="glb")
    with open(glb_path, "wb") as f_glb:
        f_glb.write(glb_bytes)
        
    tex_time = time.perf_counter() - t0_tex
    glb_size_mb = glb_path.stat().st_size / (1024 * 1024)
    print(f"[+] GLB exported: {glb_path.name} ({glb_size_mb:.2f} MB, {tex_time:.2f}s)")
    
    # Copy master deliverable to root output
    final_glb = EXP_DIR / "indoor_reconstruction.glb"
    shutil.copy(glb_path, final_glb)
    
    tex_deliverables = {
        "glb_path": str(final_glb.resolve()),
        "glb_size_mb": round(glb_size_mb, 2),
        "glb_sha256": compute_sha256(final_glb),
        "obj_path": str(obj_path.resolve()),
        "mtl_path": str((TEX_DIR / "indoor_reconstruction.mtl").resolve()),
        "atlas_path": str(atlas_png_path.resolve()),
        "atlas_resolution": f"{atlas_size}x{atlas_size}",
        "atlas_size_mb": round(atlas_png_path.stat().st_size / (1024*1024), 2),
        "runtime_sec": round(tex_time, 2)
    }
    (DIAG_DIR / "texturing_deliverables.json").write_text(json.dumps(tex_deliverables, indent=2))
    return tex_deliverables


# ==============================================================================
# MAIN ORCHESTRATION PIPELINE
# ==============================================================================
def main():
    print("=" * 80)
    print("  F.O.R.G.E. — INDOOR DIGITAL TWIN: PROVEN OUTDOOR PIPELINE APPLIED")
    print(f"  Target Video: {VIDEO_PATH.name}")
    print(f"  Output Dir:   {EXP_DIR}")
    print("=" * 80)
    
    t_start_all = time.perf_counter()
    
    # Load curated keyframes from Step 1
    manifest_p = EXP_DIR / "diagnostics/curated_frames_manifest.json"
    if not manifest_p.is_file():
        raise FileNotFoundError(f"Step 1 manifest not found at {manifest_p}. Run Step 1 first!")
    manifest = json.loads(manifest_p.read_text())
    selected_frames = manifest["selected_frames"]
    frame_files = [Path(r["file_path"]) for r in selected_frames]
    timestamps = [r["timestamp_sec"] for r in selected_frames]
    print(f"[*] Loaded {len(frame_files)} curated keyframes spanning {manifest['duration_sec']:.2f}s.")
    
    # Stage 1: Inference
    extrinsics, intrinsics, depth_maps, conf_maps, align_reports = run_stage1_inference(frame_files)
    
    # Stage 2: Depth & Filter Audit
    audit_res = run_stage2_depth_and_filter_audit(depth_maps, conf_maps, extrinsics, intrinsics)
    calibrated_depth_range = tuple(audit_res["calibrated_depth_range"])
    calibrated_conf_thresh = audit_res["calibrated_conf_thresh"]
    
    # Stage 3: Point Cloud Generation & Voxel Sensitivity
    pcd_res = run_stage3_point_cloud_pipeline(
        frame_files=frame_files,
        depth_maps=depth_maps,
        conf_maps=conf_maps,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        calibrated_depth_range=calibrated_depth_range,
        calibrated_conf_thresh=calibrated_conf_thresh,
        jump_rtol=0.12 # Canonical outdoor setting
    )
    pcd_fused = pcd_res["pcd_fused"]
    
    # Stage 4: Meshing Benchmark
    mesh_res = run_stage4_poisson_benchmark(pcd_fused)
    selected_mesh = mesh_res["selected_mesh"]
    
    # Stage 5: Multi-View Photographic Texturing
    tex_res = run_stage5_photographic_texturing(
        mesh_o3d=selected_mesh,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        depth_maps=depth_maps,
        timestamps=timestamps,
        atlas_size=4096
    )
    
    total_time = time.perf_counter() - t_start_all
    print("\n" + "=" * 80)
    print(f"  F.O.R.G.E. COMPLETE DIGITAL TWIN RECONSTRUCTION SUCCEEDED ({total_time:.2f}s)!")
    print(f"  Final GLB: {tex_res['glb_path']} ({tex_res['glb_size_mb']} MB)")
    print("=" * 80)


if __name__ == "__main__":
    main()

