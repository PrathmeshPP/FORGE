r"""
F.O.R.G.E. — High-Density Indoor Room 3D Reconstruction: Attempt #4 (New Video Source).
Input Source: C:\Users\25beevdt025\Downloads\WhatsApp Video 2026-10-30 at 16.43.49.mp4
Output: outputs/indoor/room_001/attempt_4_new_video
"""

from __future__ import annotations

import os
import sys
import json
import time
import shutil
from pathlib import Path
from dataclasses import asdict
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
import open3d as o3d
import torch
import psutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reconstruction.point_cloud import save_ply
from src.reconstruction.indoor_pipeline import (
    IndoorPipelineConfig,
    IndoorReconstructionEngine,
    compute_floor_leveling_rotation
)

VIDEO_PATH = r"C:\Users\25beevdt025\Downloads\WhatsApp Video 2026-10-30 at 16.43.49.mp4"
OUTPUT_DIR = PROJECT_ROOT / "outputs/indoor/room_001/attempt_4_new_video"


def get_system_memory_mb() -> float:
    return psutil.virtual_memory().used / (1024 ** 2)


def get_gpu_vram_mb() -> Tuple[float, float]:
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(0) / (1024 ** 2)
        res = torch.cuda.memory_reserved(0) / (1024 ** 2)
        return round(alloc, 2), round(res, 2)
    return 0.0, 0.0


def run_reconstruction():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frames_dir = OUTPUT_DIR / "frames"
    camera_dir = OUTPUT_DIR / "camera_geometry"
    mesh_dir = OUTPUT_DIR / "meshes"
    renders_dir = OUTPUT_DIR / "renders"

    for d in [frames_dir, camera_dir, mesh_dir, renders_dir]:
        d.mkdir(parents=True, exist_ok=True)

    log_path = OUTPUT_DIR / "reconstruction.log"
    log_file = open(log_path, "w", encoding="utf-8")

    def log(msg: str):
        t_str = time.strftime("%H:%M:%S")
        line = f"[{t_str}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    log("=" * 72)
    log("  F.O.R.G.E. Indoor 3D Reconstruction — Attempt #4 (New Video Source)")
    log(f"  Target Video: {VIDEO_PATH}")
    log(f"  Output Directory: {OUTPUT_DIR}")
    log(f"  CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    log("=" * 72)

    t_total_start = time.perf_counter()
    initial_ram = get_system_memory_mb()

    # ------------------------------------------------------------------
    # Stage A: Load Pre-Curated High-Density Frames
    # ------------------------------------------------------------------
    log("\n[Stage A] Ingesting Curated Frame Manifest (Sampling Density >= 1 FPS)...")
    t_stage_a = time.perf_counter()

    sampling_report_path = OUTPUT_DIR / "frame_sampling_report.json"
    if not sampling_report_path.is_file():
        raise FileNotFoundError(f"Sampling report not found at: {sampling_report_path}")

    with open(sampling_report_path, "r") as f:
        sampling_report = json.load(f)

    curated_frames_meta = sampling_report["selected_frames"]
    image_paths = [f["file_path"] for f in curated_frames_meta]
    num_frames = len(image_paths)

    for p in image_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Curated frame missing: {p}")

    stage_a_sec = time.perf_counter() - t_stage_a
    duration_sec = sampling_report["source_video"]["duration_sec"]
    log(f"  Ingested {num_frames} curated frames across {duration_sec:.2f}s video (Density: {num_frames/duration_sec:.2f} FPS)")
    for f in curated_frames_meta[:6]:
        log(f"    Frame #{f['frame_index']:04d} @ {f['timestamp_sec']:5.2f}s | Sh: {f['laplacian_sharpness']:5.1f} | Disp: {f['pairwise_disparity']:4.1f} | {'ADAPT' if f['is_adaptive_keyframe'] else '1FPS'}")
    log(f"    ... [{num_frames - 6} additional chronologically ordered frames]")

    # Configuration for Attempt 4
    config = IndoorPipelineConfig(
        image_resolution=320,  # 320x178 tokens: 4.29 FPS on RTX 4050/2070 GPU
        window_size=6,
        overlap=3,
        min_depth=0.20,
        max_depth=5.5,
        min_confidence=0.75,
        voxel_size=0.012,  # 1.2 cm high-density indoor voxel grid
        poisson_depth=9,
        density_trim_quantile=0.015,
        support_dist_multiplier=8.0,
        level_floor=True,
        min_cluster_faces=40,
        tsdf_voxel_length=0.015,
        tsdf_trunc=0.04
    )
    engine = IndoorReconstructionEngine(config)

    # ------------------------------------------------------------------
    # Stage B: Neural Camera & Geometry Estimation (Overlapping Windows)
    # ------------------------------------------------------------------
    log("\n[Stage B] Neural Camera Pose & Geometry Estimation (VGGT-Omega Overlapping Windows)...")
    poses_file = camera_dir / "camera_poses.npz"
    depth_file = camera_dir / "depth_predictions.npz"

    t_stage_b = time.perf_counter()
    window_results = engine.run_overlapping_windows(
        image_paths=image_paths,
        window_size=config.window_size,
        overlap=config.overlap,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        log_callback=log
    )
    stage_b_sec = time.perf_counter() - t_stage_b

    extrinsics = window_results["extrinsics"]
    intrinsics = window_results["intrinsics"]
    depth_maps = window_results["depth"]
    confidence_maps = window_results["confidence"]
    vggt_perf = window_results["performance"]

    vram_alloc, vram_res = get_gpu_vram_mb()
    log(f"  Stage B wall-clock duration: {stage_b_sec:.2f}s")
    log(f"  Actual VGGT inference time: {vggt_perf['total_vggt_inference_sec']:.2f}s")
    log(f"  Unique Frames Processed: {num_frames}")
    unique_throughput_fps = num_frames / max(0.001, vggt_perf['total_vggt_inference_sec'])
    log(f"  **VGGT Measured Throughput: {unique_throughput_fps:.2f} unique frames / wall-clock inference second** (Target: >= 1.0 FPS)")
    log(f"  Peak VRAM Allocated: {vram_alloc} MB | Reserved: {vram_res} MB")
    log(f"  Mean Sim(3) Alignment Residual: {vggt_perf['mean_sim3_alignment_error_m']*100:.2f} cm")

    # Save camera poses and depth predictions
    np.savez_compressed(poses_file, extrinsics=extrinsics, intrinsics=intrinsics)
    np.savez_compressed(depth_file, depth=depth_maps, confidence=confidence_maps)
    log(f"  Saved camera geometry to: {camera_dir.name}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    traj_diag = engine.assess_indoor_trajectory(extrinsics)
    log(f"  Trajectory: {traj_diag['num_cameras']} cameras, path length: {traj_diag['total_path_length_m']}m, net disp: {traj_diag['net_displacement_m']}m")

    # ------------------------------------------------------------------
    # Stage C: Point Cloud Processing & Plane Detection
    # ------------------------------------------------------------------
    log("\n[Stage C] Indoor Point-Cloud Processing & Plane Detection...")
    t_stage_c = time.perf_counter()
    frames_bgr = [cv2.imread(p) for p in image_paths]

    clean_pts, clean_cols, raw_pcd_tuple, pcd_stats = engine.backproject_indoor_pcd(
        frames_bgr=frames_bgr,
        depth_maps=depth_maps,
        confidence_maps=confidence_maps,
        extrinsics=extrinsics,
        intrinsics=intrinsics
    )
    raw_pts, raw_cols = raw_pcd_tuple
    stage_c_sec = time.perf_counter() - t_stage_c
    log(f"  Raw points: {pcd_stats['total_raw_points']:,} -> Valid: {pcd_stats['valid_masked_points']:,} -> Fused: {pcd_stats['fused_points']:,} -> Clean: {pcd_stats['clean_points']:,} (Retention: {pcd_stats['retention_rate']}%)")
    log(f"  Point cloud extents [X, Y, Z]: {pcd_stats['extents']} m")

    raw_pcd_path = OUTPUT_DIR / "raw_point_cloud.ply"
    clean_pcd_path = OUTPUT_DIR / "filtered_point_cloud.ply"
    compat_pcd_path = OUTPUT_DIR / "indoor_point_cloud.ply"

    save_ply(raw_pts, raw_cols, str(raw_pcd_path))
    save_ply(clean_pts, clean_cols, str(clean_pcd_path))
    save_ply(clean_pts, clean_cols, str(compat_pcd_path))
    log(f"  Saved raw point cloud: {raw_pcd_path.name} ({raw_pcd_path.stat().st_size / (1024**2):.2f} MB)")
    log(f"  Saved filtered point cloud: {clean_pcd_path.name} ({clean_pcd_path.stat().st_size / (1024**2):.2f} MB)")

    # Detect dominant room planes
    plane_results = engine.detect_room_planes(clean_pts)
    with open(OUTPUT_DIR / "plane_detection.json", "w") as f:
        json.dump(plane_results, f, indent=2)
    for p in plane_results["detected_planes"]:
        log(f"    Plane #{p['plane_index']} [{p['type']}]: {p['inlier_count']:,} inliers ({p['inlier_percentage']}%) | Normal: {p['normal']}")

    # Apply Floor Leveling
    cam_centers = [np.array(c) for c in traj_diag["camera_centers"]]
    if config.level_floor:
        horiz_planes = [p for p in plane_results["detected_planes"] if "horizontal" in p["type"]]
        if horiz_planes:
            best_floor = horiz_planes[0]
            R_floor = compute_floor_leveling_rotation(np.array(best_floor["normal"]))
            clean_pts = (R_floor @ clean_pts.T).T.astype(np.float32)
            raw_pts = (R_floor @ raw_pts.T).T.astype(np.float32)
            cam_centers = [(R_floor @ c) for c in cam_centers]
            log(f"  [+] Floor leveling applied: Aligned plane normal {best_floor['normal']} to [0, 1, 0]")

    # ------------------------------------------------------------------
    # Stage D: Multi-Method Mesh Generation (Poisson, TSDF, BPA)
    # ------------------------------------------------------------------
    log("\n[Stage D] Multi-Method Mesh Generation & Evaluation...")
    t_stage_d = time.perf_counter()
    meshing_evaluations = {}

    # Method 1: Screened Poisson
    log(f"  -> Executing Screened Poisson Surface Reconstruction (depth={config.poisson_depth})...")
    t_m1 = time.perf_counter()
    raw_untrimmed_mesh, mesh_poisson = engine.reconstruct_poisson_mesh(
        clean_pts, clean_cols, cam_centers, depth=config.poisson_depth
    )
    dur_m1 = time.perf_counter() - t_m1
    v_cnt1 = len(mesh_poisson.vertices)
    t_cnt1 = len(mesh_poisson.triangles)
    log(f"     Poisson mesh generated in {dur_m1:.2f}s: {v_cnt1:,} vertices, {t_cnt1:,} triangles")

    untrimmed_exp = engine.export_all_formats(raw_untrimmed_mesh, mesh_dir, base_name="raw_untrimmed_mesh")
    postproc_exp = engine.export_all_formats(mesh_poisson, mesh_dir, base_name="postprocessed_mesh")
    poisson_exp = engine.export_all_formats(mesh_poisson, mesh_dir, base_name="poisson_mesh")
    meshing_evaluations["poisson"] = {
        "vertices": v_cnt1,
        "triangles": t_cnt1,
        "generation_time_sec": round(dur_m1, 2),
        "files": poisson_exp,
        "untrimmed_files": untrimmed_exp,
        "postprocessed_files": postproc_exp
    }

    # Method 2: Open3D TSDF Volume Integration
    log(f"  -> Executing Open3D TSDF Volume Integration (voxel={config.tsdf_voxel_length}m)...")
    t_m2 = time.perf_counter()
    try:
        mesh_tsdf = engine.reconstruct_tsdf_mesh(frames_bgr, depth_maps, confidence_maps, extrinsics, intrinsics)
        dur_m2 = time.perf_counter() - t_m2
        v_cnt2 = len(mesh_tsdf.vertices)
        t_cnt2 = len(mesh_tsdf.triangles)
        log(f"     TSDF mesh generated in {dur_m2:.2f}s: {v_cnt2:,} vertices, {t_cnt2:,} triangles")
        tsdf_exp = engine.export_all_formats(mesh_tsdf, mesh_dir, base_name="tsdf_mesh")
        meshing_evaluations["tsdf"] = {
            "vertices": v_cnt2,
            "triangles": t_cnt2,
            "generation_time_sec": round(dur_m2, 2),
            "files": tsdf_exp
        }
    except Exception as e:
        log(f"     TSDF Meshing notice: {e}")
        mesh_tsdf = None
        meshing_evaluations["tsdf"] = {"error": str(e)}

    # Method 3: Multi-Scale Ball-Pivoting Algorithm (BPA)
    log(f"  -> Executing Multi-Scale Ball-Pivoting Algorithm (BPA)...")
    t_m3 = time.perf_counter()
    try:
        mesh_bpa = engine.reconstruct_bpa_mesh(clean_pts, clean_cols, cam_centers)
        dur_m3 = time.perf_counter() - t_m3
        v_cnt3 = len(mesh_bpa.vertices)
        t_cnt3 = len(mesh_bpa.triangles)
        log(f"     BPA mesh generated in {dur_m3:.2f}s: {v_cnt3:,} vertices, {t_cnt3:,} triangles")
        bpa_exp = engine.export_all_formats(mesh_bpa, mesh_dir, base_name="bpa_mesh")
        meshing_evaluations["bpa"] = {
            "vertices": v_cnt3,
            "triangles": t_cnt3,
            "generation_time_sec": round(dur_m3, 2),
            "files": bpa_exp
        }
    except Exception as e:
        log(f"     BPA Meshing notice: {e}")
        mesh_bpa = None
        meshing_evaluations["bpa"] = {"error": str(e)}

    stage_d_sec = time.perf_counter() - t_stage_d

    # Determine Best Mesh: Poisson provides smooth watertight floor, walls, and desk closures
    best_method = "poisson"
    best_mesh = mesh_poisson
    log(f"\n  Selected primary deliverable method: {best_method.upper()} for solid room geometry and surface continuity.")

    # ------------------------------------------------------------------
    # Stage E: Multi-Angle Visual Previews
    # ------------------------------------------------------------------
    log("\n[Stage E] Rendering Multi-Angle Diagnostic Previews...")
    t_stage_e = time.perf_counter()
    rendered_views = engine.render_visual_previews(best_mesh, renders_dir, prefix="attempt_4_new_video")
    stage_e_sec = time.perf_counter() - t_stage_e
    log(f"  Saved {len(rendered_views)} visual previews in {stage_e_sec:.2f}s: {[Path(p).name for p in rendered_views]}")

    total_proc_time = time.perf_counter() - t_total_start
    final_ram = get_system_memory_mb()

    # Consolidate Metadata
    metadata = {
        "attempt_identifier": "attempt_4_new_video",
        "video_source": {
            "filename": Path(VIDEO_PATH).name,
            "full_path": str(VIDEO_PATH),
            "resolution": sampling_report["source_video"]["resolution"],
            "fps": sampling_report["source_video"]["fps"],
            "duration_sec": duration_sec,
            "total_candidate_frames": sampling_report["source_video"]["total_candidate_frames"]
        },
        "curation_summary": {
            "total_candidate_frames": sampling_report["sampling_metrics"]["total_candidate_frames"],
            "number_sampled_uniform_1fps": sampling_report["sampling_metrics"]["number_sampled_uniform_1fps"],
            "number_adaptive_keyframes": sampling_report["sampling_metrics"]["number_adaptive_keyframes"],
            "total_frames_processed_vggt": num_frames,
            "sampling_density_fps": round(num_frames / duration_sec, 3),
            "total_rejected": sampling_report["sampling_metrics"]["total_rejected"],
            "rejection_percentage": sampling_report["sampling_metrics"]["rejection_percentage"]
        },
        "vggt_throughput_benchmark": {
            "target_fps": 1.0,
            "measured_unique_throughput_fps": round(unique_throughput_fps, 2),
            "target_achieved": unique_throughput_fps >= 1.0,
            "total_inference_time_sec": round(vggt_perf["total_vggt_inference_sec"], 2),
            "model_init_time_sec": round(vggt_perf["model_init_sec"], 2),
            "window_size": config.window_size,
            "overlap": config.overlap,
            "num_windows": vggt_perf["num_windows"],
            "average_window_time_sec": round(vggt_perf["average_time_per_window_sec"], 3),
            "mean_sim3_alignment_residual_m": round(vggt_perf["mean_sim3_alignment_error_m"], 4),
            "peak_vram_allocated_mb": vram_alloc,
            "peak_vram_reserved_mb": vram_res
        },
        "trajectory": traj_diag,
        "point_cloud_statistics": pcd_stats,
        "plane_detection": plane_results,
        "meshing_evaluations": meshing_evaluations,
        "best_method": best_method,
        "runtime_performance": {
            "stage_a_curation_sec": round(stage_a_sec, 2),
            "stage_b_inference_sec": round(stage_b_sec, 2),
            "stage_c_point_cloud_sec": round(stage_c_sec, 2),
            "stage_d_meshing_sec": round(stage_d_sec, 2),
            "stage_e_rendering_sec": round(stage_e_sec, 2),
            "total_processing_time_sec": round(total_proc_time, 2),
            "system_ram_start_mb": round(initial_ram, 1),
            "system_ram_end_mb": round(final_ram, 1)
        },
        "delivered_files": {
            "point_cloud": str(compat_pcd_path.resolve()),
            "raw_point_cloud": str(raw_pcd_path.resolve()),
            "filtered_point_cloud": str(clean_pcd_path.resolve()),
            "untrimmed_mesh_obj": untrimmed_exp.get("obj"),
            "untrimmed_mesh_glb": untrimmed_exp.get("glb"),
            "postprocessed_mesh_obj": postproc_exp.get("obj"),
            "postprocessed_mesh_glb": postproc_exp.get("glb"),
            "primary_mesh_glb": meshing_evaluations[best_method]["files"].get("glb"),
            "primary_mesh_obj": meshing_evaluations[best_method]["files"].get("obj"),
            "primary_mesh_mtl": meshing_evaluations[best_method]["files"].get("mtl"),
            "primary_mesh_ply": meshing_evaluations[best_method]["files"].get("ply"),
            "renders": rendered_views
        }
    }

    meta_file = OUTPUT_DIR / "attempt_metadata.json"
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)

    log(f"\n[+] Reconstruction completed in {total_proc_time:.1f}s!")
    log(f"    Metadata saved to: {meta_file}")
    log_file.close()
    return metadata


if __name__ == "__main__":
    run_reconstruction()
