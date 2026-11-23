r"""
F.O.R.G.E. — High-Frame-Density Controlled Experiment Harness.
Evaluates the impact of sampling density (1.5 FPS, 2.0 FPS, 3.0 FPS, 4.0 FPS, and Dual Pool)
on geometric reconstruction accuracy, camera alignment, point-cloud density, VRAM, and texture fidelity.
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
import open3d as o3d
import torch
import psutil
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.video.adaptive_frame_curator import AdaptiveIndoorFrameCurator
from src.reconstruction.point_cloud import save_ply
from src.reconstruction.indoor_pipeline import (
    IndoorPipelineConfig,
    IndoorReconstructionEngine,
    compute_floor_leveling_rotation
)
from src.reconstruction.stage2_texture_engine import (
    Stage2TextureEngine,
    compute_crease_aware_normals,
    save_mesh_ply_float32
)
from src.reconstruction.pose_interpolator import slerp_pose, validate_pose_reprojection

VIDEO_1_PATH = r"C:\Users\25beevdt025\Downloads\WhatsApp Video 2026-10-30 at 17.02.34.mp4"
VIDEO_2_PATH = r"C:\Users\25beevdt025\Downloads\WhatsApp Video 2026-10-30 at 17.05.00.mp4"
EXP_ROOT = PROJECT_ROOT / "outputs/indoor/frame_density_experiment"


def get_nvidia_smi_vram_mb() -> float:
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            capture_output=True, text=True, check=True
        )
        return float(res.stdout.strip())
    except Exception:
        return 0.0


def run_single_density_configuration(
    video_path: str,
    dataset_name: str,
    config_name: str,
    target_fps: float,
    output_dir: Path,
    is_dual_pool: bool = False,
    geom_fps: float = 2.0,
    tex_fps: float = 4.0,
    max_depth: float = 5.5,
    min_depth: float = 0.20,
    voxel_size: float = 0.012,
    log_func=print
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    log_func(f"\n{'='*70}")
    log_func(f"  Configuration: {config_name} on {dataset_name}")
    log_func(f"  Target FPS: {target_fps} | Dual Pool: {is_dual_pool}")
    log_func(f"  Output: {output_dir}")
    log_func(f"{'='*70}")

    vram_before = get_nvidia_smi_vram_mb()
    log_func(f"[*] Pre-run VRAM (nvidia-smi): {vram_before:.1f} MB")

    t_start = time.perf_counter()

    # 1. Adaptive Frame Curation
    t_cur_start = time.perf_counter()
    curator = AdaptiveIndoorFrameCurator(video_path)

    if not is_dual_pool:
        cur_report = curator.curate_density(target_fps=target_fps, output_frames_dir=frames_dir)
        geom_records = cur_report["frames"]
        tex_records = geom_records
    else:
        # Dual pool: curate geometry at geom_fps and texture at tex_fps
        geom_dir = frames_dir / "geom"
        tex_dir = frames_dir / "tex"
        geom_report = curator.curate_density(target_fps=geom_fps, output_frames_dir=geom_dir)
        tex_report = curator.curate_density(target_fps=tex_fps, output_frames_dir=tex_dir)
        geom_records = geom_report["frames"]
        tex_records = tex_report["frames"]
        cur_report = {
            "total_candidate_frames": geom_report["total_candidate_frames"],
            "selected_count": len(geom_records) + len(tex_records),
            "rejected_count": geom_report["rejected_count"],
            "geom_count": len(geom_records),
            "tex_count": len(tex_records),
            "rejection_breakdown": geom_report["rejection_breakdown"]
        }

    cur_time = time.perf_counter() - t_cur_start
    log_func(f"[+] Stage A Curation Complete in {cur_time:.2f}s:")
    log_func(f"    Candidates: {cur_report['total_candidate_frames']} | Selected: {len(geom_records)} geom, {len(tex_records)} tex")

    # 2. VGGT Overlapping Window Inference
    config = IndoorPipelineConfig(
        image_resolution=320,
        window_size=6,
        overlap=3,
        min_depth=min_depth,
        max_depth=max_depth,
        min_confidence=0.75,
        voxel_size=voxel_size,
        poisson_depth=9,
        density_trim_quantile=0.015,
        support_dist_multiplier=8.0,
        level_floor=True
    )
    engine = IndoorReconstructionEngine(config)

    geom_image_paths = [r["file_path"] for r in geom_records]

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    t_vggt_start = time.perf_counter()
    window_results = engine.run_overlapping_windows(
        image_paths=geom_image_paths,
        window_size=config.window_size,
        overlap=config.overlap,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        log_callback=log_func
    )
    vggt_wall_time = time.perf_counter() - t_vggt_start

    extrinsics = window_results["extrinsics"]
    intrinsics = window_results["intrinsics"]
    depth_maps = window_results["depth"]
    confidence_maps = window_results["confidence"]
    vggt_perf = window_results["performance"]

    peak_pytorch_alloc = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    peak_pytorch_res = torch.cuda.max_memory_reserved() / (1024**2) if torch.cuda.is_available() else 0.0
    peak_smi_vram = get_nvidia_smi_vram_mb()

    log_func(f"[+] Stage B VGGT Inference Complete in {vggt_wall_time:.2f}s:")
    log_func(f"    Sim(3) Residual: {vggt_perf['mean_sim3_alignment_error_m']*100:.2f} cm")
    log_func(f"    Peak VRAM: PyTorch Alloc {peak_pytorch_alloc:.1f} MB, Res {peak_pytorch_res:.1f} MB | SMI {peak_smi_vram:.1f} MB")

    # Save camera poses
    cam_dir = output_dir / "camera_geometry"
    cam_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cam_dir / "camera_poses.npz", extrinsics=extrinsics, intrinsics=intrinsics)

    # 3. Point Cloud Processing & Leveling
    t_pcd_start = time.perf_counter()
    frames_bgr = [cv2.imread(p) for p in geom_image_paths]

    clean_pts, clean_cols, raw_tuple, pcd_stats = engine.backproject_indoor_pcd(
        frames_bgr=frames_bgr,
        depth_maps=depth_maps,
        confidence_maps=confidence_maps,
        extrinsics=extrinsics,
        intrinsics=intrinsics
    )
    pcd_time = time.perf_counter() - t_pcd_start
    log_func(f"[+] Stage C Point Cloud Fused in {pcd_time:.2f}s: {pcd_stats['total_raw_points']:,} raw -> {pcd_stats['clean_points']:,} clean points")

    # Floor plane detection & leveling
    plane_results = engine.detect_room_planes(clean_pts)
    R_floor = None
    traj_diag = engine.assess_indoor_trajectory(extrinsics)
    cam_centers = [np.array(c) for c in traj_diag["camera_centers"]]

    if config.level_floor:
        horiz_planes = [p for p in plane_results["detected_planes"] if "horizontal" in p["type"]]
        if horiz_planes:
            best_floor = horiz_planes[0]
            R_floor = compute_floor_leveling_rotation(np.array(best_floor["normal"]))
            clean_pts = (R_floor @ clean_pts.T).T.astype(np.float32)
            cam_centers = [(R_floor @ c) for c in cam_centers]

    # 4. Screened Poisson Surface Reconstruction
    t_mesh_start = time.perf_counter()
    _, poisson_mesh = engine.reconstruct_poisson_mesh(
        points=clean_pts,
        colors=clean_cols,
        cam_centers=cam_centers,
        depth=config.poisson_depth
    )

    # Decimate to target 250k triangles for constant controlled benchmarking
    target_tris = 250000
    if len(poisson_mesh.triangles) > target_tris:
        poisson_mesh = poisson_mesh.simplify_quadric_decimation(target_tris)
        poisson_mesh.remove_degenerate_triangles()
        poisson_mesh.remove_duplicated_triangles()
        poisson_mesh.remove_duplicated_vertices()
        poisson_mesh.remove_non_manifold_edges()

    poisson_mesh_path = output_dir / "poisson_mesh.ply"
    o3d.io.write_triangle_mesh(str(poisson_mesh_path), poisson_mesh)
    mesh_time = time.perf_counter() - t_mesh_start
    log_func(f"[+] Stage D Meshing Complete in {mesh_time:.2f}s: {len(poisson_mesh.vertices):,} V, {len(poisson_mesh.triangles):,} F")

    # 5. Photographic Texture Baking with Stage 2 Engine
    t_tex_start = time.perf_counter()

    if not is_dual_pool:
        active_tex_paths = [Path(p) for p in geom_image_paths]
        active_extrinsics = extrinsics
        active_intrinsics = intrinsics
    else:
        # Dual pool: interpolate camera poses for texture frames
        log_func("[*] Dual Pool: Validating interpolated poses for additional texture frames...")
        geom_timestamps = np.array([r["timestamp_sec"] for r in geom_records])
        interp_tex_paths = []
        interp_ext_list = []
        interp_intr_list = []

        validated_tex_count = 0
        rejected_tex_count = 0

        for r_tex in tex_records:
            t_tex = r_tex["timestamp_sec"]
            p_tex = Path(r_tex["file_path"])

            # Find bounding geometry frames
            idx_after = np.searchsorted(geom_timestamps, t_tex)
            if idx_after == 0:
                interp_tex_paths.append(p_tex)
                interp_ext_list.append(extrinsics[0][:3, :4].copy())
                interp_intr_list.append(intrinsics[0].copy())
                validated_tex_count += 1
            elif idx_after >= len(geom_timestamps):
                interp_tex_paths.append(p_tex)
                interp_ext_list.append(extrinsics[-1][:3, :4].copy())
                interp_intr_list.append(intrinsics[-1].copy())
                validated_tex_count += 1
            else:
                idx_prev = idx_after - 1
                t_prev = geom_timestamps[idx_prev]
                t_next = geom_timestamps[idx_after]
                dt = t_next - t_prev
                alpha = (t_tex - t_prev) / dt if dt > 1e-4 else 0.0

                R_prev = extrinsics[idx_prev, :3, :3]
                t_prev_vec = extrinsics[idx_prev, :3, 3]
                R_next = extrinsics[idx_after, :3, :3]
                t_next_vec = extrinsics[idx_after, :3, 3]

                R_int, t_int = slerp_pose(R_prev, t_prev_vec, R_next, t_next_vec, alpha)
                K_int = intrinsics[idx_prev]

                # Reprojection validation
                img_prev = cv2.imread(str(geom_image_paths[idx_prev]))
                img_next = cv2.imread(str(geom_image_paths[idx_after]))
                img_tex = cv2.imread(str(p_tex))

                is_valid, err_px = validate_pose_reprojection(
                    img_prev, img_next, img_tex, R_int, t_int, K_int, max_reprojection_error_px=4.5
                )

                if is_valid:
                    ext_3x4 = np.zeros((3, 4), dtype=np.float32)
                    ext_3x4[:3, :3] = R_int
                    ext_3x4[:3, 3] = t_int
                    interp_tex_paths.append(p_tex)
                    interp_ext_list.append(ext_3x4)
                    interp_intr_list.append(K_int)
                    validated_tex_count += 1
                else:
                    rejected_tex_count += 1

        log_func(f"    Dual Pool Pose Validation: {validated_tex_count} accepted, {rejected_tex_count} rejected due to reprojection inconsistency")
        active_tex_paths = interp_tex_paths
        active_extrinsics = np.array(interp_ext_list, dtype=np.float32)
        active_intrinsics = np.array(interp_intr_list, dtype=np.float32)

    tex_engine = Stage2TextureEngine(
        mesh=poisson_mesh,
        extrinsics=active_extrinsics,
        intrinsics=active_intrinsics,
        frame_paths=active_tex_paths,
        floor_rotation=R_floor
    )

    scores, vis_mask, occ_mask = tex_engine.compute_photographic_quality_matrix()
    master_cams = tex_engine.select_master_keyframes(scores, target_cameras=16)
    best_cam_mrf, mrf_info = tex_engine.optimize_camera_charts_mrf(
        scores=scores, master_cameras=master_cams, seam_penalty=0.45, max_iterations=8
    )
    gains = tex_engine.compute_exposure_compensation_gains(
        master_cameras=master_cams, best_cam_per_face=best_cam_mrf, vis_mask=vis_mask
    )

    atlas_path = output_dir / "texture_atlas.png"
    tile_rects, atlas_info = tex_engine.synthesize_native_aspect_atlas(
        master_cameras=master_cams, exposure_gains=gains, out_atlas_path=atlas_path, atlas_dim=4096
    )
    cov_ply_path = output_dir / "surface_coverage_heatmap.ply"
    cov_stats = tex_engine.generate_four_class_coverage_mesh(
        output_ply_path=cov_ply_path, vis_mask=vis_mask, occ_mask=occ_mask, best_cam_per_face=best_cam_mrf
    )

    final_deliverables = tex_engine.export_stage2_deliverables(
        output_dir=output_dir,
        base_name="model_density_result",
        best_cam_per_face=best_cam_mrf,
        tile_rects=tile_rects,
        atlas_path=atlas_path
    )
    tex_time = time.perf_counter() - t_tex_start
    total_time = time.perf_counter() - t_start

    # Clean GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metrics = {
        "dataset": dataset_name,
        "configuration": config_name,
        "target_fps": target_fps,
        "is_dual_pool": is_dual_pool,
        "timing": {
            "curation_sec": round(cur_time, 2),
            "vggt_inference_sec": round(vggt_perf["total_vggt_inference_sec"], 2),
            "vggt_wall_clock_sec": round(vggt_wall_time, 2),
            "point_cloud_fusion_sec": round(pcd_time, 2),
            "meshing_sec": round(mesh_time, 2),
            "texturing_sec": round(tex_time, 2),
            "total_reconstruction_sec": round(total_time, 2)
        },
        "frame_counts": {
            "total_candidates": cur_report["total_candidate_frames"],
            "selected_geometry_frames": len(geom_records),
            "selected_texture_frames": len(active_tex_paths),
            "rejected_frames": cur_report["rejected_count"],
            "rejection_breakdown": cur_report["rejection_breakdown"]
        },
        "memory": {
            "peak_pytorch_allocated_mb": round(peak_pytorch_alloc, 1),
            "peak_pytorch_reserved_mb": round(peak_pytorch_res, 1),
            "peak_nvidia_smi_vram_mb": round(peak_smi_vram, 1)
        },
        "camera_alignment": {
            "mean_sim3_residual_cm": round(vggt_perf["mean_sim3_alignment_error_m"] * 100.0, 3),
            "vggt_windows_processed": vggt_perf.get("windows_processed", 0)
        },
        "point_cloud": {
            "raw_points": pcd_stats["total_raw_points"],
            "clean_points": pcd_stats["clean_points"],
            "retention_rate_pct": pcd_stats["retention_rate"]
        },
        "mesh_geometry": {
            "vertices": len(poisson_mesh.vertices),
            "triangles": len(poisson_mesh.triangles),
            "seam_count": mrf_info["optimized_mrf_seams"],
            "total_internal_edges": mrf_info["total_internal_edges"],
            "seam_ratio_pct": mrf_info["final_seam_ratio_pct"],
            "seam_reduction_pct": mrf_info["seam_reduction_pct"]
        },
        "surface_coverage": {
            "direct_green_faces": cov_stats["directly_photographed_green_faces"],
            "direct_green_pct": cov_stats["direct_coverage_pct"],
            "occluded_yellow_faces": cov_stats["physically_occluded_yellow_faces"],
            "occluded_yellow_pct": cov_stats["occluded_pct"],
            "unobserved_red_faces": cov_stats["unobserved_red_faces"],
            "unobserved_red_pct": cov_stats["unobserved_pct"],
            "seam_boundary_blue_faces": cov_stats["seam_transition_blue_faces"],
            "seam_boundary_blue_pct": cov_stats["seam_boundary_pct"]
        },
        "deliverables": {
            "glb": str(final_deliverables["glb"]),
            "obj": str(final_deliverables["obj"]),
            "ply": str(final_deliverables["ply"]),
            "atlas": str(atlas_path)
        }
    }

    report_path = output_dir / "execution_metrics.json"
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)

    log_func(f"[+] Completed {config_name} in {total_time:.1f}s | Results saved to {report_path.name}")
    return metrics


def main():
    print("=" * 72)
    print("  F.O.R.G.E. Controlled Frame-Density Reconstruction Campaign")
    print("  Evaluating: Baseline (1.5 FPS), Exp A (2.0 FPS), Exp B (3.0 FPS),")
    print("              Exp C (4.0 FPS), and Exp D (Dual-Pool)")
    print("=" * 72)

    experiments = [
        # Dataset 1: Room B 314 (40.24s)
        {
            "video_path": VIDEO_1_PATH,
            "dataset_name": "video_1_room",
            "max_depth": 5.5,
            "min_depth": 0.20,
            "voxel_size": 0.012,
            "runs": [
                {"name": "baseline_1_5fps", "target_fps": 1.5, "dual": False},
                {"name": "exp_a_2fps", "target_fps": 2.0, "dual": False},
                {"name": "exp_b_3fps", "target_fps": 3.0, "dual": False},
                {"name": "exp_c_4fps", "target_fps": 4.0, "dual": False},
                {"name": "exp_d_dual_pool", "target_fps": 2.0, "dual": True, "geom_fps": 2.0, "tex_fps": 4.0}
            ]
        },
        # Dataset 2: Hostel Corridor (82.83s)
        {
            "video_path": VIDEO_2_PATH,
            "dataset_name": "video_2_corridor",
            "max_depth": 7.5,
            "min_depth": 0.25,
            "voxel_size": 0.015,
            "runs": [
                {"name": "baseline_1_5fps", "target_fps": 1.5, "dual": False},
                {"name": "exp_a_2fps", "target_fps": 2.0, "dual": False},
                {"name": "exp_b_3fps", "target_fps": 3.0, "dual": False},
                {"name": "exp_c_4fps", "target_fps": 4.0, "dual": False},
                {"name": "exp_d_dual_pool", "target_fps": 2.0, "dual": True, "geom_fps": 2.0, "tex_fps": 4.0}
            ]
        }
    ]

    campaign_summary = {}

    for exp in experiments:
        d_name = exp["dataset_name"]
        campaign_summary[d_name] = {}
        out_base = EXP_ROOT / d_name

        for r in exp["runs"]:
            c_name = r["name"]
            run_out = out_base / c_name
            metrics_file = run_out / "execution_metrics.json"
            glb_file = run_out / "model_density_result.glb"

            if metrics_file.exists() and glb_file.exists():
                print(f"[+] Found existing completed run: {c_name} on {d_name}, reusing validated results.")
                with open(metrics_file, "r") as f:
                    res = json.load(f)
                campaign_summary[d_name][c_name] = res
                continue

            res = run_single_density_configuration(
                video_path=exp["video_path"],
                dataset_name=d_name,
                config_name=c_name,
                target_fps=r["target_fps"],
                output_dir=run_out,
                is_dual_pool=r["dual"],
                geom_fps=r.get("geom_fps", 2.0),
                tex_fps=r.get("tex_fps", 4.0),
                max_depth=exp["max_depth"],
                min_depth=exp["min_depth"],
                voxel_size=exp["voxel_size"],
                log_func=print
            )
            campaign_summary[d_name][c_name] = res

    # Master summary report
    master_report_path = EXP_ROOT / "frame_density_experiment_report.json"
    master_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(master_report_path, "w") as f:
        json.dump(campaign_summary, f, indent=2)

    print("\n" + "=" * 72)
    print(f"[+] ALL FRAME-DENSITY EXPERIMENTS COMPLETED!")
    print(f"[+] Master Report Saved to: {master_report_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()

