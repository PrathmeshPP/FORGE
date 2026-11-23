"""
F.O.R.G.E. — GENERALIZED INDOOR 3D RECONSTRUCTION & WALKTHROUGH PIPELINE

Provides end-to-end automated 3D reconstruction and verification for continuous indoor videos
across unfamiliar smartphone cameras, rooms, and corridors.

Workflow:
1. Video Calibration & Adaptive Frame Selection (OpenCV 5):
   - Calibrated relative sharpness & motion-blur detection.
   - Radiometric exposure & clipping validation.
   - Multi-region visual overlap tracking (4x4 spatial grid coverage & inlier entropy).
   - Viewpoint novelty classification (parallax translation vs rotational pan vs stationary pause).
   - Bridge-frame recovery for unexpected overlap drops.
2. Verified VGGT-Omega Window Alignment before Fusion:
   - Sim(3) quality checks: correspondence count, spatial spread, scale consistency, robust residuals.
   - Bounded weak-link recovery retry.
   - Section-level integrity reporting (connected vs disconnected sections).
3. Dense Point Cloud Backprojection & Floor Plane Leveling:
   - Confidence gating and ray-oriented normal field.
   - Automatic RANSAC floor leveling to canonical upright orientation.
4. Screened Poisson Surface Reconstruction:
   - Density-trimmed watertight manifold extraction with boundary decimation.
5. Photographic Texture Atlas Synthesis:
   - Aspect-preserving 8K atlas, depth-occlusion gating, 4-pass topological chart consolidation.
6. Scene-Specific Navigation & Route Validation:
   - Derives dynamic scene bounds, floor/ceiling levels, eye height, spawn point, obstacles.
   - Simulates walkthrough route: floor continuity, gap detection, obstacle clearance.
   - Classifies eligibility: WALKTHROUGH_READY | PARTIAL_MODEL | RECAPTURE_NEEDED.
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import shutil
import hashlib
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
import open3d as o3d
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.video.opencv5_frame_analyzer import (
    OpenCV5FrameAnalyzer,
    CurationReport,
    calibrate_video_characteristics
)
from src.reconstruction.point_cloud import depth_to_points, save_ply
from src.reconstruction.indoor_pipeline import (
    IndoorPipelineConfig,
    IndoorReconstructionEngine,
    compute_floor_leveling_rotation,
    Sim3AlignmentResult
)
from src.reconstruction.scene_navigation_generator import (
    derive_scene_navigation,
    SceneNavigationMetadata
)


def get_gpu_vram_mb() -> Tuple[float, float]:
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(0) / (1024 ** 2)
        res = torch.cuda.memory_reserved(0) / (1024 ** 2)
        return round(alloc, 2), round(res, 2)
    return 0.0, 0.0


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024 * 4):
            h.update(chunk)
    return h.hexdigest()


class GeneralizedIndoorReconstructionPipeline:
    def __init__(
        self,
        video_path: Path,
        output_dir: Path,
        target_frame_count: int = 60,
        scene_id: Optional[str] = None,
        config: Optional[IndoorPipelineConfig] = None,
        meshing_method: str = "poisson"
    ):
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.target_frame_count = target_frame_count
        self.scene_id = scene_id or self.output_dir.name
        self.config = config or IndoorPipelineConfig(
            image_resolution=320,
            window_size=6,
            overlap=3,
            min_depth=0.20,
            max_depth=5.5,
            min_confidence=0.65,
            min_cluster_faces=20,
            voxel_size=0.010,
            poisson_depth=10,
            density_trim_quantile=0.003,
            level_floor=True,
            meshing_method=meshing_method
        )
        self.config.meshing_method = meshing_method

        self.frames_dir = self.output_dir / "frames"
        self.camera_dir = self.output_dir / "camera_geometry"
        self.mesh_dir = self.output_dir / "meshes"
        self.renders_dir = self.output_dir / "renders"

        for d in [self.output_dir, self.frames_dir, self.camera_dir, self.mesh_dir, self.renders_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.log_path = self.output_dir / "generalized_reconstruction.log"
        self.log_file = open(self.log_path, "w", encoding="utf-8")

    def log(self, msg: str):
        t_str = time.strftime("%H:%M:%S")
        line = f"[{t_str}] {msg}"
        print(line, flush=True)
        self.log_file.write(line + "\n")
        self.log_file.flush()

    def run(self, reuse_cached_cameras: bool = False) -> Dict[str, Any]:
        self.log("=" * 80)
        self.log("  F.O.R.G.E. GENERALIZED INDOOR 3D RECONSTRUCTION PIPELINE")
        self.log(f"  Source Video: {self.video_path.name}")
        self.log(f"  Output Dir:   {self.output_dir}")
        self.log(f"  Target Frames: {self.target_frame_count}")
        if torch.cuda.is_available():
            free_mem, total_mem = torch.cuda.mem_get_info()
            self.log(f"  CUDA Device:  {torch.cuda.get_device_name(0)} (Free: {free_mem/(1024**2):.1f} MB / {total_mem/(1024**2):.1f} MB)")
        self.log("=" * 80)

        t0_master = time.perf_counter()
        durations: Dict[str, float] = {}

        # ----------------------------------------------------------------------
        # PHASE 1: Adaptive OpenCV Frame Selection
        # ----------------------------------------------------------------------
        t0_p1 = time.perf_counter()
        self.log("\n[PHASE 1] Adaptive OpenCV Frame Selection & Overlap Curation...")
        analyzer = OpenCV5FrameAnalyzer(orb_nfeatures=1000)

        cached_manifest = self.output_dir / "adaptive_frame_selection_manifest.json"
        if reuse_cached_cameras and cached_manifest.exists() and len(list(self.frames_dir.glob("*.png"))) > 10:
            self.log("  [+] Reusing previously curated frames from cache.")
            with open(cached_manifest, "r", encoding="utf-8") as f:
                curation_dict = json.load(f)
            frame_files = sorted(list(self.frames_dir.glob("*.png")))
            curated_records = [{"rank": i, "file_path": str(p.resolve())} for i, p in enumerate(frame_files)]
            curation_summary = curation_dict.get("selection_summary", {})
        else:
            curated_records, curation_report = analyzer.extract_curated_frames(
                video_path=str(self.video_path),
                output_dir=str(self.frames_dir),
                target_num_frames=self.target_frame_count,
                prefix="frame"
            )
            curation_dict = curation_report.to_dict()
            with open(self.output_dir / "adaptive_frame_selection_manifest.json", "w", encoding="utf-8") as f:
                json.dump(curation_dict, f, indent=2)
            curation_summary = curation_dict["selection_summary"]

        image_paths = [r["file_path"] for r in curated_records]
        num_frames = len(image_paths)
        durations["phase1_frame_selection_sec"] = round(time.perf_counter() - t0_p1, 2)

        self.log(f"  [+] Phase 1 Complete in {durations['phase1_frame_selection_sec']}s:")
        self.log(f"      Selected: {num_frames} frames (Bridges inserted: {curation_summary.get('bridge_frames_inserted', 0)}, Transitions: {curation_summary.get('transition_frames_retained', 0)})")
        self.log(f"      Mean Sharpness: {curation_summary.get('mean_selected_laplacian_sharpness', 0):.1f}, Contrast: {curation_summary.get('mean_selected_rms_contrast', 0):.1f}")
        self.log(f"      Mean Pairwise Overlap Score: {curation_summary.get('mean_pairwise_overlap_score', 0):.1f}, Inliers: {curation_summary.get('mean_pairwise_inliers', 0):.1f}")

        # ----------------------------------------------------------------------
        # PHASE 2: Verified VGGT Neural Window Alignment before Fusion
        # ----------------------------------------------------------------------
        t0_p2 = time.perf_counter()
        self.log("\n[PHASE 2] Neural Camera Pose Estimation & Verified Window Alignment (VGGT-Omega)...")
        engine = IndoorReconstructionEngine(self.config)

        cached_poses = self.camera_dir / "camera_poses.npz"
        cached_depth = self.camera_dir / "depth_predictions.npz"

        if reuse_cached_cameras and cached_poses.exists() and cached_depth.exists():
            self.log("  [+] Loading cached camera geometry & depth maps...")
            poses_data = np.load(cached_poses)
            depth_data = np.load(cached_depth)
            extrinsics = poses_data["extrinsics"]
            intrinsics = poses_data["intrinsics"]
            depth_maps = depth_data["depth"]
            confidence_maps = depth_data["confidence"]
            vggt_perf = json.load(open(self.camera_dir / "vggt_performance.json"))
            window_manifest = json.load(open(self.camera_dir / "vggt_window_alignment_manifest.json"))
            section_summary = vggt_perf.get("alignment_section_summary", {})
        else:
            window_results = engine.run_overlapping_windows(
                image_paths=image_paths,
                window_size=self.config.window_size,
                overlap=self.config.overlap,
                device="cuda:0" if torch.cuda.is_available() else "cpu",
                log_callback=self.log
            )
            extrinsics = window_results["extrinsics"]
            intrinsics = window_results["intrinsics"]
            depth_maps = window_results["depth"]
            confidence_maps = window_results["confidence"]
            vggt_perf = window_results["performance"]
            window_manifest = window_results["window_alignment_manifest"]
            section_summary = window_results["section_summary"]

            # Save camera geometry
            np.savez_compressed(cached_poses, extrinsics=extrinsics, intrinsics=intrinsics)
            np.savez_compressed(cached_depth, depth=depth_maps, confidence=confidence_maps)
            with open(self.camera_dir / "vggt_performance.json", "w") as f:
                json.dump(vggt_perf, f, indent=2)
            with open(self.camera_dir / "vggt_window_alignment_manifest.json", "w") as f:
                json.dump(window_manifest, f, indent=2)

        durations["phase2_vggt_alignment_sec"] = round(time.perf_counter() - t0_p2, 2)
        vram_alloc, vram_res = get_gpu_vram_mb()

        self.log(f"  [+] Phase 2 Complete in {durations['phase2_vggt_alignment_sec']}s:")
        self.log(f"      Total Windows: {vggt_perf.get('num_windows', 0)}, Mean Sim(3) Residual: {vggt_perf.get('mean_sim3_alignment_error_m', 0)*100:.2f} cm")
        self.log(f"      Join Integrity: {section_summary.get('alignment_integrity_rate', 1.0)*100:.1f}% (Strong: {section_summary.get('strong_joins', 0)}, Acceptable: {section_summary.get('acceptable_joins', 0)}, Recovered: {section_summary.get('recovered_joins', 0)}, Failed: {section_summary.get('failed_joins', 0)})")
        self.log(f"      Peak VRAM: {vram_alloc} MB allocated / {vram_res} MB reserved")

        traj_diag = engine.assess_indoor_trajectory(extrinsics)
        self.log(f"      Trajectory: {traj_diag['num_cameras']} cameras, path length: {traj_diag['total_path_length_m']}m, net disp: {traj_diag['net_displacement_m']}m, max span: {traj_diag.get('max_baseline_span_m', 0)}m")
        if traj_diag.get("is_stationary_pan", False):
            self.log(f"  [!] CAPTURE LIMITATION DETECTED: {traj_diag.get('capture_limitation')}")
            self.log(f"      Guidance: {traj_diag.get('actionable_guidance')}")

        # ----------------------------------------------------------------------
        # PHASE 3: Point-Cloud Backprojection, Floor Leveling & High-Fidelity Meshing
        # ----------------------------------------------------------------------
        t0_p3 = time.perf_counter()
        self.log("\n[PHASE 3] Dense Point Cloud Backprojection & Upright Leveling...")
        frames_bgr = [cv2.imread(p) for p in image_paths]
        clean_pts, clean_cols, raw_pcd_tuple, pcd_stats = engine.backproject_indoor_pcd(
            frames_bgr=frames_bgr,
            depth_maps=depth_maps,
            confidence_maps=confidence_maps,
            extrinsics=extrinsics,
            intrinsics=intrinsics
        )
        raw_pts, raw_cols = raw_pcd_tuple
        self.log(f"  Raw points: {pcd_stats['total_raw_points']:,} -> Valid: {pcd_stats['valid_masked_points']:,} -> Clean: {pcd_stats['clean_points']:,}")

        raw_pcd_path = self.output_dir / "raw_point_cloud.ply"
        clean_pcd_path = self.output_dir / "filtered_point_cloud.ply"
        save_ply(raw_pts, raw_cols, str(raw_pcd_path))
        save_ply(clean_pts, clean_cols, str(clean_pcd_path))

        cam_centers = [np.array(c) for c in traj_diag["camera_centers"]]

        # Detect horizontal planes & floor leveling with trajectory gravity awareness
        self.log("  Room Plane Detection & Upright Floor Leveling...")
        plane_results = engine.detect_room_planes(clean_pts, cam_centers=cam_centers, extrinsics=extrinsics)
        with open(self.output_dir / "plane_detection.json", "w") as f:
            json.dump(plane_results, f, indent=2)

        R_floor = np.eye(3, dtype=np.float32)
        horiz_floors = [p for p in plane_results["detected_planes"] if p["type"] == "horizontal_floor"]
        if horiz_floors and self.config.level_floor:
            best_floor = horiz_floors[0]
            R_floor = compute_floor_leveling_rotation(np.array(best_floor["normal"])).astype(np.float32)
            clean_pts = (R_floor @ clean_pts.T).T.astype(np.float32)
            raw_pts = (R_floor @ raw_pts.T).T.astype(np.float32)
            cam_centers = [(R_floor @ c) for c in cam_centers]
            
            # Level extrinsics so TSDF meshing, texture projection, and camera poses share the exact leveled frame
            for i in range(len(extrinsics)):
                extrinsics[i, :3, :3] = extrinsics[i, :3, :3] @ R_floor.T
            self.log(f"  [+] Floor leveling applied: Aligned horizontal floor normal {best_floor['normal']} to [0, 1, 0]")

            # Multi-Room Continuous Floor Height & Doorway Alignment (Controlled Experiment Toggle)
            if getattr(self.config, "enable_per_camera_height_correction", False):
                self.log("  [EXPERIMENTAL] Applying Continuous Floor Height Locking Across Doorways...")
                ref_pts = []
                for i in range(min(12, len(extrinsics))):
                    d = depth_maps[i].squeeze()
                    c = confidence_maps[i].squeeze()
                    val = (d >= self.config.min_depth) & (d <= 3.5) & (c >= 0.65)
                    p = depth_to_points(d, extrinsics[i], intrinsics[i])[val]
                    ref_pts.append(p[::10])
                if ref_pts:
                    all_ref = np.vstack(ref_pts)
                    pcd_ref = o3d.geometry.PointCloud()
                    pcd_ref.points = o3d.utility.Vector3dVector(all_ref)
                    model_ref, inliers_ref = pcd_ref.segment_plane(distance_threshold=0.03, ransac_n=3, num_iterations=500)
                    Y0 = float(np.median(all_ref[inliers_ref][:, 1]))
                    self.log(f"  [+] Target Global Floor Height Y0 = {Y0:.4f} m")

                    tracked_dy = 0.0
                    for i in range(len(extrinsics)):
                        d = depth_maps[i].squeeze()
                        c = confidence_maps[i].squeeze()
                        val = (d >= self.config.min_depth) & (d <= 3.5) & (c >= 0.60)
                        if not np.any(val):
                            extrinsics[i, :3, 3] -= extrinsics[i, :3, :3] @ np.array([0.0, tracked_dy, 0.0])
                            continue
                        temp_ext = extrinsics[i].copy()
                        temp_ext[:3, 3] -= temp_ext[:3, :3] @ np.array([0.0, tracked_dy, 0.0])
                        p = depth_to_points(d, temp_ext, intrinsics[i])[val]
                        cam_c = -temp_ext[:3, :3].T @ temp_ext[:3, 3]
                        pts_below = p[p[:, 1] > cam_c[1] + 0.25]
                        if len(pts_below) >= 100:
                            pcd_b = o3d.geometry.PointCloud()
                            pcd_b.points = o3d.utility.Vector3dVector(pts_below[::4])
                            model_b, inliers_b = pcd_b.segment_plane(distance_threshold=0.03, ransac_n=3, num_iterations=200)
                            n_b = np.array(model_b[:3])
                            n_b /= np.linalg.norm(n_b)
                            if n_b[1] < 0:
                                n_b = -n_b
                            tilt = np.degrees(np.arccos(np.clip(n_b[1], -1, 1)))
                            if tilt < 18.0:
                                fl_pts = np.asarray(pcd_b.points)[inliers_b]
                                cur_y = float(np.median(fl_pts[:, 1]))
                                res_dy = Y0 - cur_y
                                if abs(res_dy) < 0.25:
                                    tracked_dy += res_dy * 0.5
                        extrinsics[i, :3, 3] -= extrinsics[i, :3, :3] @ np.array([0.0, tracked_dy, 0.0])
        else:
            self.log("  [*] No horizontal floor detected. Maintaining upright coordinate frame.")
        cam_centers_np = np.array(cam_centers)

        # Meshing Selection (TSDF default for indoor geometry & furniture vs Poisson)
        t0_p_mesh = time.perf_counter()
        if self.config.meshing_method == "tsdf":
            self.log(f"  Executing Scalable TSDF Volume Surface Reconstruction (Furniture & Floor Preserving)...")
            pcd_span = float(np.max(pcd_stats["extents"])) if "extents" in pcd_stats else 4.0
            adaptive_voxel = max(0.008, min(self.config.tsdf_voxel_length, pcd_span / 350.0))
            self.log(f"  [+] Adaptive TSDF Voxel Length: {adaptive_voxel*100:.1f} cm (scene span: {pcd_span:.2f}m)")
            mesh_raw = engine.reconstruct_tsdf_mesh(
                frames_bgr=frames_bgr,
                depth_maps=depth_maps,
                confidence_maps=confidence_maps,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                target_voxel_length=adaptive_voxel
            )
            self.log(f"  [+] Raw TSDF surface extracted: {len(mesh_raw.vertices):,} vertices, {len(mesh_raw.triangles):,} triangles.")
        else:
            self.log(f"  Estimating Ray-Oriented Normals for Poisson (Guaranteed Interior-Facing)...")
            clean_pcd = o3d.geometry.PointCloud()
            clean_pcd.points = o3d.utility.Vector3dVector(clean_pts.astype(np.float64))
            clean_pcd.colors = o3d.utility.Vector3dVector(clean_cols.astype(np.float64) / 255.0 if clean_cols.max() > 1.0 else clean_cols.astype(np.float64))
            clean_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=25))
            normals = np.asarray(clean_pcd.normals, dtype=np.float64)
            dists_to_cams = np.linalg.norm(clean_pts[:, None, :] - cam_centers_np[None, :, :], axis=-1)
            closest_cam_idx = np.argmin(dists_to_cams, axis=1)
            view_rays = cam_centers_np[closest_cam_idx] - clean_pts
            view_rays /= (np.linalg.norm(view_rays, axis=1, keepdims=True) + 1e-6)
            dot_prod = np.sum(normals * view_rays, axis=1)
            flip_mask = dot_prod < 0.0
            normals[flip_mask] = -normals[flip_mask]
            clean_pcd.normals = o3d.utility.Vector3dVector(normals)

            self.log(f"  Executing Screened Poisson Surface Reconstruction (depth={self.config.poisson_depth})...")
            mesh_poisson, p_densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                clean_pcd, depth=self.config.poisson_depth, linear_fit=True
            )
            dens = np.asarray(p_densities)
            trim_thresh = np.percentile(dens, self.config.density_trim_quantile * 100.0)
            mesh_poisson.remove_vertices_by_mask(dens < trim_thresh)
            mesh_poisson.remove_degenerate_triangles()
            mesh_poisson.remove_duplicated_vertices()
            mesh_poisson.remove_duplicated_triangles()
            mesh_poisson.remove_non_manifold_edges()
            mesh_raw = mesh_poisson

        # Decimate to optimal triangle budget for 60 FPS Three.js rendering
        target_faces = min(600000, max(120000, len(mesh_raw.triangles)))
        if len(mesh_raw.triangles) > target_faces:
            mesh_final = mesh_raw.simplify_quadric_decimation(target_number_of_triangles=target_faces)
        else:
            mesh_final = mesh_raw
        mesh_final.compute_vertex_normals()
        mesh_final.compute_triangle_normals()
        self.log(f"  Final mesh generated in {time.perf_counter() - t0_p_mesh:.2f}s: {len(mesh_final.vertices):,} vertices, {len(mesh_final.triangles):,} triangles.")

        durations["phase3_meshing_sec"] = round(time.perf_counter() - t0_p3, 2)

        # ----------------------------------------------------------------------
        # PHASE 4: Photographic Texture Atlas & Multi-Format Deliverables
        # ----------------------------------------------------------------------
        t0_p4 = time.perf_counter()
        self.log("\n[PHASE 4] Photographic Texture Synthesis & Multi-Format Deliverables...")

        from scripts.run_nov1_191043_v3_reconstruction import V3PhotographicTextureEngine

        frame_paths_list = [Path(p) for p in image_paths]
        tex_engine = V3PhotographicTextureEngine(
            mesh=mesh_final,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            frame_paths=frame_paths_list,
            depth_maps=depth_maps,
            floor_rotation=np.eye(3),
            min_cos_theta=0.15
        )

        sharp_vals = [r.get("sharpness", 100.0) for r in curated_records]
        best_cams, scores, tex_stats = tex_engine.compute_face_visibility_and_best_cameras(sharp_vals)

        atlas_path = self.output_dir / "texture_atlas.png"
        tile_rects, atlas_stats = tex_engine.synthesize_aspect_preserving_8k_atlas(
            best_cam_per_face=best_cams,
            out_atlas_path=atlas_path,
            atlas_size=8192
        )

        deliverables = tex_engine.export_v3_deliverables(
            output_dir=self.output_dir,
            base_name="room_photographic_model",
            best_cam_per_face=best_cams,
            tile_rects=tile_rects,
            atlas_path=atlas_path,
            atlas_size=8192
        )
        durations["phase4_texturing_sec"] = round(time.perf_counter() - t0_p4, 2)
        self.log(f"  [+] Phase 4 Complete in {durations['phase4_texturing_sec']}s: GLB, OBJ, PLY exported.")

        # ----------------------------------------------------------------------
        # PHASE 5: Scene-Specific Walkthrough Navigation & Route Validation
        # ----------------------------------------------------------------------
        t0_p5 = time.perf_counter()
        self.log("\n[PHASE 5] Scene-Specific Walkthrough Metadata & Route Validation...")

        model_glb_rel = f"/outputs/indoor/{self.output_dir.name}/room_photographic_model.glb"
        # Compute canonical center consistent with deliverable export
        T_canon = np.array([1.0, -1.0, -1.0], dtype=np.float64)
        canonical_center = (np.asarray(mesh_final.vertices) * T_canon).mean(axis=0)
        canon_mesh_path = self.output_dir / "room_photographic_model.ply"
        deliverable_mesh = o3d.io.read_triangle_mesh(str(canon_mesh_path)) if canon_mesh_path.exists() else mesh_final

        nav_metadata = derive_scene_navigation(
            mesh=deliverable_mesh,
            extrinsics=extrinsics,
            detected_planes=plane_results,
            scene_id=self.scene_id,
            model_glb_url=model_glb_rel,
            player_radius=0.08,
            frame_manifest=curated_records,
            floor_rotation=np.eye(3),
            canonical_center=canonical_center
        )

        nav_dict = nav_metadata.to_dict()
        with open(self.output_dir / "scene_navigation_metadata.json", "w", encoding="utf-8") as f:
            json.dump(nav_dict, f, indent=2)

        durations["phase5_navigation_sec"] = round(time.perf_counter() - t0_p5, 2)
        durations["total_pipeline_time_sec"] = round(time.perf_counter() - t0_master, 2)

        route_val = nav_metadata.route_validation
        if traj_diag.get("is_stationary_pan", False):
            route_val.eligibility_status = "REJECTED_STATIONARY_PAN"
            route_val.actionable_notes.insert(0, "Stationary rotation/pan detected: insufficient translation baseline for spatial parallax. Recapture scene with walking motion (1-2m baseline).")
            nav_dict = nav_metadata.to_dict()
            with open(self.output_dir / "scene_navigation_metadata.json", "w", encoding="utf-8") as f:
                json.dump(nav_dict, f, indent=2)

        self.log(f"  [+] Phase 5 Complete in {durations['phase5_navigation_sec']}s:")
        self.log(f"      Eligibility Status: {route_val.eligibility_status}")
        self.log(f"      Floor Continuity: {route_val.floor_continuity_ratio*100:.1f}%, Obstacle Clearance: {route_val.obstacle_clearance_ratio*100:.1f}%")
        self.log(f"      Spawn Position: {nav_metadata.spawn_position}, Eye Height: {nav_metadata.eye_height}m")

        # ----------------------------------------------------------------------
        # Comprehensive Master Metadata & Summary Report
        # ----------------------------------------------------------------------
        total_time = durations["total_pipeline_time_sec"]
        vram_alloc, vram_res = get_gpu_vram_mb()

        master_metadata = {
            "pipeline_name": "F.O.R.G.E. Generalized Indoor 3D Reconstruction Pipeline",
            "scene_id": self.scene_id,
            "source_video": {
                "name": self.video_path.name,
                "path": str(self.video_path.resolve()),
                "sha256": compute_sha256(self.video_path),
                "resolution": curation_dict.get("native_resolution", [480, 854]),
                "fps": curation_dict.get("native_fps", 29.97),
                "duration_sec": curation_dict.get("duration_sec", 0.0),
                "total_frames": curation_dict.get("total_frames", 0),
                "orientation": curation_dict.get("orientation", "portrait")
            },
            "durations": durations,
            "memory": {
                "peak_vram_mb": vggt_perf.get("peak_vram_mb", vram_alloc),
                "vram_reserved_mb": vram_res
            },
            "frame_selection": curation_summary,
            "window_alignment": section_summary,
            "point_cloud": {
                "total_raw": pcd_stats["total_raw_points"],
                "clean_points": pcd_stats["clean_points"],
                "spatial_extents_m": pcd_stats["extents"]
            },
            "geometry": {
                "vertices": len(mesh_final.vertices),
                "triangles": len(mesh_final.triangles),
                "bounds_min": nav_metadata.bounds_xyz[:3],
                "bounds_max": nav_metadata.bounds_xyz[3:],
                "floor_y": nav_metadata.floor_y,
                "ceiling_y": nav_metadata.ceiling_y,
                "room_height_m": nav_metadata.room_height_m
            },
            "texturing": {
                "photographic_coverage_pct": tex_stats["photographic_coverage_pct"],
                "direct_photographic_faces": tex_stats["direct_photographic_faces"],
                "active_cameras_used": tex_stats["active_cameras_used"],
                "atlas_size": atlas_stats["atlas_size"],
                "tile_size": atlas_stats["tile_size"]
            },
            "navigation": {
                "eligibility_status": route_val.eligibility_status,
                "floor_continuity_pct": round(route_val.floor_continuity_ratio * 100.0, 1),
                "obstacle_clearance_pct": round(route_val.obstacle_clearance_ratio * 100.0, 1),
                "spawn_position": nav_metadata.spawn_position,
                "actionable_notes": route_val.actionable_notes
            },
            "deliverables": deliverables
        }

        with open(self.output_dir / "reconstruction_metadata.json", "w", encoding="utf-8") as f:
            json.dump(master_metadata, f, indent=2)

        # Generate markdown report
        report_md = f"""# F.O.R.G.E. Generalized Indoor Reconstruction Report
**Scene:** `{self.scene_id}`  
**Video:** `{self.video_path.name}` ({master_metadata['source_video']['duration_sec']}s, {master_metadata['source_video']['resolution'][0]}x{master_metadata['source_video']['resolution'][1]})  
**Eligibility:** `{route_val.eligibility_status}`  
**Total Runtime:** `{total_time:.1f}s` | **Peak VRAM:** `{master_metadata['memory']['peak_vram_mb']:.1f} MB`

## 1. Frame Selection Diagnostics (OpenCV 5)
- **Selected Frames:** {num_frames}
- **Bridge Frames Recovered:** {curation_summary.get('bridge_frames_inserted', 0)}
- **Transition Frames Retained:** {curation_summary.get('transition_frames_retained', 0)}
- **Mean Sharpness:** {curation_summary.get('mean_selected_laplacian_sharpness', 0):.1f}
- **Mean Pairwise Overlap Score:** {curation_summary.get('mean_pairwise_overlap_score', 0):.1f}
- **Geometric Verification Rate:** {curation_summary.get('geometric_verification_rate', 0)*100:.1f}%

## 2. VGGT Window Alignment Diagnostics
- **Windows:** {vggt_perf.get('num_windows', 0)}
- **Mean Sim(3) Alignment Residual:** {vggt_perf.get('mean_sim3_alignment_error_m', 0)*100:.2f} cm
- **Join Integrity Rate:** {section_summary.get('alignment_integrity_rate', 1.0)*100:.1f}%
- **Strong Joins:** {section_summary.get('strong_joins', 0)} | **Acceptable:** {section_summary.get('acceptable_joins', 0)} | **Recovered:** {section_summary.get('recovered_joins', 0)} | **Failed:** {section_summary.get('failed_joins', 0)}

## 3. 3D Model & Photographic Texture
- **Clean Point Cloud:** {pcd_stats['clean_points']:,} points
- **Mesh Vertices:** {len(mesh_final.vertices):,} | **Triangles:** {len(mesh_final.triangles):,}
- **Direct Photographic Coverage:** {tex_stats['photographic_coverage_pct']:.2f}%
- **Texture Atlas:** 8192x8192 aspect-preserving PBR atlas

## 4. Scene-Specific Navigation & Route Validation
- **Status:** `{route_val.eligibility_status}`
- **Floor Continuity:** {route_val.floor_continuity_ratio*100:.1f}%
- **Obstacle Clearance:** {route_val.obstacle_clearance_ratio*100:.1f}%
- **Spawn Position:** {nav_metadata.spawn_position} (Eye level: {nav_metadata.eye_height}m)
- **Actionable Advice:**
{chr(10).join(f'  - {n}' for n in route_val.actionable_notes)}
"""
        with open(self.output_dir / "reconstruction_report.md", "w", encoding="utf-8") as f:
            f.write(report_md)

        self.log(f"\n[+] RECONSTRUCTION FINISHED SUCCESSFULLY in {total_time:.1f}s!")
        self.log(f"    Master Metadata: {self.output_dir / 'reconstruction_metadata.json'}")
        self.log(f"    Report: {self.output_dir / 'reconstruction_report.md'}")
        self.log(f"    Deliverable GLB: {deliverables.get('glb', deliverables.get('glb_canonical', ''))}")
        self.log("=" * 80)

        return master_metadata


def main():
    parser = argparse.ArgumentParser(description="F.O.R.G.E. Generalized Indoor Reconstruction")
    parser.add_argument("--video", type=str, required=True, help="Path to continuous indoor video")
    parser.add_argument("--output", type=str, required=True, help="Path to output directory")
    parser.add_argument("--target-frames", type=int, default=60, help="Target curated frame count")
    parser.add_argument("--scene-id", type=str, default=None, help="Scene ID for viewer")
    parser.add_argument("--meshing", type=str, default="poisson", choices=["tsdf", "poisson"], help="Surface meshing engine")
    parser.add_argument("--reuse-cached-cameras", action="store_true", help="Reuse cached camera geometry & depth")
    args = parser.parse_args()

    pipeline = GeneralizedIndoorReconstructionPipeline(
        video_path=Path(args.video),
        output_dir=Path(args.output),
        target_frame_count=args.target_frames,
        scene_id=args.scene_id,
        meshing_method=args.meshing
    )
    pipeline.run(reuse_cached_cameras=args.reuse_cached_cameras)


if __name__ == "__main__":
    main()
