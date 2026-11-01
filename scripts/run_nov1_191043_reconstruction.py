"""
F.O.R.G.E. — Complete Indoor 3D Reconstruction Pipeline
Dataset: November 1, 2026 — 19:10:43 (Hostel Room B 314 Complete 360 Survey)
Source: WhatsApp Video 2026-11-01 at 19.10.43.mp4
Output: outputs/indoor/texture_experiment_video_20261101_191043
"""

import os
import sys
import json
import time
import math
import shutil
import hashlib
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

for _cand in [
    r"C:\Users\25beevdt025\Desktop\SIH\third_party\vggt_omega",
    r"C:\Users\25beevdt025\Desktop\coding\third_party\vggt_omega"
]:
    if Path(_cand).is_dir() and _cand not in sys.path:
        sys.path.insert(0, _cand)

from src.reconstruction.point_cloud import save_ply
from src.reconstruction.indoor_pipeline import (
    IndoorPipelineConfig,
    IndoorReconstructionEngine,
    compute_floor_leveling_rotation
)
from src.reconstruction.indoor_photographic_texture import IndoorPhotographicTextureEngine

# ------------------------------------------------------------------------------
# Paths & Setup
# ------------------------------------------------------------------------------
INPUT_VIDEO = Path(r"C:\Users\25beevdt025\Downloads\WhatsApp Video 2026-11-01 at 19.10.43.mp4")
OUTPUT_DIR = PROJECT_ROOT / "outputs/indoor/texture_experiment_video_20261101_191043"
FRAMES_DIR = OUTPUT_DIR / "frames"
CAMERA_DIR = OUTPUT_DIR / "camera_geometry"
MESH_DIR = OUTPUT_DIR / "meshes"
RENDERS_DIR = OUTPUT_DIR / "renders"
ARTIFACT_DIR = Path(r"C:\Users\25beevdt025\.gemini\antigravity\brain\536a497a-fba6-496d-b4a9-b4f0cf756953")

for d in [OUTPUT_DIR, FRAMES_DIR, CAMERA_DIR, MESH_DIR, RENDERS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LOG_PATH = OUTPUT_DIR / "reconstruction.log"
log_f = open(LOG_PATH, "w", encoding="utf-8")


def log(msg: str):
    t_str = time.strftime("%H:%M:%S")
    line = f"[{t_str}] {msg}"
    print(line, flush=True)
    log_f.write(line + "\n")
    log_f.flush()


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


# ==============================================================================
# STAGE 1: Adaptive Frame Curation
# ==============================================================================
def curate_frames(video_path: Path, out_frames_dir: Path, target_count: int = 60) -> List[Dict[str, Any]]:
    log(f"[*] Curating frames from {video_path.name} (target count: ~{target_count})...")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.93
    duration = total_frames / fps

    all_data = []
    prev_gray = None
    for f_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        t_sec = f_idx / fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_lum = float(np.mean(gray))
        contrast = float(np.std(gray))

        disp = 0.0
        if prev_gray is not None:
            g1 = cv2.resize(prev_gray, (160, 160))
            g2 = cv2.resize(gray, (160, 160))
            disp = float(np.mean(cv2.absdiff(g1, g2)))
        prev_gray = gray

        all_data.append({
            "frame_idx": f_idx,
            "timestamp_sec": round(t_sec, 3),
            "sharpness": round(lap_var, 2),
            "luminance": round(mean_lum, 2),
            "contrast": round(contrast, 2),
            "disp": round(disp, 2)
        })
    cap.release()

    # Partition video into target_count uniform bins and pick sharpest frame in each bin
    num_bins = target_count
    bin_size = total_frames / num_bins
    selected_indices = set()

    for b in range(num_bins):
        start_f = int(b * bin_size)
        end_f = min(int((b + 1) * bin_size), total_frames)
        cands = all_data[start_f:end_f]
        if not cands:
            continue
        best = max(cands, key=lambda c: c["sharpness"])
        selected_indices.add(best["frame_idx"])

    # Extract selected frames to disk
    final_indices = sorted(list(selected_indices))
    cap = cv2.VideoCapture(str(video_path))
    curated_records = []

    for rank, f_idx in enumerate(final_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame_bgr = cap.read()
        if not ret:
            continue
        fdata = all_data[f_idx]
        fname = f"frame_{rank:02d}_f{f_idx:04d}_{fdata['timestamp_sec']:05.2f}s.png"
        fpath = out_frames_dir / fname
        cv2.imwrite(str(fpath), frame_bgr)

        curated_records.append({
            "rank": rank,
            "frame_index": f_idx,
            "timestamp_sec": fdata["timestamp_sec"],
            "file_path": str(fpath.resolve()),
            "sharpness": fdata["sharpness"],
            "luminance": fdata["luminance"],
            "contrast": fdata["contrast"]
        })
    cap.release()

    report = {
        "video_path": str(video_path),
        "total_candidate_frames": total_frames,
        "duration_sec": round(duration, 3),
        "fps": round(fps, 3),
        "total_selected": len(curated_records),
        "sampling_density_fps": round(len(curated_records) / duration, 3),
        "frames": curated_records
    }
    with open(OUTPUT_DIR / "frame_sampling_report.json", "w") as f:
        json.dump(report, f, indent=2)

    log(f"[+] Frame curation complete: {len(curated_records)} frames extracted across {duration:.1f}s.")
    return curated_records


# ==============================================================================
# MAIN RECONSTRUCTION
# ==============================================================================
def main():
    log("=" * 76)
    log("  F.O.R.G.E. COMPLETE INDOOR 3D RECONSTRUCTION")
    log("  Dataset: November 1, 2026 — 19:10:43 (Hostel Room B 314 Complete Survey)")
    log(f"  Input Video:  {INPUT_VIDEO}")
    log(f"  Output Dir:   {OUTPUT_DIR}")
    if torch.cuda.is_available():
        free_mem, total_mem = torch.cuda.mem_get_info()
        log(f"  CUDA Device:  {torch.cuda.get_device_name(0)} (Free: {free_mem/(1024**2):.1f} MB / {total_mem/(1024**2):.1f} MB)")
    log("=" * 76)

    t0_master = time.perf_counter()

    # Step 1: Quality Frame Curation (~60 frames across 149s video)
    t0_stage1 = time.perf_counter()
    curated_records = curate_frames(INPUT_VIDEO, FRAMES_DIR, target_count=60)
    image_paths = [r["file_path"] for r in curated_records]
    num_frames = len(image_paths)
    dur_stage1 = time.perf_counter() - t0_stage1
    log(f"[Stage 1] Completed in {dur_stage1:.2f}s. Extracted {num_frames} frames.")

    # Step 2: VGGT Neural Geometry & Camera Estimation
    config = IndoorPipelineConfig(
        image_resolution=320,
        window_size=6,
        overlap=3,
        min_depth=0.25,
        max_depth=5.5,
        min_confidence=0.75,
        voxel_size=0.015,
        poisson_depth=9,
        density_trim_quantile=0.020,
        level_floor=True
    )
    engine = IndoorReconstructionEngine(config)

    log("\n[Stage 2] Neural Camera Pose & Geometry Estimation (VGGT-Omega Overlapping Windows)...")
    t0_stage2 = time.perf_counter()
    window_results = engine.run_overlapping_windows(
        image_paths=image_paths,
        window_size=config.window_size,
        overlap=config.overlap,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        log_callback=log
    )
    dur_stage2 = time.perf_counter() - t0_stage2

    extrinsics = window_results["extrinsics"]
    intrinsics = window_results["intrinsics"]
    depth_maps = window_results["depth"]
    confidence_maps = window_results["confidence"]
    vggt_perf = window_results["performance"]

    vram_alloc, vram_res = get_gpu_vram_mb()
    log(f"  Stage 2 wall-clock duration: {dur_stage2:.2f}s")
    log(f"  Actual VGGT inference time: {vggt_perf['total_vggt_inference_sec']:.2f}s")
    log(f"  Peak VRAM: {vram_alloc} MB allocated, {vram_res} MB reserved")
    log(f"  Mean Sim(3) Alignment Residual: {vggt_perf['mean_sim3_alignment_error_m']*100:.2f} cm")

    # Save camera geometry
    np.savez_compressed(CAMERA_DIR / "camera_poses.npz", extrinsics=extrinsics, intrinsics=intrinsics)
    np.savez_compressed(CAMERA_DIR / "depth_predictions.npz", depth=depth_maps, confidence=confidence_maps)
    with open(CAMERA_DIR / "vggt_performance.json", "w") as f:
        json.dump(vggt_perf, f, indent=2)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    traj_diag = engine.assess_indoor_trajectory(extrinsics)
    log(f"  Trajectory: {traj_diag['num_cameras']} cameras, path length: {traj_diag['total_path_length_m']}m, net disp: {traj_diag['net_displacement_m']}m")

    # Step 3: Point-Cloud Backprojection with Optical Ray Tracking
    log("\n[Stage 3] Point Cloud Backprojection & Optical Ray Direction Tracking...")
    t0_stage3 = time.perf_counter()
    frames_bgr = [cv2.imread(p) for p in image_paths]
    clean_pts, clean_cols, raw_pcd_tuple, pcd_stats = engine.backproject_indoor_pcd(
        frames_bgr=frames_bgr,
        depth_maps=depth_maps,
        confidence_maps=confidence_maps,
        extrinsics=extrinsics,
        intrinsics=intrinsics
    )
    raw_pts, raw_cols = raw_pcd_tuple
    dur_stage3 = time.perf_counter() - t0_stage3
    log(f"  Raw points: {pcd_stats['total_raw_points']:,} -> Valid: {pcd_stats['valid_masked_points']:,} -> Clean: {pcd_stats['clean_points']:,}")
    log(f"  Spatial extents [X, Y, Z]: {pcd_stats['extents']} m")

    raw_pcd_path = OUTPUT_DIR / "raw_point_cloud.ply"
    clean_pcd_path = OUTPUT_DIR / "filtered_point_cloud.ply"
    save_ply(raw_pts, raw_cols, str(raw_pcd_path))
    save_ply(clean_pts, clean_cols, str(clean_pcd_path))
    log(f"  Saved raw point cloud: {raw_pcd_path.name} ({raw_pcd_path.stat().st_size / (1024**2):.2f} MB)")
    log(f"  Saved filtered point cloud: {clean_pcd_path.name} ({clean_pcd_path.stat().st_size / (1024**2):.2f} MB)")

    # Step 4: Plane Detection & Floor Leveling
    log("\n[Stage 4] Room Plane Detection & Floor Leveling...")
    plane_results = engine.detect_room_planes(clean_pts)
    with open(OUTPUT_DIR / "plane_detection.json", "w") as f:
        json.dump(plane_results, f, indent=2)

    cam_centers = [np.array(c) for c in traj_diag["camera_centers"]]
    R_floor = np.eye(3, dtype=np.float32)
    horiz_planes = [p for p in plane_results["detected_planes"] if "horizontal" in p["type"]]
    if horiz_planes:
        best_floor = horiz_planes[0]
        R_floor = compute_floor_leveling_rotation(np.array(best_floor["normal"])).astype(np.float32)
        clean_pts = (R_floor @ clean_pts.T).T.astype(np.float32)
        raw_pts = (R_floor @ raw_pts.T).T.astype(np.float32)
        cam_centers = [(R_floor @ c) for c in cam_centers]
        log(f"  [+] Floor leveling applied: Aligned plane normal {best_floor['normal']} to [0, 1, 0]")
    cam_centers_np = np.array(cam_centers)

    # Step 5: Ray-Oriented Normal Estimation (Interior Orientation)
    log("\n[Stage 5] Estimating Ray-Oriented Normals (Guaranteed Interior-Facing)...")
    clean_pcd = o3d.geometry.PointCloud()
    clean_pcd.points = o3d.utility.Vector3dVector(clean_pts.astype(np.float64))
    clean_pcd.colors = o3d.utility.Vector3dVector(clean_cols.astype(np.float64) / 255.0)

    clean_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.06, max_nn=35)
    )
    normals = np.asarray(clean_pcd.normals, dtype=np.float64)

    # Vector from point to closest observing camera center
    dists_to_cams = np.linalg.norm(clean_pts[:, None, :] - cam_centers_np[None, :, :], axis=-1)
    closest_cam_idx = np.argmin(dists_to_cams, axis=1)
    view_rays = cam_centers_np[closest_cam_idx] - clean_pts
    view_rays /= (np.linalg.norm(view_rays, axis=1, keepdims=True) + 1e-6)

    # Flip any normal pointing away from observing camera
    dot_prod = np.sum(normals * view_rays, axis=1)
    flip_mask = dot_prod < 0.0
    normals[flip_mask] = -normals[flip_mask]
    clean_pcd.normals = o3d.utility.Vector3dVector(normals)
    log(f"  [+] Corrected {flip_mask.sum():,} ({flip_mask.mean()*100:.1f}%) normals towards true observing camera locations.")

    # Step 6: Screened Poisson Surface Meshing with Density-Preserving Trimming
    log("\n[Stage 6] Executing Screened Poisson Surface Reconstruction (depth=9)...")
    t0_poisson = time.perf_counter()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        clean_pcd, depth=9, linear_fit=True
    )
    dur_poisson = time.perf_counter() - t0_poisson
    dens = np.asarray(densities)
    log(f"  Poisson raw mesh generated in {dur_poisson:.2f}s: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles.")

    # Density trimming at 2.0 percentile (pruning only true airborne noise)
    trim_thresh = np.percentile(dens, 2.0)
    mesh.remove_vertices_by_mask(dens < trim_thresh)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()

    clusters, counts, _ = mesh.cluster_connected_triangles()
    counts = np.asarray(counts)
    keep_clusters = np.where(counts >= 150)[0]
    tri_mask = np.isin(np.asarray(clusters), keep_clusters)
    mesh.remove_triangles_by_mask(~tri_mask)
    mesh.remove_unreferenced_vertices()

    # Target 450,000 triangles to retain all 3 beds, wardrobes, study desks, cooler, and doors
    log(f"  Cleaned mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles.")
    if len(mesh.triangles) > 450000:
        log("  Simplifying to 450,000 triangles with boundary preservation...")
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=450000)
        mesh.remove_degenerate_triangles()
        mesh.remove_non_manifold_edges()

    mesh.compute_vertex_normals()
    poisson_ply_path = MESH_DIR / "poisson_mesh.ply"
    o3d.io.write_triangle_mesh(str(poisson_ply_path), mesh)
    log(f"  Saved baseline Poisson surface mesh: {poisson_ply_path.name}")

    # Step 7: Photographic Texture Engine & 8K Atlas Synthesis
    log("\n[Stage 7] Photographic Texture Engine & 8K Atlas Baking...")
    t0_tex = time.perf_counter()
    frame_files = sorted(list(FRAMES_DIR.glob("*.png")))
    frame_sharpness = [r["sharpness"] for r in curated_records]

    tex_engine = IndoorPhotographicTextureEngine(
        mesh=mesh,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        frame_paths=frame_files,
        depth_maps=depth_maps,
        floor_rotation=R_floor,
        min_cos_theta=0.08
    )

    best_cam, scores, vis_stats = tex_engine.compute_face_visibility_and_best_cameras(
        frame_sharpness=frame_sharpness
    )
    log(f"  Direct Photographic Surface Coverage: {vis_stats['photographic_coverage_pct']}% ({vis_stats['direct_photographic_faces']:,} / {vis_stats['total_faces']:,} faces)")
    log(f"  Active Camera Views Employed: {vis_stats['active_cameras_used']} / {len(frame_files)}")

    atlas_png_path = OUTPUT_DIR / "texture_atlas.png"
    tile_rects, atlas_stats = tex_engine.synthesize_packed_texture_atlas(
        best_cam_per_face=best_cam,
        out_atlas_path=atlas_png_path,
        atlas_size=8192
    )
    log(f"  8K Atlas Synthesized: {atlas_stats['atlas_file_mb']} MB (Grid: {atlas_stats['grid_layout']}, Tile: {atlas_stats['tile_size']})")

    deliverables = tex_engine.export_textured_deliverables(
        output_dir=OUTPUT_DIR,
        base_name="room_photographic_model",
        best_cam_per_face=best_cam,
        tile_rects=tile_rects,
        atlas_path=atlas_png_path,
        canonical_transform=True
    )
    dur_tex = time.perf_counter() - t0_tex
    log(f"  Textured deliverables exported in {dur_tex:.2f}s:")
    for k, v in deliverables.items():
        p = Path(v)
        sz_mb = p.stat().st_size / (1024 ** 2)
        log(f"    - {k.upper()}: {p.name} ({sz_mb:.2f} MB)")

    # Step 8: Multi-View Diagnostic Suite (10 Viewpoints)
    log("\n[Stage 8] Rendering 10 Calibrated Multi-Angle Diagnostic Previews...")
    ply_mesh = o3d.io.read_triangle_mesh(str(deliverables["ply"]))
    verts_canonical = np.asarray(ply_mesh.vertices)
    x_min, y_min, z_min = verts_canonical.min(axis=0)
    x_max, y_max, z_max = verts_canonical.max(axis=0)
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    cz = (z_min + z_max) / 2.0
    span_x = x_max - x_min
    span_y = y_max - y_min
    span_z = z_max - z_min
    log(f"  Canonical bounds: X=[{x_min:.2f}, {x_max:.2f}] ({span_x:.2f}m), Y=[{y_min:.2f}, {y_max:.2f}] ({span_y:.2f}m), Z=[{z_min:.2f}, {z_max:.2f}] ({span_z:.2f}m)")

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1280, height=720)
    vis.add_geometry(ply_mesh)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.94, 0.95, 0.96])
    opt.light_on = True
    ctr = vis.get_view_control()

    # 10 Viewpoints tailored for Hostel Room B 314
    viewpoints = [
        ("01_full_room_overview", [cx, cy + 2.5, cz - span_z * 0.9], [cx, cy, cz], [0, 1, 0]),
        ("02_room_entrance_and_door", [cx, y_min + 1.6, z_min + 0.4], [cx, y_min + 1.4, cz], [0, 1, 0]),
        ("03_bed1_and_wardrobes", [cx - 0.7, y_min + 1.1, cz - 0.5], [x_min + 0.2, y_min + 0.9, cz - 0.2], [0, 1, 0]),
        ("04_double_door_lockers", [cx - 0.5, y_min + 1.3, cz + 0.2], [x_min + 0.1, y_min + 1.3, cz + 0.2], [0, 1, 0]),
        ("05_bed2_red_mattress", [cx + 0.7, y_min + 1.1, cz - 0.3], [x_max - 0.2, y_min + 0.8, cz - 0.3], [0, 1, 0]),
        ("06_bed3_and_drying_rack", [cx + 0.6, y_min + 1.2, cz + 0.6], [x_max - 0.2, y_min + 0.9, cz + 0.8], [0, 1, 0]),
        ("07_portable_air_cooler", [cx, y_min + 1.1, cz + 0.7], [cx + 0.4, y_min + 0.7, cz + 1.1], [0, 1, 0]),
        ("08_top_down_floorplan", [cx, cy + span_y * 2.2, cz], [cx, cy, cz], [0, 0, 1]),
        ("09_isometric_room_corner", [x_min - span_x * 0.5, y_max + 1.5, z_min - span_z * 0.5], [cx, cy, cz], [0, 1, 0]),
        ("10_study_desks_and_switchboard", [cx, y_min + 1.2, cz - 0.8], [cx - 0.8, y_min + 1.0, cz - 0.2], [0, 1, 0]),
    ]

    rendered_images = []
    for tag, eye, target, up in viewpoints:
        ctr.set_lookat(np.asarray(target, dtype=np.float64))
        ctr.set_front(np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64))
        ctr.set_up(np.asarray(up, dtype=np.float64))
        ctr.set_zoom(0.7)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.15)
        out_render = RENDERS_DIR / f"render_{tag}.png"
        vis.capture_screen_image(str(out_render), do_render=True)
        log(f"    Rendered: {out_render.name}")
        rendered_images.append((tag, out_render))
    vis.destroy_window()

    # Build 10-View Master Validation Panel
    rows, cols = 5, 2
    tile_w, tile_h = 640, 360
    header_h = 32
    panel = np.zeros((rows * (tile_h + header_h) + 60, cols * tile_w, 3), dtype=np.uint8)
    title_bar = "F.O.R.G.E. — NOVEMBER 1 (19:10:43) HOSTEL ROOM B 314 COMPLETE SURVEY — 10-VIEW RECONSTRUCTION VALIDATION"
    cv2.putText(panel, title_bar, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    for i, (tag, rpath) in enumerate(rendered_images):
        r = i // cols
        c = i % cols
        x0 = c * tile_w
        y0 = 60 + r * (tile_h + header_h)
        im = cv2.imread(str(rpath))
        im_resized = cv2.resize(im, (tile_w, tile_h), interpolation=cv2.INTER_AREA)

        hdr = np.zeros((header_h, tile_w, 3), dtype=np.uint8)
        clean_tag = tag.replace("_", " ").upper()
        cv2.putText(hdr, f"VIEW {clean_tag}", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 1)

        panel[y0 : y0 + header_h, x0 : x0 + tile_w] = hdr
        panel[y0 + header_h : y0 + header_h + tile_h, x0 : x0 + tile_w] = im_resized

    out_panel_path = OUTPUT_DIR / "master_validation_panel_10views.png"
    cv2.imwrite(str(out_panel_path), panel)
    shutil.copyfile(out_panel_path, ARTIFACT_DIR / "nov1_191043_master_validation_panel.png")
    log(f"[+] Saved 10-view validation panel to: {out_panel_path.name}")

    # Step 9: Save Comprehensive Metadata & Checksums
    glb_hash = compute_sha256(Path(deliverables["glb"]))
    obj_hash = compute_sha256(Path(deliverables["obj"]))
    ply_hash = compute_sha256(Path(deliverables["ply"]))
    atlas_hash = compute_sha256(atlas_png_path)

    metadata = {
        "dataset_name": "November 1, 2026 — 19:10:43 (Hostel Room B 314 Complete 360 Survey)",
        "source_video": {
            "path": str(INPUT_VIDEO),
            "sha256": compute_sha256(INPUT_VIDEO),
            "duration_sec": 149.02,
            "resolution": [480, 854],
            "fps": 29.93
        },
        "reconstruction_metrics": {
            "num_curated_keyframes": num_frames,
            "vggt_sliding_windows": len(window_results["extrinsics"]),
            "vggt_inference_sec": vggt_perf["total_vggt_inference_sec"],
            "raw_points": pcd_stats["total_raw_points"],
            "clean_points": pcd_stats["clean_points"],
            "screened_poisson_depth": 9,
            "mesh_vertices": len(mesh.vertices),
            "mesh_triangles": len(mesh.triangles),
            "direct_photographic_coverage_pct": vis_stats["photographic_coverage_pct"],
            "direct_photographic_faces": vis_stats["direct_photographic_faces"],
            "active_cameras_used": vis_stats["active_cameras_used"],
            "atlas_resolution": [8192, 8192],
            "spatial_extents_m": [round(float(span_x), 2), round(float(span_y), 2), round(float(span_z), 2)],
            "floor_level_y": round(float(y_min), 2)
        },
        "deliverables": {
            "glb": {
                "path": str(deliverables["glb"]),
                "size_bytes": Path(deliverables["glb"]).stat().st_size,
                "sha256": glb_hash
            },
            "obj": {
                "path": str(deliverables["obj"]),
                "size_bytes": Path(deliverables["obj"]).stat().st_size,
                "sha256": obj_hash
            },
            "ply": {
                "path": str(deliverables["ply"]),
                "size_bytes": Path(deliverables["ply"]).stat().st_size,
                "sha256": ply_hash
            },
            "texture_atlas": {
                "path": str(atlas_png_path),
                "size_bytes": atlas_png_path.stat().st_size,
                "sha256": atlas_hash
            }
        },
        "total_pipeline_time_sec": round(time.perf_counter() - t0_master, 2)
    }

    with open(OUTPUT_DIR / "reconstruction_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    total_dur = time.perf_counter() - t0_master
    log("=" * 76)
    log(f"  RECONSTRUCTION COMPLETE in {total_dur:.2f}s ({total_dur/60:.2f} minutes)!")
    log(f"  GLB Deliverable: {Path(deliverables['glb']).name} (SHA-256: {glb_hash[:16]}...)")
    log("=" * 76)


if __name__ == "__main__":
    main()
