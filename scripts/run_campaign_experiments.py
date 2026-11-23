"""F.O.R.G.E — Best Reconstruction Campaign: Controlled Multi-Approach Evaluation.

Executes and benchmarks 3 distinct technical reconstruction approaches on the 16-frame dense set:
1. Candidate 1 (Poisson 16F): Screened Poisson (depth 9, support 5.5, gentle trim 0.001, hole repair 24)
2. Candidate 2 (TSDF 16F): Truncated Signed Distance Function (TSDF) volume integration
3. Candidate 3 (BPA 16F): Multi-scale Ball Pivoting Algorithm (BPA) with local point-spacing radii

All candidates:
- Use OpenCV 5 16-frame dense sampling (eliminating temporal/spatial baseline chasms).
- Use true world camera centers for optical axis and normal orientation.
- Filter low-confidence horizon/sky pseudo-depths (c >= 1.8, 0.15m <= d <= 4.5m).
- Export complete self-contained GLB models with embedded 4096x4096 photographic texture atlases.
- Generate multi-view preview contact sheets and quantitative geometric metrics.
"""

from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
import open3d as o3d
import trimesh
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reconstruction.point_cloud import depth_to_points
from src.reconstruction.multiframe_stitcher import SurfaceConnectivityMesher
from src.reconstruction.photographic_texture import (
    extract_native_frames,
    scale_camera_intrinsics,
    MultiViewTextureEngine
)
from scripts.validate_sprint1_geometry_and_filtering import compute_relative_depth_jump
from scripts.run_phase2d_recovery import (
    render_card_preview,
    render_neutral_geometry,
    render_detail_view,
    measure_mesh_metrics,
    compute_file_sha256
)

CAMPAIGN_DIR = PROJECT_ROOT / "outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign"
UPSTREAM_DIR = CAMPAIGN_DIR / "upstream_16_frames"
PRED_DIR = UPSTREAM_DIR / "predictions"
FRAMES_DIR = UPSTREAM_DIR / "frames"
VIDEO_PATH = Path(r"C:\Users\25beevdt025\Downloads\WhatsApp Video 2026-10-24 at 17.53.55.mp4")


def prepare_dense_point_cloud(
    conf_min: float = 1.8,
    d_min: float = 0.15,
    d_max: float = 4.5
) -> Tuple[o3d.geometry.PointCloud, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Path], List[float]]:
    poses = np.load(PRED_DIR / "camera_poses.npz")
    depth = np.load(PRED_DIR / "depth_predictions.npz")
    extrinsics = poses["extrinsics"] # (N, 3, 4)
    intrinsics = poses["intrinsics"] # (N, 3, 3)
    d_maps = np.squeeze(depth["depth"]) # (N, H, W)
    c_maps = np.squeeze(depth["confidence"]) # (N, H, W)

    frame_files = sorted(list(FRAMES_DIR.glob("*.png")))
    timestamps = [float(f.stem.split("_")[-1].replace("s", "")) if "_" in f.stem else float(idx * 2.0) for idx, f in enumerate(frame_files)]
    N, H_inf, W_inf = d_maps.shape

    # True camera optical centers in world: C = -R^T @ T
    cam_centers = np.array([-extrinsics[i, :3, :3].T @ extrinsics[i, :3, 3] for i in range(N)])

    pts_frames, col_frames, nrm_frames = [], [], []

    for i in range(N):
        im_bgr = cv2.imread(str(frame_files[i]))
        im_rgb = cv2.cvtColor(cv2.resize(im_bgr, (W_inf, H_inf)), cv2.COLOR_BGR2RGB) / 255.0

        d = d_maps[i]
        c = c_maps[i]
        jm = compute_relative_depth_jump(d, kernel_size=3) > 0.12
        valid = (d >= d_min) & (d <= d_max) & (c >= conf_min) & (~jm)

        pts = depth_to_points(d[None, ...], extrinsics[i:i+1], intrinsics[i:i+1])[0][valid]
        col = im_rgb[valid]

        pf = o3d.geometry.PointCloud()
        pf.points = o3d.utility.Vector3dVector(pts)
        pf.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=30))
        pf.orient_normals_towards_camera_location(camera_location=cam_centers[i])

        pts_frames.append(pts)
        col_frames.append(col)
        nrm_frames.append(np.asarray(pf.normals))

    pts_all = np.concatenate(pts_frames, axis=0)
    colors_all = np.concatenate(col_frames, axis=0)
    normals_all = np.concatenate(nrm_frames, axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_all)
    pcd.colors = o3d.utility.Vector3dVector(colors_all)
    pcd.normals = o3d.utility.Vector3dVector(normals_all)

    extents = np.ptp(pts_all, axis=0)
    fine_voxel = max(float(np.max(extents)) / 1400.0, 0.004)
    pcd_fused = pcd.voxel_down_sample(voxel_size=fine_voxel)
    print(f"[*] Dense 16-Frame Fused Point Cloud: {len(pcd_fused.points):,} points (extents: {extents.round(3)})")

    return pcd_fused, extrinsics, intrinsics, d_maps, c_maps, frame_files, timestamps


def finalize_and_export_candidate(
    mesh: o3d.geometry.TriangleMesh,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    d_maps: np.ndarray,
    timestamps: List[float],
    out_dir: Path,
    candidate_name: str,
    filter_small_clusters: bool = True
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    # Repair connectivity
    mesh_rep, rep_meta = SurfaceConnectivityMesher.repair_surface_connectivity(mesh, max_hole_vertices=24)

    # Prune tiny disconnected clusters (<100 triangles)
    if filter_small_clusters and len(mesh_rep.triangles) > 0:
        clusters, num_faces, _ = mesh_rep.cluster_connected_triangles()
        clusters = np.asarray(clusters)
        num_faces = np.asarray(num_faces)
        tiny = np.where(num_faces < 100)[0]
        if len(tiny) > 0:
            mesh_rep.remove_triangles_by_mask(np.isin(clusters, tiny))
            mesh_rep.remove_unreferenced_vertices()

    print(f"    Mesh ready for texturing: {len(mesh_rep.vertices):,} vertices, {len(mesh_rep.triangles):,} triangles")

    # Native uncompressed frames
    native_dir = out_dir / "native_frames"
    extracted_info = extract_native_frames(str(VIDEO_PATH), timestamps, native_dir)
    native_paths = [info[0] for info in extracted_info]
    native_w, native_h = extracted_info[0][1], extracted_info[0][2]
    infer_h, infer_w = d_maps.shape[1], d_maps.shape[2]

    scaled_intr = scale_camera_intrinsics(intrinsics, (infer_w, infer_h), (native_w, native_h))

    # Multi-view texture baking (4096 atlas)
    print(f"[*] Baking 4096x4096 texture atlas for {candidate_name}...")
    engine = MultiViewTextureEngine(
        mesh=mesh_rep,
        extrinsics=extrinsics,
        intrinsics_native=scaled_intr,
        frame_paths=native_paths,
        depth_maps=d_maps,
        moving_object_strategy=None
    )
    best_cam, scores, stats = engine.compute_face_visibility_and_scores()

    obj_path = out_dir / f"{candidate_name}_reconstruction.obj"
    obj_path, atlas_path, atlas_stats = engine.export_packed_atlas_obj(obj_path, best_cam, atlas_size=4096)

    ply_path = out_dir / f"{candidate_name}_reconstruction.ply"
    engine.export_photographic_vertex_ply(ply_path, best_cam)

    glb_path = out_dir / f"{candidate_name}_reconstruction.glb"
    tm = trimesh.load(str(obj_path), process=False)
    glb_bytes = tm.export(file_type="glb")
    with open(glb_path, "wb") as f_glb:
        f_glb.write(glb_bytes)
    print(f"    GLB exported: {glb_path.name} ({glb_path.stat().st_size / (1024*1024):.2f} MB)")

    # Previews
    m_ren = o3d.io.read_triangle_mesh(str(ply_path))
    card_path = render_card_preview(m_ren, out_dir / "model_preview.png", title=f"F.O.R.G.E {candidate_name.upper()}")
    neutral_path = render_neutral_geometry(m_ren, out_dir / "neutral_geometry_preview.png", front=[0.0, -0.707, -0.707], up=[0.0, -0.707, 0.707], zoom=0.65)
    detail_path = render_detail_view(m_ren, out_dir / "texture_detail_preview.png", scene_name=candidate_name)

    elapsed = time.perf_counter() - t0
    ply_metrics = measure_mesh_metrics(ply_path)
    glb_metrics = measure_mesh_metrics(glb_path)

    meta = {
        "candidate": candidate_name,
        "runtime_sec": round(elapsed, 2),
        "glb": {
            "path": str(glb_path.resolve()),
            "size_mb": glb_metrics["file_size_mb"],
            "sha256": glb_metrics["sha256"]
        },
        "obj": {
            "path": str(obj_path.resolve()),
            "size_mb": round(obj_path.stat().st_size / (1024*1024), 2),
            "sha256": compute_file_sha256(obj_path)
        },
        "ply": {
            "path": str(ply_path.resolve()),
            "size_mb": ply_metrics["file_size_mb"],
            "sha256": ply_metrics["sha256"]
        },
        "atlas": {
            "path": str(atlas_path.resolve()),
            "size_mb": round(atlas_path.stat().st_size / (1024*1024), 2),
            "sha256": compute_file_sha256(atlas_path)
        },
        "geometry": ply_metrics
    }
    with open(out_dir / "reconstruction_metadata.json", "w") as f_meta:
        json.dump(meta, f_meta, indent=2)

    return meta


def run_all_candidates():
    print("=" * 80)
    print("RUNNING BEST RECONSTRUCTION CAMPAIGN (3 DISTINCT APPROACHES)")
    print("=" * 80)

    pcd_fused, ext, intr, d_maps, c_maps, frame_files, timestamps = prepare_dense_point_cloud()

    results = {}

    # -------------------------------------------------------------
    # CANDIDATE 1: Screened Poisson HQ (16 Frames)
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CANDIDATE 1: Dense 16-Frame Screened Poisson HQ")
    print("=" * 70)
    out_c1 = CAMPAIGN_DIR / "candidate_16f_poisson"
    mesh_c1, _ = SurfaceConnectivityMesher.build_poisson_hq(
        pcd_fused,
        depth=9,
        gentle_trim_quantile=0.001,
        support_distance_factor=5.5
    )
    meta_c1 = finalize_and_export_candidate(
        mesh=mesh_c1,
        extrinsics=ext,
        intrinsics=intr,
        d_maps=d_maps,
        timestamps=timestamps,
        out_dir=out_c1,
        candidate_name="candidate_16f_poisson"
    )
    results["candidate_16f_poisson"] = meta_c1

    # -------------------------------------------------------------
    # CANDIDATE 2: TSDF Volume Integration (16 Frames)
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CANDIDATE 2: Dense 16-Frame TSDF Volume Integration")
    print("=" * 70)
    out_c2 = CAMPAIGN_DIR / "candidate_16f_tsdf"

    # Set up TSDF volume with calibrated voxel resolution
    extents = np.ptp(np.asarray(pcd_fused.points), axis=0)
    voxel_length = 0.012 # 1.2 cm voxel resolution
    sdf_trunc = 0.04    # 4 cm truncation distance

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    N, H_inf, W_inf = d_maps.shape
    for i in range(N):
        im_bgr = cv2.imread(str(frame_files[i]))
        im_rgb = cv2.cvtColor(cv2.resize(im_bgr, (W_inf, H_inf)), cv2.COLOR_BGR2RGB)
        d = d_maps[i]
        c = c_maps[i]
        jm = compute_relative_depth_jump(d, kernel_size=3) > 0.12
        # Clean depth image (zero out invalid pixels)
        valid = (d >= 0.15) & (d <= 4.5) & (c >= 1.8) & (~jm)
        d_clean = np.where(valid, d, 0.0).astype(np.float32)

        color_o3d = o3d.geometry.Image(im_rgb)
        depth_o3d = o3d.geometry.Image(d_clean)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d, depth_scale=1.0, depth_trunc=4.5, convert_rgb_to_intensity=False
        )

        pinhole_int = o3d.camera.PinholeCameraIntrinsic(
            W_inf, H_inf,
            intr[i, 0, 0], intr[i, 1, 1],
            intr[i, 0, 2], intr[i, 1, 2]
        )
        ext4x4 = np.eye(4)
        ext4x4[:3, :] = ext[i]
        volume.integrate(rgbd, pinhole_int, ext4x4)

    mesh_c2 = volume.extract_triangle_mesh()
    mesh_c2.compute_vertex_normals()
    print(f"[*] TSDF extracted mesh: {len(mesh_c2.vertices):,} vertices, {len(mesh_c2.triangles):,} triangles")

    meta_c2 = finalize_and_export_candidate(
        mesh=mesh_c2,
        extrinsics=ext,
        intrinsics=intr,
        d_maps=d_maps,
        timestamps=timestamps,
        out_dir=out_c2,
        candidate_name="candidate_16f_tsdf"
    )
    results["candidate_16f_tsdf"] = meta_c2

    # -------------------------------------------------------------
    # CANDIDATE 3: Multi-Scale Ball Pivoting (16 Frames)
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CANDIDATE 3: Dense 16-Frame Multi-Scale Ball Pivoting")
    print("=" * 70)
    out_c3 = CAMPAIGN_DIR / "candidate_16f_bpa"
    mesh_c3, _ = SurfaceConnectivityMesher.build_bpa_connected(
        pcd_fused,
        radii_multipliers=(1.5, 3.0, 6.0, 12.0)
    )
    meta_c3 = finalize_and_export_candidate(
        mesh=mesh_c3,
        extrinsics=ext,
        intrinsics=intr,
        d_maps=d_maps,
        timestamps=timestamps,
        out_dir=out_c3,
        candidate_name="candidate_16f_bpa"
    )
    results["candidate_16f_bpa"] = meta_c3

    # Summary
    summary_path = CAMPAIGN_DIR / "candidates_benchmark_summary.json"
    with open(summary_path, "w") as f_sum:
        json.dump(results, f_sum, indent=2)
    print(f"\n[+] Saved full 3-candidate benchmark summary to: {summary_path}")


if __name__ == "__main__":
    run_all_candidates()
