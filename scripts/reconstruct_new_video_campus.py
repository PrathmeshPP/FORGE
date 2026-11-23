r"""
F.O.R.G.E — Complete 3D Surface Reconstruction and Photographic Texturing
for DJI Drone Video (Campus Solar Complex).
"""

import os
import sys
import time
import json
import shutil
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
import open3d as o3d
import trimesh
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reconstruction.point_cloud import depth_to_points, voxel_grid_fusion, save_ply
from src.reconstruction.multiframe_stitcher import SurfaceConnectivityMesher
from src.reconstruction.photographic_texture import (
    compute_geometry_hash,
    scale_camera_intrinsics,
    MultiViewTextureEngine
)
from scripts.validate_sprint1_geometry_and_filtering import compute_relative_depth_jump

OUTPUT_BASE = PROJECT_ROOT / "outputs/new_video_reconstructions/whatsapp_video_2026_10_24"
FRAMES_DIR = OUTPUT_BASE / "frames"
CAMERA_DIR = OUTPUT_BASE / "camera"
POINTCLOUD_DIR = OUTPUT_BASE / "pointcloud"
MESH_DIR = OUTPUT_BASE / "mesh"
REPORTS_DIR = OUTPUT_BASE / "reports"
VIEWER_DIR = OUTPUT_BASE / "viewer"

for d in [POINTCLOUD_DIR, MESH_DIR, REPORTS_DIR, VIEWER_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1048576):
            h.update(chunk)
    return h.hexdigest()


def render_scene_views(
    mesh: o3d.geometry.TriangleMesh,
    out_dir: Path,
    prefix: str = "render_view_"
) -> Dict[str, Path]:
    """Renders top, frontal, oblique, and close-up views with high visual contrast."""
    views = {}
    W, H = 1280, 720
    ref_bbox = mesh.get_axis_aligned_bounding_box()
    ctr_pt = ref_bbox.get_center()

    # Orientation: Nadir/Oblique aerial overhead
    configs = {
        "top": {"front": [0.0, 0.0, -1.0], "up": [0.0, -1.0, 0.0], "zoom": 0.72},
        "frontal": {"front": [0.0, -0.45, -0.89], "up": [0.0, -1.0, 0.0], "zoom": 0.72},
        "oblique": {"front": [0.40, -0.45, -0.80], "up": [0.0, -1.0, 0.0], "zoom": 0.72},
        "closeup": {"front": [0.35, -0.45, -0.82], "up": [0.0, -1.0, 0.0], "zoom": 0.35}
    }

    for vname, cfg in configs.items():
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=W, height=H)
        vis.add_geometry(mesh)
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.08, 0.10, 0.14])
        opt.mesh_color_option = o3d.visualization.MeshColorOption.Color

        ctrl = vis.get_view_control()
        ctrl.set_lookat(ctr_pt)
        ctrl.set_front(cfg["front"])
        ctrl.set_up(cfg["up"])
        ctrl.set_zoom(cfg["zoom"])

        vis.poll_events()
        vis.update_renderer()
        buf = vis.capture_screen_float_buffer(do_render=True)
        vis.destroy_window()

        img_path = out_dir / f"{prefix}{vname}.png"
        Image.fromarray((np.asarray(buf) * 255).astype(np.uint8)).save(img_path)
        views[vname] = img_path

    # Neutral Clay Shading Render
    vis_clay = o3d.visualization.Visualizer()
    vis_clay.create_window(visible=False, width=W, height=H)
    m_clay = o3d.geometry.TriangleMesh(mesh)
    m_clay.compute_vertex_normals()
    gray_cols = np.full((len(m_clay.vertices), 3), 0.72)
    m_clay.vertex_colors = o3d.utility.Vector3dVector(gray_cols)
    vis_clay.add_geometry(m_clay)
    opt = vis_clay.get_render_option()
    opt.background_color = np.asarray([0.08, 0.10, 0.14])
    opt.mesh_color_option = o3d.visualization.MeshColorOption.Color
    opt.mesh_shade_option = o3d.visualization.MeshShadeOption.Color
    opt.light_on = True

    ctrl = vis_clay.get_view_control()
    ctrl.set_lookat(ctr_pt)
    ctrl.set_front([0.40, -0.45, -0.80])
    ctrl.set_up([0.0, -1.0, 0.0])
    ctrl.set_zoom(0.72)

    vis_clay.poll_events()
    vis_clay.update_renderer()
    buf_clay = vis_clay.capture_screen_float_buffer(do_render=True)
    vis_clay.destroy_window()

    clay_path = out_dir / "neutral_geometry_preview.png"
    Image.fromarray((np.asarray(buf_clay) * 255).astype(np.uint8)).save(clay_path)
    views["neutral_clay"] = clay_path

    return views


def render_4panel_visual_review(
    views: Dict[str, Path],
    atlas_path: Path,
    output_path: Path,
    title: str = "F.O.R.G.E 3D Reconstruction — Campus Solar Complex"
) -> Path:
    W_p, H_p = 640, 480
    header_h = 75
    canvas = Image.new("RGB", (W_p * 4, header_h + H_p), (14, 18, 24))
    draw = ImageDraw.Draw(canvas)

    draw.text((25, 14), title.upper(), fill=(255, 255, 255))
    draw.text((25, 42), "Verified VGGT-Omega Foundation Model & OpenCV 5 Curation | Relative Scale (Metric Unverified)", fill=(170, 200, 230))

    # Panel 1: Top Nadir View
    im1 = Image.open(views["top"]).resize((W_p, H_p), Image.Resampling.LANCZOS)
    canvas.paste(im1, (0, header_h))

    # Panel 2: 3D Oblique Perspective
    im2 = Image.open(views["oblique"]).resize((W_p, H_p), Image.Resampling.LANCZOS)
    canvas.paste(im2, (W_p, header_h))

    # Panel 3: Neutral Clay Geometry
    im3 = Image.open(views["neutral_clay"]).resize((W_p, H_p), Image.Resampling.LANCZOS)
    canvas.paste(im3, (W_p * 2, header_h))

    # Panel 4: UV Texture Atlas
    if atlas_path.is_file():
        im4 = Image.open(atlas_path).resize((W_p, H_p), Image.Resampling.LANCZOS)
    else:
        im4 = Image.open(views["closeup"]).resize((W_p, H_p), Image.Resampling.LANCZOS)
    canvas.paste(im4, (W_p * 3, header_h))

    # Badges
    badges = [
        (0, "1. Top Nadir Overview", "(Roof Panels & Courtyard)", (100, 220, 255)),
        (W_p, "2. Oblique Perspective", "(Building Facades & Foliage)", (120, 240, 160)),
        (W_p * 2, "3. Neutral Geometry", "(Screened Poisson Manifold)", (255, 200, 100)),
        (W_p * 3, "4. 4K UV Texture Atlas", f"({atlas_path.name})", (240, 140, 220))
    ]
    for x_off, c_title, c_sub, col in badges:
        draw.rectangle([x_off + 12, header_h + 12, x_off + 380, header_h + 54], fill=(12, 16, 22))
        draw.text((x_off + 20, header_h + 16), c_title, fill=col)
        draw.text((x_off + 20, header_h + 34), c_sub, fill=(200, 220, 240))

    canvas.save(output_path, "PNG")
    return output_path


def main():
    t_start = time.time()
    print("=" * 80)
    print("F.O.R.G.E — 3D Reconstruction Pipeline: Campus Solar Complex")
    print("=" * 80)

    # 1. Ingest Predictions
    print("\n[Step 1/7] Ingesting camera poses and depth predictions...")
    poses_file = CAMERA_DIR / "camera_poses.npz"
    depth_file = CAMERA_DIR / "depth_predictions.npz"

    if not poses_file.is_file() or not depth_file.is_file():
        raise FileNotFoundError(f"Missing predictions in {CAMERA_DIR}")

    poses_data = np.load(poses_file)
    depth_data = np.load(depth_file)

    extrinsics = poses_data["extrinsics"] # (N, 3, 4)
    intrinsics = poses_data["intrinsics"] # (N, 3, 3)
    d_maps = np.squeeze(depth_data["depth"]) # (N, H, W)
    c_maps = np.squeeze(depth_data["confidence"]) # (N, H, W)

    num_cams = len(extrinsics)
    infer_h, infer_w = d_maps.shape[1], d_maps.shape[2]
    print(f"  Loaded {num_cams} camera views. Inference resolution: {infer_w}x{infer_h}")

    # 2. Ingest Extracted 4K Native Frames
    print("\n[Step 2/7] Loading extracted native 4K frames...")
    frame_files = sorted(list(FRAMES_DIR.glob("*.png")))
    if len(frame_files) != num_cams:
        raise ValueError(f"Expected {num_cams} frames in {FRAMES_DIR}, found {len(frame_files)}")

    first_im = Image.open(frame_files[0])
    native_w, native_h = first_im.size
    print(f"  Native 4K resolution: {native_w}x{native_h}")

    scaled_intrinsics = scale_camera_intrinsics(
        intrinsics=intrinsics,
        infer_res=(infer_w, infer_h),
        native_res=(native_w, native_h)
    )

    # 3. Backprojection & Geometric Filtering
    print("\n[Step 3/7] Backprojecting metric depth and applying geometric filtering...")
    d_min, d_max = 0.15, 6.0
    c_min = 1.2
    rtol = 0.10

    all_raw_pts = []
    all_raw_cols = []
    all_clean_pts = []
    all_clean_cols = []

    for i in range(num_cams):
        d_i = d_maps[i]
        c_i = c_maps[i]
        ext_i = extrinsics[i]
        K_i = intrinsics[i]

        im_native = cv2.imread(str(frame_files[i]))
        im_rgb = cv2.cvtColor(cv2.resize(im_native, (infer_w, infer_h)), cv2.COLOR_BGR2RGB)

        pts_i = depth_to_points(d_i, ext_i, K_i).reshape(infer_h, infer_w, 3)
        cols_i = im_rgb

        # Valid mask
        valid_i = (
            np.isfinite(pts_i).all(axis=-1)
            & (d_i >= d_min)
            & (d_i <= d_max)
            & (c_i >= c_min)
        )

        # Depth-jump filtering (sharp boundary preservation)
        jump_i = compute_relative_depth_jump(d_i, kernel_size=3)
        edge_clean_i = valid_i & (jump_i <= rtol)

        # Horizon quantile ceiling (99th percentile)
        valid_depths = d_i[valid_i]
        h_cap = float(np.percentile(valid_depths, 99.0)) if len(valid_depths) > 0 else 6.0
        horizon_mask = d_i <= h_cap

        clean_i = edge_clean_i & horizon_mask

        all_raw_pts.append(pts_i[valid_i])
        all_raw_cols.append(cols_i[valid_i])
        all_clean_pts.append(pts_i[clean_i])
        all_clean_cols.append(cols_i[clean_i])

    raw_pts_stacked = np.vstack(all_raw_pts)
    raw_cols_stacked = np.vstack(all_raw_cols)
    clean_pts_stacked = np.vstack(all_clean_pts)
    clean_cols_stacked = np.vstack(all_clean_cols)

    print(f"  Raw Backprojected Points: {len(raw_pts_stacked):,}")
    print(f"  Clean Filtered Points:    {len(clean_pts_stacked):,} ({len(clean_pts_stacked)/len(raw_pts_stacked)*100:.1f}%)")

    # 4. Point Cloud Consolidation & Voxel Grid Fusion
    print("\n[Step 4/7] Consolidating point clouds and executing voxel grid fusion...")
    extents = np.ptp(clean_pts_stacked, axis=0)
    max_extent = float(np.max(extents))
    voxel_size = max(max_extent / 1000.0, 0.005)
    print(f"  Spatial extents [X, Y, Z]: [{extents[0]:.2f}, {extents[1]:.2f}, {extents[2]:.2f}] model units")
    print(f"  Voxel grid resolution: {voxel_size:.4f} model units")

    fused_pts, fused_cols = voxel_grid_fusion(clean_pts_stacked, clean_cols_stacked, voxel_size=voxel_size)
    print(f"  Fused point cloud vertices: {len(fused_pts):,}")

    # Export point clouds
    pcd_unfused = o3d.geometry.PointCloud()
    pcd_unfused.points = o3d.utility.Vector3dVector(raw_pts_stacked.astype(np.float64))
    pcd_unfused.colors = o3d.utility.Vector3dVector(raw_cols_stacked.astype(np.float64) / 255.0)

    pcd_fused = o3d.geometry.PointCloud()
    pcd_fused.points = o3d.utility.Vector3dVector(fused_pts.astype(np.float64))
    pcd_fused.colors = o3d.utility.Vector3dVector(fused_cols.astype(np.float64) / 255.0)

    pcd_unfused_path = POINTCLOUD_DIR / "point_cloud_unfused.ply"
    pcd_fused_path = POINTCLOUD_DIR / "point_cloud_fused.ply"
    o3d.io.write_point_cloud(str(pcd_unfused_path), pcd_unfused)
    o3d.io.write_point_cloud(str(pcd_fused_path), pcd_fused)
    print(f"  Saved unfused point cloud: {pcd_unfused_path.name} ({pcd_unfused_path.stat().st_size / (1024**2):.2f} MB)")
    print(f"  Saved fused point cloud:   {pcd_fused_path.name} ({pcd_fused_path.stat().st_size / (1024**2):.2f} MB)")

    # 5. Surface Mesh Reconstruction & Repair
    print("\n[Step 5/7] Surface reconstruction (Screened Poisson HQ + Manifold Repair)...")
    cam_centers = []
    for i in range(num_cams):
        R = extrinsics[i, :3, :3]
        t = extrinsics[i, :3, 3]
        cam_centers.append(-R.T @ t)
    mean_cam_center = np.mean(cam_centers, axis=0)

    normal_radius = max(voxel_size * 2.5, 0.015)
    pcd_fused.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
    pcd_fused.orient_normals_towards_camera_location(camera_location=mean_cam_center)

    mesh_poisson, p_meta = SurfaceConnectivityMesher.build_poisson_hq(
        pcd_fused,
        depth=9,
        gentle_trim_quantile=0.005,
        support_distance_factor=4.0
    )
    print(f"  Raw Poisson mesh: {len(mesh_poisson.vertices):,} vertices, {len(mesh_poisson.triangles):,} triangles")

    mesh_repaired, rep_meta = SurfaceConnectivityMesher.repair_surface_connectivity(
        mesh_poisson,
        max_hole_vertices=32
    )
    print(f"  Repaired mesh:    {len(mesh_repaired.vertices):,} vertices, {len(mesh_repaired.triangles):,} triangles")

    repaired_ply = MESH_DIR / "surface_mesh_repaired.ply"
    repaired_obj = MESH_DIR / "surface_mesh_repaired.obj"
    o3d.io.write_triangle_mesh(str(repaired_ply), mesh_repaired)
    o3d.io.write_triangle_mesh(str(repaired_obj), mesh_repaired)

    # 6. Photographic Texture Mapping & Export
    print("\n[Step 6/7] Photographic texture mapping with 4K multi-view projections...")
    tex_engine = MultiViewTextureEngine(
        mesh=mesh_repaired,
        extrinsics=extrinsics,
        intrinsics_native=scaled_intrinsics,
        frame_paths=frame_files,
        depth_maps=d_maps,
        min_cos_theta=0.05
    )

    best_cam_per_face, q_scores, vis_stats = tex_engine.compute_face_visibility_and_scores()
    print(f"  Photographic Face Coverage: {vis_stats['photographic_coverage_pct']}% ({vis_stats['direct_photographic_faces']:,} / {vis_stats['total_faces']:,} faces)")

    # 6A: Multi-Material Uncompressed OBJ (1:1 native 4K frame mapping)
    multiview_obj_path = MESH_DIR / "multiview_textured_mesh.obj"
    tex_engine.export_multiview_obj(multiview_obj_path, best_cam_per_face)

    # 6B: Unified UV Texture Atlas OBJ + MTL + PNG
    atlas_obj_path = MESH_DIR / "textured_mesh.obj"
    atlas_obj_path, atlas_png_path, atlas_stats = tex_engine.export_packed_atlas_obj(
        output_obj_path=atlas_obj_path,
        best_cam_per_face=best_cam_per_face,
        atlas_size=4096
    )

    # 6C: Photographic Vertex-Colored PLY
    photo_ply_path = MESH_DIR / "surface_mesh_photographic.ply"
    tex_engine.export_photographic_vertex_ply(photo_ply_path, best_cam_per_face)

    # 6D: Binary GLB Export via Trimesh
    glb_path = MESH_DIR / "surface_mesh.glb"
    v_arr = np.asarray(mesh_repaired.vertices)
    f_arr = np.asarray(mesh_repaired.triangles)
    mesh_photo = o3d.io.read_triangle_mesh(str(photo_ply_path))
    c_arr = np.asarray(mesh_photo.vertex_colors)
    colors_u8 = (c_arr * 255.0).astype(np.uint8) if len(c_arr) == len(v_arr) else None

    tri_mesh = trimesh.Trimesh(
        vertices=v_arr,
        faces=f_arr,
        vertex_colors=colors_u8,
        process=False
    )
    glb_bytes = tri_mesh.export(file_type="glb")
    with open(glb_path, "wb") as f_glb:
        f_glb.write(glb_bytes)
    print(f"  - Saved Binary GLB: {glb_path.name} ({glb_path.stat().st_size / (1024**2):.2f} MB)")

    # 7. Rendering Previews & Quality Review
    print("\n[Step 7/7] Rendering standard views and generating review sheets...")
    views = render_scene_views(mesh_photo, MESH_DIR)
    for vname, vpath in views.items():
        print(f"  Rendered {vname}: {vpath.name}")

    review_sheet_path = REPORTS_DIR / "model_visual_review.png"
    render_4panel_visual_review(views, atlas_png_path, review_sheet_path)
    print(f"  Saved 4-panel visual review sheet: {review_sheet_path.name}")

    # Geometry Validation Check
    print("\nValidating exported geometry with Open3D & Trimesh...")
    tri_check = trimesh.load(str(glb_path))
    val_report = {
        "scene_name": "whatsapp_video_2026_10_24_campus_solar_complex",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_video": "dji_fly_20261024_173224_0168_1792844321671_video.mp4",
        "drone_model": "DJI Air 3S",
        "scale_status": "relative_scale_metric_unverified",
        "artifacts": {
            "unfused_pointcloud_ply": {
                "path": str(pcd_unfused_path.resolve()),
                "size_bytes": pcd_unfused_path.stat().st_size,
                "sha256": compute_file_sha256(pcd_unfused_path),
                "point_count": len(raw_pts_stacked)
            },
            "fused_pointcloud_ply": {
                "path": str(pcd_fused_path.resolve()),
                "size_bytes": pcd_fused_path.stat().st_size,
                "sha256": compute_file_sha256(pcd_fused_path),
                "point_count": len(fused_pts)
            },
            "textured_mesh_obj": {
                "path": str(atlas_obj_path.resolve()),
                "size_bytes": atlas_obj_path.stat().st_size,
                "sha256": compute_file_sha256(atlas_obj_path),
            },
            "textured_mesh_mtl": {
                "path": str(atlas_obj_path.with_suffix('.mtl').resolve()),
                "size_bytes": atlas_obj_path.with_suffix('.mtl').stat().st_size,
                "sha256": compute_file_sha256(atlas_obj_path.with_suffix('.mtl')),
            },
            "texture_atlas_png": {
                "path": str(atlas_png_path.resolve()),
                "size_bytes": atlas_png_path.stat().st_size,
                "sha256": compute_file_sha256(atlas_png_path),
                "resolution": [4096, 4096]
            },
            "surface_mesh_photographic_ply": {
                "path": str(photo_ply_path.resolve()),
                "size_bytes": photo_ply_path.stat().st_size,
                "sha256": compute_file_sha256(photo_ply_path),
            },
            "surface_mesh_glb": {
                "path": str(glb_path.resolve()),
                "size_bytes": glb_path.stat().st_size,
                "sha256": compute_file_sha256(glb_path),
            }
        },
        "geometry_metrics": {
            "vertex_count": len(v_arr),
            "triangle_count": len(f_arr),
            "extents_xyz": [round(float(extents[i]), 4) for i in range(3)],
            "bounding_box_min": [round(float(v_arr.min(axis=0)[i]), 4) for i in range(3)],
            "bounding_box_max": [round(float(v_arr.max(axis=0)[i]), 4) for i in range(3)],
            "is_watertight": bool(mesh_repaired.is_watertight()),
            "is_edge_manifold": bool(mesh_repaired.is_edge_manifold()),
            "is_vertex_manifold": bool(mesh_repaired.is_vertex_manifold()),
            "photographic_face_coverage_pct": vis_stats['photographic_coverage_pct'],
        },
        "execution_time_sec": round(time.time() - t_start, 2)
    }

    val_json_path = REPORTS_DIR / "geometric_validation.json"
    with open(val_json_path, "w", encoding="utf-8") as f:
        json.dump(val_report, f, indent=2)
    print(f"  Saved geometric validation: {val_json_path.name}")

    print("\n" + "=" * 80)
    print(f"RECONSTRUCTION COMPLETE in {val_report['execution_time_sec']}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
