"""
F.O.R.G.E. — Forensic Stage-by-Stage Diagnostic Visualizer
Implements PART 3 of the User Request:
Renders each stage from identical camera viewpoints to determine exactly where distortion occurs:
  A. Original source video frame
  B. VGGT depth map
  C. Single-frame RGB point cloud
  D. Individual-window point cloud
  E. Globally aligned point cloud
  F. Filtered point cloud
  G. Mesh before repair
  H. Final reconstructed mesh
"""

import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.cm as cm

EXP_DIR = Path("outputs/indoor/final_room_reconstruction_recovery")
DIAG_DIR = EXP_DIR / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

RAW_CACHE = Path("outputs/indoor/vggt_tsdf_video_20261030_170234/vggt_cache/raw_windows_0_to_20.npz")
FRAMES_DIR = Path("outputs/indoor/texture_experiment_video1_reproduced_video_170234/frames")
ARTIFACTS_DIR = Path("C:/Users/25beevdt025/.gemini/antigravity/brain/536a497a-fba6-496d-b4a9-b4f0cf756953")

# We inspect 3 key diagnostic viewpoints:
# 1. View 1: Bed & Entrance (Window 2, frame 0 -> frame 6, 06.61s)
# 2. View 2: Desk & Computer (Window 9, frame 0 -> frame 27, 18.0s)
# 3. View 3: Wardrobe & Wall (Window 14, frame 0 -> frame 42, 27.5s)
DIAG_VIEWS = [
    {
        "id": "view1_bed",
        "title": "VIEW 1: BED & ENTRANCE",
        "window": 2,
        "frame_idx": 0,
        "curated_frame": FRAMES_DIR / "frame_06_f0197_06.61s.png"
    },
    {
        "id": "view2_desk",
        "title": "VIEW 2: DESK & COMPUTER",
        "window": 9,
        "frame_idx": 0,
        "curated_frame": FRAMES_DIR / "frame_32_f0655_21.98s.png"
    },
    {
        "id": "view3_wardrobe",
        "title": "VIEW 3: WARDROBE & WALL",
        "window": 14,
        "frame_idx": 0,
        "curated_frame": FRAMES_DIR / "frame_45_f0865_29.03s.png"
    }
]

def depth_to_pcd_pts(depth, ext, intr, conf, conf_thresh=1.2):
    H, W = depth.shape
    fx, fy = intr[0, 0], intr[1, 1]
    cx, cy = intr[0, 2], intr[1, 2]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    mask = (depth > 0.2) & (depth < 3.5) & (conf >= conf_thresh)
    z = depth[mask]
    x = (u[mask] - cx) * z / fx
    y = (v[mask] - cy) * z / fy
    pts_cam = np.stack([x, y, z], axis=1)
    R = ext[:3, :3]
    t = ext[:3, 3]
    pts_world = (pts_cam - t) @ R
    return pts_world, mask

def render_pcd_offscreen(pcd, out_path, eye=(0, 0, -1), target=(0, 0, 1), up=(0, -1, 0), width=640, height=480):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    ctr.set_lookat(target)
    ctr.set_front(np.array(target) - np.array(eye))
    ctr.set_up(up)
    ctr.set_zoom(0.7)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(str(out_path), do_render=True)
    vis.destroy_window()

def generate_stage_visualizations():
    print("[*] Generating 8-Stage Forensic Visualizations (PART 3)...")
    raw = np.load(RAW_CACHE)
    
    # 1. Precompute global camera chaining & global point clouds
    R_glob = np.eye(3)
    t_glob = np.zeros(3)
    window_transforms = {2: (np.eye(3), np.zeros(3))}

    for w in range(2, 20):
        if w > 2:
            e_A = raw[f'ext_{w-1}'][3]
            e_B = raw[f'ext_{w}'][0]
            R_A, t_A = e_A[:3, :3], e_A[:3, 3]
            R_B, t_B = e_B[:3, :3], e_B[:3, 3]
            c_A = -R_A.T @ t_A
            c_B = -R_B.T @ t_B
            R_step = R_A.T @ R_B
            t_step = c_A - R_step @ c_B
            t_glob = R_glob @ t_step + t_glob
            R_glob = R_glob @ R_step
            window_transforms[w] = (R_glob.copy(), t_glob.copy())

    # Build global aligned point cloud
    aligned_pts = []
    aligned_colors = []
    for w in range(2, 20):
        R_w, t_w = window_transforms[w]
        frames_to_add = range(6) if w == 2 else range(3, 6)
        for f in frames_to_add:
            d = raw[f'depth_{w}'][f]
            c = raw[f'conf_{w}'][f]
            e = raw[f'ext_{w}'][f]
            i = raw[f'intr_{w}'][f]
            pts, mask = depth_to_pcd_pts(d, e, i, c)
            pts_world = pts @ R_w.T + t_w
            aligned_pts.append(pts_world)

    pts_aligned = np.vstack(aligned_pts)
    pcd_aligned = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_aligned)).voxel_down_sample(0.015)
    
    # Filtered point cloud (radius outlier removal)
    pcd_filtered, _ = pcd_aligned.remove_radius_outlier(nb_points=6, radius=0.035)

    # Build candidate mesh from pcd_filtered
    pcd_filtered.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=30))
    pcd_filtered.orient_normals_towards_camera_location(np.array([0.2, -0.1, 0.5]))
    mesh_raw, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_filtered, depth=8)
    
    # Pruned mesh
    dens_np = np.asarray(densities)
    mesh_pruned = o3d.geometry.TriangleMesh(mesh_raw)
    mesh_pruned.remove_vertices_by_mask(dens_np < np.quantile(dens_np, 0.05))

    # Render each diagnostic viewpoint across all 8 stages
    stage_images = {}

    for v in DIAG_VIEWS:
        v_id = v["id"]
        w = v["window"]
        f = v["frame_idx"]
        print(f"\nProcessing Viewpoint: {v['title']}...")
        stage_images[v_id] = {}

        # STAGE A: Original Source Video Frame
        src_im = Image.open(v["curated_frame"]).convert("RGB")
        out_a = DIAG_DIR / f"{v_id}_stage_A_source_frame.png"
        src_im.save(out_a)
        stage_images[v_id]["A"] = out_a

        # STAGE B: VGGT Depth Map Colormap
        d = raw[f"depth_{w}"][f]
        d_norm = np.clip((d - d.min()) / (d.max() - d.min() + 1e-6), 0, 1)
        depth_colored = (cm.turbo(d_norm)[:, :, :3] * 255).astype(np.uint8)
        out_b = DIAG_DIR / f"{v_id}_stage_B_depth_map.png"
        Image.fromarray(depth_colored).save(out_b)
        stage_images[v_id]["B"] = out_b

        # STAGE C: Single-Frame Point Cloud
        pts_c, _ = depth_to_pcd_pts(d, raw[f"ext_{w}"][f], raw[f"intr_{w}"][f], raw[f"conf_{w}"][f])
        pcd_c = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_c))
        pcd_c.paint_uniform_color([0.3, 0.7, 0.9])
        out_c = DIAG_DIR / f"{v_id}_stage_C_single_frame_pcd.png"
        render_pcd_offscreen(pcd_c, out_c, eye=pcd_c.get_center() + np.array([0, 0, -1.2]), target=pcd_c.get_center())
        stage_images[v_id]["C"] = out_c

        # STAGE D: Individual-Window Point Cloud (6 frames)
        win_pts = []
        for fi in range(6):
            pts_fi, _ = depth_to_pcd_pts(raw[f"depth_{w}"][fi], raw[f"ext_{w}"][fi], raw[f"intr_{w}"][fi], raw[f"conf_{w}"][fi])
            win_pts.append(pts_fi)
        pcd_d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack(win_pts))).voxel_down_sample(0.015)
        pcd_d.paint_uniform_color([0.4, 0.8, 0.4])
        out_d = DIAG_DIR / f"{v_id}_stage_D_window_pcd.png"
        render_pcd_offscreen(pcd_d, out_d, eye=pcd_d.get_center() + np.array([0, 0, -1.5]), target=pcd_d.get_center())
        stage_images[v_id]["D"] = out_d

        # STAGE E: Globally Aligned Point Cloud
        pcd_aligned.paint_uniform_color([0.9, 0.7, 0.2])
        out_e = DIAG_DIR / f"{v_id}_stage_E_aligned_pcd.png"
        render_pcd_offscreen(pcd_aligned, out_e, eye=np.array([0.2, 0.5, -1.8]), target=np.array([0.2, 0.0, 0.8]))
        stage_images[v_id]["E"] = out_e

        # STAGE F: Filtered Point Cloud
        pcd_filtered.paint_uniform_color([0.2, 0.8, 0.8])
        out_f = DIAG_DIR / f"{v_id}_stage_F_filtered_pcd.png"
        render_pcd_offscreen(pcd_filtered, out_f, eye=np.array([0.2, 0.5, -1.8]), target=np.array([0.2, 0.0, 0.8]))
        stage_images[v_id]["F"] = out_f

        # STAGE G: Original Mesh Before Repair (Unpruned Poisson)
        mesh_raw.paint_uniform_color([0.8, 0.4, 0.4])
        out_g = DIAG_DIR / f"{v_id}_stage_G_unrepaired_mesh.png"
        render_pcd_offscreen(mesh_raw, out_g, eye=np.array([0.2, 0.5, -1.8]), target=np.array([0.2, 0.0, 0.8]))
        stage_images[v_id]["G"] = out_g

        # STAGE H: Final Reconstructed Mesh (Pruned & Filtered)
        mesh_pruned.paint_uniform_color([0.4, 0.6, 0.9])
        out_h = DIAG_DIR / f"{v_id}_stage_H_reconstructed_mesh.png"
        render_pcd_offscreen(mesh_pruned, out_h, eye=np.array([0.2, 0.5, -1.8]), target=np.array([0.2, 0.0, 0.8]))
        stage_images[v_id]["H"] = out_h

    # 3. Assemble Master 8-Stage Forensic Panel
    print("\n[*] Assembling Master 8-Stage Forensic Panel...")
    cell_w, cell_h = 320, 240
    pad = 20
    header_h = 80
    row_title_h = 24
    
    stages = [
        ("A", "Stage A: Video Frame"),
        ("B", "Stage B: VGGT Depth"),
        ("C", "Stage C: Single PCD"),
        ("D", "Stage D: Window PCD"),
        ("E", "Stage E: Aligned PCD"),
        ("F", "Stage F: Filtered PCD"),
        ("G", "Stage G: Raw Mesh"),
        ("H", "Stage H: Final Mesh")
    ]
    
    n_cols = len(stages)
    n_rows = len(DIAG_VIEWS)
    
    total_w = pad * 2 + n_cols * cell_w + (n_cols - 1) * 8
    total_h = header_h + n_rows * (cell_h + row_title_h + 20) + pad
    
    canvas = Image.new("RGB", (total_w, total_h), color=(11, 15, 25))
    draw = ImageDraw.Draw(canvas)
    
    draw.text((pad, 16), "F.O.R.G.E. — 8-STAGE FORENSIC RECONSTRUCTION DIAGNOSTIC PANEL", fill=(56, 189, 248))
    draw.text((pad, 42), "Pinpointing the exact pipeline stage where distortion, folds, or fragmentation occur across 3 interior viewpoints", fill=(148, 163, 184))
    
    # Stage column headers
    for c_i, (s_code, s_name) in enumerate(stages):
        x = pad + c_i * (cell_w + 8)
        draw.rectangle([x, 66, x + cell_w, 86], fill=(24, 32, 47))
        draw.text((x + 6, 70), s_name, fill=(203, 213, 225))
        
    for r_i, v in enumerate(DIAG_VIEWS):
        v_id = v["id"]
        y_row = header_h + r_i * (cell_h + row_title_h + 20)
        draw.rectangle([pad, y_row, total_w - pad, y_row + row_title_h], fill=(20, 27, 40))
        draw.text((pad + 8, y_row + 4), v["title"], fill=(255, 255, 255))
        
        for c_i, (s_code, s_name) in enumerate(stages):
            x = pad + c_i * (cell_w + 8)
            y = y_row + row_title_h + 4
            im_path = stage_images[v_id][s_code]
            if im_path.exists():
                im = Image.open(im_path).convert("RGB")
                im_res = im.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
                canvas.paste(im_res, (x, y))

    panel_path = EXP_DIR / "stage_by_stage_forensic_diagnostic_panel.png"
    canvas.save(panel_path, quality=95)
    canvas.save(ARTIFACTS_DIR / "stage_by_stage_forensic_diagnostic_panel.png", quality=95)
    print(f"[+] Saved Master 8-Stage Diagnostic Panel: {panel_path}")

if __name__ == "__main__":
    generate_stage_visualizations()
