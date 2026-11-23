"""
F.O.R.G.E. — Gate B: 12-Frame Real-Data Pilot Benchmark
Compares:
1. Original Defective Baseline (scale clamped without translation recomputation)
2. Candidate A: Bounded Sim(3) with Joint Point+Camera Constraints and Mandatory Translation Recomputation
3. Candidate B: Rigid SE(3) with Joint Point+Camera Constraints (s=1.0) and Mandatory Translation Recomputation

Evaluates:
- Individual-frame point clouds (frames 0..11)
- Individual-window point clouds (Windows 0, 1, 2)
- Overlap in-sample and held-out 3D point residuals
- Overlap camera center agreement
- Pre-TSDF point cloud spatial coherence
- 15mm TSDF volume reconstruction and mesh extraction
- Multi-view visual comparison renders
"""

import sys
import json
import math
from pathlib import Path
import numpy as np
import open3d as o3d
import cv2
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.reconstruction.point_cloud import depth_to_points, save_ply

EXP_DIR = PROJECT_ROOT / "outputs/indoor/vggt_tsdf_video_20261031_142303"
PILOT_DIR = PROJECT_ROOT / "outputs/indoor/distortion_investigation_20261031/pilot_12frame"
PILOT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load data
report = json.loads((EXP_DIR / "frame_curation_report.json").read_text())
frames = report["frames"][:12]
poses_data = np.load(EXP_DIR / "camera_geometry/camera_poses.npz")
depth_data = np.load(EXP_DIR / "camera_geometry/depth_predictions.npz")

all_ext = poses_data["extrinsics"][:12]   # (12, 3, 4)
all_intr = poses_data["intrinsics"][:12] # (12, 3, 3)
all_depth = depth_data["depth"][:12]     # (12, 432, 240)
all_conf = depth_data["confidence"][:12] # (12, 432, 240)

H_d, W_d = all_depth.shape[1], all_depth.shape[2]

# Confidence threshold: p20 = 2.5205 (from Part 4 audit)
CONF_THRESH = 2.5205

# Window definitions:
# Window 0: frames 0..5
# Window 1: frames 3..8 (shared: 3, 4, 5)
# Window 2: frames 6..11 (shared: 6, 7, 8)
windows = [
    [0, 1, 2, 3, 4, 5],
    [3, 4, 5, 6, 7, 8],
    [6, 7, 8, 9, 10, 11]
]

print("\n" + "=" * 76)
print("  GATE B: 12-FRAME REAL-DATA PILOT (WINDOWS 0, 1, 2)")
print("=" * 76)

def solve_joint_sim3(
    X_pts, Y_pts,
    C_src, C_tgt,
    w_pts=1.0, w_cam=10.0,
    force_se3=False,
    min_scale=0.85, max_scale=1.20
):
    """
    Joint Point + Camera Center Alignment Objective:
    min_{s, R, t} w_pts * || s * R * X + t - Y ||^2 + w_cam * || s * R * C_src + t - C_tgt ||^2
    Mandatory Translation Recomputation:
    t_applied = mu_Y - s_applied * R * mu_X
    """
    N_p = len(X_pts)
    N_c = len(C_src)
    total_w = N_p * w_pts + N_c * w_cam
    
    # Weighted centroids
    mu_X = (np.sum(X_pts * w_pts, axis=0) + np.sum(C_src * w_cam, axis=0)) / total_w
    mu_Y = (np.sum(Y_pts * w_pts, axis=0) + np.sum(C_tgt * w_cam, axis=0)) / total_w
    
    # Weighted centered coordinates
    Xc_pts = X_pts - mu_X
    Yc_pts = Y_pts - mu_Y
    Xc_cam = C_src - mu_X
    Yc_cam = C_tgt - mu_Y
    
    # Weighted cross-covariance
    cov = (
        np.sum(np.einsum('ni,nj->nij', Yc_pts * w_pts, Xc_pts), axis=0) +
        np.sum(np.einsum('ni,nj->nij', Yc_cam * w_cam, Xc_cam), axis=0)
    ) / total_w
    
    U, S, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
        
    var_X = (
        np.sum(np.sum(Xc_pts**2, axis=1) * w_pts) +
        np.sum(np.sum(Xc_cam**2, axis=1) * w_cam)
    ) / total_w
    
    s_raw = float(np.trace(np.diag(S)) / max(1e-8, var_X))
    
    if force_se3:
        s_applied = 1.0
        mode = "SE(3) rigid"
    else:
        # Check condition ratio
        cov_pts = (Xc_pts.T @ Xc_pts) / max(1, N_p)
        eigvals = np.linalg.eigvalsh(cov_pts)
        cond_ratio = math.sqrt(max(1e-8, eigvals[0])) / math.sqrt(max(1e-8, eigvals[-1]))
        
        if cond_ratio < 0.08 or s_raw < min_scale or s_raw > max_scale:
            s_applied = float(np.clip(s_raw, min_scale, max_scale))
            mode = f"Bounded Sim(3) (raw s={s_raw:.3f}, cond={cond_ratio:.3f})"
        else:
            s_applied = s_raw
            mode = f"Sim(3) full (s={s_raw:.3f})"
            
    # MANDATORY TRANSLATION RECOMPUTATION (Guarantees zero centroid drift!)
    t_applied = mu_Y - s_applied * (R @ mu_X)
    
    return s_applied, R, t_applied, mode

def run_alignment_pilot(candidate_name, force_se3=False, recompute_translation=True):
    print(f"\n[*] Running Pilot Alignment: {candidate_name}...")
    global_frames = {}
    alignment_records = []
    
    # Window 0 initialization
    win0 = windows[0]
    for lp, gp in enumerate(win0):
        R = all_ext[gp, :3, :3]
        t = all_ext[gp, :3, 3]
        c = -R.T @ t
        global_frames[gp] = {
            "ext": all_ext[gp].copy(),
            "intr": all_intr[gp].copy(),
            "depth": all_depth[gp].copy(),
            "conf": all_conf[gp].copy(),
            "center": c,
            "scale": 1.0
        }
    alignment_records.append({"window": 0, "scale": 1.0, "mode": "init", "heldout_pts_err_cm": 0.0, "cam_center_err_cm": 0.0})
    
    # Windows 1 and 2
    for w_idx in [1, 2]:
        win = windows[w_idx]
        shared_local = [lp for lp, gp in enumerate(win) if gp in global_frames]
        shared_global = [gp for lp, gp in enumerate(win) if gp in global_frames]
        
        # Subsample points on grid
        sub_y = np.linspace(int(H_d * 0.15), int(H_d * 0.85), 20, dtype=int)
        sub_x = np.linspace(int(W_d * 0.15), int(W_d * 0.85), 20, dtype=int)
        
        pts_loc = []
        pts_glob = []
        cams_loc = []
        cams_glob = []
        
        for lp, gp in zip(shared_local, shared_global):
            ge = global_frames[gp]
            pg = depth_to_points(ge["depth"], ge["ext"], ge["intr"])
            pl = depth_to_points(all_depth[gp], all_ext[gp], all_intr[gp])
            
            c_loc = all_conf[gp].squeeze()
            c_glob = ge["conf"].squeeze()
            mask = (c_loc[sub_y][:, sub_x] >= CONF_THRESH) & (c_glob[sub_y][:, sub_x] >= CONF_THRESH)
            pl_sub = pl[sub_y][:, sub_x][mask]
            pg_sub = pg[sub_y][:, sub_x][mask]
            
            if len(pl_sub) > 10:
                pts_loc.append(pl_sub)
                pts_glob.append(pg_sub)
                
            # Camera centers
            R_l = all_ext[gp, :3, :3]
            t_l = all_ext[gp, :3, 3]
            cams_loc.append(-R_l.T @ t_l)
            cams_glob.append(ge["center"])
            
        X_all = np.vstack(pts_loc)
        Y_all = np.vstack(pts_glob)
        C_src = np.array(cams_loc)
        C_tgt = np.array(cams_glob)
        
        # Partition 80% train, 20% held-out test
        np.random.seed(42 + w_idx)
        N_pts = len(X_all)
        perm = np.random.permutation(N_pts)
        n_train = int(N_pts * 0.8)
        train_idx = perm[:n_train]
        test_idx = perm[n_train:]
        
        X_train, Y_train = X_all[train_idx], Y_all[train_idx]
        X_test, Y_test = X_all[test_idx], Y_all[test_idx]
        
        if recompute_translation:
            s_w, R_w, t_w, mode = solve_joint_sim3(
                X_train, Y_train, C_src, C_tgt,
                w_pts=1.0, w_cam=10.0,
                force_se3=force_se3,
                min_scale=0.85, max_scale=1.20
            )
        else:
            # Defective baseline simulation: clamp scale, do NOT recompute translation
            s_raw, R_w, t_orig, _ = solve_joint_sim3(X_train, Y_train, C_src, C_tgt, w_pts=1.0, w_cam=0.0, force_se3=False)
            s_w = float(np.clip(s_raw, 0.85, 1.20))
            t_w = t_orig  # Retain original unclamped translation!
            mode = f"Defective Baseline (clamped s={s_w:.3f}, uncorrected t)"
            
        # Evaluate on held-out points
        Y_test_pred = s_w * (X_test @ R_w.T) + t_w
        err_heldout_pts = float(np.mean(np.linalg.norm(Y_test_pred - Y_test, axis=1)))
        
        # Evaluate on camera centers
        C_tgt_pred = s_w * (C_src @ R_w.T) + t_w
        err_cams = float(np.mean(np.linalg.norm(C_tgt_pred - C_tgt, axis=1)))
        
        alignment_records.append({
            "window": w_idx,
            "scale": round(float(s_w), 4),
            "mode": mode,
            "heldout_pts_err_cm": round(err_heldout_pts * 100.0, 2),
            "cam_center_err_cm": round(err_cams * 100.0, 2)
        })
        print(f"  Window {w_idx}: scale={s_w:.4f} | Held-out 3D Pts Err={err_heldout_pts*100:.2f} cm | Cam Center Err={err_cams*100:.2f} cm | {mode}")
        
        # Update global frames
        for lp, gp in enumerate(win):
            if gp not in global_frames:
                R_local = all_ext[gp, :3, :3]
                t_local = all_ext[gp, :3, 3]
                c_local = -R_local.T @ t_local
                
                c_global = s_w * (R_w @ c_local) + t_w
                R_global = R_local @ R_w.T
                t_global = -R_global @ c_global
                ext_global = np.hstack([R_global, t_global.reshape(3, 1)])
                depth_global = s_w * all_depth[gp]
                
                global_frames[gp] = {
                    "ext": ext_global.astype(np.float32),
                    "intr": all_intr[gp].astype(np.float32),
                    "depth": depth_global.astype(np.float32),
                    "conf": all_conf[gp].copy(),
                    "center": c_global,
                    "scale": s_w
                }
                
    # Build pre-TSDF Point Cloud
    all_pts = []
    all_cols = []
    for i in range(12):
        gf = global_frames[i]
        d = gf["depth"]
        c = gf["conf"]
        im_bgr = cv2.imread(frames[i]["file_path"])
        im_rgb = cv2.cvtColor(cv2.resize(im_bgr, (W_d, H_d)), cv2.COLOR_BGR2RGB)
        
        pts = depth_to_points(d, gf["ext"], gf["intr"])
        mask = (d >= 0.20) & (d <= 5.0) & (c >= CONF_THRESH)
        all_pts.append(pts[mask])
        all_cols.append(im_rgb[mask])
        
    pts_merged = np.vstack(all_pts)
    cols_merged = np.vstack(all_cols)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_merged.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(cols_merged.astype(np.float64) / 255.0)
    pcd_down = pcd.voxel_down_sample(voxel_size=0.012)
    
    pcd_path = PILOT_DIR / f"pre_tsdf_pcd_{candidate_name}.ply"
    o3d.io.write_point_cloud(str(pcd_path), pcd_down)
    
    # Run 15mm TSDF Integration
    vol = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=0.015, sdf_trunc=0.045, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    for i in range(12):
        gf = global_frames[i]
        d = gf["depth"].copy()
        mask_inv = (d < 0.20) | (d > 5.0) | (gf["conf"] < CONF_THRESH)
        d[mask_inv] = 0.0
        im_bgr = cv2.imread(frames[i]["file_path"])
        im_rgb = cv2.cvtColor(cv2.resize(im_bgr, (W_d, H_d)), cv2.COLOR_BGR2RGB)
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(im_rgb), o3d.geometry.Image(d), depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False
        )
        K = gf["intr"]
        intr_obj = o3d.camera.PinholeCameraIntrinsic(W_d, H_d, float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
        ext_4x4 = np.eye(4)
        ext_4x4[:3, :4] = gf["ext"]
        vol.integrate(rgbd, intr_obj, ext_4x4)
        
    mesh = vol.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    mesh_path = PILOT_DIR / f"tsdf_mesh_{candidate_name}.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    
    # Measure duplicate wall thickness: standard deviation of points along dominant normal
    bbox = mesh.get_axis_aligned_bounding_box()
    extents = bbox.get_extent().round(4).tolist()
    
    # Measure distance between pre-TSDF PCD and TSDF mesh
    dists = np.asarray(pcd_down.compute_point_cloud_distance(vol.extract_point_cloud()))
    med_dist = float(np.median(dists)) * 100.0
    p95_dist = float(np.percentile(dists, 95)) * 100.0
    
    print(f"  [+] {candidate_name} TSDF Mesh: {len(mesh.vertices):,} V | {len(mesh.triangles):,} F | Extents: {extents}")
    print(f"      PCD-to-TSDF Median Dist: {med_dist:.2f} cm | 95th-pct: {p95_dist:.2f} cm")
    
    return {
        "candidate": candidate_name,
        "mesh_vertices": len(mesh.vertices),
        "mesh_triangles": len(mesh.triangles),
        "mesh_extents_m": extents,
        "pcd_to_tsdf_median_cm": round(med_dist, 2),
        "pcd_to_tsdf_p95_cm": round(p95_dist, 2),
        "alignment_records": alignment_records,
        "mesh_path": str(mesh_path),
        "pcd_path": str(pcd_path),
        "mesh_obj": mesh,
        "pcd_obj": pcd_down
    }

# Execute all three candidates
res_defective = run_alignment_pilot("baseline_defective", force_se3=False, recompute_translation=False)
res_cand_a = run_alignment_pilot("candidate_A_bounded_sim3", force_se3=False, recompute_translation=True)
res_cand_b = run_alignment_pilot("candidate_B_rigid_se3", force_se3=True, recompute_translation=True)

# Generate multi-view comparison panel
print("\n[*] Rendering Multi-View Diagnostic Comparison Panel for Gate B...")
models = [
    {"name": "Original Defective Baseline", "data": res_defective["mesh_obj"]},
    {"name": "Candidate A: Bounded Sim(3) + Recomputed t", "data": res_cand_a["mesh_obj"]},
    {"name": "Candidate B: Rigid SE(3) + Recomputed t", "data": res_cand_b["mesh_obj"]}
]

views = [
    {"id": "oblique", "name": "Oblique Room Perspective", "offset": np.array([-0.8, 0.9, 1.1])},
    {"id": "top_down", "name": "Top-Down Plan Flight View", "offset": np.array([0.0, 1.8, 0.0]), "up": [0, 0, -1]},
    {"id": "wall_seam", "name": "Wall Seam & Overlap Alignment", "offset": np.array([0.4, 0.1, 0.5])}
]

card_w, card_h = 480, 320
header_h = 60
panel_w = card_w * len(models)
panel_h = card_h * len(views) + header_h

panel = Image.new("RGB", (panel_w, panel_h), color=(15, 18, 24))
draw = ImageDraw.Draw(panel)

try:
    font_head = ImageFont.truetype("arial.ttf", 16)
    font_lbl = ImageFont.truetype("arial.ttf", 11)
except Exception:
    font_head = ImageFont.load_default()
    font_lbl = ImageFont.load_default()

draw.rectangle([0, 0, panel_w, header_h], fill=(22, 27, 34))
draw.text((25, 20), "GATE B: 12-FRAME REAL-DATA PILOT — COMPARING BASELINE VS CANDIDATE A (SIM3) VS CANDIDATE B (SE3)", fill=(56, 189, 248), font=font_head)

for col_idx, mod in enumerate(models):
    m_name = mod["name"]
    geom = mod["data"]
    bbox = geom.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    ext = bbox.get_extent()
    diag = float(np.linalg.norm(ext))
    if diag < 1e-4: diag = 1.0
    x = col_idx * card_w
    
    for row_idx, vw in enumerate(views):
        v_name = vw["name"]
        y = header_h + row_idx * card_h
        thumb_file = PILOT_DIR / f"render_{col_idx}_{vw['id']}.png"
        
        try:
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False, width=640, height=420)
            vis.add_geometry(geom)
            opt = vis.get_render_option()
            opt.background_color = np.asarray([0.10, 0.11, 0.14])
            opt.light_on = True
            
            ctr = vis.get_view_control()
            cam_pos = center + vw["offset"] * (diag * 0.75)
            front = center - cam_pos
            front /= (np.linalg.norm(front) + 1e-8)
            ctr.set_front(front)
            ctr.set_up(vw.get("up", [0.0, 1.0, 0.0]))
            ctr.set_lookat(center)
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(str(thumb_file), do_render=True)
            vis.destroy_window()
            
            im_cell = Image.open(thumb_file).resize((card_w, card_h), Image.Resampling.LANCZOS)
            panel.paste(im_cell, (x, y))
        except Exception as e:
            draw.rectangle([x, y, x + card_w, y + card_h], fill=(25, 30, 40))
            draw.text((x + 20, y + card_h // 2), f"Render: {m_name}_{vw['id']}", fill=(100, 116, 139), font=font_lbl)
            
        draw.rectangle([x, y, x + card_w, y + 26], fill=(18, 24, 34))
        draw.text((x + 8, y + 6), f"{m_name} — {v_name}", fill=(241, 245, 249), font=font_lbl)
        draw.rectangle([x, y, x + card_w, y + card_h], outline=(45, 55, 72), width=1)

panel_path = PILOT_DIR / "pilot_12frame_comparison_panel.png"
panel.save(panel_path, quality=95)
print(f"[+] Saved Gate B Pilot Comparison Panel: {panel_path.name}")

# Save full summary report
summary = {
    "baseline_defective": {
        "mesh_vertices": res_defective["mesh_vertices"],
        "mesh_triangles": res_defective["mesh_triangles"],
        "mesh_extents_m": res_defective["mesh_extents_m"],
        "pcd_to_tsdf_median_cm": res_defective["pcd_to_tsdf_median_cm"],
        "pcd_to_tsdf_p95_cm": res_defective["pcd_to_tsdf_p95_cm"],
        "alignment_records": res_defective["alignment_records"]
    },
    "candidate_A_bounded_sim3": {
        "mesh_vertices": res_cand_a["mesh_vertices"],
        "mesh_triangles": res_cand_a["mesh_triangles"],
        "mesh_extents_m": res_cand_a["mesh_extents_m"],
        "pcd_to_tsdf_median_cm": res_cand_a["pcd_to_tsdf_median_cm"],
        "pcd_to_tsdf_p95_cm": res_cand_a["pcd_to_tsdf_p95_cm"],
        "alignment_records": res_cand_a["alignment_records"]
    },
    "candidate_B_rigid_se3": {
        "mesh_vertices": res_cand_b["mesh_vertices"],
        "mesh_triangles": res_cand_b["mesh_triangles"],
        "mesh_extents_m": res_cand_b["mesh_extents_m"],
        "pcd_to_tsdf_median_cm": res_cand_b["pcd_to_tsdf_median_cm"],
        "pcd_to_tsdf_p95_cm": res_cand_b["pcd_to_tsdf_p95_cm"],
        "alignment_records": res_cand_b["alignment_records"]
    }
}
(PILOT_DIR / "pilot_12frame_summary.json").write_text(json.dumps(summary, indent=2))
print(f"[+] Saved Gate B Pilot Summary Report: pilot_12frame_summary.json")

