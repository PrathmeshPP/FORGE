"""
F.O.R.G.E. — Gate B: Real-Data Pilot on Windows 0..4 (Frames 0..17).
Directly tests the transition at Window 4 where planar door degeneracy occurred in the experiment:
- Captures raw local window VGGT predictions.
- Compares:
  1. Defective Baseline (scale clamped to 0.85 without recomputing t_w)
  2. Candidate A: Bounded Sim(3) with joint point+camera constraints and mandatory translation recomputation
  3. Candidate B: Rigid SE(3) (s=1.0) with joint point+camera constraints and mandatory translation recomputation
- Evaluates held-out residuals, camera center residuals, and TSDF surface alignment.
- Renders multi-view comparison panel.
"""

import sys
import json
import math
from pathlib import Path
import numpy as np
import torch
import open3d as o3d
import cv2
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

for _candidate_vggt in [
    r"C:\Users\25beevdt025\Desktop\SIH\third_party\vggt_omega",
    r"C:\Users\25beevdt025\Desktop\coding\third_party\vggt_omega",
]:
    if Path(_candidate_vggt).is_dir() and _candidate_vggt not in sys.path:
        sys.path.insert(0, _candidate_vggt)

from src.vggt.inference import load_vggt_model, DEFAULT_CHECKPOINT_PATH
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera
from src.reconstruction.point_cloud import depth_to_points

EXP_DIR = PROJECT_ROOT / "outputs/indoor/vggt_tsdf_video_20261031_142303"
PILOT_DIR = PROJECT_ROOT / "outputs/indoor/distortion_investigation_20261031/pilot_12frame"
PILOT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = PILOT_DIR / "vggt_local_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

report = json.loads((EXP_DIR / "frame_curation_report.json").read_text())
frame_records = report["frames"][:18]
image_paths = [r["file_path"] for r in frame_records]

# 5 overlapping windows of 6 frames (step 3):
# Win 0: [0..5]
# Win 1: [3..8]
# Win 2: [6..11]
# Win 3: [9..14]
# Win 4: [12..17] -> Critical window where planar door degeneracy occurred!
windows = [
    list(range(0, 6)),
    list(range(3, 9)),
    list(range(6, 12)),
    list(range(9, 15)),
    list(range(12, 18))
]

CONF_THRESH = 2.5205  # p20 threshold

# 1. Obtain raw local predictions for all 5 windows
raw_windows_data = []
cache_file = CACHE_DIR / "raw_windows_0_to_4.npz"

if cache_file.exists():
    print(f"[*] Loading cached raw VGGT local windows from {cache_file.name}...")
    cached = np.load(cache_file)
    for w_idx in range(len(windows)):
        raw_windows_data.append({
            "ext": cached[f"ext_{w_idx}"],
            "intr": cached[f"intr_{w_idx}"],
            "depth": cached[f"depth_{w_idx}"],
            "conf": cached[f"conf_{w_idx}"]
        })
else:
    print("[*] Running VGGT-Omega on Windows 0..4 (RTX 2070 SUPER)...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = load_vggt_model(DEFAULT_CHECKPOINT_PATH, device)
    
    npz_dict = {}
    for w_idx, win in enumerate(windows):
        win_paths = [image_paths[i] for i in win]
        inp = load_and_preprocess_images(win_paths, image_resolution=320).to(device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            out = model(inp)
            ext, intr = encoding_to_camera(out.get("pose_enc"), image_size_hw=inp.shape[-2:])
        
        ext_np = ext.squeeze(0).cpu().numpy()
        intr_np = intr.squeeze(0).cpu().numpy()
        depth_np = out["depth"].squeeze(0).cpu().numpy()
        if depth_np.ndim == 4:
            depth_np = depth_np.squeeze(-1)
        conf_np = out["depth_conf"].squeeze(0).cpu().numpy()
        if conf_np.ndim == 4:
            conf_np = conf_np.squeeze(-1)
            
        data_w = {"ext": ext_np, "intr": intr_np, "depth": depth_np, "conf": conf_np}
        raw_windows_data.append(data_w)
        npz_dict[f"ext_{w_idx}"] = ext_np
        npz_dict[f"intr_{w_idx}"] = intr_np
        npz_dict[f"depth_{w_idx}"] = depth_np
        npz_dict[f"conf_{w_idx}"] = conf_np
        print(f"  Captured Window {w_idx} (frames {win[0]:02d}:{win[-1]:02d})")
        
    np.savez_compressed(cache_file, **npz_dict)
    print(f"[+] Cached 5 raw local windows to {cache_file.name}")

H_d, W_d = raw_windows_data[0]["depth"].shape[1], raw_windows_data[0]["depth"].shape[2]

def solve_alignment(X_pts, Y_pts, C_src, C_tgt, mode_type="bounded_sim3", w_pts=1.0, w_cam=10.0):
    N_p = len(X_pts)
    N_c = len(C_src)
    total_w = N_p * w_pts + N_c * w_cam
    
    mu_X = (np.sum(X_pts * w_pts, axis=0) + np.sum(C_src * w_cam, axis=0)) / total_w
    mu_Y = (np.sum(Y_pts * w_pts, axis=0) + np.sum(C_tgt * w_cam, axis=0)) / total_w
    
    Xc_pts = X_pts - mu_X
    Yc_pts = Y_pts - mu_Y
    Xc_cam = C_src - mu_X
    Yc_cam = C_tgt - mu_Y
    
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
    
    cov_pts = (Xc_pts.T @ Xc_pts) / max(1, N_p)
    eigvals = np.linalg.eigvalsh(cov_pts)
    cond_ratio = math.sqrt(max(1e-8, eigvals[0])) / math.sqrt(max(1e-8, eigvals[-1]))
    
    if mode_type == "defective_baseline":
        # Clamped scale, but retains unclamped translation!
        s_applied = float(np.clip(s_raw, 0.85, 1.20))
        t_unclamped = mu_Y - s_raw * (R @ mu_X)
        t_applied = t_unclamped  # DEFECT: does not recompute!
        log_mode = f"Defective (orig s={s_raw:.3f}, clamped={s_applied:.3f}, uncorrected t)"
    elif mode_type == "bounded_sim3":
        if cond_ratio < 0.08 or s_raw < 0.85 or s_raw > 1.20:
            s_applied = float(np.clip(s_raw, 0.85, 1.20))
            log_mode = f"Bounded Sim(3) (raw s={s_raw:.3f}, clamped={s_applied:.3f})"
        else:
            s_applied = s_raw
            log_mode = f"Sim(3) full (s={s_raw:.3f})"
        # MANDATORY RECOMPUTATION
        t_applied = mu_Y - s_applied * (R @ mu_X)
    elif mode_type == "rigid_se3":
        s_applied = 1.0
        log_mode = f"Rigid SE(3) (s=1.000, raw was {s_raw:.3f})"
        # MANDATORY RECOMPUTATION
        t_applied = mu_Y - 1.0 * (R @ mu_X)
        
    return s_applied, R, t_applied, log_mode

def run_pilot_experiment(candidate_key, mode_type, w_cam_weight=10.0):
    print(f"\n" + "-" * 76)
    print(f"  RUNNING PILOT: {candidate_key} (mode={mode_type}, w_cam={w_cam_weight})")
    print("-" * 76)
    
    global_frames = {}
    alignment_logs = []
    
    # Init Window 0
    w0_data = raw_windows_data[0]
    for lp, gp in enumerate(windows[0]):
        R = w0_data["ext"][lp, :3, :3]
        t = w0_data["ext"][lp, :3, 3]
        c = -R.T @ t
        global_frames[gp] = {
            "ext": w0_data["ext"][lp].copy(),
            "intr": w0_data["intr"][lp].copy(),
            "depth": w0_data["depth"][lp].copy(),
            "conf": w0_data["conf"][lp].copy(),
            "center": c,
            "scale": 1.0
        }
    alignment_logs.append({"window": 0, "scale": 1.0, "mode": "init", "heldout_err_cm": 0.0, "cam_err_cm": 0.0})
    
    for w_idx in range(1, len(windows)):
        win = windows[w_idx]
        w_data = raw_windows_data[w_idx]
        
        shared_local = [lp for lp, gp in enumerate(win) if gp in global_frames]
        shared_global = [gp for lp, gp in enumerate(win) if gp in global_frames]
        
        sub_y = np.linspace(int(H_d * 0.15), int(H_d * 0.85), 20, dtype=int)
        sub_x = np.linspace(int(W_d * 0.15), int(W_d * 0.85), 20, dtype=int)
        
        pts_loc = []
        pts_glob = []
        cams_loc = []
        cams_glob = []
        
        for lp, gp in zip(shared_local, shared_global):
            ge = global_frames[gp]
            pg = depth_to_points(ge["depth"], ge["ext"], ge["intr"])
            pl = depth_to_points(w_data["depth"][lp], w_data["ext"][lp], w_data["intr"][lp])
            
            c_loc = w_data["conf"][lp]
            c_glob = ge["conf"]
            mask = (c_loc[sub_y][:, sub_x] >= CONF_THRESH) & (c_glob[sub_y][:, sub_x] >= CONF_THRESH)
            pl_sub = pl[sub_y][:, sub_x][mask]
            pg_sub = pg[sub_y][:, sub_x][mask]
            
            if len(pl_sub) > 10:
                pts_loc.append(pl_sub)
                pts_glob.append(pg_sub)
                
            R_l = w_data["ext"][lp, :3, :3]
            t_l = w_data["ext"][lp, :3, 3]
            cams_loc.append(-R_l.T @ t_l)
            cams_glob.append(ge["center"])
            
        X_all = np.vstack(pts_loc)
        Y_all = np.vstack(pts_glob)
        C_src = np.array(cams_loc)
        C_tgt = np.array(cams_glob)
        
        # 80/20 train/test split
        np.random.seed(42 + w_idx)
        N_pts = len(X_all)
        perm = np.random.permutation(N_pts)
        n_train = int(N_pts * 0.8)
        X_train, Y_train = X_all[perm[:n_train]], Y_all[perm[:n_train]]
        X_test, Y_test = X_all[perm[n_train:]], Y_all[perm[n_train:]]
        
        s_w, R_w, t_w, log_mode = solve_alignment(
            X_train, Y_train, C_src, C_tgt,
            mode_type=mode_type,
            w_pts=1.0, w_cam=w_cam_weight
        )
        
        # Evaluate on held-out test points
        Y_test_pred = s_w * (X_test @ R_w.T) + t_w
        err_heldout = float(np.mean(np.linalg.norm(Y_test_pred - Y_test, axis=1)))
        
        # Evaluate on camera centers
        C_tgt_pred = s_w * (C_src @ R_w.T) + t_w
        err_cam = float(np.mean(np.linalg.norm(C_tgt_pred - C_tgt, axis=1)))
        
        alignment_logs.append({
            "window": w_idx,
            "scale": round(float(s_w), 4),
            "mode": log_mode,
            "heldout_err_cm": round(err_heldout * 100.0, 2),
            "cam_err_cm": round(err_cam * 100.0, 2)
        })
        print(f"  Window {w_idx}: scale={s_w:.4f} | Held-out 3D Pts Err={err_heldout*100:.2f} cm | Cam Center Err={err_cam*100:.2f} cm | {log_mode}")
        
        for lp, gp in enumerate(win):
            if gp not in global_frames:
                R_local = w_data["ext"][lp, :3, :3]
                t_local = w_data["ext"][lp, :3, 3]
                c_local = -R_local.T @ t_local
                
                c_global = s_w * (R_w @ c_local) + t_w
                R_global = R_local @ R_w.T
                t_global = -R_global @ c_global
                ext_global = np.hstack([R_global, t_global.reshape(3, 1)])
                depth_global = s_w * w_data["depth"][lp]
                
                global_frames[gp] = {
                    "ext": ext_global.astype(np.float32),
                    "intr": w_data["intr"][lp].astype(np.float32),
                    "depth": depth_global.astype(np.float32),
                    "conf": w_data["conf"][lp].copy(),
                    "center": c_global,
                    "scale": s_w
                }
                
    # Build pre-TSDF Point Cloud
    all_pts = []
    all_cols = []
    for i in range(18):
        gf = global_frames[i]
        d = gf["depth"]
        im_bgr = cv2.imread(image_paths[i])
        im_rgb = cv2.cvtColor(cv2.resize(im_bgr, (W_d, H_d)), cv2.COLOR_BGR2RGB)
        pts = depth_to_points(d, gf["ext"], gf["intr"])
        mask = (d >= 0.20) & (d <= 5.0) & (gf["conf"] >= CONF_THRESH)
        all_pts.append(pts[mask])
        all_cols.append(im_rgb[mask])
        
    pts_merged = np.vstack(all_pts)
    cols_merged = np.vstack(all_cols)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_merged.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(cols_merged.astype(np.float64) / 255.0)
    pcd_down = pcd.voxel_down_sample(voxel_size=0.012)
    o3d.io.write_point_cloud(str(PILOT_DIR / f"pre_tsdf_pcd_{candidate_key}.ply"), pcd_down)
    
    # Run 15mm TSDF Integration
    vol = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=0.015, sdf_trunc=0.045, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    for i in range(18):
        gf = global_frames[i]
        d = gf["depth"].copy()
        mask_inv = (d < 0.20) | (d > 5.0) | (gf["conf"] < CONF_THRESH)
        d[mask_inv] = 0.0
        im_bgr = cv2.imread(image_paths[i])
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
    o3d.io.write_triangle_mesh(str(PILOT_DIR / f"tsdf_mesh_{candidate_key}.ply"), mesh)
    
    bbox = mesh.get_axis_aligned_bounding_box()
    extents = bbox.get_extent().round(4).tolist()
    
    # Measure PCD to TSDF distance
    dists = np.asarray(pcd_down.compute_point_cloud_distance(vol.extract_point_cloud()))
    med_dist = float(np.median(dists)) * 100.0
    p95_dist = float(np.percentile(dists, 95)) * 100.0
    
    print(f"  [+] {candidate_key} TSDF Mesh: {len(mesh.vertices):,} V | {len(mesh.triangles):,} F | Extents: {extents}")
    print(f"      PCD-to-TSDF Median Dist: {med_dist:.2f} cm | 95th-pct: {p95_dist:.2f} cm")
    
    return {
        "candidate": candidate_key,
        "mesh_vertices": len(mesh.vertices),
        "mesh_triangles": len(mesh.triangles),
        "mesh_extents_m": extents,
        "pcd_to_tsdf_median_cm": round(med_dist, 2),
        "pcd_to_tsdf_p95_cm": round(p95_dist, 2),
        "alignment_logs": alignment_logs,
        "mesh_obj": mesh
    }

# Run the 3 comparative treatments
r_defective = run_pilot_experiment("baseline_defective", "defective_baseline", w_cam_weight=0.0)
r_cand_a = run_pilot_experiment("candidate_A_bounded_sim3", "bounded_sim3", w_cam_weight=10.0)
r_cand_b = run_pilot_experiment("candidate_B_rigid_se3", "rigid_se3", w_cam_weight=10.0)

# Multi-view rendering panel
models = [
    {"name": "Defective Baseline", "data": r_defective["mesh_obj"]},
    {"name": "Candidate A: Bounded Sim(3) + Recomputed t", "data": r_cand_a["mesh_obj"]},
    {"name": "Candidate B: Rigid SE(3) + Recomputed t", "data": r_cand_b["mesh_obj"]}
]

views = [
    {"id": "oblique", "name": "Oblique Room Overview", "offset": np.array([-0.8, 0.9, 1.1])},
    {"id": "top_down", "name": "Top-Down Plan Flight View", "offset": np.array([0.0, 1.8, 0.0]), "up": [0, 0, -1]},
    {"id": "door_seam", "name": "Window 4 Planar Door Boundary", "offset": np.array([0.4, 0.0, 0.6])}
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
draw.text((25, 20), "GATE B PILOT — REAL VIDEO COMPARISON: DEFECTIVE BASELINE vs CANDIDATE A (SIM3) vs CANDIDATE B (SE3)", fill=(56, 189, 248), font=font_head)

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

panel_path = PILOT_DIR / "pilot_comparison_panel_real_vggt.png"
panel.save(panel_path, quality=95)
print(f"\n[+] Saved Gate B Real Pilot Comparison Panel: {panel_path.name}")

final_report = {
    "baseline_defective": r_defective,
    "candidate_A_bounded_sim3": r_cand_a,
    "candidate_B_rigid_se3": r_cand_b
}
# Strip mesh_obj for json serialization
for k in final_report:
    del final_report[k]["mesh_obj"]

out_summary_file = PILOT_DIR / "gate_b_pilot_summary.json"
out_summary_file.write_text(json.dumps(final_report, indent=2))
print(f"[+] Saved Gate B Pilot Summary to {out_summary_file.name}")

