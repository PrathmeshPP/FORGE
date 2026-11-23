"""
F.O.R.G.E. Stage 2 — Candidate A Alignment & Point-Cloud Generation for Video 1
Applies:
- Bounded Sim(3) with mandatory translation recomputation
- Joint camera-pose constraints (w_cam=10.0, w_pts=1.0)
- Calibrated p20 confidence threshold from Stage 1
- Full point cloud generation & spatial metrics
"""

import sys
import os
import json
import math
import time
from pathlib import Path
import numpy as np
import open3d as o3d
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.reconstruction.point_cloud import depth_to_points

OUTPUT_DIR = PROJECT_ROOT / "outputs/indoor/vggt_tsdf_video_20261030_170234"
FRAMES_DIR = OUTPUT_DIR / "frames"
CACHE_DIR = OUTPUT_DIR / "vggt_cache"
CAM_DIR = OUTPUT_DIR / "camera_geometry"
PCD_DIR = OUTPUT_DIR / "point_clouds"
DIAG_DIR = OUTPUT_DIR / "diagnostics"
RENDERS_DIR = OUTPUT_DIR / "renders"

for d in [CAM_DIR, PCD_DIR, DIAG_DIR, RENDERS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 76)
print("  F.O.R.G.E. — STAGE 2: CANDIDATE A ALIGNMENT & POINT-CLOUD GENERATION")
print("=" * 76)

# 1. Load curation report & frames
curation_p = OUTPUT_DIR / "frame_curation_report.json"
curation = json.loads(curation_p.read_text())
frame_records = curation["frames"]
N_frames = len(frame_records)
image_paths = [Path(r["file_path"]) for r in frame_records]

# 2. Windows setup
window_size = 6
overlap = 3
step = window_size - overlap
windows = []
start = 0
while start + window_size <= N_frames:
    windows.append(list(range(start, start + window_size)))
    start += step
if start < N_frames:
    windows.append(list(range(max(0, N_frames - window_size), N_frames)))
N_windows = len(windows)

# 3. Load cached raw windows
cache_file = CACHE_DIR / f"raw_windows_0_to_{N_windows-1}.npz"
if not cache_file.exists():
    raise FileNotFoundError(f"Missing cache file: {cache_file}")

print(f"[*] Loading {N_windows} raw windows from {cache_file.name}...")
cached = np.load(cache_file)
raw_windows_data = []
for w in range(N_windows):
    raw_windows_data.append({
        "ext": cached[f"ext_{w}"],
        "intr": cached[f"intr_{w}"],
        "depth": cached[f"depth_{w}"],
        "conf": cached[f"conf_{w}"]
    })

# 4. Load calibrated confidence threshold
conf_diag_p = DIAG_DIR / "confidence_distribution.json"
if conf_diag_p.exists():
    conf_diag = json.loads(conf_diag_p.read_text())
    CONF_THRESH = conf_diag.get("recommended_p20_threshold", 2.0)
else:
    CONF_THRESH = 2.0
print(f"[*] Using Calibrated Depth Confidence Threshold (p20): {CONF_THRESH:.4f}")

H_d, W_d = raw_windows_data[0]["depth"].shape[1], raw_windows_data[0]["depth"].shape[2]

# Solver
def solve_alignment_candidate_a(X_pts, Y_pts, C_src, C_tgt, w_pts=1.0, w_cam=10.0):
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

    if N_p > 0:
        cov_pts = (Xc_pts.T @ Xc_pts) / max(1, N_p)
        eigvals = np.linalg.eigvalsh(cov_pts)
        cond_ratio = math.sqrt(max(1e-8, eigvals[0])) / math.sqrt(max(1e-8, eigvals[-1]))
    else:
        cond_ratio = 1.0

    # Bounded Sim(3) Decision
    if cond_ratio < 0.08 or s_raw < 0.85 or s_raw > 1.20:
        s_applied = float(np.clip(s_raw, 0.85, 1.20))
        reg_mode = f"Bounded Sim(3) (raw s={s_raw:.4f}, clamped={s_applied:.4f}, cond={cond_ratio:.3f})"
    else:
        s_applied = s_raw
        reg_mode = f"Sim(3) full (s={s_raw:.4f}, cond={cond_ratio:.3f})"

    # MANDATORY TRANSLATION RECOMPUTATION
    t_applied = mu_Y - s_applied * (R @ mu_X)

    return s_applied, R, t_applied, s_raw, cond_ratio, reg_mode


# 5. Chronological Alignment
print("\n[*] Aligning 20 windows chronologically with Candidate A...")
global_frames = {}
alignment_logs = []
cumulative_scale = 1.0

# Initialize Window 0
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

alignment_logs.append({
    "window": 0,
    "frames": f"{windows[0][0]:02d}:{windows[0][-1]:02d}",
    "scale_applied": 1.0,
    "scale_raw": 1.0,
    "cumulative_scale": 1.0,
    "condition_ratio": 1.0,
    "rotation_trace": 3.0,
    "translation_norm_m": 0.0,
    "corr_count": 0,
    "inlier_count": 0,
    "inlier_ratio": 1.0,
    "train_residual_cm": 0.0,
    "heldout_residual_cm": 0.0,
    "cam_residual_cm": 0.0,
    "cum_trajectory_len_m": 0.0,
    "mode": "init"
})

sub_y = np.linspace(int(H_d * 0.10), int(H_d * 0.90), 30, dtype=int)
sub_x = np.linspace(int(W_d * 0.10), int(W_d * 0.90), 30, dtype=int)

for w_idx in range(1, N_windows):
    win = windows[w_idx]
    if win == windows[w_idx - 1]:
        print(f"  Win {w_idx:02d}: skipping duplicate window.")
        continue
    w_data = raw_windows_data[w_idx]

    shared_local = [lp for lp, gp in enumerate(win) if gp in global_frames]
    shared_global = [gp for lp, gp in enumerate(win) if gp in global_frames]

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

        if len(pl_sub) > 5:
            pts_loc.append(pl_sub)
            pts_glob.append(pg_sub)
        else:
            # Fallback to top 50 highest confidence points in overlap
            conf_combined = c_loc[sub_y][:, sub_x] + c_glob[sub_y][:, sub_x]
            flat_idx = np.argsort(conf_combined.ravel())[-50:]
            pl_top = pl[sub_y][:, sub_x].reshape(-1, 3)[flat_idx]
            pg_top = pg[sub_y][:, sub_x].reshape(-1, 3)[flat_idx]
            pts_loc.append(pl_top)
            pts_glob.append(pg_top)

        R_l = w_data["ext"][lp, :3, :3]
        t_l = w_data["ext"][lp, :3, 3]
        cams_loc.append(-R_l.T @ t_l)
        cams_glob.append(ge["center"])

    X_all = np.vstack(pts_loc)
    Y_all = np.vstack(pts_glob)
    C_src = np.array(cams_loc)
    C_tgt = np.array(cams_glob)
    N_corr = len(X_all)

    # 80/20 train/test split
    np.random.seed(42 + w_idx)
    perm = np.random.permutation(N_corr)
    n_train = max(1, int(N_corr * 0.8))
    X_train, Y_train = X_all[perm[:n_train]], Y_all[perm[:n_train]]
    X_test, Y_test = X_all[perm[n_train:]], Y_all[perm[n_train:]]
    if len(X_test) == 0:
        X_test, Y_test = X_train, Y_train

    s_w, R_w, t_w, s_raw, cond_ratio, reg_mode = solve_alignment_candidate_a(
        X_train, Y_train, C_src, C_tgt, w_pts=1.0, w_cam=10.0
    )

    # Residuals
    Y_train_pred = s_w * (X_train @ R_w.T) + t_w
    train_res_cm = float(np.mean(np.linalg.norm(Y_train_pred - Y_train, axis=1))) * 100.0

    inliers = np.sum(np.linalg.norm(Y_train_pred - Y_train, axis=1) < 0.03)
    inlier_ratio = float(inliers / max(1, len(X_train)))

    Y_test_pred = s_w * (X_test @ R_w.T) + t_w
    heldout_res_cm = float(np.mean(np.linalg.norm(Y_test_pred - Y_test, axis=1))) * 100.0

    C_tgt_pred = s_w * (C_src @ R_w.T) + t_w
    cam_res_cm = float(np.mean(np.linalg.norm(C_tgt_pred - C_tgt, axis=1))) * 100.0

    cumulative_scale *= s_w
    t_norm = float(np.linalg.norm(t_w))

    # Propagate global frames
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

    curr_frames_sorted = sorted(list(global_frames.keys()))
    centers_so_far = np.array([global_frames[k]["center"] for k in curr_frames_sorted])
    cum_traj_m = float(np.sum(np.linalg.norm(np.diff(centers_so_far, axis=0), axis=1)))

    alignment_logs.append({
        "window": w_idx,
        "frames": f"{win[0]:02d}:{win[-1]:02d}",
        "scale_applied": round(s_w, 4),
        "scale_raw": round(s_raw, 4),
        "cumulative_scale": round(cumulative_scale, 4),
        "condition_ratio": round(cond_ratio, 4),
        "rotation_trace": round(float(np.trace(R_w)), 4),
        "translation_norm_m": round(t_norm, 4),
        "corr_count": N_corr,
        "inlier_count": int(inliers),
        "inlier_ratio": round(inlier_ratio, 4),
        "train_residual_cm": round(train_res_cm, 3),
        "heldout_residual_cm": round(heldout_res_cm, 3),
        "cam_residual_cm": round(cam_res_cm, 3),
        "cum_trajectory_len_m": round(cum_traj_m, 3),
        "mode": reg_mode
    })

    print(f"  Win {w_idx:02d} [{win[0]:02d}:{win[-1]:02d}]: scale={s_w:.3f} (raw={s_raw:.3f}, cum={cumulative_scale:.3f}) | heldout={heldout_res_cm:.2f}cm | cam_err={cam_res_cm:.2f}cm | traj={cum_traj_m:.2f}m | {reg_mode}")

# Summary metrics
heldouts = [l["heldout_residual_cm"] for l in alignment_logs[1:]]
cam_errs = [l["cam_residual_cm"] for l in alignment_logs[1:]]
print("\n" + "=" * 60)
print(f"  Candidate A Alignment Summary (Video 1):")
print(f"  Mean Held-Out 3D Residual:   {np.mean(heldouts):.2f} cm (Median: {np.median(heldouts):.2f} cm, Max: {np.max(heldouts):.2f} cm)")
print(f"  Mean Camera Center Residual: {np.mean(cam_errs):.2f} cm (Median: {np.median(cam_errs):.2f} cm)")
print(f"  Total Trajectory Length:     {alignment_logs[-1]['cum_trajectory_len_m']:.3f} m")
print(f"  Cumulative Scale:            {cumulative_scale:.4f}")
print("=" * 60)

# Save camera poses
sorted_keys = sorted(list(global_frames.keys()))
all_extrinsics = np.stack([global_frames[k]["ext"] for k in sorted_keys])
all_intrinsics = np.stack([global_frames[k]["intr"] for k in sorted_keys])
all_depths = np.stack([global_frames[k]["depth"] for k in sorted_keys])
all_confs = np.stack([global_frames[k]["conf"] for k in sorted_keys])
all_centers = np.stack([global_frames[k]["center"] for k in sorted_keys])

np.savez_compressed(
    CAM_DIR / "camera_poses_candidate_a.npz",
    extrinsics=all_extrinsics,
    intrinsics=all_intrinsics,
    depth=all_depths,
    confidence=all_confs,
    camera_centers=all_centers
)
(DIAG_DIR / "candidate_a_alignment_log.json").write_text(json.dumps(alignment_logs, indent=2))

# 6. Generate Global Point Cloud
print("\n[*] Unprojecting full 63-frame sequence...")
all_pts = []
all_colors = []
conf_filtered_pts = []
conf_filtered_colors = []

# Subsample step for dense unprojection
step_u = 2

for k in sorted_keys:
    gf = global_frames[k]
    img_bgr = cv2.imread(str(image_paths[k]))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H_img, W_img = img_rgb.shape[:2]

    # Resize RGB to match depth map
    rgb_small = cv2.resize(img_rgb, (W_d, H_d), interpolation=cv2.INTER_AREA)

    pts = depth_to_points(gf["depth"], gf["ext"], gf["intr"])
    conf = gf["conf"]

    # Filter depth range [0.15, 6.0]
    valid_depth = (gf["depth"] >= 0.15) & (gf["depth"] <= 6.0)

    # Subsample for unprojected cloud
    v_sub = valid_depth[::step_u, ::step_u]
    pts_sub = pts[::step_u, ::step_u][v_sub]
    colors_sub = rgb_small[::step_u, ::step_u][v_sub] / 255.0

    all_pts.append(pts_sub)
    all_colors.append(colors_sub)

    # Confidence filtered
    c_sub = conf[::step_u, ::step_u][v_sub]
    c_mask = c_sub >= CONF_THRESH
    conf_filtered_pts.append(pts_sub[c_mask])
    conf_filtered_colors.append(colors_sub[c_mask])

all_pts = np.vstack(all_pts)
all_colors = np.vstack(all_colors)
conf_pts = np.vstack(conf_filtered_pts)
conf_colors = np.vstack(conf_filtered_colors)

print(f"[+] Total Raw Unprojected Points: {len(all_pts):,}")
print(f"[+] Confidence-Filtered Points (thresh >= {CONF_THRESH:.3f}): {len(conf_pts):,} ({len(conf_pts)/len(all_pts)*100:.1f}%)")

# Save Raw Point Cloud (sampled for file size)
pcd_raw = o3d.geometry.PointCloud()
sample_raw_idx = np.random.choice(len(all_pts), min(len(all_pts), 500000), replace=False)
pcd_raw.points = o3d.utility.Vector3dVector(all_pts[sample_raw_idx])
pcd_raw.colors = o3d.utility.Vector3dVector(all_colors[sample_raw_idx])
o3d.io.write_point_cloud(str(PCD_DIR / "raw_point_cloud.ply"), pcd_raw)

# Save Confidence Filtered Point Cloud
pcd_conf = o3d.geometry.PointCloud()
sample_conf_idx = np.random.choice(len(conf_pts), min(len(conf_pts), 500000), replace=False)
pcd_conf.points = o3d.utility.Vector3dVector(conf_pts[sample_conf_idx])
pcd_conf.colors = o3d.utility.Vector3dVector(conf_colors[sample_conf_idx])
o3d.io.write_point_cloud(str(PCD_DIR / "confidence_filtered_point_cloud.ply"), pcd_conf)

# 7. Voxel Downsampling (12mm) & Outlier Filtering
pcd_conf_full = o3d.geometry.PointCloud()
pcd_conf_full.points = o3d.utility.Vector3dVector(conf_pts)
pcd_conf_full.colors = o3d.utility.Vector3dVector(conf_colors)

print("[*] Voxel downsampling (12mm) for globally aligned point cloud...")
pcd_down = pcd_conf_full.voxel_down_sample(voxel_size=0.012)
print(f"[+] Downsampled Point Count: {len(pcd_down.points):,}")
o3d.io.write_point_cloud(str(PCD_DIR / "globally_aligned_point_cloud.ply"), pcd_down)

print("[*] Statistical outlier removal (nb_neighbors=30, std_ratio=1.5)...")
pcd_clean, ind = pcd_down.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
pcd_clean.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
pcd_clean.orient_normals_towards_camera_location(camera_location=np.mean(all_centers, axis=0))
print(f"[+] Outlier-Filtered Clean Points: {len(pcd_clean.points):,}")
o3d.io.write_point_cloud(str(PCD_DIR / "fused_point_cloud_clean.ply"), pcd_clean)

# Bounding box & Room dimensions
bbox = pcd_clean.get_axis_aligned_bounding_box()
extent = bbox.get_extent()
min_b = bbox.get_min_bound()
max_b = bbox.get_max_bound()
print(f"  Physical Room Extents: X={extent[0]:.2f}m, Y={extent[1]:.2f}m, Z={extent[2]:.2f}m")
print(f"  Bounds: X [{min_b[0]:.2f}, {max_b[0]:.2f}], Y [{min_b[1]:.2f}, {max_b[1]:.2f}], Z [{min_b[2]:.2f}, {max_b[2]:.2f}]")

# Spatial Density Heatmap
print("[*] Computing spatial-density heatmap...")
kdtree = o3d.geometry.KDTreeFlann(pcd_clean)
densities = []
pts_clean = np.asarray(pcd_clean.points)
for pt in pts_clean:
    [k, idx, _] = kdtree.search_radius_vector_3d(pt, radius=0.05)
    densities.append(k)
densities = np.array(densities, dtype=np.float32)
d_min, d_max = np.percentile(densities, 5), np.percentile(densities, 95)
norm_d = np.clip((densities - d_min) / max(1e-5, d_max - d_min), 0.0, 1.0)

# Pure numpy turbo/plasma-like colormap (no matplotlib dependency)
def turbo_colormap(val):
    # val is in [0, 1]
    # Simple jet: blue -> cyan -> green -> yellow -> red
    r = np.clip(1.5 - np.abs(val * 4.0 - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(val * 4.0 - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(val * 4.0 - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)

heatmap_colors = turbo_colormap(norm_d)

pcd_heatmap = o3d.geometry.PointCloud()
pcd_heatmap.points = o3d.utility.Vector3dVector(pts_clean)
pcd_heatmap.colors = o3d.utility.Vector3dVector(heatmap_colors)
o3d.io.write_point_cloud(str(PCD_DIR / "spatial_density_heatmap.ply"), pcd_heatmap)

pcd_summary = {
    "total_raw_points": int(len(all_pts)),
    "confidence_filtered_points": int(len(conf_pts)),
    "confidence_threshold": float(CONF_THRESH),
    "voxel_downsampled_points": int(len(pcd_down.points)),
    "clean_fused_points": int(len(pcd_clean.points)),
    "room_dimensions_m": {
        "width_x": round(float(extent[0]), 3),
        "height_y": round(float(extent[1]), 3),
        "length_z": round(float(extent[2]), 3)
    },
    "bounding_box_min": [round(float(v), 3) for v in min_b],
    "bounding_box_max": [round(float(v), 3) for v in max_b],
    "spatial_density_pts_per_m3": round(len(pcd_clean.points) / max(1e-3, float(extent[0] * extent[1] * extent[2])), 1),
    "mean_heldout_residual_cm": round(float(np.mean(heldouts)), 3),
    "median_heldout_residual_cm": round(float(np.median(heldouts)), 3),
    "mean_cam_residual_cm": round(float(np.mean(cam_errs)), 3),
    "trajectory_length_m": round(float(alignment_logs[-1]["cum_trajectory_len_m"]), 3)
}
(DIAG_DIR / "point_cloud_summary.json").write_text(json.dumps(pcd_summary, indent=2))
print(f"[+] Saved point cloud summary to {DIAG_DIR / 'point_cloud_summary.json'}")

# 8. Render Point Cloud Views
print("[*] Rendering point cloud views...")
def render_pcd_offscreen(pcd, out_png, view_type="overview"):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1280, height=720)
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = 3.0
    vis.get_render_option().background_color = np.array([0.10, 0.11, 0.13])

    ctr = vis.get_view_control()
    if view_type == "top_plan":
        ctr.set_front([0, -1, 0])
        ctr.set_up([0, 0, -1])
        ctr.set_lookat(pcd.get_center())
        ctr.set_zoom(0.65)
    elif view_type == "side_elevation":
        ctr.set_front([1, 0, 0])
        ctr.set_up([0, -1, 0])
        ctr.set_lookat(pcd.get_center())
        ctr.set_zoom(0.70)
    else:  # overview
        ctr.set_front([0.5, -0.6, -0.6])
        ctr.set_up([0, -1, 0])
        ctr.set_lookat(pcd.get_center())
        ctr.set_zoom(0.70)

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(str(out_png), do_render=True)
    vis.destroy_window()

render_pcd_offscreen(pcd_clean, RENDERS_DIR / "pcd_overview.png", "overview")
render_pcd_offscreen(pcd_clean, RENDERS_DIR / "pcd_top_plan.png", "top_plan")
render_pcd_offscreen(pcd_clean, RENDERS_DIR / "pcd_side_elevation.png", "side_elevation")
render_pcd_offscreen(pcd_heatmap, RENDERS_DIR / "pcd_density_heatmap.png", "overview")
print("[+] Rendered point cloud visual previews.")
