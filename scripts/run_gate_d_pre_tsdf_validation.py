"""
F.O.R.G.E. — GATE D: Full-Duration Alignment Validation (179 Frames, 59 Windows).
1. Reuses full 59-window VGGT inference from raw_windows_0_to_58.npz cache.
2. Applies Candidate A across all 59 windows:
   - Bounded Sim(3) with scale clamping in [0.85, 1.20]
   - Mandatory translation recomputation after scale clamping: t_applied = mu_Y - s_applied * R * mu_X
   - Joint camera constraints (w_cam=10.0, w_pts=1.0)
   - Calibrated p20 confidence filtering (thresh=2.5205) with adaptive fallback on sparse frames
3. Records per-window metrics (rotation, translation, scale, inliers, residuals, held-out errors, camera errors).
4. Generates the complete pre-TSDF point cloud FIRST:
   - Full-room isometric view
   - Full-room top-down view (floorplan check)
   - Side elevation view
   - Camera trajectory lineset overlay
   - Spatial-density heatmap point cloud
   - Per-window point clouds for representative windows
   - Point cloud coverage and spatial dimensions report
5. STOPS before TSDF fusion for user review.
"""

import sys
import json
import math
import time
from pathlib import Path
import numpy as np
import open3d as o3d
import cv2
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.reconstruction.point_cloud import depth_to_points

EXP_DIR = PROJECT_ROOT / "outputs/indoor/vggt_tsdf_video_20261031_142303"
INVEST_DIR = PROJECT_ROOT / "outputs/indoor/distortion_investigation_20261031"
GATE_D_DIR = INVEST_DIR / "gate_d_full_duration"
GATE_D_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = GATE_D_DIR / "vggt_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PER_WIN_DIR = GATE_D_DIR / "per_window_pcd"
PER_WIN_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = Path(r"C:\Users\25beevdt025\.gemini\antigravity\brain\536a497a-fba6-496d-b4a9-b4f0cf756953")

print("=" * 76)
print("  GATE D — STEP 1: AUDIT & REUSE EXISTING INFERENCE RESULTS")
print("=" * 76)

curation_report_p = EXP_DIR / "frame_curation_report.json"
curation_report = json.loads(curation_report_p.read_text())
frame_records = curation_report["frames"]
N_frames = len(frame_records)
image_paths = [r["file_path"] for r in frame_records]

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
print(f"[*] Total Curated Frames: {N_frames}")
print(f"[*] Total Sliding Windows: {N_windows} (size={window_size}, overlap={overlap})")

full_cache_p = CACHE_DIR / "raw_windows_0_to_58.npz"
if not full_cache_p.exists():
    raise FileNotFoundError(f"Missing cache file: {full_cache_p}")

print(f"[*] Loading all 59 raw local VGGT windows from {full_cache_p.name}...")
cached_raw = np.load(full_cache_p)
raw_windows_data = []
for w in range(N_windows):
    raw_windows_data.append({
        "ext": cached_raw[f"ext_{w}"],
        "intr": cached_raw[f"intr_{w}"],
        "depth": cached_raw[f"depth_{w}"],
        "conf": cached_raw[f"conf_{w}"]
    })
print(f"[+] Successfully loaded all {N_windows} windows from cache!")

H_d, W_d = raw_windows_data[0]["depth"].shape[1], raw_windows_data[0]["depth"].shape[2]
CONF_THRESH = 2.5205  # p20 calibrated threshold

# Alignment Solver for Candidate A
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

# ==============================================================================
# STAGE 2: EXECUTE FULL 59-WINDOW CHRONOLOGICAL ALIGNMENT
# ==============================================================================
print("\n" + "=" * 76)
print("  GATE D — STEP 2: FULL 59-WINDOW CANDIDATE A CHRONOLOGICAL ALIGNMENT")
print("=" * 76)

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
    "frames": f"{windows[0][0]:03d}:{windows[0][-1]:03d}",
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

    # Evaluate residuals
    Y_train_pred = s_w * (X_train @ R_w.T) + t_w
    train_res_cm = float(np.mean(np.linalg.norm(Y_train_pred - Y_train, axis=1))) * 100.0

    inliers = np.sum(np.linalg.norm(Y_train_pred - Y_train, axis=1) < 0.03)  # < 3cm inlier
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
        "frames": f"{win[0]:03d}:{win[-1]:03d}",
        "scale_applied": round(float(s_w), 4),
        "scale_raw": round(float(s_raw), 4),
        "cumulative_scale": round(float(cumulative_scale), 4),
        "condition_ratio": round(float(cond_ratio), 4),
        "rotation_trace": round(float(np.trace(R_w)), 4),
        "translation_norm_m": round(t_norm, 4),
        "corr_count": N_corr,
        "inlier_count": int(inliers),
        "inlier_ratio": round(inlier_ratio, 3),
        "train_residual_cm": round(train_res_cm, 2),
        "heldout_residual_cm": round(heldout_res_cm, 2),
        "cam_residual_cm": round(cam_res_cm, 2),
        "cum_trajectory_len_m": round(cum_traj_m, 3),
        "mode": reg_mode
    })

    if w_idx % 5 == 0 or w_idx == N_windows - 1 or "Bounded" in reg_mode:
        print(f"  Win {w_idx:02d} ({win[0]:03d}:{win[-1]:03d}): s={s_w:.4f} (raw={s_raw:.4f}) | cum_s={cumulative_scale:.3f} | Held-out={heldout_res_cm:5.2f}cm | Cam={cam_res_cm:5.2f}cm | {reg_mode}")

# Summary statistics
all_heldout = [log["heldout_residual_cm"] for log in alignment_logs[1:]]
all_cam = [log["cam_residual_cm"] for log in alignment_logs[1:]]
all_scales = [log["scale_applied"] for log in alignment_logs[1:]]
bounded_count = sum(1 for log in alignment_logs[1:] if "Bounded" in log["mode"])

print(f"\n[+] 59-Window Alignment Complete:")
print(f"    Total Windows Aligned: {N_windows}")
print(f"    Bounded Scale Windows: {bounded_count} / {N_windows-1}")
print(f"    Mean Held-out 3D Residual: {np.mean(all_heldout):.2f} cm (Median: {np.median(all_heldout):.2f} cm, Max: {np.max(all_heldout):.2f} cm)")
print(f"    Mean Camera Center Residual: {np.mean(all_cam):.2f} cm (Median: {np.median(all_cam):.2f} cm)")
print(f"    Cumulative Scale Factor: {cumulative_scale:.4f}")

# Trajectory analysis
all_centers = np.array([global_frames[i]["center"] for i in range(N_frames)])
traj_diffs = np.linalg.norm(np.diff(all_centers, axis=0), axis=1)
total_traj_len = float(np.sum(traj_diffs))
max_step = float(np.max(traj_diffs))
median_step = float(np.median(traj_diffs))
traj_start = all_centers[0].tolist()
traj_end = all_centers[-1].tolist()
net_displacement = float(np.linalg.norm(all_centers[-1] - all_centers[0]))

print(f"    Trajectory: Total Length = {total_traj_len:.3f} m | Median Step = {median_step*100:.2f} cm | Max Step = {max_step*100:.2f} cm")
print(f"    Net Camera Displacement: {net_displacement:.3f} m")

# Save Camera Poses for Gate D
camera_poses_d = {
    "extrinsics": np.array([global_frames[i]["ext"] for i in range(N_frames)]),
    "intrinsics": np.array([global_frames[i]["intr"] for i in range(N_frames)]),
    "centers": all_centers
}
np.savez_compressed(GATE_D_DIR / "camera_poses_candidate_a.npz", **camera_poses_d)
print(f"[+] Saved corrected camera poses to camera_poses_candidate_a.npz")

# ==============================================================================
# STAGE 3: GENERATE COMPLETE PRE-TSDF POINT CLOUD FIRST
# ==============================================================================
print("\n" + "=" * 76)
print("  GATE D — STEP 3: GENERATE COMPLETE PRE-TSDF POINT CLOUD FIRST")
print("=" * 76)

all_pts_list = []
all_cols_list = []
all_frames_id_list = []

for i in range(N_frames):
    gf = global_frames[i]
    d = gf["depth"]
    im_bgr = cv2.imread(image_paths[i])
    im_rgb = cv2.cvtColor(cv2.resize(im_bgr, (W_d, H_d)), cv2.COLOR_BGR2RGB)
    pts = depth_to_points(d, gf["ext"], gf["intr"])

    mask = (d >= 0.20) & (d <= 5.0) & (gf["conf"] >= CONF_THRESH)
    pts_valid = pts[mask]
    cols_valid = im_rgb[mask]

    all_pts_list.append(pts_valid)
    all_cols_list.append(cols_valid)
    all_frames_id_list.append(np.full(len(pts_valid), i, dtype=np.int32))

    if (i + 1) % 30 == 0 or i == N_frames - 1:
        print(f"    Unprojected frame {i+1:03d}/{N_frames:03d} ({len(pts_valid):,} valid pts)")

merged_pts = np.vstack(all_pts_list)
merged_cols = np.vstack(all_cols_list)
merged_frame_ids = np.concatenate(all_frames_id_list)

print(f"\n[+] Raw Unfiltered Merged Point Cloud: {len(merged_pts):,} total points")

# Downsample to 12mm voxel grid for structural clarity
pcd_raw = o3d.geometry.PointCloud()
pcd_raw.points = o3d.utility.Vector3dVector(merged_pts.astype(np.float64))
pcd_raw.colors = o3d.utility.Vector3dVector(merged_cols.astype(np.float64) / 255.0)

print("[*] Voxel downsampling (12mm) for spatial analysis...")
pcd_down = pcd_raw.voxel_down_sample(voxel_size=0.012)
print(f"[+] Clean Merged Point Cloud: {len(pcd_down.points):,} points (voxel 12mm)")

pcd_down_path = GATE_D_DIR / "full_pre_tsdf_point_cloud_12mm.ply"
o3d.io.write_point_cloud(str(pcd_down_path), pcd_down)
print(f"[+] Saved Full Pre-TSDF Point Cloud: {pcd_down_path.name} ({pcd_down_path.stat().st_size / 1024 / 1024:.2f} MB)")

# Statistical outlier filter
cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=25, std_ratio=2.0)
pcd_clean = pcd_down.select_by_index(ind)
pcd_clean_path = GATE_D_DIR / "full_pre_tsdf_point_cloud_clean.ply"
o3d.io.write_point_cloud(str(pcd_clean_path), pcd_clean)
print(f"[+] Saved Statistical-Filtered Point Cloud: {pcd_clean_path.name} ({len(pcd_clean.points):,} pts)")

# Spatial Bounding Box & Dimensions
bbox = pcd_clean.get_axis_aligned_bounding_box()
pcd_min = bbox.get_min_bound().round(3).tolist()
pcd_max = bbox.get_max_bound().round(3).tolist()
pcd_extents = bbox.get_extent().round(3).tolist()
pcd_center = bbox.get_center().round(3).tolist()

print(f"\n[*] Spatial Room Extents: X={pcd_extents[0]:.2f}m, Y={pcd_extents[1]:.2f}m, Z={pcd_extents[2]:.2f}m")
print(f"    Bounds Min: {pcd_min}")
print(f"    Bounds Max: {pcd_max}")
print(f"    Room Center: {pcd_center}")

# Per-window point clouds
key_sample_windows = [0, 10, 20, 30, 40, 50, 58]
for kw in key_sample_windows:
    kw_frames = windows[kw]
    kw_mask = np.isin(merged_frame_ids, kw_frames)
    p_kw = o3d.geometry.PointCloud()
    p_kw.points = o3d.utility.Vector3dVector(merged_pts[kw_mask].astype(np.float64))
    p_kw.colors = o3d.utility.Vector3dVector(merged_cols[kw_mask].astype(np.float64) / 255.0)
    p_kw_down = p_kw.voxel_down_sample(voxel_size=0.015)
    kw_path = PER_WIN_DIR / f"window_{kw:02d}_frames_{kw_frames[0]:03d}_{kw_frames[-1]:03d}.ply"
    o3d.io.write_point_cloud(str(kw_path), p_kw_down)

print(f"[+] Saved {len(key_sample_windows)} representative window point clouds to {PER_WIN_DIR.name}/")

# Generate Density Heatmap Point Cloud
print("\n[*] Computing spatial density heatmap...")
kdtree = o3d.geometry.KDTreeFlann(pcd_clean)
pts_np = np.asarray(pcd_clean.points)
densities = np.zeros(len(pts_np), dtype=np.float32)
for idx in range(len(pts_np)):
    [k, idxs, _] = kdtree.search_radius_vector_3d(pts_np[idx], 0.05)
    densities[idx] = k

d_min = float(np.percentile(densities, 5))
d_max = float(np.percentile(densities, 95))
d_norm = np.clip((densities - d_min) / max(1e-4, d_max - d_min), 0.0, 1.0)

heatmap_cols = np.zeros((len(pts_np), 3), dtype=np.float32)
heatmap_cols[:, 0] = np.clip(1.5 * d_norm - 0.5, 0.0, 1.0)
heatmap_cols[:, 1] = np.sin(d_norm * np.pi)
heatmap_cols[:, 2] = np.clip(1.0 - 1.5 * d_norm, 0.0, 1.0)

pcd_heatmap = o3d.geometry.PointCloud()
pcd_heatmap.points = pcd_clean.points
pcd_heatmap.colors = o3d.utility.Vector3dVector(heatmap_cols)
heatmap_path = GATE_D_DIR / "full_pre_tsdf_density_heatmap.ply"
o3d.io.write_point_cloud(str(heatmap_path), pcd_heatmap)
print(f"[+] Saved Density Heatmap Point Cloud: {heatmap_path.name}")

# Trajectory Geometry
line_points = [all_centers[i] for i in range(len(all_centers))]
line_indices = [[i, i + 1] for i in range(len(all_centers) - 1)]
lineset = o3d.geometry.LineSet()
lineset.points = o3d.utility.Vector3dVector(line_points)
lineset.lines = o3d.utility.Vector2iVector(line_indices)
line_cols = [[0.2, 0.8, 1.0] for _ in range(len(line_indices))]
lineset.colors = o3d.utility.Vector3dVector(line_cols)

# ==============================================================================
# STAGE 4: RENDER COMPREHENSIVE MULTI-VIEW DIAGNOSTIC PANEL
# ==============================================================================
print("\n" + "=" * 76)
print("  GATE D — STEP 4: RENDER COMPREHENSIVE MULTI-VIEW DIAGNOSTIC PANEL")
print("=" * 76)

diag_render_views = [
    {
        "id": "isometric_overview",
        "name": "Full-Room Isometric Overview",
        "offset": np.array([-1.0, 0.9, 1.2]),
        "up": [0.0, 1.0, 0.0]
    },
    {
        "id": "top_down_plan",
        "name": "Full-Room Top-Down Plan (Wall Continuity Check)",
        "offset": np.array([0.0, 2.8, 0.0]),
        "up": [0.0, 0.0, -1.0]
    },
    {
        "id": "side_elevation",
        "name": "Side Profile (Floor & Ceiling Levelling Check)",
        "offset": np.array([2.2, 0.1, 0.0]),
        "up": [0.0, 1.0, 0.0]
    },
    {
        "id": "density_heatmap_iso",
        "name": "Spatial-Density Heatmap (Point Density Concentration)",
        "offset": np.array([-1.0, 0.9, 1.2]),
        "up": [0.0, 1.0, 0.0],
        "use_heatmap": True
    }
]

rendered_images = {}
center = np.array(pcd_center)
ext = np.array(pcd_extents)
diag_scale = float(np.linalg.norm(ext))

for vw in diag_render_views:
    v_id = vw["id"]
    geom_to_render = pcd_heatmap if vw.get("use_heatmap", False) else pcd_clean
    out_img_p = GATE_D_DIR / f"render_{v_id}.png"

    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=1280, height=840)
        vis.add_geometry(geom_to_render)
        vis.add_geometry(lineset)

        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.08, 0.09, 0.12])
        opt.point_size = 2.8
        opt.light_on = False

        ctr = vis.get_view_control()
        cam_pos = center + vw["offset"] * (diag_scale * 0.70)
        front = center - cam_pos
        front /= (np.linalg.norm(front) + 1e-8)
        ctr.set_front(front)
        ctr.set_up(vw["up"])
        ctr.set_lookat(center)

        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(str(out_img_p), do_render=True)
        vis.destroy_window()
        rendered_images[v_id] = out_img_p
        print(f"  [+] Rendered: {v_id} -> {out_img_p.name}")
    except Exception as e:
        print(f"  [-] Failed rendering {v_id}: {e}")

# Assemble 2x2 Master Inspection Panel
panel_w, panel_h = 1600, 1100
cell_w, cell_h = 790, 480
header_h = 90

panel = Image.new("RGB", (panel_w, panel_h), color=(15, 23, 42))
draw = ImageDraw.Draw(panel)

try:
    font_title = ImageFont.truetype("arialbd.ttf", 22)
    font_sub = ImageFont.truetype("arial.ttf", 13)
    font_cell = ImageFont.truetype("arialbd.ttf", 13)
except Exception:
    font_title = ImageFont.load_default()
    font_sub = ImageFont.load_default()
    font_cell = ImageFont.load_default()

# Header
draw.rectangle([0, 0, panel_w, header_h], fill=(24, 32, 47))
draw.text((30, 18), "F.O.R.G.E. — GATE D: FULL-DURATION (136s, 59 WINDOWS) PRE-TSDF POINT CLOUD VALIDATION", fill=(56, 189, 248), font=font_title)
draw.text((30, 48), f"Candidate A Bounded Sim(3) Alignment | 179 Frames | Extents: [{pcd_extents[0]:.2f}m, {pcd_extents[1]:.2f}m, {pcd_extents[2]:.2f}m] | Mean 3D Residual: {np.mean(all_heldout):.2f} cm", fill=(148, 163, 184), font=font_sub)

grid_positions = [
    (0, 0, "isometric_overview"),
    (1, 0, "top_down_plan"),
    (0, 1, "side_elevation"),
    (1, 1, "density_heatmap_iso")
]

for col, row, vid in grid_positions:
    x = 10 + col * (cell_w + 10)
    y = header_h + 10 + row * (cell_h + 10)

    if vid in rendered_images and rendered_images[vid].exists():
        im = Image.open(rendered_images[vid]).resize((cell_w, cell_h), Image.Resampling.LANCZOS)
        panel.paste(im, (x, y))
    else:
        draw.rectangle([x, y, x + cell_w, y + cell_h], fill=(30, 41, 59))
        draw.text((x + 20, y + cell_h // 2), f"Missing render: {vid}", fill=(148, 163, 184), font=font_cell)

    vw_meta = next(v for v in diag_render_views if v["id"] == vid)
    draw.rectangle([x, y, x + cell_w, y + 26], fill=(15, 23, 42, 220))
    draw.text((x + 10, y + 6), f"{vw_meta['name']}", fill=(241, 245, 249), font=font_cell)
    draw.rectangle([x, y, x + cell_w, y + cell_h], outline=(51, 65, 85), width=1)

panel_path = GATE_D_DIR / "gate_d_pre_tsdf_diagnostic_panel.png"
panel.save(panel_path, quality=95)
print(f"\n[+] Master Diagnostic Panel Saved: {panel_path.name}")

# Copy to artifacts directory
if ARTIFACT_DIR.is_dir():
    import shutil
    shutil.copy(panel_path, ARTIFACT_DIR / "gate_d_pre_tsdf_diagnostic_panel.png")
    shutil.copy(pcd_down_path, ARTIFACT_DIR / "full_pre_tsdf_point_cloud_12mm.ply")
    for vid, p in rendered_images.items():
        shutil.copy(p, ARTIFACT_DIR / p.name)
    print(f"[+] Diagnostic assets copied to artifacts directory.")

# Save Comprehensive Gate D Audit Report
final_report = {
    "validation_gate": "GATE D — FULL-DURATION PRE-TSDF VALIDATION",
    "video_duration_sec": curation_report["duration_sec"],
    "total_frames": N_frames,
    "total_windows": N_windows,
    "reused_windows_count": N_windows,
    "alignment_formulation": "Candidate A: Bounded Sim(3) + Mandatory Translation Recomputation + Joint Camera Constraints",
    "p20_confidence_threshold": CONF_THRESH,
    "camera_constraint_weight": 10.0,
    "scale_bounds": [0.85, 1.20],
    "bounded_scale_windows_count": bounded_count,
    "cumulative_scale_factor": round(cumulative_scale, 4),
    "mean_heldout_residual_cm": round(float(np.mean(all_heldout)), 2),
    "median_heldout_residual_cm": round(float(np.median(all_heldout)), 2),
    "max_heldout_residual_cm": round(float(np.max(all_heldout)), 2),
    "mean_camera_center_residual_cm": round(float(np.mean(all_cam)), 2),
    "median_camera_center_residual_cm": round(float(np.median(all_cam)), 2),
    "trajectory_metrics": {
        "total_trajectory_len_m": round(total_traj_len, 3),
        "median_step_between_frames_cm": round(median_step * 100.0, 2),
        "max_step_between_frames_cm": round(max_step * 100.0, 2),
        "net_camera_displacement_m": round(net_displacement, 3),
        "start_camera_pos_m": [round(x, 4) for x in traj_start],
        "end_camera_pos_m": [round(x, 4) for x in traj_end]
    },
    "spatial_point_cloud_metrics": {
        "raw_points_count": int(len(merged_pts)),
        "voxel_downsampled_12mm_count": int(len(pcd_down.points)),
        "statistical_filtered_clean_count": int(len(pcd_clean.points)),
        "room_extents_m": pcd_extents,
        "bounds_min_m": pcd_min,
        "bounds_max_m": pcd_max,
        "room_center_m": pcd_center
    },
    "alignment_window_logs": alignment_logs
}

report_out_p = GATE_D_DIR / "gate_d_pre_tsdf_report.json"
report_out_p.write_text(json.dumps(final_report, indent=2))
print(f"[+] Saved Gate D Pre-TSDF Report: {report_out_p.name}")
print("\n" + "=" * 76)
print("  GATE D CHECKPOINT: PRE-TSDF STAGE COMPLETE. STOPPING BEFORE TSDF FUSION.")
print("=" * 76)
