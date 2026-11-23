"""
Diagnostic script to test 59-window alignment on cached raw VGGT data:
Inspects scale factors, condition ratios, correspondence counts across all 59 windows.
"""

import sys
import json
import math
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.reconstruction.point_cloud import depth_to_points

CACHE_FILE = PROJECT_ROOT / "outputs/indoor/distortion_investigation_20261031/gate_d_full_duration/vggt_cache/raw_windows_0_to_58.npz"
cached = np.load(CACHE_FILE)

N_frames = 179
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
CONF_THRESH = 2.5205

H_d, W_d = cached["depth_0"].shape[1], cached["depth_0"].shape[2]

def solve_alignment_test(X_pts, Y_pts, C_src, C_tgt, w_pts=1.0, w_cam=10.0):
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

    if cond_ratio < 0.08 or s_raw < 0.85 or s_raw > 1.20:
        s_applied = float(np.clip(s_raw, 0.85, 1.20))
        reg_mode = f"Bounded (raw={s_raw:.3f}, clamped={s_applied:.3f}, cond={cond_ratio:.3f})"
    else:
        s_applied = s_raw
        reg_mode = f"Sim3 full (s={s_raw:.3f}, cond={cond_ratio:.3f})"

    t_applied = mu_Y - s_applied * (R @ mu_X)
    return s_applied, R, t_applied, s_raw, cond_ratio, reg_mode

global_frames = {}
w0_ext = cached["ext_0"]
w0_intr = cached["intr_0"]
w0_depth = cached["depth_0"]
w0_conf = cached["conf_0"]

for lp, gp in enumerate(windows[0]):
    R = w0_ext[lp, :3, :3]
    t = w0_ext[lp, :3, 3]
    c = -R.T @ t
    global_frames[gp] = {
        "ext": w0_ext[lp].copy(),
        "intr": w0_intr[lp].copy(),
        "depth": w0_depth[lp].copy(),
        "conf": w0_conf[lp].copy(),
        "center": c,
        "scale": 1.0
    }

sub_y = np.linspace(int(H_d * 0.10), int(H_d * 0.90), 30, dtype=int)
sub_x = np.linspace(int(W_d * 0.10), int(W_d * 0.90), 30, dtype=int)

scale_trace = []

for w_idx in range(1, N_windows):
    win = windows[w_idx]
    ext_w = cached[f"ext_{w_idx}"]
    intr_w = cached[f"intr_{w_idx}"]
    depth_w = cached[f"depth_{w_idx}"]
    conf_w = cached[f"conf_{w_idx}"]

    shared_local = [lp for lp, gp in enumerate(win) if gp in global_frames]
    shared_global = [gp for lp, gp in enumerate(win) if gp in global_frames]

    pts_loc = []
    pts_glob = []
    cams_loc = []
    cams_glob = []

    for lp, gp in zip(shared_local, shared_global):
        ge = global_frames[gp]
        pg = depth_to_points(ge["depth"], ge["ext"], ge["intr"])
        pl = depth_to_points(depth_w[lp], ext_w[lp], intr_w[lp])

        c_loc = conf_w[lp]
        c_glob = ge["conf"]
        mask = (c_loc[sub_y][:, sub_x] >= CONF_THRESH) & (c_glob[sub_y][:, sub_x] >= CONF_THRESH)
        pl_sub = pl[sub_y][:, sub_x][mask]
        pg_sub = pg[sub_y][:, sub_x][mask]

        if len(pl_sub) > 5:
            pts_loc.append(pl_sub)
            pts_glob.append(pg_sub)
        else:
            # Fallback to top 50 highest confidence points in overlap if threshold is too strict
            conf_combined = c_loc[sub_y][:, sub_x] + c_glob[sub_y][:, sub_x]
            flat_idx = np.argsort(conf_combined.ravel())[-50:]
            pl_top = pl[sub_y][:, sub_x].reshape(-1, 3)[flat_idx]
            pg_top = pg[sub_y][:, sub_x].reshape(-1, 3)[flat_idx]
            pts_loc.append(pl_top)
            pts_glob.append(pg_top)

        R_l = ext_w[lp, :3, :3]
        t_l = ext_w[lp, :3, 3]
        cams_loc.append(-R_l.T @ t_l)
        cams_glob.append(ge["center"])

    X_all = np.vstack(pts_loc)
    Y_all = np.vstack(pts_glob)
    C_src = np.array(cams_loc)
    C_tgt = np.array(cams_glob)

    s_w, R_w, t_w, s_raw, cond_ratio, reg_mode = solve_alignment_test(
        X_all, Y_all, C_src, C_tgt, w_pts=1.0, w_cam=10.0
    )

    scale_trace.append((w_idx, s_w, s_raw, cond_ratio, len(X_all)))

    for lp, gp in enumerate(win):
        if gp not in global_frames:
            R_local = ext_w[lp, :3, :3]
            t_local = ext_w[lp, :3, 3]
            c_local = -R_local.T @ t_local

            c_global = s_w * (R_w @ c_local) + t_w
            R_global = R_local @ R_w.T
            t_global = -R_global @ c_global
            ext_global = np.hstack([R_global, t_global.reshape(3, 1)])
            depth_global = s_w * depth_w[lp]

            global_frames[gp] = {
                "ext": ext_global.astype(np.float32),
                "intr": intr_w[lp].astype(np.float32),
                "depth": depth_global.astype(np.float32),
                "conf": conf_w[lp].copy(),
                "center": c_global,
                "scale": s_w
            }

print(f"Alignment test completed for all {N_windows} windows!")
for w_idx, s_w, s_raw, cond, n_pts in scale_trace:
    print(f"Win {w_idx:02d}: s_w={s_w:.4f} (raw={s_raw:.4f}, cond={cond:.3f}, pts={n_pts})")

