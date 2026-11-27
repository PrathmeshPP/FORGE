"""
F.O.R.G.E. V2 — Global Sim(3) Pose Graph, Revisit Loop Closures,
and Independently-Validated Sparse Bundle Adjustment (BA).

Replaces open-loop sequential Sim(3) chaining with:
1. Per-window internal consistency quality checks (within-window pose & depth gate).
2. Stationary visual feature tracking (SIFT + spatial grid gating, excluding dynamic masks & depth boundaries).
3. Verified long-range revisit loop-closure detection across non-adjacent windows.
4. Global 7-DoF Sim(3) window pose graph optimization (resolving scale drift & yaw/pitch warp).
5. Sparse multi-view Bundle Adjustment with mandatory stratified held-out track validation:
   - Preserves baseline solution explicitly.
   - Splits multi-view tracks into 80% optimization / 20% independent held-out validation.
   - Accepts refined poses ONLY if held-out cross-view / reprojection & trajectory checks improve
     or stay strictly non-degrading; otherwise rejects BA and retains baseline.
6. Alignment acceptance gate (`ALIGNMENT_PASS`, `ALIGNMENT_RECOVERABLE`, `PARTIAL_ALIGNMENT`, `ALIGNMENT_FAILED`).
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from src.reconstruction.canonical_camera import (
    apply_sim3_to_extrinsic,
    apply_sim3_to_points,
    backproject_pixels,
    camera_center_from_extrinsic,
    extrinsic_from_center_and_rotation,
    project_world_points,
)


@dataclass
class AlignmentQAThresholds:
    """Calibrated QA acceptance thresholds derived from indoor regression scenes."""

    max_median_reproj_px: float = 4.5
    max_within_window_depth_rel: float = 0.08
    max_shared_frame_cam_drift_m: float = 0.08
    max_weak_join_ratio: float = 0.25
    max_trajectory_step_m: float = 0.85
    min_heldout_tracks_for_ba_eval: int = 12
    max_heldout_reproj_regression_px: float = 0.15
    min_loop_inliers: int = 22
    min_loop_grid_cells: int = 5
    max_loop_sim3_median_m: float = 0.065


@dataclass
class GlobalAlignmentReport:
    status: str  # ALIGNMENT_PASS | ALIGNMENT_RECOVERABLE | PARTIAL_ALIGNMENT | ALIGNMENT_FAILED
    ba_decision: str  # ACCEPTED | REJECTED | SKIPPED_INSUFFICIENT_EVIDENCE | DISABLED
    ba_reason: str
    num_windows: int
    within_window_passed: int
    within_window_failed: int
    sequential_edges: int
    loop_closure_edges: int
    weak_joins: List[int] = field(default_factory=list)
    failed_joins: List[int] = field(default_factory=list)
    total_tracks: int = 0
    opt_tracks: int = 0
    heldout_tracks: int = 0
    baseline_metrics: Dict[str, Any] = field(default_factory=dict)
    pose_graph_metrics: Dict[str, Any] = field(default_factory=dict)
    ba_refined_metrics: Dict[str, Any] = field(default_factory=dict)
    accepted_metrics: Dict[str, Any] = field(default_factory=dict)
    window_scales: List[float] = field(default_factory=list)
    loop_closures_detail: List[Dict[str, Any]] = field(default_factory=list)
    within_window_details: List[Dict[str, Any]] = field(default_factory=list)
    gate_failures: List[str] = field(default_factory=list)
    duration_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _rodrigues_to_mat(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return R


def _mat_to_rodrigues(R: np.ndarray) -> np.ndarray:
    rv, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64))
    return rv.flatten()


def _rotation_angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    tr = float(np.trace(np.asarray(R1, dtype=np.float64) @ np.asarray(R2, dtype=np.float64).T))
    return float(np.degrees(math.acos(np.clip((tr - 1.0) * 0.5, -1.0, 1.0))))


def _umeyama_ransac_sim3(
    X: np.ndarray,
    Y: np.ndarray,
    max_res: float = 0.05,
    trials: int = 45,
) -> Optional[Tuple[float, np.ndarray, np.ndarray, float, float, float]]:
    """
    Solves robust 7-DoF Y ≈ s * (X @ R.T) + t with collinearity & scale sanity guard.
    Returns (s, R, t, med_res, p90_res, inlier_ratio) or None.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    N = len(X)
    if N < 6:
        return None
    spread_x = float(np.linalg.norm(np.ptp(X, axis=0)))
    if spread_x < 0.08:
        return None

    rng = np.random.RandomState(42)
    best_inl = None
    best_cnt = 0
    if N >= 24:
        for _ in range(trials):
            idx = rng.choice(N, size=min(10, N), replace=False)
            xs, ys = X[idx], Y[idx]
            mx, my = xs.mean(axis=0), ys.mean(axis=0)
            xc, yc = xs - mx, ys - my
            vx = float(np.mean(np.sum(xc ** 2, axis=1)))
            if vx < 1e-6:
                continue
            U, D, Vt = np.linalg.svd((yc.T @ xc) / len(idx))
            S = np.eye(3)
            if np.linalg.det(U @ Vt) < 0:
                S[2, 2] = -1.0
            Rc = U @ S @ Vt
            sc = float(np.sum(D * np.diag(S)) / vx)
            if not (0.35 <= sc <= 2.6):
                continue
            tc = my - sc * (Rc @ mx)
            res = np.linalg.norm(Y - (sc * (X @ Rc.T) + tc), axis=1)
            inl = np.where(res <= max_res)[0]
            if len(inl) > best_cnt:
                best_cnt = len(inl)
                best_inl = inl

    Xin, Yin = (X[best_inl], Y[best_inl]) if (best_inl is not None and best_cnt >= 8) else (X, Y)
    mx, my = Xin.mean(axis=0), Yin.mean(axis=0)
    xc, yc = Xin - mx, Yin - my
    vx = float(np.mean(np.sum(xc ** 2, axis=1)))
    if vx < 1e-8:
        return None
    U, D, Vt = np.linalg.svd((yc.T @ xc) / len(Xin))
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    s = float(np.sum(D * np.diag(S)) / vx)
    if not (0.25 <= s <= 3.5) or not math.isfinite(s):
        return None
    t = my - s * (R @ mx)
    all_res = np.linalg.norm(Y - (s * (X @ R.T) + t), axis=1)
    return (
        s, R, t,
        float(np.median(all_res)),
        float(np.percentile(all_res, 90)),
        float(np.mean(all_res <= max_res)),
    )


def check_within_window_consistency(
    win_idx: int,
    ext: np.ndarray,
    intr: np.ndarray,
    depth: np.ndarray,
    conf: np.ndarray,
    min_depth: float = 0.20,
    max_depth: float = 5.50,
    max_rel_depth_err: float = 0.08,
) -> Dict[str, Any]:
    """
    Independently verifies that predictions *inside* a single VGGT window are internally
    consistent before admitting the window to global alignment.
    """
    W_cnt = len(ext)
    d_arr = depth[..., 0] if depth.ndim == 4 else depth
    finite_ok = bool(np.all(np.isfinite(ext)) and np.all(np.isfinite(intr)) and np.all(np.isfinite(d_arr)))
    per_frame_valid = [
        float(np.mean((d_arr[k] >= min_depth) & (d_arr[k] <= max_depth) & (conf[k] >= 0.65)))
        for k in range(W_cnt)
    ]
    valid_frac = float(np.mean(per_frame_valid))
    min_frame_valid = float(min(per_frame_valid)) if per_frame_valid else 0.0
    med_conf = float(np.median(conf))

    H, W = d_arr.shape[1], d_arr.shape[2]
    ys = np.linspace(int(H * 0.15), int(H * 0.85), 14, dtype=int)
    xs = np.linspace(int(W * 0.15), int(W * 0.85), 14, dtype=int)
    ug, vg = np.meshgrid(xs, ys)
    uf, vf = ug.ravel().astype(np.float64), vg.ravel().astype(np.float64)

    rel_errs = []
    centers = [camera_center_from_extrinsic(ext[k]) for k in range(W_cnt)]
    cam_steps = [float(np.linalg.norm(centers[k + 1] - centers[k])) for k in range(W_cnt - 1)]

    for k in range(W_cnt - 1):
        dk = d_arr[k, vg, ug].ravel()
        ck = conf[k, vg, ug].ravel()
        mk = (dk >= min_depth) & (dk <= max_depth) & (ck >= 0.7)
        if np.count_nonzero(mk) < 12:
            continue
        pw = backproject_pixels(uf[mk], vf[mk], dk[mk], intr[k], ext[k])
        u2, v2, z2 = project_world_points(pw, intr[k + 1], ext[k + 1])
        ok = (z2 > min_depth) & (u2 >= 0) & (u2 <= W - 1) & (v2 >= 0) & (v2 <= H - 1)
        if np.count_nonzero(ok) < 8:
            continue
        ui = np.clip(np.round(u2[ok]).astype(int), 0, W - 1)
        vi = np.clip(np.round(v2[ok]).astype(int), 0, H - 1)
        d2_obs = d_arr[k + 1, vi, ui]
        val2 = np.isfinite(d2_obs) & (d2_obs >= min_depth) & (d2_obs <= max_depth * 2.0)
        if np.any(val2):
            rel = np.abs(z2[ok][val2] - d2_obs[val2]) / np.maximum(z2[ok][val2], 1e-3)
            rel_errs.append(float(np.median(rel)))

    med_rel_err = float(np.median(rel_errs)) if rel_errs else 1.0
    max_rel_err = float(max(rel_errs)) if rel_errs else 1.0
    max_step = max(cam_steps) if cam_steps else 0.0
    passed = bool(
        finite_ok
        and valid_frac >= 0.15
        and min_frame_valid >= 0.05
        and med_rel_err <= max_rel_depth_err
        and max_rel_err <= max_rel_depth_err * 2.5
        and max_step <= 1.2
    )
    return {
        "window_index": int(win_idx),
        "finite_ok": finite_ok,
        "valid_depth_fraction": round(valid_frac, 4),
        "min_frame_valid_fraction": round(min_frame_valid, 4),
        "median_confidence": round(med_conf, 3),
        "internal_median_rel_depth_err": round(med_rel_err, 4),
        "internal_max_rel_depth_err": round(max_rel_err, 4),
        "max_internal_cam_step": round(max_step, 4),
        "passed": passed,
    }


def _bilin_depth_and_grad(depth_2d: np.ndarray, u: float, v: float) -> Tuple[float, float]:
    H, W = depth_2d.shape
    u_c = float(np.clip(u, 0.0, W - 1.001))
    v_c = float(np.clip(v, 0.0, H - 1.001))
    u0, v0 = int(math.floor(u_c)), int(math.floor(v_c))
    du, dv = u_c - u0, v_c - v0
    d = (
        depth_2d[v0, u0] * (1 - du) * (1 - dv)
        + depth_2d[v0, min(u0 + 1, W - 1)] * du * (1 - dv)
        + depth_2d[min(v0 + 1, H - 1), u0] * (1 - du) * dv
        + depth_2d[min(v0 + 1, H - 1), min(u0 + 1, W - 1)] * du * dv
    )
    uc, vc = min(max(int(round(u_c)), 1), W - 2), min(max(int(round(v_c)), 1), H - 2)
    gx = 0.5 * (depth_2d[vc, uc + 1] - depth_2d[vc, uc - 1])
    gy = 0.5 * (depth_2d[vc + 1, uc] - depth_2d[vc - 1, uc])
    return float(d), float(math.hypot(gx, gy))


def detect_stationary_features_and_tracks(
    images_rgb: List[np.ndarray],
    depth_maps: np.ndarray,
    confidence_maps: np.ndarray,
    intrinsics_infer: np.ndarray,
    dynamic_masks_infer: Optional[List[Optional[np.ndarray]]] = None,
    min_depth: float = 0.20,
    max_depth: float = 5.50,
    min_conf: float = 0.65,
    max_depth_grad: float = 0.06,
    revisit_gap: int = 5,
    top_revisit_candidates: int = 5,
    min_loop_inliers: int = 22,
    min_loop_grid_cells: int = 5,
) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int, np.ndarray, np.ndarray, int]], Dict[str, Any]]:
    """
    Extracts SIFT keypoints in inference coordinates, excludes dynamic actor masks and
    silhouette depth-bleed boundaries, matches sequential neighbors + non-local revisit
    candidates with strict USAC_MAGSAC + 4x4 grid spread filtering, and builds multi-view tracks.

    Returns:
        tracks: list of {track_id, obs: [(cam_idx, u, v, d_local_raw, w_3d)], is_revisit: bool, span: int}
        verified_pairs: list of (idx_a, idx_b, pts_a_uv, pts_b_uv, grid_cells)
        summary: diagnostics dict
    """
    N = len(images_rgb)
    d_arr = depth_maps[..., 0] if depth_maps.ndim == 4 else depth_maps
    H_d, W_d = d_arr.shape[1], d_arr.shape[2]

    sift = cv2.SIFT_create(nfeatures=900)
    kps_list: List[List[Tuple[float, float, float, float]]] = []  # (u, v, d, w)
    des_list: List[Optional[np.ndarray]] = []
    global_vecs = []

    for i in range(N):
        img_resized = cv2.resize(images_rgb[i], (W_d, H_d), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)

        # Compact normalized descriptor for fast non-local revisit candidate shortlisting
        tiny = cv2.resize(gray, (24, 24), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        tiny -= tiny.mean()
        nrm = float(np.linalg.norm(tiny))
        global_vecs.append(tiny / max(nrm, 1e-6))

        mask_static = np.full((H_d, W_d), 255, dtype=np.uint8)
        if dynamic_masks_infer is not None and dynamic_masks_infer[i] is not None:
            dyn = cv2.resize(dynamic_masks_infer[i].astype(np.uint8), (W_d, H_d), interpolation=cv2.INTER_NEAREST)
            mask_static[dyn > 0] = 0

        kps_raw, des_raw = sift.detectAndCompute(gray, mask_static)
        kept_kps = []
        kept_idx = []
        if kps_raw and des_raw is not None:
            for k_i, kp in enumerate(kps_raw):
                u, v = float(kp.pt[0]), float(kp.pt[1])
                ui = min(max(int(round(u)), 0), W_d - 1)
                vi = min(max(int(round(v)), 0), H_d - 1)
                c_val = float(confidence_maps[i, vi, ui])
                d_val, dgrad = _bilin_depth_and_grad(d_arr[i], u, v)
                if (min_depth <= d_val <= max_depth) and (c_val >= min_conf) and (dgrad <= max_depth_grad):
                    # Penalize steep silhouette proximity gently in 3D weight
                    w_3d = float(np.clip(c_val / 3.0, 0.25, 1.0) * math.exp(-dgrad * 12.0))
                    kept_kps.append((u, v, d_val, w_3d))
                    kept_idx.append(k_i)
        kps_list.append(kept_kps)
        des_list.append(des_raw[kept_idx] if kept_idx else None)

    gv = np.vstack(global_vecs) if N > 0 else np.zeros((0, 576))
    sim_mat = gv @ gv.T

    # Candidate pairs: local sequential window neighbors (gap 1..3) + top non-local revisit candidates
    cand_pairs = set()
    for i in range(N):
        for g in (1, 2, 3):
            if i + g < N:
                cand_pairs.add((i, i + g))
        if i + revisit_gap < N:
            scores = sim_mat[i].copy()
            scores[: i + revisit_gap] = -1.0
            top_js = np.argsort(scores)[::-1][:top_revisit_candidates]
            for j in top_js:
                if scores[j] > 0.38:
                    cand_pairs.add((int(i), int(j)))

    bf = cv2.BFMatcher(cv2.NORM_L2)
    verified_pairs: List[Tuple[int, int, np.ndarray, np.ndarray, int]] = []
    # Union-Find to merge verified feature matches across views cleanly into multi-view tracks
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}

    def _find(x: Tuple[int, int]) -> Tuple[int, int]:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def _union(a: Tuple[int, int], b: Tuple[int, int]):
        ra, rb = _find(a), _find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    loop_pair_count = 0
    for ia, ib in sorted(cand_pairs):
        da, db = des_list[ia], des_list[ib]
        if da is None or db is None or len(da) < 12 or len(db) < 12:
            continue
        raw = bf.knnMatch(da, db, k=2)
        good = [m[0] for m in raw if len(m) == 2 and m[0].distance < 0.72 * m[1].distance]
        if len(good) < 16:
            continue
        pts_a = np.float32([(kps_list[ia][m.queryIdx][0], kps_list[ia][m.queryIdx][1]) for m in good])
        pts_b = np.float32([(kps_list[ib][m.trainIdx][0], kps_list[ib][m.trainIdx][1]) for m in good])

        try:
            F, mask = cv2.findFundamentalMat(pts_a, pts_b, cv2.USAC_MAGSAC, 1.75, 0.999)
        except Exception:
            F, mask = cv2.findFundamentalMat(pts_a, pts_b, cv2.FM_RANSAC, 2.0, 0.99)
        if F is None or mask is None:
            continue
        inl_idx = np.where(mask.ravel() > 0)[0]
        is_revisit = (ib - ia) >= revisit_gap
        min_req = min_loop_inliers if is_revisit else 14
        if len(inl_idx) < min_req:
            continue

        # Spatial 4x4 grid distribution gate (blocks localized poster/ceiling-fan false matches)
        grid = np.zeros((4, 4), dtype=int)
        for u_pt, v_pt in pts_b[inl_idx]:
            grid[min(3, int(v_pt / max(H_d / 4.0, 1.0))), min(3, int(u_pt / max(W_d / 4.0, 1.0)))] += 1
        occ_cells = int(np.count_nonzero(grid))
        if is_revisit and occ_cells < min_loop_grid_cells:
            continue
        if not is_revisit and occ_cells < 3:
            continue

        verified_pairs.append((ia, ib, pts_a[inl_idx], pts_b[inl_idx], occ_cells))
        if is_revisit:
            loop_pair_count += 1
        for k_in in inl_idx:
            m = good[k_in]
            ka = (ia, int(m.queryIdx))
            kb = (ib, int(m.trainIdx))
            parent.setdefault(ka, ka)
            parent.setdefault(kb, kb)
            _union(ka, kb)

    # Group connected components into conflict-free tracks (at most 1 observation per camera)
    comp_map: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for node in parent:
        comp_map.setdefault(_find(node), []).append(node)

    tracks: List[Dict[str, Any]] = []
    for root, nodes in sorted(comp_map.items()):
        by_cam: Dict[int, List[Tuple[int, int]]] = {}
        for cam_i, kp_i in nodes:
            by_cam.setdefault(cam_i, []).append((cam_i, kp_i))
        if len(by_cam) < 3:
            continue
        # Only keep unique single-keypoint observations per camera to avoid merged ambiguous tracks
        if any(len(v) > 1 for v in by_cam.values()):
            continue
        obs = []
        cams_sorted = sorted(by_cam.keys())
        for cam_i in cams_sorted:
            _, kp_i = by_cam[cam_i][0]
            u, v, d_val, w_3d = kps_list[cam_i][kp_i]
            obs.append((cam_i, u, v, d_val, w_3d))
        span = cams_sorted[-1] - cams_sorted[0]
        tracks.append({
            "track_id": len(tracks),
            "obs": obs,
            "is_revisit": bool(any(cams_sorted[t + 1] - cams_sorted[t] >= revisit_gap for t in range(len(cams_sorted) - 1))),
            "span": int(span),
        })

    # Keep top well-distributed tracks (preferring longer tracks & revisit bridges) for fast tractable BA
    tracks.sort(key=lambda tr: (int(tr["is_revisit"]), len(tr["obs"]), tr["span"]), reverse=True)
    if len(tracks) > 650:
        tracks = tracks[:650]

    summary = {
        "verified_pairs_total": len(verified_pairs),
        "verified_revisit_pairs": loop_pair_count,
        "multi_view_tracks_total": len(tracks),
        "revisit_tracks": sum(1 for t in tracks if t["is_revisit"]),
    }
    return tracks, verified_pairs, summary


def evaluate_pose_solution(
    extrinsics: np.ndarray,
    intrinsics_infer: np.ndarray,
    depth_maps: np.ndarray,
    tracks: List[Dict[str, Any]],
    window_scales: Optional[List[float]] = None,
    raw_local_depths: Optional[List[Dict[int, Tuple[float, np.ndarray]]]] = None,
) -> Dict[str, Any]:
    """
    Independently evaluates ANY camera-pose + scaled-depth solution on a track set
    (e.g., held-out validation tracks ONLY) without touching optimization residuals.
    Measures:
      - Median & RMSE 2D reprojection error of triangulated linear/weighted 3D points (px).
      - Cross-view 3D backprojected scatter across observations of each track (m).
      - Separate Revisit-track scatter (loop-closure consistency, m).
      - Trajectory continuity (max & p95 step jump, m) and relative scale spread.
    """
    ext = np.asarray(extrinsics, dtype=np.float64)
    intr = np.asarray(intrinsics_infer, dtype=np.float64)
    d_arr = depth_maps[..., 0] if depth_maps.ndim == 4 else depth_maps

    reproj_errs = []
    scatter_all = []
    scatter_revisit = []

    for tr in tracks:
        pts_w = []
        obs_valid = []
        for cam_i, u, v, _, _ in tr["obs"]:
            ui = min(max(int(round(u)), 0), d_arr.shape[2] - 1)
            vi = min(max(int(round(v)), 0), d_arr.shape[1] - 1)
            d_glob, _ = _bilin_depth_and_grad(d_arr[cam_i], u, v)
            if d_glob <= 0.1 or not math.isfinite(d_glob):
                continue
            pw = backproject_pixels(np.array([u]), np.array([v]), np.array([d_glob]), intr[cam_i], ext[cam_i])[0]
            pts_w.append(pw)
            obs_valid.append((cam_i, u, v))
        if len(pts_w) < 2:
            continue
        pts_np = np.asarray(pts_w, dtype=np.float64)
        mean_pt = np.median(pts_np, axis=0)
        dists = np.linalg.norm(pts_np - mean_pt, axis=1)
        med_dist = float(np.median(dists))
        scatter_all.append(med_dist)
        if tr.get("is_revisit", False):
            scatter_revisit.append(med_dist)

        for cam_i, u, v in obs_valid:
            up, vp, zp = project_world_points(mean_pt[None, :], intr[cam_i], ext[cam_i])
            if zp[0] > 0.1:
                reproj_errs.append(float(math.hypot(up[0] - u, vp[0] - v)))

    centers = np.array([camera_center_from_extrinsic(ext[i]) for i in range(len(ext))], dtype=np.float64)
    steps = np.linalg.norm(centers[1:] - centers[:-1], axis=1) if len(centers) > 1 else np.zeros(1)
    scales = np.asarray(window_scales, dtype=np.float64) if window_scales else np.ones(1)

    return {
        "num_eval_tracks": len(scatter_all),
        "num_eval_revisit_tracks": len(scatter_revisit),
        "reproj_median_px": round(float(np.median(reproj_errs)) if reproj_errs else 999.0, 3),
        "reproj_rmse_px": round(float(np.sqrt(np.mean(np.square(reproj_errs)))) if reproj_errs else 999.0, 3),
        "crossview_scatter_median_m": round(float(np.median(scatter_all)) if scatter_all else 999.0, 4),
        "crossview_scatter_p90_m": round(float(np.percentile(scatter_all, 90)) if scatter_all else 999.0, 4),
        "revisit_scatter_median_m": round(float(np.median(scatter_revisit)) if scatter_revisit else -1.0, 4),
        "max_traj_step_m": round(float(np.max(steps)), 4),
        "p95_traj_step_m": round(float(np.percentile(steps, 95)), 4),
        "scale_std": round(float(np.std(scales)), 4),
    }


def run_global_alignment_and_validated_ba(
    raw_windows: List[Dict[str, Any]],
    images_rgb: List[np.ndarray],
    baseline_extrinsics: np.ndarray,
    baseline_intrinsics_infer: np.ndarray,
    baseline_depths: np.ndarray,
    baseline_confs: np.ndarray,
    window_joins_baseline: List[Dict[str, Any]],
    dynamic_masks_infer: Optional[List[Optional[np.ndarray]]] = None,
    enable_pose_graph: bool = True,
    enable_ba: bool = True,
    thresholds: Optional[AlignmentQAThresholds] = None,
    log_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, GlobalAlignmentReport]:
    """
    Executes within-window checks, stationary feature tracking, revisit loop closures,
    global Sim(3) window pose graph, sparse BA, and strict held-out validation gate.
    Always returns `(accepted_extrinsics, accepted_depths, accepted_confs, report)`.
    """
    def _log(msg: str):
        if log_callback:
            log_callback(msg)

    t0 = time.perf_counter()
    th = thresholds or AlignmentQAThresholds()
    N = len(baseline_extrinsics)
    M = len(raw_windows)

    # 1. Within-Window Internal Consistency Checks
    ww_details = []
    for w_idx, w in enumerate(raw_windows):
        ww = check_within_window_consistency(
            win_idx=w_idx,
            ext=w["local_extrinsics"],
            intr=w["local_intrinsics"],
            depth=w["local_depth"],
            conf=w["local_conf"],
            max_rel_depth_err=th.max_within_window_depth_rel,
        )
        ww_details.append(ww)
    ww_pass = sum(1 for d in ww_details if d["passed"])
    ww_fail = len(ww_details) - ww_pass
    _log(f"  [Within-Window QC] Passed: {ww_pass}/{M} windows")

    # 2. Extract Stationary Feature Tracks & Verified Revisit Pairs
    tracks, verified_pairs, trk_sum = detect_stationary_features_and_tracks(
        images_rgb=images_rgb,
        depth_maps=baseline_depths,
        confidence_maps=baseline_confs,
        intrinsics_infer=baseline_intrinsics_infer,
        dynamic_masks_infer=dynamic_masks_infer,
        min_loop_inliers=th.min_loop_inliers,
        min_loop_grid_cells=th.min_loop_grid_cells,
    )
    _log(f"  [Tracks & Loops] Verified pairs: {trk_sum['verified_pairs_total']} (revisits: {trk_sum['verified_revisit_pairs']}), multi-view tracks: {len(tracks)} (revisit tracks: {trk_sum['revisit_tracks']})")

    # Deterministic Stratified 80% Optimization / 20% Held-Out Validation Split
    opt_tracks = [t for idx, t in enumerate(tracks) if idx % 5 != 0]
    heldout_tracks = [t for idx, t in enumerate(tracks) if idx % 5 == 0]

    # Map frame -> list of windows containing it
    frame_to_wins: Dict[int, List[Tuple[int, int]]] = {}
    for w_idx, w in enumerate(raw_windows):
        for l_pos, g_idx in enumerate(w["frame_indices"]):
            frame_to_wins.setdefault(int(g_idx), []).append((w_idx, l_pos))

    # Build relative window edges from verified sequential overlaps + non-local revisit matches
    # First initialize global window Sim(3) transforms T_w mapping window w local coords -> global world
    init_win_sim3: List[Tuple[float, np.ndarray, np.ndarray]] = [(1.0, np.eye(3), np.zeros(3))]
    for w_idx in range(1, M):
        w_cur = raw_windows[w_idx]
        pts_l, pts_g = [], []
        for l_pos, g_idx in enumerate(w_cur["frame_indices"]):
            if g_idx in frame_to_wins and any(pw < w_idx for pw, _ in frame_to_wins[g_idx]):
                d_glob = baseline_depths[g_idx].squeeze()
                c_glob = baseline_confs[g_idx].squeeze()
                d_loc = w_cur["local_depth"][l_pos].squeeze()
                c_loc = w_cur["local_conf"][l_pos].squeeze()
                H_d, W_d = d_loc.shape
                ys = np.linspace(int(H_d * 0.15), int(H_d * 0.85), 14, dtype=int)
                xs = np.linspace(int(W_d * 0.15), int(W_d * 0.85), 14, dtype=int)
                ug, vg = np.meshgrid(xs, ys)
                m = (c_glob[vg, ug] >= 0.7) & (c_loc[vg, ug] >= 0.7) & (d_loc[vg, ug] >= 0.2) & (d_glob[vg, ug] >= 0.2)
                if np.count_nonzero(m) >= 8:
                    pg = backproject_pixels(ug[m], vg[m], d_glob[vg, ug][m], baseline_intrinsics_infer[g_idx], baseline_extrinsics[g_idx])
                    pl = backproject_pixels(ug[m], vg[m], d_loc[vg, ug][m], w_cur["local_intrinsics"][l_pos], w_cur["local_extrinsics"][l_pos])
                    pts_l.append(pl)
                    pts_g.append(pg)
        if pts_l:
            fit = _umeyama_ransac_sim3(np.vstack(pts_l), np.vstack(pts_g), max_res=0.06)
            if fit is not None:
                init_win_sim3.append((fit[0], fit[1], fit[2]))
                continue
        # Fallback: derive R_w from R_glob = R_loc @ R_w.T => R_w = R_glob.T @ R_loc
        g0 = int(w_cur["frame_indices"][0])
        R_g0 = baseline_extrinsics[g0, :3, :3]
        R_l0 = w_cur["local_extrinsics"][0, :3, :3]
        R_w = R_g0.T @ R_l0
        c_g0 = camera_center_from_extrinsic(baseline_extrinsics[g0])
        c_l0 = camera_center_from_extrinsic(w_cur["local_extrinsics"][0])
        init_win_sim3.append((1.0, R_w, c_g0 - R_w @ c_l0))

    base_scales = [x[0] for x in init_win_sim3]
    baseline_metrics = evaluate_pose_solution(
        baseline_extrinsics, baseline_intrinsics_infer, baseline_depths, heldout_tracks or tracks, base_scales
    )
    _log(f"  [Baseline Held-Out] reproj_med={baseline_metrics['reproj_median_px']}px, "
         f"scatter={baseline_metrics['crossview_scatter_median_m']:.4f}m, "
         f"revisit_scatter={baseline_metrics['revisit_scatter_median_m']:.4f}m")

    # 3. Build Global Pose Graph Edges: Sequential Window Edges + Verified Non-Local Loop Closures
    pg_edges: List[Dict[str, Any]] = []
    # Sequential edges (w_a -> w_b where frames overlap)
    for w_a in range(M - 1):
        w_b = w_a + 1
        shared = sorted(set(raw_windows[w_a]["frame_indices"]) & set(raw_windows[w_b]["frame_indices"]))
        if len(shared) >= 2:
            Xa_l, Xb_l = [], []
            for g_idx in shared:
                pa = raw_windows[w_a]["frame_indices"].index(g_idx)
                pb = raw_windows[w_b]["frame_indices"].index(g_idx)
                da = raw_windows[w_a]["local_depth"][pa].squeeze()
                db = raw_windows[w_b]["local_depth"][pb].squeeze()
                ca = raw_windows[w_a]["local_conf"][pa].squeeze()
                cb = raw_windows[w_b]["local_conf"][pb].squeeze()
                H_d, W_d = da.shape
                ys = np.linspace(int(H_d * 0.15), int(H_d * 0.85), 14, dtype=int)
                xs = np.linspace(int(W_d * 0.15), int(W_d * 0.85), 14, dtype=int)
                ug, vg = np.meshgrid(xs, ys)
                mk = (ca[vg, ug] >= 0.7) & (cb[vg, ug] >= 0.7) & (da[vg, ug] >= 0.2) & (db[vg, ug] >= 0.2)
                if np.count_nonzero(mk) >= 8:
                    pa_pts = backproject_pixels(ug[mk], vg[mk], da[vg, ug][mk], raw_windows[w_a]["local_intrinsics"][pa], raw_windows[w_a]["local_extrinsics"][pa])
                    pb_pts = backproject_pixels(ug[mk], vg[mk], db[vg, ug][mk], raw_windows[w_b]["local_intrinsics"][pb], raw_windows[w_b]["local_extrinsics"][pb])
                    Xa_l.append(pa_pts)
                    Xb_l.append(pb_pts)
            if Xa_l:
                fit = _umeyama_ransac_sim3(np.vstack(Xb_l), np.vstack(Xa_l), max_res=0.05)
                if fit is not None:
                    pg_edges.append({
                        "w_src": w_b, "w_dst": w_a, "kind": "sequential",
                        "s_rel": fit[0], "R_rel": fit[1], "t_rel": fit[2],
                        "weight": 1.0,
                    })

    # Revisit Loop Closure Edges from Verified Feature Pairs across distinct windows
    loop_details = []
    seen_win_pairs = set()
    for ia, ib, pts_a_uv, pts_b_uv, occ_cells in verified_pairs:
        if (ib - ia) < 5:
            continue
        wa = frame_to_wins[ia][0][0]
        wb = frame_to_wins[ib][0][0]
        if abs(wa - wb) <= 1 or (wa, wb) in seen_win_pairs or (wb, wa) in seen_win_pairs:
            continue
        pa = raw_windows[wa]["frame_indices"].index(ia)
        pb = raw_windows[wb]["frame_indices"].index(ib)
        da = raw_windows[wa]["local_depth"][pa].squeeze()
        db = raw_windows[wb]["local_depth"][pb].squeeze()
        pts3_a, pts3_b = [], []
        for (ua, va), (ub, vb) in zip(pts_a_uv, pts_b_uv):
            da_v, ga = _bilin_depth_and_grad(da, ua, va)
            db_v, gb = _bilin_depth_and_grad(db, ub, vb)
            if (0.2 <= da_v <= 5.5) and (0.2 <= db_v <= 5.5) and ga <= 0.06 and gb <= 0.06:
                p3a = backproject_pixels(np.array([ua]), np.array([va]), np.array([da_v]), raw_windows[wa]["local_intrinsics"][pa], raw_windows[wa]["local_extrinsics"][pa])[0]
                p3b = backproject_pixels(np.array([ub]), np.array([vb]), np.array([db_v]), raw_windows[wb]["local_intrinsics"][pb], raw_windows[wb]["local_extrinsics"][pb])[0]
                pts3_a.append(p3a)
                pts3_b.append(p3b)
        if len(pts3_a) < 12:
            continue
        fit_lp = _umeyama_ransac_sim3(np.asarray(pts3_b), np.asarray(pts3_a), max_res=th.max_loop_sim3_median_m)
        if fit_lp is None or fit_lp[3] > th.max_loop_sim3_median_m or fit_lp[5] < 0.50:
            continue
        seen_win_pairs.add((wa, wb))
        loop_w = min(2.0, 0.8 + 0.02 * len(pts3_a)) * fit_lp[5]
        pg_edges.append({
            "w_src": wb, "w_dst": wa, "kind": "loop_closure",
            "s_rel": fit_lp[0], "R_rel": fit_lp[1], "t_rel": fit_lp[2],
            "weight": float(loop_w),
        })
        loop_details.append({
            "win_a": wa, "win_b": wb, "frame_a": ia, "frame_b": ib,
            "matches_used": len(pts3_a), "grid_cells": occ_cells,
            "s_rel": round(fit_lp[0], 4), "med_res_local_m": round(fit_lp[3], 4),
            "inlier_ratio": round(fit_lp[5], 3),
        })

    _log(f"  [Pose Graph] Sequential edges: {sum(1 for e in pg_edges if e['kind']=='sequential')}, Loop edges: {len(loop_details)}")

    # 4. Optimize Global Sim(3) Window Pose Graph
    opt_win_sim3 = list(init_win_sim3)
    if enable_pose_graph and M >= 2 and pg_edges:
        # Parameterize windows 1..M-1 relative to fixed window 0 (gauge fixed)
        p0 = np.zeros(7 * (M - 1), dtype=np.float64)
        for w_idx in range(1, M):
            sw, Rw, tw = init_win_sim3[w_idx]
            off = 7 * (w_idx - 1)
            p0[off] = math.log(max(sw, 1e-3))
            p0[off + 1:off + 4] = _mat_to_rodrigues(Rw)
            p0[off + 4:off + 7] = tw

        win_centers_local = [
            np.mean([camera_center_from_extrinsic(raw_windows[w]["local_extrinsics"][k]) for k in range(len(raw_windows[w]["frame_indices"]))], axis=0)
            for w in range(M)
        ]

        def _pg_res(p: np.ndarray) -> np.ndarray:
            T_list: List[Tuple[float, np.ndarray, np.ndarray]] = [(1.0, np.eye(3), np.zeros(3))]
            for w_idx in range(1, M):
                off = 7 * (w_idx - 1)
                sw = math.exp(np.clip(p[off], -1.6, 1.6))
                Rw = _rodrigues_to_mat(p[off + 1:off + 4])
                tw = p[off + 4:off + 7]
                T_list.append((sw, Rw, tw))

            res = []
            for e in pg_edges:
                wa = e["w_dst"]
                wb = e["w_src"]
                sa, Ra, ta = T_list[wa]
                sb, Rb, tb = T_list[wb]
                wt = e["weight"]

                # Target transform T_b_pred = T_a * T_rel =>
                # s_pred = sa * s_rel, R_pred = Ra @ R_rel, t_pred = sa * (Ra @ t_rel) + ta
                s_pred = sa * e["s_rel"]
                R_pred = Ra @ e["R_rel"]
                t_pred = sa * (Ra @ e["t_rel"]) + ta

                c_b = win_centers_local[wb]
                pos_b_actual = sb * (Rb @ c_b) + tb
                pos_b_pred = s_pred * (R_pred @ c_b) + t_pred
                res.extend(((pos_b_actual - pos_b_pred) * 2.5 * wt).tolist())
                res.extend(((_mat_to_rodrigues(Rb @ R_pred.T)) * 1.5 * wt).tolist())
                res.append((math.log(max(sb, 1e-4)) - math.log(max(s_pred, 1e-4))) * 2.0 * wt)
            return np.asarray(res, dtype=np.float64)

        sol_pg = least_squares(_pg_res, p0, loss="huber", f_scale=0.04, max_nfev=80)
        opt_win_sim3 = [(1.0, np.eye(3), np.zeros(3))]
        for w_idx in range(1, M):
            off = 7 * (w_idx - 1)
            sw = math.exp(np.clip(sol_pg.x[off], -1.6, 1.6))
            Rw = _rodrigues_to_mat(sol_pg.x[off + 1:off + 4])
            tw = sol_pg.x[off + 4:off + 7]
            opt_win_sim3.append((sw, Rw, tw))

    # Synthesize Pose-Graph Extrinsics & Rescaled Depths (blending multi-window observations per frame)
    pg_ext = np.zeros_like(baseline_extrinsics, dtype=np.float64)
    pg_depths = np.zeros_like(baseline_depths, dtype=np.float64)
    pg_confs = np.zeros_like(baseline_confs, dtype=np.float64)
    for g_idx in range(N):
        obs_w = frame_to_wins[g_idx]
        c_sum = np.zeros(3, dtype=np.float64)
        R_sum = np.zeros((3, 3), dtype=np.float64)
        d_acc = np.zeros(baseline_depths.shape[1:3], dtype=np.float64)
        w_acc = np.zeros(baseline_depths.shape[1:3], dtype=np.float64)
        c_max = np.zeros(baseline_confs.shape[1:3], dtype=np.float64)
        for w_idx, l_pos in obs_w:
            sw, Rw, tw = opt_win_sim3[w_idx]
            ext_g = apply_sim3_to_extrinsic(raw_windows[w_idx]["local_extrinsics"][l_pos], sw, Rw, tw)
            R_sum += ext_g[:3, :3]
            c_sum += camera_center_from_extrinsic(ext_g)
            d_loc = raw_windows[w_idx]["local_depth"][l_pos].squeeze()
            cf = raw_windows[w_idx]["local_conf"][l_pos].squeeze()
            d_acc += (sw * d_loc) * np.maximum(cf, 0.05)
            w_acc += np.maximum(cf, 0.05)
            c_max = np.maximum(c_max, cf)
        U, _, Vt = np.linalg.svd(R_sum / max(len(obs_w), 1))
        R_blend = U @ Vt
        if np.linalg.det(R_blend) < 0:
            U[:, 2] *= -1
            R_blend = U @ Vt
        c_blend = c_sum / max(len(obs_w), 1)
        pg_ext[g_idx] = extrinsic_from_center_and_rotation(R_blend, c_blend)
        d_res = d_acc / np.maximum(w_acc, 1e-6)
        if pg_depths.ndim == 4:
            pg_depths[g_idx, ..., 0] = d_res.astype(pg_depths.dtype)
        else:
            pg_depths[g_idx] = d_res.astype(pg_depths.dtype)
        pg_confs[g_idx] = c_max.astype(pg_confs.dtype)

    pg_scales = [x[0] for x in opt_win_sim3]
    pg_metrics = evaluate_pose_solution(
        pg_ext, baseline_intrinsics_infer, pg_depths, heldout_tracks or tracks, pg_scales
    )
    _log(f"  [Pose Graph Held-Out] reproj_med={pg_metrics['reproj_median_px']}px, "
         f"scatter={pg_metrics['crossview_scatter_median_m']:.4f}m, "
         f"revisit_scatter={pg_metrics['revisit_scatter_median_m']:.4f}m")

    # 5. Sparse Multi-View Track Bundle Adjustment + Held-Out Validation Gate
    ba_ext = pg_ext.copy()
    ba_depths = pg_depths.copy()
    ba_metrics = dict(pg_metrics)
    ba_decision = "DISABLED" if not enable_ba else "SKIPPED_INSUFFICIENT_EVIDENCE"
    ba_reason = "BA disabled in config"

    # Choose best pre-BA starting point between Baseline and Pose Graph
    cur_best_ext, cur_best_dep, cur_best_m, cur_best_scales, cur_best_name = (
        (pg_ext, pg_depths, pg_metrics, pg_scales, "pose_graph")
        if (pg_metrics["crossview_scatter_median_m"] <= baseline_metrics["crossview_scatter_median_m"] + 0.004
            and pg_metrics["reproj_median_px"] <= baseline_metrics["reproj_median_px"] + th.max_heldout_reproj_regression_px)
        else (baseline_extrinsics.astype(np.float64), baseline_depths.astype(np.float64), baseline_metrics, base_scales, "baseline")
    )

    if enable_ba and len(opt_tracks) >= 12 and len(heldout_tracks) >= th.min_heldout_tracks_for_ba_eval:
        # Parameters: per-camera (cam 1..N-1) [rvec(3), center(3), log_s(1)] + 3D track pts X_k(3)
        n_cam_vars = (N - 1) * 7
        n_tracks_ba = len(opt_tracks)
        x0 = np.zeros(n_cam_vars + n_tracks_ba * 3, dtype=np.float64)

        ref_centers = np.array([camera_center_from_extrinsic(cur_best_ext[i]) for i in range(N)], dtype=np.float64)
        ref_rvecs = np.array([_mat_to_rodrigues(cur_best_ext[i, :3, :3]) for i in range(N)], dtype=np.float64)
        for i in range(1, N):
            off = (i - 1) * 7
            x0[off:off + 3] = ref_rvecs[i]
            x0[off + 3:off + 6] = ref_centers[i]
            x0[off + 6] = 0.0

        # Initialize 3D points as robust median backprojection across track observations
        d_curr_2d = cur_best_dep[..., 0] if cur_best_dep.ndim == 4 else cur_best_dep
        track_obs_scaled_d = []
        for k_idx, tr in enumerate(opt_tracks):
            pws = []
            obs_k = []
            for cam_i, u, v, _, w3 in tr["obs"]:
                d_v, _ = _bilin_depth_and_grad(d_curr_2d[cam_i], u, v)
                if d_v <= 0.1:
                    continue
                pw = backproject_pixels(np.array([u]), np.array([v]), np.array([d_v]), baseline_intrinsics_infer[cam_i], cur_best_ext[cam_i])[0]
                pws.append(pw)
                obs_k.append((cam_i, u, v, d_v, w3))
            track_obs_scaled_d.append(obs_k)
            init_x = np.median(np.asarray(pws), axis=0) if pws else np.zeros(3)
            x0[n_cam_vars + k_idx * 3: n_cam_vars + (k_idx + 1) * 3] = init_x

        # Build sparse Jacobian pattern for ultra-fast Trust-Region-Reflective solve
        obs_flat: List[Tuple[int, int, float, float, float, float]] = []
        for k_idx, obs_k in enumerate(track_obs_scaled_d):
            for cam_i, u, v, d_v, w3 in obs_k:
                obs_flat.append((k_idx, cam_i, u, v, d_v, w3))

        n_res = len(obs_flat) * 5 + (N - 1) * 7
        J_pat = lil_matrix((n_res, len(x0)), dtype=int)
        row = 0
        for k_idx, cam_i, _, _, _, _ in obs_flat:
            pt_c = n_cam_vars + k_idx * 3
            J_pat[row:row + 5, pt_c:pt_c + 3] = 1
            if cam_i > 0:
                cc = (cam_i - 1) * 7
                J_pat[row:row + 5, cc:cc + 7] = 1
            row += 5
        for i in range(1, N):
            cc = (i - 1) * 7
            J_pat[row:row + 7, cc:cc + 7] = 1
            row += 7

        def _ba_residuals(x: np.ndarray) -> np.ndarray:
            res = np.zeros(n_res, dtype=np.float64)
            Rmats = [cur_best_ext[0, :3, :3]]
            Cvecs = [ref_centers[0]]
            scales_i = [1.0]
            for i in range(1, N):
                off = (i - 1) * 7
                Rmats.append(_rodrigues_to_mat(x[off:off + 3]))
                Cvecs.append(x[off + 3:off + 6])
                scales_i.append(math.exp(np.clip(x[off + 6], -0.45, 0.45)))

            r_i = 0
            for k_idx, cam_i, u, v, d_v, w3 in obs_flat:
                Xk = x[n_cam_vars + k_idx * 3: n_cam_vars + (k_idx + 1) * 3]
                Ri, Ci, si = Rmats[cam_i], Cvecs[cam_i], scales_i[cam_i]
                K = baseline_intrinsics_infer[cam_i]
                pc = Ri @ (Xk - Ci)
                z_safe = max(pc[2], 0.15)
                up = K[0, 0] * (pc[0] / z_safe) + K[0, 2]
                vp = K[1, 1] * (pc[1] / z_safe) + K[1, 2]
                # Normalized pixel residual (~1 px scale)
                res[r_i] = (up - u) * 0.45
                res[r_i + 1] = (vp - v) * 0.45
                # 3D backprojection agreement residual (meters * weight)
                d_s = si * d_v
                p_obs_c = np.array([(u - K[0, 2]) / K[0, 0] * d_s, (v - K[1, 2]) / K[1, 1] * d_s, d_s])
                p_obs_w = Ci + Ri.T @ p_obs_c
                res[r_i + 2:r_i + 5] = (Xk - p_obs_w) * (14.0 * w3)
                r_i += 5

            # Regularize camera poses lightly toward starting solution so pure-monocular views stay bounded
            for i in range(1, N):
                off = (i - 1) * 7
                res[r_i:r_i + 3] = (x[off:off + 3] - ref_rvecs[i]) * 2.0
                res[r_i + 3:r_i + 6] = (x[off + 3:off + 6] - ref_centers[i]) * 2.5
                res[r_i + 6] = x[off + 6] * 4.0
                r_i += 7
            return res

        sol_ba = least_squares(
            _ba_residuals, x0, jac_sparsity=J_pat, loss="huber", f_scale=1.2, max_nfev=35, x_scale="jac"
        )
        ba_ext = cur_best_ext.copy()
        ba_depths = cur_best_dep.copy()
        ba_scales = list(cur_best_scales)
        for i in range(1, N):
            off = (i - 1) * 7
            Ri = _rodrigues_to_mat(sol_ba.x[off:off + 3])
            Ci = sol_ba.x[off + 3:off + 6]
            si = math.exp(np.clip(sol_ba.x[off + 6], -0.45, 0.45))
            ba_ext[i] = extrinsic_from_center_and_rotation(Ri, Ci)
            if ba_depths.ndim == 4:
                ba_depths[i, ..., 0] *= si
            else:
                ba_depths[i] *= si

        ba_metrics = evaluate_pose_solution(
            ba_ext, baseline_intrinsics_infer, ba_depths, heldout_tracks, ba_scales
        )
        _log(f"  [BA Held-Out] reproj_med={ba_metrics['reproj_median_px']}px, "
             f"scatter={ba_metrics['crossview_scatter_median_m']:.4f}m, "
             f"revisit_scatter={ba_metrics['revisit_scatter_median_m']:.4f}m")

        # Mandatory Independent Acceptance Gate on Held-Out Tracks + Trajectory Continuity
        reproj_ok = ba_metrics["reproj_median_px"] <= cur_best_m["reproj_median_px"] + th.max_heldout_reproj_regression_px
        scatter_improved = ba_metrics["crossview_scatter_median_m"] <= cur_best_m["crossview_scatter_median_m"] - 1e-4
        reproj_improved = ba_metrics["reproj_median_px"] <= cur_best_m["reproj_median_px"] - 0.03
        revisit_non_degrading = (
            cur_best_m["revisit_scatter_median_m"] < 0
            or ba_metrics["revisit_scatter_median_m"] <= cur_best_m["revisit_scatter_median_m"] + 0.005
        )
        traj_non_degrading = ba_metrics["max_traj_step_m"] <= max(cur_best_m["max_traj_step_m"] * 1.15, th.max_trajectory_step_m)

        if reproj_ok and (scatter_improved or reproj_improved) and revisit_non_degrading and traj_non_degrading:
            ba_decision = "ACCEPTED"
            ba_reason = (
                f"Held-out validation improved vs {cur_best_name}: "
                f"reproj {cur_best_m['reproj_median_px']}->{ba_metrics['reproj_median_px']}px, "
                f"scatter {cur_best_m['crossview_scatter_median_m']}->{ba_metrics['crossview_scatter_median_m']}m"
            )
            cur_best_ext, cur_best_dep, cur_best_m = ba_ext, ba_depths, ba_metrics
        else:
            ba_decision = "REJECTED"
            ba_reason = (
                f"Rejected BA on held-out validation vs {cur_best_name}: "
                f"reproj {cur_best_m['reproj_median_px']} vs {ba_metrics['reproj_median_px']}px, "
                f"scatter {cur_best_m['crossview_scatter_median_m']} vs {ba_metrics['crossview_scatter_median_m']}m"
            )
    elif enable_ba:
        ba_reason = f"Held-out tracks ({len(heldout_tracks)}) < minimum required ({th.min_heldout_tracks_for_ba_eval}); retaining {cur_best_name}."

    _log(f"  [BA Decision] {ba_decision}: {ba_reason}")

    # 6. Final Alignment Acceptance Gate
    weak_joins = [
        int(j.get("window_index", idx + 1))
        for idx, j in enumerate(window_joins_baseline)
        if j.get("join_category", j.get("quality_status")) in ("WEAK", "WEAK_RECOVERABLE", "UNRELIABLE")
    ]
    failed_joins = [
        int(j.get("window_index", idx + 1))
        for idx, j in enumerate(window_joins_baseline)
        if j.get("join_category", j.get("quality_status")) == "FAILED"
    ]

    gate_failures = []
    if ww_fail > max(1, M // 4):
        gate_failures.append(f"within_window_failures ({ww_fail}/{M})")
    if cur_best_m["reproj_median_px"] > th.max_median_reproj_px and cur_best_m["num_eval_tracks"] > 0:
        gate_failures.append(f"heldout_reproj_exceeds_gate ({cur_best_m['reproj_median_px']}px > {th.max_median_reproj_px}px)")
    if len(failed_joins) > 0:
        gate_failures.append(f"disconnected_windows ({len(failed_joins)})")

    if len(failed_joins) >= max(1, M // 2) or ww_pass == 0:
        align_status = "ALIGNMENT_FAILED"
    elif failed_joins or len(weak_joins) / max(1, len(window_joins_baseline)) > th.max_weak_join_ratio:
        align_status = "PARTIAL_ALIGNMENT" if failed_joins else "ALIGNMENT_RECOVERABLE"
    elif gate_failures:
        align_status = "ALIGNMENT_RECOVERABLE"
    else:
        align_status = "ALIGNMENT_PASS"

    report = GlobalAlignmentReport(
        status=align_status,
        ba_decision=ba_decision,
        ba_reason=ba_reason,
        num_windows=M,
        within_window_passed=ww_pass,
        within_window_failed=ww_fail,
        sequential_edges=sum(1 for e in pg_edges if e["kind"] == "sequential"),
        loop_closure_edges=len(loop_details),
        weak_joins=weak_joins,
        failed_joins=failed_joins,
        total_tracks=len(tracks),
        opt_tracks=len(opt_tracks),
        heldout_tracks=len(heldout_tracks),
        baseline_metrics=baseline_metrics,
        pose_graph_metrics=pg_metrics,
        ba_refined_metrics=ba_metrics,
        accepted_metrics=cur_best_m,
        window_scales=[round(float(s), 4) for s in cur_best_scales],
        loop_closures_detail=loop_details,
        within_window_details=ww_details,
        gate_failures=gate_failures,
        duration_sec=round(time.perf_counter() - t0, 3),
    )

    return cur_best_ext.astype(np.float32), cur_best_dep.astype(np.float32), baseline_confs, report
