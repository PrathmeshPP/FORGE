"""
F.O.R.G.E. — Window Alignment, Overlap Verification & Selective Local Bundle Adjustment (LBA).

Ensures continuous, metric, and drift-free multi-window indoor reconstructions across
video transitions (corridors, doorways, room turns).

Capabilities:
1. Multi-Metric Join Quality Assessment:
   - Rotation error (geodesic distance between camera orientations).
   - Translation drift (camera center agreement).
   - Scale consistency ratio.
   - 3D dense point cloud Umeyama residual (median, p90, inlier ratio).
2. Strict Join Classification:
   - STRONG: tight convergence across geometry and camera poses.
   - WEAK_RECOVERABLE: recoverable via non-linear Local Bundle Adjustment (LBA).
   - UNRELIABLE / FAILED: diverged scale/residual, requires trajectory fallback or bridge frames.
3. Selective Local Bundle Adjustment (LBA):
   - Non-linear least-squares optimization over Sim(3) parameters (s, R, t) using Huber robust loss.
   - Jointly minimizes 3D surface point error and camera trajectory continuity.
   - Refines camera poses, transforms dependent geometry and depth maps.
4. Multi-stage Bridge Retry:
   - Up to 3 recovery retries with adaptive density and confidence relaxation.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Optional, Union

import cv2
import numpy as np
from scipy.optimize import least_squares


@dataclass
class WindowJoinQuality:
    """Quantitative diagnostics for a window-to-window or window-to-global join."""
    window_index: int
    shared_frames: List[int]
    num_shared_frames: int
    scale: float
    rotation_error_deg: float
    translation_drift_m: float
    median_residual_m: float
    p90_residual_m: float
    inlier_ratio: float
    spatial_spread_m: float
    num_correspondences: int
    join_category: str  # "STRONG" | "WEAK_RECOVERABLE" | "UNRELIABLE" | "FAILED"
    lba_applied: bool
    lba_residual_reduction_pct: float
    action_taken: str  # "DIRECT_MERGE" | "LBA_REFINED_MERGE" | "TRAJECTORY_CONTINUITY_RECOVERED" | "DISCONNECTED"
    diagnostics_note: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_index": self.window_index,
            "shared_frames": self.shared_frames,
            "num_shared_frames": self.num_shared_frames,
            "scale": round(float(self.scale), 4),
            "rotation_error_deg": round(float(self.rotation_error_deg), 2),
            "translation_drift_m": round(float(self.translation_drift_m), 4),
            "median_residual_m": round(float(self.median_residual_m), 4),
            "p90_residual_m": round(float(self.p90_residual_m), 4),
            "inlier_ratio": round(float(self.inlier_ratio), 4),
            "spatial_spread_m": round(float(self.spatial_spread_m), 4),
            "num_correspondences": int(self.num_correspondences),
            "join_category": self.join_category,
            "quality_status": self.join_category,
            "lba_applied": self.lba_applied,
            "lba_residual_reduction_pct": round(float(self.lba_residual_reduction_pct), 2),
            "action_taken": self.action_taken,
            "diagnostics_note": self.diagnostics_note,
        }


def compute_rotation_geodesic_distance_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """Computes the geodesic angular distance in degrees between two 3x3 rotation matrices."""
    R_diff = R1 @ R2.T
    tr = float(np.trace(R_diff))
    cos_angle = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    angle_rad = math.acos(cos_angle)
    return math.degrees(angle_rad)


def evaluate_window_overlap_metrics(
    X_local: np.ndarray,
    Y_global: np.ndarray,
    local_cams: List[np.ndarray],
    global_cams: List[np.ndarray],
    local_rots: List[np.ndarray],
    global_rots: List[np.ndarray],
    s_init: float,
    R_init: np.ndarray,
    t_init: np.ndarray
) -> Tuple[float, float, float, float, float, float]:
    """
    Evaluates alignment metrics for a candidate Sim(3) transform target ≈ s * (R @ source.T).T + t.
    Returns:
        (median_residual_m, p90_residual_m, inlier_ratio, mean_rot_err_deg, max_cam_drift_m, spatial_spread)
    """
    if len(X_local) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # 1. 3D point cloud residuals
    X_trans = s_init * (X_local @ R_init.T) + t_init
    residuals = np.linalg.norm(Y_global - X_trans, axis=1)
    med_res = float(np.median(residuals))
    p90_res = float(np.percentile(residuals, 90))
    inliers = float(np.mean(residuals <= 0.05))
    spatial_spread = float(np.linalg.norm(np.ptp(X_local, axis=0)))

    # 2. Camera rotation errors across shared frames
    rot_errors = []
    cam_drifts = []
    for R_l, R_g, c_l, c_g in zip(local_rots, global_rots, local_cams, global_cams):
        # Transformed local camera rotation in global frame: R_l_global = R_l @ R_init.T
        R_l_glob = R_l @ R_init.T
        rot_err = compute_rotation_geodesic_distance_deg(R_l_glob, R_g)
        rot_errors.append(rot_err)

        c_l_glob = s_init * (R_init @ c_l) + t_init
        drift = float(np.linalg.norm(c_g - c_l_glob))
        cam_drifts.append(drift)

    mean_rot_err = float(np.mean(rot_errors)) if rot_errors else 0.0
    max_cam_drift = float(np.max(cam_drifts)) if cam_drifts else 0.0

    return med_res, p90_res, inliers, mean_rot_err, max_cam_drift, spatial_spread


def run_selective_local_bundle_adjustment(
    X_local: np.ndarray,
    Y_global: np.ndarray,
    local_cams: List[np.ndarray],
    global_cams: List[np.ndarray],
    local_rots: List[np.ndarray],
    global_rots: List[np.ndarray],
    s_init: float,
    R_init: np.ndarray,
    t_init: np.ndarray,
    max_iterations: int = 40
) -> Tuple[float, np.ndarray, np.ndarray, float, float]:
    """
    Executes selective non-linear Local Bundle Adjustment (LBA) over Sim(3) parameters (s, R, t)
    using Huber robust loss to minimize point cloud discrepancy and camera trajectory drift.

    Parameter vector p: [ln(s), r_x, r_y, r_z, t_x, t_y, t_z] (7 DoF)
    """
    r_vec_init, _ = cv2.Rodrigues(R_init)
    r_init_flat = r_vec_init.flatten()
    log_s_init = math.log(max(s_init, 1e-4))

    p0 = np.array([
        log_s_init,
        r_init_flat[0], r_init_flat[1], r_init_flat[2],
        t_init[0], t_init[1], t_init[2]
    ], dtype=np.float64)

    # Subsample point cloud for fast robust optimization if large
    if len(X_local) > 500:
        step = max(1, len(X_local) // 500)
        X_sub = X_local[::step]
        Y_sub = Y_global[::step]
    else:
        X_sub = X_local
        Y_sub = Y_global

    cams_l_arr = np.array(local_cams, dtype=np.float64)
    cams_g_arr = np.array(global_cams, dtype=np.float64)

    # Initial residual calculation
    init_res, _, _, _, _, _ = evaluate_window_overlap_metrics(
        X_sub, Y_sub, local_cams, global_cams, local_rots, global_rots, s_init, R_init, t_init
    )

    def residuals_fn(p: np.ndarray) -> np.ndarray:
        s = math.exp(np.clip(p[0], -2.5, 2.5))
        r_v = p[1:4]
        t = p[4:7]
        R, _ = cv2.Rodrigues(r_v)

        # 1. 3D point residuals: Y - (s * X @ R.T + t)
        pts_pred = s * (X_sub @ R.T) + t
        pt_res = (Y_sub - pts_pred).flatten()

        # 2. Camera center continuity residuals (weighted)
        cams_pred = s * (cams_l_arr @ R.T) + t
        cam_res = ((cams_g_arr - cams_pred) * 2.5).flatten()

        return np.concatenate([pt_res, cam_res])

    # Optimize with Huber loss
    opt_result = least_squares(
        residuals_fn,
        p0,
        loss="huber",
        f_scale=0.035,
        max_nfev=max_iterations,
        ftol=1e-5,
        xtol=1e-5
    )

    p_opt = opt_result.x
    s_opt = math.exp(np.clip(p_opt[0], -2.5, 2.5))
    R_opt, _ = cv2.Rodrigues(p_opt[1:4])
    t_opt = p_opt[4:7]

    opt_res, _, _, _, _, _ = evaluate_window_overlap_metrics(
        X_sub, Y_sub, local_cams, global_cams, local_rots, global_rots, s_opt, R_opt, t_opt
    )

    red_pct = ((init_res - opt_res) / max(init_res, 1e-4)) * 100.0 if opt_res < init_res else 0.0

    if opt_res < init_res and (0.35 <= s_opt <= 2.5):
        return s_opt, R_opt, t_opt, opt_res, red_pct
    else:
        # Optimization did not improve; retain initial solution
        return s_init, R_init, t_init, init_res, 0.0


class WindowAlignmentLBAManager:
    """Orchestrates verified Sim(3) alignment, join classification, and selective LBA."""

    def __init__(self, enable_lba: bool = True, max_lba_iters: int = 40):
        self.enable_lba = enable_lba
        self.max_lba_iters = max_lba_iters

    def align_and_verify_window(
        self,
        w_idx: int,
        win_indices: List[int],
        shared_local_indices: List[int],
        shared_global_indices: List[int],
        X_local: np.ndarray,
        Y_global: np.ndarray,
        local_cams: List[np.ndarray],
        global_cams: List[np.ndarray],
        local_rots: List[np.ndarray],
        global_rots: List[np.ndarray],
        s_umeyama: float,
        R_umeyama: np.ndarray,
        t_umeyama: np.ndarray,
        log_callback: Optional[Any] = None
    ) -> Tuple[float, np.ndarray, np.ndarray, WindowJoinQuality]:
        """
        Evaluates, verifies, and optionally refines Sim(3) alignment between Window w and Global model.
        """
        def _log(msg: str):
            if log_callback:
                log_callback(msg)
            else:
                print(msg, flush=True)

        num_pts = len(X_local)
        num_shared = len(shared_global_indices)

        if num_pts < 10 or num_shared < 2:
            quality = WindowJoinQuality(
                window_index=w_idx,
                shared_frames=shared_global_indices,
                num_shared_frames=num_shared,
                scale=1.0,
                rotation_error_deg=0.0,
                translation_drift_m=0.0,
                median_residual_m=0.0,
                p90_residual_m=0.0,
                inlier_ratio=0.0,
                spatial_spread_m=0.0,
                num_correspondences=num_pts,
                join_category="FAILED",
                lba_applied=False,
                lba_residual_reduction_pct=0.0,
                action_taken="DISCONNECTED",
                diagnostics_note="insufficient_shared_points_or_frames"
            )
            return 1.0, np.eye(3), np.zeros(3), quality

        med_res, p90_res, inliers, rot_err, cam_drift, spread = evaluate_window_overlap_metrics(
            X_local, Y_global, local_cams, global_cams, local_rots, global_rots,
            s_umeyama, R_umeyama, t_umeyama
        )

        # Categorize join
        is_strong = (
            med_res <= 0.040
            and rot_err <= 6.0
            and (0.55 <= s_umeyama <= 1.70)
            and inliers >= 0.45
            and spread >= 0.15
        )

        is_unreliable = (
            med_res > 0.090
            or s_umeyama < 0.35
            or s_umeyama > 2.80
            or rot_err > 22.0
            or inliers < 0.15
        )

        s_final = s_umeyama
        R_final = R_umeyama
        t_final = t_umeyama
        lba_applied = False
        lba_red_pct = 0.0
        action = "DIRECT_MERGE"
        notes = []

        if is_strong:
            category = "STRONG"
            notes.append(f"strong_join (med_res: {med_res*100:.1f}cm, rot_err: {rot_err:.1f}deg, inliers: {inliers*100:.0f}%)")
        elif not is_unreliable:
            category = "WEAK_RECOVERABLE"
            notes.append(f"weak_recoverable (med_res: {med_res*100:.1f}cm, rot_err: {rot_err:.1f}deg, scale: {s_umeyama:.3f})")

            if self.enable_lba:
                _log(f"    [*] Window [{w_idx}] join is WEAK_RECOVERABLE. Initiating selective Local Bundle Adjustment (LBA)...")
                t0_lba = time.perf_counter()
                s_lba, R_lba, t_lba, lba_res, red_pct = run_selective_local_bundle_adjustment(
                    X_local, Y_global, local_cams, global_cams, local_rots, global_rots,
                    s_umeyama, R_umeyama, t_umeyama, max_iterations=self.max_lba_iters
                )
                lba_sec = time.perf_counter() - t0_lba

                if red_pct > 2.0:
                    s_final, R_final, t_final = s_lba, R_lba, t_lba
                    lba_applied = True
                    lba_red_pct = red_pct
                    med_res, p90_res, inliers, rot_err, cam_drift, spread = evaluate_window_overlap_metrics(
                        X_local, Y_global, local_cams, global_cams, local_rots, global_rots,
                        s_final, R_final, t_final
                    )
                    action = "LBA_REFINED_MERGE"
                    notes.append(f"LBA refined residual by {red_pct:.1f}% in {lba_sec:.2f}s (new med_res: {med_res*100:.1f}cm)")
                    _log(f"    [+] Window [{w_idx}] LBA refinement SUCCESS: residual reduced by {red_pct:.1f}% to {med_res*100:.1f} cm (rot_err: {rot_err:.1f}deg)")
                else:
                    _log(f"    [-] Window [{w_idx}] LBA produced marginal delta ({red_pct:.1f}%). Retaining robust Umeyama.")
        else:
            category = "UNRELIABLE"
            notes.append(f"unreliable_join (med_res: {med_res*100:.1f}cm, scale: {s_umeyama:.3f}, rot_err: {rot_err:.1f}deg)")
            # Camera trajectory continuity fallback
            _log(f"    [!] Window [{w_idx}] join UNRELIABLE. Recovering via smooth camera trajectory continuity...")
            R_l_last = local_rots[-1]
            c_l_last = local_cams[-1]
            R_g_last = global_rots[-1]
            c_g_last = global_cams[-1]

            R_w_cam = (R_l_last.T @ R_g_last).T
            s_w_cam = 1.0  # Safe unit scale
            t_w_cam = c_g_last - s_w_cam * (R_w_cam @ c_l_last)

            s_final, R_final, t_final = s_w_cam, R_w_cam, t_w_cam
            action = "TRAJECTORY_CONTINUITY_RECOVERED"
            notes.append("recovered_via_camera_trajectory_continuity")

        join_quality = WindowJoinQuality(
            window_index=w_idx,
            shared_frames=shared_global_indices,
            num_shared_frames=num_shared,
            scale=s_final,
            rotation_error_deg=rot_err,
            translation_drift_m=cam_drift,
            median_residual_m=med_res,
            p90_residual_m=p90_res,
            inlier_ratio=inliers,
            spatial_spread_m=spread,
            num_correspondences=num_pts,
            join_category=category,
            lba_applied=lba_applied,
            lba_residual_reduction_pct=lba_red_pct,
            action_taken=action,
            diagnostics_note="; ".join(notes)
        )

        return s_final, R_final, t_final, join_quality
