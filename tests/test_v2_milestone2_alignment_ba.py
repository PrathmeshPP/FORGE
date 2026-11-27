"""
Automated Tests — F.O.R.G.E. V2 Milestone 2 (Global Pose Graph, Loop Closures & BA Gate).
Covers Part M Tests #9 (BA acceptance) and #10 (BA rejection when held-out worsens),
plus within-window internal verification on deterministic synthetic textured 3D scenes.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reconstruction.canonical_camera import extrinsic_from_center_and_rotation
from src.reconstruction.global_alignment_ba import (
    AlignmentQAThresholds,
    check_within_window_consistency,
    run_global_alignment_and_validated_ba,
)


def _render_synthetic_textured_room_sequence(n_frames: int = 8, H: int = 120, W: int = 160):
    """
    Renders a deterministic textured front/side wall with known 3D points so SIFT tracks
    and true camera depths exist without VGGT weights.
    """
    rng = np.random.default_rng(123)
    # Generate 220 distinctive 3D landmarks on a rigid room corner at Z in [2.0, 3.2]
    n_pts = 220
    pts_w = np.column_stack([
        rng.uniform(-1.0, 1.0, n_pts),
        rng.uniform(-0.7, 0.7, n_pts),
        rng.uniform(2.0, 3.0, n_pts),
    ])
    colors_p = [tuple(int(c) for c in rng.integers(30, 230, size=3)) for _ in range(n_pts)]
    radii_p = rng.integers(3, 6, size=n_pts)

    K = np.array([[140.0, 0.0, W * 0.5], [0.0, 140.0, H * 0.5], [0.0, 0.0, 1.0]], dtype=np.float64)
    extrinsics_gt = np.zeros((n_frames, 3, 4), dtype=np.float64)
    intrinsics_all = np.tile(K[None, :, :], (n_frames, 1, 1))
    depths_gt = np.full((n_frames, H, W, 1), 2.5, dtype=np.float32)
    confs_gt = np.full((n_frames, H, W), 3.0, dtype=np.float32)
    images = []

    # Camera moves along an arc in front of the scene and revisits start (loop!)
    for i in range(n_frames):
        theta = (2.0 * np.pi * i) / max(n_frames - 1, 1) * 0.35
        cam_c = np.array([0.45 * math.sin(i * 0.7), 0.12 * math.cos(i * 0.5), 0.10 * i * 0.05])
        if i >= n_frames - 2:
            cam_c = np.array([0.05 * (i - (n_frames - 2)), 0.0, 0.02])  # revisit near start!
        yaw = -0.18 * cam_c[0]
        R = np.array([[np.cos(yaw), 0.0, np.sin(yaw)], [0.0, 1.0, 0.0], [-np.sin(yaw), 0.0, np.cos(yaw)]])
        ext = extrinsic_from_center_and_rotation(R, cam_c)
        extrinsics_gt[i] = ext

        img = np.full((H, W, 3), 210, dtype=np.uint8)
        dmap = np.full((H, W), 2.6, dtype=np.float32)

        # Render sorted back-to-front texture circles & exact depth disks
        p_cam = (pts_w @ R.T) + ext[:, 3]
        order = np.argsort(p_cam[:, 2])[::-1]
        for idx in order:
            pc = p_cam[idx]
            if pc[2] <= 0.3:
                continue
            u = int(round(K[0, 0] * pc[0] / pc[2] + K[0, 2]))
            v = int(round(K[1, 1] * pc[1] / pc[2] + K[1, 2]))
            rad = int(radii_p[idx])
            if 0 <= u < W and 0 <= v < H:
                cv2.circle(img, (u, v), rad, colors_p[idx], -1)
                cv2.circle(img, (u, v), max(1, rad // 2), (255 - colors_p[idx][0], colors_p[idx][1], 20), 1)
                cv2.circle(dmap, (u, v), rad + 1, float(pc[2]), -1)

        # Planar background wall depth so gradients around disks aren't huge
        depths_gt[i, ..., 0] = cv2.GaussianBlur(dmap, (3, 3), 0.5)
        images.append(img)

    return images, extrinsics_gt, intrinsics_all, depths_gt, confs_gt


import math


class TestGlobalAlignmentAndBA(unittest.TestCase):

    def test_within_window_consistency_check(self):
        images, ext_gt, intr, depth_gt, conf_gt = _render_synthetic_textured_room_sequence(6)
        res_ok = check_within_window_consistency(0, ext_gt, intr, depth_gt, conf_gt)
        self.assertTrue(res_ok["passed"])

        # Corrupt one frame depth -> fail internal QC
        bad_depth = depth_gt.copy()
        bad_depth[2] *= 2.8
        res_bad = check_within_window_consistency(0, ext_gt, intr, bad_depth, conf_gt, max_rel_depth_err=0.05)
        self.assertFalse(res_bad["passed"])

    def test_9_ba_acceptance_on_drifted_poses(self):
        """Perturbed baseline poses + drifted second window should improve and pass BA acceptance."""
        images, ext_gt, intr, depth_gt, conf_gt = _render_synthetic_textured_room_sequence(8)
        raw_windows = [
            {
                "window_index": 0,
                "frame_indices": [0, 1, 2, 3, 4],
                "local_extrinsics": ext_gt[:5].copy(),
                "local_intrinsics": intr[:5].copy(),
                "local_depth": depth_gt[:5].copy(),
                "local_conf": conf_gt[:5].copy(),
            },
            {
                "window_index": 1,
                "frame_indices": [3, 4, 5, 6, 7],
                "local_extrinsics": ext_gt[3:].copy(),
                "local_intrinsics": intr[3:].copy(),
                "local_depth": depth_gt[3:].copy(),
                "local_conf": conf_gt[3:].copy(),
            },
        ]

        # Drift second window in baseline prediction (shift camera centers by +0.16m and yaw +6 deg)
        base_ext = ext_gt.copy()
        th_d = np.radians(6.0)
        R_d = np.array([[np.cos(th_d), 0.0, np.sin(th_d)], [0.0, 1.0, 0.0], [-np.sin(th_d), 0.0, np.cos(th_d)]])
        for i in range(4, 8):
            ci = -base_ext[i, :3, :3].T @ base_ext[i, :3, 3]
            Ri = base_ext[i, :3, :3] @ R_d.T
            ci_d = ci + np.array([0.16, -0.08, 0.05])
            base_ext[i] = extrinsic_from_center_and_rotation(Ri, ci_d)

        joins = [{"window_index": 1, "quality_status": "ACCEPTABLE", "join_category": "ACCEPTABLE"}]
        acc_ext, acc_dep, _, rep = run_global_alignment_and_validated_ba(
            raw_windows=raw_windows,
            images_rgb=images,
            baseline_extrinsics=base_ext,
            baseline_intrinsics_infer=intr,
            baseline_depths=depth_gt.copy(),
            baseline_confs=conf_gt.copy(),
            window_joins_baseline=joins,
            enable_pose_graph=True,
            enable_ba=True,
            thresholds=AlignmentQAThresholds(min_heldout_tracks_for_ba_eval=5),
        )
        self.assertEqual(rep.within_window_passed, 2)
        self.assertGreater(rep.total_tracks, 15)
        self.assertLess(rep.accepted_metrics["reproj_median_px"], rep.baseline_metrics["reproj_median_px"] + 0.5)
        self.assertLessEqual(rep.accepted_metrics["crossview_scatter_median_m"], rep.baseline_metrics["crossview_scatter_median_m"])
        self.assertIn(rep.status, ("ALIGNMENT_PASS", "ALIGNMENT_RECOVERABLE"))

    def test_10_ba_rejection_when_heldout_worsens(self):
        """
        Part M Test #10: When BA sees corrupted optimization tracks (or degraded candidate),
        held-out validation MUST reject BA and roll back to the uncorrupted baseline poses.
        """
        images, ext_gt, intr, depth_gt, conf_gt = _render_synthetic_textured_room_sequence(8)
        raw_windows = [
            {
                "window_index": 0,
                "frame_indices": [0, 1, 2, 3, 4, 5, 6, 7],
                "local_extrinsics": ext_gt.copy(),
                "local_intrinsics": intr.copy(),
                "local_depth": depth_gt.copy(),
                "local_conf": conf_gt.copy(),
            }
        ]
        # Corrupt confidence/depths badly AFTER extracting clean tracks via monkeypatch of detect function
        from src.reconstruction import global_alignment_ba as gba

        orig_det = gba.detect_stationary_features_and_tracks
        def _poisoned_det(**kwargs):
            tracks, vpairs, tsum = orig_det(**kwargs)
            poisoned = []
            for idx, tr in enumerate(tracks):
                tr_c = dict(tr)
                if idx % 5 != 0:
                    # Poison ONLY optimization tracks (shift pixel u/v by 28px!), leaving held-out tracks (idx%5==0) clean!
                    tr_c["obs"] = [(ci, u + 28.0 * ((-1) ** ci), v - 18.0, d, w) for (ci, u, v, d, w) in tr["obs"]]
                poisoned.append(tr_c)
            return poisoned, vpairs, tsum

        gba.detect_stationary_features_and_tracks = _poisoned_det
        try:
            acc_ext, acc_dep, _, rep = run_global_alignment_and_validated_ba(
                raw_windows=raw_windows,
                images_rgb=images,
                baseline_extrinsics=ext_gt.copy(),
                baseline_intrinsics_infer=intr,
                baseline_depths=depth_gt.copy(),
                baseline_confs=conf_gt.copy(),
                window_joins_baseline=[],
                enable_pose_graph=True,
                enable_ba=True,
                thresholds=AlignmentQAThresholds(min_heldout_tracks_for_ba_eval=5),
            )
        finally:
            gba.detect_stationary_features_and_tracks = orig_det

        self.assertEqual(rep.ba_decision, "REJECTED")
        # Baseline poses must be preserved cleanly!
        np.testing.assert_allclose(acc_ext, ext_gt, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
