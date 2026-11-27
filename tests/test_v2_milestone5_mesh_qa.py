"""
Automated Tests — F.O.R.G.E. V2 Milestone 5 (Global TSDF + Conditional Localized Poisson QA).
Covers Part M / Section 19 Tests #15–#20:
  - Test #15: Global TSDF produces primary surface without 6-submap switching/fallback.
  - Test #16: No forced PASS when observed TSDF fails QA gates.
  - Test #17: Small cracks/gaps (Case A) do NOT trigger localized Poisson.
  - Test #18: Qualifying enclosed interior supported gap (Case B) triggers localized Poisson,
              exports patch PLY, and leaves `mesh_global_tsdf_observed.ply` unmodified.
  - Test #19: Open doorway / KNOWN_FREE_SPACE gap (Case C) is preserved untouched (never sealed).
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reconstruction.canonical_camera import extrinsic_from_center_and_rotation
from src.reconstruction.depth_consistency_filter import FreeSpaceVoxelEvidence
from src.reconstruction.mesh_branches_and_qa import (
    MeshBranchQAConfig,
    build_canonical_scene_geometry,
    run_global_tsdf_and_conditional_repair,
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestGlobalTSDFAndConditionalRepairQA(unittest.TestCase):

    def test_15_16_global_tsdf_no_submap_fallback_and_honest_qa_fail(self):
        """
        Tests #15 & #16: Verify zero submap TSDF switching and strict NO-forced-PASS when
        synchronized depths are empty/invalid.
        """
        rng = np.random.default_rng(19)
        pts = rng.uniform([-0.5, -0.5, 2.0], [0.5, 0.5, 2.5], size=(250, 3)).astype(np.float32)
        cols = np.full((250, 3), 180, dtype=np.uint8)
        H, W = 64, 64
        K = np.array([[60.0, 0.0, 32.0], [0.0, 60.0, 32.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        ext = np.stack([
            extrinsic_from_center_and_rotation(np.eye(3), np.array([0.0, 0.0, 0.0])),
            extrinsic_from_center_and_rotation(np.eye(3), np.array([0.2, 0.0, 0.0])),
        ])
        intr = np.tile(K[None, :, :], (2, 1, 1))
        dmaps = np.zeros((2, H, W, 1), dtype=np.float32)
        wmaps = np.zeros((2, H, W), dtype=np.float32)
        frames = [np.full((H, W, 3), 150, dtype=np.uint8) for _ in range(2)]

        geom = build_canonical_scene_geometry(
            points=pts, colors=cols, cam_centers=[np.zeros(3)],
            extrinsics=ext, intrinsics=intr,
            synchronized_depth_maps=dmaps, validated_weight_maps=wmaps,
            voxel_size=0.02,
        )
        cfg = MeshBranchQAConfig(
            voxel_size=0.02, tsdf_voxel_length=0.02, tsdf_trunc=0.06,
            qa_min_triangles=5000, qa_min_obs_support_ratio=0.85,
        )
        with tempfile.TemporaryDirectory() as td:
            res = run_global_tsdf_and_conditional_repair(geom, frames, Path(td), cfg=cfg)
        self.assertEqual(res["selection_status"], "QA_FAILED")
        self.assertFalse(res["tsdf_stats"].get("submap_tsdf_enabled", True))
        self.assertEqual(res["tsdf_stats"].get("num_submaps"), 1)
        self.assertFalse(res["qa_observed"].passes_qa)

    def test_17_small_case_a_hole_never_triggers_poisson(self):
        """
        Test #17: A flat observed wall with a tiny pinhole (diameter ~4cm, Case A) must NOT
        invoke localized Poisson repair.
        """
        H, W = 90, 120
        K = np.array([[100.0, 0.0, 60.0], [0.0, 100.0, 45.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        ext = np.stack([
            extrinsic_from_center_and_rotation(np.eye(3), np.array([-0.15, 0.0, 0.2])),
            extrinsic_from_center_and_rotation(np.eye(3), np.array([0.0, 0.0, 0.2])),
            extrinsic_from_center_and_rotation(np.eye(3), np.array([0.15, 0.0, 0.2])),
        ])
        intr = np.tile(K[None, :, :], (3, 1, 1))
        dmaps = np.full((3, H, W, 1), 1.8, dtype=np.float32)
        cmaps = np.full((3, H, W), 3.0, dtype=np.float32)
        # Tiny pinhole centered at world (0,0,2.0) -> diameter ~0.05m (2 voxels) -> strictly Case A!
        xs_px = np.arange(W, dtype=np.float64)[None, :]
        ys_px = np.arange(H, dtype=np.float64)[:, None]
        for ci, x_cam in enumerate((-0.15, 0.0, 0.15)):
            wx = (xs_px - 60.0) / 100.0 * 1.8 + x_cam
            wy = (ys_px - 45.0) / 100.0 * 1.8
            pin = (np.abs(wx) < 0.025) & (np.abs(wy) < 0.025)
            dmaps[ci, pin, 0] = 0.0
        frames = [np.full((H, W, 3), 210, dtype=np.uint8) for _ in range(3)]

        xw, yw = np.meshgrid(np.linspace(-1.25, 1.25, 80), np.linspace(-0.85, 0.85, 55))
        pts = np.column_stack([xw.ravel(), yw.ravel(), np.full_like(xw, 2.0).ravel()]).astype(np.float32)
        cols = np.full((len(pts), 3), 210, dtype=np.uint8)

        fse = FreeSpaceVoxelEvidence(voxel_size_m=0.05)
        fse.build_from_depth_maps(dmaps, cmaps, ext, intr)
        geom = build_canonical_scene_geometry(
            points=pts, colors=cols, cam_centers=[np.array([0.0, 0.0, 0.2])],
            extrinsics=ext, intrinsics=intr,
            synchronized_depth_maps=dmaps, validated_weight_maps=cmaps,
            voxel_size=0.025, free_space_evidence=fse,
        )
        cfg = MeshBranchQAConfig(
            voxel_size=0.025, tsdf_voxel_length=0.025, tsdf_trunc=0.075,
            qa_min_triangles=400, qa_min_obs_support_ratio=0.80, qa_min_completeness_ratio=0.65,
            enable_conditional_repair=True,
            small_hole_max_voxel_diam=6.0,
        )
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            res = run_global_tsdf_and_conditional_repair(geom, frames, td_p, cfg=cfg)
            self.assertEqual(res["selection_status"], "PASS")
            self.assertFalse(res["repair_info"]["poisson_triggered"])
            self.assertEqual(len(res["repair_info"]["case_b_patches"]), 0)
            self.assertGreaterEqual(len(res["repair_info"]["case_a_decisions"]), 1)
            self.assertEqual(len(list(td_p.glob("poisson_repair_patch_*.ply"))), 0)
            self.assertTrue((td_p / "mesh_global_tsdf_observed.ply").is_file())

    def test_18_qualifying_case_b_triggers_localized_poisson_and_preserves_observed(self):
        """
        Test #18: An enclosed interior supported wall gap (~0.28m wide with zero free-space
        penetration through it) qualifies as Case B, extracts localized Poisson patches,
        creates `mesh_global_tsdf_repaired.ply`, and leaves `mesh_global_tsdf_observed.ply` intact.
        """
        H, W = 90, 120
        K = np.array([[100.0, 0.0, 60.0], [0.0, 100.0, 45.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        ext = np.stack([
            extrinsic_from_center_and_rotation(np.eye(3), np.array([-0.15, 0.0, 0.2])),
            extrinsic_from_center_and_rotation(np.eye(3), np.array([0.0, 0.0, 0.2])),
            extrinsic_from_center_and_rotation(np.eye(3), np.array([0.15, 0.0, 0.2])),
        ])
        intr = np.tile(K[None, :, :], (3, 1, 1))
        dmaps = np.full((3, H, W, 1), 1.8, dtype=np.float32)
        cmaps = np.full((3, H, W), 3.0, dtype=np.float32)
        frames = [np.full((H, W, 3), 210, dtype=np.uint8) for _ in range(3)]

        # Curved non-planar wall (saddle/wave z ~ 2.0 + 0.14*sin(5*x) + 0.09*cos(4.5*y)) around 0.30m gap -> triggers Case B2 Poisson!
        xs_px = np.arange(W, dtype=np.float64)[None, :]
        ys_px = np.arange(H, dtype=np.float64)[:, None]
        for ci, x_cam in enumerate((-0.15, 0.0, 0.15)):
            xw_a = (xs_px - 60.0) / 100.0 * 1.8 + x_cam
            yw_a = (ys_px - 45.0) / 100.0 * 1.8
            z_wall = 2.0 + 0.14 * np.sin(5.0 * xw_a) + 0.09 * np.cos(4.5 * yw_a)
            wx = (xs_px - 60.0) / 100.0 * (z_wall - 0.2) + x_cam
            wy = (ys_px - 45.0) / 100.0 * (z_wall - 0.2)
            dmaps[ci, :, :, 0] = (z_wall - 0.2).astype(np.float32)
            blind = (np.abs(wx) < 0.15) & (np.abs(wy) < 0.15)
            dmaps[ci, blind, 0] = 0.0

        xw, yw = np.meshgrid(np.linspace(-1.25, 1.25, 95), np.linspace(-0.85, 0.85, 65))
        zw = 2.0 + 0.14 * np.sin(5.0 * xw) + 0.09 * np.cos(4.5 * yw)
        pts_all = np.column_stack([xw.ravel(), yw.ravel(), zw.ravel()])
        # Keep points surrounding the 0.28m gap (ring of support points stays present!)
        in_gap = (np.abs(pts_all[:, 0]) < 0.13) & (np.abs(pts_all[:, 1]) < 0.13)
        pts = pts_all[~in_gap].astype(np.float32)
        cols = np.full((len(pts), 3), 210, dtype=np.uint8)

        fse = FreeSpaceVoxelEvidence(voxel_size_m=0.05)
        fse.build_from_depth_maps(dmaps, cmaps, ext, intr)
        geom = build_canonical_scene_geometry(
            points=pts, colors=cols, cam_centers=[np.array([0.0, 0.0, 0.2])],
            extrinsics=ext, intrinsics=intr,
            synchronized_depth_maps=dmaps, validated_weight_maps=cmaps,
            voxel_size=0.02, free_space_evidence=fse,
        )
        cfg = MeshBranchQAConfig(
            voxel_size=0.02, tsdf_voxel_length=0.02, tsdf_trunc=0.06,
            poisson_depth=6, qa_min_triangles=400, qa_min_obs_support_ratio=0.75,
            qa_min_completeness_ratio=0.60, enable_conditional_repair=True,
            small_hole_max_voxel_diam=5.0, large_hole_max_diam_m=0.80,
            case_b_min_support_pts=60, b1_planar_mad_max_m=0.004,
            case_b_max_normal_std_deg=35.0,
        )
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            res = run_global_tsdf_and_conditional_repair(geom, frames, td_p, cfg=cfg)
            obs_ply = td_p / "mesh_global_tsdf_observed.ply"
            self.assertTrue(obs_ply.is_file())
            hash_obs = _sha256_file(obs_ply)
            rep = res["repair_info"]
            self.assertTrue(rep["poisson_triggered"], f"Expected Case B localized Poisson trigger, got {rep}")
            self.assertGreaterEqual(len(rep["case_b_patches"]), 1)
            patch_files = list(td_p.glob("poisson_repair_patch_*.ply"))
            self.assertGreaterEqual(len(patch_files), 1)
            self.assertTrue((td_p / "mesh_global_tsdf_repaired.ply").is_file())
            # Re-verify observed file stayed unchanged & intact
            self.assertEqual(hash_obs, _sha256_file(obs_ply))

    def test_19_case_c_doorway_freespace_never_sealed(self):
        """
        Test #19: When rays penetrate through an opening (doorway to d=4.5m -> KNOWN_FREE_SPACE),
        the boundary hole is classified Case C and NEVER sealed by Poisson!
        """
        H, W = 90, 120
        K = np.array([[100.0, 0.0, 60.0], [0.0, 100.0, 45.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        ext = np.stack([
            extrinsic_from_center_and_rotation(np.eye(3), np.array([-0.15, 0.0, 0.2])),
            extrinsic_from_center_and_rotation(np.eye(3), np.array([0.0, 0.0, 0.2])),
            extrinsic_from_center_and_rotation(np.eye(3), np.array([0.15, 0.0, 0.2])),
        ])
        intr = np.tile(K[None, :, :], (3, 1, 1))
        dmaps = np.full((3, H, W, 1), 1.8, dtype=np.float32)
        cmaps = np.full((3, H, W), 3.0, dtype=np.float32)
        frames = [np.full((H, W, 3), 210, dtype=np.uint8) for _ in range(3)]

        # Punch doorway at world [-0.15..0.15] where rays travel straight to depth=4.5m background!
        xs_px = np.arange(W, dtype=np.float64)[None, :]
        ys_px = np.arange(H, dtype=np.float64)[:, None]
        for ci, x_cam in enumerate((-0.15, 0.0, 0.15)):
            wx = (xs_px - 60.0) / 100.0 * 1.8 + x_cam
            wy = (ys_px - 45.0) / 100.0 * 1.8
            door = (np.abs(wx) < 0.15) & (np.abs(wy) < 0.18)
            dmaps[ci, door, 0] = 4.5  # unobstructed hallway beyond the wall -> free space!

        xw, yw = np.meshgrid(np.linspace(-1.25, 1.25, 90), np.linspace(-0.85, 0.85, 60))
        pts_all = np.column_stack([xw.ravel(), yw.ravel(), np.full_like(xw, 2.0).ravel()])
        in_door = (np.abs(pts_all[:, 0]) < 0.15) & (np.abs(pts_all[:, 1]) < 0.18)
        pts = pts_all[~in_door].astype(np.float32)
        cols = np.full((len(pts), 3), 210, dtype=np.uint8)

        fse = FreeSpaceVoxelEvidence(voxel_size_m=0.04)
        fse.build_from_depth_maps(dmaps, cmaps, ext, intr, max_depth=5.0, pixel_stride=1)
        # Exclude doorway rays from TSDF integration after carving free-space (simulating hallway beyond max_depth clip or open rim)
        dmaps_tsdf = np.where(dmaps <= 3.0, dmaps, 0.0)
        geom = build_canonical_scene_geometry(
            points=pts, colors=cols, cam_centers=[np.array([0.0, 0.0, 0.2])],
            extrinsics=ext, intrinsics=intr,
            synchronized_depth_maps=dmaps_tsdf, validated_weight_maps=cmaps,
            voxel_size=0.02, free_space_evidence=fse,
        )
        cfg = MeshBranchQAConfig(
            voxel_size=0.02, tsdf_voxel_length=0.02, tsdf_trunc=0.06,
            poisson_depth=6, qa_min_triangles=400, qa_min_obs_support_ratio=0.75,
            qa_min_completeness_ratio=0.55, enable_conditional_repair=True,
            small_hole_max_voxel_diam=5.0, large_hole_max_diam_m=0.80,
        )
        with tempfile.TemporaryDirectory() as td:
            res = run_global_tsdf_and_conditional_repair(geom, frames, Path(td), cfg=cfg)
            rep = res["repair_info"]
            self.assertFalse(rep["poisson_triggered"], f"Doorway free-space must stay Case C and never trigger Poisson: {rep}")
            self.assertGreaterEqual(len(rep["case_c_preserved_gaps"]), 1)


if __name__ == "__main__":
    unittest.main()
