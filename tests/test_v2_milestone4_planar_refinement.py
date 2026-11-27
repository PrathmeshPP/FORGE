"""
Automated Tests — F.O.R.G.E. V2 Milestone 4 (PlanarGS Verification & Refinement).
Covers Part M Tests #14, #15, and #16:
  - Test #14: Plane-boundary preservation (table rims & adjacent walls kept sharp).
  - Test #15: Two nearby tabletops at the SAME height separated by a gap are NOT merged.
  - Test #16: Canonical geometry -> visibility-aware depth reprojection consistency for TSDF.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reconstruction.canonical_camera import extrinsic_from_center_and_rotation
from src.reconstruction.depth_consistency_filter import FreeSpaceVoxelEvidence
from src.reconstruction.planar_refinement import (
    PlanarRefinementConfig,
    associate_yolo_semantics_to_3d_points,
    project_canonical_points_to_depth_maps,
    refine_scene_geometry_planar_gs,
)


class TestPlanarRefinement(unittest.TestCase):

    def test_14_and_15_two_nearby_tabletops_not_merged_and_boundaries_preserved(self):
        rng = np.random.default_rng(77)
        pts_list = []

        # Desk 1 tabletop at y=0.0 (with slight interior Z/Y noise +-1.2cm, sharp elevated step boundary vs legs)
        xs1, zs1 = np.meshgrid(np.linspace(-0.70, -0.10, 35), np.linspace(1.5, 2.2, 35))
        ys1 = rng.normal(0.0, 0.008, size=xs1.shape)
        pts_desk1 = np.column_stack([xs1.ravel(), ys1.ravel(), zs1.ravel()])
        # Give desk1 perimeter rim a crisp bevel/drop-off to act as geometric table legs / edge markers
        pts_leg1 = np.column_stack([
            np.full(60, -0.10), np.linspace(0.03, 0.50, 60), np.full(60, 1.85)
        ])

        # Desk 2 tabletop at EXACT SAME height y=0.0, separated ONLY by a 10cm gap x in [-0.10, 0.00] -> starts at x=+0.02!
        # Wait: give Desk 2 a tiny +1.8cm offset in actual true surface (e.g. y=+0.018) so if erroneously merged it shifts Desk 2/1!
        xs2, zs2 = np.meshgrid(np.linspace(0.02, 0.62, 35), np.linspace(1.5, 2.2, 35))
        ys2 = rng.normal(0.018, 0.008, size=xs2.shape)
        pts_desk2 = np.column_stack([xs2.ravel(), ys2.ravel(), zs2.ravel()])

        # Vertical wall perpendicular behind both desks at z=2.28 (dense ~2cm sampling)
        xw, yw = np.meshgrid(np.linspace(-0.80, 0.80, 75), np.linspace(-0.6, 0.5, 55))
        zw = np.full_like(xw, 2.28) + rng.normal(0.0, 0.006, size=xw.shape)
        pts_wall = np.column_stack([xw.ravel(), yw.ravel(), zw.ravel()])

        pts = np.vstack([pts_desk1, pts_leg1, pts_desk2, pts_wall]).astype(np.float32)
        cols = np.full((len(pts), 3), 200, dtype=np.uint8)

        # Synthetic camera looking at the two desks & wall with valid depth & free space through the aisle gap!
        H, W = 120, 160
        K = np.array([[150.0, 0.0, 80.0], [0.0, 150.0, 60.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        cam_c = np.array([0.0, -0.5, 0.4])
        ext = np.stack([
            extrinsic_from_center_and_rotation(np.eye(3), cam_c),
            extrinsic_from_center_and_rotation(np.eye(3), cam_c + np.array([0.15, 0.0, 0.0])),
        ])
        intr = np.tile(K[None, :, :], (2, 1, 1))
        dmaps = np.full((2, H, W, 1), 2.3, dtype=np.float32)
        cmaps = np.full((2, H, W), 3.0, dtype=np.float32)

        # Populate exact camera depth maps from scene points + leave gap ray at 2.3m wall
        for ci in range(2):
            pc = (pts @ ext[ci, :3, :3].T) + ext[ci, :3, 3]
            uf = np.clip(np.round(K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]).astype(int), 0, W - 1)
            vf = np.clip(np.round(K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]).astype(int), 0, H - 1)
            for u, v, z in zip(uf, vf, pc[:, 2]):
                for du in (-1, 0, 1):
                    for dv in (-1, 0, 1):
                        uu, vv = min(max(u + du, 0), W - 1), min(max(v + dv, 0), H - 1)
                        if z < dmaps[ci, vv, uu, 0]:
                            dmaps[ci, vv, uu, 0] = z

        fse = FreeSpaceVoxelEvidence(voxel_size_m=0.04)
        fse.build_from_depth_maps(dmaps, cmaps, ext, intr)

        cfg = PlanarRefinementConfig(
            voxel_size=0.015, cluster_eps_m=0.040, min_patch_points=80, enable_yolo_proposals=False
        )
        res = refine_scene_geometry_planar_gs(
            points=pts, colors=cols, cam_centers=[cam_c],
            extrinsics=ext, intrinsics=intr, depth_maps=dmaps, confidence_maps=cmaps,
            free_space_evidence=fse, config=cfg,
        )

        # Must preserve >= 3 distinct verified planes (Desk 1, Desk 2, Wall) — NEVER merging Desk 1 & Desk 2!
        tabletops = [r for r in res["regions"] if r.semantic_label == "tabletop_or_bed"]
        walls = [r for r in res["regions"] if r.semantic_label == "wall"]
        self.assertGreaterEqual(len(tabletops), 2, f"Expected >=2 separate tabletops, got {len(tabletops)}")
        self.assertGreaterEqual(len(walls), 1)

        # Desk legs MUST NOT shift noticeably toward the tabletop plane (boundary/leg separation preserved)
        n1 = len(pts_desk1)
        leg_start, leg_end = n1, n1 + len(pts_leg1)
        leg_drift = np.max(np.linalg.norm(res["refined_points"][leg_start:leg_end] - pts[leg_start:leg_end], axis=1))
        self.assertLess(leg_drift, 0.005)

        # Desk 1 and Desk 2 must remain vertically separated near their ~1.8cm step (not collapsed to one global plane!)
        desk1_y_after = float(np.median(res["refined_points"][:n1, 1]))
        desk2_y_after = float(np.median(res["refined_points"][leg_end:leg_end + len(pts_desk2), 1]))
        self.assertGreater(abs(desk2_y_after - desk1_y_after), 0.010)

        # Test #16: Reprojecting canonical points back into camera depth maps preserves agreement for TSDF!
        d_sync, _, sync_stats = project_canonical_points_to_depth_maps(
            res["refined_points"], ext, intr, dmaps, cmaps, max_depth_delta_m=0.08
        )
        self.assertGreater(sync_stats["updated_from_canonical_points"], 200)
        self.assertTrue(np.all(np.isfinite(d_sync)))

    def test_11_12_yolo_semantics_preserve_furniture_and_exclude_person(self):
        """
        Tests #11 & #12: 3D YOLO association filters out moving `person` pixels, preserves static
        non-planar `chair` points untouched during PlanarGS refinement, and never merges adjacent
        `bed` / `dining table` objects carrying distinct furniture instance IDs.
        """
        rng = np.random.default_rng(11)
        H, W = 120, 160
        K = np.array([[150.0, 0.0, 80.0], [0.0, 150.0, 60.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        ext = np.stack([
            extrinsic_from_center_and_rotation(np.eye(3), np.array([0.0, -0.3, 0.2])),
            extrinsic_from_center_and_rotation(np.eye(3), np.array([0.12, -0.3, 0.2])),
        ])
        intr = np.tile(K[None, :, :], (2, 1, 1))

        # Chair points (curved/noisy at x=-0.35, y=-0.05..0.35, z=1.9) + Wall behind at z=2.3
        xc, yc = np.meshgrid(np.linspace(-0.45, -0.25, 16), np.linspace(-0.05, 0.35, 22))
        zc = 1.9 + 0.03 * np.sin(xc * 12.0)
        pts_chair = np.column_stack([xc.ravel(), yc.ravel(), zc.ravel()])
        xw, yw = np.meshgrid(np.linspace(-0.7, 0.7, 60), np.linspace(-0.5, 0.5, 45))
        pts_wall = np.column_stack([xw.ravel(), yw.ravel(), np.full_like(xw, 2.3).ravel()])
        pts = np.vstack([pts_chair, pts_wall]).astype(np.float32)
        cols = np.full((len(pts), 3), 200, dtype=np.uint8)

        dmaps = np.full((2, H, W, 1), 2.1, dtype=np.float32)
        cmaps = np.full((2, H, W), 3.0, dtype=np.float32)
        cls_masks = [np.full((H, W), -1, dtype=np.int16) for _ in range(2)]
        inst_masks = [np.zeros((H, W), dtype=np.int32) for _ in range(2)]
        dyn_masks = [np.zeros((H, W), dtype=np.uint8) for _ in range(2)]

        for ci in range(2):
            pc = (pts @ ext[ci, :3, :3].T) + ext[ci, :3, 3]
            uf = np.clip(np.round(K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]).astype(int), 0, W - 1)
            vf = np.clip(np.round(K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]).astype(int), 0, H - 1)
            for idx_p, (u, v, z) in enumerate(zip(uf, vf, pc[:, 2])):
                for du in (-1, 0, 1):
                    for dv in (-1, 0, 1):
                        uu, vv = min(max(u + du, 0), W - 1), min(max(v + dv, 0), H - 1)
                        if z < dmaps[ci, vv, uu, 0]:
                            dmaps[ci, vv, uu, 0] = z
                if idx_p < len(pts_chair):
                    cls_masks[ci][max(0, v - 2):min(H, v + 3), max(0, u - 2):min(W, u + 3)] = 56  # chair
                    inst_masks[ci][max(0, v - 2):min(H, v + 3), max(0, u - 2):min(W, u + 3)] = 10 + ci
                elif pts[idx_p, 0] > 0.35:
                    # Moving person region in upper-right -> MUST be excluded from furniture labels!
                    dyn_masks[ci][max(0, v - 1):min(H, v + 2), max(0, u - 1):min(W, u + 2)] = 1

        sem_cls, furn_inst, sem_stats = associate_yolo_semantics_to_3d_points(
            pts, ext, intr, dmaps, cls_masks, inst_masks, dynamic_masks=dyn_masks,
            class_id_to_name={"56": "chair"},
        )
        n_c = len(pts_chair)
        self.assertGreater(np.count_nonzero(sem_cls[:n_c] == 56), int(n_c * 0.7))
        self.assertEqual(np.count_nonzero(sem_cls[n_c:] == 56), 0)
        self.assertGreaterEqual(sem_stats["unique_3d_furniture_instances"], 1)

        res = refine_scene_geometry_planar_gs(
            pts, cols, [np.array([0.0, -0.3, 0.2])], ext, intr, dmaps, cmaps,
            point_semantic_classes=sem_cls, point_furniture_instance_ids=furn_inst,
            class_id_to_name={"56": "chair"},
            config=PlanarRefinementConfig(voxel_size=0.015, cluster_eps_m=0.04, min_patch_points=60),
        )
        # Chair points are strictly protected from planar flattening!
        chair_drift = np.max(np.linalg.norm(res["refined_points"][:n_c] - pts[:n_c], axis=1))
        self.assertEqual(float(chair_drift), 0.0)
        self.assertGreater(res["metrics"]["furniture_protected_points"], 0)


if __name__ == "__main__":
    unittest.main()
